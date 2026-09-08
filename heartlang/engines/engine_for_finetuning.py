# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# Based on BEiT-v2, timm, DeiT, DINO and LaBraM code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/935963004/LaBraM
# ---------------------------------------------------------

"""分类微调阶段的训练与评估循环。"""

import copy
import math
import sys
from contextlib import nullcontext
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import heartlang.utils.utils as utils
from einops import rearrange
import torch.distributed as dist

from heartlang.project_paths import (
    RESULTS_ECG_DSA_PREDICTIONS_DIR,
    RESULTS_PREDICTIONS_DIR,
    dataset_path_slug,
)


def build_regression_criterion(loss_type="mse", beta=1.0):
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return torch.nn.MSELoss()
    if loss_type == "smooth_l1":
        return torch.nn.SmoothL1Loss(beta=beta)
    if loss_type == "huber":
        return torch.nn.HuberLoss(delta=beta)
    raise ValueError(f"Unsupported regression_loss_type: {loss_type}")


def split_model_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["main"], outputs.get("aux_regression")
    return outputs, None


def split_batch_targets(target):
    if isinstance(target, dict):
        return (
            target["main"],
            target.get("aux_regression"),
            target.get("aux_regression_mask"),
        )
    return target, None, None


def move_target_to_device(target, device):
    if isinstance(target, dict):
        return {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
    return target.to(device, non_blocking=True)


def masked_regression_loss(criterion, prediction, target, mask):
    if mask is None:
        return criterion(prediction, target)
    mask = mask.float().type_as(prediction)
    valid_count = mask.sum()
    if valid_count.item() <= 0:
        return prediction.new_zeros(())
    masked_prediction = prediction * mask
    masked_target = target * mask
    loss = criterion(masked_prediction, masked_target)
    if criterion.reduction == "mean":
        loss = loss * (mask.numel() / valid_count)
    return loss


def train_class_batch(
    model,
    samples,
    target,
    criterion,
    aux_regression_criterion,
    aux_regression_loss_weight,
    in_chan_matrix,
    in_time_matrix,
    task_type="classification",
    is_binary=True,
):
    # 单个 batch 的前向与 loss 计算。
    outputs = model(
        samples, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
    )
    main_output, aux_output = split_model_outputs(outputs)
    main_target_raw, aux_target_raw, aux_mask_raw = split_batch_targets(target)
    main_target = prepare_targets_for_loss(
        main_target_raw, main_output, task_type=task_type, is_binary=is_binary
    )
    main_loss = criterion(main_output, main_target)

    aux_target = None
    aux_loss = None
    total_loss = main_loss
    if aux_output is not None:
        if aux_target_raw is None:
            raise ValueError("Model returned auxiliary regression outputs but dataset did not provide them.")
        aux_target = aux_target_raw.float().type_as(aux_output)
        if aux_target.shape != aux_output.shape:
            aux_target = aux_target.reshape_as(aux_output)
        aux_mask = None
        if aux_mask_raw is not None:
            aux_mask = aux_mask_raw.float().type_as(aux_output)
            if aux_mask.shape != aux_output.shape:
                aux_mask = aux_mask.reshape_as(aux_output)
        aux_loss = masked_regression_loss(
            aux_regression_criterion, aux_output, aux_target, aux_mask
        )
        total_loss = total_loss + aux_regression_loss_weight * aux_loss

    return total_loss, outputs, main_target, main_loss, aux_target, aux_loss


def prepare_targets_for_loss(target, outputs, task_type="classification", is_binary=True):
    if task_type == "regression":
        target = target.float().type_as(outputs)
        if target.ndim == 1 and outputs.ndim == 2:
            target = target.unsqueeze(-1)
        return target.reshape_as(outputs) if target.shape != outputs.shape else target

    if is_binary:
        return target.float().type_as(outputs)

    if target.ndim > 1 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    elif target.ndim > 1 and target.shape[-1] > 1:
        target = target.argmax(dim=-1)
    return target.long()


def postprocess_outputs_for_metrics(output, task_type="classification", is_binary=True):
    if task_type == "regression":
        return output
    if is_binary:
        return torch.sigmoid(output)
    return torch.softmax(output, dim=-1)


def get_loss_scale_for_deepspeed(model):
    # 兼容不同版本 deepspeed optimizer 的 loss scale 字段。
    optimizer = model.optimizer
    return (
        optimizer.loss_scale
        if hasattr(optimizer, "loss_scale")
        else optimizer.cur_scale
    )


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    aux_regression_criterion: Optional[torch.nn.Module],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    is_binary=True,
    task_type="classification",
    use_amp=True,
    aux_regression_loss_weight=1.0,
):
    # 微调阶段的一个 epoch，支持 AMP、梯度累积、EMA 和 DDP。
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    all_outputs = []
    all_targets = []

    for data_iter_step, (samples, targets, in_chan_matrix, in_time_matrix) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # 在每个梯度累积窗口的起点更新 LR / WD。
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get(
                        "lr_scale", 1.0
                    )
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True)
        targets = move_target_to_device(targets, device)
        in_chan_matrix = in_chan_matrix.to(device, non_blocking=True)
        in_time_matrix = in_time_matrix.to(device, non_blocking=True)

        if loss_scaler is None:
            # Deepspeed 分支。
            samples = samples.half()
            loss, output, targets_for_loss, main_loss, aux_targets_for_loss, aux_loss = train_class_batch(
                model,
                samples,
                targets,
                criterion,
                aux_regression_criterion,
                aux_regression_loss_weight,
                in_chan_matrix,
                in_time_matrix,
                task_type=task_type,
                is_binary=is_binary,
            )
        else:
            # 普通 AMP 分支。
            amp_context = (
                torch.amp.autocast("cuda")
                if use_amp and torch.cuda.is_available()
                else nullcontext()
            )
            with amp_context:
                loss, output, targets_for_loss, main_loss, aux_targets_for_loss, aux_loss = train_class_batch(
                    model,
                    samples,
                    targets,
                    criterion,
                    aux_regression_criterion,
                    aux_regression_loss_weight,
                    in_chan_matrix,
                    in_time_matrix,
                    task_type=task_type,
                    is_binary=is_binary,
                )


        main_output, _ = split_model_outputs(output)
        metric_output = postprocess_outputs_for_metrics(
            main_output, task_type=task_type, is_binary=is_binary
        )

        loss_value = loss.item()
        main_loss_value = main_loss.item()
        aux_loss_value = aux_loss.item() if aux_loss is not None else None

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            if use_amp:
                grad_norm = loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    create_graph=is_second_order,
                    update_grad=(data_iter_step + 1) % update_freq == 0,
                )
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)
            else:
                loss.backward(create_graph=is_second_order)
                if (data_iter_step + 1) % update_freq == 0:
                    if max_norm is not None and max_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm
                        )
                    else:
                        grad_norm = utils.get_grad_norm_(model.parameters())
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                else:
                    grad_norm = None
                loss_scale_value = 1.0

        all_outputs.append(metric_output.detach())
        all_targets.append(targets_for_loss.detach())

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if task_type == "regression":
            train_metric_name = "mae"
            train_metric_value = utils.analyze_regression(
                metric_output.detach().cpu().numpy(),
                targets_for_loss.detach().cpu().numpy(),
            )["mae"]
        elif is_binary:
            train_metric_name = "class_acc"
            train_metric_value = utils.analyze_ecg_classification(
                metric_output.detach().cpu().numpy(),
                targets_for_loss.detach().cpu().numpy(),
                is_binary,
            )["accuracy"]
        else:
            train_metric_name = "class_acc"
            train_metric_value = (
                metric_output.max(-1)[-1] == targets_for_loss.squeeze()
            ).float().mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(main_loss=main_loss_value)
        if aux_loss_value is not None:
            metric_logger.update(aux_reg_loss=aux_loss_value)
        metric_logger.update(**{train_metric_name: train_metric_value})
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="train")
            log_writer.update(main_loss=main_loss_value, head="train")
            if aux_loss_value is not None:
                log_writer.update(aux_reg_loss=aux_loss_value, head="train")
            log_writer.update(**{train_metric_name: train_metric_value}, head="train")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # 汇总多卡统计量。
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    header="Test:",
    metrics=["acc"],
    is_binary=True,
    dataset_dir="",
    task_type="classification",
    regression_loss_type="mse",
    regression_loss_beta=1.0,
    use_amp=True,
    aux_regression_loss_weight=1.0,
):
    # 评估阶段同时计算 batch loss、分类指标和全局 ROC-AUC。
    del metrics
    if task_type == "regression":
        criterion = build_regression_criterion(regression_loss_type, regression_loss_beta)
    elif is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    aux_regression_criterion = build_regression_criterion(
        regression_loss_type, regression_loss_beta
    )

    metric_logger = utils.MetricLogger(delimiter="  ")

    # 切换到 eval 模式。
    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        ECG = batch[0]
        target = batch[1]
        in_chan_matrix = batch[2]
        in_time_matrix = batch[3]
        ECG = ECG.float().to(device, non_blocking=True)
        target = move_target_to_device(target, device)
        in_chan_matrix = in_chan_matrix.to(device, non_blocking=True)
        in_time_matrix = in_time_matrix.to(device, non_blocking=True)

        # 只做前向推理，不更新参数。
        amp_context = (
            torch.amp.autocast("cuda")
            if use_amp and torch.cuda.is_available()
            else nullcontext()
        )
        with amp_context:
            output = model(
                ECG, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
            )
            main_output, aux_output = split_model_outputs(output)
            main_target_raw, aux_target_raw, aux_mask_raw = split_batch_targets(target)
            target_for_loss = prepare_targets_for_loss(
                main_target_raw, main_output, task_type=task_type, is_binary=is_binary
            )
            main_loss = criterion(main_output, target_for_loss)
            aux_loss = None
            loss = main_loss
            if aux_output is not None:
                if aux_target_raw is None:
                    raise ValueError("Model returned auxiliary regression outputs but dataset did not provide them.")
                aux_target_for_loss = aux_target_raw.float().type_as(aux_output)
                if aux_target_for_loss.shape != aux_output.shape:
                    aux_target_for_loss = aux_target_for_loss.reshape_as(aux_output)
                aux_mask_for_loss = None
                if aux_mask_raw is not None:
                    aux_mask_for_loss = aux_mask_raw.float().type_as(aux_output)
                    if aux_mask_for_loss.shape != aux_output.shape:
                        aux_mask_for_loss = aux_mask_for_loss.reshape_as(aux_output)
                aux_loss = masked_regression_loss(
                    aux_regression_criterion,
                    aux_output,
                    aux_target_for_loss,
                    aux_mask_for_loss,
                )
                loss = loss + aux_regression_loss_weight * aux_loss

        output = postprocess_outputs_for_metrics(
            main_output, task_type=task_type, is_binary=is_binary
        ).cpu()

        target = target_for_loss.cpu()

        pred.append(output)
        true.append(target)

        batch_size = ECG.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(main_loss=main_loss.item())
        if aux_loss is not None:
            metric_logger.update(aux_reg_loss=aux_loss.item())
        if task_type != "regression":
            results = utils.analyze_ecg_classification(
                output.numpy(), target.numpy(), is_binary, threshold=0.5
            )
            for key, value in results.items():
                metric_logger.meters[key].update(value, n=batch_size)

    # print when running
    all_outputs = torch.cat(pred, dim=0).to(device)
    all_targets = torch.cat(true, dim=0).to(device)

    if utils.is_dist_avail_and_initialized() and utils.get_world_size() > 1:
        gather_all_outputs = utils.gather_tensor(all_outputs)
        gather_all_targets = utils.gather_tensor(all_targets)
        print(f"gather_all_outputs shape: {gather_all_outputs.shape}")
        print(f"gather_all_targets shape: {gather_all_targets.shape}")
    else:
        gather_all_outputs = copy.deepcopy(all_outputs)
        gather_all_targets = copy.deepcopy(all_targets)

    if header == "Test:":
        # 测试阶段把预测值和真实标签落盘，方便后处理分析。
        dataset_name = dataset_path_slug(dataset_dir)
        predictions_dir = (
            RESULTS_ECG_DSA_PREDICTIONS_DIR
            if "ECG_DSA" in dataset_name
            else RESULTS_PREDICTIONS_DIR
        )

        output_file_path = predictions_dir / f"{dataset_name}_outputs.csv"
        target_file_path = predictions_dir / f"{dataset_name}_targets.csv"

        utils.save_tensor_to_csv(gather_all_outputs, output_file_path)
        utils.save_tensor_to_csv(gather_all_targets, target_file_path)

    if task_type == "regression":
        ret = utils.analyze_regression(
            gather_all_outputs.cpu().numpy(),
            gather_all_targets.cpu().numpy(),
        )
        metric_logger.synchronize_between_processes()
        print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))
        ret["loss"] = metric_logger.loss.global_avg
        ret["main_loss"] = metric_logger.main_loss.global_avg
        if "aux_reg_loss" in metric_logger.meters:
            ret["aux_reg_loss"] = metric_logger.aux_reg_loss.global_avg
        return ret

    roc_auc = utils.compute_roc_auc(
        gather_all_outputs.cpu().numpy(),
        gather_all_targets.cpu().numpy(),
    )
    metric_logger.update(roc_auc=roc_auc)
    # 汇总多卡统计。
    metric_logger.synchronize_between_processes()
    print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

    ret = utils.analyze_ecg_classification(
        all_outputs.cpu().numpy(), all_targets.cpu().numpy(), is_binary, 0.5
    )
    ret["loss"] = metric_logger.loss.global_avg
    ret["main_loss"] = metric_logger.main_loss.global_avg
    if "aux_reg_loss" in metric_logger.meters:
        ret["aux_reg_loss"] = metric_logger.aux_reg_loss.global_avg
    ret["roc_auc"] = roc_auc
    return ret
