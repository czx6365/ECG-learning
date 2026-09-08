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

import math
import sys
from contextlib import nullcontext
from typing import Iterable
import torch
import heartlang.utils.utils as utils


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    model: torch.nn.Module,
    data_loader_list: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    clip_grad: float = 0,
    log_writer=None,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    args=None,
):
    # VQ tokenizer 的一个 epoch：训练 encoder / quantizer / decoder 做重建。
    model.train()
    model_without_ddp = unwrap_model(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if hasattr(model_without_ddp, "quantize"):
        try:
            # 每个 epoch 开始前重置 codebook 统计，便于观察本轮使用情况。
            model_without_ddp.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before each epoch")
        except:
            pass
    step_loader = 0
    for data_loader in data_loader_list:
        for step, (batch) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            # 按 step 更新学习率。
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group.get(
                            "lr_scale", 1.0
                        )

            # batch 中包含波形 token、导联索引和时间索引。
            ECG = batch[0].float().to(device, non_blocking=True)
            in_chan_matrix = batch[1].to(device, non_blocking=True)
            in_time_matrix = batch[2].to(device, non_blocking=True)
            amp_context = (
                torch.cuda.amp.autocast(enabled=True)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                # 模型返回总损失，以及量化/重建等分项日志。
                loss, log_loss = model(
                    ECG, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
                )

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value), force=True)
                sys.exit(1)

            optimizer.zero_grad()
            # 兼容 timm 中的二阶优化器实现。
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=clip_grad,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )
            loss_scale_value = loss_scaler.state_dict()["scale"]

            if device.type == "cuda":
                torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            new_log_loss = {
                k.split("/")[-1]: v
                for k, v in log_loss.items()
                if k not in ["total_loss"]
            }
            metric_logger.update(**new_log_loss)

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
                log_writer.update(**new_log_loss, head="train/loss")

                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.update(loss_scale=loss_scale_value, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # 汇总多卡统计量，得到全局平均损失。
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 额外统计本 epoch 未被使用到的 codebook 项数。
    if hasattr(model_without_ddp, "quantize"):
        try:
            codebook_cluster_size = model_without_ddp.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model_without_ddp.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat["Unused_code"] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return train_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader_list, model, device, log_writer=None, epoch=None, args=None):
    # 验证阶段与训练阶段类似，但不更新参数，可选输出 codebook 可视化。
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation:"
    model_without_ddp = unwrap_model(model)

    # 切换到 eval 模式，关闭 dropout 等训练态行为。
    model.eval()

    if hasattr(model_without_ddp, "quantize"):
        try:
            # 验证前同样重置 codebook 统计。
            model_without_ddp.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before testing")
        except:
            pass

    for data_loader in data_loader_list:
        for step, (batch) in enumerate(
            metric_logger.log_every(data_loader, 10, header)
        ):
            ECG = batch[0].float().to(device, non_blocking=True)
            in_chan_matrix = batch[1].to(device, non_blocking=True)
            in_time_matrix = batch[2].to(device, non_blocking=True)

            loss, log_loss = model(
                ECG, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
            )
            metric_logger.update(loss=loss.item())
            new_log_loss = {
                k.split("/")[-1]: v
                for k, v in log_loss.items()
                if k not in ["total_loss"]
            }
            metric_logger.update(**new_log_loss)

    # 汇总多卡统计。
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 验证阶段也报告未使用 code 的数量。
    if hasattr(model_without_ddp, "quantize"):
        try:
            codebook_cluster_size = model_without_ddp.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model_without_ddp.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat["unused_code"] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return test_stat

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

