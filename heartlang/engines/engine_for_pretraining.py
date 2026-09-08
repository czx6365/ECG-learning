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
from typing import Iterable
import torch
import torch.nn as nn
from contextlib import nullcontext
import heartlang.utils.utils as utils


def random_masking(x, mask_ratio):
    # 对每个样本的 token 序列独立做随机打乱，
    # 再按 mask_ratio 还原出布尔 mask。
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask.to(torch.bool)


def _normalize_token_score(score: torch.Tensor):
    """
    对每个 token 的分数做标准化处理。
    输入:
        score: shape 通常是 [num_tokens]，表示每个 token 的某种分数，
               例如 token_loss 或 token_entropy。
    输出:
        标准化后的分数:
            mean ≈ 0
            std  ≈ 1
    作用:
        因为 token_loss 和 entropy 的数值尺度可能不同，
        直接相加会导致某一项主导 selective_score。
        所以先做 normalization，再按权重组合。
    """

    # 转成 float，避免半精度 float16 下均值/方差计算不稳定
    score = score.float()
    # 如果只有 0 或 1 个 token，标准差没有意义，直接返回全 0
    if score.numel() <= 1:
        return torch.zeros_like(score)
    # z-score 标准化:
    # (当前值 - 均值) / 标准差
    # unbiased=False 表示使用总体标准差，而不是样本标准差
    # clamp_min(1e-6) 是为了防止 std 太小导致除零或数值爆炸
    return (score - score.mean()) / score.std(unbiased=False).clamp_min(1e-6)


def _prediction_entropy(logits: torch.Tensor):
    """
    计算每个 token 的预测熵 entropy。
    输入:
        logits: shape 通常是 [num_tokens, vocab_size]
                表示每个 token 对整个词表的预测分数。
    输出:
        entropy: shape 是 [num_tokens]
                 每个 token 一个熵值。
    熵的含义:
        - 熵越低: 模型预测越确定，例如某个类别概率特别高
        - 熵越高: 模型预测越不确定，概率分布比较平均
    """

    # 先计算 log softmax，得到 log probability
    # logits.float() 是为了提高数值稳定性
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # entropy = - sum(p * log p)
    # log_probs.exp() 就是 softmax 后的概率 p
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def selective_masked_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_loss: torch.Tensor,
    keep_ratio: float,
    loss_weight: float,
    entropy_weight: float,
    min_keep_tokens: int,
):
    """
    选择性 masked token loss。
    核心思想:
        不是把所有 masked token 的 loss 都平均，
        而是先给每个 token 打分 selective_score，
        然后只保留一部分 token 参与最终 loss。
    输入:
        logits:
            shape 通常是 [num_tokens, vocab_size]
            每个 token 对 ECG vocabulary / codebook 的预测结果。
        labels:
            shape 通常是 [num_tokens]
            每个 token 的真实类别，也就是目标 collective ECG word index。
        token_loss:
            shape 通常是 [num_tokens]
            每个 token 单独的 cross entropy loss。
            注意这里一般要求 reduction='none'，否则拿不到逐 token loss
        keep_ratio:
            保留多少比例的 token。
            例如 0.7 表示保留 70% token 参与最终 loss。
        loss_weight:
            token_loss 在 selective_score 中的权重。
        entropy_weight:
            token_entropy 在 selective_score 中的权重。
        min_keep_tokens:
            最少保留多少个 token，避免保留数量太少导致 loss 不稳定。

    输出:
        loss:
            最终只在 keep_mask 选中的 token 上计算的平均 loss。
        keep_mask.float().mean().item():
            实际保留 token 的比例。
        token_entropy.mean().item():
            当前 batch/token 集合的平均预测熵。
        mlm_acc:
            所有 token 上的 masked language modeling accuracy。
        kept_mlm_acc:
            被保留下来的 token 上的 accuracy。
    """

    # 当前一共有多少个 token 需要计算 loss
    num_tokens = token_loss.numel()

    # 把 keep_ratio 限制在 [0, 1] 范围内，防止传入非法值
    keep_ratio = min(max(float(keep_ratio), 0.0), 1.0)

    # 根据 keep_ratio 计算理论上要保留多少 token
    # math.ceil 表示向上取整，例如 10 * 0.15 = 1.5，会保留 2 个
    keep_tokens = int(math.ceil(num_tokens * keep_ratio))

    # 最终保留数量需要满足:
    # 1. 不超过 num_tokens
    # 2. 至少 min_keep_tokens
    # 3. 至少 1 个 token
    keep_tokens = min(num_tokens, max(int(min_keep_tokens), keep_tokens, 1))

    # 下面只是选择 token，不希望这个选择过程参与反向传播
    # 所以用 torch.no_grad()
    with torch.no_grad():
        # 计算每个 token 的预测熵 shape: [num_tokens]
        token_entropy = _prediction_entropy(logits)

        # 计算每个 token 的选择分数 selective_score
        # token_loss 越大:说明这个 token 当前预测得越差。
        # token_entropy 越大:说明模型对这个 token 越不确定。
        # 这里 largest=False，后面会选择 selective_score 最小的 token。
        # 因此当前逻辑更偏向保留:
        #   - loss 较小的 token
        #   - entropy 较小的 token
        # 也就是模型更有把握、相对更容易的 token。
        selective_score = (
            loss_weight * _normalize_token_score(token_loss.detach())
            + entropy_weight * _normalize_token_score(token_entropy.detach())
        )

        # 选出 selective_score 最小的 keep_tokens 个 token
        # keep_indices 是被保留 token 的下标
        _, keep_indices = torch.topk(
            selective_score,
            k=keep_tokens,
            largest=False
        )

        # 初始化 keep_mask，默认所有 token 都不保留
        keep_mask = torch.zeros(
            num_tokens,
            dtype=torch.bool,
            device=logits.device
        )

        # 把被选中的 token 位置设为 True
        keep_mask[keep_indices] = True

    # 只对被保留的 token 计算平均 loss
    loss = token_loss[keep_mask].mean()

    # 计算所有 token 上的预测准确率
    # logits.max(-1)[1] 等价于 argmax，得到预测类别 index
    mlm_acc = (
        (logits.max(-1)[1] == labels)
        .float()
        .mean()
        .item()
    )

    # 只计算被保留 token 上的预测准确率
    kept_mlm_acc = (
        (logits[keep_mask].max(-1)[1] == labels[keep_mask])
        .float()
        .mean()
        .item()
    )

    # 返回:
    # 1. selective loss
    # 2. 实际保留比例
    # 3. 平均 entropy
    # 4. 全部 token accuracy
    # 5. 保留 token accuracy
    return (
        loss,
        keep_mask.float().mean().item(),
        token_entropy.mean().item(),
        mlm_acc,
        kept_mlm_acc
    )

def train_one_epoch(model: torch.nn.Module, vqhbr: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    # 这里实现 HeartLang 的 masked token 预训练主循环。
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    step_loader = 0
    for data_loader in data_loader_list:
        if len(data_loader) == 0:
            continue
        for step, (batch) in enumerate(
                metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # 按全局 step 动态更新学习率和 weight decay。
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            # samples: [bs, seq_len, time_window]
            # 其中 seq_len 是一句 ECG sentence 里的 token 数量，
            # time_window 是每个 token 对应的波形长度。
            samples = batch[0]
            samples = samples.float().to(device, non_blocking=True)
            in_chan_matrix = batch[1].to(device, non_blocking=True)
            in_time_matrix = batch[2].to(device, non_blocking=True)

            # 随机选择 50% 的 token 作为预测目标。
            bool_masked_pos = random_masking(samples, mask_ratio=0.5).to(device, non_blocking=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # 先用冻结的 VQ tokenizer 把连续 ECG token 离散化成 codebook id。
                    input_ids = vqhbr.get_codebook_indices(samples)
                # 监督信号只保留被 mask 的位置。
                labels = input_ids[bool_masked_pos]

            my_context = model.no_sync if args.distributed and (
                    step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            with my_context():
                with torch.cuda.amp.autocast():  # enabled=False
                    # HeartLang 只输出被 mask 位置的词表 logits。
                    x_rec = model(samples, mask_bool_matrix=bool_masked_pos, in_chan_matrix=in_chan_matrix,
                                  in_time_matrix=in_time_matrix, return_qrs_tokens=False, return_all_tokens=False)
                    # 目标是恢复出原始离散 token id。
                    token_loss = loss_fn(x_rec, labels)
                    use_selective_loss = getattr(args, "selective_mlm_loss", False) and (
                        epoch > getattr(args, "selective_mlm_warmup_epochs", 1)
                    )
                    if use_selective_loss:
                        (
                            loss,
                            selective_keep_ratio,
                            token_entropy,
                            mlm_acc,
                            kept_mlm_acc,
                        ) = selective_masked_token_loss(
                            x_rec,
                            labels,
                            token_loss,
                            getattr(args, "selective_mlm_keep_ratio", 0.85),
                            getattr(args, "selective_mlm_loss_weight", 1.0),
                            getattr(args, "selective_mlm_entropy_weight", 1.0),
                            getattr(args, "selective_mlm_min_keep_tokens", 1),
                        )
                    else:
                        loss = token_loss.mean()
                        selective_keep_ratio = 1.0
                        with torch.no_grad():
                            token_entropy = _prediction_entropy(x_rec).mean().item()
                        mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
                        kept_mlm_acc = mlm_acc

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                sys.exit(1)

            # 梯度累积时，只有到达 accumulation 边界才真正执行 optimizer update。
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= args.gradient_accumulation_steps
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            # mlm_acc 表示被 mask token 的恢复准确率。
            metric_logger.update(mlm_acc=mlm_acc)
            metric_logger.update(kept_mlm_acc=kept_mlm_acc)
            metric_logger.update(selective_keep_ratio=selective_keep_ratio)
            metric_logger.update(token_entropy=token_entropy)
            metric_logger.update(loss=loss.item())

            if log_writer is not None:
                log_writer.update(mlm_acc=mlm_acc, head="loss")
                log_writer.update(kept_mlm_acc=kept_mlm_acc, head="loss")
                log_writer.update(selective_keep_ratio=selective_keep_ratio, head="loss")
                log_writer.update(token_entropy=token_entropy, head="loss")
                log_writer.update(loss=loss.item(), head="loss")

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
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
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # 汇总多卡上的统计量，得到全局平均日志。
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
