#!/usr/bin/env python3
"""C4b ordinary SID MLM plus whole-lead masked prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import train_heartlang_sid_pretraining as base


def parse_args_with_cross_lead_weight():
    flag = "--cross_lead_weight"
    value = 0.1
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if index + 1 >= len(sys.argv):
            raise ValueError(f"{flag} requires a value")
        value = float(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    args = original_parse_args()
    args.cross_lead_weight = value
    if value < 0:
        raise ValueError("cross_lead_weight must be non-negative")
    return args


def load_c21_sentence_weights(model, optimizer, scaler, args, device):
    del optimizer, scaler
    if args.resume is None:
        raise ValueError("C4b requires --resume pointing to the C2.1 Sentence MLM checkpoint.")
    checkpoint_path = base.resolve_path(args.resume)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result = model.load_state_dict(checkpoint["model"], strict=True)
    print(
        f"Initialized C4b from {checkpoint_path}; source_epoch={checkpoint.get('epoch')} "
        f"load_missing={list(result.missing_keys)} load_unexpected={list(result.unexpected_keys)}",
        flush=True,
    )
    return 0, None


def whole_lead_mask(valid, chans):
    """Choose one observed lead per record and mask all of its valid SID tokens."""
    mask = torch.zeros_like(valid, dtype=torch.bool)
    for row in range(valid.shape[0]):
        lead_ids = torch.unique(chans[row][valid[row] & (chans[row] > 0)])
        if lead_ids.numel() < 2:
            continue
        selected = lead_ids[torch.randint(lead_ids.numel(), (1,), device=valid.device)]
        mask[row] = valid[row] & (chans[row] == selected)
    return mask


def run_epoch(model, loader, optimizer, scaler, args, device, epoch, total_steps, warmup_steps, train=True):
    model.train(train)
    keys = (
        "loss", "mlm_loss", "cross_lead_loss", "mlm_acc", "cross_lead_acc",
        "token_entropy", "masked_tokens", "cross_lead_tokens",
    )
    sums = {key: 0.0 for key in keys}
    count = 0
    max_steps = args.max_train_steps_per_epoch if train else args.max_eval_steps

    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        sid = batch[0].long().to(device, non_blocking=True)
        chans = batch[1].long().to(device, non_blocking=True)
        times = batch[2].long().to(device, non_blocking=True)
        valid = base.valid_sid_mask(sid, chans, times, args)
        mlm_mask = base.random_masking_valid(valid, args.mask_ratio)
        lead_mask = whole_lead_mask(valid, chans)
        mlm_labels = sid[mlm_mask]
        lead_labels = sid[lead_mask]
        if mlm_labels.numel() == 0 or lead_labels.numel() == 0:
            continue

        if train:
            global_step = (epoch - 1) * len(loader) + step
            lr = base.lr_for_step(global_step, total_steps, warmup_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            mlm_logits = model(
                sid, mask_bool_matrix=mlm_mask,
                in_chan_matrix=chans, in_time_matrix=times,
            )
            lead_logits = model(
                sid, mask_bool_matrix=lead_mask,
                in_chan_matrix=chans, in_time_matrix=times,
            )
            mlm_loss = F.cross_entropy(mlm_logits.float(), mlm_labels.long())
            cross_lead_loss = F.cross_entropy(lead_logits.float(), lead_labels.long())
            loss = mlm_loss + args.cross_lead_weight * cross_lead_loss

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad and args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

        with torch.no_grad():
            mlm_acc = (mlm_logits.argmax(dim=-1) == mlm_labels).float().mean()
            lead_acc = (lead_logits.argmax(dim=-1) == lead_labels).float().mean()
            probs = torch.softmax(mlm_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
        values = {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "cross_lead_loss": cross_lead_loss,
            "mlm_acc": mlm_acc,
            "cross_lead_acc": lead_acc,
            "token_entropy": entropy,
            "masked_tokens": mlm_labels.numel(),
            "cross_lead_tokens": lead_labels.numel(),
        }
        for key, value in values.items():
            sums[key] += float(value.item() if torch.is_tensor(value) else value)
        count += 1

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            phase = "train" if train else "val"
            print(
                f"{phase} epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss.item():.4f} mlm={mlm_loss.item():.4f} "
                f"cross={cross_lead_loss.item():.4f} mlm_acc={mlm_acc.item():.4f} "
                f"cross_acc={lead_acc.item():.4f} cross_tokens={lead_labels.numel()}",
                flush=True,
            )

    if count == 0:
        return {key: float("nan") for key in sums}
    return {key: value / count for key, value in sums.items()}


original_parse_args = base.parse_args
base.parse_args = parse_args_with_cross_lead_weight
base.load_checkpoint = load_c21_sentence_weights
base.run_epoch = run_epoch


if __name__ == "__main__":
    base.main()
