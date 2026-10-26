#!/usr/bin/env python3
"""C5a: fine-tune the final SID Transformer block with a binary auxiliary loss."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import train_heartlang_sid_regression as base
from heartlang.utils.utils import set_seed


def pop_arg(name: str, cast, default):
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} requires a value")
    value = cast(sys.argv[index + 1])
    del sys.argv[index:index + 2]
    return value


def parse_args():
    resume_checkpoint = pop_arg("--resume_checkpoint", str, "")
    backbone_lr = pop_arg("--backbone_lr", float, 1e-5)
    backbone_min_lr = pop_arg("--backbone_min_lr", float, 1e-7)
    binary_aux_weight = pop_arg("--binary_aux_weight", float, 0.2)
    max_train_records = pop_arg("--max_train_records", int, 0)
    max_val_records = pop_arg("--max_val_records", int, 0)
    max_test_records = pop_arg("--max_test_records", int, 0)
    max_train_steps = pop_arg("--max_train_steps_per_epoch", int, 0)
    max_eval_steps = pop_arg("--max_eval_steps", int, 0)
    args = base.parse_args()
    args.backbone_lr = backbone_lr
    args.resume_checkpoint = resume_checkpoint
    args.backbone_min_lr = backbone_min_lr
    args.binary_aux_weight = binary_aux_weight
    args.max_train_records = max_train_records
    args.max_val_records = max_val_records
    args.max_test_records = max_test_records
    args.max_train_steps_per_epoch = max_train_steps
    args.max_eval_steps = max_eval_steps
    if args.task_head_type != "mlp2":
        raise ValueError("C5a requires --task_head_type mlp2")
    if binary_aux_weight < 0:
        raise ValueError("binary_aux_weight must be non-negative")
    return args


def binary_arrays(pred, target):
    return (
        np.column_stack([pred[:, :3].sum(axis=1), pred[:, 3]]),
        np.column_stack([target[:, :3].sum(axis=1), target[:, 3]]),
    )


def binary_metrics(pred, target):
    pred2, target2 = binary_arrays(pred, target)
    per_r2 = [float(r2_score(target2[:, i], pred2[:, i])) for i in range(2)]
    return {
        "r2": float(np.mean(per_r2)),
        "per_output_r2": {"Disease": per_r2[0], "Healthy": per_r2[1]},
        "mse": float(np.mean((pred2 - target2) ** 2)),
        "mae": float(np.mean(np.abs(pred2 - target2))),
    }


def cosine_lr(step, total_steps, warmup_steps, peak, minimum, warmup_start):
    if warmup_steps > 0 and step < warmup_steps:
        alpha = step / max(1, warmup_steps)
        return warmup_start + alpha * (peak - warmup_start)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def run_epoch(model, loader, optimizer, args, device, epoch, total_steps, warmup_steps, train, output_weights):
    model.train(train)
    outputs, targets = [], []
    sums = {"loss": 0.0, "four_loss": 0.0, "binary_loss": 0.0}
    steps = 0
    max_steps = args.max_train_steps_per_epoch if train else args.max_eval_steps
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        sid = batch[0].long().to(device, non_blocking=True)
        target = batch[1].float().to(device, non_blocking=True)
        chans = batch[2].long().to(device, non_blocking=True)
        times = batch[3].long().to(device, non_blocking=True)
        if train:
            global_step = (epoch - 1) * len(loader) + step
            head_lr = cosine_lr(
                global_step, total_steps, warmup_steps,
                args.lr, args.min_lr, args.warmup_lr,
            )
            backbone_lr = cosine_lr(
                global_step, total_steps, warmup_steps,
                args.backbone_lr, args.backbone_min_lr, min(args.warmup_lr, args.backbone_lr),
            )
            optimizer.param_groups[0]["lr"] = head_lr
            optimizer.param_groups[1]["lr"] = backbone_lr
            optimizer.zero_grad(set_to_none=True)

        pred = model(sid, in_chan_matrix=chans, in_time_matrix=times)
        four_loss = ((pred - target).square() * output_weights.view(1, -1)).mean()
        pred_binary = torch.stack([pred[:, :3].sum(dim=1), pred[:, 3]], dim=1)
        target_binary = torch.stack([target[:, :3].sum(dim=1), target[:, 3]], dim=1)
        binary_loss = (pred_binary - target_binary).square().mean()
        loss = four_loss + args.binary_aux_weight * binary_loss
        if train:
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad], args.clip_grad
                )
            optimizer.step()

        sums["loss"] += float(loss.item())
        sums["four_loss"] += float(four_loss.item())
        sums["binary_loss"] += float(binary_loss.item())
        steps += 1
        outputs.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == len(loader):
            phase = "train" if train else "eval"
            print(
                f"{phase} epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss.item():.5f} four={four_loss.item():.5f} binary={binary_loss.item():.5f}",
                flush=True,
            )

    pred_np = torch.cat(outputs).numpy()
    target_np = torch.cat(targets).numpy()
    metrics = base.metrics_from_arrays(pred_np, target_np, base.LABEL_NAMES)
    metrics.update({key: value / max(steps, 1) for key, value in sums.items()})
    metrics["binary"] = binary_metrics(pred_np, target_np)
    return metrics, pred_np, target_np


def limited_dataset(dataset, maximum):
    return Subset(dataset, range(min(len(dataset), maximum))) if maximum > 0 else dataset


def trainable_report(model):
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    illegal = [
        name for name in names
        if not (
            name.startswith("mlp_head.")
            or name.startswith("backbone.transformer.layers.11.")
            or name.startswith("backbone.norm_layer.")
        )
    ]
    if illegal:
        raise RuntimeError(f"Unexpected trainable C5a parameters: {illegal}")
    required = ("mlp_head.", "backbone.transformer.layers.11.", "backbone.norm_layer.")
    for prefix in required:
        if not any(name.startswith(prefix) for name in names):
            raise RuntimeError(f"No trainable parameter found for required prefix: {prefix}")
    return names


def main():
    args = parse_args()
    args.sid_data_dir = base.resolve_path(args.sid_data_dir)
    args.pretrained_checkpoint = base.resolve_path(args.pretrained_checkpoint)
    args.resume_checkpoint = base.resolve_path(Path(args.resume_checkpoint)) if args.resume_checkpoint else None
    args.output_dir = base.resolve_path(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)

    full_train = base.SIDRegressionDataset(args.sid_data_dir, "train")
    full_val = base.SIDRegressionDataset(args.sid_data_dir, "val")
    full_test = base.SIDRegressionDataset(args.sid_data_dir, "test")
    train_ds = limited_dataset(full_train, args.max_train_records)
    val_ds = limited_dataset(full_val, args.max_val_records)
    test_ds = limited_dataset(full_test, args.max_test_records)
    output_dim = int(full_train.labels.shape[1])
    variance = np.var(np.asarray(full_train.labels, dtype=np.float64), axis=0)
    raw_weights = 1.0 / np.maximum(variance, 1e-8)
    weights = raw_weights / raw_weights.mean()
    output_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    args.train_label_variance = variance.tolist()
    args.output_loss_weights = weights.tolist()

    model = base.build_model(args, output_dim).to(device)
    base.load_pretrained_backbone(model, args.pretrained_checkpoint, device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.mlp_head.parameters():
        param.requires_grad = True
    for param in model.backbone.transformer.layers[-1].parameters():
        param.requires_grad = True
    for param in model.backbone.norm_layer.parameters():
        param.requires_grad = True
    names = trainable_report(model)

    head_params = list(model.mlp_head.parameters())
    backbone_params = list(model.backbone.transformer.layers[-1].parameters()) + list(model.backbone.norm_layer.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr, "group_name": "head"},
            {"params": backbone_params, "lr": args.backbone_lr, "group_name": "last_block_and_norm"},
        ],
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    best_val_r2 = -float("inf")
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_r2 = float(checkpoint.get("metrics", {}).get("r2", -float("inf")))
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint epoch {start_epoch - 1} is not below target epochs {args.epochs}"
            )
        shutil.copy2(args.resume_checkpoint, args.output_dir / "checkpoint-best.pth")
        print(
            f"Resuming from {args.resume_checkpoint} at epoch={start_epoch}; "
            f"initial best validation R2={best_val_r2:.9f}",
            flush=True,
        )
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True),
        "val": DataLoader(val_ds, batch_size=int(1.5 * args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem),
        "test": DataLoader(test_ds, batch_size=int(1.5 * args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem),
    }
    total_steps = args.epochs * len(loaders["train"])
    warmup_steps = args.warmup_epochs * len(loaders["train"])
    args.trainable_parameter_names = names
    args.head_trainable_parameters = sum(param.numel() for param in head_params)
    args.backbone_trainable_parameters = sum(param.numel() for param in backbone_params)
    print(f"Train/val/test records: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Head trainable parameters: {args.head_trainable_parameters}")
    print(f"Backbone trainable parameters: {args.backbone_trainable_parameters}")
    print(f"Trainable prefixes verified; parameter tensors={len(names)}")
    print(f"Head/backbone peak LR: {args.lr}/{args.backbone_lr}")
    print(f"Loss: variance_normalized_mse + {args.binary_aux_weight} * binary_mse")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2))

    history = args.output_dir / "training_history.csv"
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, loaders["train"], optimizer, args, device, epoch, total_steps, warmup_steps, True, output_weights)
        with torch.no_grad():
            val_metrics, _, _ = run_epoch(model, loaders["val"], optimizer, args, device, epoch, total_steps, warmup_steps, False, output_weights)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_four_loss": train_metrics["four_loss"],
            "train_binary_loss": train_metrics["binary_loss"],
            "train_r2": train_metrics["r2"],
            "train_binary_r2": train_metrics["binary"]["r2"],
            "val_loss": val_metrics["loss"],
            "val_four_loss": val_metrics["four_loss"],
            "val_binary_loss": val_metrics["binary_loss"],
            "val_r2": val_metrics["r2"],
            "val_binary_r2": val_metrics["binary"]["r2"],
            "val_per_output_r2": val_metrics["per_output_r2"],
        }
        base.write_history(history, row)
        with (args.output_dir / "log.txt").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            base.save_checkpoint(model, optimizer, args, epoch, tag="best", metrics=val_metrics)
        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs:
            base.save_checkpoint(model, optimizer, args, epoch, metrics=val_metrics)

    checkpoint = torch.load(args.output_dir / "checkpoint-best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        test_metrics, pred, target = run_epoch(model, loaders["test"], optimizer, args, device, args.epochs, total_steps, warmup_steps, False, output_weights)
    test_metrics.update({
        "best_val_r2": float(best_val_r2),
        "best_epoch": int(checkpoint["epoch"]),
        "label_names": base.LABEL_NAMES,
        "n_test": len(test_ds),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "loss_type": "variance_normalized_mse_plus_binary_aux",
        "binary_aux_weight": args.binary_aux_weight,
        "head_lr": args.lr,
        "backbone_lr": args.backbone_lr,
    })
    (args.output_dir / "test_regression_summary.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    base.save_predictions(args.output_dir, "test", pred, target)
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"Training time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
