#!/usr/bin/env python3
"""Linear-probe four-output regression on a pretrained HeartLangSID backbone."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models.modeling_heartlang_sid import HeartLangSIDForRegression
from heartlang.utils.utils import set_seed


LABEL_NAMES = ["OMI", "NOMI", "Clinical Control", "Healthy"]


class SIDRegressionDataset(Dataset):
    def __init__(self, sid_data_dir: Path, split: str):
        self.sid = np.load(sid_data_dir / f"{split}_sid.npy", mmap_mode="r")
        self.chans = np.load(sid_data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
        self.times = np.load(sid_data_dir / f"{split}_data_in_times.npy", mmap_mode="r")
        labels = np.load(sid_data_dir / f"{split}_labels.npy", mmap_mode="r", allow_pickle=True)
        self.labels = labels.astype(np.float32)
        if self.labels.ndim == 1:
            self.labels = self.labels.reshape(-1, 1)

    def __len__(self):
        return len(self.sid)

    def __getitem__(self, idx):
        return (
            np.array(self.sid[idx], copy=True),
            np.array(self.labels[idx], copy=True),
            np.array(self.chans[idx], copy=True),
            np.array(self.times[idx], copy=True),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HeartLangSID four-output linear regression")
    parser.add_argument("--sid_data_dir", type=Path, required=True)
    parser.add_argument("--pretrained_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--save_ckpt_freq", type=int, default=10)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--pad_sid", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb_dropout", type=float, default=0.1)
    parser.add_argument("--no_spatial_embedding", action="store_false", dest="use_spatial_embedding")
    parser.add_argument("--no_temporal_embedding", action="store_false", dest="use_temporal_embedding")
    parser.set_defaults(use_spatial_embedding=True, use_temporal_embedding=True)
    parser.add_argument("--task_head_type", choices=["linear", "mlp2"], default="linear")
    parser.add_argument("--task_head_hidden_dim", type=int, default=256)
    parser.add_argument("--task_head_dropout", type=float, default=0.1)
    parser.add_argument(
        "--loss_type",
        choices=["mse", "variance_normalized_mse"],
        default="mse",
        help="variance_normalized_mse weights each output by inverse train-label variance.",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def build_model(args, num_outputs: int):
    return HeartLangSIDForRegression(
        num_outputs=num_outputs,
        task_head_type=args.task_head_type,
        task_head_hidden_dim=args.task_head_hidden_dim,
        task_head_dropout=args.task_head_dropout,
        freeze_backbone=True,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        pad_sid=args.pad_sid,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        use_spatial_embedding=args.use_spatial_embedding,
        use_temporal_embedding=args.use_temporal_embedding,
    )


def load_pretrained_backbone(model, checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(f"No backbone.* keys found in {checkpoint_path}")
    result = model.backbone.load_state_dict(backbone_state, strict=True)
    print(f"Loaded pretrained SID backbone from {checkpoint_path}")
    print(f"Backbone load result: {result}")


def lr_for_step(global_step, total_steps, warmup_steps, args):
    if warmup_steps > 0 and global_step < warmup_steps:
        alpha = global_step / max(1, warmup_steps)
        return args.warmup_lr + alpha * (args.lr - args.warmup_lr)
    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, args, epoch, tag=None, metrics=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{epoch}.pth" if tag is None else f"checkpoint-{tag}.pth"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "model_type": "heartlang_sid_regression_linear",
        "metrics": metrics or {},
    }
    torch.save(payload, args.output_dir / name)
    torch.save(payload, args.output_dir / "checkpoint.pth")


def metrics_from_arrays(pred, target, label_names):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse_by_dim = np.mean((pred - target) ** 2, axis=0)
    mae_by_dim = np.mean(np.abs(pred - target), axis=0)
    per_output_r2 = {}
    for idx in range(target.shape[1]):
        name = label_names[idx] if idx < len(label_names) else f"output_{idx}"
        try:
            per_output_r2[name] = float(r2_score(target[:, idx], pred[:, idx]))
        except ValueError:
            per_output_r2[name] = float("nan")
    try:
        r2 = float(r2_score(target, pred, multioutput="uniform_average"))
    except ValueError:
        r2 = float("nan")
    return {
        "r2": r2,
        "per_output_r2": per_output_r2,
        "mse": float(np.mean(mse_by_dim)),
        "rmse": float(math.sqrt(float(np.mean(mse_by_dim)))),
        "mae": float(np.mean(mae_by_dim)),
        "per_output_mse": {
            (label_names[i] if i < len(label_names) else f"output_{i}"): float(v)
            for i, v in enumerate(mse_by_dim)
        },
        "per_output_mae": {
            (label_names[i] if i < len(label_names) else f"output_{i}"): float(v)
            for i, v in enumerate(mae_by_dim)
        },
    }


def run_epoch(
    model, loader, optimizer, args, device, epoch, total_steps, warmup_steps,
    train=True, output_weights=None,
):
    model.train(train)
    outputs = []
    targets = []
    loss_sum = 0.0
    steps = 0
    for step, batch in enumerate(loader):
        sid = batch[0].long().to(device, non_blocking=True)
        target = batch[1].float().to(device, non_blocking=True)
        chans = batch[2].long().to(device, non_blocking=True)
        times = batch[3].long().to(device, non_blocking=True)
        if train:
            global_step = (epoch - 1) * len(loader) + step
            lr = lr_for_step(global_step, total_steps, warmup_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

        pred = model(sid, in_chan_matrix=chans, in_time_matrix=times)
        squared_error = (pred - target).square()
        if output_weights is None:
            loss = squared_error.mean()
        else:
            loss = (squared_error * output_weights.view(1, -1)).mean()
        if train:
            loss.backward()
            if args.clip_grad and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.clip_grad,
                )
            optimizer.step()

        loss_sum += float(loss.item())
        steps += 1
        outputs.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == len(loader):
            phase = "train" if train else "eval"
            print(f"{phase} epoch={epoch} step={step + 1}/{len(loader)} loss={loss.item():.5f}", flush=True)
    pred_np = torch.cat(outputs).numpy()
    target_np = torch.cat(targets).numpy()
    metrics = metrics_from_arrays(pred_np, target_np, LABEL_NAMES)
    metrics["loss"] = float(loss_sum / max(steps, 1))
    return metrics, pred_np, target_np


def write_history(path: Path, row: dict):
    flat = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def save_predictions(output_dir: Path, split: str, pred, target):
    pdict = {f"pred_{name}": pred[:, i] for i, name in enumerate(LABEL_NAMES[: pred.shape[1]])}
    tdict = {f"target_{name}": target[:, i] for i, name in enumerate(LABEL_NAMES[: target.shape[1]])}
    with (output_dir / f"{split}_outputs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pdict.keys()))
        writer.writeheader()
        for row_idx in range(pred.shape[0]):
            writer.writerow({key: float(values[row_idx]) for key, values in pdict.items()})
    with (output_dir / f"{split}_targets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(tdict.keys()))
        writer.writeheader()
        for row_idx in range(target.shape[0]):
            writer.writerow({key: float(values[row_idx]) for key, values in tdict.items()})


def main() -> None:
    args = parse_args()
    args.sid_data_dir = resolve_path(args.sid_data_dir)
    args.pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)

    train_ds = SIDRegressionDataset(args.sid_data_dir, "train")
    val_ds = SIDRegressionDataset(args.sid_data_dir, "val")
    test_ds = SIDRegressionDataset(args.sid_data_dir, "test")
    output_dim = int(train_ds.labels.shape[1])
    train_label_variance = np.var(np.asarray(train_ds.labels, dtype=np.float64), axis=0)
    if args.loss_type == "variance_normalized_mse":
        raw_weights = 1.0 / np.maximum(train_label_variance, 1e-8)
        normalized_weights = raw_weights / raw_weights.mean()
    else:
        normalized_weights = np.ones(output_dim, dtype=np.float64)
    output_weights = torch.as_tensor(normalized_weights, dtype=torch.float32, device=device)
    args.train_label_variance = train_label_variance.tolist()
    args.output_loss_weights = normalized_weights.tolist()
    model = build_model(args, output_dim).to(device)
    load_pretrained_backbone(model, args.pretrained_checkpoint, device)
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True),
        "val": DataLoader(val_ds, batch_size=int(1.5 * args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem),
        "test": DataLoader(test_ds, batch_size=int(1.5 * args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem),
    }
    total_steps = args.epochs * len(loaders["train"])
    warmup_steps = args.warmup_epochs * len(loaders["train"])
    print(f"SID downstream data: {args.sid_data_dir}")
    print(f"Train/val/test records: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Output dim: {output_dim}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Loss type: {args.loss_type}")
    print(f"Train label variance: {args.train_label_variance}")
    print(f"Normalized output loss weights: {args.output_loss_weights}")
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_val_r2 = -float("inf")
    history_path = args.output_dir / "training_history.csv"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, loaders["train"], optimizer, args, device, epoch, total_steps, warmup_steps,
            train=True, output_weights=output_weights,
        )
        with torch.no_grad():
            val_metrics, _, _ = run_epoch(
                model, loaders["val"], optimizer, args, device, epoch, total_steps, warmup_steps,
                train=False, output_weights=output_weights,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_r2": val_metrics["r2"],
            "val_per_output_r2": val_metrics["per_output_r2"],
        }
        write_history(history_path, row)
        with (args.output_dir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            save_checkpoint(model, optimizer, args, epoch, tag="best", metrics=val_metrics)
        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, args, epoch, metrics=val_metrics)

    best_ckpt = torch.load(args.output_dir / "checkpoint-best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"], strict=True)
    with torch.no_grad():
        test_metrics, test_pred, test_target = run_epoch(
            model, loaders["test"], optimizer, args, device, epoch=args.epochs,
            total_steps=total_steps, warmup_steps=warmup_steps, train=False,
            output_weights=output_weights,
        )
    test_metrics["best_val_r2"] = float(best_val_r2)
    test_metrics["label_names"] = LABEL_NAMES[:output_dim]
    test_metrics["n_test"] = int(len(test_ds))
    test_metrics["pretrained_checkpoint"] = str(args.pretrained_checkpoint)
    test_metrics["loss_type"] = args.loss_type
    test_metrics["train_label_variance"] = args.train_label_variance
    test_metrics["output_loss_weights"] = args.output_loss_weights
    (args.output_dir / "test_regression_summary.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_predictions(args.output_dir, "test", test_pred, test_target)
    print("=== Test metrics ===")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"Training time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
