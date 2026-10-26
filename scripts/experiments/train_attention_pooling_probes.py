#!/usr/bin/env python3
"""Train attention pooling plus a linear downstream head on frozen heartbeat tokens."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_information_flow_linear_probes import (
    LABEL_NAMES,
    classification_metrics,
    lr_for_step,
    regression_metrics,
    set_seed,
)


class TokenFeatureDataset(Dataset):
    def __init__(self, run_dir: Path, split: str, representation: str, task: str):
        self.tokens = np.load(run_dir / f"{split}_{representation}_tokens.npy", mmap_mode="r")
        self.valid = np.load(run_dir / f"{split}_valid_heartbeat_mask.npy", mmap_mode="r")
        labels = np.load(run_dir / f"{split}_labels.npy", mmap_mode="r").astype(np.float32)
        self.labels = labels if task == "regression" else labels.argmax(axis=1).astype(np.int64)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return (
            np.array(self.tokens[index], copy=True),
            np.array(self.valid[index], copy=True),
            np.array(self.labels[index], copy=True),
            np.int64(index),
        )


class AttentionPoolingHead(nn.Module):
    def __init__(self, code_dim: int = 64, hidden_dim: int = 64, outputs: int = 4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Linear(code_dim, outputs)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, return_attention: bool = False):
        valid = valid.bool()
        scores = self.attention(tokens).squeeze(-1)
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        output = self.head(pooled)
        return (output, weights) if return_attention else output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Frozen-heartbeat attention-pooling probes")
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--representation", action="append", choices=["z_e", "z_q"], required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--class_weight", choices=["none", "normalized_inverse_freq"], default="normalized_inverse_freq")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_loaders(run_dir, representation, task, args):
    datasets = {
        split: TokenFeatureDataset(run_dir, split, representation, task)
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True, generator=generator),
        "val": DataLoader(datasets["val"], batch_size=int(1.5 * args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=int(1.5 * args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }
    return datasets, loaders


def run_epoch(model, loader, optimizer, args, device, epoch, total_steps, warmup_steps, task, train, class_weights=None):
    model.train(train)
    outputs, targets = [], []
    loss_sum = 0.0
    for step, (tokens, valid, target, _indices) in enumerate(loader):
        tokens = tokens.float().to(device, non_blocking=True)
        valid = valid.bool().to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if train:
            global_step = (epoch - 1) * len(loader) + step
            lr = lr_for_step(global_step, total_steps, warmup_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        output = model(tokens, valid)
        loss = F.mse_loss(output, target.float()) if task == "regression" else F.cross_entropy(output, target.long(), weight=class_weights)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        outputs.append(output.detach().cpu())
        targets.append(target.detach().cpu())
        loss_sum += float(loss.item())
    output_np = torch.cat(outputs).numpy()
    target_np = torch.cat(targets).numpy()
    metrics = regression_metrics(output_np, target_np) if task == "regression" else classification_metrics(output_np, target_np.astype(np.int64))
    metrics["loss"] = loss_sum / max(1, len(loader))
    return metrics


@torch.inference_mode()
def save_attention_weights(model, loader, output_dir: Path, split: str, device, class_names):
    # Export every ECG in stable dataset order.  The training loader intentionally
    # shuffles and drops its last partial batch, so it must not be reused here.
    loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=loader.num_workers,
        pin_memory=True,
    )
    count = len(loader.dataset)
    seq_len = loader.dataset.valid.shape[1]
    weights_out = np.lib.format.open_memmap(output_dir / f"{split}_attention_weights.npy", mode="w+", dtype=np.float32, shape=(count, seq_len))
    top_indices = np.full((count, 10), -1, dtype=np.int16)
    valid_counts = np.zeros(count, dtype=np.int16)
    labels = np.zeros(count, dtype=np.int64)
    for tokens, valid, target, indices in loader:
        tokens = tokens.float().to(device, non_blocking=True)
        valid_device = valid.bool().to(device, non_blocking=True)
        _output, weights = model(tokens, valid_device, return_attention=True)
        weights_np = weights.cpu().numpy()
        indices_np = indices.numpy()
        weights_out[indices_np] = weights_np
        valid_counts[indices_np] = valid.sum(dim=1).numpy().astype(np.int16)
        labels[indices_np] = target.numpy().argmax(axis=1) if target.ndim == 2 else target.numpy().astype(np.int64)
        for local_idx, record_idx in enumerate(indices_np):
            k = min(10, int(valid_counts[record_idx]))
            if k:
                top_indices[record_idx, :k] = np.argsort(weights_np[local_idx])[-k:][::-1].astype(np.int16)
    weights_out.flush()
    np.save(output_dir / f"{split}_top10_heartbeat_indices.npy", top_indices)
    np.save(output_dir / f"{split}_valid_heartbeat_counts.npy", valid_counts)

    weights = np.load(output_dir / f"{split}_attention_weights.npy", mmap_mode="r")
    summaries = {}
    for class_idx, class_name in enumerate(class_names):
        class_selected = labels == class_idx
        selected = class_selected & (valid_counts > 0)
        if not class_selected.any():
            continue
        if not selected.any():
            summaries[class_name] = {
                "records": int(class_selected.sum()),
                "valid_records": 0,
                "empty_records": int(class_selected.sum()),
            }
            continue
        class_weights = np.asarray(weights[selected])
        class_valid_counts = valid_counts[selected].astype(np.float32)
        entropy = -(class_weights * np.log(np.clip(class_weights, 1e-12, None))).sum(axis=1)
        effective = 1.0 / np.square(class_weights).sum(axis=1).clip(min=1e-12)
        normalized_entropy = entropy / np.log(np.maximum(class_valid_counts, 2.0))
        sorted_weights = np.sort(class_weights, axis=1)[:, ::-1]
        summaries[class_name] = {
            "records": int(class_selected.sum()),
            "valid_records": int(selected.sum()),
            "empty_records": int((class_selected & (valid_counts == 0)).sum()),
            "mean_valid_heartbeat_count": float(class_valid_counts.mean()),
            "mean_attention_entropy": float(entropy.mean()),
            "mean_normalized_attention_entropy": float(normalized_entropy.mean()),
            "mean_effective_heartbeat_count": float(effective.mean()),
            "mean_effective_fraction": float((effective / class_valid_counts).mean()),
            "mean_top1_mass": float(sorted_weights[:, :1].sum(axis=1).mean()),
            "mean_top5_mass": float(sorted_weights[:, :5].sum(axis=1).mean()),
            "mean_top10_mass": float(sorted_weights[:, :10].sum(axis=1).mean()),
        }
    return summaries


def train_task(run_dir, representation, task, args, output_dir, device):
    set_seed(args.seed)
    datasets, loaders = make_loaders(run_dir, representation, task, args)
    model = AttentionPoolingHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = None
    if task == "classification" and args.class_weight == "normalized_inverse_freq":
        counts = np.bincount(datasets["train"].labels, minlength=4).astype(np.float64)
        class_weights = torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=device)
    total_steps = args.epochs * len(loaders["train"])
    warmup_steps = args.warmup_epochs * len(loaders["train"])
    best_score, best_epoch, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, args, device, epoch, total_steps, warmup_steps, task, True, class_weights)
        with torch.no_grad():
            val_metrics = run_epoch(model, loaders["val"], optimizer, args, device, epoch, total_steps, warmup_steps, task, False, class_weights)
        score = val_metrics["r2"] if task == "regression" else val_metrics["macro_roc_auc"]
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(json.dumps({"representation": representation, "task": task, "epoch": epoch, "val_score": score}), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        test_metrics = run_epoch(model, loaders["test"], optimizer, args, device, args.epochs, total_steps, warmup_steps, task, False, class_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "best_epoch": best_epoch, "best_val_score": best_score}, output_dir / "checkpoint-best.pth")
    (output_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    attention_summary = {
        split: save_attention_weights(model, loaders[split], output_dir, split, device, LABEL_NAMES)
        for split in ("train", "val", "test")
    }
    summary = {"best_epoch": best_epoch, "best_val_score": best_score, "test": test_metrics, "attention_distribution": attention_summary}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    args.feature_root = args.feature_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    run_dir = args.feature_root / args.run
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for representation in args.representation:
        results = {}
        for task in ("regression", "classification"):
            results[task] = train_task(run_dir, representation, task, args, args.output_dir / args.run / representation / task, device)
        reg, cls = results["regression"]["test"], results["classification"]["test"]
        rows.append({
            "model": args.run,
            "representation": representation,
            "pooling": "attention",
            "r2": reg["r2"],
            "omi_r2": reg["per_output_r2"]["OMI"],
            "nomi_r2": reg["per_output_r2"]["NOMI"],
            "control_r2": reg["per_output_r2"]["Clinical Control"],
            "healthy_r2": reg["per_output_r2"]["Healthy"],
            "accuracy": cls["accuracy"],
            "balanced_accuracy": cls["balanced_accuracy"],
            "macro_f1": cls["macro_f1"],
            "macro_roc_auc": cls["macro_roc_auc"],
            "regression_best_epoch": results["regression"]["best_epoch"],
            "classification_best_epoch": results["classification"]["best_epoch"],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "attention_probe_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
