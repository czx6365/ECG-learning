#!/usr/bin/env python3
"""Train identical frozen-representation linear probes for z_e, z_q, and HeartLang."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LABEL_NAMES = ["OMI", "NOMI", "Clinical Control", "Healthy"]
REPRESENTATIONS = ["z_e", "z_q", "heartlang"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Information-flow frozen linear probes")
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument(
        "--representation",
        action="append",
        choices=["z_e", "z_q", "heartlang", "z_e_mean_max", "z_q_mean_max"],
        help="Representation to probe; repeat as needed. Defaults to z_e, z_q, and heartlang.",
    )
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
    parser.add_argument("--standardize", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_for_step(step: int, total_steps: int, warmup_steps: int, args) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        alpha = step / max(1, warmup_steps)
        return args.warmup_lr + alpha * (args.lr - args.warmup_lr)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(target, pred, multioutput="uniform_average")),
        "per_output_r2": {
            name: float(r2_score(target[:, idx], pred[:, idx]))
            for idx, name in enumerate(LABEL_NAMES)
        },
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
    }


def classification_metrics(logits: np.ndarray, target: np.ndarray) -> dict:
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    pred = prob.argmax(axis=1)
    per_class_auc = {}
    for idx, name in enumerate(LABEL_NAMES):
        per_class_auc[name] = float(roc_auc_score((target == idx).astype(np.int64), prob[:, idx]))
    return {
        "accuracy": float(accuracy_score(target, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, average="macro", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(target, prob, multi_class="ovr", average="macro")),
        "per_class_auc": per_class_auc,
    }


def prepare_features(run_dir: Path, representation: str, standardize: bool):
    arrays = {
        split: np.load(run_dir / f"{split}_{representation}.npy").astype(np.float32)
        for split in ("train", "val", "test")
    }
    labels = {
        split: np.load(run_dir / f"{split}_labels.npy").astype(np.float32)
        for split in ("train", "val", "test")
    }
    normalization = None
    if standardize:
        mean = arrays["train"].mean(axis=0, keepdims=True)
        std = arrays["train"].std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        arrays = {split: (values - mean) / std for split, values in arrays.items()}
        normalization = {"mean_shape": list(mean.shape), "zero_variance_dims": int(np.sum(std == 1.0))}
    return arrays, labels, normalization


def make_loader(x, y, batch_size, shuffle, seed, num_workers, drop_last=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=generator,
    )


def run_epoch(model, loader, optimizer, args, device, epoch, total_steps, warmup_steps, task, train):
    model.train(train)
    outputs, targets = [], []
    loss_sum = 0.0
    class_weights = getattr(args, "active_class_weights", None)
    for step, (x, y) in enumerate(loader):
        x = x.float().to(device)
        y = y.to(device)
        if train:
            global_step = (epoch - 1) * len(loader) + step
            lr = lr_for_step(global_step, total_steps, warmup_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = F.mse_loss(output, y.float()) if task == "regression" else F.cross_entropy(output, y.long(), weight=class_weights)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        outputs.append(output.detach().cpu())
        targets.append(y.detach().cpu())
        loss_sum += float(loss.item())
    output_np = torch.cat(outputs).numpy()
    target_np = torch.cat(targets).numpy()
    metrics = regression_metrics(output_np, target_np) if task == "regression" else classification_metrics(output_np, target_np.astype(np.int64))
    metrics["loss"] = loss_sum / max(1, len(loader))
    return metrics


def train_task(arrays, labels, input_dim, task, args, output_dir: Path, device) -> dict:
    set_seed(args.seed)
    task_labels = labels if task == "regression" else {k: v.argmax(axis=1).astype(np.int64) for k, v in labels.items()}
    loaders = {
        "train": make_loader(arrays["train"], task_labels["train"], args.batch_size, True, args.seed, args.num_workers, True),
        "val": make_loader(arrays["val"], task_labels["val"], int(1.5 * args.batch_size), False, args.seed, args.num_workers),
        "test": make_loader(arrays["test"], task_labels["test"], int(1.5 * args.batch_size), False, args.seed, args.num_workers),
    }
    model = nn.Linear(input_dim, 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if task == "classification" and args.class_weight == "normalized_inverse_freq":
        counts = np.bincount(task_labels["train"], minlength=4).astype(np.float64)
        args.active_class_weights = torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=device)
    else:
        args.active_class_weights = None
    total_steps = args.epochs * len(loaders["train"])
    warmup_steps = args.warmup_epochs * len(loaders["train"])
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, args, device, epoch, total_steps, warmup_steps, task, True)
        with torch.no_grad():
            val_metrics = run_epoch(model, loaders["val"], optimizer, args, device, epoch, total_steps, warmup_steps, task, False)
        score = val_metrics["r2"] if task == "regression" else val_metrics["macro_roc_auc"]
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        test_metrics = run_epoch(model, loaders["test"], optimizer, args, device, args.epochs, total_steps, warmup_steps, task, False)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "best_epoch": best_epoch, "best_val_score": best_score}, output_dir / f"{task}_best.pth")
    (output_dir / f"{task}_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"best_epoch": best_epoch, "best_val_score": best_score, "test": test_metrics}
    (output_dir / f"{task}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.feature_root = args.feature_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    representations = args.representation or REPRESENTATIONS
    rows = []
    manifest = {
        "seed": args.seed,
        "epochs": args.epochs,
        "early_stopping": "disabled for all representations; best epoch selected on validation metric",
        "optimizer": "AdamW",
        "scheduler": "per-step linear warmup then cosine decay",
        "standardize_from_train_split": args.standardize,
        "class_weight": args.class_weight,
        "runs": [],
    }
    for run_name in args.run:
        run_dir = args.feature_root / run_name
        run_result = {"model": run_name, "representations": {}}
        for representation in representations:
            arrays, labels, normalization = prepare_features(run_dir, representation, args.standardize)
            rep_out = args.output_dir / run_name / representation
            regression = train_task(arrays, labels, arrays["train"].shape[1], "regression", args, rep_out, device)
            classification = train_task(arrays, labels, arrays["train"].shape[1], "classification", args, rep_out, device)
            reg = regression["test"]
            cls = classification["test"]
            row = {
                "model": run_name,
                "representation": representation,
                "r2": reg["r2"],
                "omi_r2": reg["per_output_r2"]["OMI"],
                "nomi_r2": reg["per_output_r2"]["NOMI"],
                "control_r2": reg["per_output_r2"]["Clinical Control"],
                "healthy_r2": reg["per_output_r2"]["Healthy"],
                "accuracy": cls["accuracy"],
                "balanced_accuracy": cls["balanced_accuracy"],
                "macro_f1": cls["macro_f1"],
                "macro_roc_auc": cls["macro_roc_auc"],
                "regression_best_epoch": regression["best_epoch"],
                "classification_best_epoch": classification["best_epoch"],
            }
            rows.append(row)
            run_result["representations"][representation] = {
                "normalization": normalization,
                "regression": regression,
                "classification": classification,
            }
        if {"z_e", "z_q", "heartlang"}.issubset(run_result["representations"]):
            z_e = run_result["representations"]["z_e"]
            z_q = run_result["representations"]["z_q"]
            hl = run_result["representations"]["heartlang"]
            run_result["gaps"] = {
                "delta_vq_r2": z_q["regression"]["test"]["r2"] - z_e["regression"]["test"]["r2"],
                "delta_vq_balanced_accuracy": z_q["classification"]["test"]["balanced_accuracy"] - z_e["classification"]["test"]["balanced_accuracy"],
                "delta_pretrain_r2": hl["regression"]["test"]["r2"] - z_q["regression"]["test"]["r2"],
                "delta_pretrain_balanced_accuracy": hl["classification"]["test"]["balanced_accuracy"] - z_q["classification"]["test"]["balanced_accuracy"],
            }
        manifest["runs"].append(run_result)
    with (args.output_dir / "representation_probe_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "information_flow_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
