#!/usr/bin/env python3
"""Linear-probe four-class classification on a pretrained HeartLangSID backbone."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models.modeling_heartlang_sid import HeartLangSIDBackbone
from heartlang.utils.utils import set_seed


LABEL_NAMES = ["OMI", "NOMI", "Clinical Control", "Healthy"]


class SIDClassificationDataset(Dataset):
    def __init__(self, sid_data_dir: Path, split: str):
        self.sid = np.load(sid_data_dir / f"{split}_sid.npy", mmap_mode="r")
        self.chans = np.load(sid_data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
        self.times = np.load(sid_data_dir / f"{split}_data_in_times.npy", mmap_mode="r")
        labels = np.load(sid_data_dir / f"{split}_labels.npy", mmap_mode="r", allow_pickle=True)
        labels = np.asarray(labels)
        if labels.ndim == 2:
            self.labels = labels.argmax(axis=1).astype(np.int64)
            self.one_hot = labels.astype(np.int64)
        elif labels.ndim == 1:
            self.labels = labels.astype(np.int64)
            self.one_hot = np.eye(len(LABEL_NAMES), dtype=np.int64)[self.labels]
        else:
            raise ValueError(f"Unsupported label shape: {labels.shape}")

    def __len__(self):
        return len(self.sid)

    def __getitem__(self, idx):
        return (
            np.array(self.sid[idx], copy=True),
            np.int64(self.labels[idx]),
            np.array(self.chans[idx], copy=True),
            np.array(self.times[idx], copy=True),
        )


class HeartLangSIDForClassification(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.backbone = HeartLangSIDBackbone(**kwargs)
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, sid_ids, in_chan_matrix=None, in_time_matrix=None):
        features = self.backbone(
            sid_ids,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
        )
        return self.head(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HeartLangSID frozen-backbone linear classification")
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
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--class_weight",
        choices=["none", "inverse_freq", "normalized_inverse_freq"],
        default="none",
        help="Class-weighting mode for CrossEntropyLoss. normalized_inverse_freq uses total/(K*count).",
    )
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


def build_model(args, num_classes: int):
    return HeartLangSIDForClassification(
        num_classes=num_classes,
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
        "model_type": "heartlang_sid_frozen_backbone_linear_classification",
        "metrics": metrics or {},
    }
    torch.save(payload, args.output_dir / name)
    torch.save(payload, args.output_dir / "checkpoint.pth")


def safe_auc(y_true, prob, average):
    try:
        return float(roc_auc_score(y_true, prob, multi_class="ovr", average=average))
    except ValueError:
        return float("nan")


def metrics_from_arrays(logits, target, label_names):
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = prob / prob.sum(axis=1, keepdims=True)
    pred = prob.argmax(axis=1)
    per_class_auc = {}
    for idx, name in enumerate(label_names):
        try:
            per_class_auc[name] = float(roc_auc_score((target == idx).astype(np.int64), prob[:, idx]))
        except ValueError:
            per_class_auc[name] = float("nan")
    return {
        "accuracy": float(accuracy_score(target, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(target, pred, average="weighted", zero_division=0)),
        "macro_roc_auc": safe_auc(target, prob, "macro"),
        "weighted_roc_auc": safe_auc(target, prob, "weighted"),
        "per_class_auc": per_class_auc,
        "per_class_recall": {
            label_names[i]: float(recall_score(target == i, pred == i, zero_division=0))
            for i in range(len(label_names))
        },
        "omi_recall": float(recall_score(target == 0, pred == 0, zero_division=0)),
        "confusion_matrix": confusion_matrix(target, pred, labels=list(range(len(label_names)))).tolist(),
    }


def make_class_weights(labels, num_classes: int, mode: str, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"Cannot compute class weights with empty classes: {counts.tolist()}")
    if mode == "none":
        return None, counts.tolist()
    if mode == "inverse_freq":
        weights = 1.0 / counts
        weights = weights / weights.mean()
    elif mode == "normalized_inverse_freq":
        weights = counts.sum() / (num_classes * counts)
    else:
        raise ValueError(f"Unsupported class_weight: {mode}")
    return torch.as_tensor(weights, dtype=torch.float32, device=device), counts.tolist()


def run_epoch(
    model,
    loader,
    optimizer,
    args,
    device,
    epoch,
    total_steps,
    warmup_steps,
    train=True,
    class_weights=None,
):
    model.train(train)
    logits_out = []
    targets = []
    loss_sum = 0.0
    steps = 0
    for step, batch in enumerate(loader):
        sid = batch[0].long().to(device, non_blocking=True)
        target = batch[1].long().to(device, non_blocking=True)
        chans = batch[2].long().to(device, non_blocking=True)
        times = batch[3].long().to(device, non_blocking=True)
        if train:
            global_step = (epoch - 1) * len(loader) + step
            lr = lr_for_step(global_step, total_steps, warmup_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

        logits = model(sid, in_chan_matrix=chans, in_time_matrix=times)
        loss = F.cross_entropy(logits, target, weight=class_weights)
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
        logits_out.append(logits.detach().cpu())
        targets.append(target.detach().cpu())
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == len(loader):
            phase = "train" if train else "eval"
            pred = logits.argmax(dim=1)
            acc = (pred == target).float().mean().item()
            print(
                f"{phase} epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss.item():.5f} acc={acc:.4f}",
                flush=True,
            )
    logits_np = torch.cat(logits_out).numpy()
    target_np = torch.cat(targets).numpy()
    metrics = metrics_from_arrays(logits_np, target_np, LABEL_NAMES)
    metrics["loss"] = float(loss_sum / max(steps, 1))
    return metrics, logits_np, target_np


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


def save_predictions(output_dir: Path, split: str, logits, target):
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = prob / prob.sum(axis=1, keepdims=True)
    pred = prob.argmax(axis=1)
    fieldnames = ["index", "target", "pred"] + [f"prob_{name}" for name in LABEL_NAMES]
    with (output_dir / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(target)):
            row = {
                "index": idx,
                "target": LABEL_NAMES[int(target[idx])],
                "pred": LABEL_NAMES[int(pred[idx])],
            }
            for class_idx, name in enumerate(LABEL_NAMES):
                row[f"prob_{name}"] = float(prob[idx, class_idx])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.sid_data_dir = resolve_path(args.sid_data_dir)
    args.pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)

    train_ds = SIDClassificationDataset(args.sid_data_dir, "train")
    val_ds = SIDClassificationDataset(args.sid_data_dir, "val")
    test_ds = SIDClassificationDataset(args.sid_data_dir, "test")
    num_classes = len(LABEL_NAMES)
    model = build_model(args, num_classes).to(device)
    load_pretrained_backbone(model, args.pretrained_checkpoint, device)
    class_weights, train_counts = make_class_weights(
        train_ds.labels,
        num_classes=num_classes,
        mode=args.class_weight,
        device=device,
    )
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
    print(f"Label names: {LABEL_NAMES}")
    print(f"Train label counts: {train_counts}")
    print(f"Val label counts: {np.bincount(val_ds.labels, minlength=num_classes).tolist()}")
    print(f"Test label counts: {np.bincount(test_ds.labels, minlength=num_classes).tolist()}")
    if class_weights is not None:
        print(f"Class weight mode: {args.class_weight}")
        print(f"Class weights: {[float(x) for x in class_weights.detach().cpu().tolist()]}")
    else:
        print("Class weight mode: none")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_val_macro_auc = -float("inf")
    history_path = args.output_dir / "training_history.csv"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model,
            loaders["train"],
            optimizer,
            args,
            device,
            epoch,
            total_steps,
            warmup_steps,
            train=True,
            class_weights=class_weights,
        )
        with torch.no_grad():
            val_metrics, _, _ = run_epoch(
                model,
                loaders["val"],
                optimizer,
                args,
                device,
                epoch,
                total_steps,
                warmup_steps,
                train=False,
                class_weights=class_weights,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_macro_roc_auc": train_metrics["macro_roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_roc_auc": val_metrics["macro_roc_auc"],
            "val_per_class_auc": val_metrics["per_class_auc"],
            "val_omi_recall": val_metrics["omi_recall"],
        }
        write_history(history_path, row)
        with (args.output_dir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = val_metrics["macro_roc_auc"]
        if math.isnan(score):
            score = val_metrics["balanced_accuracy"]
        if score > best_val_macro_auc:
            best_val_macro_auc = score
            save_checkpoint(model, optimizer, args, epoch, tag="best", metrics=val_metrics)
        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, args, epoch, metrics=val_metrics)

    best_ckpt = torch.load(args.output_dir / "checkpoint-best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"], strict=True)
    with torch.no_grad():
        test_metrics, test_logits, test_target = run_epoch(
            model,
            loaders["test"],
            optimizer,
            args,
            device,
            epoch=args.epochs,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            train=False,
            class_weights=class_weights,
        )
    test_metrics["best_val_selection_score"] = float(best_val_macro_auc)
    test_metrics["class_weight"] = args.class_weight
    test_metrics["class_weights"] = (
        None if class_weights is None else [float(x) for x in class_weights.detach().cpu().tolist()]
    )
    test_metrics["label_names"] = LABEL_NAMES
    test_metrics["n_test"] = int(len(test_ds))
    test_metrics["pretrained_checkpoint"] = str(args.pretrained_checkpoint)
    (args.output_dir / "test_classification_summary.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_predictions(args.output_dir, "test", test_logits, test_target)
    print("=== Test metrics ===")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"Training time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
