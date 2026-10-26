#!/usr/bin/env python3
"""Pretrain HeartLang on Morphology-V2 SID sequences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models.modeling_heartlang_sid import HeartLangSIDForMLM
from heartlang.utils.utils import set_seed


class SIDPretrainDataset(Dataset):
    def __init__(self, sid_data_dir: Path, split: str, max_records: int = 0):
        self.sid = np.load(sid_data_dir / f"{split}_sid.npy", mmap_mode="r")
        self.chans = np.load(sid_data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
        self.times = np.load(sid_data_dir / f"{split}_data_in_times.npy", mmap_mode="r")
        self.length = min(len(self.sid), len(self.chans), len(self.times))
        if max_records > 0:
            self.length = min(self.length, int(max_records))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            np.array(self.sid[idx], copy=True),
            np.array(self.chans[idx], copy=True),
            np.array(self.times[idx], copy=True),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HeartLangSID MLM pretraining")
    parser.add_argument("--sid_data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_ckpt_freq", type=int, default=50)
    parser.add_argument("--val_freq", type=int, default=10)
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
    parser.add_argument("--mask_ratio", type=float, default=0.5)
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
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_train_records", type=int, default=0)
    parser.add_argument("--max_val_records", type=int, default=0)
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def valid_sid_mask(sid, chans, times, args):
    valid = (sid >= 0) & (sid < args.vocab_size)
    if args.use_spatial_embedding:
        valid &= chans > 0
    if args.use_temporal_embedding:
        valid &= times > 0
    return valid


def random_masking_valid(valid, mask_ratio):
    mask = torch.zeros_like(valid, dtype=torch.bool)
    for row in range(valid.shape[0]):
        idx = torch.nonzero(valid[row], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        keep = max(1, int(round(float(mask_ratio) * idx.numel())))
        chosen = idx[torch.randperm(idx.numel(), device=valid.device)[:keep]]
        mask[row, chosen] = True
    return mask


def build_model(args):
    return HeartLangSIDForMLM(
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


def lr_for_step(global_step, total_steps, warmup_steps, args):
    if warmup_steps > 0 and global_step < warmup_steps:
        alpha = global_step / max(1, warmup_steps)
        return args.warmup_lr + alpha * (args.lr - args.warmup_lr)
    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, scaler, args, epoch, tag=None, best_metric=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{epoch}.pth" if tag is None else f"checkpoint-{tag}.pth"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "args": vars(args),
        "model_type": "heartlang_sid_mlm",
        "best_metric": best_metric,
    }
    torch.save(payload, args.output_dir / name)
    torch.save(payload, args.output_dir / "checkpoint.pth")


def load_checkpoint(model, optimizer, scaler, args, device):
    if args.resume is None:
        return 0, None
    ckpt_path = resolve_path(args.resume)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), ckpt.get("best_metric")


def run_epoch(model, loader, optimizer, scaler, args, device, epoch, total_steps, warmup_steps, train=True):
    model.train(train)
    sums = {"loss": 0.0, "mlm_acc": 0.0, "token_entropy": 0.0, "masked_tokens": 0.0}
    count = 0
    max_steps = args.max_train_steps_per_epoch if train else args.max_eval_steps

    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        sid = batch[0].long().to(device, non_blocking=True)
        chans = batch[1].long().to(device, non_blocking=True)
        times = batch[2].long().to(device, non_blocking=True)
        valid = valid_sid_mask(sid, chans, times, args)
        mask = random_masking_valid(valid, args.mask_ratio)
        labels = sid[mask]
        if labels.numel() == 0:
            continue

        if train:
            global_step = (epoch - 1) * len(loader) + step
            lr = lr_for_step(global_step, total_steps, warmup_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(
                sid,
                mask_bool_matrix=mask,
                in_chan_matrix=chans,
                in_time_matrix=times,
            )
            loss = F.cross_entropy(logits.float(), labels.long())

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
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
        sums["loss"] += float(loss.item())
        sums["mlm_acc"] += float(acc.item())
        sums["token_entropy"] += float(entropy.item())
        sums["masked_tokens"] += float(labels.numel())
        count += 1

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            phase = "train" if train else "val"
            print(
                f"{phase} epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss.item():.4f} acc={acc.item():.4f} masked={labels.numel()}",
                flush=True,
            )

    if count == 0:
        return {key: float("nan") for key in sums}
    return {key: value / count for key, value in sums.items()}


def write_history(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.sid_data_dir = resolve_path(args.sid_data_dir)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is not None:
        args.log_dir = resolve_path(args.log_dir)
        args.log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)

    train_ds = SIDPretrainDataset(args.sid_data_dir, "train", args.max_train_records)
    val_ds = SIDPretrainDataset(args.sid_data_dir, "val", args.max_val_records)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(1.5 * args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scaler, args, device)
    if best_val_loss is None:
        best_val_loss = float("inf")

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    print(f"SID data: {args.sid_data_dir}")
    print(f"Train/val records: {len(train_ds)}/{len(val_ds)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Start HeartLangSID MLM pretraining for {args.epochs} epochs")

    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path = args.output_dir / "training_history.csv"
    start_time = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_stats = run_epoch(
            model, train_loader, optimizer, scaler, args, device, epoch, total_steps, warmup_steps, train=True
        )
        row = {f"train_{k}": v for k, v in train_stats.items()}
        row["epoch"] = epoch

        should_val = epoch == 1 or epoch % args.val_freq == 0 or epoch == args.epochs
        if should_val:
            with torch.no_grad():
                val_stats = run_epoch(
                    model, val_loader, optimizer, scaler, args, device, epoch, total_steps, warmup_steps, train=False
                )
            row.update({f"val_{k}": v for k, v in val_stats.items()})
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                save_checkpoint(model, optimizer, scaler, args, epoch, tag="best", best_metric=best_val_loss)

        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, scaler, args, epoch, best_metric=best_val_loss)
        write_history(history_path, row)
        with (args.output_dir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    print(f"Training time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
