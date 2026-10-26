#!/usr/bin/env python3
"""Train Curriculum V1 C1: heartbeat reconstruction plus EMA VQ only.

This experiment intentionally removes lead/time/position context before SID
assignment. Each 96-point heartbeat is encoded independently:

heartbeat waveform -> Conv1D morphology encoder -> EMA VQ -> decoder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import heartlang.utils.utils as utils
from heartlang.utils.norm_ema_quantizer import NormEMAVectorQuantizer
from heartlang.utils.optim_factory import create_optimizer


class MorphologyEncoder(nn.Module):
    """A per-heartbeat encoder. No batch norm, no sequence context."""

    def __init__(self, token_len: int = 96, code_dim: int = 64):
        super().__init__()
        self.token_len = token_len
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(256 * math.ceil(token_len / 8), 256),
            nn.GELU(),
            nn.Linear(256, code_dim),
        )

    def forward(self, beats: torch.Tensor) -> torch.Tensor:
        z = self.net(beats.unsqueeze(1))
        return F.normalize(z, p=2, dim=-1)


class MorphologyDecoder(nn.Module):
    """Decode one quantized morphology vector back to one 96-point heartbeat."""

    def __init__(self, token_len: int = 96, code_dim: int = 64):
        super().__init__()
        self.token_len = token_len
        self.seed_len = math.ceil(token_len / 8)
        self.fc = nn.Sequential(
            nn.Linear(code_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256 * self.seed_len),
            nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.fc(z_q).view(z_q.shape[0], 256, self.seed_len)
        x = self.deconv(x).squeeze(1)
        return x[:, : self.token_len]


class MorphologyOnlyVQ(nn.Module):
    """Morphology-only tokenizer compatible with existing SID eval scripts."""

    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 64,
        token_len: int = 96,
        ema_decay: float = 0.99,
        quantize_kmeans_init: bool = False,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.token_len = int(token_len)
        self.encoder = MorphologyEncoder(token_len=token_len, code_dim=codebook_dim)
        self.decoder = MorphologyDecoder(token_len=token_len, code_dim=codebook_dim)
        self.quantize = NormEMAVectorQuantizer(
            n_embed=codebook_size,
            embedding_dim=codebook_dim,
            beta=1.0,
            decay=ema_decay,
            kmeans_init=quantize_kmeans_init,
        )

    @staticmethod
    def valid_mask(x: torch.Tensor, in_chan_matrix=None, in_time_matrix=None) -> torch.Tensor:
        valid = torch.abs(x).amax(dim=-1) > 1e-8
        if in_chan_matrix is not None:
            valid = valid & (in_chan_matrix > 0)
        if in_time_matrix is not None:
            valid = valid & (in_time_matrix > 0)
        return valid

    def encode_to_quantizer_features(self, x, in_chan_matrix=None, in_time_matrix=None):
        valid = self.valid_mask(x, in_chan_matrix, in_time_matrix)
        z_full = torch.zeros(*x.shape[:2], self.codebook_dim, device=x.device, dtype=x.dtype)
        if valid.any():
            z_full[valid] = self.encoder(x[valid])
        return z_full

    def encode_valid(self, x, in_chan_matrix=None, in_time_matrix=None):
        valid = self.valid_mask(x, in_chan_matrix, in_time_matrix)
        if not valid.any():
            empty_z = torch.empty(0, self.codebook_dim, device=x.device, dtype=x.dtype)
            empty_beats = torch.empty(0, self.token_len, device=x.device, dtype=x.dtype)
            return valid, empty_beats, empty_z, empty_z, torch.empty(0, dtype=torch.long, device=x.device), x.sum() * 0.0
        beats = x[valid]
        z = self.encoder(beats)
        z_q, quant_loss, embed_ind = self.quantize(z)
        return valid, beats, z, z_q, embed_ind, quant_loss

    def forward(self, x, in_chan_matrix=None, in_time_matrix=None):
        # SID assignment 前只用 heartbeat waveform；lead/time 只用于过滤 padding。
        valid, beats, _z, z_q, embed_ind, quant_loss = self.encode_valid(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )
        if beats.numel() == 0:
            rec_loss = x.sum() * 0.0
            rec = torch.zeros_like(beats)
        else:
            rec = self.decoder(z_q)
            rec_loss = F.mse_loss(rec, beats)
        loss = quant_loss + rec_loss
        split = "train" if self.training else "val"
        log = {
            f"{split}/quant_loss": quant_loss.detach(),
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/valid_tokens": torch.as_tensor(float(beats.shape[0]), device=x.device),
        }
        return loss, log

    @torch.no_grad()
    def reconstruct(self, x, in_chan_matrix=None, in_time_matrix=None):
        valid, beats, _z, z_q, embed_ind, _quant_loss = self.encode_valid(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )
        rec_full = torch.zeros_like(x)
        if beats.numel() > 0:
            rec_full[valid] = self.decoder(z_q)
        ids_full = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        if embed_ind.numel() > 0:
            ids_full[valid] = embed_ind
        return rec_full, ids_full

    @torch.no_grad()
    def get_codebook_indices(self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        _rec, ids = self.reconstruct(x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix)
        return ids

    @torch.no_grad()
    def get_codebook_quantize(self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        return self.encode_to_quantizer_features(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )


class IndexedC1Dataset(Dataset):
    """Read a grouped split through small indices without copying the 9.3 GB base arrays."""

    SOURCE_SPLITS = ("train", "val")

    def __init__(self, data_dir: Path, index_dir: Path, split: str):
        self.data = {
            name: np.load(data_dir / f"{name}_data.npy", mmap_mode="r")
            for name in self.SOURCE_SPLITS
        }
        self.chans = {
            name: np.load(data_dir / f"{name}_data_in_chans.npy", mmap_mode="r")
            for name in self.SOURCE_SPLITS
        }
        self.times = {
            name: np.load(data_dir / f"{name}_data_in_times.npy", mmap_mode="r")
            for name in self.SOURCE_SPLITS
        }
        self.source_code = np.load(index_dir / f"{split}_source_code.npy", mmap_mode="r")
        self.source_index = np.load(index_dir / f"{split}_source_index.npy", mmap_mode="r")
        if len(self.source_code) != len(self.source_index):
            raise RuntimeError(f"C1 index length mismatch for {split}")

    def __len__(self):
        return len(self.source_index)

    def __getitem__(self, idx: int):
        source = self.SOURCE_SPLITS[int(self.source_code[idx])]
        row = int(self.source_index[idx])
        return (
            np.array(self.data[source][row], copy=True),
            np.array(self.chans[source][row], copy=True),
            np.array(self.times[source][row], copy=True),
        )


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Curriculum V1 C1 heartbeat reconstruction and VQ training")
    parser.add_argument("--pretrain_data_dir", type=Path, required=True)
    parser.add_argument("--split_index_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_ckpt_freq", type=int, default=50)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--codebook_n_emd", type=int, default=1024)
    parser.add_argument("--codebook_emd_dim", type=int, default=64)
    parser.add_argument("--token_len", type=int, default=96)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--quantize_kmeans_init", action="store_true")
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=None, type=float, nargs="+")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decay_end", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--write_assignments", action="store_true")
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def build_loaders(args):
    data_dir = resolve_path(args.pretrain_data_dir)
    index_dir = resolve_path(args.split_index_dir)
    dataset_train = IndexedC1Dataset(data_dir, index_dir, "train")
    dataset_val = IndexedC1Dataset(data_dir, index_dir, "val")
    sampler_train = DistributedSampler(dataset_train, num_replicas=1, rank=0, shuffle=True)
    loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_dir, index_dir, dataset_train, dataset_val, sampler_train, loader_train, loader_val


def build_model(args) -> MorphologyOnlyVQ:
    return MorphologyOnlyVQ(
        codebook_size=args.codebook_n_emd,
        codebook_dim=args.codebook_emd_dim,
        token_len=args.token_len,
        ema_decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
    )


def save_checkpoint(model, optimizer, args, epoch, tag=None) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{epoch}.pth" if tag is None else f"checkpoint-{tag}.pth"
    path = args.output_dir / name
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "args": vars(args),
            "model_type": "morphology_only_vq",
        },
        path,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "args": vars(args),
            "model_type": "morphology_only_vq",
        },
        args.output_dir / "checkpoint.pth",
    )
    np.save(
        args.output_dir / "codebook_vectors.npy",
        model.quantize.embedding.weight.detach().cpu().numpy(),
    )
    return path


def load_checkpoint(model, optimizer, resume: Path, device: torch.device):
    patch_checkpoint_paths()
    ckpt = torch.load(resolve_path(resume), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


def train_one_epoch(model, loader, sampler, optimizer, device, epoch, args, lr_schedule, start_step):
    model.train()
    sampler.set_epoch(epoch - 1)
    model.quantize.reset_cluster_size(device)
    meters = defaultdict_float()
    step_count = 0
    for step, batch in enumerate(loader):
        if args.max_train_steps_per_epoch and step >= args.max_train_steps_per_epoch:
            break
        global_step = start_step + step
        if global_step < len(lr_schedule):
            set_lr(optimizer, lr_schedule[global_step])
        ecg = batch[0].float().to(device, non_blocking=True)
        chans = batch[1].to(device, non_blocking=True)
        times = batch[2].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss, log = model(ecg, in_chan_matrix=chans, in_time_matrix=times)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad) if args.clip_grad else utils.get_grad_norm_(model.parameters())
        optimizer.step()
        meters["loss"] += float(loss.item())
        meters["grad_norm"] += float(grad_norm)
        meters["lr"] += float(optimizer.param_groups[0]["lr"])
        for key, value in log.items():
            meters[key.split("/")[-1]] += float(value.detach().item())
        step_count += 1
        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"Epoch {epoch} step {step + 1}/{len(loader)} "
                f"loss={loss.item():.6f} rec={log['train/rec_loss'].item():.6f} "
                f"quant={log['train/quant_loss'].item():.6f} valid={int(log['train/valid_tokens'].item())}",
                flush=True,
            )
    stats = {key: value / max(step_count, 1) for key, value in meters.items()}
    usage = model.quantize.cluster_size.detach()
    stats["unused_code"] = int((usage <= 0).sum().item())
    return stats, step_count


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    model.quantize.reset_cluster_size(device)
    meters = defaultdict_float()
    total_valid = 0
    step_count = 0
    for step, batch in enumerate(loader):
        if args.max_eval_steps and step >= args.max_eval_steps:
            break
        ecg = batch[0].float().to(device, non_blocking=True)
        chans = batch[1].to(device, non_blocking=True)
        times = batch[2].to(device, non_blocking=True)
        loss, log = model(ecg, in_chan_matrix=chans, in_time_matrix=times)
        meters["loss"] += float(loss.item())
        for key, value in log.items():
            meters[key.split("/")[-1]] += float(value.detach().item())
        total_valid += int(log["val/valid_tokens"].item())
        step_count += 1
    stats = {key: value / max(step_count, 1) for key, value in meters.items()}
    usage = model.quantize.cluster_size.detach().cpu().numpy()
    total_usage = usage.sum()
    probs = usage[usage > 0] / total_usage if total_usage > 0 else np.array([])
    stats["valid_tokens"] = total_valid
    stats["active_code"] = int((usage > 0).sum())
    stats["unused_code"] = int((usage <= 0).sum())
    stats["perplexity"] = float(np.exp(-(probs * np.log(probs)).sum())) if probs.size else 0.0
    return stats


def defaultdict_float():
    from collections import defaultdict

    return defaultdict(float)


@torch.no_grad()
def write_assignments(model, data_dir: Path, split: str, loader, device, args):
    out = np.lib.format.open_memmap(
        args.output_dir / f"{split}_sid_assignments.npy",
        mode="w+",
        dtype=np.int32,
        shape=(len(loader.dataset), 256),
    )
    offset = 0
    for step, batch in enumerate(loader):
        ecg = batch[0].float().to(device, non_blocking=True)
        chans = batch[1].to(device, non_blocking=True)
        times = batch[2].to(device, non_blocking=True)
        ids = model.get_codebook_indices(ecg, in_chan_matrix=chans, in_time_matrix=times)
        bs = ids.shape[0]
        out[offset : offset + bs] = ids.cpu().numpy().astype(np.int32, copy=False)
        offset += bs
        if step == 0 or (step + 1) % 20 == 0:
            print(f"Wrote {split} SID assignments: {offset}/{len(loader.dataset)}", flush=True)
    out.flush()


def main():
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    if args.log_dir is not None:
        args.log_dir = resolve_path(args.log_dir)
        args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    utils.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    data_dir, index_dir, dataset_train, dataset_val, sampler_train, loader_train, loader_val = build_loaders(args)
    model = build_model(args).to(device)

    total_batch_size = args.batch_size
    base_lr = args.lr
    args.lr = total_batch_size / 128 * args.lr
    optimizer = create_optimizer(args, model)
    steps_per_epoch = len(loader_train)
    if args.max_train_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps_per_epoch)
    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    start_epoch = 1
    if args.resume:
        loaded_epoch = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch = loaded_epoch + 1
        print(f"Resumed from {args.resume}, next epoch={start_epoch}")

    config = {
        "experiment": "curriculum_v1_c1_word_reconstruction_vq",
        "stage": "C1",
        "data_dir": str(data_dir),
        "split_index_dir": str(index_dir),
        "split_index_summary": json.loads((index_dir / "summary.json").read_text(encoding="utf-8")),
        "train_records": len(dataset_train),
        "val_records": len(dataset_val),
        "codebook_size": args.codebook_n_emd,
        "codebook_dim": args.codebook_emd_dim,
        "token_len": args.token_len,
        "batch_size_records": args.batch_size,
        "base_lr_before_scaling": base_lr,
        "actual_lr_after_batch_scaling": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "epochs": args.epochs,
        "losses": {"waveform_mse": 1.0, "vq_commitment": 1.0},
        "disabled_objectives": ["physiology_pattern", "sentence_mlm", "cross_lead", "four_class"],
        "main_control_variable": "C1 heartbeat reconstruction and basic codebook only",
        "removed_before_sid": ["spatial embedding", "temporal embedding", "position embedding", "other heartbeat context"],
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    print(model, flush=True)

    if args.eval:
        stats = evaluate(model, loader_val, device, args)
        print(json.dumps({"val": stats}, ensure_ascii=False, indent=2), flush=True)
        if args.write_assignments:
            write_assignments(model, data_dir, "val", loader_val, device, args)
        return

    start_time = time.time()
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats, train_steps = train_one_epoch(
            model, loader_train, sampler_train, optimizer, device, epoch, args, lr_schedule, (epoch - 1) * steps_per_epoch
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            val_stats = evaluate(model, loader_val, device, args)
            row.update({f"val_{k}": v for k, v in val_stats.items()})
            print("Validation:", json.dumps(val_stats, ensure_ascii=False), flush=True)
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_path = save_checkpoint(model, optimizer, args, epoch, tag="best")
                print(f"Saved best checkpoint: {best_path} val_loss={best_val_loss:.6f}", flush=True)
        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs:
            ckpt_path = save_checkpoint(model, optimizer, args, epoch)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)
        with (args.output_dir / "log.txt").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = str(dt.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}", flush=True)
    write_assignments(model, data_dir, "val", loader_val, device, args)


if __name__ == "__main__":
    main()
