#!/usr/bin/env python3
"""E2 A/B/C/D 统一 MLM 训练入口，支持 pilot 与正式独立运行。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

from ecg_representation_e2 import ECGRepresentationMLM, encode_independent_teacher, initialize_matched_core, random_valid_mask, warmstart_from_sid_mlm
from evaluate_vocabulary_pilot import load_model

SOURCE = ROOT / "datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
INDEX = ROOT / "datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict"
SID = ROOT / "datasets/ecg_datasets/ECG_DSA/pretrain_sid/e3_healthy_patient_replay_20261011_strict_mixed"
TOKENIZER = ROOT / "checkpoints/curriculum_v1/e3_healthy_patient_replay_20261011_adapt_e30/checkpoint-best.pth"
MLM = ROOT / "checkpoints/curriculum_v1/e3_healthy_patient_replay_20261011_sid_mlm_e100/checkpoint-best.pth"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class StrictDataset:
    """通过冻结 index 找回 waveform，与已有 strict SID 逐行对应。"""

    def __init__(self, split: str, limit: int):
        self.codes = np.load(INDEX / f"{split}_source_code.npy", mmap_mode="r")
        self.positions = np.load(INDEX / f"{split}_source_index.npy", mmap_mode="r")
        self.sid = np.load(SID / f"{split}_sid.npy", mmap_mode="r")
        self.chans = np.load(SID / f"{split}_data_in_chans.npy", mmap_mode="r")
        self.times = np.load(SID / f"{split}_data_in_times.npy", mmap_mode="r")
        self.wave = {name: np.load(SOURCE / f"{name}_data.npy", mmap_mode="r") for name in ("train", "val")}
        if len(self.codes) != len(self.sid) or len(self.positions) != len(self.sid):
            raise ValueError("严格索引与 SID 数量不同")
        self.n = len(self.sid) if limit <= 0 else min(limit, len(self.sid))

    def __len__(self):
        return self.n

    def batch(self, indices, device):
        waves = np.stack([self.wave[("train", "val")[int(self.codes[i])]][int(self.positions[i])] for i in indices])
        sid = np.asarray(self.sid[indices]).copy()
        chans = np.asarray(self.chans[indices]).copy()
        times = np.asarray(self.times[indices]).copy()
        return (torch.from_numpy(waves).float().to(device), torch.from_numpy(sid).long().to(device),
                torch.from_numpy(chans).long().to(device), torch.from_numpy(times).long().to(device))


def mask_for_batch(valid, seed):
    gen = torch.Generator(device=valid.device).manual_seed(seed)
    return random_valid_mask(valid, 0.5, gen)


def eval_one(model, tokenizer, dataset, batch_size, device):
    model.eval()
    totals = {"loss_sum": 0.0, "correct": 0, "tokens": 0, "records": 0}
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            ix = np.arange(start, min(start + batch_size, len(dataset)))
            wave, sid, chans, times = dataset.batch(ix, device)
            valid, ze, _zq, _online_sid = encode_independent_teacher(tokenizer, wave, chans, times)
            assert torch.equal(valid, (sid >= 0) & (sid < 1024))
            mask = mask_for_batch(valid, 71_000 + start)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(wave, sid, ze, valid, mask, chans, times)
                loss = torch.nn.functional.cross_entropy(logits.float(), sid[mask])
            totals["loss_sum"] += float(loss.item()) * int(mask.sum())
            totals["correct"] += int((logits.argmax(-1) == sid[mask]).sum())
            totals["tokens"] += int(mask.sum())
            totals["records"] += len(ix)
    return {"mlm_loss": totals["loss_sum"] / max(1, totals["tokens"]),
            "mlm_accuracy": totals["correct"] / max(1, totals["tokens"]),
            "masked_tokens": totals["tokens"], "records": totals["records"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(ECGRepresentationMLM.MODES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmstart", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_val", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    public_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    out = args.output.resolve() if args.output.is_absolute() else (ROOT / args.output).resolve()
    if out.exists():
        raise FileExistsError(f"拒绝覆盖已有产物: {out}")
    out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = StrictDataset("train", args.max_train), StrictDataset("val", args.max_val)
    tokenizer = load_model(TOKENIZER, device).eval().requires_grad_(False)
    model = ECGRepresentationMLM(args.mode)
    init = (warmstart_from_sid_mlm(model, MLM) if args.warmstart
            else initialize_matched_core(model, args.seed))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    steps = math.ceil(len(train) / args.batch)
    total_steps = steps * args.epochs
    warm_steps = min(total_steps, steps * 5)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        epoch_loss = 0.0
        seen = 0
        for step, start in enumerate(range(0, len(train), args.batch)):
            ix = order[start:start + args.batch]
            wave, sid, chans, times = train.batch(ix, device)
            with torch.no_grad():
                valid, ze, _zq, _online_sid = encode_independent_teacher(tokenizer, wave, chans, times)
            if not torch.equal(valid, (sid >= 0) & (sid < 1024)):
                raise ValueError("teacher 有效心拍与固定 SID 不对齐")
            mask = mask_for_batch(valid, args.seed * 10**8 + epoch * len(train) + start)
            if not mask.any():
                continue
            global_step = (epoch - 1) * steps + step
            if global_step < warm_steps:
                lr = args.lr * (global_step + 1) / max(1, warm_steps)
            else:
                fraction = (global_step - warm_steps) / max(1, total_steps - warm_steps)
                lr = 1e-5 + (args.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = model.mlm_loss(wave, sid, ze, valid, mask, chans, times)
            if not torch.isfinite(loss):
                raise FloatingPointError("非有限 MLM loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item()) * int(mask.sum())
            seen += int(mask.sum())
        metrics = eval_one(model, tokenizer, val, args.batch, device)
        record = {"epoch": epoch, "train_mlm_loss": epoch_loss / max(1, seen), "val": metrics,
                  "seconds_elapsed": time.monotonic() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["mlm_loss"] < best_loss:
            best_loss, best_epoch = metrics["mlm_loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "args": public_args, "best_epoch": best_epoch,
                        "val": metrics}, out / "checkpoint-best.pth")
    summary = {"mode": args.mode, "warmstart": args.warmstart, "best_epoch": best_epoch,
               "best_val_mlm_loss": best_loss, "history": history,
               "duration_seconds": time.monotonic() - started,
               "total_parameters": sum(p.numel() for p in model.parameters()),
               "pretrain_train_records": len(train), "pretrain_val_records": len(val),
               "teacher": str(TOKENIZER), "warmstart_checkpoint": str(MLM) if args.warmstart else None,
               "source_hashes": {str(p): sha(p) for p in (TOKENIZER, MLM, INDEX / "summary.json", SID / "strict_sid_summary.json")},
               "initialization_audit": init, "hyperparameters": public_args,
               "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
               "mask_safety": "all source modalities masked before input projection; independent teacher; key padding mask"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "COMPLETE").touch()


if __name__ == "__main__":
    main()
