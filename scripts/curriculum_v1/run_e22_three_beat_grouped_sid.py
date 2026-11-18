#!/usr/bin/env python3
"""E22: non-overlapping same-lead three-heartbeat VQ atoms and grouped-SID MLM.

The experiment changes only the tokenizer atom from one 96-point heartbeat to
three consecutive heartbeats from the same measured lead.  It trains a new
1024x64 EMA codebook and a new E12-A continuous-input Transformer from the
same random initialization.  Text, lead embeddings, and time embeddings are
disabled.  Historical E3/E11/E12 artifacts are read-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

from heartlang.utils.norm_ema_quantizer import NormEMAVectorQuantizer
from heartlang.utils.utils import set_seed
import run_e12_continuous_input as e12
import run_e18_ecg_text_contrastive as e18


DEFAULT_RUN = ROOT / "results/e22_three_beat_grouped_sid"
PRETRAIN_SOURCE = e12.PRETRAIN_SOURCE
PRETRAIN_INDEX = e12.PRETRAIN_INDEX
E12_RUN = e12.DEFAULT_RUN
DOWNSTREAM = e12.DOWNSTREAM
MAX_GROUPS = 96
SEQ_LEN = 256
GROUP_SIZE = 3
TOKEN_LEN = 96
CODEBOOK_SIZE = 1024
CODE_DIM = 64


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state: dict[str, torch.Tensor], ignore_cluster_size: bool = False) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        if ignore_cluster_size and key.endswith("cluster_size"):
            continue
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_status(run: Path, value: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "status.txt").write_text(value, encoding="utf-8")
    print(value, flush=True)


def source_arrays(kind: str) -> dict[str, np.ndarray]:
    return {
        split: np.load(PRETRAIN_SOURCE / f"{split}_{kind}.npy", mmap_mode="r")
        for split in ("train", "val")
    }


def stable_leads(chans: np.ndarray, valid: np.ndarray) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in chans[valid].tolist():
        lead = int(value)
        if lead > 0 and lead not in seen:
            seen.add(lead)
            result.append(lead)
    return result


def group_indices_for_record(
    wave: np.ndarray,
    chans: np.ndarray,
    times: np.ndarray,
    max_groups: int = MAX_GROUPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [G,3] source indices, group lead and representative real time."""
    valid = (chans > 0) & (times > 0) & (np.abs(wave).max(axis=-1) > 1e-8)
    out = np.full((max_groups, GROUP_SIZE), -1, dtype=np.int16)
    group_chans = np.zeros(max_groups, dtype=np.int16)
    group_times = np.zeros(max_groups, dtype=np.int16)
    cursor = 0
    for lead in stable_leads(chans, valid):
        positions = np.flatnonzero(valid & (chans == lead))
        # The QRS tokenizer is lead-major and already chronological; explicitly
        # stable-sort by real R-peak-derived time to make this contract robust.
        positions = positions[np.argsort(times[positions], kind="stable")]
        for start in range(0, len(positions), GROUP_SIZE):
            if cursor >= max_groups:
                raise RuntimeError(f"More than {max_groups} same-lead groups in one ECG")
            chunk = positions[start : start + GROUP_SIZE]
            out[cursor, : len(chunk)] = chunk.astype(np.int16)
            group_chans[cursor] = lead
            group_times[cursor] = int(times[chunk[min(1, len(chunk) - 1)]])
            cursor += 1
    return out, group_chans, group_times


def materialize_groups(wave: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.zeros((len(indices), GROUP_SIZE, TOKEN_LEN), dtype=np.float32)
    slot_valid = indices >= 0
    rows, slots = np.nonzero(slot_valid)
    if len(rows):
        groups[rows, slots] = np.asarray(wave[indices[rows, slots]], dtype=np.float32)
    return groups, slot_valid


def build_group_index(run: Path) -> None:
    out = run / "group_index"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    waves = source_arrays("data")
    chans = source_arrays("data_in_chans")
    times = source_arrays("data_in_times")
    summary: dict[str, object] = {
        "grouping": "non-overlapping consecutive triples within each real measured lead",
        "group_size": GROUP_SIZE,
        "max_groups_storage": MAX_GROUPS,
        "padding": "zero waveform slots with explicit inner slot mask; no repeated beats",
        "cross_lead_groups_allowed": False,
        "splits": {},
    }
    for split in ("train", "val"):
        codes = np.load(PRETRAIN_INDEX / f"{split}_source_code.npy", mmap_mode="r")
        rows = np.load(PRETRAIN_INDEX / f"{split}_source_index.npy", mmap_mode="r")
        idx_out = np.lib.format.open_memmap(out / f"{split}_indices.npy", mode="w+", dtype=np.int16,
                                            shape=(len(rows), MAX_GROUPS, GROUP_SIZE))
        chan_out = np.lib.format.open_memmap(out / f"{split}_chans.npy", mode="w+", dtype=np.int16,
                                             shape=(len(rows), MAX_GROUPS))
        time_out = np.lib.format.open_memmap(out / f"{split}_times.npy", mode="w+", dtype=np.int16,
                                             shape=(len(rows), MAX_GROUPS))
        idx_out[:] = -1; chan_out[:] = 0; time_out[:] = 0
        beats = groups = incomplete = pad_slots = max_seen = 0
        remainders = [0, 0, 0]
        per_record: list[int] = []
        for j, (code, row) in enumerate(zip(codes, rows)):
            source = "train" if int(code) == 0 else "val"
            w = np.asarray(waves[source][int(row)])
            c = np.asarray(chans[source][int(row)])
            t = np.asarray(times[source][int(row)])
            gi, gc, gt = group_indices_for_record(w, c, t)
            idx_out[j], chan_out[j], time_out[j] = gi, gc, gt
            n = int((gc > 0).sum()); real_slots = int((gi >= 0).sum())
            per_record.append(n); beats += real_slots; groups += n
            max_seen = max(max_seen, n); incomplete += int(((gi[:n] >= 0).sum(1) < 3).sum())
            pad_slots += 3 * n - real_slots
            for lead in stable_leads(c, (c > 0) & (t > 0) & (np.abs(w).max(-1) > 1e-8)):
                remainders[int(((c == lead) & (t > 0) & (np.abs(w).max(-1) > 1e-8)).sum()) % 3] += 1
            if j == 0 or (j + 1) % 5000 == 0 or j + 1 == len(rows):
                print(f"audit {split}: {j+1}/{len(rows)}", flush=True)
        idx_out.flush(); chan_out.flush(); time_out.flush()
        summary["splits"][split] = {
            "records": int(len(rows)), "real_beats": beats, "groups": groups,
            "mean_groups_per_record": float(np.mean(per_record)), "p95_groups": float(np.percentile(per_record, 95)),
            "max_groups": max_seen, "incomplete_groups": incomplete,
            "incomplete_group_fraction": incomplete / max(1, groups),
            "padded_slot_fraction": pad_slots / max(1, groups * GROUP_SIZE),
            "lead_remainders_0_1_2": remainders,
        }
        if max_seen > MAX_GROUPS:
            raise RuntimeError(f"{split} max groups {max_seen} exceeds storage {MAX_GROUPS}")
    # Audit downstream too, because feature extraction must obey the same bound.
    downstream_stats = {}
    for split in ("train", "val", "test"):
        dw = np.load(DOWNSTREAM / f"{split}_data.npy", mmap_mode="r")
        dc = np.load(DOWNSTREAM / f"{split}_data_in_chans.npy", mmap_mode="r")
        dt = np.load(DOWNSTREAM / f"{split}_data_in_times.npy", mmap_mode="r")
        counts = []
        for j in range(len(dw)):
            _gi, gc, _gt = group_indices_for_record(np.asarray(dw[j]), np.asarray(dc[j]), np.asarray(dt[j]))
            counts.append(int((gc > 0).sum()))
        downstream_stats[split] = {"records": len(counts), "max_groups": max(counts),
                                   "mean_groups": float(np.mean(counts))}
    summary["downstream"] = downstream_stats
    summary["checks"] = {"all_max_groups_le_96": all(v["max_groups"] <= MAX_GROUPS for v in summary["splits"].values())
                                              and all(v["max_groups"] <= MAX_GROUPS for v in downstream_stats.values()),
                         "no_cross_lead_grouping": True, "real_time_order_preserved": True}
    dump(out / "audit.json", summary)
    (out / "COMPLETE").touch()


class GroupEncoder(nn.Module):
    def __init__(self, code_dim: int = CODE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Flatten(), nn.Linear(256 * math.ceil(TOKEN_LEN / 8), 256), nn.GELU(), nn.Linear(256, code_dim),
        )

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(groups), p=2, dim=-1)


class GroupDecoder(nn.Module):
    def __init__(self, code_dim: int = CODE_DIM):
        super().__init__()
        seed = math.ceil(TOKEN_LEN / 8)
        self.seed = seed
        self.fc = nn.Sequential(nn.Linear(code_dim, 256), nn.GELU(), nn.Linear(256, 256 * seed), nn.GELU())
        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 3, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(z).view(len(z), 256, self.seed))[..., :TOKEN_LEN]


class ThreeBeatVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GroupEncoder()
        self.decoder = GroupDecoder()
        self.quantize = NormEMAVectorQuantizer(n_embed=CODEBOOK_SIZE, embedding_dim=CODE_DIM,
                                               beta=1.0, decay=0.99, kmeans_init=False)

    def encode_valid(self, groups: torch.Tensor, slot_valid: torch.Tensor):
        valid = slot_valid.any(-1)
        if not valid.any():
            empty = groups.new_empty((0, CODE_DIM))
            return valid, groups.new_empty((0, 3, TOKEN_LEN)), slot_valid.new_empty((0, 3)), empty, empty, torch.empty(0, dtype=torch.long, device=groups.device), groups.sum() * 0
        x = groups[valid]
        slots = slot_valid[valid]
        z = self.encoder(x)
        zq, qloss, ids = self.quantize(z)
        return valid, x, slots, z, zq, ids, qloss

    def forward(self, groups: torch.Tensor, slot_valid: torch.Tensor):
        valid, x, slots, _z, zq, ids, qloss = self.encode_valid(groups, slot_valid)
        rec = self.decoder(zq)
        weight = slots.unsqueeze(-1).to(rec.dtype)
        rec_loss = ((rec - x).square() * weight).sum() / (weight.sum().clamp_min(1) * TOKEN_LEN)
        return qloss + rec_loss, {"quant_loss": qloss.detach(), "rec_loss": rec_loss.detach(),
                                  "valid_groups": valid.sum().detach(), "real_beats": slots.sum().detach()}, ids

    @torch.no_grad()
    def encode_full(self, groups: torch.Tensor, slot_valid: torch.Tensor):
        valid, _x, _slots, z, _zq, ids, _ql = self.encode_valid(groups, slot_valid)
        zfull = groups.new_zeros((*groups.shape[:2], CODE_DIM))
        sid = torch.full(groups.shape[:2], CODEBOOK_SIZE, dtype=torch.long, device=groups.device)
        zfull[valid] = z
        sid[valid] = ids
        return zfull, sid, valid


class GroupDataset(Dataset):
    SOURCE_SPLITS = ("train", "val")

    def __init__(self, run: Path, split: str):
        self.data = source_arrays("data")
        self.codes = np.load(PRETRAIN_INDEX / f"{split}_source_code.npy", mmap_mode="r")
        self.rows = np.load(PRETRAIN_INDEX / f"{split}_source_index.npy", mmap_mode="r")
        self.indices = np.load(run / "group_index" / f"{split}_indices.npy", mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        source = self.SOURCE_SPLITS[int(self.codes[i])]
        wave = np.asarray(self.data[source][int(self.rows[i])])
        return materialize_groups(wave, np.asarray(self.indices[i]))


def codebook_checkpoint(model: nn.Module, optimizer, epoch: int, best: float) -> dict:
    return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch, "best_val_loss": best, "model_type": "three_beat_group_vq",
            "config": {"group_size": 3, "token_len": 96, "codebook_size": 1024, "code_dim": 64}}


@torch.no_grad()
def eval_codebook(model: ThreeBeatVQ, loader, device: torch.device) -> dict:
    model.eval(); model.quantize.reset_cluster_size(device)
    sums = {"loss": 0.0, "rec_loss": 0.0, "quant_loss": 0.0}; batches = groups = beats = 0
    for group, slots in loader:
        group = group.float().to(device, non_blocking=True); slots = slots.bool().to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss, log, _ids = model(group, slots)
        sums["loss"] += float(loss); sums["rec_loss"] += float(log["rec_loss"]); sums["quant_loss"] += float(log["quant_loss"])
        groups += int(log["valid_groups"]); beats += int(log["real_beats"]); batches += 1
    usage = model.quantize.cluster_size.detach().cpu().numpy(); p = usage[usage > 0]
    if p.size: p = p / p.sum()
    return {k: v / max(1, batches) for k, v in sums.items()} | {
        "batches": batches, "valid_groups": groups, "real_beats": beats,
        "active_codes": int((usage > 0).sum()), "unused_codes": int((usage <= 0).sum()),
        "perplexity": float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0,
    }


def train_codebook(run: Path, device: torch.device, epochs: int = 50) -> None:
    out = run / "codebook"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(0); model = ThreeBeatVQ().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_ds, val_ds = GroupDataset(run, "train"), GroupDataset(run, "val")
    sampler = DistributedSampler(train_ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    train_loader = DataLoader(train_ds, sampler=sampler, batch_size=256, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, sampler=SequentialSampler(val_ds), batch_size=384, num_workers=8, pin_memory=True)
    start_epoch, best, best_epoch = 1, float("inf"), None
    last = out / "checkpoint-last.pth"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False); model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"]); start_epoch = int(ckpt["epoch"]) + 1
        best = float(ckpt.get("best_val_loss", best)); best_epoch = ckpt.get("best_epoch")
    steps = len(train_loader); history = out / "training_history.jsonl"; started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train(); sampler.set_epoch(epoch - 1); model.quantize.reset_cluster_size(device)
        sums = {"loss": 0.0, "rec_loss": 0.0, "quant_loss": 0.0}; n = 0
        for step, (group, slots) in enumerate(train_loader):
            global_step = (epoch - 1) * steps + step
            lr = e12.lr_at(global_step, epochs * steps, 5 * steps, 1e-4, 1e-5, 1e-6)
            for pg in optimizer.param_groups: pg["lr"] = lr
            group = group.float().to(device, non_blocking=True); slots = slots.bool().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, log, _ids = model(group, slots)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
            sums["loss"] += float(loss); sums["rec_loss"] += float(log["rec_loss"]); sums["quant_loss"] += float(log["quant_loss"]); n += 1
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
                print(f"codebook epoch={epoch} step={step+1}/{steps} loss={float(loss):.6f} rec={float(log['rec_loss']):.6f}", flush=True)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **{f"train_{k}": v / max(1, n) for k, v in sums.items()}}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            stats = eval_codebook(model, val_loader, device); row.update({f"val_{k}": v for k, v in stats.items()})
            if stats["loss"] < best:
                best, best_epoch = stats["loss"], epoch
                torch.save(codebook_checkpoint(model, optimizer, epoch, best) | {"best_epoch": best_epoch}, out / "checkpoint-best.pth")
            print(json.dumps(row), flush=True)
        torch.save(codebook_checkpoint(model, optimizer, epoch, best) | {"best_epoch": best_epoch}, last)
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ckpt = torch.load(out / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"]); final = eval_codebook(model, val_loader, device)
    np.save(out / "codebook_vectors.npy", model.quantize.embedding.weight.detach().cpu().numpy())
    dump(out / "summary.json", {"best_epoch": best_epoch, "best_val_loss": best, "best_validation": final,
                                 "epochs": epochs, "elapsed_seconds_this_process": time.time() - started,
                                 "train_records": len(train_ds), "val_records": len(val_ds)})
    (out / "COMPLETE").touch()


def load_group_vq(path: Path, device: torch.device) -> ThreeBeatVQ:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ThreeBeatVQ(); model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def cache_group_tokens(run: Path, device: torch.device) -> None:
    out = run / "pretrain_group_cache"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=True)
    model = load_group_vq(run / "codebook/checkpoint-best.pth", device)
    before = state_sha(model.state_dict(), ignore_cluster_size=True)
    usage = model.quantize.cluster_size.detach().clone()
    audit = {"dtype": "float32", "sequence_storage": SEQ_LEN, "valid_group_storage": MAX_GROUPS,
             "codebook_checkpoint": str(run / "codebook/checkpoint-best.pth"), "splits": {}}
    for split in ("train", "val"):
        ds = GroupDataset(run, split); loader = DataLoader(ds, batch_size=192, shuffle=False, num_workers=8, pin_memory=True)
        ze_out = np.lib.format.open_memmap(out / f"{split}_ze.npy", mode="w+", dtype=np.float32,
                                           shape=(len(ds), SEQ_LEN, CODE_DIM))
        sid_out = np.lib.format.open_memmap(out / f"{split}_sid.npy", mode="w+", dtype=np.int16,
                                            shape=(len(ds), SEQ_LEN))
        ze_out[:] = 0; sid_out[:] = CODEBOOK_SIZE
        offset = valid_total = 0
        for step, (groups, slots) in enumerate(loader):
            groups = groups.float().to(device, non_blocking=True); slots = slots.bool().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                ze, sid, valid = model.encode_full(groups, slots)
            model.quantize.cluster_size.copy_(usage)
            bs = len(groups); ze_out[offset:offset+bs, :MAX_GROUPS] = ze.float().cpu().numpy()
            sid_out[offset:offset+bs, :MAX_GROUPS] = sid.cpu().numpy().astype(np.int16)
            valid_total += int(valid.sum()); offset += bs
            if step == 0 or (step + 1) % 20 == 0 or offset == len(ds):
                print(f"cache {split}: {offset}/{len(ds)}", flush=True)
        ze_out.flush(); sid_out.flush(); del ze_out, sid_out
        audit["splits"][split] = {"records": len(ds), "valid_groups": valid_total,
                                     "ze_sha256": sha256(out / f"{split}_ze.npy"),
                                     "sid_sha256": sha256(out / f"{split}_sid.npy")}
    if state_sha(model.state_dict(), ignore_cluster_size=True) != before:
        raise RuntimeError("Frozen three-beat tokenizer parameters changed during caching")
    audit["frozen_codebook_parameters_unchanged"] = True
    dump(out / "audit.json", audit); (out / "COMPLETE").touch()


class GroupTokenDataset(Dataset):
    def __init__(self, run: Path, split: str):
        cache = run / "pretrain_group_cache"
        self.ze = np.load(cache / f"{split}_ze.npy", mmap_mode="r")
        self.sid = np.load(cache / f"{split}_sid.npy", mmap_mode="r")
    def __len__(self): return len(self.sid)
    def __getitem__(self, i): return np.array(self.ze[i], copy=True), np.array(self.sid[i], copy=True)


def mlm_checkpoint(model, optimizer, scaler, epoch: int, best: float, best_epoch,
                   loader_generator: torch.Generator) -> dict:
    return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
            "best_val_loss": best, "best_epoch": best_epoch, "mode": "three_beat_scratch",
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_generator_state": loader_generator.get_state(),
            "config": e12.model_config()}


def pretrain_mlm(run: Path, device: torch.device, epochs: int = 100) -> None:
    out = run / "mlm"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=True)
    model, init_audit = e12.initialize_continuous("scratch", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_ds, val_ds = GroupTokenDataset(run, "train"), GroupTokenDataset(run, "val")
    val_loader = DataLoader(val_ds, batch_size=384, shuffle=False, num_workers=8, pin_memory=True)
    start_epoch, best, best_epoch = 1, float("inf"), None
    loader_generator = torch.Generator().manual_seed(0)
    last = out / "checkpoint-last.pth"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False); model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"]); scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1; best = float(ckpt["best_val_loss"]); best_epoch = ckpt.get("best_epoch")
        if ckpt.get("torch_rng_state") is not None: torch.set_rng_state(ckpt["torch_rng_state"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None: torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        if ckpt.get("loader_generator_state") is not None: loader_generator.set_state(ckpt["loader_generator_state"])
    dump(out / "initialization_audit.json", init_audit)
    history = out / "training_history.jsonl"; started = time.time()
    args = argparse.Namespace(epochs=epochs)
    for epoch in range(start_epoch, epochs + 1):
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=loader_generator,
                                  num_workers=8, pin_memory=True, drop_last=True)
        train_stats = e12.run_mlm_epoch(model, train_loader, device, optimizer, scaler, epoch, args, True)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            # Validation must not retain an autograd graph across batches. E12-A
            # wraps this call in no_grad; match that behavior to avoid GPU OOM.
            with torch.no_grad():
                val_stats = e12.run_mlm_epoch(model, val_loader, device, optimizer, scaler, epoch, args, False)
            row.update({f"val_{k}": v for k, v in val_stats.items()})
            if val_stats["loss"] < best:
                best, best_epoch = val_stats["loss"], epoch
                torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "epoch": epoch, "best_val_loss": best, "mode": "three_beat_scratch",
                            "config": e12.model_config()}, out / "checkpoint-best.pth")
        torch.save(mlm_checkpoint(model, optimizer, scaler, epoch, best, best_epoch, loader_generator), last)
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row), flush=True)
    dump(out / "summary.json", {"epochs": epochs, "best_epoch": best_epoch, "best_val_loss": best,
                                 "pretrain_seed": 0, "train_records": len(train_ds), "val_records": len(val_ds),
                                 "elapsed_seconds_this_process": time.time() - started,
                                 "grouped_token_mean_train": json.loads((run / "group_index/audit.json").read_text())["splits"]["train"]["mean_groups_per_record"]})
    (out / "COMPLETE").touch()


@torch.inference_mode()
def fixed_mlm(run: Path, device: torch.device) -> None:
    path = run / "fixed_mask_mlm_comparison.json"
    if path.exists(): return
    ds = GroupTokenDataset(run, "val")
    model = e12.load_continuous_checkpoint(run / "mlm/checkpoint-best.pth", device)
    total_loss = correct = tokens = 0
    for start in range(0, len(ds), 128):
        end = min(start + 128, len(ds)); ze_np = np.asarray(ds.ze[start:end]); sid_np = np.asarray(ds.sid[start:end])
        mask_np = e12.fixed_masks(sid_np, start); ze = torch.from_numpy(ze_np.copy()).float().to(device)
        sid = torch.from_numpy(sid_np.copy()).long().to(device); mask = torch.from_numpy(mask_np).to(device)
        valid = (sid >= 0) & (sid < CODEBOOK_SIZE); labels = sid[mask]
        logits = model(ze, valid, mask)
        total_loss += float(F.cross_entropy(logits.float(), labels, reduction="sum"));
        correct += int((logits.argmax(-1) == labels).sum()); tokens += int(labels.numel())
    baseline = json.loads((E12_RUN / "fixed_mask_mlm_comparison.json").read_text())["e12a_scratch"]
    dump(path, {"mask_policy": "stateless 50% of valid tokens in each model",
                "comparability_note": "E12-A and E22 predict different codebooks, so raw CE/accuracy are diagnostic rather than directly comparable task scores",
                "e12a_single_beat": baseline,
                "e22_three_beat": {"loss": total_loss / tokens, "masked_sid_accuracy": correct / tokens, "masked_tokens": tokens},
                "checkpoint": str(run / "mlm/checkpoint-best.pth")})


def group_batch(waves: np.ndarray, chans: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.zeros((len(waves), MAX_GROUPS, GROUP_SIZE, TOKEN_LEN), dtype=np.float32)
    slots = np.zeros((len(waves), MAX_GROUPS, GROUP_SIZE), dtype=bool)
    for j in range(len(waves)):
        idx, _gc, _gt = group_indices_for_record(waves[j], chans[j], times[j])
        groups[j], slots[j] = materialize_groups(waves[j], idx)
    return groups, slots


@torch.inference_mode()
def extract_features(run: Path, device: torch.device) -> None:
    out = run / "features"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_group_vq(run / "codebook/checkpoint-best.pth", device)
    model = e12.load_continuous_checkpoint(run / "mlm/checkpoint-best.pth", device).requires_grad_(False)
    audit = {"pooling": "valid grouped-token mean+max", "group_size": 3, "text_used": False, "splits": {}}
    for split in ("train", "val", "test"):
        waves = np.load(DOWNSTREAM / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(DOWNSTREAM / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(DOWNSTREAM / f"{split}_data_in_times.npy", mmap_mode="r")
        result = np.empty((len(waves), 1536), dtype=np.float32); valid_total = 0
        for start in range(0, len(waves), 32):
            end = min(start + 32, len(waves))
            g_np, s_np = group_batch(np.asarray(waves[start:end]), np.asarray(chans[start:end]), np.asarray(times[start:end]))
            g = torch.from_numpy(g_np).to(device); slots = torch.from_numpy(s_np).to(device)
            ze96, _sid96, valid96 = tokenizer.encode_full(g, slots)
            ze = torch.zeros(len(g), SEQ_LEN, CODE_DIM, device=device); valid = torch.zeros(len(g), SEQ_LEN, dtype=torch.bool, device=device)
            ze[:, :MAX_GROUPS] = ze96.float(); valid[:, :MAX_GROUPS] = valid96
            tokens_out = model.extract_tokens(ze, valid).float(); count = valid.sum(1).clamp_min(1).unsqueeze(-1)
            mean = (tokens_out * valid.unsqueeze(-1)).sum(1) / count
            maximum = tokens_out.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1); maximum[~valid.any(1)] = 0
            result[start:end] = torch.cat([mean, maximum], 1).cpu().numpy(); valid_total += int(valid.sum())
        np.save(out / f"{split}_heartlang_mean_max.npy", result)
        audit["splits"][split] = {"records": len(result), "valid_groups": valid_total, "shape": list(result.shape),
                                   "sha256": sha256(out / f"{split}_heartlang_mean_max.npy"), "finite": bool(np.isfinite(result).all())}
    dump(out / "audit.json", audit); (out / "COMPLETE").touch()


def aggregate(items: list[dict]) -> dict:
    keys = ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy", "brier_score", "ece15_top_label")
    result = {key: {"mean": float(np.mean([row["val"][key] for row in items])),
                    "sd": float(np.std([row["val"][key] for row in items], ddof=1))} for key in keys}
    result["per_output_r2"] = {name: {"mean": float(np.mean([row["val"]["per_output_r2"][name] for row in items])),
                                             "sd": float(np.std([row["val"]["per_output_r2"][name] for row in items], ddof=1))}
                               for name in e12.LABEL_NAMES}
    result["patient_mean_r2"] = {"mean": float(np.mean([np.mean([row["val"]["per_output_r2"][n] for n in e12.LABEL_NAMES[:3]]) for row in items])),
                                 "sd": float(np.std([np.mean([row["val"]["per_output_r2"][n] for n in e12.LABEL_NAMES[:3]]) for row in items], ddof=1))}
    return result


def downstream(run: Path, device: torch.device) -> None:
    path = run / "validation_summary.json"
    if path.exists(): return
    rows = []
    for seed in (0, 1, 2):
        seed_dir = run / "downstream" / f"seed{seed}"
        if seed_dir.exists() and not (seed_dir / "VAL_COMPLETE").exists() and not any(seed_dir.iterdir()): seed_dir.rmdir()
        item = e18.train_downstream_seed(run, seed, device); item["method"] = "e22_three_beat_grouped_sid"
        dump(seed_dir / "val_summary.json", item); rows.append(item)
    base_rows = json.loads((E12_RUN / "downstream/validation_selection.json").read_text())["methods"]["e12a_region"]["per_seed"]
    dump(path, {"validation_only": True, "test_run": False,
                "e12a_historical": {"aggregate": aggregate(base_rows), "per_seed": base_rows},
                "e22_three_beat": {"aggregate": aggregate(rows), "per_seed": rows}})


def smoke(run: Path, device: torch.device) -> None:
    path = run / "smoke.json"
    if path.exists(): return
    audit = json.loads((run / "group_index/audit.json").read_text())
    if not audit["checks"]["all_max_groups_le_96"]: raise RuntimeError("Group length audit failed")
    model = ThreeBeatVQ().to(device); model.train()
    group = torch.randn(4, 7, 3, 96, device=device); slots = torch.ones(4, 7, 3, dtype=torch.bool, device=device)
    slots[0, -1, 1:] = False; group[0, -1, 1:] = 0
    loss, log, ids = model(group, slots); loss.backward()
    if not torch.isfinite(loss) or model.encoder.net[0].weight.grad is None: raise RuntimeError("Three-beat VQ backward smoke failed")
    mlm, init = e12.initialize_continuous("scratch", device); mlm.eval()
    ze = torch.randn(2, 256, 64, device=device); valid = torch.zeros(2, 256, dtype=torch.bool, device=device); valid[:, :41] = True
    mask = torch.zeros_like(valid); mask[:, [2, 7, 13]] = True
    with torch.no_grad():
        a = mlm(ze, valid, mask); changed = ze.clone(); changed[mask] = torch.randn_like(changed[mask]) * 1e5; b = mlm(changed, valid, mask)
    delta = float((a - b).abs().max())
    if delta != 0: raise RuntimeError(f"Masked grouped z leakage: {delta}")
    dump(path, {"passed": True, "group_shape": [3, 96], "same_lead_nonoverlap": True,
                "zero_padding_with_inner_mask": True, "vq_loss_finite": True,
                "vq_encoder_gradient_finite": bool(torch.isfinite(model.encoder.net[0].weight.grad).all()),
                "ids_shape": list(ids.shape), "masked_group_ze_mutation_delta": delta,
                "explicit_padding_key_mask": True, "text_used": False, "e12_initialization": init})


def report(run: Path) -> None:
    summary = json.loads((run / "validation_summary.json").read_text()); fixed = json.loads((run / "fixed_mask_mlm_comparison.json").read_text())
    grouping = json.loads((run / "group_index/audit.json").read_text()); codebook = json.loads((run / "codebook/summary.json").read_text())
    base, new = summary["e12a_historical"]["aggregate"], summary["e22_three_beat"]["aggregate"]
    rows = []
    for label, key in (("E12-A", "e12a_historical"), ("E22-3Beat", "e22_three_beat")):
        for item in summary[key]["per_seed"]:
            val = item["val"]
            rows.append({"method": label, "statistic": "seed", "seed": item.get("seed"), "overall_r2": val["r2"],
                         "patient_mean_r2": np.mean([val["per_output_r2"][n] for n in e12.LABEL_NAMES[:3]]),
                         **{f"{n.lower().replace(' ', '_')}_r2": val["per_output_r2"][n] for n in e12.LABEL_NAMES},
                         "macro_auc": val["macro_roc_auc"], "macro_f1": val["macro_f1"]})
        agg = summary[key]["aggregate"]
        for stat in ("mean", "sd"):
            rows.append({"method": label, "statistic": stat, "seed": np.nan, "overall_r2": agg["r2"][stat],
                         "patient_mean_r2": agg["patient_mean_r2"][stat],
                         **{f"{n.lower().replace(' ', '_')}_r2": agg["per_output_r2"][n][stat] for n in e12.LABEL_NAMES},
                         "macro_auc": agg["macro_roc_auc"][stat], "macro_f1": agg["macro_f1"][stat]})
    pd.DataFrame(rows).to_csv(run / "metrics_summary.csv", index=False)
    def metric_row(label: str, x: dict) -> str:
        return (f"| {label} | {x['r2']['mean']:.6f} ± {x['r2']['sd']:.6f} | {x['patient_mean_r2']['mean']:.6f} | "
                + " | ".join(f"{x['per_output_r2'][n]['mean']:.6f}" for n in e12.LABEL_NAMES)
                + f" | {x['macro_roc_auc']['mean']:.6f} | {x['macro_f1']['mean']:.6f} |")
    cb = codebook["best_validation"]; tr = grouping["splits"]["train"]
    text = ["# E22：三心拍组 Codebook 与组级 SID MLM 消融", "", "## 方法", "",
            "同一实测导联内按真实时间顺序把连续三个96点心拍组成一个token；不足三个时补零并用内部mask排除补齐位置的重建损失。",
            "重新训练1024×64 EMA Codebook，一个三心拍组对应一个连续z_g和一个SID；MLM从与E12-A相同的随机初始化训练，并Mask 50%有效组。", "",
            "- 不跨导联分组；不复制末尾心拍；不使用文字、Lead或Time Embedding。",
            "- Transformer结构、256位置长度、优化器、100 epoch和下游Mean+Max+E11区域融合与E12-A一致。",
            "- 下游seed 0/1/2，仅Validation，未运行Test。", "", "## 分组审计", "",
            f"- Train：{tr['records']}条ECG，{tr['real_beats']}个真实心拍，{tr['groups']}个三心拍组，平均{tr['mean_groups_per_record']:.3f}组/ECG。",
            f"- 不完整组占{tr['incomplete_group_fraction']:.4%}；补齐槽位占{tr['padded_slot_fraction']:.4%}；最大{tr['max_groups']}组。", "",
            "## Codebook", "", f"- 最佳epoch：{codebook['best_epoch']}；Val总损失：{codebook['best_val_loss']:.6f}。",
            f"- Val重建MSE：{cb['rec_loss']:.6f}；活跃SID：{cb['active_codes']}/1024；perplexity：{cb['perplexity']:.3f}。", "",
            "## 固定Mask MLM（目标Codebook不同，数值仅作各自任务诊断）", "", "| 方法 | Val CE | Masked SID Accuracy |", "|---|---:|---:|",
            f"| E12-A单心拍 | {fixed['e12a_single_beat']['loss']:.6f} | {fixed['e12a_single_beat']['masked_sid_accuracy']:.6f} |",
            f"| E22三心拍 | {fixed['e22_three_beat']['loss']:.6f} | {fixed['e22_three_beat']['masked_sid_accuracy']:.6f} |", "",
            "## 下游Validation（三个seed）", "",
            "| 方法 | Overall R² | 患者三类均值 | OMI | NOMI | Clinical Control | Healthy | Macro AUC | Macro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|", metric_row("E12-A", base), metric_row("E22-3Beat", new), "",
            f"E22相对E12-A：Overall R² **{new['r2']['mean']-base['r2']['mean']:+.6f}**；患者三类平均R² **{new['patient_mean_r2']['mean']-base['patient_mean_r2']['mean']:+.6f}**；",
            f"OMI **{new['per_output_r2']['OMI']['mean']-base['per_output_r2']['OMI']['mean']:+.6f}**；NOMI **{new['per_output_r2']['NOMI']['mean']-base['per_output_r2']['NOMI']['mean']:+.6f}**。", ""]
    (run / "README.md").write_text("\n".join(text), encoding="utf-8")
    set_status(run, "complete"); (run / "COMPLETE").touch()


def protocol(run: Path, codebook_epochs: int, mlm_epochs: int) -> None:
    path = run / "protocol.json"
    if path.exists(): return
    dump(path, {"experiment": "E22 same-lead non-overlapping three-heartbeat grouped SID ablation",
                "baseline": "E12-A historical read-only", "group_size": 3, "heartbeat_points": 96,
                "remainder_policy": "zero pad to three and exclude padded slots from reconstruction loss",
                "cross_lead_grouping": False, "codebook": "new 1024x64 EMA VQ trained from scratch",
                "codebook_epochs": codebook_epochs, "mlm": "continuous grouped z_g input predicts grouped SID",
                "mlm_epochs": mlm_epochs, "mlm_seed": 0, "same_e12a_transformer_initialization": True,
                "mask_ratio_valid_groups": 0.5, "seq_len": 256,
                "disabled": ["text", "lead_embedding", "time_embedding", "clinical_labels_in_pretraining"],
                "downstream": "frozen grouped Transformer Mean+Max1536 plus unchanged E11 region residual; seeds0,1,2",
                "evaluation": "Validation only; Test prohibited"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "smoke", "codebook", "cache", "pretrain", "fixed_mlm", "features", "downstream", "report"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--codebook_epochs", type=int, default=50)
    parser.add_argument("--mlm_epochs", type=int, default=100)
    args = parser.parse_args(); run = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run.mkdir(parents=True, exist_ok=True); protocol(run, args.codebook_epochs, args.mlm_epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "audit": build_group_index(run)
    elif args.stage == "smoke": smoke(run, device)
    elif args.stage == "codebook": train_codebook(run, device, args.codebook_epochs)
    elif args.stage == "cache": cache_group_tokens(run, device)
    elif args.stage == "pretrain": pretrain_mlm(run, device, args.mlm_epochs)
    elif args.stage == "fixed_mlm": fixed_mlm(run, device)
    elif args.stage == "features": extract_features(run, device)
    elif args.stage == "downstream": downstream(run, device)
    elif args.stage == "report": report(run)


if __name__ == "__main__":
    main()
