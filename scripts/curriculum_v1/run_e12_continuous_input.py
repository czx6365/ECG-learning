#!/usr/bin/env python3
"""E12: continuous z_e input, frozen-codebook SID targets, and E11 downstream.

The frozen E3 morphology encoder supplies continuous per-heartbeat z_e.  The
frozen E3 codebook supplies only the precomputed SID targets.  E12-A trains the
Sentence Transformer from a matched random initialization; E12-B warm-starts
the Transformer and MLM head from the trained E3 SID MLM.  Downstream model
selection is validation-only and test is evaluated only for the selected E12
architecture.
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
from einops import rearrange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

from heartlang.models.modeling_heartlang_sid import HeartLangSIDForMLM
from heartlang.models.modeling_pretrain import Transformer
from heartlang.utils.utils import set_seed
import run_e4_aggregation as e4
import run_e7_objective as e7
import run_e9_ze_fusion_r2 as e9
import run_e10_lead_aware_ze_fusion as e10
import run_e11_region_residual_fusion as e11


EXP = "e12_continuous_ze_sentence_20261018"
DEFAULT_RUN = ROOT / "results/curriculum_v1" / EXP
PRETRAIN_SOURCE = ROOT / "datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
PRETRAIN_INDEX = ROOT / "datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict"
PRETRAIN_SID = ROOT / "datasets/ecg_datasets/ECG_DSA/pretrain_sid/e3_healthy_patient_replay_20261011_strict_mixed"
DOWNSTREAM = e9.DATA
TOKENIZER = e9.TOKENIZER
E3_MLM = e9.MLM
E11_RUN = ROOT / "results/curriculum_v1/e11_region_residual_fusion_20261018"
E10_RUN = ROOT / "results/curriculum_v1/e10_lead_aware_ze_fusion_20261018"
LABEL_NAMES = e9.LABEL_NAMES


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_status(run_dir: Path, value: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.txt").write_text(value, encoding="utf-8")
    print(value, flush=True)


def model_config() -> dict:
    return dict(seq_len=256, code_dim=64, vocab_size=1024, embed_dim=768,
                depth=12, heads=8, dim_head=64, mlp_dim=1024,
                dropout=0.1, emb_dropout=0.1)


def masked_transformer_forward(transformer: Transformer, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Run the original HeartLang blocks with an explicit key-padding mask.

    `valid` includes CLS. Invalid queries are zeroed after every block, while
    invalid keys are excluded before softmax. Module names/weights stay exactly
    compatible with the E3 Transformer checkpoint.
    """
    key_valid = valid[:, None, None, :]
    query_valid = valid.unsqueeze(-1).to(x.dtype)
    for attn, ff in transformer.layers:
        y = attn.norm(x)
        qkv = attn.to_qkv(y).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=attn.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
        dots = dots.masked_fill(~key_valid, torch.finfo(dots.dtype).min)
        weights = attn.dropout(attn.attend(dots))
        out = torch.matmul(weights, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        x = (attn.to_out(out) + x) * query_valid
        x = (ff(x) + x) * query_valid
    return x


class ContinuousBackbone(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        self.seq_len = int(cfg["seq_len"])
        self.code_dim = int(cfg["code_dim"])
        self.embed_dim = int(cfg["embed_dim"])
        self.depth = int(cfg["depth"])
        self.input_proj = nn.Linear(self.code_dim, self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len + 1, self.embed_dim))
        self.dropout = nn.Dropout(float(cfg["emb_dropout"]))
        self.transformer = Transformer(embed_dim=self.embed_dim, depth=self.depth,
                                       heads=int(cfg["heads"]), dim_head=int(cfg["dim_head"]),
                                       mlp_dim=int(cfg["mlp_dim"]), dropout=float(cfg["dropout"]))
        self.norm_layer = nn.LayerNorm(self.embed_dim)

    def forward_feature(self, ze: torch.Tensor, valid: torch.Tensor,
                        mask_bool_matrix: torch.Tensor | None = None) -> torch.Tensor:
        if ze.ndim != 3 or ze.shape[1:] != (self.seq_len, self.code_dim):
            raise ValueError(f"Expected z_e [B,{self.seq_len},{self.code_dim}], got {tuple(ze.shape)}")
        valid = valid.bool()
        x = self.input_proj(ze.float())
        x = x * valid.unsqueeze(-1).to(x.dtype)
        if mask_bool_matrix is None:
            mask = torch.zeros_like(valid)
        else:
            mask = mask_bool_matrix.bool() & valid
        # Complete replacement after the per-token Linear: no masked z_e path remains.
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
        cls = self.cls_token.view(1, 1, -1).expand(len(x), 1, -1)
        x = torch.cat([cls, x], dim=1)
        full_valid = torch.cat([torch.ones(len(x), 1, dtype=torch.bool, device=x.device), valid], dim=1)
        x = self.dropout(x + self.pos_embed)
        x = masked_transformer_forward(self.transformer, x, full_valid)
        x = self.norm_layer(x) * full_valid.unsqueeze(-1).to(x.dtype)
        return x

    def token_features(self, ze: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.forward_feature(ze, valid)[:, 1:]


class ContinuousSIDMLM(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        self.backbone = ContinuousBackbone(**cfg)
        self.lm_head = nn.Linear(int(cfg["embed_dim"]), int(cfg["vocab_size"]))

    def forward(self, ze: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_feature(ze, valid, mask)[:, 1:]
        return self.lm_head(tokens[mask.bool() & valid.bool()])

    def extract_tokens(self, ze: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.backbone.token_features(ze, valid)


def fresh_sid_model(seed: int = 0) -> HeartLangSIDForMLM:
    set_seed(seed)
    cfg = model_config()
    return HeartLangSIDForMLM(seq_len=cfg["seq_len"], vocab_size=cfg["vocab_size"], pad_sid=1024,
                              embed_dim=cfg["embed_dim"], depth=cfg["depth"], heads=cfg["heads"],
                              dim_head=cfg["dim_head"], mlp_dim=cfg["mlp_dim"],
                              dropout=cfg["dropout"], emb_dropout=cfg["emb_dropout"],
                              use_spatial_embedding=False, use_temporal_embedding=False)


def initialize_continuous(mode: str, device: torch.device) -> tuple[ContinuousSIDMLM, dict]:
    if mode not in ("scratch", "warm"):
        raise ValueError(mode)
    # A and B receive an identical independently seeded input projection.
    set_seed(120012)
    model = ContinuousSIDMLM(**model_config())
    if mode == "scratch":
        source = fresh_sid_model(0).state_dict()
        source_desc = "matched random E3 SID-model initialization (untrained shared weights)"
    else:
        ckpt = torch.load(E3_MLM, map_location="cpu", weights_only=False)
        source = ckpt["model"]
        source_desc = str(E3_MLM)
    target = model.state_dict()
    copied, skipped = [], []
    for key, value in source.items():
        if key == "backbone.sid_embed.weight":
            skipped.append(key)
        elif key in target and target[key].shape == value.shape:
            target[key] = value.detach().clone()
            copied.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(target, strict=True)
    audit = {"mode": mode, "source": source_desc, "copied_keys": len(copied),
             "skipped_source_keys": skipped,
             "input_projection_sha256": state_sha({k: v for k, v in model.state_dict().items() if k.startswith("backbone.input_proj.")}),
             "full_initial_state_sha256": state_sha(model.state_dict())}
    return model.to(device), audit


class ZEDataset(Dataset):
    def __init__(self, cache: Path, split: str):
        self.ze = np.load(cache / f"{split}_ze.npy", mmap_mode="r")
        self.sid = np.load(PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        if self.ze.shape[:2] != self.sid.shape or self.ze.shape[2] != 64:
            raise RuntimeError(f"{split}: z_e/SID shape mismatch")

    def __len__(self):
        return len(self.sid)

    def __getitem__(self, index):
        return np.array(self.ze[index], copy=True), np.array(self.sid[index], copy=True)


def random_mask(valid: torch.Tensor, ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    mask = torch.zeros_like(valid)
    for row in range(len(valid)):
        idx = torch.nonzero(valid[row], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        n = max(1, int(round(float(ratio) * idx.numel())))
        perm = torch.randperm(idx.numel(), generator=generator, device=idx.device)
        mask[row, idx[perm[:n]]] = True
    return mask


def gather_source(sources: dict[str, np.ndarray], codes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    out = np.empty((len(indices),) + tuple(sources["train"].shape[1:]), dtype=np.float32)
    for code, name in enumerate(("train", "val")):
        pos = np.flatnonzero(codes == code)
        if len(pos):
            out[pos] = np.asarray(sources[name][indices[pos]], dtype=np.float32)
    return out


@torch.inference_mode()
def cache_pretrain(run_dir: Path, device: torch.device, batch_size: int) -> None:
    cache = run_dir / "pretrain_ze_cache"
    if (cache / "COMPLETE").exists():
        print("Reuse complete pretrain z_e cache", flush=True); return
    cache.mkdir(parents=True, exist_ok=False)
    tokenizer = e4.load_model(TOKENIZER, device).eval().requires_grad_(False)
    frozen_before = state_sha(tokenizer.state_dict())
    # This quantizer updates its eval-time usage counter even though the
    # codebook centers remain frozen. Preserve and restore the counter around
    # the read-only SID audit so the entire tokenizer state is bit-identical.
    usage_counter = tokenizer.quantize.cluster_size.detach().clone()
    wave_sources = {s: np.load(PRETRAIN_SOURCE / f"{s}_data.npy", mmap_mode="r") for s in ("train", "val")}
    chan_sources = {s: np.load(PRETRAIN_SOURCE / f"{s}_data_in_chans.npy", mmap_mode="r") for s in ("train", "val")}
    time_sources = {s: np.load(PRETRAIN_SOURCE / f"{s}_data_in_times.npy", mmap_mode="r") for s in ("train", "val")}
    summary = {"tokenizer": str(TOKENIZER), "tokenizer_sha256": sha256(TOKENIZER),
               "dtype": "float32", "splits": {}}
    for split in ("train", "val"):
        codes = np.load(PRETRAIN_INDEX / f"{split}_source_code.npy")
        indices = np.load(PRETRAIN_INDEX / f"{split}_source_index.npy")
        targets = np.load(PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        out = np.lib.format.open_memmap(cache / f"{split}_ze.npy", mode="w+", dtype=np.float32,
                                        shape=(len(indices), 256, 64))
        sid_match, sid_total, valid_total, zero_invalid_max = 0, 0, 0, 0.0
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            c = codes[start:end]; ii = indices[start:end]
            wave = gather_source(wave_sources, c, ii)
            chan = gather_source(chan_sources, c, ii).astype(np.int64)
            times = gather_source(time_sources, c, ii).astype(np.int64)
            x = torch.from_numpy(wave).to(device)
            ch = torch.from_numpy(chan).long().to(device)
            ti = torch.from_numpy(times).long().to(device)
            valid = tokenizer.valid_mask(x, ch, ti)
            ze = tokenizer.encode_to_quantizer_features(x, ch, ti).float()
            if (~valid).any():
                zero_invalid_max = max(zero_invalid_max, float(ze[~valid].abs().max()))
            out[start:end] = ze.cpu().numpy()
            # Reproduce the original SID-generation autocast path for an exact audit.
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                valid2, _beats, _z, _zq, ids, _ql = tokenizer.encode_valid(x, ch, ti)
            tokenizer.quantize.cluster_size.copy_(usage_counter)
            pred = np.full((end - start, 256), 1024, dtype=np.int16)
            pred[valid2.cpu().numpy()] = ids.cpu().numpy().astype(np.int16)
            tgt = np.asarray(targets[start:end])
            v = (tgt >= 0) & (tgt < 1024)
            sid_match += int((pred[v] == tgt[v]).sum()); sid_total += int(v.sum())
            valid_total += int(valid.sum())
            if start == 0 or end == len(indices) or (start // batch_size) % 20 == 0:
                print(f"cache {split} {end}/{len(indices)}", flush=True)
        out.flush(); del out
        summary["splits"][split] = {"records": int(len(indices)), "valid_tokens": valid_total,
                                            "sid_target_match": sid_match / max(1, sid_total),
                                            "invalid_ze_abs_max": zero_invalid_max,
                                            "ze_sha256": sha256(cache / f"{split}_ze.npy"),
                                            "sid_sha256": sha256(PRETRAIN_SID / f"{split}_sid.npy")}
    if state_sha(tokenizer.state_dict()) != frozen_before:
        raise RuntimeError("Frozen E3 tokenizer changed while caching")
    summary["frozen_tokenizer_unchanged"] = True
    dump(cache / "audit.json", summary); (cache / "COMPLETE").touch()


def lr_at(step: int, total: int, warmup: int, peak: float, minimum: float, warmup_lr: float) -> float:
    if step < warmup:
        return warmup_lr + (peak - warmup_lr) * step / max(1, warmup)
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * p))


def run_mlm_epoch(model, loader, device, optimizer, scaler, epoch, args, train: bool) -> dict:
    model.train(train)
    # Match the E3 trainer exactly: report each metric as the unweighted mean
    # over batches (including masked-token count), and use that Val CE for
    # checkpoint selection.
    sums = {"loss": 0.0, "mlm_acc": 0.0, "token_entropy": 0.0,
            "masked_tokens": 0.0, "batches": 0}
    for step, (ze, sid) in enumerate(loader):
        ze = ze.to(device, non_blocking=True).float(); sid = sid.to(device, non_blocking=True).long()
        valid = (sid >= 0) & (sid < 1024)
        mask = random_mask(valid, 0.5)
        labels = sid[mask]
        if not labels.numel():
            continue
        if train:
            global_step = (epoch - 1) * len(loader) + step
            lr = lr_at(global_step, args.epochs * len(loader), 5 * len(loader), 5e-4, 1e-5, 1e-6)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(ze, valid, mask)
            loss = F.cross_entropy(logits.float(), labels)
        if train:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            prob = torch.softmax(logits.float(), -1)
            sums["loss"] += float(loss)
            sums["mlm_acc"] += float((logits.argmax(-1) == labels).float().mean())
            sums["token_entropy"] += float((-(prob * prob.clamp_min(1e-12).log()).sum(-1)).mean())
            sums["masked_tokens"] += float(labels.numel())
            sums["batches"] += 1
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            print(f"{'train' if train else 'val'} epoch={epoch} step={step+1}/{len(loader)} loss={float(loss):.5f}", flush=True)
    n = max(1, sums["batches"])
    return {"loss": sums["loss"] / n, "mlm_acc": sums["mlm_acc"] / n,
            "token_entropy": sums["token_entropy"] / n,
            "masked_tokens": sums["masked_tokens"] / n,
            "batches": sums["batches"]}


def pretrain(run_dir: Path, mode: str, device: torch.device, args) -> None:
    out = run_dir / "mlm" / mode
    if (out / "COMPLETE").exists():
        print(f"Reuse completed {mode} MLM", flush=True); return
    out.mkdir(parents=True, exist_ok=False)
    model, init_audit = initialize_continuous(mode, device)
    dump(out / "initialization_audit.json", init_audit)
    cache = run_dir / "pretrain_ze_cache"
    train_ds, val_ds = ZEDataset(cache, "train"), ZEDataset(cache, "val")
    generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=generator,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=384, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_epoch = float("inf"), None
    history_path = out / "training_history.csv"
    stat_names = ("loss", "mlm_acc", "token_entropy", "masked_tokens", "batches")
    history_fields = ["epoch"] + [f"train_{k}" for k in stat_names] + [f"val_{k}" for k in stat_names]
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = run_mlm_epoch(model, train_loader, device, optimizer, scaler, epoch, args, True)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                val_stats = run_mlm_epoch(model, val_loader, device, optimizer, scaler, epoch, args, False)
            row.update({f"val_{k}": v for k, v in val_stats.items()})
            if val_stats["loss"] < best:
                best, best_epoch = val_stats["loss"], epoch
                torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "epoch": epoch, "best_val_loss": best, "mode": mode,
                            "config": model_config()}, out / "checkpoint-best.pth")
        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=history_fields)
            if f.tell() == 0: w.writeheader()
            w.writerow(row)
        print(json.dumps(row), flush=True)
        if epoch in (50, args.epochs):
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "best_val_loss": best, "mode": mode,
                        "config": model_config()}, out / f"checkpoint-{epoch}.pth")
    dump(out / "summary.json", {"mode": mode, "epochs": args.epochs, "best_epoch": best_epoch,
                                 "best_val_loss": best, "elapsed_seconds": time.time() - start,
                                 "pretrain_seed": 0, "train_records": len(train_ds), "val_records": len(val_ds)})
    (out / "COMPLETE").touch()


def load_continuous_checkpoint(path: Path, device: torch.device) -> ContinuousSIDMLM:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ContinuousSIDMLM(**ckpt.get("config", model_config()))
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()


def fixed_masks(sid: np.ndarray, offset: int) -> np.ndarray:
    out = np.zeros_like(sid, dtype=bool)
    for j, row in enumerate(sid):
        idx = np.flatnonzero((row >= 0) & (row < 1024))
        if len(idx):
            rng = np.random.default_rng(20261018 + offset + j)
            out[j, rng.permutation(idx)[:max(1, int(round(.5 * len(idx))))]] = True
    return out


@torch.inference_mode()
def fixed_mlm_audit(run_dir: Path, device: torch.device) -> None:
    path = run_dir / "fixed_mask_mlm_comparison.json"
    if path.exists(): return
    sid_np = np.load(PRETRAIN_SID / "val_sid.npy", mmap_mode="r")
    ze_np = np.load(run_dir / "pretrain_ze_cache/val_ze.npy", mmap_mode="r")
    e3_ckpt = torch.load(E3_MLM, map_location="cpu", weights_only=False)
    e3 = fresh_sid_model(0); e3.load_state_dict(e3_ckpt["model"], strict=True); e3.to(device).eval()
    models = {"e3_sid": e3,
              "e12a_scratch": load_continuous_checkpoint(run_dir / "mlm/scratch/checkpoint-best.pth", device),
              "e12b_warm": load_continuous_checkpoint(run_dir / "mlm/warm/checkpoint-best.pth", device)}
    sums = {k: {"loss": 0.0, "correct": 0, "tokens": 0} for k in models}
    batch = 128
    for start in range(0, len(sid_np), batch):
        end = min(start + batch, len(sid_np))
        sid = np.asarray(sid_np[start:end]); mask_np = fixed_masks(sid, start)
        s = torch.from_numpy(sid.copy()).long().to(device); mask = torch.from_numpy(mask_np).to(device)
        valid = (s >= 0) & (s < 1024); labels = s[mask]
        ze = torch.from_numpy(np.asarray(ze_np[start:end]).copy()).float().to(device)
        logits_map = {"e3_sid": e3(s, mask_bool_matrix=mask),
                      "e12a_scratch": models["e12a_scratch"](ze, valid, mask),
                      "e12b_warm": models["e12b_warm"](ze, valid, mask)}
        for key, logits in logits_map.items():
            sums[key]["loss"] += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
            sums[key]["correct"] += int((logits.argmax(-1) == labels).sum())
            sums[key]["tokens"] += int(labels.numel())
    result = {k: {"loss": v["loss"] / v["tokens"], "masked_sid_accuracy": v["correct"] / v["tokens"],
                  "masked_tokens": v["tokens"]} for k, v in sums.items()}
    result["mask_policy"] = "same stateless 50% valid-position mask for all three models"
    result["e3_historical_best"] = {"epoch": int(e3_ckpt.get("epoch", -1)),
                                     "best_val_loss": float(e3_ckpt.get("best_metric", float("nan")))}
    dump(path, result)


def leakage_smoke(run_dir: Path, device: torch.device) -> None:
    out = run_dir / "smoke"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    model, audit = initialize_continuous("scratch", device); model.eval()
    ze = torch.randn(3, 256, 64, device=device)
    valid = torch.zeros(3, 256, dtype=torch.bool, device=device); valid[:, :31] = True
    mask = torch.zeros_like(valid); mask[:, [2, 7, 13]] = True
    with torch.no_grad():
        a = model(ze, valid, mask)
        changed = ze.clone(); changed[mask] = torch.randn_like(changed[mask]) * 1e5
        b = model(changed, valid, mask)
    if not torch.equal(a, b):
        raise RuntimeError(f"Masked z_e leakage detected: max diff={float((a-b).abs().max())}")
    grad_logits = model(ze, valid, mask)
    loss = F.cross_entropy(grad_logits.float(), torch.randint(0, 1024, (grad_logits.shape[0],), device=device)); loss.backward()
    if model.backbone.input_proj.weight.grad is None or not torch.isfinite(loss):
        raise RuntimeError("Smoke backward failed")
    dump(out / "audit.json", {"passed": True, "masked_value_replacement_exact": True,
                               "explicit_key_padding_mask": True, "spatial_embedding": False,
                               "temporal_embedding": False, "input_projection_receives_gradient": True,
                               "initialization": audit})
    (out / "COMPLETE").touch()


@torch.inference_mode()
def extract_downstream(run_dir: Path, mode: str, device: torch.device, batch_size: int = 32) -> None:
    out = run_dir / "features" / mode
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = e4.load_model(TOKENIZER, device).eval().requires_grad_(False)
    model = load_continuous_checkpoint(run_dir / f"mlm/{mode}/checkpoint-best.pth", device).requires_grad_(False)
    audit = {"mode": mode, "tokenizer": str(TOKENIZER), "mlm_checkpoint": str(run_dir / f"mlm/{mode}/checkpoint-best.pth"),
             "pooling": "masked valid-token mean+max", "splits": {}}
    for split in ("train", "val", "test"):
        waves = np.load(DOWNSTREAM / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(DOWNSTREAM / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(DOWNSTREAM / f"{split}_data_in_times.npy", mmap_mode="r")
        result = np.empty((len(waves), 1536), np.float32)
        for start in range(0, len(waves), batch_size):
            end = min(start + batch_size, len(waves))
            x = torch.from_numpy(np.asarray(waves[start:end], dtype=np.float32).copy()).to(device)
            c = torch.from_numpy(np.asarray(chans[start:end]).copy()).long().to(device)
            t = torch.from_numpy(np.asarray(times[start:end]).copy()).long().to(device)
            valid = tokenizer.valid_mask(x, c, t)
            ze = tokenizer.encode_to_quantizer_features(x, c, t).float()
            tokens = model.extract_tokens(ze, valid).float()
            count = valid.sum(1).clamp_min(1).unsqueeze(-1)
            mean = (tokens * valid.unsqueeze(-1)).sum(1) / count
            maximum = tokens.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
            maximum[~valid.any(1)] = 0
            result[start:end] = torch.cat([mean, maximum], 1).cpu().numpy()
        np.save(out / f"{split}_heartlang_mean_max.npy", result)
        audit["splits"][split] = {"records": len(result), "shape": list(result.shape),
                                           "sha256": sha256(out / f"{split}_heartlang_mean_max.npy"),
                                           "mean": float(result.mean()), "std": float(result.std()),
                                           "finite": bool(np.isfinite(result).all())}
    dump(out / "audit.json", audit); (out / "COMPLETE").touch()


class MLMOnly(nn.Module):
    def __init__(self, seed: int):
        super().__init__(); e4.set_seed(seed); self.classifier = e9.DualHead(1536, seed)

    def training_outputs(self, mlm): return self.classifier.training_outputs(mlm)
    def forward(self, mlm): return self.classifier(mlm)


def probability_metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    base, prob = e7.classification(logits, labels, "CE")
    true = labels.argmax(1); pred = prob.argmax(1); correct = pred == true; conf = prob.max(1)
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any(): ece += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    base.update(brier_score=float(np.mean(np.sum((prob - labels) ** 2, axis=1))), ece15_top_label=ece)
    return base, prob


@torch.no_grad()
def eval_downstream(model, method: str, arrays: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval(); chunks = []
    regions, region_mask = e11.masked_region_means(arrays["lead"], arrays["lead_mask"])
    for start in range(0, len(arrays["labels"]), 96):
        if method == "e12a_mlm_only": raw = model(arrays["mlm"][start:start+96])
        else: raw = model(arrays["mlm"][start:start+96], arrays["global"][start:start+96],
                          regions[start:start+96], region_mask[start:start+96])
        chunks.append(raw.cpu().numpy())
    raw = np.concatenate(chunks)
    metrics, prob = probability_metrics(raw, arrays["labels"].cpu().numpy())
    return metrics, prob, raw


def load_downstream_arrays(run_dir: Path, mode: str, device: torch.device) -> dict:
    data = {}
    for split in ("train", "val", "test"):
        labels = np.load(DOWNSTREAM / f"{split}_labels.npy").astype(np.float32)
        mlm = np.load(run_dir / f"features/{mode}/{split}_heartlang_mean_max.npy")
        lead = np.load(E10_RUN / f"features/{split}_lead_ze_mean_max.npy")
        lead_mask = np.load(E10_RUN / f"features/{split}_lead_present.npy")
        global_ze = np.load(E10_RUN / f"features/{split}_global_ze_mean_max.npy")
        if len({len(labels), len(mlm), len(lead), len(global_ze)}) != 1:
            raise RuntimeError(f"{split}: feature alignment mismatch")
        data[split] = {"labels": torch.from_numpy(labels).to(device),
                       "mlm": torch.from_numpy(mlm).float().to(device),
                       "lead": torch.from_numpy(lead).float().to(device),
                       "lead_mask": torch.from_numpy(lead_mask).bool().to(device),
                       "global": torch.from_numpy(global_ze).float().to(device)}
    return data


def build_downstream(method: str, seed: int, device: torch.device) -> nn.Module:
    if method == "e12a_mlm_only": return MLMOnly(seed).to(device)
    return e11.E11RegionResidual(seed).to(device)


def train_downstream_one(run_dir: Path, method: str, seed: int, args, device: torch.device) -> dict:
    out = run_dir / "downstream" / method / f"seed{seed}"
    if (out / "VAL_COMPLETE").exists(): return json.loads((out / "val_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    mode = "scratch" if method.startswith("e12a") else "warm"
    data = load_downstream_arrays(run_dir, mode, device)
    model = build_downstream(method, seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.05)
    target = data["train"]["labels"].argmax(1); steps = len(target) // 64
    generator = torch.Generator().manual_seed(seed)
    history, best, best_epoch, best_state = [], -float("inf"), None, None
    train_regions, train_region_mask = e11.masked_region_means(data["train"]["lead"], data["train"]["lead_mask"])
    for epoch in range(1, 31):
        model.train(); order = torch.randperm(len(target), generator=generator).to(device)
        main_losses, aux_losses = [], []
        for step in range(steps):
            idx = order[step*64:(step+1)*64]
            lr = lr_at((epoch-1)*steps+step, 30*steps, 5*steps, 5e-4, 1e-5, 1e-6)
            for g in optimizer.param_groups: g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if method == "e12a_mlm_only":
                main, aux = model.training_outputs(data["train"]["mlm"][idx])
            else:
                main, aux = model.training_outputs(data["train"]["mlm"][idx], data["train"]["global"][idx],
                                                   train_regions[idx], train_region_mask[idx])
            lm = F.cross_entropy(main, target[idx]); la = e9.patient_auxiliary_loss(aux, target[idx]); loss = lm + .2 * la
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
            main_losses.append(float(lm)); aux_losses.append(float(la))
        val, _, _ = eval_downstream(model, method, data["val"])
        row = {"epoch": epoch, "train_main_ce": float(np.mean(main_losses)),
               "train_patient_ce": float(np.mean(aux_losses)), "val": val}; history.append(row)
        if val["r2"] > best:
            best, best_epoch, best_state = val["r2"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val, val_prob, val_logits = eval_downstream(model, method, data["val"])
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "method": method, "seed": seed,
                "best_epoch": best_epoch, "selection": "Val Overall R2"}, out / "checkpoint-best.pth")
    np.save(out / "val_logits.npy", val_logits); np.save(out / "val_probabilities.npy", val_prob)
    dump(out / "history.json", history)
    result = {"method": method, "sentence_mode": mode, "pretrain_seed": 0, "downstream_seed": seed,
              "best_epoch": best_epoch, "val": val,
              "trainable_parameters": sum(p.numel() for p in model.parameters())}
    dump(out / "val_summary.json", result); (out / "VAL_COMPLETE").touch()
    return result


def train_all_downstream(run_dir: Path, args, device: torch.device) -> None:
    summary_path = run_dir / "downstream/validation_selection.json"
    if summary_path.exists(): return
    rows = []
    for method in ("e12a_region", "e12b_region", "e12a_mlm_only"):
        for seed in (0, 1, 2):
            set_status(run_dir, f"downstream_{method}_seed{seed}_validation_only")
            rows.append(train_downstream_one(run_dir, method, seed, args, device))
    methods = {}
    for method in ("e12a_region", "e12b_region", "e12a_mlm_only"):
        rr = [x for x in rows if x["method"] == method]
        methods[method] = {"val_r2_mean": float(np.mean([x["val"]["r2"] for x in rr])),
                           "val_r2_sd": float(np.std([x["val"]["r2"] for x in rr], ddof=1)),
                           "per_seed": rr}
    selected = max(("e12a_region", "e12b_region"), key=lambda m: methods[m]["val_r2_mean"])
    dump(summary_path, {"selection_split": "validation", "test_used": False,
                        "selected_e12_method": selected, "methods": methods,
                        "region_value_e12a_val_delta": methods["e12a_region"]["val_r2_mean"] - methods["e12a_mlm_only"]["val_r2_mean"]})


def finalize_test(run_dir: Path, device: torch.device) -> None:
    final_path = run_dir / "downstream/final_test_summary.json"
    if final_path.exists(): return
    selection = json.loads((run_dir / "downstream/validation_selection.json").read_text())
    method = selection["selected_e12_method"]; mode = "scratch" if method.startswith("e12a") else "warm"
    data = load_downstream_arrays(run_dir, mode, device); rows = []
    for seed in (0, 1, 2):
        out = run_dir / "downstream" / method / f"seed{seed}"
        ckpt = torch.load(out / "checkpoint-best.pth", map_location=device, weights_only=False)
        model = build_downstream(method, seed, device); model.load_state_dict(ckpt["model"], strict=True)
        test, prob, logits = eval_downstream(model, method, data["test"])
        np.save(out / "test_logits.npy", logits); np.save(out / "test_probabilities.npy", prob)
        rows.append({"seed": seed, "test": test})
    keys = ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy", "brier_score", "ece15_top_label")
    agg = {k: {"mean": float(np.mean([r["test"][k] for r in rows])),
               "sd": float(np.std([r["test"][k] for r in rows], ddof=1))} for k in keys}
    agg["per_output_r2"] = {name: {"mean": float(np.mean([r["test"]["per_output_r2"][name] for r in rows])),
                                           "sd": float(np.std([r["test"]["per_output_r2"][name] for r in rows], ddof=1))}
                             for name in LABEL_NAMES}
    dump(final_path, {"selected_by": "mean validation Overall R2 across downstream seeds",
                      "method": method, "test_evaluated_methods": [method], "rows": rows, "aggregate": agg})


def overlap_audit(run_dir: Path) -> None:
    path = run_dir / "pretraining_downstream_overlap_audit.json"
    if path.exists(): return
    out = {}
    for ps in ("train", "val"):
        p = pd.read_csv(PRETRAIN_INDEX / f"{ps}_metadata.csv", low_memory=False); out[ps] = {}
        for ds in ("train", "val", "test"):
            d = pd.read_csv(DOWNSTREAM / f"{ds}_metadata.csv", low_memory=False)
            out[ps][ds] = {key: len(set(p[key].dropna().astype(str)) & set(d[key].dropna().astype(str)))
                           for key in ("entity_id", "csv_name", "member_name")}
    if any(out[ps][ds][k] for ps in out for ds in ("val", "test") for k in out[ps][ds]):
        raise RuntimeError("Downstream val/test overlap detected in E12 MLM pretraining")
    dump(path, {"overlap_counts": out, "downstream_val_test_overlap_zero": True,
                "note": "Locked downstream train is intentionally present in unsupervised pretrain train, matching E3."})


def aggregate_historical_e11() -> dict:
    summary = json.loads((E11_RUN / "metrics/aggregate_summary.json").read_text())
    b = summary["methods"]["e11_region_residual"]
    return {"val": b["val"], "test": b["test"], "source": str(E11_RUN), "retrained": False}


def report(run_dir: Path) -> None:
    selection = json.loads((run_dir / "downstream/validation_selection.json").read_text())
    final = json.loads((run_dir / "downstream/final_test_summary.json").read_text())
    mlm = json.loads((run_dir / "fixed_mask_mlm_comparison.json").read_text())
    e11h = aggregate_historical_e11(); methods = selection["methods"]; selected = final["method"]; test = final["aggregate"]
    names = {"e12a_region": "E12-A 从头连续输入 + E11区域", "e12b_region": "E12-B Warm-start连续输入 + E11区域",
             "e12a_mlm_only": "E12-A仅Transformer Mean+Max"}
    lines = ["# E12：连续 z_e 输入替代 SID Embedding", "", "## 结论摘要", "",
             f"按三种下游seed的平均Val Overall R²，选中的E12方案是 **{names[selected]}**。",
             f"其Test Overall R²为 **{test['r2']['mean']:.6f} ± {test['r2']['sd']:.6f}**；历史E11为 **{e11h['test']['r2']['mean']:.6f}**。",
             "所有架构和checkpoint选择只使用Val；Test仅对选中的E12方案评估一次。", "",
             "## 数据与冻结审计", "",
             "- E3 MorphologyEncoder和Codebook冻结；Codebook只提供已有SID监督目标。",
             "- E3无监督池为87,066/8,941条train/val；与下游Val/Test按entity_id、csv_name、member_name核对均无重叠。",
             "- Mask位置的z_e在Linear之后被完整替换；显式key-padding mask已启用；Spatial/Temporal Embedding关闭。", "",
             "## 固定Mask的MLM验证比较", "", "| 模型 | Val CE | Masked SID Accuracy |", "|---|---:|---:|"]
    for key, label in (("e3_sid", "E3 SID输入"), ("e12a_scratch", "E12-A连续输入"), ("e12b_warm", "E12-B连续Warm-start")):
        lines.append(f"| {label} | {mlm[key]['loss']:.6f} | {mlm[key]['masked_sid_accuracy']:.6f} |")
    lines += ["", "## 下游验证集比较", "", "| 方法 | Val Overall R² mean±sd |", "|---|---:|"]
    lines.append(f"| 历史E11 SID输入区域融合 | {e11h['val']['r2']['mean']:.6f} ± {e11h['val']['r2']['std_ddof1']:.6f} |")
    for key in ("e12a_region", "e12b_region", "e12a_mlm_only"):
        b = methods[key]; lines.append(f"| {names[key]} | {b['val_r2_mean']:.6f} ± {b['val_r2_sd']:.6f} |")
    lines += ["", f"E12-A加入E11区域残差后的Val R²变化为 **{selection['region_value_e12a_val_delta']:+.6f}**。", "",
              "## 最终测试集（仅Val选中的E12方案）", "",
              "| 模型 | Overall R² | OMI | NOMI | Clinical Control | Healthy | Macro AUC | Macro F1 | MSE | Brier | ECE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ht = e11h["test"]
    lines.append("| 历史E11 | " + " | ".join([f"{ht['r2']['mean']:.6f}"] + [f"{ht['per_output_r2'][n]['mean']:.6f}" for n in LABEL_NAMES] + [f"{ht['macro_roc_auc']['mean']:.6f}", f"{ht['macro_f1']['mean']:.6f}", f"{ht['mse']['mean']:.6f}", "—", "—"]) + " |")
    lines.append("| " + names[selected] + " | " + " | ".join([f"{test['r2']['mean']:.6f}"] + [f"{test['per_output_r2'][n]['mean']:.6f}" for n in LABEL_NAMES] + [f"{test['macro_roc_auc']['mean']:.6f}", f"{test['macro_f1']['mean']:.6f}", f"{test['mse']['mean']:.6f}", f"{test['brier_score']['mean']:.6f}", f"{test['ece15_top_label']['mean']:.6f}"]) + " |")
    delta = test["r2"]["mean"] - ht["r2"]["mean"]
    lines += ["", "## 回答六个问题", "",
              f"1. **连续输入是否优于SID输入：** Test Overall R²差值为 **{delta:+.6f}**；架构选择依据仍以Val表格为准。",
              "2. **MLM与下游是否一致：** 见固定Mask MLM表与Val R²表；两者分别测量SID可预测性和诊断概率质量，不假定单调对应。",
              "3. **类别改善：** 以上表中选中E12与历史E11四类R²的差值用于判断，重点查看OMI/NOMI/Clinical Control。",
              f"4. **区域分支价值：** E12-A区域分支Val增量为 {selection['region_value_e12a_val_delta']:+.6f}。",
              f"5. **从头与Warm-start：** Val选择结果为 {selected}；未使用Test作此判断。",
              "6. **是否直接输入96点波形：** 仅当连续z_e在Val上显示稳定优势或MLM明显受益时，才建议做严格同预算的原始波形输入对照；否则不宜直接扩大计算预算。", "",
              "## 局限", "", "- 只有一个MLM预训练seed（seed0）；下游seed0/1/2不能替代独立MLM种子。",
              "- 当前连续输入是E3 MorphologyEncoder的z_e，不是HeartLang原文的96点原始心拍Conv输入。",
              "- 历史E11未重训；其特征、划分和下游协议保持一致并作为只读基线。", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def protocol(run_dir: Path, epochs: int) -> None:
    if (run_dir / "protocol.json").exists(): return
    dump(run_dir / "protocol.json", {"experiment": EXP, "tokenizer": str(TOKENIZER), "e3_mlm": str(E3_MLM),
                                      "pretrain_seed": 0, "downstream_seeds": [0,1,2], "mlm_epochs": epochs,
                                      "mlm_batch": 256, "mask_ratio": .5, "lr": 5e-4, "min_lr": 1e-5,
                                      "warmup_epochs": 5, "weight_decay": .05, "clip": 3,
                                      "spatial_embedding": False, "temporal_embedding": False,
                                      "explicit_key_padding_mask": True,
                                      "downstream_loss": "CE4 + 0.2*patient CE3", "downstream_epochs": 30,
                                      "selection": "MLM best Val CE; downstream checkpoint best Val Overall R2; architecture mean Val Overall R2",
                                      "test_policy": "only selected E12 architecture after all validation decisions",
                                      "historical_e11_overwritten": False})


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["smoke", "cache", "pretrain_a", "pretrain_b", "fixed_mlm", "features_a", "features_b", "downstream", "test", "report", "audit"])
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_RUN); ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args(); run_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage != "smoke": run_dir.mkdir(parents=True, exist_ok=True); protocol(run_dir, args.epochs)
    if args.stage == "smoke": leakage_smoke(run_dir, device)
    elif args.stage == "audit": overlap_audit(run_dir)
    elif args.stage == "cache": cache_pretrain(run_dir, device, args.batch_size)
    elif args.stage == "pretrain_a": pretrain(run_dir, "scratch", device, args)
    elif args.stage == "pretrain_b": pretrain(run_dir, "warm", device, args)
    elif args.stage == "fixed_mlm": fixed_mlm_audit(run_dir, device)
    elif args.stage == "features_a": extract_downstream(run_dir, "scratch", device)
    elif args.stage == "features_b": extract_downstream(run_dir, "warm", device)
    elif args.stage == "downstream": train_all_downstream(run_dir, args, device)
    elif args.stage == "test": finalize_test(run_dir, device)
    elif args.stage == "report": report(run_dir); set_status(run_dir, "complete"); (run_dir / "COMPLETE").touch()


if __name__ == "__main__": main()
