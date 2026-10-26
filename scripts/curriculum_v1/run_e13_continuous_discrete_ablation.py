#!/usr/bin/env python3
"""E13: continuous ECG representation and discrete SID supervision ablations.

This file deliberately reuses the locked E12 data contract and E11 downstream
head.  Historical checkpoints are read-only.  Architecture selection uses the
mean validation Overall R2 over downstream seeds 0/1/2; only that architecture
is evaluated on test.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

from heartlang.models.modeling_pretrain import TokenEmbedding
from heartlang.utils.utils import set_seed
import run_e4_aggregation as e4
import run_e9_ze_fusion_r2 as e9
import run_e10_lead_aware_ze_fusion as e10
import run_e11_region_residual_fusion as e11
import run_e12_continuous_input as e12


EXP = "e13_continuous_discrete_ablation"
DEFAULT_RUN = ROOT / "results" / EXP
E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
E10_RUN = ROOT / "results/curriculum_v1/e10_lead_aware_ze_fusion_20261018"
DOWNSTREAM = e9.DATA
LABEL_NAMES = e9.LABEL_NAMES


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_status(run_dir: Path, value: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.txt").write_text(value, encoding="utf-8")
    print(value, flush=True)


def state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


class ZERegressionMLM(nn.Module):
    """E13-C: masked continuous z_e reconstruction."""
    def __init__(self):
        super().__init__()
        base, audit = e12.initialize_continuous("scratch", torch.device("cpu"))
        self.backbone = base.backbone
        set_seed(130003)
        self.reg_head = nn.Linear(768, 64)
        self.init_audit = audit

    def forward(self, ze, valid, mask):
        h = self.backbone.forward_feature(ze, valid, mask)[:, 1:]
        use = mask.bool() & valid.bool()
        return self.reg_head(h[use])

    def extract_tokens(self, ze, valid):
        return self.backbone.token_features(ze, valid)


class HybridSIDMLM(nn.Module):
    """E13-D: additive z_e + SID embedding, jointly masked."""
    def __init__(self):
        super().__init__()
        base, audit = e12.initialize_continuous("scratch", torch.device("cpu"))
        sid_source = e12.fresh_sid_model(0)
        self.input_proj = base.backbone.input_proj
        self.sid_embed = copy.deepcopy(sid_source.backbone.sid_embed)
        self.mask_token = base.backbone.mask_token
        self.cls_token = base.backbone.cls_token
        self.pos_embed = base.backbone.pos_embed
        self.dropout = base.backbone.dropout
        self.transformer = base.backbone.transformer
        self.norm_layer = base.backbone.norm_layer
        self.lm_head = base.lm_head
        self.init_audit = audit

    def features(self, ze, sid, valid, mask=None):
        valid = valid.bool()
        safe_sid = sid.clamp(0, 1024)
        # 1/sqrt(2) keeps the initial variance close to a single input branch.
        x = (self.input_proj(ze.float()) + self.sid_embed(safe_sid.long())) / math.sqrt(2.0)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        use = torch.zeros_like(valid) if mask is None else (mask.bool() & valid)
        # Both continuous and discrete inputs are replaced together.
        x = torch.where(use.unsqueeze(-1), self.mask_token.expand_as(x), x)
        cls = self.cls_token.view(1, 1, -1).expand(len(x), 1, -1)
        x = torch.cat([cls, x], 1)
        full_valid = torch.cat([torch.ones(len(x), 1, dtype=torch.bool, device=x.device), valid], 1)
        x = self.dropout(x + self.pos_embed)
        x = e12.masked_transformer_forward(self.transformer, x, full_valid)
        return self.norm_layer(x) * full_valid.unsqueeze(-1).to(x.dtype)

    def forward(self, ze, sid, valid, mask):
        h = self.features(ze, sid, valid, mask)[:, 1:]
        return self.lm_head(h[mask.bool() & valid.bool()])

    def extract_tokens(self, ze, sid, valid):
        return self.features(ze, sid, valid)[:, 1:]


class WaveformSIDMLM(nn.Module):
    """E13-E: official TokenEmbedding Conv1d with leak-safe raw masking."""
    def __init__(self):
        super().__init__()
        base, audit = e12.initialize_continuous("scratch", torch.device("cpu"))
        set_seed(130005)
        self.token_embed = TokenEmbedding(c_in=96, d_model=768)
        self.raw_mask_token = nn.Parameter(torch.zeros(1, 1, 96))
        self.mask_token = base.backbone.mask_token
        self.cls_token = base.backbone.cls_token
        self.pos_embed = base.backbone.pos_embed
        self.dropout = base.backbone.dropout
        self.transformer = base.backbone.transformer
        self.norm_layer = base.backbone.norm_layer
        self.lm_head = base.lm_head
        self.init_audit = audit

    def features(self, wave, valid, mask=None):
        valid = valid.bool()
        use = torch.zeros_like(valid) if mask is None else (mask.bool() & valid)
        # Mask before Conv1d so adjacent visible positions cannot receive the
        # hidden heartbeat through the kernel. Invalid tokens are zeroed too.
        raw = wave.float() * valid.unsqueeze(-1).to(wave.dtype)
        raw = torch.where(use.unsqueeze(-1), self.raw_mask_token.expand_as(raw), raw)
        x = self.token_embed(raw)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        # Also replace the masked center after convolution for exact removal.
        x = torch.where(use.unsqueeze(-1), self.mask_token.expand_as(x), x)
        cls = self.cls_token.view(1, 1, -1).expand(len(x), 1, -1)
        x = torch.cat([cls, x], 1)
        full_valid = torch.cat([torch.ones(len(x), 1, dtype=torch.bool, device=x.device), valid], 1)
        x = self.dropout(x + self.pos_embed)
        x = e12.masked_transformer_forward(self.transformer, x, full_valid)
        return self.norm_layer(x) * full_valid.unsqueeze(-1).to(x.dtype)

    def forward(self, wave, valid, mask):
        h = self.features(wave, valid, mask)[:, 1:]
        return self.lm_head(h[mask.bool() & valid.bool()])

    def extract_tokens(self, wave, valid):
        return self.features(wave, valid)[:, 1:]


class WaveDataset(Dataset):
    def __init__(self, cache: Path, split: str):
        self.wave = np.load(cache / f"{split}_wave.npy", mmap_mode="r")
        self.sid = np.load(e12.PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        if self.wave.shape[:2] != self.sid.shape or self.wave.shape[2] != 96:
            raise RuntimeError(f"{split}: waveform/SID mismatch")
    def __len__(self): return len(self.sid)
    def __getitem__(self, i):
        return np.array(self.wave[i], copy=True), np.array(self.sid[i], copy=True)


def prepare_a(run_dir: Path) -> None:
    out = run_dir / "pretrain/e13_a_no_mlm"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    model, audit = e12.initialize_continuous("scratch", torch.device("cpu"))
    torch.save({"model": state_cpu(model), "variant": "e13_a", "pretrained": False,
                "config": e12.model_config()}, out / "checkpoint-random-init.pth")
    audit.update({"semantics": "random Transformer frozen during downstream; only E11 fusion/head is trained",
                  "mlm_epochs": 0, "selection": "none"})
    dump(out / "initialization_audit.json", audit)
    (out / "COMPLETE").touch()


def load_model(variant: str, run_dir: Path, device: torch.device):
    if variant == "e13_a":
        model, _ = e12.initialize_continuous("scratch", device)
        ck = torch.load(run_dir / "pretrain/e13_a_no_mlm/checkpoint-random-init.pth", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
    elif variant == "e13_c":
        model = ZERegressionMLM().to(device)
        ck = torch.load(run_dir / "pretrain/e13_c_ze_regression/checkpoint-best.pth", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
    elif variant == "e13_d":
        model = HybridSIDMLM().to(device)
        ck = torch.load(run_dir / "pretrain/e13_d_hybrid_sid/checkpoint-best.pth", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
    elif variant == "e13_e":
        model = WaveformSIDMLM().to(device)
        ck = torch.load(run_dir / "pretrain/e13_e_waveform_sid/checkpoint-best.pth", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
    else:
        raise ValueError(variant)
    return model.eval()


def make_loaders(run_dir: Path, variant: str):
    if variant == "e13_e":
        train = WaveDataset(run_dir / "pretrain_wave_cache", "train")
        val = WaveDataset(run_dir / "pretrain_wave_cache", "val")
    else:
        train = e12.ZEDataset(E12_RUN / "pretrain_ze_cache", "train")
        val = e12.ZEDataset(E12_RUN / "pretrain_ze_cache", "val")
    gen = torch.Generator().manual_seed(0)
    return (DataLoader(train, 256, shuffle=True, generator=gen, num_workers=8, pin_memory=True, drop_last=True),
            DataLoader(val, 384, shuffle=False, num_workers=8, pin_memory=True))


def reconstruction_stats(pred, target):
    return {"mse": float(F.mse_loss(pred, target)), "mae": float(F.l1_loss(pred, target)),
            "cosine_similarity": float(F.cosine_similarity(pred, target, dim=-1).mean())}


def epoch(run_dir, variant, model, loader, optimizer, scaler, ep, epochs, device, train):
    model.train(train); sums = {}; batches = 0
    for step, (source, sid) in enumerate(loader):
        source = source.to(device, non_blocking=True).float(); sid = sid.to(device, non_blocking=True).long()
        valid = (sid >= 0) & (sid < 1024); mask = e12.random_mask(valid, .5); labels = sid[mask]
        if not labels.numel(): continue
        if train:
            gs = (ep - 1) * len(loader) + step
            lr = e12.lr_at(gs, epochs * len(loader), 5 * len(loader), 5e-4, 1e-5, 1e-6)
            for g in optimizer.param_groups: g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        # Continuous regression is numerically unstable in fp16 once the
        # cosine schedule reaches its high-LR region. BF16 keeps the same
        # memory/computation budget while retaining the exponent range of
        # float32. The objective remains the declared plain MSE.
        amp_dtype = torch.bfloat16 if variant == "e13_c" else torch.float16
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
            if variant == "e13_c":
                pred = model(source, valid, mask).float(); target = source[mask].float()
                loss = F.mse_loss(pred, target)
            elif variant == "e13_d":
                pred = model(source, sid, valid, mask).float(); loss = F.cross_entropy(pred, labels)
            else:
                pred = model(source, valid, mask).float(); loss = F.cross_entropy(pred, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{variant}: non-finite loss at epoch={ep} step={step+1}")
        if train:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            stat = reconstruction_stats(pred, target) if variant == "e13_c" else {
                "ce": float(loss), "masked_sid_accuracy": float((pred.argmax(-1) == labels).float().mean())}
            stat["loss"] = float(loss)
            for k, v in stat.items(): sums[k] = sums.get(k, 0.0) + v
            batches += 1
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            print(f"{variant} {'train' if train else 'val'} epoch={ep} {step+1}/{len(loader)} loss={float(loss):.6f}", flush=True)
    return {k: v / max(1, batches) for k, v in sums.items()} | {"batches": batches}


def pretrain(run_dir: Path, variant: str, device: torch.device, epochs: int) -> None:
    names = {"e13_c": "e13_c_ze_regression", "e13_d": "e13_d_hybrid_sid", "e13_e": "e13_e_waveform_sid"}
    out = run_dir / "pretrain" / names[variant]
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    set_seed(0)
    model = {"e13_c": ZERegressionMLM, "e13_d": HybridSIDMLM, "e13_e": WaveformSIDMLM}[variant]().to(device)
    dump(out / "initialization_audit.json", model.init_audit)
    train_loader, val_loader = make_loaders(run_dir, variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and variant != "e13_c")
    best = float("inf"); best_epoch = None; history = []
    start = time.time()
    for ep in range(1, epochs + 1):
        tr = epoch(run_dir, variant, model, train_loader, optimizer, scaler, ep, epochs, device, True)
        row = {"epoch": ep, "train": tr}
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            with torch.no_grad(): va = epoch(run_dir, variant, model, val_loader, None, None, ep, epochs, device, False)
            row["val"] = va
            if va["loss"] < best:
                best = va["loss"]; best_epoch = ep
                torch.save({"model": state_cpu(model), "variant": variant, "epoch": ep,
                            "best_val_loss": best, "config": e12.model_config()}, out / "checkpoint-best.pth")
        history.append(row); dump(out / "training_history.json", history)
        print(json.dumps(row), flush=True)
    total = time.time() - start
    dump(out / "summary.json", {"variant": variant, "pretrain_seed": 0, "epochs": epochs,
                                 "best_epoch": best_epoch, "best_val_loss": best,
                                 "gpu_wall_seconds": total,
                                 "trainable_parameters": sum(p.numel() for p in model.parameters())})
    (out / "COMPLETE").touch()


def cache_wave(run_dir: Path) -> None:
    out = run_dir / "pretrain_wave_cache"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    sources = {s: np.load(e12.PRETRAIN_SOURCE / f"{s}_data.npy", mmap_mode="r") for s in ("train", "val")}
    summary = {}
    for split in ("train", "val"):
        codes = np.load(e12.PRETRAIN_INDEX / f"{split}_source_code.npy")
        indices = np.load(e12.PRETRAIN_INDEX / f"{split}_source_index.npy")
        mm = np.lib.format.open_memmap(out / f"{split}_wave.npy", mode="w+", dtype=np.float32,
                                       shape=(len(indices), 256, 96))
        for st in range(0, len(indices), 512):
            en = min(st + 512, len(indices)); mm[st:en] = e12.gather_source(sources, codes[st:en], indices[st:en])
            if st == 0 or en == len(indices) or (st // 512) % 20 == 0: print(f"wave cache {split} {en}/{len(indices)}", flush=True)
        mm.flush(); del mm
        summary[split] = {"records": len(indices), "shape": [len(indices), 256, 96]}
    dump(out / "audit.json", summary); (out / "COMPLETE").touch()


@torch.inference_mode()
def leakage_smoke(run_dir: Path, device: torch.device) -> None:
    out = run_dir / "smoke"; out.mkdir(parents=True, exist_ok=True)
    set_seed(9); b = 2
    ze = F.normalize(torch.randn(b, 256, 64, device=device), dim=-1)
    sid = torch.randint(0, 1024, (b, 256), device=device); valid = torch.ones(b, 256, dtype=torch.bool, device=device)
    valid[:, 220:] = False; sid[~valid] = 1024
    mask = torch.zeros_like(valid); mask[:, 10] = True; mask[:, 50] = True
    results = {}
    for name, model, source in (("e13_c", ZERegressionMLM().to(device).eval(), ze),
                                ("e13_d", HybridSIDMLM().to(device).eval(), ze),
                                ("e13_e", WaveformSIDMLM().to(device).eval(), torch.randn(b,256,96,device=device))):
        source2 = source.clone(); source2[mask] = torch.randn_like(source2[mask]) * 1000
        if name == "e13_d":
            sid2 = sid.clone(); sid2[mask] = (sid2[mask] + 333) % 1024
            a = model(source, sid, valid, mask); c = model(source2, sid2, valid, mask)
        else:
            a = model(source, valid, mask); c = model(source2, valid, mask)
        delta = float((a - c).abs().max())
        results[name] = {"masked_target_mutation_max_logit_or_pred_delta": delta, "passed": delta < 1e-5}
    # Explicitly verify the legacy ordering is unsafe for the official Conv.
    emb = TokenEmbedding(c_in=96, d_model=768).to(device).eval()
    w = torch.randn(1, 8, 96, device=device); w2 = w.clone(); w2[:, 3] += 100
    legacy_delta = float((emb(w)[:, 2] - emb(w2)[:, 2]).abs().max())
    results["legacy_conv_audit"] = {"neighbor_delta_if_mask_after_conv": legacy_delta, "unsafe": legacy_delta > 0}
    if not all(results[k]["passed"] for k in ("e13_c", "e13_d", "e13_e")):
        raise RuntimeError(results)
    dump(out / "leakage_checks.json", results); (out / "COMPLETE").touch()


@torch.inference_mode()
def extract_features(run_dir: Path, variant: str, device: torch.device, batch_size=32) -> None:
    out = run_dir / "features" / variant
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    model = load_model(variant, run_dir, device).requires_grad_(False)
    tokenizer = None if variant == "e13_e" else e4.load_model(e12.TOKENIZER, device).eval().requires_grad_(False)
    audit = {"variant": variant, "pooling": "valid token Mean+Max", "splits": {}}
    for split in ("train", "val", "test"):
        waves = np.load(DOWNSTREAM / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(DOWNSTREAM / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(DOWNSTREAM / f"{split}_data_in_times.npy", mmap_mode="r")
        fixed_sid = np.load(e9.SID / f"{split}_sid.npy", mmap_mode="r") if variant == "e13_d" else None
        result = np.empty((len(waves), 1536), np.float32)
        for st in range(0, len(waves), batch_size):
            en = min(st + batch_size, len(waves))
            x = torch.from_numpy(np.asarray(waves[st:en], dtype=np.float32).copy()).to(device)
            c = torch.from_numpy(np.asarray(chans[st:en]).copy()).long().to(device)
            t = torch.from_numpy(np.asarray(times[st:en]).copy()).long().to(device)
            if variant == "e13_e":
                valid = (x.abs().amax(-1) > 1e-8) & (c > 0) & (t > 0)
                tokens = model.extract_tokens(x, valid).float()
            else:
                valid = tokenizer.valid_mask(x, c, t)
                ze = tokenizer.encode_to_quantizer_features(x, c, t).float()
                if variant == "e13_d":
                    s = torch.from_numpy(np.asarray(fixed_sid[st:en]).copy()).long().to(device)
                    valid = (s >= 0) & (s < 1024); tokens = model.extract_tokens(ze, s, valid).float()
                else: tokens = model.extract_tokens(ze, valid).float()
            count = valid.sum(1).clamp_min(1).unsqueeze(-1)
            mean = (tokens * valid.unsqueeze(-1)).sum(1) / count
            mx = tokens.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1); mx[~valid.any(1)] = 0
            result[st:en] = torch.cat([mean, mx], 1).cpu().numpy()
        np.save(out / f"{split}_heartlang_mean_max.npy", result)
        audit["splits"][split] = {"shape": list(result.shape), "finite": bool(np.isfinite(result).all()),
                                   "mean": float(result.mean()), "std": float(result.std())}
    dump(out / "audit.json", audit); (out / "COMPLETE").touch()


def probability_metrics(logits, labels): return e12.probability_metrics(logits, labels)


def load_arrays(run_dir: Path, variant: str, device):
    data = {}
    feature_root = E12_RUN / "features/scratch" if variant == "e13_b" else run_dir / "features" / variant
    for split in ("train", "val", "test"):
        labels = np.load(DOWNSTREAM / f"{split}_labels.npy").astype(np.float32)
        mlm = np.load(feature_root / f"{split}_heartlang_mean_max.npy")
        lead = np.load(E10_RUN / f"features/{split}_lead_ze_mean_max.npy")
        lead_mask = np.load(E10_RUN / f"features/{split}_lead_present.npy")
        glob = np.load(E10_RUN / f"features/{split}_global_ze_mean_max.npy")
        data[split] = {"labels": torch.from_numpy(labels).to(device), "mlm": torch.from_numpy(mlm).float().to(device),
                       "lead": torch.from_numpy(lead).float().to(device), "lead_mask": torch.from_numpy(lead_mask).bool().to(device),
                       "global": torch.from_numpy(glob).float().to(device)}
    return data


@torch.no_grad()
def eval_head(model, arrays):
    model.eval(); regions, rm = e11.masked_region_means(arrays["lead"], arrays["lead_mask"]); chunks=[]
    for st in range(0, len(arrays["labels"]), 96):
        chunks.append(model(arrays["mlm"][st:st+96], arrays["global"][st:st+96], regions[st:st+96], rm[st:st+96]).cpu().numpy())
    logits=np.concatenate(chunks); metrics, prob=probability_metrics(logits, arrays["labels"].cpu().numpy())
    return metrics, prob, logits


def train_head_one(run_dir: Path, variant: str, seed: int, device):
    out = run_dir / "downstream" / variant / f"seed{seed}"
    if (out / "VAL_COMPLETE").exists(): return json.loads((out / "val_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False); data=load_arrays(run_dir, variant, device)
    model=e11.E11RegionResidual(seed).to(device); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
    target=data["train"]["labels"].argmax(1); steps=len(target)//64; gen=torch.Generator().manual_seed(seed)
    regions,rm=e11.masked_region_means(data["train"]["lead"],data["train"]["lead_mask"])
    best=-float("inf"); state=None; best_ep=None; history=[]
    for ep in range(1,31):
        model.train(); order=torch.randperm(len(target),generator=gen).to(device); losses=[]
        for step in range(steps):
            idx=order[step*64:(step+1)*64]
            lr=e12.lr_at((ep-1)*steps+step,30*steps,5*steps,5e-4,1e-5,1e-6)
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True)
            main,aux=model.training_outputs(data["train"]["mlm"][idx],data["train"]["global"][idx],regions[idx],rm[idx])
            loss=F.cross_entropy(main,target[idx])+.2*e9.patient_auxiliary_loss(aux,target[idx]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),3.0);opt.step();losses.append(float(loss))
        val,_,_=eval_head(model,data["val"]); history.append({"epoch":ep,"loss":float(np.mean(losses)),"val":val})
        if val["r2"]>best:best=val["r2"];best_ep=ep;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);val,prob,logits=eval_head(model,data["val"])
    torch.save({"model":{k:v.cpu() for k,v in state.items()},"variant":variant,"seed":seed,"best_epoch":best_ep},out/"checkpoint-best.pth")
    np.save(out/"val_logits.npy",logits);np.save(out/"val_probabilities.npy",prob);dump(out/"history.json",history)
    result={"variant":variant,"pretrain_seed":0 if variant!="e13_a" else None,"downstream_seed":seed,"best_epoch":best_ep,"val":val}
    dump(out/"val_summary.json",result);(out/"VAL_COMPLETE").touch();return result


def train_heads(run_dir: Path, variant: str, device):
    out=run_dir/"downstream"/variant/"aggregate_val.json"
    if out.exists():return
    rows=[train_head_one(run_dir,variant,s,device) for s in (0,1,2)]
    keys=("r2","mse","mae","macro_roc_auc","macro_f1","accuracy")
    agg={k:{"mean":float(np.mean([r["val"][k] for r in rows])),"sd":float(np.std([r["val"][k] for r in rows],ddof=1))} for k in keys}
    agg["per_output_r2"]={n:{"mean":float(np.mean([r["val"]["per_output_r2"][n] for r in rows])),"sd":float(np.std([r["val"]["per_output_r2"][n] for r in rows],ddof=1))} for n in LABEL_NAMES}
    dump(out,{"variant":variant,"rows":rows,"aggregate":agg,"selection_split":"val"})


def reuse_b(run_dir: Path):
    out=run_dir/"downstream/e13_b/aggregate_val.json"
    if out.exists():return
    src=json.loads((E12_RUN/"downstream/validation_selection.json").read_text())["methods"]["e12a_region"]
    rows=[]
    for r in src["per_seed"]:
        q=copy.deepcopy(r);q["variant"]="e13_b";q["source_method"]="e12a_region";rows.append(q)
    keys=("r2","mse","mae","macro_roc_auc","macro_f1","accuracy")
    agg={k:{"mean":float(np.mean([r["val"][k] for r in rows])),"sd":float(np.std([r["val"][k] for r in rows],ddof=1))} for k in keys}
    agg["per_output_r2"]={n:{"mean":float(np.mean([r["val"]["per_output_r2"][n] for r in rows])),"sd":float(np.std([r["val"]["per_output_r2"][n] for r in rows],ddof=1))} for n in LABEL_NAMES}
    dump(out,{"variant":"e13_b","reused":True,"source":str(E12_RUN),"rows":rows,"aggregate":agg})


def select_and_test(run_dir: Path, device):
    final=run_dir/"final_test_summary.json"
    if final.exists():return
    variants=("e13_a","e13_b","e13_c","e13_d","e13_e")
    vals={v:json.loads((run_dir/f"downstream/{v}/aggregate_val.json").read_text()) for v in variants}
    selected=max(variants,key=lambda v:vals[v]["aggregate"]["r2"]["mean"])
    dump(run_dir/"validation_selection.json",{"selected":selected,"criterion":"mean Val Overall R2 across downstream seeds 0/1/2","test_used":False,
                                              "val":{v:vals[v]["aggregate"] for v in variants}})
    if selected=="e13_b":
        hist=json.loads((E12_RUN/"downstream/final_test_summary.json").read_text())
        dump(final,{"selected":selected,"reused_historical_e12_test":True,"source":str(E12_RUN),"aggregate":hist["aggregate"],"rows":hist["rows"]});return
    data=load_arrays(run_dir,selected,device);rows=[]
    for seed in (0,1,2):
        ck=torch.load(run_dir/f"downstream/{selected}/seed{seed}/checkpoint-best.pth",map_location=device,weights_only=False)
        model=e11.E11RegionResidual(seed).to(device);model.load_state_dict(ck["model"],strict=True)
        met,prob,logits=eval_head(model,data["test"]);od=run_dir/f"downstream/{selected}/seed{seed}"
        np.save(od/"test_logits.npy",logits);np.save(od/"test_probabilities.npy",prob);rows.append({"seed":seed,"test":met})
    keys=("r2","mse","mae","macro_roc_auc","macro_f1","accuracy","brier_score","ece15_top_label")
    agg={k:{"mean":float(np.mean([r["test"][k] for r in rows])),"sd":float(np.std([r["test"][k] for r in rows],ddof=1))} for k in keys}
    agg["per_output_r2"]={n:{"mean":float(np.mean([r["test"]["per_output_r2"][n] for r in rows])),"sd":float(np.std([r["test"]["per_output_r2"][n] for r in rows],ddof=1))} for n in LABEL_NAMES}
    dump(final,{"selected":selected,"rows":rows,"aggregate":agg,"test_evaluated_new_methods":[selected]})


def report(run_dir: Path):
    sel=json.loads((run_dir/"validation_selection.json").read_text());fin=json.loads((run_dir/"final_test_summary.json").read_text())
    labels={"e13_a":"A 无MLM随机Transformer","e13_b":"B E12 Masked SID","e13_c":"C Masked z_e重建","e13_d":"D z_e+SID Hybrid","e13_e":"E 原始96点波形"}
    lines=["# E13：连续 ECG 表示与离散 SID 监督消融","","## 协议与审计","",
           "- E3 MorphologyEncoder与Codebook冻结；历史E3/E11/E12只读。",
           "- E13-A冻结随机Transformer，只训练相同E11融合/分类头，因此直接隔离MLM表征预训练的价值。",
           "- C/D/E与E12使用同一87,066/8,941无监督划分、100 epoch、50%有效位置Mask和显式padding mask。",
           "- E13-E在原始96点波形上先Mask再使用官方TokenEmbedding Conv1d，避免跨心拍卷积泄漏。",
           "- 所有选择仅依据三种下游seed的平均Val Overall R²；Test只评估Val预选方案。","",
           "## Validation结果","","| 方法 | Overall R² mean±sd | OMI | NOMI | Control | Healthy | Macro AUC | Macro F1 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v in labels:
        a=sel["val"][v];pr=a["per_output_r2"]
        lines.append(f"| {labels[v]} | {a['r2']['mean']:.6f} ± {a['r2']['sd']:.6f} | "+" | ".join(f"{pr[n]['mean']:.6f}" for n in LABEL_NAMES)+f" | {a['macro_roc_auc']['mean']:.6f} | {a['macro_f1']['mean']:.6f} |")
    t=fin["aggregate"]
    lines += ["",f"Val选中：**{labels[sel['selected']]}**。","","## 锁定Test（仅选中方案）","",
              f"- Overall R²: {t['r2']['mean']:.6f} ± {t['r2']['sd']:.6f}",
              f"- Macro AUC: {t['macro_roc_auc']['mean']:.6f}; Macro F1: {t['macro_f1']['mean']:.6f}",
              "- 四类R²: "+", ".join(f"{n}={t['per_output_r2'][n]['mean']:.6f}" for n in LABEL_NAMES),"",
              "## 解释","",
              "1. A与B的Val差值量化Masked SID预训练相对随机冻结Transformer的收益。",
              "2. B与C比较SID分类监督和连续重建；两种预训练损失数值不可直接横比。",
              "3. D与B比较SID作为输入是否在连续z_e之外仍提供增量。",
              "4. E与B/C比较原始波形和E3形态编码；E采用无泄漏的官方Conv结构。",
              "5. 当前完整筛选只有预训练seed0；三种下游seed不等于三个独立预训练seed。后续应只给Val最有希望的方案补预训练seed1/2。","",
              "## 文件","","- `protocol.json`：锁定协议。","- `pretrain/`：checkpoint、历史与预训练汇总。","- `downstream/`：每seed checkpoint、Val/Test预测。","- `validation_selection.json`与`final_test_summary.json`：选择和锁定Test。",""]
    (run_dir/"README.md").write_text("\n".join(lines),encoding="utf-8");(run_dir/"COMPLETE").touch();set_status(run_dir,"complete")


def protocol(run_dir: Path, epochs: int):
    if (run_dir/"protocol.json").exists():return
    dump(run_dir/"protocol.json",{"experiment":EXP,"e12_source":str(E12_RUN),"pretrain_seed_screen":[0],"downstream_seeds":[0,1,2],
        "variants":{"e13_a":"random continuous Transformer; no MLM; frozen downstream","e13_b":"reuse E12-A masked SID",
                    "e13_c":"continuous z_e masked regression MSE","e13_d":"(z_e projection + SID embedding)/sqrt(2), masked SID",
                    "e13_e":"raw 96 point heartbeat, mask-before official Conv1d, masked SID"},
        "epochs":epochs,"batch":256,"mask_ratio":.5,"peak_lr":5e-4,"min_lr":1e-5,"warmup_epochs":5,"weight_decay":.05,"clip":3,
        "downstream":"frozen sentence Mean+Max + E11 region residual, CE4+0.2 patient CE3, 30 epochs",
        "selection":"best pretrain Val loss; best downstream Val Overall R2; architecture mean Val Overall R2",
        "test_policy":"exactly one Val-selected architecture; E12 test reused read-only if B selected"})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["smoke","audit","prepare_a","cache_wave","pretrain_c","pretrain_d","pretrain_e","features_a","features_c","features_d","features_e","downstream_a","reuse_b","downstream_c","downstream_d","downstream_e","test","report"])
    ap.add_argument("--output_dir",type=Path,default=DEFAULT_RUN);ap.add_argument("--epochs",type=int,default=100)
    args=ap.parse_args();run=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir;device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run.mkdir(parents=True,exist_ok=True);protocol(run,args.epochs)
    if args.stage=="smoke":leakage_smoke(run,device)
    elif args.stage=="audit":e12.overlap_audit(run)
    elif args.stage=="prepare_a":prepare_a(run)
    elif args.stage=="cache_wave":cache_wave(run)
    elif args.stage.startswith("pretrain_"):pretrain(run,"e13_"+args.stage[-1],device,args.epochs)
    elif args.stage.startswith("features_"):extract_features(run,"e13_"+args.stage[-1],device)
    elif args.stage.startswith("downstream_"):train_heads(run,"e13_"+args.stage[-1],device)
    elif args.stage=="reuse_b":reuse_b(run)
    elif args.stage=="test":select_and_test(run,device)
    elif args.stage=="report":report(run)


if __name__=="__main__":main()
