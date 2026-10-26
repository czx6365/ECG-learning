#!/usr/bin/env python3
"""E15-C: reliability-aware selective SID MLM on the E12-A backbone.

The architecture, initialization, data, masking, and downstream protocol match
E12-A. Epochs 1-10 optimize all masked valid tokens. Epochs 11-100 use
0.5 * L_all + 0.5 * L_select, where approximately 80% of positions are selected
without gradients using historical prediction residual stability and frozen
codebook assignment margin. Rare SIDs are always retained and quotas are
balanced by measured lead.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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

import run_e12_continuous_input as e12
import run_e11_region_residual_fusion as e11
import run_e14_lead_time_embedding as e14

RUN_DEFAULT = ROOT / "results/e15_selective_mlm"
E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
E13_RUN = ROOT / "results/e13_continuous_discrete_ablation"
SEQ = 256


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class SelectiveDataset(Dataset):
    def __init__(self, run: Path, split: str):
        self.ze = np.load(E12_RUN / f"pretrain_ze_cache/{split}_ze.npy", mmap_mode="r")
        self.sid = np.load(e12.PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        self.lead = np.load(run / f"selection_meta/{split}_lead.npy", mmap_mode="r")
        self.margin = np.load(run / f"selection_meta/{split}_codebook_margin.npy", mmap_mode="r")
        self.quality = np.load(run / f"selection_meta/{split}_quality.npy", mmap_mode="r")

    def __len__(self):
        return len(self.sid)

    def __getitem__(self, index):
        return (index, np.array(self.ze[index], copy=True), np.array(self.sid[index], copy=True),
                np.array(self.lead[index], copy=True), np.array(self.margin[index], copy=True),
                np.array(self.quality[index], copy=True))


def _source_arrays(split: str):
    codes = np.load(e12.PRETRAIN_INDEX / f"{split}_source_code.npy")
    indices = np.load(e12.PRETRAIN_INDEX / f"{split}_source_index.npy")
    waves = {s: np.load(e12.PRETRAIN_SOURCE / f"{s}_data.npy", mmap_mode="r") for s in ("train", "val")}
    leads = {s: np.load(e12.PRETRAIN_SOURCE / f"{s}_data_in_chans.npy", mmap_mode="r") for s in ("train", "val")}
    return codes, indices, waves, leads


@torch.inference_mode()
def prepare_meta(run: Path, device: torch.device) -> None:
    out = run / "selection_meta"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=False)
    tok = e12.e4.load_model(e12.TOKENIZER, device).eval().requires_grad_(False)
    codebook = tok.quantize.embedding.weight.detach().float().to(device)
    summary = {}
    for split in ("train", "val"):
        ze = np.load(E12_RUN / f"pretrain_ze_cache/{split}_ze.npy", mmap_mode="r")
        sid = np.load(e12.PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        codes, indices, wave_sources, lead_sources = _source_arrays(split)
        lead_out = np.lib.format.open_memmap(out / f"{split}_lead.npy", mode="w+", dtype=np.int16,
                                             shape=sid.shape)
        margin_out = np.lib.format.open_memmap(out / f"{split}_codebook_margin.npy", mode="w+",
                                               dtype=np.float16, shape=sid.shape)
        # ptp, std, flat-difference fraction, absolute peak offset from sample 48.
        quality_out = np.lib.format.open_memmap(out / f"{split}_quality.npy", mode="w+",
                                                dtype=np.float16, shape=sid.shape + (4,))
        valid_count = match = 0
        sid_counts = np.zeros(1024, dtype=np.int64)
        for st in range(0, len(sid), 192):
            en = min(st + 192, len(sid)); c = codes[st:en]; ii = indices[st:en]
            lead = e12.gather_source(lead_sources, c, ii).astype(np.int16)
            wave = e12.gather_source(wave_sources, c, ii).astype(np.float32)
            s = np.asarray(sid[st:en]); valid = (s >= 0) & (s < 1024)
            lead_out[st:en] = lead
            q = np.zeros(s.shape + (4,), dtype=np.float32)
            q[..., 0] = np.ptp(wave, axis=-1)
            q[..., 1] = wave.std(axis=-1)
            q[..., 2] = (np.abs(np.diff(wave, axis=-1)) < 1e-4).mean(axis=-1)
            q[..., 3] = np.abs(np.abs(wave).argmax(axis=-1) - 48)
            q[~valid] = 0; quality_out[st:en] = q.astype(np.float16)
            margins = np.zeros(s.shape, dtype=np.float32)
            if valid.any():
                z = torch.from_numpy(np.asarray(ze[st:en])[valid].copy()).float().to(device)
                z = F.normalize(z, dim=-1)
                labels = torch.from_numpy(s[valid].astype(np.int64)).to(device)
                # Exact frozen-codebook squared-distance rule used by VQ-HBR.
                d = z.square().sum(1, keepdim=True) + codebook.square().sum(1) - 2 * z @ codebook.T
                target_d = d.gather(1, labels[:, None]).squeeze(1)
                top_d, top_i = d.topk(2, dim=1, largest=False)
                other_d = torch.where(top_i[:, 0] == labels, top_d[:, 1], top_d[:, 0])
                margins[valid] = (other_d - target_d).cpu().numpy()
                match += int((top_i[:, 0] == labels).sum())
                valid_count += int(labels.numel())
                sid_counts += np.bincount(labels.cpu().numpy(), minlength=1024)
            margin_out[st:en] = margins.astype(np.float16)
            if st == 0 or en == len(sid) or (st // 192) % 50 == 0:
                print(f"meta {split} {en}/{len(sid)}", flush=True)
        lead_out.flush(); margin_out.flush(); quality_out.flush()
        active = np.flatnonzero(sid_counts)
        positive = sid_counts[active]
        rare_cut = int(np.quantile(positive, .25, method="higher")) if len(positive) else 0
        rare = active[positive <= rare_cut]
        np.save(out / f"{split}_sid_counts.npy", sid_counts)
        summary[split] = {"records": int(len(sid)), "valid_tokens": valid_count,
                          "nearest_code_matches_sid": match / max(1, valid_count),
                          "active_sids": int(len(active)), "rare_sid_count": int(len(rare)),
                          "rare_sid_frequency_cutoff": rare_cut,
                          "observed_leads": sorted(int(x) for x in np.unique(lead[valid]) if x > 0)}
    train_counts = np.load(out / "train_sid_counts.npy")
    active = np.flatnonzero(train_counts); cut = int(np.quantile(train_counts[active], .25, method="higher"))
    rare_mask = np.zeros(1024, dtype=np.bool_); rare_mask[active[train_counts[active] <= cut]] = True
    np.save(out / "rare_sid_mask.npy", rare_mask)
    dump(out / "audit.json", {"splits": summary, "quality_columns": ["wave_ptp", "wave_std", "flat_diff_fraction", "abs_peak_offset_from_48"],
                              "selection_inputs": ["historical_true_SID_residual_EMA", "historical_residual_change_EMA", "frozen_codebook_target_margin"],
                              "clinical_labels_used": False, "rare_sid_policy": "bottom frequency quartile always retained"})
    (out / "COMPLETE").touch()


def _zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)


@torch.no_grad()
def select_positions(labels: torch.Tensor, leads: torch.Tensor, margin: torch.Tensor,
                     sample_rows: torch.Tensor, hist_residual: torch.Tensor,
                     hist_change: torch.Tensor, rare_sid: torch.Tensor,
                     keep_ratio: float = .8) -> tuple[torch.Tensor, torch.Tensor]:
    """Select by detached history+codebook reliability, with lead and rare-SID guards."""
    historical_bad = hist_residual + hist_change
    score = _zscore(historical_bad.float()) + _zscore((-margin).float())
    kept = rare_sid[labels].clone()
    for lead_id in torch.unique(leads):
        group = torch.nonzero(leads == lead_id, as_tuple=False).flatten()
        target = max(1, int(math.ceil(float(keep_ratio) * group.numel())))
        already = int(kept[group].sum())
        if already < target:
            candidates = group[~kept[group]]
            choose = candidates[torch.topk(score[candidates], k=min(target - already, candidates.numel()), largest=False).indices]
            kept[choose] = True
    # Safety: every ECG with masked valid tokens contributes to L_select.
    for row in torch.unique(sample_rows):
        group = torch.nonzero(sample_rows == row, as_tuple=False).flatten()
        if not kept[group].any():
            kept[group[score[group].argmin()]] = True
    return kept, score


class History:
    def __init__(self, records: int):
        self.mean = np.zeros(records * SEQ, np.float32)
        self.change = np.zeros(records * SEQ, np.float32)
        self.count = np.zeros(records * SEQ, np.uint8)

    def get(self, flat: np.ndarray, device: torch.device):
        return (torch.from_numpy(self.mean[flat].copy()).to(device),
                torch.from_numpy(self.change[flat].copy()).to(device))

    def update(self, flat: np.ndarray, residual: np.ndarray):
        old = self.mean[flat].copy(); seen = self.count[flat] > 0
        self.mean[flat] = np.where(seen, .9 * old + .1 * residual, residual)
        delta = np.abs(residual - old)
        self.change[flat] = np.where(seen, .9 * self.change[flat] + .1 * delta, 0.)
        self.count[flat] = np.minimum(self.count[flat].astype(np.int16) + 1, 255).astype(np.uint8)


def _accum_group(total: np.ndarray, kept: np.ndarray, ids: np.ndarray, keep: np.ndarray):
    total += np.bincount(ids, minlength=len(total))
    kept += np.bincount(ids[keep], minlength=len(kept))


def train_epoch(model, loader, device, optimizer, scaler, epoch, epochs, history: History,
                rare_sid: torch.Tensor, train: bool) -> tuple[dict, dict]:
    model.train(train)
    scalar = {k: 0. for k in ("loss", "loss_all", "loss_select", "ce_kept", "ce_filtered", "mlm_acc", "keep_ratio", "masked_tokens", "batches")}
    sid_total = np.zeros(1024, np.int64); sid_kept = np.zeros(1024, np.int64)
    lead_total = np.zeros(16, np.int64); lead_kept = np.zeros(16, np.int64)
    q_keep = np.zeros(4, np.float64); q_drop = np.zeros(4, np.float64); nk = nd = 0
    margin_keep = margin_drop = hist_keep = hist_drop = 0.
    for step, (indices, ze, sid, lead, margin, quality) in enumerate(loader):
        ze = ze.float().to(device, non_blocking=True); sid = sid.long().to(device, non_blocking=True)
        lead = lead.long().to(device, non_blocking=True); margin = margin.float().to(device, non_blocking=True)
        valid = (sid >= 0) & (sid < 1024); mask = e12.random_mask(valid, .5); coords = mask.nonzero(as_tuple=False)
        labels = sid[mask]; lead_m = lead[mask]; margin_m = margin[mask]
        coords_cpu = coords.cpu()
        flat = (indices[coords_cpu[:, 0]].numpy().astype(np.int64) * SEQ + coords_cpu[:, 1].numpy())
        hmean, hchange = history.get(flat, device)
        if train:
            gs = (epoch - 1) * len(loader) + step
            lr = e12.lr_at(gs, epochs * len(loader), 5 * len(loader), 5e-4, 1e-5, 1e-6)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(ze, valid, mask)
            token_ce = F.cross_entropy(logits.float(), labels, reduction="none")
            loss_all = token_ce.mean()
            if train and epoch > 10:
                keep, _ = select_positions(labels, lead_m, margin_m, coords[:, 0], hmean, hchange, rare_sid)
                loss_select = token_ce[keep].mean()
                loss = .5 * loss_all + .5 * loss_select
            else:
                keep = torch.ones_like(labels, dtype=torch.bool)
                loss_select = loss_all
                loss = loss_all
        if train:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.); scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            probs = torch.softmax(logits.float(), -1); ptrue = probs.gather(1, labels[:, None]).squeeze(1)
            residual = (1 - ptrue).cpu().numpy();
            if train: history.update(flat, residual)
            drop = ~keep; ce = token_ce.detach()
            scalar["loss"] += float(loss); scalar["loss_all"] += float(loss_all); scalar["loss_select"] += float(loss_select)
            scalar["ce_kept"] += float(ce[keep].mean()); scalar["ce_filtered"] += float(ce[drop].mean()) if drop.any() else 0.
            scalar["mlm_acc"] += float((logits.argmax(-1) == labels).float().mean()); scalar["keep_ratio"] += float(keep.float().mean())
            scalar["masked_tokens"] += labels.numel(); scalar["batches"] += 1
            lab_np = labels.cpu().numpy(); lead_np = lead_m.cpu().numpy(); keep_np = keep.cpu().numpy()
            _accum_group(sid_total, sid_kept, lab_np, keep_np); _accum_group(lead_total, lead_kept, lead_np, keep_np)
            q = quality[mask.cpu()].numpy().astype(np.float64); q_keep += q[keep_np].sum(0); nk += int(keep_np.sum())
            if (~keep_np).any(): q_drop += q[~keep_np].sum(0); nd += int((~keep_np).sum())
            m_np = margin_m.cpu().numpy(); h_np = (hmean + hchange).cpu().numpy()
            margin_keep += float(m_np[keep_np].sum()); hist_keep += float(h_np[keep_np].sum())
            if (~keep_np).any(): margin_drop += float(m_np[~keep_np].sum()); hist_drop += float(h_np[~keep_np].sum())
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            print(f"{'train' if train else 'val'} epoch={epoch} {step+1}/{len(loader)} loss={float(loss):.6f} keep={float(keep.float().mean()):.4f}", flush=True)
    n = max(1, int(scalar["batches"])); stats = {k: (v / n if k not in ("batches",) else v) for k, v in scalar.items()}
    detail = {"sid_total": sid_total.tolist(), "sid_kept": sid_kept.tolist(), "lead_total": lead_total.tolist(), "lead_kept": lead_kept.tolist(),
              "kept_tokens": nk, "filtered_tokens": nd,
              "quality_kept_mean": (q_keep / max(1, nk)).tolist(), "quality_filtered_mean": (q_drop / max(1, nd)).tolist(),
              "codebook_margin_kept_mean": margin_keep / max(1, nk), "codebook_margin_filtered_mean": margin_drop / max(1, nd),
              "historical_badness_kept_mean": hist_keep / max(1, nk), "historical_badness_filtered_mean": hist_drop / max(1, nd)}
    return stats, detail


def pretrain(run: Path, device: torch.device, epochs: int) -> None:
    out = run / "mlm/selective"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    model, init = e12.initialize_continuous("scratch", device); dump(out / "initialization_audit.json", init)
    tr_ds = SelectiveDataset(run, "train"); va_ds = SelectiveDataset(run, "val")
    gen = torch.Generator().manual_seed(0)
    tr = DataLoader(tr_ds, batch_size=256, shuffle=True, generator=gen, num_workers=8, pin_memory=True, drop_last=True)
    va = DataLoader(va_ds, batch_size=384, shuffle=False, num_workers=8, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    hist = History(len(tr_ds)); rare = torch.from_numpy(np.load(run / "selection_meta/rare_sid_mask.npy")).bool().to(device)
    best = float("inf"); best_ep = None; rows = []; start = time.time()
    stat_file = out / "selection_stats.jsonl"
    for ep in range(1, epochs + 1):
        train_stats, detail = train_epoch(model, tr, device, opt, scaler, ep, epochs, hist, rare, True)
        row = {"epoch": ep, **{f"train_{k}": v for k, v in train_stats.items()}}
        with stat_file.open("a", encoding="utf-8") as f: f.write(json.dumps({"epoch": ep, **detail}) + "\n")
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            with torch.no_grad(): val_stats, _ = train_epoch(model, va, device, None, None, ep, epochs, History(len(va_ds)), rare, False)
            row.update({f"val_{k}": v for k, v in val_stats.items()})
            if val_stats["loss_all"] < best:
                best = val_stats["loss_all"]; best_ep = ep
                torch.save({"model": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "config": e12.model_config(), "epoch": ep, "best_val_loss": best}, out / "checkpoint-best.pth")
        rows.append(row); dump(out / "training_history.json", rows); print(json.dumps(row), flush=True)
        if ep in (10, 50, epochs):
            torch.save({"model": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "config": e12.model_config(), "epoch": ep}, out / f"checkpoint-{ep}.pth")
    dump(out / "summary.json", {"best_epoch": best_ep, "best_val_loss": best, "epochs": epochs, "pretrain_seed": 0,
                                 "elapsed_seconds": time.time()-start, "trainable_parameters": sum(p.numel() for p in model.parameters()),
                                 "loss": "epochs1-10 L_all; epochs11-100 0.5*L_all+0.5*L_select", "target_keep_ratio": .8})
    (out / "COMPLETE").touch()


@torch.inference_mode()
def fixed_mlm(run: Path, device: torch.device) -> None:
    out = run / "fixed_mask_mlm_comparison.json"
    if out.exists(): return
    base = e12.load_continuous_checkpoint(E12_RUN / "mlm/scratch/checkpoint-best.pth", device)
    new = e12.load_continuous_checkpoint(run / "mlm/selective/checkpoint-best.pth", device)
    sid_np = np.load(e12.PRETRAIN_SID / "val_sid.npy", mmap_mode="r")
    ze_np = np.load(E12_RUN / "pretrain_ze_cache/val_ze.npy", mmap_mode="r")
    sums = {k:{"loss":0.,"correct":0,"tokens":0} for k in ("e12_a","e15_c")}
    for st in range(0, len(sid_np), 128):
        en=min(st+128,len(sid_np)); s0=np.asarray(sid_np[st:en]); mn=e12.fixed_masks(s0,st)
        sid=torch.from_numpy(s0.copy()).long().to(device); mask=torch.from_numpy(mn).to(device); valid=(sid>=0)&(sid<1024); labels=sid[mask]
        ze=torch.from_numpy(np.asarray(ze_np[st:en]).copy()).float().to(device)
        for key,logits in {"e12_a":base(ze,valid,mask),"e15_c":new(ze,valid,mask)}.items():
            sums[key]["loss"] += float(F.cross_entropy(logits.float(),labels,reduction="sum")); sums[key]["correct"] += int((logits.argmax(-1)==labels).sum()); sums[key]["tokens"] += labels.numel()
    dump(out,{k:{"val_ce":v["loss"]/v["tokens"],"masked_sid_accuracy":v["correct"]/v["tokens"],"masked_tokens":v["tokens"]} for k,v in sums.items()} | {"mask_policy":"identical fixed 50% valid-token mask; validation uses all masked positions"})


def features(run: Path, device: torch.device) -> None:
    e12.extract_downstream(run, "selective", device, 32)


def train_one(run: Path, seed: int, device: torch.device) -> dict:
    out=run/f"downstream/e15c/seed{seed}"
    if (out/"VAL_COMPLETE").exists(): return json.loads((out/"val_summary.json").read_text())
    out.mkdir(parents=True,exist_ok=False); data=e12.load_downstream_arrays(run,"selective",device)
    model=e11.E11RegionResidual(seed).to(device); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
    target=data["train"]["labels"].argmax(1); steps=len(target)//64; gen=torch.Generator().manual_seed(seed)
    best=-float("inf"); best_ep=None; state=None; history=[]; reg,regmask=e11.masked_region_means(data["train"]["lead"],data["train"]["lead_mask"])
    for ep in range(1,31):
        model.train(); order=torch.randperm(len(target),generator=gen).to(device); lm=[]; la=[]
        for step in range(steps):
            idx=order[step*64:(step+1)*64]; lr=e12.lr_at((ep-1)*steps+step,30*steps,5*steps,5e-4,1e-5,1e-6)
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True); main,aux=model.training_outputs(data["train"]["mlm"][idx],data["train"]["global"][idx],reg[idx],regmask[idx])
            l1=F.cross_entropy(main,target[idx]);l2=e12.e9.patient_auxiliary_loss(aux,target[idx]);loss=l1+.2*l2;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();lm.append(float(l1));la.append(float(l2))
        val,_,_=e12.eval_downstream(model,"e15c",data["val"]);history.append({"epoch":ep,"main_ce":float(np.mean(lm)),"patient_ce":float(np.mean(la)),"val":val})
        if val["r2"]>best:best=val["r2"];best_ep=ep;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);val,prob,logits=e12.eval_downstream(model,"e15c",data["val"])
    torch.save({"model":{k:v.cpu() for k,v in state.items()},"seed":seed,"best_epoch":best_ep},out/"checkpoint-best.pth");np.save(out/"val_logits.npy",logits);np.save(out/"val_probabilities.npy",prob);dump(out/"history.json",history)
    res={"seed":seed,"pretrain_seed":0,"best_epoch":best_ep,"val":val};dump(out/"val_summary.json",res);(out/"VAL_COMPLETE").touch();return res


def downstream(run: Path, device: torch.device) -> None:
    out=run/"validation_summary.json"
    if out.exists():return
    rows=[train_one(run,s,device) for s in (0,1,2)];base=json.loads((E13_RUN/"validation_selection.json").read_text())["val"]["e13_b"]
    def agg(k):
        x=[r["val"][k] for r in rows];return {"mean":float(np.mean(x)),"sd":float(np.std(x,ddof=1))}
    new={k:agg(k) for k in ("r2","mse","mae","macro_roc_auc","macro_f1","accuracy")};new["per_output_r2"]={n:{"mean":float(np.mean([r["val"]["per_output_r2"][n] for r in rows])),"sd":float(np.std([r["val"]["per_output_r2"][n] for r in rows],ddof=1))} for n in e12.LABEL_NAMES}
    dump(out,{"selection_split":"validation","test_run":False,"e12_a":base,"e15_c":new,"delta":{"r2":new["r2"]["mean"]-base["r2"]["mean"],**{n:new["per_output_r2"][n]["mean"]-base["per_output_r2"][n]["mean"] for n in e12.LABEL_NAMES}},"per_seed":rows})


def aggregate_selection(run: Path) -> dict:
    rows=[json.loads(x) for x in (run/"mlm/selective/selection_stats.jsonl").read_text().splitlines() if x.strip() and json.loads(x)["epoch"]>10]
    sid_t=np.sum([np.asarray(r["sid_total"],np.int64) for r in rows],0);sid_k=np.sum([np.asarray(r["sid_kept"],np.int64) for r in rows],0)
    lead_t=np.sum([np.asarray(r["lead_total"],np.int64) for r in rows],0);lead_k=np.sum([np.asarray(r["lead_kept"],np.int64) for r in rows],0)
    nk=sum(r["kept_tokens"] for r in rows);nd=sum(r["filtered_tokens"] for r in rows)
    def weighted(key, nkey):
        den=sum(r[nkey] for r in rows);return sum(np.asarray(r[key])*r[nkey] for r in rows).tolist() if isinstance(rows[0][key],list) else sum(r[key]*r[nkey] for r in rows)/max(1,den)
    qk=sum(np.asarray(r["quality_kept_mean"])*r["kept_tokens"] for r in rows)/max(1,nk);qd=sum(np.asarray(r["quality_filtered_mean"])*r["filtered_tokens"] for r in rows)/max(1,nd)
    active=np.flatnonzero(sid_t);ret=np.divide(sid_k,sid_t,out=np.zeros_like(sid_k,dtype=float),where=sid_t>0)
    lowest=active[np.argsort(ret[active])[:20]];highest=active[np.argsort(ret[active])[-20:][::-1]]
    return {"epochs":len(rows),"kept_tokens":int(nk),"filtered_tokens":int(nd),"actual_keep_ratio":nk/max(1,nk+nd),
            "quality_columns":["wave_ptp","wave_std","flat_diff_fraction","abs_peak_offset_from_48"],"quality_kept_mean":qk.tolist(),"quality_filtered_mean":qd.tolist(),
            "codebook_margin_kept_mean":sum(r["codebook_margin_kept_mean"]*r["kept_tokens"] for r in rows)/max(1,nk),
            "codebook_margin_filtered_mean":sum(r["codebook_margin_filtered_mean"]*r["filtered_tokens"] for r in rows)/max(1,nd),
            "historical_badness_kept_mean":sum(r["historical_badness_kept_mean"]*r["kept_tokens"] for r in rows)/max(1,nk),
            "historical_badness_filtered_mean":sum(r["historical_badness_filtered_mean"]*r["filtered_tokens"] for r in rows)/max(1,nd),
            "lead_retention":{str(i):float(lead_k[i]/lead_t[i]) for i in np.flatnonzero(lead_t)},
            "lowest_retention_sids":{str(int(i)):float(ret[i]) for i in lowest},"highest_retention_sids":{str(int(i)):float(ret[i]) for i in highest},
            "rare_sid_guard":"bottom train-frequency quartile retained at 100% whenever masked"}


def report(run: Path) -> None:
    s=json.loads((run/"validation_summary.json").read_text());m=json.loads((run/"fixed_mask_mlm_comparison.json").read_text());sel=aggregate_selection(run);dump(run/"selection_summary.json",sel)
    a=s["e12_a"];b=s["e15_c"]
    metrics=[("Overall R²",a["r2"]["mean"],b["r2"]["mean"])]+[(n+" R²",a["per_output_r2"][n]["mean"],b["per_output_r2"][n]["mean"]) for n in e12.LABEL_NAMES]+[("Macro AUC",a["macro_roc_auc"]["mean"],b["macro_roc_auc"]["mean"]),("Macro F1",a["macro_f1"]["mean"],b["macro_f1"]["mean"])]
    with (run/"metrics_summary.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["metric","E12-A","E15-C","delta"]);w.writerows([(n,x,y,y-x) for n,x,y in metrics])
    qn=sel["quality_columns"];qk=sel["quality_kept_mean"];qd=sel["quality_filtered_mean"]
    lines=["# E15-C：Selective Learning 优化 ECG MLM 预训练","","## 方法审计","","- 模型、初始化、数据、50%有效心拍Mask、优化器和训练预算与E12-A一致。","- Epoch 1–10使用全部Mask位置；Epoch 11–100使用`0.5×L_all + 0.5×L_select`。","- 筛选依据为跨epoch真实SID预测残差稳定性与冻结Codebook目标原型间隔；不使用下游临床标签，也不直接删除当前CE最高的20%。","- 按实测导联维持约80%配额；低频四分位SID强制保留；每个样本至少保留一个位置。","- 这是对Selective Learning思想的SID-MLM改造，不等同于论文的滑窗预测残差熵与残差下界双Mask。","- Validation不做筛选；未运行Test。","","## 固定Mask MLM","","|模型|Val CE|Masked SID Accuracy|","|---|---:|---:|",f"|E12-A|{m['e12_a']['val_ce']:.6f}|{m['e12_a']['masked_sid_accuracy']:.6f}|",f"|E15-C|{m['e15_c']['val_ce']:.6f}|{m['e15_c']['masked_sid_accuracy']:.6f}|","","## Validation（三下游seed均值）","","|指标|E12-A|E15-C|差值|","|---|---:|---:|---:|"]
    lines += [f"|{n}|{x:.6f}|{y:.6f}|{y-x:+.6f}|" for n,x,y in metrics]
    lines += ["","## 筛选出的20%位置特征","",f"- 实际保留比例：{sel['actual_keep_ratio']:.6f}。",f"- Codebook目标间隔：保留{sel['codebook_margin_kept_mean']:.6f}，过滤{sel['codebook_margin_filtered_mean']:.6f}（越低越模糊）。",f"- 历史预测不稳定度：保留{sel['historical_badness_kept_mean']:.6f}，过滤{sel['historical_badness_filtered_mean']:.6f}（越高越不稳定）。"]
    lines += [f"- {qn[i]}：保留{qk[i]:.6f}，过滤{qd[i]:.6f}。" for i in range(4)]
    lines += ["- 各导联保留率："+", ".join(f"{e14.LEAD_MAP.get(int(k),k)}={v:.4f}" for k,v in sel["lead_retention"].items()),"- 低频SID由强制保留规则保护，因此不会被长期排除。","","## 结论","",f"Val Overall R²：{a['r2']['mean']:.6f} → {b['r2']['mean']:.6f}（{b['r2']['mean']-a['r2']['mean']:+.6f}）。",f"OMI：{s['delta']['OMI']:+.6f}；NOMI：{s['delta']['NOMI']:+.6f}。"]
    (run/"README.md").write_text("\n".join(lines),encoding="utf-8");(run/"COMPLETE").touch();(run/"status.txt").write_text("complete")


def smoke(run: Path, device: torch.device) -> None:
    out=run/"smoke";out.mkdir(parents=True,exist_ok=False);model,_=e12.initialize_continuous("scratch",device);model.eval()
    ze=torch.randn(3,256,64,device=device);sid=torch.full((3,256),1024,dtype=torch.long,device=device);sid[:,:20]=torch.randint(0,1024,(3,20),device=device);valid=sid<1024;mask=torch.zeros_like(valid);mask[:,[2,5,9,13,17]]=True
    with torch.no_grad():a=model(ze,valid,mask);z2=ze.clone();z2[mask]=torch.randn_like(z2[mask])*1e6;b=model(z2,valid,mask)
    delta=float((a-b).abs().max());labels=sid[mask];leads=torch.tensor([1,2,3,7,8]*3,device=device);margin=torch.randn(len(labels),device=device);rows=mask.nonzero()[:,0];rare=torch.zeros(1024,dtype=torch.bool,device=device);rare[labels[0]]=True
    keep,score=select_positions(labels,leads,margin,rows,torch.rand(len(labels),device=device),torch.rand(len(labels),device=device),rare)
    if delta!=0 or not keep[0] or any(not keep[rows==r].any() for r in rows.unique()):raise RuntimeError("smoke failed")
    loss=F.cross_entropy(model(ze,valid,mask),labels);loss.backward();dump(out/"audit.json",{"passed":True,"masked_z_e_mutation_delta":delta,"selection_detached":not score.requires_grad,"rare_sid_retained":bool(keep[0]),"every_sample_has_selected":True,"finite_backward":bool(torch.isfinite(loss))});(out/"COMPLETE").touch()


def audit(run: Path) -> None:
    e12.overlap_audit(run)
    dump(run/"method_audit.json",{"paper":"Selective Learning for Deep Time Series Forecasting, NeurIPS 2025, arXiv:2510.25207","official_code":"https://github.com/GestaltCogTeam/selective-learning","existing_project_implementation":"heartlang/engines/engine_for_pretraining.py uses current CE + single-pass class entropy and pure selected loss","e15_adaptation":"history-based SID residual stability + frozen-codebook target margin; 50/50 all/selected loss; lead quota; rare-SID guard","not_equivalent_to_paper":True,"reason":"SID MLM lacks repeated sliding-window forecasts and continuous residual lower-bound target used by the forecasting paper","test_policy":"validation only"})


def protocol(run: Path) -> None:
    if (run/"protocol.json").exists():return
    dump(run/"protocol.json",{"experiment":"E15-C","baseline":"read-only E12-A/E13-B","architecture":"unchanged E12-A","pretrain_records":[87066,8941],"epochs":100,"batch":256,"pretrain_seed":0,"downstream_seeds":[0,1,2],"mask_ratio":.5,"warmup_full_loss_epochs":10,"select_keep_ratio":.8,"loss_after_warmup":"0.5 L_all + 0.5 L_select","selection":"historical SID residual stability + frozen codebook margin, lead quota, rare SID guard","validation":"all fixed masked valid positions","test_policy":"no new test"})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["audit","smoke","meta","pretrain","fixed_mlm","features","downstream","report"]);ap.add_argument("--output_dir",type=Path,default=RUN_DEFAULT);ap.add_argument("--epochs",type=int,default=100);args=ap.parse_args();run=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir;run.mkdir(parents=True,exist_ok=True);protocol(run);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage=="audit":audit(run)
    elif args.stage=="smoke":smoke(run,device)
    elif args.stage=="meta":prepare_meta(run,device)
    elif args.stage=="pretrain":pretrain(run,device,args.epochs)
    elif args.stage=="fixed_mlm":fixed_mlm(run,device)
    elif args.stage=="features":features(run,device)
    elif args.stage=="downstream":downstream(run,device)
    elif args.stage=="report":report(run)


if __name__=="__main__":main()
