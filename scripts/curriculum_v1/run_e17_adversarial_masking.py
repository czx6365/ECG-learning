#!/usr/bin/env python3
"""E17-B: heartbeat-token adversarial masking on the frozen-E3 E12-A path.

This is an adaptation of Bo et al. (2022), not a reproduction.  Their mask
network acts on raw ECGs for contrastive learning; here a small MLP scores
frozen E3 z_e tokens and is alternately optimized against SID MLM loss.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

RUN_DEFAULT = ROOT / "results/e17_adversarial_masking"
E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
E13_RUN = ROOT / "results/e13_continuous_discrete_ablation"
SEQ = 256


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


class AdvDataset(Dataset):
    def __init__(self, run: Path, split: str):
        self.ze = np.load(E12_RUN / f"pretrain_ze_cache/{split}_ze.npy", mmap_mode="r")
        self.sid = np.load(e12.PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        self.lead = np.load(run / f"mask_meta/{split}_lead.npy", mmap_mode="r")
        if self.ze.shape[:2] != self.sid.shape or self.lead.shape != self.sid.shape:
            raise RuntimeError(f"{split}: z_e/SID/lead shape mismatch")

    def __len__(self): return len(self.sid)

    def __getitem__(self, i):
        return (i, np.array(self.ze[i], copy=True), np.array(self.sid[i], copy=True),
                np.array(self.lead[i], copy=True))


class MaskGenerator(nn.Module):
    """Small per-heartbeat scorer; no clinical label or SID enters this module."""
    def __init__(self, dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, ze: torch.Tensor) -> torch.Tensor:
        # Bounded scores prevent the scale explosion observed in the archived
        # straight-through pilot while preserving the complete ranking range.
        return 5.0 * torch.tanh(self.net(ze.float()).squeeze(-1) / 5.0)


def adv_fraction(epoch: int) -> float:
    if epoch <= 10: return 0.0
    if epoch <= 20: return 0.30 * float(epoch - 11) / 9.0
    return 0.30


def _quota(counts: torch.Tensor, total: int) -> list[int]:
    """Hamilton allocation with exact total and group-capacity constraints."""
    if total <= 0: return [0] * len(counts)
    c = counts.detach().cpu().numpy().astype(np.int64)
    raw = total * c / max(1, c.sum())
    q = np.minimum(np.floor(raw).astype(np.int64), c)
    order = np.argsort(-(raw - np.floor(raw)))
    left = total - int(q.sum())
    while left:
        changed = False
        for j in order:
            if q[j] < c[j]:
                q[j] += 1; left -= 1; changed = True
                if left == 0: break
        if not changed: raise RuntimeError("Cannot allocate adversarial quota")
    return q.tolist()


def _soft_exact_k(scores: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    """Sigmoid threshold relaxation with detached bisection threshold."""
    if k <= 0: return torch.zeros_like(scores)
    if k >= scores.numel(): return torch.ones_like(scores)
    with torch.no_grad():
        lo = scores.min() - 20 * tau
        hi = scores.max() + 20 * tau
        for _ in range(32):
            mid = (lo + hi) / 2
            if torch.sigmoid((scores - mid) / tau).sum() > k: lo = mid
            else: hi = mid
        threshold = (lo + hi) / 2
    return torch.sigmoid((scores - threshold) / tau)


def _spread_topk(group: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    """Prefer high scores while avoiding adjacent adversarial beats within a lead."""
    order = torch.argsort(scores, descending=True)
    chosen: list[int] = []
    for j in order.tolist():
        pos = int(group[j])
        if all(abs(pos - int(group[x])) > 1 for x in chosen):
            chosen.append(j)
            if len(chosen) == k: break
    if len(chosen) < k:
        for j in order.tolist():
            if j not in chosen:
                chosen.append(j)
                if len(chosen) == k: break
    return torch.tensor(chosen, device=group.device, dtype=torch.long)


def build_hybrid_mask(valid: torch.Tensor, lead: torch.Tensor, scores: torch.Tensor,
                      fraction: float, tau: float = .5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Exact 50% hard mask plus an ST relaxed gate, balanced by measured lead."""
    hard = torch.zeros_like(valid); random_part = torch.zeros_like(valid); adv_part = torch.zeros_like(valid)
    gate = torch.zeros_like(scores, dtype=torch.float32)
    target_total = target_adv = 0
    for row in range(len(valid)):
        idx = valid[row].nonzero(as_tuple=False).flatten()
        if not idx.numel(): continue
        k = max(1, int(round(.5 * idx.numel())))
        ka = min(k, int(round(fraction * k))); kr = k - ka
        perm = torch.randperm(idx.numel(), device=idx.device)
        ridx = idx[perm[:kr]]
        random_part[row, ridx] = True; hard[row, ridx] = True; gate[row, ridx] = 1.
        remain = idx[~torch.isin(idx, ridx)] if kr else idx
        if ka:
            lead_ids = torch.unique(lead[row, remain], sorted=True)
            groups = [remain[lead[row, remain] == x] for x in lead_ids]
            quotas = _quota(torch.tensor([g.numel() for g in groups]), ka)
            for group, q in zip(groups, quotas):
                if not q: continue
                sg = scores[row, group]
                local = _spread_topk(group, sg, q)
                chosen = group[local]
                h = torch.zeros_like(sg); h[local] = 1.
                soft = _soft_exact_k(sg, q, tau)
                st = h + soft - soft.detach()
                gate[row, group] = st
                adv_part[row, chosen] = True; hard[row, chosen] = True
        target_total += k; target_adv += ka
    meta = {"target_total": target_total, "target_adv": target_adv,
            "actual_total": int(hard.sum()), "actual_adv": int(adv_part.sum())}
    return hard, random_part, adv_part, gate, meta


def gated_logits(model: e12.ContinuousSIDMLM, ze: torch.Tensor, valid: torch.Tensor,
                 gate: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:
    """E12 forward with a straight-through float mask gate for generator updates."""
    b = model.backbone
    x = b.input_proj(ze.float()) * valid.unsqueeze(-1).to(ze.dtype)
    g = (gate * valid).unsqueeze(-1).to(x.dtype)
    x = x * (1 - g) + b.mask_token.expand_as(x) * g
    cls = b.cls_token.view(1, 1, -1).expand(len(x), 1, -1)
    x = torch.cat([cls, x], 1)
    full_valid = torch.cat([torch.ones(len(x), 1, dtype=torch.bool, device=x.device), valid], 1)
    x = b.dropout(x + b.pos_embed)
    x = e12.masked_transformer_forward(b.transformer, x, full_valid)
    x = b.norm_layer(x) * full_valid.unsqueeze(-1).to(x.dtype)
    return model.lm_head(x[:, 1:][hard_mask & valid])


def prepare_meta(run: Path) -> None:
    out = run / "mask_meta"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False); summary = {}
    for split in ("train", "val"):
        lead, _ = e14.source_meta(split)
        sid = np.load(e12.PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        valid = (sid >= 0) & (sid < 1024)
        if lead.shape != sid.shape or np.any(valid & (lead <= 0)):
            raise RuntimeError(f"{split}: lead metadata invalid")
        np.save(out / f"{split}_lead.npy", lead)
        ids, counts = np.unique(lead[valid], return_counts=True)
        summary[split] = {"records": len(sid), "valid_tokens": int(valid.sum()),
                          "lead_counts": {str(int(k)): int(v) for k, v in zip(ids, counts)}}
    dump(out / "audit.json", {"splits": summary, "generator_inputs": ["frozen E3 z_e"],
                               "clinical_labels_used": False, "sid_used_by_generator": False,
                               "lead_use": "hard proportional quota only; lead is not an MLP input"})
    (out / "COMPLETE").touch()


def _grad_norm(parameters) -> float:
    vals = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(vals).sum().sqrt()) if vals else 0.


@torch.no_grad()
def _probe_difficulty(model, ze, sid, valid):
    """Measure token difficulty only where a random 50% probe truly masks z_e."""
    probe = e12.random_mask(valid, .5)
    logits = model(ze, valid, probe)
    ce = F.cross_entropy(logits.float(), sid[probe], reduction="none")
    return probe, ce


def generator_objective(model, masker, ze, sid, valid, lead, tau: float = .5):
    """Maximize relaxed expected CE measured under an independent random mask.

    Hard top-k is never claimed to carry gradients.  The current Transformer
    first supplies detached, leakage-free CE rewards at randomly masked tokens.
    Within each record and measured lead, an exact-k sigmoid relaxation then
    assigns differentiable generator mass to those rewards.
    """
    probe, token_ce = _probe_difficulty(model, ze, sid, valid)
    coords = probe.nonzero(as_tuple=False); scores = masker(ze.detach())
    objectives = []
    for row in torch.unique(coords[:, 0]):
        ridx = torch.nonzero(coords[:, 0] == row, as_tuple=False).flatten()
        positions = coords[ridx, 1]
        for lid in torch.unique(lead[row, positions]):
            pick = ridx[lead[row, positions] == lid]
            if not pick.numel(): continue
            pos = coords[pick, 1]; k = max(1, int(round(.30 * pos.numel())))
            weight = _soft_exact_k(scores[row, pos], k, tau)
            reward = token_ce[pick].detach()
            objectives.append((weight * reward).sum() / weight.sum().clamp_min(1e-6))
    if not objectives: raise RuntimeError("No valid generator reward groups")
    expected_difficulty = torch.stack(objectives).mean()
    regularizer = 1e-4 * scores[valid].square().mean()
    return -expected_difficulty + regularizer, float(token_ce.mean()), probe


def _js(count_a: np.ndarray, count_b: np.ndarray) -> float:
    a = count_a.astype(float); b = count_b.astype(float)
    if not a.sum() or not b.sum(): return float("nan")
    a /= a.sum(); b /= b.sum(); m = .5 * (a + b)
    def kl(x):
        z = x > 0
        return float(np.sum(x[z] * np.log(x[z] / m[z])))
    return .5 * (kl(a) + kl(b))


def epoch_loop(model, masker, loader, device, model_opt, gen_opt, model_scaler,
               epoch: int, epochs: int, train: bool) -> tuple[dict, dict]:
    model.train(train); masker.train(train)
    frac = adv_fraction(epoch) if train else 0.
    scalar = {k: 0. for k in ("loss", "mlm_acc", "masked_tokens", "total_mask_ratio", "adv_of_mask",
                               "random_ce", "adv_ce", "generator_loss", "generator_grad_norm",
                               "score_adv", "score_other", "max_consecutive_adversarial", "batches")}
    sid_r = np.zeros(1024, np.int64); sid_a = np.zeros(1024, np.int64)
    lead_r = np.zeros(16, np.int64); lead_a = np.zeros(16, np.int64)
    for step, (indices, ze, sid, lead) in enumerate(loader):
        ze = ze.float().to(device, non_blocking=True); sid = sid.long().to(device, non_blocking=True)
        lead = lead.long().to(device, non_blocking=True); valid = (sid >= 0) & (sid < 1024)
        if train:
            gs = (epoch - 1) * len(loader) + step
            lr = e12.lr_at(gs, epochs * len(loader), 5 * len(loader), 5e-4, 1e-5, 1e-6)
            for g in model_opt.param_groups: g["lr"] = lr
            for g in gen_opt.param_groups: g["lr"] = lr

        gen_loss_value = grad_value = 0.
        hard = rand = adv = None
        # Adversary update: model weights frozen, ST gate remains differentiable.
        if train and frac > 0:
            for p in model.parameters(): p.requires_grad_(False)
            gen_opt.zero_grad(set_to_none=True)
            gen_loss, probe_ce, probe = generator_objective(model, masker, ze, sid, valid, lead)
            gen_loss.backward()
            grad_value = _grad_norm(masker.parameters())
            if not math.isfinite(grad_value) or grad_value <= 0:
                raise RuntimeError(f"Mask generator gradient invalid: {grad_value}")
            nn.utils.clip_grad_norm_(masker.parameters(), 3.); gen_opt.step()
            gen_loss_value = float(gen_loss.detach())
            for p in model.parameters(): p.requires_grad_(True)
            # The updated scorer now constructs the exact hard mask seen by the Transformer.
            with torch.no_grad():
                scores = masker(ze.detach()).masked_fill(~valid, -20.)
                hard, rand, adv, gate_g, meta_g = build_hybrid_mask(valid, lead, scores, frac)

        if train and frac == 0:
            hard = e12.random_mask(valid, .5); rand = hard; adv = torch.zeros_like(hard)
        if not train:
            # Stateless masks make every validation checkpoint directly comparable.
            mask_np = np.zeros(tuple(sid.shape), dtype=np.bool_)
            sid_np = sid.cpu().numpy(); idx_np = indices.cpu().numpy()
            for j, record_idx in enumerate(idx_np):
                mask_np[j] = e12.fixed_masks(sid_np[j:j+1], int(record_idx))[0]
            hard = torch.from_numpy(mask_np).to(device); rand = hard; adv = torch.zeros_like(hard)
        labels = sid[hard]
        if train:
            model_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(ze, valid, hard); loss = F.cross_entropy(logits.float(), labels)
        if train:
            model_scaler.scale(loss).backward(); model_scaler.unscale_(model_opt)
            nn.utils.clip_grad_norm_(model.parameters(), 3.); model_scaler.step(model_opt); model_scaler.update()
        with torch.no_grad():
            ce = F.cross_entropy(logits.float(), labels, reduction="none")
            is_adv = adv[hard]; is_rand = rand[hard]
            scalar["loss"] += float(loss); scalar["mlm_acc"] += float((logits.argmax(-1) == labels).float().mean())
            scalar["masked_tokens"] += labels.numel(); scalar["total_mask_ratio"] += float(hard.sum() / valid.sum())
            scalar["adv_of_mask"] += float(adv.sum() / hard.sum()); scalar["random_ce"] += float(ce[is_rand].mean())
            scalar["adv_ce"] += float(ce[is_adv].mean()) if is_adv.any() else 0.
            scalar["generator_loss"] += gen_loss_value; scalar["generator_grad_norm"] += grad_value; scalar["batches"] += 1
            s = masker(ze).detach(); scalar["score_adv"] += float(s[adv].mean()) if adv.any() else 0.
            scalar["score_other"] += float(s[valid & ~hard].mean()) if (valid & ~hard).any() else 0.
            adv_np=adv.cpu().numpy();lead_np_full=lead.cpu().numpy();batch_max=0
            for rr in range(len(adv_np)):
                run_now=0;prev_pos=-2;prev_lead=-1
                for pos in np.flatnonzero(adv_np[rr]):
                    lid=int(lead_np_full[rr,pos])
                    run_now=run_now+1 if lid==prev_lead and pos==prev_pos+1 else 1
                    batch_max=max(batch_max,run_now);prev_pos=int(pos);prev_lead=lid
            scalar["max_consecutive_adversarial"] += batch_max
            np.add.at(sid_r, labels[is_rand].cpu().numpy(), 1); np.add.at(sid_a, labels[is_adv].cpu().numpy(), 1)
            np.add.at(lead_r, lead[hard][is_rand].cpu().numpy(), 1); np.add.at(lead_a, lead[hard][is_adv].cpu().numpy(), 1)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            print(f"{'train' if train else 'val'} epoch={epoch} {step+1}/{len(loader)} loss={float(loss):.6f} adv={float(adv.sum()/hard.sum()):.4f} ggrad={grad_value:.4g}", flush=True)
    n = max(1, int(scalar["batches"])); stats = {k: (v / n if k != "batches" else v) for k, v in scalar.items()}
    detail = {"epoch": epoch, "target_adversarial_fraction": frac, "sid_random": sid_r.tolist(), "sid_adversarial": sid_a.tolist(),
              "lead_random": lead_r.tolist(), "lead_adversarial": lead_a.tolist(),
              "sid_js_adversarial_vs_random": _js(sid_a, sid_r), "lead_js_adversarial_vs_random": _js(lead_a, lead_r),
              "active_sid_random": int((sid_r > 0).sum()), "active_sid_adversarial": int((sid_a > 0).sum())}
    return stats, detail


def pretrain(run: Path, device: torch.device, epochs: int) -> None:
    out = run / "mlm/adversarial"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    model, init = e12.initialize_continuous("scratch", device)
    torch.manual_seed(170017); masker = MaskGenerator().to(device)
    dump(out / "initialization_audit.json", init | {
        "generator_seed": 170017,
        "generator_parameters": sum(p.numel() for p in masker.parameters()),
        "model_initial_state_matches_e12a": True})
    tr_ds, va_ds = AdvDataset(run, "train"), AdvDataset(run, "val")
    loader_gen = torch.Generator().manual_seed(0)
    tr = DataLoader(tr_ds, batch_size=256, shuffle=True, generator=loader_gen,
                    num_workers=8, pin_memory=True, drop_last=True)
    va = DataLoader(va_ds, batch_size=384, shuffle=False, num_workers=8,
                    pin_memory=True, drop_last=False)
    model_opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.05)
    gen_opt = torch.optim.AdamW(masker.parameters(), lr=5e-4, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf"); best_ep = None; rows = []; start = time.time()
    detail_path = out / "mask_generator_stats.jsonl"
    for ep in range(1, epochs + 1):
        tr_stat, detail = epoch_loop(model, masker, tr, device, model_opt, gen_opt, scaler, ep, epochs, True)
        with detail_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(detail) + "\n")
        row = {"epoch": ep, **{f"train_{k}": v for k, v in tr_stat.items()}}
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            with torch.no_grad():
                va_stat, _ = epoch_loop(model, masker, va, device, None, None, None, ep, epochs, False)
            row.update({f"val_{k}": v for k, v in va_stat.items()})
            if va_stat["loss"] < best:
                best = va_stat["loss"]; best_ep = ep
                torch.save({"model": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                            "mask_generator": {k:v.detach().cpu() for k,v in masker.state_dict().items()},
                            "config": e12.model_config(), "epoch": ep, "best_val_loss": best,
                            "generator_config": {"input_dim":64,"hidden_dim":128,"tau":.5}},
                           out / "checkpoint-best.pth")
        rows.append(row); dump(out / "training_history.json", rows); print(json.dumps(row), flush=True)
        if ep in (10, 50, epochs):
            torch.save({"model": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "mask_generator": {k:v.detach().cpu() for k,v in masker.state_dict().items()},
                        "config": e12.model_config(), "epoch": ep}, out / f"checkpoint-{ep}.pth")
    dump(out / "summary.json", {"epochs": epochs, "best_epoch": best_ep, "best_val_loss": best,
          "elapsed_seconds": time.time()-start, "pretrain_seed":0,
          "transformer_trainable_parameters": sum(p.numel() for p in model.parameters()),
          "generator_parameters": sum(p.numel() for p in masker.parameters()),
          "alternating_updates_after_epoch11": True, "test_run": False})
    (out / "COMPLETE").touch()


def load_masker(checkpoint: Path, device: torch.device) -> MaskGenerator:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    g = MaskGenerator(); g.load_state_dict(ck["mask_generator"], strict=True)
    return g.to(device).eval()


@torch.inference_mode()
def fixed_mlm(run: Path, device: torch.device) -> None:
    out = run / "fixed_mask_mlm_comparison.json"
    if out.exists(): return
    base = e12.load_continuous_checkpoint(E12_RUN / "mlm/scratch/checkpoint-best.pth", device)
    new = e12.load_continuous_checkpoint(run / "mlm/adversarial/checkpoint-best.pth", device)
    sid_np = np.load(e12.PRETRAIN_SID / "val_sid.npy", mmap_mode="r")
    ze_np = np.load(E12_RUN / "pretrain_ze_cache/val_ze.npy", mmap_mode="r")
    sums = {k:{"loss":0.,"correct":0,"tokens":0} for k in ("e12_a","e17_b")}
    for st in range(0, len(sid_np), 128):
        en=min(st+128,len(sid_np)); s0=np.asarray(sid_np[st:en]); mn=e12.fixed_masks(s0,st)
        sid=torch.from_numpy(s0.copy()).long().to(device); mask=torch.from_numpy(mn).to(device)
        valid=(sid>=0)&(sid<1024); labels=sid[mask]
        ze=torch.from_numpy(np.asarray(ze_np[st:en]).copy()).float().to(device)
        for key,logits in {"e12_a":base(ze,valid,mask),"e17_b":new(ze,valid,mask)}.items():
            sums[key]["loss"] += float(F.cross_entropy(logits.float(),labels,reduction="sum"))
            sums[key]["correct"] += int((logits.argmax(-1)==labels).sum()); sums[key]["tokens"] += labels.numel()
    dump(out,{k:{"val_ce":v["loss"]/v["tokens"],"masked_sid_accuracy":v["correct"]/v["tokens"],
                 "masked_tokens":v["tokens"]} for k,v in sums.items()} |
             {"mask_policy":"identical stateless 50% valid-token random mask; generator disabled for validation"})
    visualize_masks(run, device)


@torch.inference_mode()
def visualize_masks(run: Path, device: torch.device) -> None:
    png = run / "random_vs_adversarial_masks.png"
    if png.exists(): return
    ds = AdvDataset(run, "train"); masker = load_masker(run / "mlm/adversarial/checkpoint-best.pth", device)
    records = [17, 103, 709, 1709, 3709, 7709]; csv_rows=[]
    fig, axes = plt.subplots(len(records), 1, figsize=(16, 2.6*len(records)), constrained_layout=True)
    torch.manual_seed(202617)
    for ax, rec in zip(axes, records):
        _, z0, s0, l0 = ds[rec]
        ze=torch.from_numpy(z0[None]).float().to(device); sid=torch.from_numpy(s0[None]).long().to(device)
        lead=torch.from_numpy(l0[None]).long().to(device); valid=(sid>=0)&(sid<1024); score=masker(ze)
        hard, rnd, adv, gate, _=build_hybrid_mask(valid,lead,score,.30)
        baseline=torch.from_numpy(e12.fixed_masks(s0[None],rec)).to(device)
        n=int(valid.sum()); x=np.arange(n); norm=np.linalg.norm(z0[:n],axis=1)
        ax.plot(x,norm,color="0.75",lw=1,label="z_e norm")
        ax.scatter(x[baseline[0,:n].cpu()],norm[baseline[0,:n].cpu()],s=13,c="#377eb8",label="random 50%")
        ax.scatter(x[adv[0,:n].cpu()],norm[adv[0,:n].cpu()],s=28,c="#e41a1c",marker="x",label="adversarial 15%")
        ax.set_title(f"record {rec}, valid beats={n}"); ax.set_ylabel("||z_e||")
        for j in range(n):
            csv_rows.append([rec,j,int(l0[j]),int(s0[j]),float(score[0,j]),int(baseline[0,j]),int(rnd[0,j]),int(adv[0,j])])
    axes[-1].set_xlabel("token position"); axes[0].legend(ncol=3,loc="upper right")
    fig.savefig(png,dpi=170); plt.close(fig)
    with (run / "random_vs_adversarial_masks.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["record","position","lead_id","sid","generator_score","baseline_random_mask","hybrid_random_part","hybrid_adversarial_part"]);w.writerows(csv_rows)


def features(run: Path, device: torch.device) -> None:
    e12.extract_downstream(run, "adversarial", device, 32)


def train_one(run: Path, seed: int, device: torch.device) -> dict:
    out=run/f"downstream/e17b/seed{seed}"
    if (out/"VAL_COMPLETE").exists(): return json.loads((out/"val_summary.json").read_text())
    out.mkdir(parents=True,exist_ok=False); data=e12.load_downstream_arrays(run,"adversarial",device)
    model=e11.E11RegionResidual(seed).to(device); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
    target=data["train"]["labels"].argmax(1);steps=len(target)//64;gen=torch.Generator().manual_seed(seed)
    best=-float("inf");best_ep=None;state=None;history=[];reg,regmask=e11.masked_region_means(data["train"]["lead"],data["train"]["lead_mask"])
    for ep in range(1,31):
        model.train();order=torch.randperm(len(target),generator=gen).to(device);lm=[];la=[]
        for step in range(steps):
            idx=order[step*64:(step+1)*64];lr=e12.lr_at((ep-1)*steps+step,30*steps,5*steps,5e-4,1e-5,1e-6)
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True);main,aux=model.training_outputs(data["train"]["mlm"][idx],data["train"]["global"][idx],reg[idx],regmask[idx])
            l1=F.cross_entropy(main,target[idx]);l2=e12.e9.patient_auxiliary_loss(aux,target[idx]);loss=l1+.2*l2
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();lm.append(float(l1));la.append(float(l2))
        val,_,_=e12.eval_downstream(model,"e17b",data["val"]);history.append({"epoch":ep,"main_ce":float(np.mean(lm)),"patient_ce":float(np.mean(la)),"val":val})
        if val["r2"]>best:best=val["r2"];best_ep=ep;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);val,prob,logits=e12.eval_downstream(model,"e17b",data["val"])
    torch.save({"model":{k:v.cpu() for k,v in state.items()},"seed":seed,"best_epoch":best_ep},out/"checkpoint-best.pth")
    np.save(out/"val_logits.npy",logits);np.save(out/"val_probabilities.npy",prob);dump(out/"history.json",history)
    res={"seed":seed,"pretrain_seed":0,"best_epoch":best_ep,"val":val};dump(out/"val_summary.json",res);(out/"VAL_COMPLETE").touch();return res


def downstream(run: Path, device: torch.device) -> None:
    out=run/"validation_summary.json"
    if out.exists():return
    rows=[train_one(run,s,device) for s in (0,1,2)]
    base=json.loads((E13_RUN/"validation_selection.json").read_text())["val"]["e13_b"]
    def agg(k):
        x=[r["val"][k] for r in rows];return {"mean":float(np.mean(x)),"sd":float(np.std(x,ddof=1))}
    new={k:agg(k) for k in ("r2","mse","mae","macro_roc_auc","macro_f1","accuracy")}
    new["per_output_r2"]={n:{"mean":float(np.mean([r["val"]["per_output_r2"][n] for r in rows])),
                                "sd":float(np.std([r["val"]["per_output_r2"][n] for r in rows],ddof=1))} for n in e12.LABEL_NAMES}
    dump(out,{"selection_split":"validation","test_run":False,"e12_a":base,"e17_b":new,
              "delta":{"r2":new["r2"]["mean"]-base["r2"]["mean"],**{n:new["per_output_r2"][n]["mean"]-base["per_output_r2"][n]["mean"] for n in e12.LABEL_NAMES}},"per_seed":rows})


def aggregate_generator(run: Path) -> dict:
    rows=[json.loads(x) for x in (run/"mlm/adversarial/mask_generator_stats.jsonl").read_text().splitlines() if x.strip()]
    full=[r for r in rows if r["epoch"]>=21]
    sid_r=np.sum([np.asarray(r["sid_random"],np.int64) for r in full],0);sid_a=np.sum([np.asarray(r["sid_adversarial"],np.int64) for r in full],0)
    lead_r=np.sum([np.asarray(r["lead_random"],np.int64) for r in full],0);lead_a=np.sum([np.asarray(r["lead_adversarial"],np.int64) for r in full],0)
    return {"full_adversarial_epochs":len(full),"sid_js_adversarial_vs_random":_js(sid_a,sid_r),"lead_js_adversarial_vs_random":_js(lead_a,lead_r),
            "active_sid_random":int((sid_r>0).sum()),"active_sid_adversarial":int((sid_a>0).sum()),
            "lead_adversarial_share":{str(i):float(lead_a[i]/max(1,lead_a.sum())) for i in np.flatnonzero(lead_a)},
            "sid_adversarial_top20":{str(int(i)):int(sid_a[i]) for i in np.argsort(sid_a)[-20:][::-1]},
            "random_masked_tokens":int(sid_r.sum()),"adversarial_masked_tokens":int(sid_a.sum())}


def report(run: Path) -> None:
    s=json.loads((run/"validation_summary.json").read_text());m=json.loads((run/"fixed_mask_mlm_comparison.json").read_text());g=aggregate_generator(run);dump(run/"mask_generator_summary.json",g)
    hist=json.loads((run/"mlm/adversarial/training_history.json").read_text());full=[x for x in hist if x["epoch"]>=21]
    a=s["e12_a"];b=s["e17_b"]
    metrics=[("Overall R²",a["r2"]["mean"],b["r2"]["mean"])]
    metrics += [(n+" R²",a["per_output_r2"][n]["mean"],b["per_output_r2"][n]["mean"]) for n in e12.LABEL_NAMES]
    metrics += [("Macro AUC",a["macro_roc_auc"]["mean"],b["macro_roc_auc"]["mean"]),("Macro F1",a["macro_f1"]["mean"],b["macro_f1"]["mean"])]
    with (run/"metrics_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["metric","E12-A","E17-B","delta"]);w.writerows((n,x,y,y-x) for n,x,y in metrics)
    mean_grad=float(np.mean([x["train_generator_grad_norm"] for x in full]));mean_adv=float(np.mean([x["train_adv_of_mask"] for x in full]))
    adv_ce=float(np.mean([x["train_adv_ce"] for x in full]));rnd_ce=float(np.mean([x["train_random_ce"] for x in full]))
    lines=["# E17：Adversarial Masking 优化 ECG 自监督预训练","","## 方法与边界","",
           "- E17-A为只读E12-A；E17-B仅改变Mask位置选择，其余架构、数据、SID CE、优化器和下游协议不变。",
           "- Epoch 1–10全随机；11–20从0线性增加；21–100为70%随机+30%对抗，总Mask严格为每条记录有效心拍的50%。",
           "- 生成器是训练期LayerNorm+两层MLP；随机50%探测Mask先提供无泄漏的逐位置CE奖励，再以sigmoid阈值exact-k松弛最大化期望困难度；Transformer真实前向使用硬Top-K。按实测导联分配对抗配额，Padding不参与。",
           "- 生成器与Transformer交替更新：生成器最大化随机探测位置的期望SID CE，Transformer最小化混合硬Mask的SID CE。未使用疾病标签，也未运行Test。",
           "- 原论文在原始波形上用1D U-Net生成软遮挡并优化SimCLR；本实验迁移为心拍级SID MLM，因此是改造实验，不是完整复现。","",
           "## 固定Mask MLM","","|模型|Val CE|Masked SID Accuracy|","|---|---:|---:|",
           f"|E12-A|{m['e12_a']['val_ce']:.6f}|{m['e12_a']['masked_sid_accuracy']:.6f}|",
           f"|E17-B|{m['e17_b']['val_ce']:.6f}|{m['e17_b']['masked_sid_accuracy']:.6f}|","",
           "## Validation（三下游seed均值）","","|指标|E12-A|E17-B|差值|","|---|---:|---:|---:|"]
    lines += [f"|{n}|{x:.6f}|{y:.6f}|{y-x:+.6f}|" for n,x,y in metrics]
    lines += ["","## Generator是否真正学习","",
              f"- Epoch 21–100实际对抗Mask占全部Mask：{mean_adv:.6f}；Generator平均梯度范数：{mean_grad:.6f}。",
              f"- 对抗位置平均CE：{adv_ce:.6f}；随机位置平均CE：{rnd_ce:.6f}。",
              f"- SID分布JS散度（对抗 vs 随机）：{g['sid_js_adversarial_vs_random']:.6f}；导联分布JS散度：{g['lead_js_adversarial_vs_random']:.6f}。",
              f"- 对抗Mask覆盖active SID数：{g['active_sid_adversarial']}；随机部分覆盖：{g['active_sid_random']}。",
              "- 位置示意见 `random_vs_adversarial_masks.png`，逐位置数据见同名CSV。","",
              "## 结论","",
              f"Val Overall R²：{a['r2']['mean']:.6f} → {b['r2']['mean']:.6f}（{b['r2']['mean']-a['r2']['mean']:+.6f}）。",
              f"OMI变化：{s['delta']['OMI']:+.6f}；NOMI变化：{s['delta']['NOMI']:+.6f}。",
              "结合对抗/随机位置CE、SID与导联JS散度及梯度范数，判断收益或损害更可能来自遮挡难度、覆盖偏移，还是对抗优化不稳定。",""]
    (run/"README.md").write_text("\n".join(lines),encoding="utf-8");(run/"COMPLETE").touch();(run/"status.txt").write_text("complete")


def smoke(run: Path, device: torch.device) -> None:
    out=run/"smoke"
    if (out/"COMPLETE").exists(): return
    out.mkdir(parents=True,exist_ok=False);model,_=e12.initialize_continuous("scratch",device);torch.manual_seed(170017);masker=MaskGenerator().to(device)
    model.eval();masker.train();ze=torch.randn(4,256,64,device=device);sid=torch.full((4,256),1024,dtype=torch.long,device=device)
    sid[:,:40]=torch.randint(0,1024,(4,40),device=device);valid=sid<1024
    lead=torch.zeros_like(sid);lead[:,:40]=torch.tensor(([1]*5+[2]*5+[3]*5+[7]*5+[8]*5+[9]*5+[10]*5+[11]*5),device=device)
    scores=masker(ze).masked_fill(~valid,-20.);hard,rnd,adv,gate,meta=build_hybrid_mask(valid,lead,scores,.30)
    expected=torch.tensor([round(.5*int(x)) for x in valid.sum(1)],device=device)
    if not torch.equal(hard.sum(1),expected) or (hard&~valid).any() or meta["actual_adv"]!=meta["target_adv"]:
        raise RuntimeError("exact mask budget smoke failed")
    with torch.no_grad():
        a=model(ze,valid,hard);z2=ze.clone();z2[hard]=torch.randn_like(z2[hard])*1e6;b=model(z2,valid,hard)
        c=gated_logits(model,ze,valid,gate,hard)
    leakage=float((a-b).abs().max());equiv=float((a-c).abs().max())
    for p in model.parameters():p.requires_grad_(False)
    loss,probe_ce,probe=generator_objective(model,masker,ze,sid,valid,lead);loss.backward();gn=_grad_norm(masker.parameters())
    model_grad=any(p.grad is not None for p in model.parameters())
    for p in model.parameters():p.requires_grad_(True)
    if leakage!=0 or equiv>1e-5 or not math.isfinite(gn) or gn<=0 or model_grad:raise RuntimeError(f"gradient/leak smoke failed {leakage=} {equiv=} {gn=} {model_grad=}")
    lead_counts={str(int(x)):int(adv[lead==x].sum()) for x in torch.unique(lead[valid])}
    dump(out/"audit.json",{"passed":True,"exact_total_mask_ratio":.5,"target_adversarial_of_mask":.3,
          "actual_adversarial_of_mask":meta["actual_adv"]/meta["actual_total"],"padding_masked":False,
          "hard_vs_st_forward_max_delta":equiv,"masked_z_e_mutation_max_delta":leakage,
          "generator_gradient_norm":gn,"generator_receives_gradient":True,"transformer_gradient_during_generator_step":model_grad,
          "generator_objective":"differentiable exact-k expected CE on independently random-masked probe tokens",
          "probe_mask_ratio":float(probe.sum()/valid.sum()),"probe_mean_ce":probe_ce,
          "lead_balanced_adversarial_counts":lead_counts,"clinical_labels_used":False})
    (out/"COMPLETE").touch()


def protocol(run: Path) -> None:
    dump(run / "protocol.json", {"experiment": "E17-B", "baseline": "read-only E12-A/E13-B",
          "method_scope": "heartbeat-token SID-MLM adaptation of Bo et al. 2022; not a complete reproduction",
          "architecture": "unchanged E12-A plus training-only LayerNorm-MLP(64,128,1) mask generator",
          "pretrain_records": [87066, 8941], "epochs": 100, "batch": 256, "pretrain_seed": 0,
          "downstream_seeds": [0,1,2], "mask_ratio": .5, "mask_mix": "70% random + 30% adversarial at full schedule",
          "schedule": "epoch1-10 random; epoch11 0%; epoch20 30%; epoch21-100 30% adversarial",
          "relaxation": "sigmoid-threshold exact-k relaxation over leakage-free random-probe token CE rewards, tau=0.5; hard per-lead top-k for Transformer forward",
          "optimization": "alternate generator maximize relaxed expected random-probe SID CE and Transformer minimize hard-mask SID CE; AdamW same LR schedule",
          "generator_constraints": "proportional measured-lead quota; padding excluded; exact per-record 50% total budget",
          "validation": "fixed random 50% valid-token masks; generator unused", "test_policy": "no new test"})


def audit(run: Path) -> None:
    e12.overlap_audit(run)
    dump(run / "method_audit.json", {
        "paper": "Pretraining ECG Data with Adversarial Masking Improves Model Generalizability for Data-Scarce Tasks, arXiv:2211.07889",
        "official_code": "https://github.com/jessica-bo/advmask_ecg",
        "paper_method": "raw-waveform 1D U-Net soft mask, alternating max/min SimCLR optimization with sparsity constraint",
        "project_existing_flags": "run_heartlang_pretraining.py exposes adversarial-mask CLI arguments",
        "project_existing_engine": "engine_for_pretraining.py always invokes random_masking and does not consume those adversarial flags",
        "existing_implementation_reusable": False,
        "e17_adaptation": "heartbeat z_e MLP scorer + exact-budget lead-balanced hard top-k; differentiable exact-k expected-CE maximization from independently random-masked probe tokens",
        "not_equivalent_to_paper": True, "clinical_labels_used": False, "test_policy": "validation only"})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["audit","meta","smoke","pretrain","fixed_mlm","features","downstream","report"])
    ap.add_argument("--output_dir", type=Path, default=RUN_DEFAULT); ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args(); run = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run.mkdir(parents=True, exist_ok=True); protocol(run); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "audit": audit(run)
    elif args.stage == "meta": prepare_meta(run)
    elif args.stage == "smoke": smoke(run,device)
    elif args.stage == "pretrain": pretrain(run,device,args.epochs)
    elif args.stage == "fixed_mlm": fixed_mlm(run,device)
    elif args.stage == "features": features(run,device)
    elif args.stage == "downstream": downstream(run,device)
    elif args.stage == "report": report(run)


if __name__ == "__main__": main()
