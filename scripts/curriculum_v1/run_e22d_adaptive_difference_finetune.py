#!/usr/bin/env python3
"""E22-D: adapt E22-C downstream with local-difference pooling and a sample gate.

The E22-C checkpoint is the only ECG representation source.  Blocks 1--11 are
cached and remain frozen.  Only block 12, the final LayerNorm, the original E11
four-class head, and a zero-start local-difference residual are optimized.
Validation selects checkpoints; Test is never loaded.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]
import run_e12_continuous_input as e12
import run_e22c_pretraining_only_local_decoder as e22c

DEFAULT_RUN = ROOT / "results/e22d_adaptive_difference_finetune"
E22C_RUN = ROOT / "results/e22c_pretraining_only_local_decoder"
LAST_BLOCK_LR = 1e-5
HEAD_LR = 5e-4
EPOCHS = 30


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_status(run: Path, value: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "status.txt").write_text(value + "\n", encoding="utf-8")
    print(value, flush=True)


def block_forward(block, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    attn, ff = block
    key_valid = valid[:, None, None, :]
    query_valid = valid.unsqueeze(-1).to(x.dtype)
    y = attn.norm(x)
    q, k, v = (rearrange(z, "b n (h d) -> b h n d", h=attn.heads)
               for z in attn.to_qkv(y).chunk(3, dim=-1))
    dots = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
    dots = dots.masked_fill(~key_valid, torch.finfo(dots.dtype).min)
    weights = attn.dropout(attn.attend(dots))
    z = torch.matmul(weights, v)
    z = rearrange(z, "b h n d -> b n (h d)")
    x = (attn.to_out(z) + x) * query_valid
    return (ff(x) + x) * query_valid


@torch.inference_mode()
def cache_penultimate(run: Path, device: torch.device, batch_size: int = 16) -> None:
    out = run / "penultimate_cache"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = e12.e4.load_model(e12.TOKENIZER, device).eval().requires_grad_(False)
    checkpoint = E22C_RUN / "mlm/checkpoint-best.pth"
    model = e22c.load_model(checkpoint, device).eval().requires_grad_(False)
    backbone = model.backbone
    audit = {"source_checkpoint": str(checkpoint),
             "source_epoch": int(torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"]),
             "cache_point": "input to E22-C Transformer block 12",
             "dtype": "float16", "test_loaded": False, "splits": {}}
    for split in ("train", "val"):
        waves = np.load(e12.DOWNSTREAM / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(e12.DOWNSTREAM / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(e12.DOWNSTREAM / f"{split}_data_in_times.npy", mmap_mode="r")
        hidden = np.lib.format.open_memmap(out / f"{split}_prelast.npy", mode="w+", dtype=np.float16,
                                           shape=(len(waves), 257, 768))
        valid_out = np.lib.format.open_memmap(out / f"{split}_valid.npy", mode="w+", dtype=bool,
                                              shape=(len(waves), 257))
        neighbors = np.lib.format.open_memmap(out / f"{split}_neighbors.npy", mode="w+", dtype=np.int16,
                                              shape=(len(waves), 256, 3))
        valid_tokens = full_triplets = cross_lead = negative_time = 0
        for start in range(0, len(waves), batch_size):
            end = min(start + batch_size, len(waves))
            wn = np.asarray(waves[start:end], dtype=np.float32)
            cn = np.asarray(chans[start:end]).copy()
            tn = np.asarray(times[start:end]).copy()
            x0 = torch.from_numpy(wn.copy()).to(device)
            ch = torch.from_numpy(cn).long().to(device)
            ti = torch.from_numpy(tn).long().to(device)
            valid = tokenizer.valid_mask(x0, ch, ti)
            ze = tokenizer.encode_to_quantizer_features(x0, ch, ti).float()
            x = backbone.input_proj(ze) * valid.unsqueeze(-1)
            cls = backbone.cls_token.view(1, 1, -1).expand(len(x), 1, -1)
            x = torch.cat([cls, x], 1)
            full_valid = torch.cat([torch.ones(len(x), 1, dtype=torch.bool, device=device), valid], 1)
            x = backbone.dropout(x + backbone.pos_embed)
            for block in backbone.transformer.layers[:-1]:
                x = block_forward(block, x, full_valid)
            vn = valid.cpu().numpy()
            nei = e22c.make_neighbors(cn, tn, vn)
            hidden[start:end] = x.half().cpu().numpy()
            valid_out[start:end] = full_valid.cpu().numpy()
            neighbors[start:end] = nei
            valid_tokens += int(vn.sum())
            full_triplets += int(((nei[:, :, 0] >= 0) & (nei[:, :, 2] >= 0)).sum())
            for b in range(len(cn)):
                for slot in (0, 2):
                    q = np.flatnonzero(nei[b, :, slot] >= 0)
                    if len(q):
                        cross_lead += int(np.sum(cn[b, q] != cn[b, nei[b, q, slot]]))
                        if slot == 0:
                            negative_time += int(np.sum(tn[b, q] < tn[b, nei[b, q, slot]]))
                        else:
                            negative_time += int(np.sum(tn[b, q] > tn[b, nei[b, q, slot]]))
            if start == 0 or end == len(waves) or start // batch_size % 50 == 0:
                print(f"cache {split} {end}/{len(waves)}", flush=True)
        hidden.flush(); valid_out.flush(); neighbors.flush()
        audit["splits"][split] = {
            "records": int(len(waves)), "valid_tokens": valid_tokens,
            "full_triplet_fraction": full_triplets / max(1, valid_tokens),
            "cross_lead_neighbors": cross_lead, "time_order_violations": negative_time,
            "hidden_shape": [int(len(waves)), 257, 768],
        }
    if any(v["cross_lead_neighbors"] or v["time_order_violations"] for v in audit["splits"].values()):
        raise RuntimeError(f"neighbor audit failed: {audit}")
    torch.save({"block": copy.deepcopy(backbone.transformer.layers[-1].state_dict()),
                "norm": copy.deepcopy(backbone.norm_layer.state_dict())},
               out / "last_block_initial_state.pth")
    dump(out / "audit.json", audit)
    (out / "COMPLETE").touch()


class CacheArrays:
    def __init__(self, run: Path, split: str):
        root = run / "penultimate_cache"
        self.hidden = np.load(root / f"{split}_prelast.npy", mmap_mode="r")
        self.valid = np.load(root / f"{split}_valid.npy", mmap_mode="r")
        self.neighbors = np.load(root / f"{split}_neighbors.npy", mmap_mode="r")

    def batch(self, idx: np.ndarray, device: torch.device):
        h = torch.from_numpy(np.asarray(self.hidden[idx], dtype=np.float32)).to(device)
        v = torch.from_numpy(np.asarray(self.valid[idx]).copy()).bool().to(device)
        n = torch.from_numpy(np.asarray(self.neighbors[idx]).copy()).long().to(device)
        return h, v, n


class AdaptiveDifferenceModel(nn.Module):
    def __init__(self, seed: int, initial_state: dict):
        super().__init__()
        template = e22c.load_model(E22C_RUN / "mlm/checkpoint-best.pth", torch.device("cpu")).backbone
        self.last_block = copy.deepcopy(template.transformer.layers[-1])
        self.final_norm = copy.deepcopy(template.norm_layer)
        self.last_block.load_state_dict(initial_state["block"])
        self.final_norm.load_state_dict(initial_state["norm"])
        self.region_head = e12.e11.E11RegionResidual(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(220040 + int(seed))
            self.difference_project = nn.Sequential(
                nn.Linear(1536, 256, bias=False), nn.GELU(), nn.Linear(256, 1536, bias=False))
            self.sample_gate = nn.Sequential(nn.Linear(1536, 128), nn.GELU(), nn.Linear(128, 1))
        # Exact E22-C + E11 starting function: residual branch initially contributes zero.
        nn.init.zeros_(self.difference_project[-1].weight)
        nn.init.zeros_(self.sample_gate[-1].weight)
        nn.init.constant_(self.sample_gate[-1].bias, -4.0)

    @staticmethod
    def masked_mean_max(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        count = valid.sum(1).clamp_min(1).unsqueeze(-1)
        mean = (x * valid.unsqueeze(-1)).sum(1) / count
        maximum = x.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(valid.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], 1)

    def representations(self, prelast, full_valid, neighbors):
        x = block_forward(self.last_block, prelast, full_valid)
        x = self.final_norm(x) * full_valid.unsqueeze(-1)
        token, valid = x[:, 1:], full_valid[:, 1:]
        base = self.masked_mean_max(token, valid)
        side = neighbors[:, :, [0, 2]]
        idx = side.clamp_min(0)
        bi = torch.arange(len(token), device=token.device)[:, None, None]
        gathered = token[bi, idx]
        side_valid = (side >= 0) & valid[bi, idx]
        denom = side_valid.sum(2).clamp_min(1).unsqueeze(-1)
        local_mean = (gathered * side_valid.unsqueeze(-1)).sum(2) / denom
        diff_valid = valid & side_valid.any(2)
        local_diff = (token - local_mean).abs() * diff_valid.unsqueeze(-1)
        diff = self.masked_mean_max(local_diff, diff_valid)
        delta = self.difference_project(diff)
        gate = torch.sigmoid(self.sample_gate(base))
        return base + gate * delta, base, diff, gate

    def forward(self, prelast, full_valid, neighbors, global_ze, regions, region_mask):
        adapted, _, _, _ = self.representations(prelast, full_valid, neighbors)
        return self.region_head(adapted, global_ze, regions, region_mask)

    def training_outputs(self, prelast, full_valid, neighbors, global_ze, regions, region_mask):
        adapted, _, _, gate = self.representations(prelast, full_valid, neighbors)
        main, aux = self.region_head.training_outputs(adapted, global_ze, regions, region_mask)
        return main, aux, gate


def load_small_data(device: torch.device) -> dict:
    data = {}
    for split in ("train", "val"):
        labels = np.load(e12.DOWNSTREAM / f"{split}_labels.npy").astype(np.float32)
        lead = np.load(e12.E10_RUN / f"features/{split}_lead_ze_mean_max.npy")
        lead_mask = np.load(e12.E10_RUN / f"features/{split}_lead_present.npy")
        global_ze = np.load(e12.E10_RUN / f"features/{split}_global_ze_mean_max.npy")
        data[split] = {"labels": torch.from_numpy(labels).to(device),
                       "lead": torch.from_numpy(lead).float().to(device),
                       "lead_mask": torch.from_numpy(lead_mask).bool().to(device),
                       "global": torch.from_numpy(global_ze).float().to(device)}
    if [len(data[s]["labels"]) for s in ("train", "val")] != [6640, 1344]:
        raise RuntimeError("downstream patient split changed")
    return data


@torch.no_grad()
def evaluate(model, cache, arrays, device):
    model.eval(); chunks = []; gates = []
    regions, region_mask = e12.e11.masked_region_means(arrays["lead"], arrays["lead_mask"])
    for start in range(0, len(arrays["labels"]), 64):
        idx = np.arange(start, min(start + 64, len(arrays["labels"])))
        h, v, n = cache.batch(idx, device)
        adapted, _, _, gate = model.representations(h, v, n)
        raw = model.region_head(adapted, arrays["global"][idx], regions[idx], region_mask[idx])
        chunks.append(raw.float().cpu().numpy()); gates.append(gate.float().cpu().numpy())
    logits = np.concatenate(chunks); gate_np = np.concatenate(gates).reshape(-1)
    metrics, prob = e12.probability_metrics(logits, arrays["labels"].cpu().numpy())
    metrics["patient_mean_r2"] = float(np.mean([metrics["per_output_r2"][n] for n in e12.LABEL_NAMES[:3]]))
    metrics["gate_mean"] = float(gate_np.mean()); metrics["gate_sd"] = float(gate_np.std())
    metrics["gate_p05"] = float(np.quantile(gate_np, .05)); metrics["gate_p95"] = float(np.quantile(gate_np, .95))
    return metrics, prob, logits, gate_np


def smoke(run: Path, device: torch.device) -> None:
    path = run / "smoke.json"
    if path.exists(): return
    if not (run / "penultimate_cache/COMPLETE").exists():
        raise RuntimeError("cache incomplete")
    state = torch.load(run / "penultimate_cache/last_block_initial_state.pth", map_location="cpu", weights_only=False)
    model = AdaptiveDifferenceModel(0, state).to(device)
    data = load_small_data(device); cache = CacheArrays(run, "train")
    idx = np.arange(16); idx_t = torch.from_numpy(idx).long().to(device)
    h, v, n = cache.batch(idx, device)
    regions, rm = e12.e11.masked_region_means(data["train"]["lead"][idx_t], data["train"]["lead_mask"][idx_t])
    with torch.no_grad():
        adapted, base, diff, gate = model.representations(h, v, n)
        initial_delta = float((adapted - base).abs().max())
    target = data["train"]["labels"][idx_t].argmax(1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    grad_rows = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        main, aux, gate_now = model.training_outputs(h, v, n, data["train"]["global"][idx_t], regions, rm)
        loss = F.cross_entropy(main.float(), target) + .2 * e12.e9.patient_auxiliary_loss(aux.float(), target)
        loss.backward()
        grad_rows.append({"step": step + 1,
                          "last_block": float(next(model.last_block.parameters()).grad.norm()),
                          "difference_output": float(model.difference_project[-1].weight.grad.norm()),
                          "sample_gate": float(model.sample_gate[-1].weight.grad.norm())})
        opt.step()
    passed = (initial_delta == 0 and grad_rows[0]["last_block"] > 0 and
              grad_rows[0]["difference_output"] > 0 and grad_rows[1]["sample_gate"] > 0 and
              math.isfinite(float(loss)) and float(diff.abs().sum()) > 0)
    result = {"passed": passed, "source": "E22-C checkpoint-best", "initial_adapted_vs_base_max_abs_diff": initial_delta,
              "initial_gate_mean": float(gate.mean()), "local_difference_nonzero": bool(diff.abs().sum() > 0),
              "grad_norms": grad_rows, "blocks_1_to_11": "cached and absent from optimizer",
              "trainable": "last block, final norm, E11 head, difference projection, sample gate",
              "text_age_sex_used": False, "test_loaded": False}
    if not passed: raise RuntimeError(f"smoke failed: {result}")
    dump(path, result)


def train_seed(run: Path, seed: int, device: torch.device) -> dict:
    out = run / "downstream" / f"seed{seed}"
    if (out / "VAL_COMPLETE").exists():
        return json.loads((out / "val_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    state = torch.load(run / "penultimate_cache/last_block_initial_state.pth", map_location="cpu", weights_only=False)
    model = AdaptiveDifferenceModel(seed, state).to(device)
    last_params = list(model.last_block.parameters()) + list(model.final_norm.parameters())
    last_ids = {id(p) for p in last_params}
    head_params = [p for p in model.parameters() if id(p) not in last_ids]
    optimizer = torch.optim.AdamW([{"params": last_params, "lr": LAST_BLOCK_LR},
                                  {"params": head_params, "lr": HEAD_LR}], weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    data = load_small_data(device); train_cache = CacheArrays(run, "train"); val_cache = CacheArrays(run, "val")
    target = data["train"]["labels"].argmax(1)
    regions, rm = e12.e11.masked_region_means(data["train"]["lead"], data["train"]["lead_mask"])
    rng = np.random.default_rng(seed); steps = len(target) // 64
    best = -float("inf"); best_epoch = None; best_state = None; history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); order = rng.permutation(len(target)); losses = []; gate_values = []
        for step in range(steps):
            idx = order[step*64:(step+1)*64]; idx_t = torch.from_numpy(idx).long().to(device)
            progress = ((epoch - 1) * steps + step) / max(1, EPOCHS * steps - 1)
            warm = min(1., ((epoch - 1) * steps + step + 1) / max(1, 5 * steps))
            cosine = .5 * (1 + math.cos(math.pi * progress))
            for group, peak in zip(optimizer.param_groups, (LAST_BLOCK_LR, HEAD_LR)):
                group["lr"] = (1e-6 + (peak - 1e-6) * warm) if warm < 1 else 1e-6 + (peak - 1e-6) * cosine
            h, v, n = train_cache.batch(idx, device); optimizer.zero_grad(set_to_none=True)
            # E11 uses a -1e9 region-mask sentinel, which is intentionally kept
            # in float32 (it overflows IEEE fp16).  The cached first 11 blocks
            # already remove the expensive part, so AMP is unnecessary here.
            main, aux, gate = model.training_outputs(h, v, n, data["train"]["global"][idx_t], regions[idx_t], rm[idx_t])
            loss = F.cross_entropy(main.float(), target[idx_t]) + .2 * e12.e9.patient_auxiliary_loss(aux.float(), target[idx_t])
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.); scaler.step(optimizer); scaler.update()
            losses.append(float(loss)); gate_values.append(float(gate.mean()))
        val, _, _, _ = evaluate(model, val_cache, data["val"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_gate_mean": float(np.mean(gate_values)), "val": val}
        history.append(row); print(json.dumps({"seed": seed, **row}, ensure_ascii=False), flush=True)
        if val["r2"] > best:
            best = val["r2"]; best_epoch = epoch
            best_state = copy.deepcopy({k: x.detach().cpu() for k, x in model.state_dict().items()})
    model.load_state_dict(best_state)
    val, prob, logits, gates = evaluate(model, val_cache, data["val"], device)
    torch.save({"model": best_state, "seed": seed, "best_epoch": best_epoch,
                "selection": "Val Overall R2", "test_run": False}, out / "checkpoint-best.pth")
    np.save(out / "val_probabilities.npy", prob); np.save(out / "val_logits.npy", logits); np.save(out / "val_gates.npy", gates)
    dump(out / "history.json", history)
    result = {"method": "e22d_adaptive_difference_finetune", "seed": seed, "best_epoch": best_epoch,
              "val": val, "test_run": False, "last_block_lr": LAST_BLOCK_LR, "head_lr": HEAD_LR,
              "trainable_parameters": sum(p.numel() for p in model.parameters())}
    dump(out / "val_summary.json", result); (out / "VAL_COMPLETE").touch(); return result


def aggregate(rows: list[dict]) -> dict:
    vals = [r["val"] for r in rows]
    out = {}
    for key in ("r2", "macro_roc_auc", "macro_f1", "mse", "gate_mean", "gate_sd"):
        a = np.array([v[key] for v in vals], float); out[key] = {"mean": float(a.mean()), "sd": float(a.std(ddof=0))}
    out["patient_mean_r2"] = {"mean": float(np.mean([v["patient_mean_r2"] for v in vals])),
                              "sd": float(np.std([v["patient_mean_r2"] for v in vals]))}
    out["per_output_r2"] = {}
    for name in e12.LABEL_NAMES:
        a = np.array([v["per_output_r2"][name] for v in vals], float)
        out["per_output_r2"][name] = {"mean": float(a.mean()), "sd": float(a.std(ddof=0))}
    return out


def downstream(run: Path, device: torch.device) -> None:
    path = run / "validation_summary.json"
    if path.exists(): return
    rows = [train_seed(run, seed, device) for seed in (0, 1, 2)]
    old = json.loads((E22C_RUN / "validation_summary.json").read_text())
    dump(path, {"validation_only": True, "test_run": False,
                "e12a_historical": old["e12a_historical"],
                "e22b_writeback_historical": old["e22b_writeback_historical"],
                "e22c_frozen_historical": old["e22c_pretraining_only_decoder"],
                "e22d_adaptive_difference_finetune": {"aggregate": aggregate(rows), "per_seed": rows}})


def report(run: Path) -> None:
    summary = json.loads((run / "validation_summary.json").read_text())
    cache = json.loads((run / "penultimate_cache/audit.json").read_text())
    rows = []
    names = [("E12-A", "e12a_historical"), ("E22-B-Writeback", "e22b_writeback_historical"),
             ("E22-C-Frozen", "e22c_frozen_historical"), ("E22-D-Adaptive", "e22d_adaptive_difference_finetune")]
    for label, key in names:
        for item in summary[key]["per_seed"]:
            v = item["val"]
            rows.append({"method": label, "seed": item.get("seed", item.get("downstream_seed")),
                         "overall_r2": v["r2"], "patient_mean_r2": float(np.mean([v["per_output_r2"][n] for n in e12.LABEL_NAMES[:3]])),
                         **{f"{n.lower().replace(' ', '_')}_r2": v["per_output_r2"][n] for n in e12.LABEL_NAMES},
                         "macro_auc": v["macro_roc_auc"], "macro_f1": v["macro_f1"],
                         "gate_mean": v.get("gate_mean")})
    pd.DataFrame(rows).to_csv(run / "metrics_per_seed.csv", index=False)
    def fmt(label, x):
        return (f"| {label} | {x['r2']['mean']:.6f} ± {x['r2']['sd']:.6f} | {x['patient_mean_r2']['mean']:.6f} | " +
                " | ".join(f"{x['per_output_r2'][n]['mean']:.6f}" for n in e12.LABEL_NAMES) +
                f" | {x['macro_roc_auc']['mean']:.6f} | {x['macro_f1']['mean']:.6f} |")
    base = summary["e12a_historical"]["aggregate"]; c = summary["e22c_frozen_historical"]["aggregate"]
    new = summary["e22d_adaptive_difference_finetune"]["aggregate"]
    lines = ["# E22-D：最后一个Block微调 + 三心拍局部差异池化 + 样本自适应门控", "",
             "## 方法", "", "- 从E22-C最佳checkpoint开始；Transformer Blocks 1–11冻结并缓存。",
             "- 仅微调Block 12与最终LayerNorm；保留每个96点心拍为独立token。",
             "- 每个token与同导联真实时间顺序的前/后拍均值做绝对差，差异特征Mean+Max池化。",
             "- 样本门控将差异残差加到原始E22-C Mean+Max特征；残差投影零初始化，初始函数严格不变。",
             "- 保留原E12-A四分类和E11区域融合；无文本、年龄、性别；只做Validation，未运行Test。", "",
             "## 邻域审计", "", f"- Train完整前后邻域比例：{cache['splits']['train']['full_triplet_fraction']:.2%}；跨导联邻居：{cache['splits']['train']['cross_lead_neighbors']}。", "",
             "## Validation（三个seed）", "", "| 方法 | Overall R² | 患者三类均值 | OMI | NOMI | Control | Healthy | Macro AUC | Macro F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
             fmt("E12-A", base), fmt("E22-B写回上下文", summary['e22b_writeback_historical']['aggregate']),
             fmt("E22-C冻结下游", c), fmt("E22-D自适应差异微调", new), "",
             f"E22-D相对E22-C Overall R²：**{new['r2']['mean']-c['r2']['mean']:+.6f}**；相对E12-A：**{new['r2']['mean']-base['r2']['mean']:+.6f}**。",
             f"E22-D Validation门控均值：{new['gate_mean']['mean']:.6f}。", ""]
    (run / "README.md").write_text("\n".join(lines), encoding="utf-8")
    set_status(run, "complete"); (run / "COMPLETE").touch()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=["cache", "smoke", "downstream", "report"])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN); parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "cache": set_status(args.run_dir, "cache_penultimate"); cache_penultimate(args.run_dir, device, args.batch_size)
    elif args.stage == "smoke": set_status(args.run_dir, "smoke"); smoke(args.run_dir, device)
    elif args.stage == "downstream": set_status(args.run_dir, "downstream"); downstream(args.run_dir, device)
    else: set_status(args.run_dir, "report"); report(args.run_dir)


if __name__ == "__main__": main()
