#!/usr/bin/env python3
"""E19: hierarchical patient classification with training-only report concepts.

E19-A trains an ECG-only hierarchical head and the final E12-A Transformer block.
E19-B adds a masked multi-label auxiliary objective derived from ECG-observable
concepts in the row-composite Chinese report. Healthy records have no report and
never receive fabricated negative concept labels. Validation inference is ECG-only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
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


E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
E18B_RUN = ROOT / "results/e18b_downstream_text_auxiliary"
DEFAULT_RUN = ROOT / "results/e19_hierarchical_text_concepts"
CONCEPT_WEIGHT = 0.1
HEAD_LR = 5e-4
LAST_BLOCK_LR = 1e-5

# Explicit, frozen before looking at downstream validation. These are report
# concepts that can plausibly be expressed by ECG morphology/rhythm. They are
# not constructed from OMI/NOMI/Control labels.
CONCEPT_PATTERNS = {
    "sinus_rhythm": r"窦性",
    "st_elevation": r"ST[^。；,，]{0,10}(?:抬高|弓背|上抬)",
    "st_depression": r"ST[^。；,，]{0,10}(?:压低|下移|下降)",
    "st_t_change": r"ST[-－— ]?T|ST段改变|ST改变|复极",
    "q_wave": r"(?:异常|病理|坏死性)?Q波|QS波",
    "t_inversion_or_flat": r"T波[^。；,，]{0,8}(?:倒置|低平|改变|双向)",
    "af_or_flutter": r"房颤|心房颤动|房扑|心房扑动",
    "conduction_block": r"传导阻滞|束支阻滞|房室阻滞",
    "low_voltage": r"低电压|低电位",
    "premature_beat": r"早搏|期前收缩",
    "tachycardia": r"心动过速",
    "bradycardia": r"心动过缓",
    "axis_deviation": r"电轴[^。；,，]{0,10}(?:偏|转)",
    "qt_abnormal": r"QT[^。；,，]{0,10}(?:延长|缩短)|长QT|短QT",
    "mi_interpretation": r"心肌梗死|心梗",
    "ischemia_or_injury": r"心肌缺血|心肌损伤|缺血性改变",
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_status(run: Path, value: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "status.txt").write_text(value + "\n", encoding="utf-8")
    print(value, flush=True)


def build_concepts(run: Path) -> None:
    out = run / "concepts"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=False)
    pairs = pd.read_csv(E18B_RUN / "text_manifest/train_report_pairs.csv", low_memory=False)
    labels = np.load(e12.DOWNSTREAM / "train_labels.npy").argmax(1)
    n, c = len(labels), len(CONCEPT_PATTERNS)
    targets = np.zeros((n, c), dtype=np.float32)
    available = np.zeros(n, dtype=bool)
    names = list(CONCEPT_PATTERNS)
    compiled = [re.compile(CONCEPT_PATTERNS[name], re.IGNORECASE) for name in names]
    for row in pairs.itertuples(index=False):
        i = int(row.downstream_index)
        if labels[i] == 3:
            raise RuntimeError("Healthy record unexpectedly has a report")
        text = str(row.report_text)
        available[i] = True
        targets[i] = [float(bool(p.search(text))) for p in compiled]
    patient = labels != 3
    if not np.all(available[patient]) or np.any(available[~patient]):
        raise RuntimeError("Report availability does not exactly match Train patient status")
    np.save(out / "train_targets.npy", targets)
    np.save(out / "train_available.npy", available)
    rows = []
    for j, name in enumerate(names):
        item = {"concept": name, "pattern": CONCEPT_PATTERNS[name],
                "patient_positive": int(targets[patient, j].sum()),
                "patient_prevalence": float(targets[patient, j].mean())}
        for class_id, class_name in enumerate(e12.LABEL_NAMES[:3]):
            m = labels == class_id
            item[class_name] = int(targets[m, j].sum())
        rows.append(item)
    pd.DataFrame(rows).to_csv(out / "concept_prevalence.csv", index=False)
    dump(out / "audit.json", {
        "source": str(E18B_RUN / "text_manifest/train_report_pairs.csv"),
        "mapping": "same-row composite of diagnosis_1..6 assigned to every same-row ECG",
        "concepts": rows,
        "train_records": int(n), "patient_reports": int(available.sum()),
        "healthy_reports": int(available[labels == 3].sum()),
        "validation_test_reports_loaded": False,
        "healthy_policy": "concept loss masked out; healthy is not treated as all-negative",
        "label_policy": "patterns fixed from ECG-observable report terms; no OMI/NOMI/Control label used",
    })
    (out / "COMPLETE").touch()


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
    model = e12.load_continuous_checkpoint(E12_RUN / "mlm/scratch/checkpoint-best.pth", device)
    model.eval().requires_grad_(False)
    backbone = model.backbone
    summary = {"source_checkpoint": str(E12_RUN / "mlm/scratch/checkpoint-best.pth"),
               "cache_point": "input to E12-A Transformer block 12", "dtype": "float16", "splits": {}}
    for split in ("train", "val"):
        waves = np.load(e12.DOWNSTREAM / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(e12.DOWNSTREAM / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(e12.DOWNSTREAM / f"{split}_data_in_times.npy", mmap_mode="r")
        hidden = np.lib.format.open_memmap(out / f"{split}_prelast.npy", mode="w+", dtype=np.float16,
                                           shape=(len(waves), 257, 768))
        valid_out = np.lib.format.open_memmap(out / f"{split}_valid.npy", mode="w+", dtype=bool,
                                              shape=(len(waves), 257))
        for start in range(0, len(waves), batch_size):
            end = min(start + batch_size, len(waves))
            x0 = torch.from_numpy(np.asarray(waves[start:end], dtype=np.float32).copy()).to(device)
            ch = torch.from_numpy(np.asarray(chans[start:end]).copy()).long().to(device)
            ti = torch.from_numpy(np.asarray(times[start:end]).copy()).long().to(device)
            valid = tokenizer.valid_mask(x0, ch, ti)
            ze = tokenizer.encode_to_quantizer_features(x0, ch, ti).float()
            x = backbone.input_proj(ze) * valid.unsqueeze(-1)
            cls = backbone.cls_token.view(1, 1, -1).expand(len(x), 1, -1)
            x = torch.cat([cls, x], 1)
            full_valid = torch.cat([torch.ones(len(x), 1, dtype=torch.bool, device=device), valid], 1)
            x = backbone.dropout(x + backbone.pos_embed)
            for block in backbone.transformer.layers[:-1]:
                x = block_forward(block, x, full_valid)
            hidden[start:end] = x.half().cpu().numpy()
            valid_out[start:end] = full_valid.cpu().numpy()
            if start == 0 or end == len(waves) or start // batch_size % 50 == 0:
                print(f"cache {split} {end}/{len(waves)}", flush=True)
        hidden.flush(); valid_out.flush()
        summary["splits"][split] = {"records": int(len(waves)),
                                    "hidden_shape": [int(len(waves)), 257, 768]}
    last_state = {"block": copy.deepcopy(backbone.transformer.layers[-1].state_dict()),
                  "norm": copy.deepcopy(backbone.norm_layer.state_dict())}
    torch.save(last_state, out / "last_block_initial_state.pth")
    dump(out / "audit.json", summary)
    (out / "COMPLETE").touch()


class E19Model(nn.Module):
    def __init__(self, seed: int, initial_state: dict, concept_dim: int):
        super().__init__()
        template = e12.load_continuous_checkpoint(E12_RUN / "mlm/scratch/checkpoint-best.pth",
                                                   torch.device("cpu")).backbone
        self.last_block = copy.deepcopy(template.transformer.layers[-1])
        self.final_norm = copy.deepcopy(template.norm_layer)
        self.last_block.load_state_dict(initial_state["block"])
        self.final_norm.load_state_dict(initial_state["norm"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(190000 + int(seed))
            self.global_project = nn.Linear(128, 128)
            self.region_embedding = nn.Parameter(torch.empty(5, 128))
            nn.init.normal_(self.region_embedding, std=0.02)
            self.region_attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
            self.region_project = nn.Linear(128, 128)
            self.residual_scale = nn.Parameter(torch.tensor(0.0))
            self.shared = nn.Sequential(nn.Linear(1664, 128), nn.GELU(), nn.Dropout(0.1))
            self.patient_logit = nn.Linear(128, 1)
            self.subtype_logits = nn.Linear(128, 3)
            self.concept_logits = nn.Linear(128, concept_dim)

    def pool_regions(self, regions, region_mask):
        score = self.region_attention(regions + self.region_embedding.unsqueeze(0)).squeeze(-1)
        score = score.masked_fill(~region_mask, -1e9)
        weight = torch.softmax(score, 1) * region_mask.float()
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-12)
        return (weight.unsqueeze(-1) * regions).sum(1)

    def forward_parts(self, prelast, full_valid, global_ze, regions, region_mask):
        x = block_forward(self.last_block, prelast, full_valid)
        x = self.final_norm(x) * full_valid.unsqueeze(-1)
        token, valid = x[:, 1:], full_valid[:, 1:]
        count = valid.sum(1).clamp_min(1).unsqueeze(-1)
        mean = (token * valid.unsqueeze(-1)).sum(1) / count
        maximum = token.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(valid.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        mlm = torch.cat([mean, maximum], 1)
        region = self.pool_regions(regions, region_mask)
        fused = self.global_project(global_ze) + torch.tanh(self.residual_scale) * self.region_project(region)
        hidden = self.shared(torch.cat([mlm, fused], 1))
        return self.patient_logit(hidden).squeeze(-1), self.subtype_logits(hidden), self.concept_logits(hidden)

    @staticmethod
    def hierarchy_log_prob(patient_logit, subtype_logits):
        log_patient = F.logsigmoid(patient_logit)
        log_healthy = F.logsigmoid(-patient_logit)
        return torch.cat([log_patient[:, None] + F.log_softmax(subtype_logits, 1), log_healthy[:, None]], 1)

    def forward(self, prelast, full_valid, global_ze, regions, region_mask):
        p, s, _ = self.forward_parts(prelast, full_valid, global_ze, regions, region_mask)
        return self.hierarchy_log_prob(p, s)


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
    return data


class CacheArrays:
    def __init__(self, run: Path, split: str):
        self.hidden = np.load(run / f"penultimate_cache/{split}_prelast.npy", mmap_mode="r")
        self.valid = np.load(run / f"penultimate_cache/{split}_valid.npy", mmap_mode="r")

    def batch(self, idx: np.ndarray, device: torch.device):
        h = torch.from_numpy(np.asarray(self.hidden[idx], dtype=np.float32)).to(device)
        v = torch.from_numpy(np.asarray(self.valid[idx]).copy()).bool().to(device)
        return h, v


@torch.no_grad()
def evaluate(model, cache: CacheArrays, arrays: dict, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval(); chunks = []
    regions, region_mask = e12.e11.masked_region_means(arrays["lead"], arrays["lead_mask"])
    for start in range(0, len(arrays["labels"]), 64):
        idx = np.arange(start, min(start + 64, len(arrays["labels"])))
        h, v = cache.batch(idx, device)
        raw = model(h, v, arrays["global"][idx], regions[idx], region_mask[idx])
        chunks.append(raw.float().cpu().numpy())
    logits = np.concatenate(chunks)
    metrics, prob = e12.probability_metrics(logits, arrays["labels"].cpu().numpy())
    patient_mean = float(np.mean([metrics["per_output_r2"][name] for name in e12.LABEL_NAMES[:3]]))
    metrics["patient_mean_r2"] = patient_mean
    return metrics, prob, logits


def smoke(run: Path, device: torch.device) -> None:
    path = run / "smoke.json"
    if path.exists(): return
    build_concepts(run)
    if not (run / "penultimate_cache/COMPLETE").exists():
        raise RuntimeError("Penultimate cache must be complete before smoke")
    state = torch.load(run / "penultimate_cache/last_block_initial_state.pth", map_location="cpu", weights_only=False)
    model = E19Model(0, state, len(CONCEPT_PATTERNS)).to(device)
    data = load_small_data(device); cache = CacheArrays(run, "train")
    labels = data["train"]["labels"].argmax(1)
    patient = torch.nonzero(labels != 3, as_tuple=False).flatten()[:16].cpu().numpy()
    healthy = torch.nonzero(labels == 3, as_tuple=False).flatten()[:16].cpu().numpy()
    idx = np.concatenate([patient, healthy]); idx_t = torch.from_numpy(idx).long().to(device)
    regions, region_mask = e12.e11.masked_region_means(data["train"]["lead"][idx_t], data["train"]["lead_mask"][idx_t])
    h, v = cache.batch(idx, device)
    p, s, concepts = model.forward_parts(h, v, data["train"]["global"][idx_t], regions, region_mask)
    logp = model.hierarchy_log_prob(p, s)
    targets = torch.from_numpy(np.load(run / "concepts/train_targets.npy")[idx]).float().to(device)
    available = torch.from_numpy(np.load(run / "concepts/train_available.npy")[idx]).bool().to(device)
    loss = F.nll_loss(logp, labels[idx_t]) + CONCEPT_WEIGHT * F.binary_cross_entropy_with_logits(
        concepts[available], targets[available])
    loss.backward()
    grads = {"last_block": float(next(model.last_block.parameters()).grad.norm()),
             "hierarchy": float(model.subtype_logits.weight.grad.norm()),
             "concept": float(model.concept_logits.weight.grad.norm())}
    if not math.isfinite(float(loss)) or not all(math.isfinite(x) and x > 0 for x in grads.values()):
        raise RuntimeError(f"Smoke failed: loss={loss}, grads={grads}")
    prob = logp.exp()
    dump(path, {"loss_finite": True, "grad_norms": grads,
                "hierarchical_probability_sum_max_error": float((prob.sum(1) - 1).abs().max()),
                "patient_records_with_concepts": int(available.sum()),
                "healthy_records_with_concepts": int(available[16:].sum()),
                "validation_test_text_used": False})


def train_seed(run: Path, method: str, seed: int, device: torch.device) -> dict:
    out = run / "downstream" / method / f"seed{seed}"
    if (out / "VAL_COMPLETE").exists():
        return json.loads((out / "val_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    use_concepts = method == "e19b_hierarchical_concepts"
    state = torch.load(run / "penultimate_cache/last_block_initial_state.pth", map_location="cpu", weights_only=False)
    model = E19Model(seed, state, len(CONCEPT_PATTERNS)).to(device)
    last_params = list(model.last_block.parameters()) + list(model.final_norm.parameters())
    last_ids = {id(p) for p in last_params}
    head_params = [p for p in model.parameters() if id(p) not in last_ids]
    optimizer = torch.optim.AdamW([{"params": last_params, "lr": LAST_BLOCK_LR},
                                  {"params": head_params, "lr": HEAD_LR}], weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    data = load_small_data(device)
    train_cache, val_cache = CacheArrays(run, "train"), CacheArrays(run, "val")
    target = data["train"]["labels"].argmax(1)
    concept_target = torch.from_numpy(np.load(run / "concepts/train_targets.npy")).float().to(device)
    concept_available = torch.from_numpy(np.load(run / "concepts/train_available.npy")).bool().to(device)
    positive = concept_target[concept_available].sum(0)
    negative = concept_available.sum() - positive
    pos_weight = (negative / positive.clamp_min(1)).clamp(1, 10)
    regions, region_mask = e12.e11.masked_region_means(data["train"]["lead"], data["train"]["lead_mask"])
    rng = np.random.default_rng(seed)
    steps = len(target) // 64
    best, best_epoch, best_state, history = -float("inf"), None, None, []
    for epoch in range(1, 31):
        model.train(); order = rng.permutation(len(target)); sums = {"nll": 0., "concept": 0.}
        for step in range(steps):
            idx = order[step * 64:(step + 1) * 64]; idx_t = torch.from_numpy(idx).long().to(device)
            progress = ((epoch - 1) * steps + step) / max(1, 30 * steps - 1)
            warm = min(1.0, ((epoch - 1) * steps + step + 1) / max(1, 5 * steps))
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            for group, peak in zip(optimizer.param_groups, (LAST_BLOCK_LR, HEAD_LR)):
                group["lr"] = (1e-6 + (peak - 1e-6) * warm) if warm < 1 else 1e-6 + (peak - 1e-6) * cosine
            h, v = train_cache.batch(idx, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                p, s, concept = model.forward_parts(h, v, data["train"]["global"][idx_t],
                                                     regions[idx_t], region_mask[idx_t])
                logp = model.hierarchy_log_prob(p, s)
                nll = F.nll_loss(logp.float(), target[idx_t])
                available = concept_available[idx_t]
                concept_loss = F.binary_cross_entropy_with_logits(
                    concept[available].float(), concept_target[idx_t][available], pos_weight=pos_weight)
                loss = nll + (CONCEPT_WEIGHT * concept_loss if use_concepts else 0.0)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(optimizer); scaler.update()
            sums["nll"] += float(nll); sums["concept"] += float(concept_loss)
        val, _, _ = evaluate(model, val_cache, data["val"], device)
        row = {"epoch": epoch, "train_hierarchical_nll": sums["nll"] / steps,
               "train_concept_bce": sums["concept"] / steps, "val": val}
        history.append(row); print(json.dumps({"method": method, "seed": seed, **row}, ensure_ascii=False), flush=True)
        if val["r2"] > best:
            best, best_epoch = val["r2"], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    val, prob, logits = evaluate(model, val_cache, data["val"], device)
    torch.save({"model": best_state, "method": method, "seed": seed, "best_epoch": best_epoch,
                "selection": "Val Overall R2", "test_run": False}, out / "checkpoint-best.pth")
    np.save(out / "val_probabilities.npy", prob); np.save(out / "val_logits.npy", logits)
    dump(out / "history.json", history)
    result = {"method": method, "seed": seed, "best_epoch": best_epoch, "val": val,
              "test_run": False, "concept_weight": CONCEPT_WEIGHT if use_concepts else 0.0,
              "last_block_lr": LAST_BLOCK_LR, "head_lr": HEAD_LR}
    dump(out / "val_summary.json", result); (out / "VAL_COMPLETE").touch()
    return result


def aggregate(items: list[dict]) -> dict:
    keys = ("r2", "patient_mean_r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy")
    out = {}
    for key in keys:
        vals = [x["val"].get(key, np.mean([x["val"]["per_output_r2"][n] for n in e12.LABEL_NAMES[:3]])) for x in items]
        out[key] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    out["per_output_r2"] = {name: {"mean": float(np.mean([x["val"]["per_output_r2"][name] for x in items])),
                                           "sd": float(np.std([x["val"]["per_output_r2"][name] for x in items], ddof=1))}
                              for name in e12.LABEL_NAMES}
    return out


def train_all(run: Path, device: torch.device) -> None:
    path = run / "validation_summary.json"
    if path.exists(): return
    methods = {}
    for method in ("e19a_hierarchical", "e19b_hierarchical_concepts"):
        rows = [train_seed(run, method, seed, device) for seed in (0, 1, 2)]
        methods[method] = {"aggregate": aggregate(rows), "per_seed": rows}
    base = json.loads((E12_RUN / "downstream/validation_selection.json").read_text())["methods"]["e12a_region"]
    base_rows = base["per_seed"]
    for row in base_rows:
        row["val"]["patient_mean_r2"] = float(np.mean([row["val"]["per_output_r2"][n] for n in e12.LABEL_NAMES[:3]]))
    methods["e12a_historical"] = {"aggregate": aggregate(base_rows), "per_seed": base_rows}
    dump(path, {"validation_only": True, "test_run": False,
                "selection": "per-seed best Val Overall R2", "methods": methods})


def report(run: Path) -> None:
    summary = json.loads((run / "validation_summary.json").read_text())
    display = {"e12a_historical": "E12-A", "e19a_hierarchical": "E19-A",
               "e19b_hierarchical_concepts": "E19-B"}
    rows = []
    for method, block in summary["methods"].items():
        for item in block["per_seed"]:
            val = item["val"]
            rows.append({"method": display[method], "seed": item.get("seed", item.get("downstream_seed")),
                         "best_epoch": item.get("best_epoch"), "overall_r2": val["r2"],
                         "patient_mean_r2": val.get("patient_mean_r2", np.mean([val["per_output_r2"][n] for n in e12.LABEL_NAMES[:3]])),
                         **{f"{n.lower().replace(' ', '_')}_r2": val["per_output_r2"][n] for n in e12.LABEL_NAMES},
                         "macro_auc": val["macro_roc_auc"], "macro_f1": val["macro_f1"], "mse": val["mse"]})
    pd.DataFrame(rows).to_csv(run / "metrics_per_seed.csv", index=False)
    lines = ["# E19：患者内层级分类与结构化文本概念监督", "", "## 协议", "",
             "- E12-A、E3与历史实验只读；只微调E12-A最后一个Transformer Block及下游头。",
             "- E19-A：患者/Healthy层级分类 + 患者内部OMI/NOMI/Control分类。",
             "- E19-B：在E19-A上加入训练患者报告的16个ECG可观察概念，多标签权重0.1。",
             "- Healthy无报告，概念损失完全屏蔽；Validation推理只使用ECG。",
             "- 每个seed按Val Overall R²选checkpoint；未运行Test。", "", "## Validation", "",
             "| 方法 | Overall R² | Patient mean R² | OMI | NOMI | Control | Healthy | Macro AUC | Macro F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("e12a_historical", "e19a_hierarchical", "e19b_hierarchical_concepts"):
        a = summary["methods"][method]["aggregate"]
        lines.append(f"| {display[method]} | {a['r2']['mean']:.6f} ± {a['r2']['sd']:.6f} | {a['patient_mean_r2']['mean']:.6f} | " +
                     " | ".join(f"{a['per_output_r2'][n]['mean']:.6f}" for n in e12.LABEL_NAMES) +
                     f" | {a['macro_roc_auc']['mean']:.6f} | {a['macro_f1']['mean']:.6f} |")
    base = summary["methods"]["e12a_historical"]["aggregate"]
    b = summary["methods"]["e19b_hierarchical_concepts"]["aggregate"]
    lines += ["", f"E19-B相对E12-A Overall R²：**{b['r2']['mean']-base['r2']['mean']:+.6f}**。",
              f"患者三类平均R²差值：**{b['patient_mean_r2']['mean']-base['patient_mean_r2']['mean']:+.6f}**。", ""]
    (run / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "cache", "smoke", "train", "report"])
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args(); run = args.output_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "audit": set_status(run, "audit_report_concepts"); build_concepts(run)
    elif args.stage == "cache": set_status(run, "cache_e12_penultimate_features"); cache_penultimate(run, device, args.batch_size)
    elif args.stage == "smoke": set_status(run, "smoke_hierarchy_and_concept_gradients"); smoke(run, device)
    elif args.stage == "train": set_status(run, "train_e19a_e19b_seeds_0_1_2"); train_all(run, device)
    elif args.stage == "report": report(run); set_status(run, "complete")


if __name__ == "__main__":
    main()
