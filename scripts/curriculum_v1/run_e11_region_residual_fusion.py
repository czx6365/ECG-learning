#!/usr/bin/env python3
"""E11: region-aware residual fusion over frozen E3/E4 representations.

All representation models remain frozen.  E10 diagnostics use validation data
only.  New checkpoints are selected by validation four-output Overall R2 and
the test split is evaluated once after selection.
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
import torch
from torch import nn


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]
import run_e4_aggregation as e4
import run_e7_objective as e7
import run_e9_ze_fusion_r2 as e9
import run_e10_lead_aware_ze_fusion as e10


EXP = "e11_region_residual_fusion_20261018"
DEFAULT_RUN = ROOT / "results/curriculum_v1" / EXP
DATA, TOKENIZER, MLM, E4_FEATURES = e9.DATA, e9.TOKENIZER, e9.MLM, e9.E4_FEATURES
E9_RUN = ROOT / "results/curriculum_v1/e9_ze_fusion_r2_20261018"
E10_RUN = ROOT / "results/curriculum_v1/e10_lead_aware_ze_fusion_20261018"
LABEL_NAMES, LEAD_NAMES = e9.LABEL_NAMES, e10.LEAD_NAMES
REGION_NAMES = ["Anteroseptal_V1-V3", "Anterior_V3-V5", "Anterolateral_I-V5-V6", "Inferior_II-III", "Precordial_V1-V6"]
# I, II, III, V1, V2, V3, V4, V5, V6
REGION_MEMBERS = [
    [3, 4, 5],       # V1-V3
    [5, 6, 7],       # V3-V5
    [0, 7, 8],       # I, V5, V6
    [1, 2],          # II, III
    [3, 4, 5, 6, 7, 8],  # V1-V6
]
NEW_METHODS = ["e11_global_projection", "e11_residual_global_control", "e11_region_residual"]
BASELINES = ["e9_fusion", "e10a_repeat_control", "e10b_lead"]
ALL_METHODS = [*BASELINES, *NEW_METHODS]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_status(run_dir: Path, value: str) -> None:
    (run_dir / "status.txt").write_text(value, encoding="utf-8")
    print(value, flush=True)


def state_sha(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_features(device: torch.device) -> tuple[dict, dict]:
    """Load E10's already audited lead features and independently re-audit them."""
    data, audit = {}, {"source": str(E10_RUN / "features"), "splits": {}}
    for split in ("train", "val", "test"):
        lead_np = np.load(E10_RUN / "features" / f"{split}_lead_ze_mean_max.npy")
        mask_np = np.load(E10_RUN / "features" / f"{split}_lead_present.npy")
        counts_np = np.load(E10_RUN / "features" / f"{split}_lead_counts.npy")
        global_np = np.load(E10_RUN / "features" / f"{split}_global_ze_mean_max.npy")
        mlm_np = np.load(E4_FEATURES / f"{split}_heartlang_mean_max.npy", mmap_mode="r")
        labels_np = np.load(DATA / f"{split}_labels.npy").astype(np.float32)
        n = len(labels_np)
        if lead_np.shape != (n, 9, 128) or mask_np.shape != (n, 9) or counts_np.shape != (n, 9):
            raise RuntimeError(f"{split}: malformed E10 lead feature cache")
        if global_np.shape != (n, 128) or mlm_np.shape != (n, 1536):
            raise RuntimeError(f"{split}: malformed global/MLM cache")
        if not np.array_equal(mask_np, counts_np > 0):
            raise RuntimeError(f"{split}: mask/count mismatch")
        if np.any(lead_np[~mask_np] != 0):
            raise RuntimeError(f"{split}: missing lead slot is nonzero")
        weighted = (lead_np[:, :, :64] * counts_np[:, :, None]).sum(1) / counts_np.sum(1).clip(min=1)[:, None]
        maximum = np.where(mask_np[:, :, None], lead_np[:, :, 64:], -np.inf).max(1)
        maximum[~mask_np.any(1)] = 0
        reconstructed = np.concatenate([weighted, maximum], axis=1).astype(np.float32)
        diff = float(np.max(np.abs(reconstructed - global_np)))
        if not np.allclose(reconstructed, global_np, atol=2e-6, rtol=2e-5):
            raise RuntimeError(f"{split}: global reconstruction mismatch {diff}")
        e9_global = np.load(E9_RUN / "features" / f"{split}_ze_mean_max.npy", mmap_mode="r")
        e9_diff = float(np.max(np.abs(global_np - e9_global)))
        if not np.array_equal(global_np, e9_global):
            raise RuntimeError(f"{split}: E10/E9 global z_e mismatch {e9_diff}")
        data[split] = {
            "mlm": torch.from_numpy(np.asarray(mlm_np).copy()).float().to(device),
            "lead": torch.from_numpy(lead_np).float().to(device),
            "mask": torch.from_numpy(mask_np).bool().to(device),
            "global": torch.from_numpy(global_np).float().to(device),
            "labels": torch.from_numpy(labels_np).float().to(device),
        }
        audit["splits"][split] = {
            "records": n, "empty_ecg": int((~mask_np.any(1)).sum()),
            "lead_to_global_reconstruction_max_abs_diff": diff,
            "e10_global_vs_e9_global_max_abs_diff": e9_diff,
            "lead_feature_sha256": sha256(E10_RUN / "features" / f"{split}_lead_ze_mean_max.npy"),
            "mlm_feature_sha256": sha256(E4_FEATURES / f"{split}_heartlang_mean_max.npy"),
        }
    if [len(data[s]["labels"]) for s in ("train", "val", "test")] != [6640, 1344, 1366]:
        raise RuntimeError("Patient split counts changed")
    return data, audit


def masked_region_means(lead: torch.Tensor, lead_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    membership = torch.zeros(5, 9, dtype=torch.bool, device=lead.device)
    for r, members in enumerate(REGION_MEMBERS):
        membership[r, members] = True
    valid = lead_mask[:, None, :] & membership[None, :, :]
    count = valid.sum(2)
    regions = (lead[:, None, :, :] * valid.unsqueeze(-1)).sum(2) / count.clamp_min(1).unsqueeze(-1)
    regions = regions * (count > 0).unsqueeze(-1)
    return regions, count > 0


class E11GlobalProjection(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(93000 + int(seed))
            self.global_project = nn.Linear(128, 128)
        self.classifier = e9.DualHead(1664, seed)
        self.classifier.load_state_dict(e9.paired_initial_state(seed), strict=True)

    def encode(self, mlm, global_ze):
        return torch.cat([mlm, self.global_project(global_ze)], dim=1)

    def forward(self, mlm, global_ze):
        return self.classifier(self.encode(mlm, global_ze))

    def training_outputs(self, mlm, global_ze):
        return self.classifier.training_outputs(self.encode(mlm, global_ze))


class E11RegionResidual(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(93000 + int(seed))
            self.global_project = nn.Linear(128, 128)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(94000 + int(seed))
            self.region_embedding = nn.Parameter(torch.empty(5, 128))
            nn.init.normal_(self.region_embedding, std=0.02)
            self.region_attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
            self.region_project = nn.Linear(128, 128)
        # Exact global-projection starting point; the residual is introduced by training.
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self.classifier = e9.DualHead(1664, seed)
        self.classifier.load_state_dict(e9.paired_initial_state(seed), strict=True)

    def pool_regions(self, regions, region_mask):
        score = self.region_attention(regions + self.region_embedding.unsqueeze(0)).squeeze(-1)
        score = score.masked_fill(~region_mask, -1e9)
        weight = torch.softmax(score, dim=1) * region_mask.float()
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-12)
        pooled = (weight.unsqueeze(-1) * regions).sum(1)
        return pooled, weight

    def encode(self, mlm, global_ze, regions, region_mask):
        region, weight = self.pool_regions(regions, region_mask)
        alpha = torch.tanh(self.residual_scale)
        fused = self.global_project(global_ze) + alpha * self.region_project(region)
        return torch.cat([mlm, fused], dim=1), weight, alpha

    def forward(self, mlm, global_ze, regions, region_mask):
        x, _, _ = self.encode(mlm, global_ze, regions, region_mask)
        return self.classifier(x)

    def training_outputs(self, mlm, global_ze, regions, region_mask):
        x, _, _ = self.encode(mlm, global_ze, regions, region_mask)
        return self.classifier.training_outputs(x)


def build_model(method: str, seed: int, device: torch.device) -> nn.Module:
    model = E11GlobalProjection(seed) if method == "e11_global_projection" else E11RegionResidual(seed)
    e4.set_seed(seed)
    return model.to(device)


def region_inputs(method: str, arrays: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if method == "e11_global_projection":
        return None, None
    if method == "e11_region_residual":
        return masked_region_means(arrays["lead"], arrays["mask"])
    # Same architecture and initialization, but without regional differences.
    any_valid = arrays["mask"].any(1)
    regions = arrays["global"].unsqueeze(1).expand(-1, 5, -1) * any_valid[:, None, None]
    region_mask = any_valid[:, None].expand(-1, 5)
    return regions, region_mask


@torch.no_grad()
def evaluate(model, method: str, arrays: dict):
    model.eval()
    regions, region_mask = region_inputs(method, arrays)
    raw_chunks, weight_chunks = [], []
    for i in range(0, len(arrays["labels"]), 96):
        if isinstance(model, E11GlobalProjection):
            raw = model(arrays["mlm"][i:i + 96], arrays["global"][i:i + 96])
        else:
            x, weight, _ = model.encode(arrays["mlm"][i:i + 96], arrays["global"][i:i + 96],
                                        regions[i:i + 96], region_mask[i:i + 96])
            raw = model.classifier(x)
            weight_chunks.append(weight.cpu().numpy())
        raw_chunks.append(raw.cpu().numpy())
    raw = np.concatenate(raw_chunks)
    metrics, scores = e7.classification(raw, arrays["labels"].cpu().numpy(), "CE")
    weights = np.concatenate(weight_chunks) if weight_chunks else None
    return metrics, scores, raw, weights


def train_one(method, seed, data, run_dir, args, device):
    out = run_dir / "checkpoints" / method / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    model = build_model(method, seed, device)
    initial_sha = state_sha(model)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "initial_state.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    target = data["train"]["labels"].argmax(1)
    train_regions, train_region_mask = region_inputs(method, data["train"])
    steps = len(target) // args.batch_size
    generator = torch.Generator().manual_seed(seed)
    history, best_r2, best_epoch, best_state = [], -float("inf"), None, None
    first_grad = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=generator).to(device)
        main_losses, aux_losses = [], []
        for step in range(steps):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            lr = e4.lr_for_step((epoch - 1) * steps + step, args.epochs * steps,
                                args.warmup_epochs * steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, E11GlobalProjection):
                main, aux = model.training_outputs(data["train"]["mlm"][idx], data["train"]["global"][idx])
            else:
                main, aux = model.training_outputs(data["train"]["mlm"][idx], data["train"]["global"][idx],
                                                   train_regions[idx], train_region_mask[idx])
            loss_main = nn.functional.cross_entropy(main, target[idx])
            loss_aux = e9.patient_auxiliary_loss(aux, target[idx])
            loss = loss_main + 0.2 * loss_aux
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}/seed{seed}: non-finite loss")
            loss.backward()
            if epoch == 1 and step == 0:
                first_grad = {"global_project": float(model.global_project.weight.grad.norm()),
                              "classifier": float(model.classifier.shared[0].weight.grad.norm())}
                if isinstance(model, E11RegionResidual):
                    first_grad["residual_scale"] = float(model.residual_scale.grad.abs())
                    first_grad["region_project"] = 0.0 if model.region_project.weight.grad is None else float(model.region_project.weight.grad.norm())
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            main_losses.append(float(loss_main)); aux_losses.append(float(loss_aux))
        val, _, _, _ = evaluate(model, method, data["val"])
        row = {"epoch": epoch, "train_main_ce": float(np.mean(main_losses)),
               "train_patient_ce": float(np.mean(aux_losses)), "val": val}
        if isinstance(model, E11RegionResidual):
            row["residual_alpha_tanh"] = float(torch.tanh(model.residual_scale).detach())
        history.append(row)
        if val["r2"] > best_r2:
            best_r2, best_epoch, best_state = val["r2"], epoch, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{method}/seed{seed}: no checkpoint selected")
    model.load_state_dict(best_state)
    val, val_scores, val_raw, val_weights = evaluate(model, method, data["val"])
    # Only test evaluation for this run, after validation-only selection.
    test, test_scores, test_raw, test_weights = evaluate(model, method, data["test"])
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "best_epoch": best_epoch,
                "selection": "validation four-output uniform-average R2", "method": method, "seed": seed}, out / "checkpoint-best.pth")
    dump(out / "history.json", history)
    pred = run_dir / "predictions" / method / f"seed{seed}"
    pred.mkdir(parents=True, exist_ok=False)
    for split, raw, prob in (("val", val_raw, val_scores), ("test", test_raw, test_scores)):
        np.save(pred / f"{split}_logits.npy", raw); np.save(pred / f"{split}_probabilities.npy", prob)
    attention, alpha = None, None
    if isinstance(model, E11RegionResidual):
        np.save(pred / "val_region_attention.npy", val_weights)
        np.save(pred / "test_region_attention.npy", test_weights)
        alpha = float(torch.tanh(model.residual_scale).detach())
        true = data["test"]["labels"].argmax(1).cpu().numpy()
        attention = {"overall": {}, "by_true_class": {}}
        for j, name in enumerate(REGION_NAMES):
            attention["overall"][name] = float(test_weights[:, j].mean())
        for c, label in enumerate(LABEL_NAMES):
            attention["by_true_class"][label] = {name: float(test_weights[true == c, j].mean()) for j, name in enumerate(REGION_NAMES)}
    result = {
        "method": method, "seed": seed, "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "selection": "val Overall R2", "initial_state_sha256": initial_sha,
        "first_batch_gradient_norm": first_grad, "residual_alpha_tanh": alpha,
        "test_region_attention": attention, "val": val, "test": test,
    }
    dump(out / "summary.json", result)
    print(json.dumps({"method": method, "seed": seed, "best_epoch": best_epoch,
                      "val_r2": val["r2"], "test_r2": test["r2"], "alpha": alpha}), flush=True)
    return result


@torch.no_grad()
def analyze_e10b_validation(data: dict, run_dir: Path, device: torch.device) -> dict:
    """Class-conditional attention and lead removal on validation only."""
    val = data["val"]
    true = val["labels"].argmax(1).cpu().numpy()
    attention_sum = {label: {lead: [] for lead in LEAD_NAMES} for label in LABEL_NAMES}
    ablations = []
    per_seed_attention = []
    for seed in (0, 1, 2):
        model = e10.E10B(seed).to(device)
        ckpt = torch.load(E10_RUN / f"checkpoints/e10b_lead/seed{seed}/checkpoint-best.pth", map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        baseline, _, _, weights = e10.evaluate(model, "e10b_lead", val, val["labels"])
        seed_att = {"seed": seed, "by_true_class": {}}
        present = val["mask"].cpu().numpy()
        for c, label in enumerate(LABEL_NAMES):
            rows = true == c
            seed_att["by_true_class"][label] = {}
            for j, lead_name in enumerate(LEAD_NAMES):
                keep = rows & present[:, j]
                value = float(weights[keep, j].mean()) if keep.any() else float("nan")
                seed_att["by_true_class"][label][lead_name] = value
                attention_sum[label][lead_name].append(value)
        per_seed_attention.append(seed_att)
        for j, lead_name in enumerate(LEAD_NAMES):
            changed = dict(val)
            changed["lead"] = val["lead"].clone()
            changed["mask"] = val["mask"].clone()
            changed["lead"][:, j] = 0
            changed["mask"][:, j] = False
            metric, _, _, _ = e10.evaluate(model, "e10b_lead", changed, val["labels"])
            ablations.append({
                "seed": seed, "removed_lead": lead_name,
                "baseline_r2": baseline["r2"], "ablated_r2": metric["r2"],
                "delta_r2_ablated_minus_baseline": metric["r2"] - baseline["r2"],
                "baseline_macro_auc": baseline["macro_roc_auc"], "ablated_macro_auc": metric["macro_roc_auc"],
                "delta_auc_ablated_minus_baseline": metric["macro_roc_auc"] - baseline["macro_roc_auc"],
                "per_output_r2_delta": {k: metric["per_output_r2"][k] - baseline["per_output_r2"][k] for k in LABEL_NAMES},
            })
    attention_mean = {label: {lead: float(np.mean(vals)) for lead, vals in leads.items()} for label, leads in attention_sum.items()}
    ablation_aggregate = {}
    for lead in LEAD_NAMES:
        rows = [r for r in ablations if r["removed_lead"] == lead]
        ablation_aggregate[lead] = {
            "delta_r2_mean": float(np.mean([r["delta_r2_ablated_minus_baseline"] for r in rows])),
            "delta_r2_std_ddof1": float(np.std([r["delta_r2_ablated_minus_baseline"] for r in rows], ddof=1)),
            "delta_auc_mean": float(np.mean([r["delta_auc_ablated_minus_baseline"] for r in rows])),
            "per_output_r2_delta_mean": {k: float(np.mean([r["per_output_r2_delta"][k] for r in rows])) for k in LABEL_NAMES},
        }
    result = {
        "split": "validation only", "seeds": [0, 1, 2],
        "attention_by_true_class_mean_across_seeds": attention_mean,
        "attention_by_true_class_per_seed": per_seed_attention,
        "lead_removal_per_seed": ablations,
        "lead_removal_aggregate": ablation_aggregate,
        "interpretation_rule": "negative ablated-minus-baseline delta means the removed lead supported the metric",
    }
    dump(run_dir / "analysis/e10b_validation_diagnostics.json", result)
    with (run_dir / "analysis/e10b_attention_by_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["true_class", *LEAD_NAMES])
        for label in LABEL_NAMES:
            w.writerow([label, *[attention_mean[label][x] for x in LEAD_NAMES]])
    with (run_dir / "analysis/e10b_lead_removal.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["lead", "delta_r2_mean", "delta_r2_sd", "delta_auc_mean", *[f"delta_{x}_r2" for x in LABEL_NAMES]])
        for lead in LEAD_NAMES:
            x = ablation_aggregate[lead]
            w.writerow([lead, x["delta_r2_mean"], x["delta_r2_std_ddof1"], x["delta_auc_mean"], *[x["per_output_r2_delta_mean"][k] for k in LABEL_NAMES]])
    return result


def calibration_stats(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    result = {"macro_ece": 0.0, "per_class": {}}
    for c, label in enumerate(LABEL_NAMES):
        p, t = prob[:, c], y[:, c]
        ece = 0.0
        for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
            m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
            if m.any():
                ece += float(m.mean()) * abs(float(p[m].mean()) - float(t[m].mean()))
        result["per_class"][label] = {
            "prevalence": float(t.mean()), "mean_probability": float(p.mean()),
            "mean_bias": float((p - t).mean()), "probability_std": float(p.std()), "ece10": ece,
        }
    result["macro_ece"] = float(np.mean([v["ece10"] for v in result["per_class"].values()]))
    return result


def analyze_e10a_auc_vs_r2(run_dir: Path) -> dict:
    summary = json.loads((E10_RUN / "metrics/aggregate_summary.json").read_text())
    labels = np.load(DATA / "test_labels.npy")
    methods = ["e9_fusion", "e10a_repeat_control", "e10a_lead"]
    output = {"split": "existing completed E9/E10 test predictions; descriptive only", "methods": {}}
    for method in methods:
        stats = []
        source = E9_RUN if method == "e9_fusion" else E10_RUN
        for seed in (0, 1, 2):
            prob = np.load(source / f"predictions/{method}/seed{seed}/test_probabilities.npy")
            stats.append(calibration_stats(prob, labels))
        block = summary["methods"][method] if method in summary["methods"] else json.loads((E9_RUN / "metrics/aggregate_summary.json").read_text())["methods"][method]
        output["methods"][method] = {
            "test_r2": block["test"]["r2"], "test_macro_auc": block["test"]["macro_roc_auc"],
            "test_per_output_r2": block["test"]["per_output_r2"],
            "macro_ece10_mean": float(np.mean([x["macro_ece"] for x in stats])),
            "calibration_per_seed": stats,
        }
    a, b = output["methods"]["e9_fusion"], output["methods"]["e10a_lead"]
    output["e10a_lead_minus_e9"] = {
        "overall_r2": b["test_r2"]["mean"] - a["test_r2"]["mean"],
        "macro_auc": b["test_macro_auc"]["mean"] - a["test_macro_auc"]["mean"],
        "macro_ece10": b["macro_ece10_mean"] - a["macro_ece10_mean"],
        "per_output_r2": {k: b["test_per_output_r2"][k]["mean"] - a["test_per_output_r2"][k]["mean"] for k in LABEL_NAMES},
    }
    output["explanation"] = "AUC measures ranking and is invariant to monotonic score scaling; probability R2 penalizes magnitude/calibration error. E10-A gains on NOMI/Healthy are offset by OMI/Control R2 losses."
    dump(run_dir / "analysis/e10a_auc_vs_r2.json", output)
    return output


def load_baselines() -> list[dict]:
    e9_summary = json.loads((E9_RUN / "metrics/aggregate_summary.json").read_text())
    e10_summary = json.loads((E10_RUN / "metrics/aggregate_summary.json").read_text())
    rows = []
    for r in e9_summary["per_seed"]:
        if r["method"] == "e9_fusion":
            x = copy.deepcopy(r); x["reused_from"] = str(E9_RUN); rows.append(x)
    for r in e10_summary["per_seed"]:
        if r["method"] in ("e10a_repeat_control", "e10b_lead"):
            x = copy.deepcopy(r); x["reused_from"] = str(E10_RUN); rows.append(x)
    for method in BASELINES:
        if len([r for r in rows if r["method"] == method]) != 3:
            raise RuntimeError(f"Missing baseline {method} seeds")
    return rows


def aggregate(results: list[dict]) -> dict:
    output = {"per_seed": results, "methods": {}}
    for method in ALL_METHODS:
        rows = [r for r in results if r["method"] == method]
        if len(rows) != 3:
            raise RuntimeError(f"{method}: expected 3 seeds")
        block = {"trainable_parameters": rows[0]["trainable_parameters"]}
        for split in ("val", "test"):
            block[split] = {}
            for key in ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy"):
                vals = np.array([r[split][key] for r in rows], dtype=float)
                block[split][key] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
            block[split]["per_output_r2"] = {}
            for label in LABEL_NAMES:
                vals = np.array([r[split]["per_output_r2"][label] for r in rows], dtype=float)
                block[split]["per_output_r2"][label] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
        output["methods"][method] = block
    mean = lambda m: output["methods"][m]["test"]["r2"]["mean"]
    output["contrasts"] = {
        "global_projection_minus_e9": mean("e11_global_projection") - mean("e9_fusion"),
        "global_projection_minus_e10a_repeat_control": mean("e11_global_projection") - mean("e10a_repeat_control"),
        "region_residual_minus_e9": mean("e11_region_residual") - mean("e9_fusion"),
        "region_residual_minus_e10b": mean("e11_region_residual") - mean("e10b_lead"),
        "region_residual_minus_same_parameter_global_control": mean("e11_region_residual") - mean("e11_residual_global_control"),
    }
    return output


def write_csv(run_dir: Path, results: list[dict], summary: dict) -> None:
    metrics = run_dir / "metrics"; metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "results_by_seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["method", "seed", "parameters", "best_epoch", "split", "overall_r2", *[f"{x}_r2" for x in LABEL_NAMES], "mse", "mae", "macro_auc", "macro_f1", "accuracy", "residual_alpha"])
        for r in results:
            for split in ("val", "test"):
                m = r[split]
                w.writerow([r["method"], r["seed"], r["trainable_parameters"], r["best_epoch"], split, m["r2"],
                            *[m["per_output_r2"][x] for x in LABEL_NAMES], m["mse"], m["mae"], m["macro_roc_auc"],
                            m["macro_f1"], m["accuracy"], r.get("residual_alpha_tanh")])
    with (metrics / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["method", "parameters", "test_r2_mean", "test_r2_sd", "macro_auc_mean", "macro_auc_sd"])
        for m in ALL_METHODS:
            b = summary["methods"][m]
            w.writerow([m, b["trainable_parameters"], b["test"]["r2"]["mean"], b["test"]["r2"]["std_ddof1"],
                        b["test"]["macro_roc_auc"]["mean"], b["test"]["macro_roc_auc"]["std_ddof1"]])


def write_report(run_dir: Path, summary: dict, results: list[dict], diagnostics: dict, auc_analysis: dict) -> None:
    names = {
        "e9_fusion": "E9 全局 z_e", "e10a_repeat_control": "E10-A 全局重复对照",
        "e10b_lead": "E10-B 导联Attention", "e11_global_projection": "E11 全局投影",
        "e11_residual_global_control": "E11 等参数全局残差对照", "e11_region_residual": "E11 区域残差融合",
    }
    c = summary["contrasts"]
    best = max(ALL_METHODS, key=lambda m: summary["methods"][m]["test"]["r2"]["mean"])
    lines = ["# E11：Region-aware Residual Fusion", "", "## 主要结论", "",
             f"六种方法中三种子 Test Overall R² 最高的是 **{names[best]}**（{summary['methods'][best]['test']['r2']['mean']:.6f}）。",
             f"E11 区域残差相对 E9 为 **{c['region_residual_minus_e9']:+.6f}**，相对 E10-B 为 **{c['region_residual_minus_e10b']:+.6f}**，相对完全相同参数量的全局残差对照为 **{c['region_residual_minus_same_parameter_global_control']:+.6f}**。",
             f"直接全局 128→128 投影相对 E9 为 **{c['global_projection_minus_e9']:+.6f}**，相对 E10-A 全局重复对照为 **{c['global_projection_minus_e10a_repeat_control']:+.6f}**。", "",
             "## E10-B 验证集诊断", "", "### 按真实类别的平均导联 Attention", "",
             "| true class | " + " | ".join(LEAD_NAMES) + " |", "|---|" + "---:|" * 9]
    att = diagnostics["attention_by_true_class_mean_across_seeds"]
    for label in LABEL_NAMES:
        lines.append("| " + label + " | " + " | ".join(f"{att[label][lead]:.4f}" for lead in LEAD_NAMES) + " |")
    lines += ["", "### 单导联移除（Val；负值表示移除后变差）", "",
              "| lead removed | ΔOverall R² | sd | ΔMacro AUC |", "|---|---:|---:|---:|"]
    abl = diagnostics["lead_removal_aggregate"]
    for lead in sorted(LEAD_NAMES, key=lambda x: abl[x]["delta_r2_mean"]):
        x = abl[lead]
        lines.append(f"| {lead} | {x['delta_r2_mean']:+.6f} | {x['delta_r2_std_ddof1']:.6f} | {x['delta_auc_mean']:+.6f} |")
    d = auc_analysis["e10a_lead_minus_e9"]
    lines += ["", "## E10-A：为什么 AUC 提升而 R² 未提升", "",
              f"E10-A 相对 E9 的 Macro AUC 为 **{d['macro_auc']:+.6f}**，但 Overall R² 为 **{d['overall_r2']:+.6f}**。AUC只评价排序，对单调缩放不敏感；R²使用Softmax概率与one-hot标签的平方误差，同时要求概率幅度准确。",
              "E10-A 的类别R²变化为：" + "，".join(f"{k} {v:+.6f}" for k, v in d["per_output_r2"].items()) + "。NOMI与Healthy的收益被OMI与Clinical Control下降抵消。",
              f"10-bin macro ECE 相对 E9 的差值为 {d['macro_ece10']:+.6f}；该值仅作为概率校准的描述性检查。", "",
              "## E11 架构", "",
              "区域定义严格使用现有9导联：前间壁V1–V3、前壁V3–V5、前侧壁I/V5/V6、下壁II/III、整体胸前V1–V6。重叠导联分别参与对应区域；缺失导联从区域均值和Attention中屏蔽；空ECG区域分支为零。未使用aVL、aVF或V7–V9。",
              "区域Attention得到128维表示；最终连续分支为 `GlobalProject(z_e_global) + tanh(alpha) × RegionProject(region)`，与MLM 1536维拼接进入原E8双头分类器。alpha从0开始，因此初始状态严格等于全局投影基线。", "",
              "## 三种子总体结果", "", "| 方法 | 参数 | Test R² mean±sd | Macro AUC mean±sd | ΔR² vs E9 |", "|---|---:|---:|---:|---:|"]
    e9mean = summary["methods"]["e9_fusion"]["test"]["r2"]["mean"]
    for m in ALL_METHODS:
        b = summary["methods"][m]
        lines.append(f"| {names[m]} | {b['trainable_parameters']} | {b['test']['r2']['mean']:.6f} ± {b['test']['r2']['std_ddof1']:.6f} | {b['test']['macro_roc_auc']['mean']:.6f} ± {b['test']['macro_roc_auc']['std_ddof1']:.6f} | {b['test']['r2']['mean']-e9mean:+.6f} |")
    lines += ["", "## Test 四类别 R²", "", "| 方法 | OMI | NOMI | Clinical Control | Healthy | Overall |", "|---|---:|---:|---:|---:|---:|"]
    for m in ALL_METHODS:
        b = summary["methods"][m]["test"]
        lines.append("| " + names[m] + " | " + " | ".join([f"{b['per_output_r2'][x]['mean']:.6f}" for x in LABEL_NAMES] + [f"{b['r2']['mean']:.6f}"]) + " |")
    lines += ["", "## 新实验每个 seed", "", "| 方法 | seed | best epoch | Val R² | Test R² | Macro AUC | residual alpha |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in [x for x in results if x["method"] in NEW_METHODS]:
        alpha = r.get("residual_alpha_tanh")
        lines.append(f"| {names[r['method']]} | {r['seed']} | {r['best_epoch']} | {r['val']['r2']:.6f} | {r['test']['r2']:.6f} | {r['test']['macro_roc_auc']:.6f} | {'—' if alpha is None else f'{alpha:.6f}'} |")
    lines += ["", "## 实验纪律", "", "- 固定患者级6640/1344/1366；冻结MorphologyEncoder、Codebook、Sentence MLM；复用E10已审计导联特征。",
              "- seed 0/1/2，30 epoch，batch64，AdamW峰值5e-4，5轮warmup，cosine到1e-5，weight decay0.05，clip3。",
              "- 损失为四分类CE + 0.2×患者三分类CE；按Val Overall R²选模；恢复最佳checkpoint后test只评估一次。",
              "- E10诊断和导联移除只使用验证集；测试集没有用于架构选择或调参。", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def smoke(run_dir: Path, device: torch.device) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    e4.set_seed(0)
    mlm, global_ze = torch.randn(6, 1536, device=device), torch.randn(6, 128, device=device)
    lead = torch.randn(6, 9, 128, device=device)
    mask = torch.tensor([[1] * 9, [1, 0, 1, 0, 1, 0, 1, 0, 1], [0] * 9, [1] * 8 + [0], [0, 1, 0, 1, 0, 1, 0, 1, 0], [1] * 9], dtype=torch.bool, device=device)
    lead *= mask.unsqueeze(-1); global_ze[2] = 0
    arrays = {"mlm": mlm, "global": global_ze, "lead": lead, "mask": mask,
              "labels": torch.eye(4, device=device)[torch.tensor([0, 1, 2, 3, 0, 1], device=device)]}
    regions, region_mask = masked_region_means(lead, mask)
    if not torch.equal(regions[2], torch.zeros_like(regions[2])) or region_mask[2].any():
        raise RuntimeError("Empty ECG region handling failed")
    control = E11RegionResidual(0).to(device); actual = E11RegionResidual(0).to(device)
    if state_sha(control) != state_sha(actual):
        raise RuntimeError("Residual same-parameter control init mismatch")
    gp = E11GlobalProjection(0).to(device)
    # Shared global path/classifier must start exactly equal.
    if not all(torch.equal(gp.state_dict()[f"global_project.{k}"], actual.state_dict()[f"global_project.{k}"]) for k in ("weight", "bias")):
        raise RuntimeError("Global projection init mismatch")
    for k, v in gp.classifier.state_dict().items():
        if not torch.equal(v, actual.classifier.state_dict()[k]):
            raise RuntimeError("Classifier init mismatch")
    gp.eval(); actual.eval()
    with torch.no_grad():
        g_logits = gp(mlm, global_ze)
        r_logits = actual(mlm, global_ze, regions, region_mask)
        if not torch.equal(g_logits, r_logits):
            raise RuntimeError("Zero residual does not exactly reproduce global projection")
        pooled, weights = actual.pool_regions(regions, region_mask)
        if not torch.isfinite(weights).all() or not torch.equal(pooled[2], torch.zeros_like(pooled[2])):
            raise RuntimeError("Region attention mask failed")
    main, aux = actual.training_outputs(mlm, global_ze, regions, region_mask)
    target = arrays["labels"].argmax(1)
    loss = nn.functional.cross_entropy(main, target) + 0.2 * e9.patient_auxiliary_loss(aux, target)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke loss non-finite")
    dump(run_dir / "smoke.json", {"passed": True, "region_names": REGION_NAMES,
                                   "region_members": [[LEAD_NAMES[j] for j in x] for x in REGION_MEMBERS],
                                   "zero_residual_exactly_matches_global_projection": True,
                                   "empty_ecg_and_missing_mask_passed": True,
                                   "global_projection_parameters": sum(p.numel() for p in gp.parameters()),
                                   "region_residual_parameters": sum(p.numel() for p in actual.parameters())})
    (run_dir / "COMPLETE").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=3.0)
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_only:
        smoke(args.output_dir, device); return
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "pipeline.pid").write_text(str(os.getpid()))
    start = time.time()
    protocol = {
        "experiment": EXP, "baselines": BASELINES, "new_methods": NEW_METHODS, "seeds": [0, 1, 2],
        "regions": {name: [LEAD_NAMES[j] for j in members] for name, members in zip(REGION_NAMES, REGION_MEMBERS)},
        "region_overlap": "intentional; each lead contributes independently to every listed region",
        "missing_policy": "exclude missing leads; mask empty regions; all-zero output for empty ECG",
        "e11_residual": "global_project(global_z_e) + tanh(alpha)*region_project(attention(region means)); alpha init 0",
        "same_parameter_control": "five copies of global z_e through exact same residual model and initialization",
        "primary_metric": "softmax vs one-hot uniform-average four-output R2", "selection": "Val Overall R2 only",
        "test_policy": "once after selection for each new method/seed", "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs, "weight_decay": args.weight_decay, "clip": args.clip,
        "tokenizer": str(TOKENIZER), "sentence_mlm": str(MLM), "e10_features": str(E10_RUN / "features"),
        "frozen_tokenizer_codebook_mlm": True,
    }
    dump(run_dir / "protocol.json", protocol)
    sources = [TOKENIZER, MLM, ROOT / "scripts/curriculum_v1/run_e9_ze_fusion_r2.py",
               ROOT / "scripts/curriculum_v1/run_e10_lead_aware_ze_fusion.py", Path(__file__)]
    source_hashes = {str(p): sha256(p) for p in sources}
    dump(run_dir / "source_fingerprints.json", source_hashes)
    try:
        set_status(run_dir, "loading_and_auditing_frozen_features")
        data, feature_audit = load_features(device)
        dump(run_dir / "feature_audit.json", feature_audit)
        set_status(run_dir, "analyzing_e10b_validation_attention_and_ablation")
        diagnostics = analyze_e10b_validation(data, run_dir, device)
        set_status(run_dir, "analyzing_e10a_auc_vs_r2")
        auc_analysis = analyze_e10a_auc_vs_r2(run_dir)

        # Pair audits: residual actual/control exact; global path matches standalone.
        pair_audit = {}
        for seed in (0, 1, 2):
            a = build_model("e11_region_residual", seed, device)
            b = build_model("e11_residual_global_control", seed, device)
            g = build_model("e11_global_projection", seed, device)
            if state_sha(a) != state_sha(b):
                raise RuntimeError(f"seed{seed}: residual control initial state mismatch")
            for key, value in g.global_project.state_dict().items():
                if not torch.equal(value, a.global_project.state_dict()[key]):
                    raise RuntimeError(f"seed{seed}: global projection init mismatch")
            for key, value in g.classifier.state_dict().items():
                if not torch.equal(value, a.classifier.state_dict()[key]):
                    raise RuntimeError(f"seed{seed}: classifier init mismatch")
            pair_audit[f"seed{seed}"] = {"residual_state_sha256": state_sha(a),
                                         "global_projection_parameters": sum(p.numel() for p in g.parameters()),
                                         "residual_parameters": sum(p.numel() for p in a.parameters())}
            del a, b, g
        dump(run_dir / "paired_control_audit.json", pair_audit)

        results = load_baselines()
        dump(run_dir / "partial_results.json", results)
        for method in NEW_METHODS:
            for seed in (0, 1, 2):
                set_status(run_dir, f"training_{method}_seed{seed}")
                results.append(train_one(method, seed, data, run_dir, args, device))
                dump(run_dir / "partial_results.json", results)
        summary = aggregate(results)
        dump(run_dir / "metrics/aggregate_summary.json", summary)
        write_csv(run_dir, results, summary)
        write_report(run_dir, summary, results, diagnostics, auc_analysis)
        if any(sha256(p) != source_hashes[str(p)] for p in sources):
            raise RuntimeError("Frozen source changed during E11")
        dump(run_dir / "completion_audit.json", {"source_hashes_unchanged": True,
                                                   "split_counts": {s: len(data[s]["labels"]) for s in data},
                                                   "new_runs": 9, "reused_baseline_runs": 9,
                                                   "e10_diagnostics_split": "validation", "test_evaluations_per_new_run": 1,
                                                   "selection_uses_test": False, "elapsed_seconds": time.time() - start})
        set_status(run_dir, "complete")
        (run_dir / "exit_code.txt").write_text("0")
        (run_dir / "COMPLETE").touch()
    except BaseException:
        set_status(run_dir, "failed")
        (run_dir / "exit_code.txt").write_text("1")
        raise


if __name__ == "__main__":
    main()
