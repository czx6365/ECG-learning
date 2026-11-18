#!/usr/bin/env python3
"""E23: patient-only age auxiliary supervision on the E12-A downstream head.

The ECG inference path is exactly E12-A Mean+Max plus the E11 regional residual
classifier. Age is never an input. During downstream training only, a small
regression head predicts standardized age for patient ECGs. Healthy ECGs have
no fabricated age target, and Test is never loaded or evaluated.
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
from torch import nn


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

import run_e12_continuous_input as e12


E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
DEFAULT_RUN = ROOT / "results/e23_downstream_age_auxiliary"
AGE_WEIGHT = 0.1


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_status(run: Path, value: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "status.txt").write_text(value + "\n", encoding="utf-8")
    print(value, flush=True)


def load_ecg_data(device: torch.device) -> dict:
    """Load Train/Val ECG features only. Test is intentionally untouched."""
    data = {}
    for split in ("train", "val"):
        labels = np.load(e12.DOWNSTREAM / f"{split}_labels.npy").astype(np.float32)
        mlm = np.load(E12_RUN / f"features/scratch/{split}_heartlang_mean_max.npy")
        lead = np.load(e12.E10_RUN / f"features/{split}_lead_ze_mean_max.npy")
        lead_mask = np.load(e12.E10_RUN / f"features/{split}_lead_present.npy")
        global_ze = np.load(e12.E10_RUN / f"features/{split}_global_ze_mean_max.npy")
        if len({len(labels), len(mlm), len(lead), len(lead_mask), len(global_ze)}) != 1:
            raise RuntimeError(f"{split}: feature alignment mismatch")
        data[split] = {
            "labels": torch.from_numpy(labels).to(device),
            "mlm": torch.from_numpy(mlm).float().to(device),
            "lead": torch.from_numpy(lead).float().to(device),
            "lead_mask": torch.from_numpy(lead_mask).bool().to(device),
            "global": torch.from_numpy(global_ze).float().to(device),
        }
    return data


def build_age_targets(run: Path) -> None:
    out = run / "age"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=False)
    frames = {}
    raw = {}
    masks = {}
    audit = {
        "experiment": "E23 downstream patient-only age auxiliary",
        "age_source": "frozen downstream metadata column 年龄",
        "age_is_model_input": False,
        "healthy_policy": "missing age; auxiliary loss masked; no fabricated target",
        "training_scope": "downstream Train patients only",
        "validation_policy": "classification checkpoint selected only by Val Overall R2; age used only as a post-selection diagnostic",
        "test_policy": "Test metadata/features/labels are not loaded or evaluated",
        "splits": {},
    }
    for split in ("train", "val"):
        frame = pd.read_csv(e12.DOWNSTREAM / f"{split}_metadata.csv", low_memory=False)
        labels = np.load(e12.DOWNSTREAM / f"{split}_labels.npy").argmax(1)
        if len(frame) != len(labels):
            raise RuntimeError(f"{split}: metadata/label length mismatch")
        patient = labels != 3
        cohort_patient = frame["cohort"].astype(str).eq("patient").to_numpy()
        if not np.array_equal(patient, cohort_patient):
            raise RuntimeError(f"{split}: patient cohort disagrees with class label")
        age = pd.to_numeric(frame["年龄"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(age[patient]).all() or np.isfinite(age[~patient]).any():
            raise RuntimeError(f"{split}: age presence does not exactly match patient status")
        if ((age[patient] < 18) | (age[patient] > 110)).any():
            raise RuntimeError(f"{split}: implausible patient age")
        per_patient = frame.loc[patient, ["patient_sn", "年龄"]].copy()
        per_patient["年龄"] = pd.to_numeric(per_patient["年龄"], errors="raise")
        grouped_age = per_patient.groupby("patient_sn")["年龄"].agg(["count", "nunique", "min", "max"])
        conflicting = grouped_age[grouped_age["nunique"] > 1]
        frames[split] = frame
        raw[split] = age
        masks[split] = patient
        by_class = {}
        for class_id, name in enumerate(e12.LABEL_NAMES):
            m = labels == class_id
            values = age[m & patient]
            by_class[name] = {
                "records": int(m.sum()), "age_present": int(np.isfinite(age[m]).sum()),
                "mean": None if not len(values) else float(values.mean()),
                "sd": None if not len(values) else float(values.std()),
                "min": None if not len(values) else float(values.min()),
                "max": None if not len(values) else float(values.max()),
            }
        audit["splits"][split] = {
            "records": int(len(frame)), "patient_records": int(patient.sum()),
            "healthy_records": int((~patient).sum()),
            "unique_patients": int(frame.loc[patient, "patient_sn"].nunique()),
            "patient_age_present": int(np.isfinite(age[patient]).sum()),
            "healthy_age_present": int(np.isfinite(age[~patient]).sum()),
            "repeated_patients_with_multiple_recorded_ages": int(len(conflicting)),
            "maximum_within_patient_age_span_years": float(
                (conflicting["max"] - conflicting["min"]).max()) if len(conflicting) else 0.0,
            "by_class": by_class,
        }

    train_age = raw["train"][masks["train"]]
    mean, sd = float(train_age.mean()), float(train_age.std(ddof=0))
    if not (math.isfinite(mean) and math.isfinite(sd) and sd > 0):
        raise RuntimeError("invalid Train patient age normalization")
    audit["normalization"] = {
        "basis": "Train patient ECG records; row-specific ages retained for longitudinal records",
        "n": int(len(train_age)),
        "mean_years": mean, "sd_years": sd,
    }
    for split in ("train", "val"):
        years = raw[split].astype(np.float32)
        mask = masks[split]
        z = np.zeros(len(years), dtype=np.float32)
        z[mask] = ((years[mask] - mean) / sd).astype(np.float32)
        np.save(out / f"{split}_age_years.npy", years)
        np.save(out / f"{split}_age_z.npy", z)
        np.save(out / f"{split}_age_mask.npy", mask.astype(bool))
    dump(out / "audit.json", audit)
    (out / "COMPLETE").touch()


class AgeAuxE11(e12.e11.E11RegionResidual):
    def __init__(self, seed: int):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(230000 + int(seed))
            self.age_head = nn.Linear(128, 1)

    def training_outputs_with_age(self, mlm, global_ze, regions, region_mask):
        x, _, _ = self.encode(mlm, global_ze, regions, region_mask)
        hidden = self.classifier.shared(x)
        return self.classifier.main(hidden), self.classifier.aux(hidden), self.age_head(hidden).squeeze(-1)

    def predict_age_z(self, mlm, global_ze, regions, region_mask):
        x, _, _ = self.encode(mlm, global_ze, regions, region_mask)
        return self.age_head(self.classifier.shared(x)).squeeze(-1)


def age_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return F.smooth_l1_loss(prediction[mask].float(), target[mask].float(), beta=1.0)
    return prediction.sum() * 0.0


def smoke(run: Path, device: torch.device) -> None:
    path = run / "smoke.json"
    if path.exists():
        return
    build_age_targets(run)
    data = load_ecg_data(device)
    mask_np = np.load(run / "age/train_age_mask.npy")
    age_np = np.load(run / "age/train_age_z.npy")
    patient = np.flatnonzero(mask_np)[:32]
    healthy = np.flatnonzero(~mask_np)[:32]
    idx_np = np.concatenate([patient, healthy])
    idx = torch.from_numpy(idx_np).long().to(device)
    age = torch.from_numpy(age_np[idx_np]).float().to(device)
    age_mask = torch.from_numpy(mask_np[idx_np]).bool().to(device)
    model = AgeAuxE11(0).to(device)
    baseline = e12.e11.E11RegionResidual(0).to(device)
    base_state, state = baseline.state_dict(), model.state_dict()
    common_diff = max(float((state[k] - value).abs().max()) for k, value in base_state.items())
    regions, region_mask = e12.e11.masked_region_means(data["train"]["lead"][idx], data["train"]["lead_mask"][idx])
    main, aux, predicted_age = model.training_outputs_with_age(
        data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask)
    loss = age_loss(predicted_age, age, age_mask)
    loss.backward()
    grads = {
        "shared": float(model.classifier.shared[0].weight.grad.norm()),
        "age_head": float(model.age_head.weight.grad.norm()),
    }
    model.eval(); baseline.eval()
    with torch.no_grad():
        base_logits = baseline(data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask)
        model_logits = model(data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask)
    result = {
        "common_initialization_max_abs_diff": common_diff,
        "initial_inference_logits_max_abs_diff": float((base_logits - model_logits).abs().max()),
        "age_only_gradient_norms": grads,
        "patient_age_targets": int(age_mask[:len(patient)].sum()),
        "healthy_age_targets": int(age_mask[len(patient):].sum()),
        "age_loss_finite": bool(torch.isfinite(loss)),
        "age_is_inference_input": False, "test_loaded": False,
    }
    if (not result["age_loss_finite"] or common_diff != 0
            or result["initial_inference_logits_max_abs_diff"] != 0
            or result["healthy_age_targets"] != 0
            or not all(math.isfinite(x) and x > 0 for x in grads.values())):
        raise RuntimeError(f"Smoke failed: {result}")
    dump(path, result)


def aggregate(items: list[dict]) -> dict:
    keys = ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy")
    result = {key: {"mean": float(np.mean([row["val"][key] for row in items])),
                    "sd": float(np.std([row["val"][key] for row in items], ddof=1))} for key in keys}
    result["per_output_r2"] = {
        name: {"mean": float(np.mean([row["val"]["per_output_r2"][name] for row in items])),
               "sd": float(np.std([row["val"]["per_output_r2"][name] for row in items], ddof=1))}
        for name in e12.LABEL_NAMES
    }
    patient_values = [np.mean([row["val"]["per_output_r2"][name] for name in e12.LABEL_NAMES[:3]]) for row in items]
    result["patient_mean_r2"] = {"mean": float(np.mean(patient_values)),
                                 "sd": float(np.std(patient_values, ddof=1))}
    if all("age_diagnostic" in row for row in items):
        result["age_diagnostic"] = {
            key: {"mean": float(np.mean([row["age_diagnostic"][key] for row in items])),
                  "sd": float(np.std([row["age_diagnostic"][key] for row in items], ddof=1))}
            for key in ("mae_years", "rmse_years", "r2")
        }
    return result


@torch.no_grad()
def evaluate_age(model: AgeAuxE11, arrays: dict, age_z: torch.Tensor, age_mask: torch.Tensor,
                 mean: float, sd: float) -> dict:
    model.eval()
    regions, region_mask = e12.e11.masked_region_means(arrays["lead"], arrays["lead_mask"])
    chunks = []
    for start in range(0, len(age_z), 96):
        chunks.append(model.predict_age_z(arrays["mlm"][start:start+96], arrays["global"][start:start+96],
                                          regions[start:start+96], region_mask[start:start+96]).cpu())
    pred_z = torch.cat(chunks).numpy()
    mask = age_mask.cpu().numpy().astype(bool)
    true_years = age_z.cpu().numpy()[mask] * sd + mean
    pred_years = pred_z[mask] * sd + mean
    error = pred_years - true_years
    denom = float(np.sum((true_years - true_years.mean()) ** 2))
    return {
        "patients": int(mask.sum()), "mae_years": float(np.mean(np.abs(error))),
        "rmse_years": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / denom),
        "selection_metric": False,
    }


def train_seed(run: Path, seed: int, device: torch.device) -> dict:
    out = run / "downstream" / f"seed{seed}"
    if (out / "VAL_COMPLETE").exists():
        return json.loads((out / "val_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    data = load_ecg_data(device)
    audit = json.loads((run / "age/audit.json").read_text())
    mean = float(audit["normalization"]["mean_years"])
    sd = float(audit["normalization"]["sd_years"])
    age = {split: torch.from_numpy(np.load(run / f"age/{split}_age_z.npy")).float().to(device)
           for split in ("train", "val")}
    age_mask = {split: torch.from_numpy(np.load(run / f"age/{split}_age_mask.npy")).bool().to(device)
                for split in ("train", "val")}
    target = data["train"]["labels"].argmax(1)
    model = AgeAuxE11(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    steps = len(target) // 64
    generator = torch.Generator().manual_seed(seed)
    regions, region_mask = e12.e11.masked_region_means(data["train"]["lead"], data["train"]["lead_mask"])
    history, best, best_epoch, best_state = [], -float("inf"), None, None
    for epoch in range(1, 31):
        model.train(); order = torch.randperm(len(target), generator=generator).to(device)
        sums = {"main": 0.0, "patient": 0.0, "age": 0.0, "batches": 0}
        for step in range(steps):
            idx = order[step * 64:(step + 1) * 64]
            lr = e12.lr_at((epoch - 1) * steps + step, 30 * steps, 5 * steps, 5e-4, 1e-5, 1e-6)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            main, aux, predicted_age = model.training_outputs_with_age(
                data["train"]["mlm"][idx], data["train"]["global"][idx], regions[idx], region_mask[idx])
            main_loss = F.cross_entropy(main, target[idx])
            patient_loss = e12.e9.patient_auxiliary_loss(aux, target[idx])
            a_loss = age_loss(predicted_age, age["train"][idx], age_mask["train"][idx])
            loss = main_loss + 0.2 * patient_loss + AGE_WEIGHT * a_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"seed{seed}/epoch{epoch}: non-finite loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
            sums["main"] += float(main_loss); sums["patient"] += float(patient_loss)
            sums["age"] += float(a_loss); sums["batches"] += 1
        val, _, _ = e12.eval_downstream(model, "e12a_region", data["val"])
        age_diag = evaluate_age(model, data["val"], age["val"], age_mask["val"], mean, sd)
        row = {"epoch": epoch, "train_main_ce": sums["main"] / sums["batches"],
               "train_patient_ce": sums["patient"] / sums["batches"],
               "train_age_huber": sums["age"] / sums["batches"],
               "val": val, "val_age_diagnostic": age_diag}
        history.append(row); print(json.dumps({"seed": seed, **row}, ensure_ascii=False), flush=True)
        if val["r2"] > best:
            best, best_epoch, best_state = val["r2"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val, prob, logits = e12.eval_downstream(model, "e12a_region", data["val"])
    age_diag = evaluate_age(model, data["val"], age["val"], age_mask["val"], mean, sd)
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "seed": seed,
                "best_epoch": best_epoch, "selection": "Val Overall R2", "test_run": False,
                "age_weight": AGE_WEIGHT, "age_mean_years": mean, "age_sd_years": sd},
               out / "checkpoint-best.pth")
    np.save(out / "val_logits.npy", logits); np.save(out / "val_probabilities.npy", prob)
    dump(out / "history.json", history)
    result = {
        "method": "e23_patient_age_auxiliary", "seed": seed, "best_epoch": best_epoch,
        "val": val, "age_diagnostic": age_diag, "test_run": False,
        "age_weight": AGE_WEIGHT, "age_loss": "SmoothL1 on Train-patient standardized age",
        "inference_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                               if not n.startswith("age_head")),
        "training_only_age_parameters": sum(p.numel() for n, p in model.named_parameters()
                                            if n.startswith("age_head")),
    }
    dump(out / "val_summary.json", result); (out / "VAL_COMPLETE").touch()
    return result


def train_all(run: Path, device: torch.device) -> None:
    path = run / "validation_summary.json"
    if path.exists():
        return
    rows = [train_seed(run, seed, device) for seed in (0, 1, 2)]
    baseline = json.loads((E12_RUN / "downstream/validation_selection.json").read_text())["methods"]["e12a_region"]
    dump(path, {"validation_only": True, "test_run": False,
                "e12a_historical": {"aggregate": aggregate(baseline["per_seed"]),
                                     "per_seed": baseline["per_seed"]},
                "e23_age_auxiliary": {"aggregate": aggregate(rows), "per_seed": rows}})


def report(run: Path) -> None:
    summary = json.loads((run / "validation_summary.json").read_text())
    audit = json.loads((run / "age/audit.json").read_text())
    base = summary["e12a_historical"]["aggregate"]
    model = summary["e23_age_auxiliary"]["aggregate"]
    rows = []
    for label, key in (("E12-A", "e12a_historical"), ("E23-Age", "e23_age_auxiliary")):
        for item in summary[key]["per_seed"]:
            val = item["val"]
            row = {"method": label, "seed": item.get("seed", item.get("downstream_seed")),
                   "overall_r2": val["r2"],
                   **{f"{name.lower().replace(' ', '_')}_r2": val["per_output_r2"][name]
                      for name in e12.LABEL_NAMES},
                   "macro_auc": val["macro_roc_auc"], "macro_f1": val["macro_f1"], "mse": val["mse"]}
            if "age_diagnostic" in item:
                row.update({f"age_{k}": v for k, v in item["age_diagnostic"].items()
                            if isinstance(v, (int, float, bool))})
            rows.append(row)
    pd.DataFrame(rows).to_csv(run / "metrics_per_seed.csv", index=False)
    delta = model["r2"]["mean"] - base["r2"]["mean"]
    lines = [
        "# E23：E12-A 下游患者年龄辅助监督", "", "## 设计", "",
        "- 完整复用E12-A冻结ECG表示、E11区域残差、原始四分类主头和患者三分类辅助头。",
        "- 年龄不是模型输入；仅Train患者计算标准化年龄SmoothL1辅助损失，Healthy不伪造年龄。",
        f"- 年龄按Train患者ECG记录均值/标准差标准化；跨年度同患者记录保留行级年龄；辅助权重预先固定为{AGE_WEIGHT:g}。",
        "- Validation年龄只用于选定分类checkpoint后的诊断；checkpoint仍只按Val Overall R²选择。",
        "- 下游seed 0/1/2、30 epoch；未加载或运行Test。", "",
        "## 年龄审计", "",
        f"- Train：{audit['splits']['train']['patient_records']}条患者ECG、"
        f"{audit['splits']['train']['unique_patients']}名患者，年龄完整；"
        f"{audit['splits']['train']['healthy_records']}条Healthy均无年龄。",
        f"- Train患者ECG年龄标准化：mean={audit['normalization']['mean_years']:.3f}岁，"
        f"sd={audit['normalization']['sd_years']:.3f}岁。", "",
        "## Validation（三个seed）", "",
        "| 方法 | Overall R² | Patient mean R² | OMI | NOMI | Control | Healthy | Macro AUC | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| E12-A | {base['r2']['mean']:.6f} ± {base['r2']['sd']:.6f} | {base['patient_mean_r2']['mean']:.6f} | " +
        " | ".join(f"{base['per_output_r2'][name]['mean']:.6f}" for name in e12.LABEL_NAMES) +
        f" | {base['macro_roc_auc']['mean']:.6f} | {base['macro_f1']['mean']:.6f} |",
        f"| E23-Age | {model['r2']['mean']:.6f} ± {model['r2']['sd']:.6f} | {model['patient_mean_r2']['mean']:.6f} | " +
        " | ".join(f"{model['per_output_r2'][name]['mean']:.6f}" for name in e12.LABEL_NAMES) +
        f" | {model['macro_roc_auc']['mean']:.6f} | {model['macro_f1']['mean']:.6f} |", "",
        f"Overall R²差值：**{delta:+.6f}**。",
        f"患者三类平均R²差值：**{model['patient_mean_r2']['mean'] - base['patient_mean_r2']['mean']:+.6f}**。",
        f"OMI / NOMI / Control / Healthy差值：**"
        f"{model['per_output_r2']['OMI']['mean'] - base['per_output_r2']['OMI']['mean']:+.6f} / "
        f"{model['per_output_r2']['NOMI']['mean'] - base['per_output_r2']['NOMI']['mean']:+.6f} / "
        f"{model['per_output_r2']['Clinical Control']['mean'] - base['per_output_r2']['Clinical Control']['mean']:+.6f} / "
        f"{model['per_output_r2']['Healthy']['mean'] - base['per_output_r2']['Healthy']['mean']:+.6f}**。", "",
        f"选定checkpoint的患者年龄诊断：MAE **{model['age_diagnostic']['mae_years']['mean']:.3f}岁**，"
        f"年龄R² **{model['age_diagnostic']['r2']['mean']:.4f}**。", "",
    ]
    (run / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global AGE_WEIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "smoke", "train", "report"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--age_weight", type=float, default=AGE_WEIGHT)
    args = parser.parse_args(); run = args.output_dir.resolve()
    if args.age_weight < 0:
        parser.error("--age_weight must be non-negative")
    AGE_WEIGHT = float(args.age_weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "audit":
        set_status(run, "audit_patient_age_targets"); build_age_targets(run)
    elif args.stage == "smoke":
        set_status(run, "smoke_exact_e12_inference_and_age_gradients"); smoke(run, device)
    elif args.stage == "train":
        set_status(run, "train_e23_age_auxiliary_seeds_0_1_2"); train_all(run, device)
    else:
        report(run); set_status(run, "complete"); (run / "COMPLETE").touch()


if __name__ == "__main__":
    main()
