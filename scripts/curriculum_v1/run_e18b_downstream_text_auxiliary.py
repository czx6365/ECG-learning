#!/usr/bin/env python3
"""E18-B: report-assisted downstream training on the frozen E12-A representation.

The inference path is exactly the E12-A Mean+Max + E11 regional residual model.
Chinese ECG reports are an auxiliary training-only signal for OMI/NOMI/Control;
healthy ECGs have no report and never receive a fabricated text target.  Validation
and test inference consume ECG features only.  This exploratory run is Val-only.
"""

from __future__ import annotations

import argparse
import copy
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
import run_e18_ecg_text_contrastive as e18


E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
E18_RUN = ROOT / "results/e18_ecg_text_contrastive"
DEFAULT_RUN = ROOT / "results/e18b_downstream_text_auxiliary"
TEXT_WEIGHT = 0.1
TEXT_DIM = 768
PROJ_DIM = 256


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_status(run_dir: Path, value: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.txt").write_text(value + "\n", encoding="utf-8")
    print(value, flush=True)


def build_manifest(run_dir: Path) -> None:
    out = run_dir / "text_manifest"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=False)
    pairs, source_audit = e18.label_pairs()
    reports = [json.loads(line) for line in
               (E18_RUN / "text_alignment/report_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    text_to_id = {row["text"]: int(row["report_id"]) for row in reports}
    embeddings = np.load(E18_RUN / "text_alignment/medbert_report_embeddings.npy", mmap_mode="r")
    if embeddings.shape != (len(reports), TEXT_DIM) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Invalid frozen text embeddings: {embeddings.shape}")

    split_audit = {}
    train_index = None
    for split in ("train", "val", "test"):
        meta = pd.read_csv(e12.DOWNSTREAM / f"{split}_metadata.csv", low_memory=False).copy()
        labels = np.load(e12.DOWNSTREAM / f"{split}_labels.npy").argmax(1)
        meta["csv_stem"] = meta["csv_name"].astype(str).map(lambda x: Path(x).stem)
        merged = meta.merge(
            pairs[["csv_stem", "patient_sn", "excel_row", "report_text"]],
            on="csv_stem", how="left", validate="one_to_one", suffixes=("", "_report"),
        )
        patient = labels != 3
        has_report = merged["report_text"].fillna("").str.len().gt(0).to_numpy()
        if not np.all(has_report[patient]) or np.any(has_report[~patient]):
            raise RuntimeError(f"{split}: report presence does not match patient/healthy status")
        mismatch = has_report & (merged["patient_sn"].astype(str) != merged["patient_sn_report"].astype(str))
        if mismatch.any():
            raise RuntimeError(f"{split}: ECG/report patient mismatch")
        ids = np.full(len(merged), -1, dtype=np.int64)
        if split == "train":
            for i in np.flatnonzero(patient):
                text = str(merged.iloc[i]["report_text"])
                if text not in text_to_id:
                    raise RuntimeError(f"Train report missing frozen MedBERT embedding: row {i}")
                ids[i] = text_to_id[text]
            train_index = ids
            manifest = merged.loc[patient, ["csv_name", "patient_sn", "excel_row", "report_text"]].copy()
            manifest.insert(0, "downstream_index", np.flatnonzero(patient))
            manifest["report_id"] = ids[patient]
            manifest.to_csv(out / "train_report_pairs.csv", index=False)
        by_class = {}
        for class_id, name in enumerate(e12.LABEL_NAMES):
            mask = labels == class_id
            by_class[name] = {
                "records": int(mask.sum()), "with_report": int(has_report[mask].sum()),
                "without_report": int((~has_report[mask]).sum()),
            }
        split_audit[split] = {"records": len(labels), "by_class": by_class}

    np.save(out / "train_report_index.npy", train_index)
    audit = {
        "experiment": "E18-B downstream text auxiliary",
        "mapping": "row-composite diagnosis_1..6; same-row ECGs share report_id",
        "text_training_scope": "downstream Train OMI/NOMI/Clinical Control only",
        "healthy_policy": "no report, no fabricated text, classification losses only",
        "validation_test_policy": "reports audited for presence only; never loaded by training/evaluation",
        "inference_inputs": "ECG features only",
        "source_audit": source_audit,
        "frozen_text_embeddings": str(E18_RUN / "text_alignment/medbert_report_embeddings.npy"),
        "splits": split_audit,
    }
    dump(out / "audit.json", audit)
    (out / "COMPLETE").touch()


def load_data(device: torch.device) -> dict:
    data = {}
    for split in ("train", "val", "test"):
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


class TextAuxE11(e12.e11.E11RegionResidual):
    def __init__(self, seed: int):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(189000 + int(seed))
            self.ecg_text_projection = nn.Linear(128, PROJ_DIM)
            self.text_projection = nn.Sequential(nn.LayerNorm(TEXT_DIM), nn.Linear(TEXT_DIM, PROJ_DIM))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

    def training_outputs_with_text(self, mlm, global_ze, regions, region_mask, frozen_text):
        x, _, _ = self.encode(mlm, global_ze, regions, region_mask)
        hidden = self.classifier.shared(x)
        main, aux = self.classifier.main(hidden), self.classifier.aux(hidden)
        ecg = F.normalize(self.ecg_text_projection(hidden), dim=-1)
        text = F.normalize(self.text_projection(frozen_text), dim=-1)
        return main, aux, ecg, text


def smoke(run_dir: Path, device: torch.device) -> None:
    path = run_dir / "smoke.json"
    if path.exists():
        return
    build_manifest(run_dir)
    data = load_data(device)
    report_index = np.load(run_dir / "text_manifest/train_report_index.npy")
    patient = np.flatnonzero(report_index >= 0)[:32]
    healthy = np.flatnonzero(report_index < 0)[:32]
    idx_np = np.concatenate([patient, healthy])
    idx = torch.from_numpy(idx_np).long().to(device)
    text_table = torch.from_numpy(
        np.asarray(np.load(E18_RUN / "text_alignment/medbert_report_embeddings.npy"), dtype=np.float32)
    ).to(device)
    report = torch.from_numpy(report_index[idx_np]).long().to(device)

    model = TextAuxE11(0).to(device)
    baseline = e12.e11.E11RegionResidual(0).to(device)
    common = baseline.state_dict()
    model_state = model.state_dict()
    common_max_diff = max(float((model_state[k] - v).abs().max()) for k, v in common.items())
    if common_max_diff != 0:
        raise RuntimeError(f"E12 downstream initialization changed: {common_max_diff}")
    regions, region_mask = e12.e11.masked_region_means(data["train"]["lead"][idx], data["train"]["lead_mask"][idx])
    paired = report >= 0
    main, aux, ecg, text = model.training_outputs_with_text(
        data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask,
        text_table[report.clamp_min(0)],
    )
    target = data["train"]["labels"][idx].argmax(1)
    clip, stats = e18.contrastive_loss(ecg[paired], text[paired], report[paired], model.logit_scale)
    loss = F.cross_entropy(main, target) + 0.2 * e12.e9.patient_auxiliary_loss(aux, target) + TEXT_WEIGHT * clip
    loss.backward()
    grads = {
        "classifier_shared": float(model.classifier.shared[0].weight.grad.norm()),
        "ecg_projection": float(model.ecg_text_projection.weight.grad.norm()),
        "text_projection": float(model.text_projection[1].weight.grad.norm()),
    }
    if not math.isfinite(float(loss)) or not all(math.isfinite(v) and v > 0 for v in grads.values()):
        raise RuntimeError(f"Non-finite or missing gradients: loss={loss}, grads={grads}")
    model.eval(); baseline.eval()
    with torch.no_grad():
        base_logits = baseline(data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask)
        model_logits = model(data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask)
    dump(path, {
        "common_initialization_max_abs_diff": common_max_diff,
        "initial_inference_logits_max_abs_diff": float((base_logits - model_logits).abs().max()),
        "batch_patient_reports": int(paired.sum()), "batch_healthy_without_reports": int((~paired).sum()),
        "loss_finite": True, "grad_norms": grads, "contrastive": stats,
        "validation_test_text_used": False,
    })


def train_seed(run_dir: Path, seed: int, device: torch.device) -> dict:
    out = run_dir / "downstream" / f"seed{seed}"
    if (out / "VAL_COMPLETE").exists():
        return json.loads((out / "val_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    data = load_data(device)
    report_np = np.load(run_dir / "text_manifest/train_report_index.npy")
    report_index = torch.from_numpy(report_np).long().to(device)
    text_table = torch.from_numpy(
        np.asarray(np.load(E18_RUN / "text_alignment/medbert_report_embeddings.npy"), dtype=np.float32)
    ).to(device)
    model = TextAuxE11(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    target = data["train"]["labels"].argmax(1)
    steps = len(target) // 64
    generator = torch.Generator().manual_seed(seed)
    train_regions, train_region_mask = e12.e11.masked_region_means(
        data["train"]["lead"], data["train"]["lead_mask"])
    history, best, best_epoch, best_state = [], -float("inf"), None, None
    for epoch in range(1, 31):
        model.train()
        order = torch.randperm(len(target), generator=generator).to(device)
        sums = {"main": 0.0, "patient": 0.0, "text": 0.0, "r1": 0.0, "batches": 0}
        for step in range(steps):
            idx = order[step * 64:(step + 1) * 64]
            lr = e12.lr_at((epoch - 1) * steps + step, 30 * steps, 5 * steps, 5e-4, 1e-5, 1e-6)
            for group in optimizer.param_groups:
                group["lr"] = lr
            report = report_index[idx]
            paired = report >= 0
            optimizer.zero_grad(set_to_none=True)
            main, aux, ecg, text = model.training_outputs_with_text(
                data["train"]["mlm"][idx], data["train"]["global"][idx],
                train_regions[idx], train_region_mask[idx], text_table[report.clamp_min(0)],
            )
            main_loss = F.cross_entropy(main, target[idx])
            patient_loss = e12.e9.patient_auxiliary_loss(aux, target[idx])
            text_loss, text_stats = e18.contrastive_loss(
                ecg[paired], text[paired], report[paired], model.logit_scale)
            loss = main_loss + 0.2 * patient_loss + TEXT_WEIGHT * text_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            model.logit_scale.data.clamp_(max=math.log(100.0))
            sums["main"] += float(main_loss); sums["patient"] += float(patient_loss)
            sums["text"] += float(text_loss); sums["r1"] += text_stats["retrieval_r1"]
            sums["batches"] += 1
        val, _, _ = e12.eval_downstream(model, "e12a_region", data["val"])
        row = {
            "epoch": epoch, "train_main_ce": sums["main"] / sums["batches"],
            "train_patient_ce": sums["patient"] / sums["batches"],
            "train_text_ce": sums["text"] / sums["batches"],
            "train_text_r1": sums["r1"] / sums["batches"],
            "temperature": float(1 / model.logit_scale.exp()), "val": val,
        }
        history.append(row)
        print(json.dumps({"seed": seed, **row}, ensure_ascii=False), flush=True)
        if val["r2"] > best:
            best, best_epoch, best_state = val["r2"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val, prob, logits = e12.eval_downstream(model, "e12a_region", data["val"])
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "seed": seed,
                "best_epoch": best_epoch, "selection": "Val Overall R2",
                "test_run": False}, out / "checkpoint-best.pth")
    np.save(out / "val_logits.npy", logits)
    np.save(out / "val_probabilities.npy", prob)
    dump(out / "history.json", history)
    result = {"method": "e18b_downstream_text_aux", "seed": seed, "best_epoch": best_epoch,
              "val": val, "test_run": False,
              "inference_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                    if not n.startswith(("ecg_text_projection", "text_projection", "logit_scale"))),
              "training_only_text_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                   if n.startswith(("ecg_text_projection", "text_projection", "logit_scale")))}
    dump(out / "val_summary.json", result)
    (out / "VAL_COMPLETE").touch()
    return result


def train_all(run_dir: Path, device: torch.device) -> None:
    path = run_dir / "validation_summary.json"
    if path.exists():
        return
    rows = [train_seed(run_dir, seed, device) for seed in (0, 1, 2)]
    baseline = json.loads((E12_RUN / "downstream/validation_selection.json").read_text())["methods"]["e12a_region"]
    base_rows = baseline["per_seed"]
    keys = ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy")

    def aggregate(items):
        result = {key: {"mean": float(np.mean([row["val"][key] for row in items])),
                        "sd": float(np.std([row["val"][key] for row in items], ddof=1))} for key in keys}
        result["per_output_r2"] = {
            name: {"mean": float(np.mean([row["val"]["per_output_r2"][name] for row in items])),
                   "sd": float(np.std([row["val"]["per_output_r2"][name] for row in items], ddof=1))}
            for name in e12.LABEL_NAMES
        }
        return result

    dump(path, {"validation_only": True, "test_run": False,
                "e12a_historical": {"aggregate": aggregate(base_rows), "per_seed": base_rows},
                "e18b_text_aux": {"aggregate": aggregate(rows), "per_seed": rows}})


def report(run_dir: Path) -> None:
    summary = json.loads((run_dir / "validation_summary.json").read_text())
    base = summary["e12a_historical"]["aggregate"]
    text = summary["e18b_text_aux"]["aggregate"]
    rows = []
    for label, key in (("E12-A", "e12a_historical"), ("E18-B text auxiliary", "e18b_text_aux")):
        for item in summary[key]["per_seed"]:
            val = item["val"]
            rows.append({"method": label, "seed": item.get("seed", item.get("downstream_seed")),
                         "overall_r2": val["r2"],
                         **{f"{name.lower().replace(' ', '_')}_r2": val["per_output_r2"][name]
                            for name in e12.LABEL_NAMES},
                         "macro_auc": val["macro_roc_auc"], "macro_f1": val["macro_f1"],
                         "mse": val["mse"]})
    pd.DataFrame(rows).to_csv(run_dir / "metrics_per_seed.csv", index=False)
    lines = [
        "# E18-B：下游中文报告辅助训练", "", "## 设计", "",
        "- 冻结并复用E12-A ECG表示；推理主路径与E12-A相同。",
        "- OMI/NOMI/Clinical Control训练样本使用同一Excel行的完整诊断报告作为对比辅助监督。",
        "- Healthy没有诊断报告，只计算四分类CE；不生成或填充伪文本。",
        "- 总损失：CE4 + 0.2×患者CE3 + 0.1×患者ECG-文本多正样本对比CE。",
        "- Validation和推理只输入ECG；未运行Test。", "", "## Validation（三个下游seed）", "",
        "| 方法 | Overall R² | OMI | NOMI | Clinical Control | Healthy | Macro AUC | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| E12-A | {base['r2']['mean']:.6f} ± {base['r2']['sd']:.6f} | " +
        " | ".join(f"{base['per_output_r2'][name]['mean']:.6f}" for name in e12.LABEL_NAMES) +
        f" | {base['macro_roc_auc']['mean']:.6f} | {base['macro_f1']['mean']:.6f} |",
        f"| E18-B | {text['r2']['mean']:.6f} ± {text['r2']['sd']:.6f} | " +
        " | ".join(f"{text['per_output_r2'][name]['mean']:.6f}" for name in e12.LABEL_NAMES) +
        f" | {text['macro_roc_auc']['mean']:.6f} | {text['macro_f1']['mean']:.6f} |", "",
        f"Overall R²差值：**{text['r2']['mean'] - base['r2']['mean']:+.6f}**。",
        f"OMI差值：**{text['per_output_r2']['OMI']['mean'] - base['per_output_r2']['OMI']['mean']:+.6f}**；"
        f"NOMI差值：**{text['per_output_r2']['NOMI']['mean'] - base['per_output_r2']['NOMI']['mean']:+.6f}**。", "",
        "## 边界", "", "- 文本只属于训练期辅助监督，不是推理输入。",
        "- 这是一个预先固定权重的Validation实验；没有使用Test调参。", "",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "smoke", "train", "report"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_dir = args.output_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "audit":
        set_status(run_dir, "audit_train_only_report_mapping")
        build_manifest(run_dir)
    elif args.stage == "smoke":
        set_status(run_dir, "smoke_text_auxiliary_gradients")
        smoke(run_dir, device)
    elif args.stage == "train":
        set_status(run_dir, "downstream_text_auxiliary_seeds_0_1_2")
        train_all(run_dir, device)
    elif args.stage == "report":
        report(run_dir)
        set_status(run_dir, "complete")


if __name__ == "__main__":
    main()
