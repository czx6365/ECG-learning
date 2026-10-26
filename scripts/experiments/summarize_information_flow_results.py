#!/usr/bin/env python3
"""Assemble codebook, pretraining, downstream, and representation-probe results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
INFO_ROOT = ROOT / "results" / "information_flow"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pretrain_summary(directory: Path) -> dict:
    rows = list(csv.DictReader((directory / "training_history.csv").open()))
    val_rows = [row for row in rows if row.get("val_mlm_acc") not in (None, "")]
    best_acc = max(val_rows, key=lambda row: float(row["val_mlm_acc"]))
    checkpoint = torch.load(directory / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    return {
        "checkpoint": str(directory / "checkpoint-best.pth"),
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "max_val_mlm_accuracy": float(best_acc["val_mlm_acc"]),
        "max_val_mlm_accuracy_epoch": int(float(best_acc["epoch"])),
        "best_checkpoint_metric": checkpoint.get("best_metric"),
        "validation_frequency": 10,
    }


def select_codebook_summary(comparison_dir: Path, run_name: str) -> dict:
    summary = read_json(comparison_dir / run_name / "summary.json")
    phys = summary["physiology"]
    return {
        "checkpoint": summary["checkpoint"],
        "reconstruction_rmse": summary["reconstruction_rmse"],
        "reconstruction_mae": summary["reconstruction_mae"],
        "reconstruction_corr": summary["reconstruction_corr"],
        "active_sid": summary["active_sid"],
        "dead_sid": summary["dead_sid"],
        "perplexity": summary["perplexity"],
        "usage_entropy_nats": float(__import__("math").log(summary["perplexity"])),
        "top1pct_usage": summary.get("top1pct_usage"),
        "macro_intra_shift_corr_similarity": summary["macro_intra_shift_corr_similarity"],
        "inter_over_intra_shift_corr": summary["inter_over_intra_shift_corr"],
        "weighted_physiology_purity": phys["weighted_average_physiology_purity"],
        "r_purity": phys["discrete_purity_weighted"]["r_polarity"],
        "t_purity": phys["discrete_purity_weighted"]["t_polarity"],
        "st_purity": phys["discrete_purity_weighted"]["st_state"],
        "qrs_purity": phys["discrete_purity_weighted"]["qrs_type"],
        "high_corr_sid_disagreement": summary["global_morphology_consistent_sid_disagreement"],
    }


def direction(value: float, epsilon: float) -> str:
    if value < -epsilon:
        return "negative"
    if value > epsilon:
        return "positive"
    return "approximately_zero"


def main() -> None:
    v1_eval = INFO_ROOT / "codebook" / "MorphologyV2_vs_V1PlusPhysiology_val_shift4"
    no_cont_eval = ROOT / "results" / "morphology_v2_codebook" / "MorphologyV2_vs_NoContinuous_val_shift4"
    probe_summary = read_json(INFO_ROOT / "probes" / "information_flow_summary.json")

    experiments = {
        "V1-Plus-Physiology": {
            "training": {
                "data": "patient_healthy_raw_v2_clean_le5pct",
                "codebook": "1024x64",
                "epochs": 50,
                "losses": "waveform + VQ + discrete/continuous physiology; corr=0; derivative=0",
                "seed": 0,
            },
            "codebook": select_codebook_summary(v1_eval, "V1-Plus-Physiology"),
            "pretrain": pretrain_summary(ROOT / "checkpoints/pretrain/HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_rawv2_e100"),
            "regression": read_json(ROOT / "checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_e100_linear/test_regression_summary.json"),
            "classification": read_json(ROOT / "checkpoints/finetune/ecg_dsa/classification_label4_HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_e100_linear_weighted_ce/test_classification_summary.json"),
        },
        "NoContinuous": {
            "training": {
                "data": "patient_healthy_raw_v2_clean_le5pct",
                "codebook": "1024x64",
                "epochs": 50,
                "losses": "waveform + VQ + corr + derivative + discrete physiology; continuous physiology disabled",
                "seed": 0,
            },
            "codebook": select_codebook_summary(no_cont_eval, "Morphology-V2-NoContinuous"),
            "pretrain": pretrain_summary(ROOT / "checkpoints/pretrain/HeartLangSID_MorphologyV2NoContinuous_SIDOnly_rawv2_e100"),
            "regression": read_json(ROOT / "checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_MorphologyV2NoContinuous_SIDOnly_e100_linear/test_regression_summary.json"),
            "classification": read_json(ROOT / "checkpoints/finetune/ecg_dsa/classification_label4_HeartLangSID_MorphologyV2NoContinuous_SIDOnly_e100_linear_weighted_ce/test_classification_summary.json"),
        },
    }

    diagnoses = {}
    for run in probe_summary["runs"]:
        gaps = run["gaps"]
        vq_r2_state = direction(gaps["delta_vq_r2"], 0.005)
        vq_cls_state = direction(gaps["delta_vq_balanced_accuracy"], 0.005)
        pre_r2_state = direction(gaps["delta_pretrain_r2"], 0.005)
        pre_cls_state = direction(gaps["delta_pretrain_balanced_accuracy"], 0.005)
        diagnoses[run["model"]] = {
            "gaps": gaps,
            "quantization_regression": vq_r2_state,
            "quantization_classification": vq_cls_state,
            "pretraining_regression": pre_r2_state,
            "pretraining_classification": pre_cls_state,
            "interpretation": {
                "vq": "VQ loses downstream information" if "negative" in (vq_r2_state, vq_cls_state) else "No clear VQ loss at the 0.005 practical threshold",
                "heartlang": "HeartLang adds downstream information" if "positive" in (pre_r2_state, pre_cls_state) else ("HeartLang loses downstream information" if "negative" in (pre_r2_state, pre_cls_state) else "HeartLang adds little measurable downstream information"),
            },
        }

    report = {
        "git_commit": None,
        "git_commit_note": "Remote project snapshot has no .git metadata; SHA-256 fingerprints are in execution_manifest.txt",
        "probe_protocol": {
            "patient_split": "canonical existing train/val/test split, 6640/1344/1366 records",
            "z_e_z_q_pooling": "masked mean over valid heartbeat tokens",
            "heartlang_pooling": "CLS token",
            "feature_standardization": False,
            "frozen_backbones": True,
            "linear_head_only": True,
            "epochs": 30,
            "early_stopping": "disabled identically; validation-best checkpoint selected",
            "seed": 0,
        },
        "experiments": experiments,
        "representation_probes": probe_summary,
        "diagnoses": diagnoses,
    }
    (INFO_ROOT / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    table = list(csv.DictReader((INFO_ROOT / "probes" / "representation_probe_table.csv").open()))
    lines = [
        "# Codebook → Pretrain → Downstream information-flow report",
        "",
        "| Model | Representation | R² | OMI R² | NOMI R² | Control R² | Healthy R² | Accuracy | BAcc | Macro F1 | Macro AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['representation']} | {float(row['r2']):.4f} | "
            f"{float(row['omi_r2']):.4f} | {float(row['nomi_r2']):.4f} | "
            f"{float(row['control_r2']):.4f} | {float(row['healthy_r2']):.4f} | "
            f"{float(row['accuracy']):.4f} | {float(row['balanced_accuracy']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['macro_roc_auc']):.4f} |"
        )
    lines += ["", "## Information gaps", ""]
    for model, diagnosis in diagnoses.items():
        gaps = diagnosis["gaps"]
        lines.append(
            f"- {model}: ΔVQ R²={gaps['delta_vq_r2']:+.4f}, "
            f"ΔVQ BAcc={gaps['delta_vq_balanced_accuracy']:+.4f}; "
            f"ΔPretrain R²={gaps['delta_pretrain_r2']:+.4f}, "
            f"ΔPretrain BAcc={gaps['delta_pretrain_balanced_accuracy']:+.4f}."
        )
    (INFO_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
