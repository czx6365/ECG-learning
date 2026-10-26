#!/usr/bin/env python3
"""Evaluate C5b-lite tokenizer with the exact C2.1 shift/purity/usage protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT / "scripts" / "curriculum_v1"))
import evaluate_c21_tokenizers as evaluator


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c5b_checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    data_dir = ROOT / "datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
    cache_dir = ROOT / "results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct"
    data = np.load(data_dir / "val_data.npy", mmap_mode="r")
    chans = np.load(data_dir / "val_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_dir / "val_data_in_times.npy", mmap_mode="r")
    labels = np.load(cache_dir / "val_phys_discrete.npy", mmap_mode="r")
    checkpoints = {
        "C2.1_pattern_shift": ROOT / "checkpoints/curriculum_v1/c21_pattern_shift_from_c2_e50/checkpoint-best.pth",
        "C5b_lite_joint": resolve(args.c5b_checkpoint),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        name: evaluator.evaluate(name, path, data, chans, times, labels, device, args.batch_size)
        for name, path in checkpoints.items()
    }
    c21 = results["C2.1_pattern_shift"]
    c5b = results["C5b_lite_joint"]
    results["delta_c5b_minus_c21"] = {
        "active_sid": c5b["active_sid"] - c21["active_sid"],
        "perplexity": c5b["perplexity"] - c21["perplexity"],
        "sid_disagreement": {
            key: c5b["sid_disagreement"][key] - c21["sid_disagreement"][key]
            for key in c21["sid_disagreement"]
        },
        "morphology_purity": {
            key: c5b["morphology_purity"][key] - c21["morphology_purity"][key]
            for key in c21["morphology_purity"]
        },
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
