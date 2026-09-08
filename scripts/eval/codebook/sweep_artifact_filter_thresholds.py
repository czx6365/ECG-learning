#!/usr/bin/env python3
"""Sweep artifact-filter thresholds to keep removal rate under a target fraction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ECG token artifact-filter thresholds.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--target_filter_fraction", type=float, default=0.05)
    return parser.parse_args()


def max_flat_run_batch(waves: np.ndarray, eps: float) -> np.ndarray:
    flat = np.abs(np.diff(waves, axis=1)) <= eps
    out = np.ones(waves.shape[0], dtype=np.int16)
    cur = np.ones(waves.shape[0], dtype=np.int16)
    for idx in range(flat.shape[1]):
        cur = np.where(flat[:, idx], cur + 1, 1)
        out = np.maximum(out, cur)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_path.expanduser().resolve()
    data = np.load(data_path / f"{args.split}_data.npy", mmap_mode="r")
    chans = np.load(data_path / f"{args.split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_path / f"{args.split}_data_in_times.npy", mmap_mode="r")

    flat_runs = []
    peak_indices = []
    valid_total = 0
    for sample_id in range(data.shape[0]):
        valid = (chans[sample_id] > 0) & (times[sample_id] > 0)
        if not np.any(valid):
            continue
        waves = np.asarray(data[sample_id, valid], dtype=np.float32)
        flat_runs.append(max_flat_run_batch(waves, args.eps))
        peak_indices.append(np.argmax(np.abs(waves), axis=1).astype(np.int16))
        valid_total += int(waves.shape[0])
        if sample_id == 0 or sample_id == data.shape[0] - 1 or sample_id % 1000 == 0:
            print(f"Scanned {sample_id + 1}/{data.shape[0]} samples; valid={valid_total}", flush=True)

    flat_runs = np.concatenate(flat_runs)
    peak_indices = np.concatenate(peak_indices)
    assert len(flat_runs) == valid_total

    flat_thresholds = [20, 24, 28, 32, 36, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    peak_ranges = [
        (0, 95),
        (5, 90),
        (10, 88),
        (15, 85),
        (20, 82),
        (25, 80),
        (30, 75),
        (35, 70),
        (40, 65),
    ]

    rows = []
    for flat_th in flat_thresholds:
        flat_bad = flat_runs >= flat_th
        for pmin, pmax in peak_ranges:
            peak_bad = (peak_indices < pmin) | (peak_indices > pmax)
            union_bad = flat_bad | peak_bad
            rows.append(
                {
                    "max_flat_run_ge": flat_th,
                    "peak_idx_min": pmin,
                    "peak_idx_max": pmax,
                    "flat_bad_count": int(flat_bad.sum()),
                    "flat_bad_fraction": float(flat_bad.mean()),
                    "peak_bad_count": int(peak_bad.sum()),
                    "peak_bad_fraction": float(peak_bad.mean()),
                    "union_bad_count": int(union_bad.sum()),
                    "union_bad_fraction": float(union_bad.mean()),
                    "kept_count": int((~union_bad).sum()),
                    "kept_fraction": float((~union_bad).mean()),
                    "under_target": bool(union_bad.mean() <= args.target_filter_fraction),
                }
            )

    out_csv = args.output_dir / "artifact_filter_threshold_sweep.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    quantiles = {
        "valid_tokens": int(valid_total),
        "max_flat_run_quantiles": {
            str(q): float(np.percentile(flat_runs, q)) for q in [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]
        },
        "peak_idx_quantiles": {
            str(q): float(np.percentile(peak_indices, q)) for q in [0.1, 0.5, 1, 2, 5, 10, 50, 90, 95, 98, 99, 99.5, 99.9]
        },
        "target_filter_fraction": float(args.target_filter_fraction),
    }

    candidates = [row for row in rows if row["under_target"]]
    candidates = sorted(candidates, key=lambda r: (abs(r["union_bad_fraction"] - args.target_filter_fraction), -r["union_bad_fraction"]))
    quantiles["recommended_candidates"] = candidates[:10]

    with open(args.output_dir / "artifact_filter_threshold_sweep_summary.json", "w", encoding="utf-8") as handle:
        json.dump(quantiles, handle, indent=2, ensure_ascii=False)
    print(json.dumps(quantiles, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
