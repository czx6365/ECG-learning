#!/usr/bin/env python3
"""Evaluate how strongly SID assignments organize continuous heartbeat features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Continuous physiology variance within and between SIDs")
    parser.add_argument("--phys_dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--run", action="append", required=True, help="NAME,ASSIGNMENT_NPY")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--chunk_records", type=int, default=512)
    parser.add_argument("--exclude_features", nargs="*", default=["rr_interval_ms"])
    return parser.parse_args()


def parse_run(spec: str) -> tuple[str, Path]:
    parts = [item.strip() for item in spec.split(",", 1)]
    if len(parts) != 2:
        raise ValueError("--run must be NAME,ASSIGNMENT_NPY")
    return parts[0], Path(parts[1]).expanduser().resolve()


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def evaluate_run(
    name: str,
    assignment_path: Path,
    continuous: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    feature_indices: list[int],
    codebook_size: int,
    chunk_records: int,
    output_dir: Path,
) -> dict:
    assignments = np.load(assignment_path, mmap_mode="r")
    if assignments.shape[1:] != continuous.shape[1:2] or len(assignments) > len(continuous):
        raise ValueError(
            f"{name}: assignments {assignments.shape} are not a valid prefix of feature cache {continuous.shape[:2]}"
        )
    n_features = len(feature_indices)
    counts = np.zeros((codebook_size, n_features), dtype=np.int64)
    sums = np.zeros((codebook_size, n_features), dtype=np.float64)
    sums_sq = np.zeros((codebook_size, n_features), dtype=np.float64)

    for start in range(0, len(assignments), chunk_records):
        end = min(start + chunk_records, len(assignments))
        sid = np.asarray(assignments[start:end], dtype=np.int64).reshape(-1)
        values = np.asarray(continuous[start:end], dtype=np.float64)[..., feature_indices].reshape(-1, n_features)
        valid = np.asarray(masks[start:end], dtype=bool)[..., feature_indices].reshape(-1, n_features)
        sid_valid = (sid >= 0) & (sid < codebook_size)
        for feature_idx in range(n_features):
            keep = sid_valid & valid[:, feature_idx] & np.isfinite(values[:, feature_idx])
            if not keep.any():
                continue
            feature_sid = sid[keep]
            feature_values = values[keep, feature_idx]
            counts[:, feature_idx] += np.bincount(feature_sid, minlength=codebook_size)
            sums[:, feature_idx] += np.bincount(
                feature_sid, weights=feature_values, minlength=codebook_size
            )
            sums_sq[:, feature_idx] += np.bincount(
                feature_sid, weights=np.square(feature_values), minlength=codebook_size
            )

    means = safe_divide(sums, counts)
    variances = safe_divide(sums_sq, counts) - np.square(means)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances)
    rows = []
    for sid in range(codebook_size):
        row = {"sid": sid}
        for feature_idx, feature in enumerate(feature_names):
            row[f"{feature}_count"] = int(counts[sid, feature_idx])
            row[f"{feature}_mean"] = float(means[sid, feature_idx]) if counts[sid, feature_idx] else ""
            row[f"{feature}_std"] = float(stds[sid, feature_idx]) if counts[sid, feature_idx] else ""
        rows.append(row)
    per_sid_path = output_dir / f"{name}_per_sid_continuous_stats.csv"
    with per_sid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    feature_summaries = {}
    for feature_idx, feature in enumerate(feature_names):
        feature_counts = counts[:, feature_idx]
        used = feature_counts > 0
        total_count = int(feature_counts.sum())
        global_sum = float(sums[:, feature_idx].sum())
        global_sum_sq = float(sums_sq[:, feature_idx].sum())
        global_mean = global_sum / total_count
        total_variance = max(global_sum_sq / total_count - global_mean**2, 0.0)
        within_sse = np.sum(
            sums_sq[used, feature_idx]
            - np.square(sums[used, feature_idx]) / feature_counts[used]
        )
        within_variance = max(float(within_sse / total_count), 0.0)
        between_variance = max(total_variance - within_variance, 0.0)
        weighted_sid_std = float(
            np.sum(feature_counts[used] * stds[used, feature_idx]) / total_count
        )
        feature_summaries[feature] = {
            "valid_tokens": total_count,
            "active_sid": int(used.sum()),
            "global_mean": global_mean,
            "total_variance": total_variance,
            "within_sid_variance": within_variance,
            "between_sid_variance": between_variance,
            "between_over_within": (
                float(between_variance / within_variance) if within_variance > 0 else None
            ),
            "sid_explained_variance": (
                float(between_variance / total_variance) if total_variance > 0 else None
            ),
            "weighted_within_sid_std": weighted_sid_std,
        }
    explained = [
        item["sid_explained_variance"]
        for item in feature_summaries.values()
        if item["sid_explained_variance"] is not None
    ]
    ratios = [
        item["between_over_within"]
        for item in feature_summaries.values()
        if item["between_over_within"] is not None
    ]
    summary = {
        "run": name,
        "assignment_path": str(assignment_path),
        "features": feature_names,
        "feature_metrics": feature_summaries,
        "macro_sid_explained_variance": float(np.mean(explained)),
        "macro_between_over_within": float(np.mean(ratios)),
        "per_sid_stats": str(per_sid_path),
    }
    (output_dir / f"{name}_continuous_variance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    phys_dir = args.phys_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((phys_dir / "physiology_feature_schema.json").read_text(encoding="utf-8"))
    all_features = [str(name) for name in schema["continuous_features"]]
    excluded = set(args.exclude_features)
    feature_indices = [idx for idx, name in enumerate(all_features) if name not in excluded]
    feature_names = [all_features[idx] for idx in feature_indices]
    continuous = np.load(phys_dir / f"{args.split}_phys_continuous.npy", mmap_mode="r")
    masks = np.load(phys_dir / f"{args.split}_phys_continuous_mask.npy", mmap_mode="r")
    summaries = [
        evaluate_run(
            name,
            path,
            continuous,
            masks,
            feature_names,
            feature_indices,
            args.codebook_size,
            args.chunk_records,
            output_dir,
        )
        for name, path in (parse_run(spec) for spec in args.run)
    ]
    comparison_rows = []
    for summary in summaries:
        row = {
            "run": summary["run"],
            "macro_sid_explained_variance": summary["macro_sid_explained_variance"],
            "macro_between_over_within": summary["macro_between_over_within"],
        }
        for feature, metrics in summary["feature_metrics"].items():
            row[f"{feature}_explained_variance"] = metrics["sid_explained_variance"]
            row[f"{feature}_between_over_within"] = metrics["between_over_within"]
            row[f"{feature}_within_variance"] = metrics["within_sid_variance"]
        comparison_rows.append(row)
    comparison_path = output_dir / "continuous_variance_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    result = {
        "phys_dir": str(phys_dir),
        "split": args.split,
        "excluded_features": sorted(excluded),
        "runs": summaries,
        "comparison_csv": str(comparison_path),
    }
    (output_dir / "continuous_variance_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
