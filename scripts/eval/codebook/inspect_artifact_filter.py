#!/usr/bin/env python3
"""Count and visualize ECG tokens removed by simple artifact filters."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ECG token artifact filters.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_flat_run", type=int, default=12)
    parser.add_argument("--edge_len", type=int, default=12)
    parser.add_argument("--peak_idx_min", type=int, default=44)
    parser.add_argument("--peak_idx_max", type=int, default=58)
    parser.add_argument("--check_prefix_flat", action="store_true")
    parser.add_argument("--check_suffix_flat", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--sample_per_reason", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def max_flat_run(wave: np.ndarray, eps: float) -> int:
    diffs = np.abs(np.diff(wave))
    best = 1
    cur = 1
    for diff in diffs:
        if diff <= eps:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def edge_is_flat(wave: np.ndarray, edge_len: int, eps: float, prefix: bool) -> bool:
    edge = wave[:edge_len] if prefix else wave[-edge_len:]
    return bool(np.max(np.abs(np.diff(edge))) <= eps)


def reasons_for_wave(wave: np.ndarray, args: argparse.Namespace) -> tuple[list[str], dict]:
    flat_run = max_flat_run(wave, args.eps)
    prefix_flat = edge_is_flat(wave, args.edge_len, args.eps, prefix=True)
    suffix_flat = edge_is_flat(wave, args.edge_len, args.eps, prefix=False)
    peak_idx = int(np.argmax(np.abs(wave)))
    reasons = []
    if flat_run >= args.max_flat_run:
        reasons.append("long_flat_run")
    if args.check_prefix_flat and prefix_flat:
        reasons.append("prefix_flat")
    if args.check_suffix_flat and suffix_flat:
        reasons.append("suffix_flat")
    if peak_idx < args.peak_idx_min:
        reasons.append("peak_too_early")
    if peak_idx > args.peak_idx_max:
        reasons.append("peak_too_late")
    features = {
        "max_flat_run": int(flat_run),
        "prefix_flat": bool(prefix_flat),
        "suffix_flat": bool(suffix_flat),
        "peak_idx": int(peak_idx),
        "std": float(np.std(wave)),
        "min": float(np.min(wave)),
        "max": float(np.max(wave)),
        "range": float(np.max(wave) - np.min(wave)),
    }
    return reasons, features


def reservoir_add(bucket: list, item: dict, limit: int) -> None:
    seen = bucket[0]["_seen"] + 1 if bucket and "_seen" in bucket[0] else len(bucket) + 1
    if len(bucket) < limit:
        item["_seen"] = seen
        bucket.append(item)
        return
    replace_idx = random.randrange(seen)
    bucket[0]["_seen"] = seen
    if replace_idx < limit:
        item["_seen"] = seen
        bucket[replace_idx] = item


def zscore(wave: np.ndarray) -> np.ndarray:
    return (wave - wave.mean()) / max(float(wave.std()), 1e-6)


def plot_reason_examples(reason: str, examples: list[dict], output_dir: Path) -> None:
    if not examples:
        return
    n = min(len(examples), 12)
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.5), sharex=True)
    axes = axes.ravel()
    for ax, ex in zip(axes, examples[:n]):
        wave = ex["wave"]
        ax.plot(wave, color="#4c78a8", linewidth=1.3)
        ax.axvline(ex["peak_idx"], color="#d62728", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"s{ex['sample_id']} t{ex['token_index']} L{ex['lead_id']}\n"
            f"peak={ex['peak_idx']} flat={ex['max_flat_run']}",
            fontsize=8,
        )
        ax.grid(alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Filtered examples: {reason}")
    fig.tight_layout()
    fig.savefig(output_dir / f"filtered_examples_{reason}.png", dpi=180)
    plt.close(fig)


def plot_overlay(reason: str, examples: list[dict], output_dir: Path) -> None:
    if not examples:
        return
    waves = np.asarray([ex["wave"] for ex in examples], dtype=np.float32)
    norm = np.asarray([zscore(w) for w in waves], dtype=np.float32)
    x = np.arange(waves.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for wave in waves:
        axes[0].plot(x, wave, color="#4c78a8", alpha=0.22, linewidth=0.9)
    axes[0].plot(x, waves.mean(axis=0), color="#d1495b", linewidth=2)
    axes[0].set_title("Raw overlay")
    for wave in norm:
        axes[1].plot(x, wave, color="#59a14f", alpha=0.22, linewidth=0.9)
    axes[1].plot(x, norm.mean(axis=0), color="#d1495b", linewidth=2)
    axes[1].set_title("Z-score overlay")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlabel("time index")
    fig.suptitle(f"Filtered overlay: {reason}")
    fig.tight_layout()
    fig.savefig(output_dir / f"filtered_overlay_{reason}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_path.expanduser().resolve()
    data = np.load(data_path / f"{args.split}_data.npy", mmap_mode="r")
    chans = np.load(data_path / f"{args.split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_path / f"{args.split}_data_in_times.npy", mmap_mode="r")

    valid_total = 0
    kept_total = 0
    reason_counts = defaultdict(int)
    primary_counts = defaultdict(int)
    samples_by_reason = defaultdict(list)
    feature_sums = defaultdict(float)

    priority = ("long_flat_run", "prefix_flat", "suffix_flat", "peak_too_early", "peak_too_late")

    for sample_id in range(data.shape[0]):
        valid_positions = np.where((chans[sample_id] > 0) & (times[sample_id] > 0))[0]
        for token_index in valid_positions:
            wave = np.asarray(data[sample_id, token_index], dtype=np.float32)
            reasons, features = reasons_for_wave(wave, args)
            valid_total += 1
            for key, value in features.items():
                if isinstance(value, bool):
                    feature_sums[key] += float(value)
                else:
                    feature_sums[key] += value
            if not reasons:
                kept_total += 1
                continue
            for reason in reasons:
                reason_counts[reason] += 1
                item = {
                    "sample_id": int(sample_id),
                    "token_index": int(token_index),
                    "lead_id": int(chans[sample_id, token_index]),
                    "time_id": int(times[sample_id, token_index]),
                    "wave": wave.copy(),
                    **features,
                }
                reservoir_add(samples_by_reason[reason], item, args.sample_per_reason)
            for reason in priority:
                if reason in reasons:
                    primary_counts[reason] += 1
                    break
        if sample_id == 0 or sample_id == data.shape[0] - 1 or sample_id % 1000 == 0:
            print(f"Scanned {sample_id + 1}/{data.shape[0]} samples", flush=True)

    filtered_total = valid_total - kept_total
    summary = {
        "split": args.split,
        "num_samples": int(data.shape[0]),
        "valid_tokens_before_filter": int(valid_total),
        "kept_tokens_after_filter": int(kept_total),
        "filtered_tokens_union": int(filtered_total),
        "kept_fraction": float(kept_total / valid_total) if valid_total else 0.0,
        "filtered_fraction": float(filtered_total / valid_total) if valid_total else 0.0,
        "rule_counts_overlap_allowed": {
            key: {
                "count": int(reason_counts[key]),
                "fraction_of_valid": float(reason_counts[key] / valid_total) if valid_total else 0.0,
            }
            for key in priority
        },
        "primary_rule_counts_no_overlap": {
            key: {
                "count": int(primary_counts[key]),
                "fraction_of_valid": float(primary_counts[key] / valid_total) if valid_total else 0.0,
                "fraction_of_filtered": float(primary_counts[key] / filtered_total) if filtered_total else 0.0,
            }
            for key in priority
        },
        "mean_features_all_valid": {
            key: float(value / valid_total) if valid_total else 0.0 for key, value in feature_sums.items()
        },
        "filter_config": {
            "max_flat_run": int(args.max_flat_run),
            "edge_len": int(args.edge_len),
            "peak_idx_min": int(args.peak_idx_min),
            "peak_idx_max": int(args.peak_idx_max),
            "eps": float(args.eps),
        },
    }

    with open(args.output_dir / "artifact_filter_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    rows = []
    for reason in priority:
        for item in samples_by_reason[reason]:
            row = {k: v for k, v in item.items() if k not in ("wave", "_seen")}
            row["reason"] = reason
            rows.append(row)
        plot_reason_examples(reason, samples_by_reason[reason], args.output_dir)
        plot_overlay(reason, samples_by_reason[reason], args.output_dir)
    if rows:
        import csv

        with open(args.output_dir / "filtered_example_index.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
