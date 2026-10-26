#!/usr/bin/env python3
"""Split high-correlation adjacent heartbeat SID disagreement by physiology labels.

This script reuses saved SID assignments and the cached physiology labels from
Morphology-V2 training/evaluation. It does not run model inference or rewrite
any SID files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DISCRETE_LABELS = ("r_polarity", "t_polarity", "st_state", "qrs_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "High-correlation adjacent pair physiology-stratified SID disagreement."
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--phys_cache", type=Path, required=True)
    parser.add_argument("--v1_sid", type=Path, required=True)
    parser.add_argument("--v2_sid", type=Path, required=True)
    parser.add_argument("--v1_name", default="Morphology-V1")
    parser.add_argument("--v2_name", default="Morphology-V2")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_shift", type=int, default=4)
    parser.add_argument("--corr_threshold", type=float, default=0.9)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--lead_id", type=int, default=0, help="If >0, evaluate only this channel id.")
    return parser.parse_args()


def rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(a.shape[0], dtype=np.float32)
    valid = denom > 1e-8
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return np.clip(out, -1.0, 1.0)


def adjacent_shift_corr(waves: np.ndarray, max_shift: int) -> np.ndarray:
    """Best Pearson correlation for adjacent waves allowing local sample shifts."""
    if len(waves) < 2:
        return np.empty(0, dtype=np.float32)
    a = waves[:-1].astype(np.float32, copy=False)
    b = waves[1:].astype(np.float32, copy=False)
    best = np.full(a.shape[0], -np.inf, dtype=np.float32)
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            aa, bb = a, b
        elif shift > 0:
            aa, bb = a[:, shift:], b[:, :-shift]
        else:
            shift_abs = -shift
            aa, bb = a[:, :-shift_abs], b[:, shift_abs:]
        if aa.shape[1] < 8:
            continue
        best = np.maximum(best, rowwise_corr(aa, bb))
    best[~np.isfinite(best)] = 0.0
    return best


def valid_token_mask(data: np.ndarray, chans: np.ndarray, times: np.ndarray, lead_id: int = 0) -> np.ndarray:
    valid = (chans > 0) & (times > 0) & (np.abs(data).max(axis=-1) > 1e-8)
    if lead_id > 0:
        valid &= chans == lead_id
    return valid


def init_counts() -> dict[str, int]:
    return {
        "same_phys_pairs": 0,
        "same_phys_sid_diff": 0,
        "diff_phys_pairs": 0,
        "diff_phys_sid_diff": 0,
    }


def update_counts(
    counts: dict[str, int],
    sids: np.ndarray,
    high_corr: np.ndarray,
    same_phys: np.ndarray,
) -> None:
    sid_diff = sids[:-1] != sids[1:]
    same_mask = high_corr & same_phys
    diff_mask = high_corr & ~same_phys
    counts["same_phys_pairs"] += int(same_mask.sum())
    counts["same_phys_sid_diff"] += int((same_mask & sid_diff).sum())
    counts["diff_phys_pairs"] += int(diff_mask.sum())
    counts["diff_phys_sid_diff"] += int((diff_mask & sid_diff).sum())


def safe_rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else float("nan")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_path / f"{args.split}_data.npy", mmap_mode="r")
    chans = np.load(args.data_path / f"{args.split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(args.data_path / f"{args.split}_data_in_times.npy", mmap_mode="r")
    phys = np.load(args.phys_cache / f"{args.split}_phys_discrete.npy", mmap_mode="r")
    sid_runs = {
        args.v1_name: np.load(args.v1_sid, mmap_mode="r"),
        args.v2_name: np.load(args.v2_sid, mmap_mode="r"),
    }

    rows = []
    per_run_counts = {name: init_counts() for name in sid_runs}
    total_adjacent_pairs = 0
    total_high_corr_pairs = 0
    high_corr_same_phys_pairs = 0
    high_corr_diff_phys_pairs = 0

    for row_idx in range(data.shape[0]):
        row_data = np.asarray(data[row_idx])
        row_chans = np.asarray(chans[row_idx])
        row_times = np.asarray(times[row_idx])
        row_phys = np.asarray(phys[row_idx])
        valid = valid_token_mask(row_data, row_chans, row_times, args.lead_id)
        valid &= np.all(row_phys >= 0, axis=1)
        if not valid.any():
            continue

        for lead in np.unique(row_chans[valid]).astype(int):
            token_indices = np.flatnonzero(valid & (row_chans == lead))
            if len(token_indices) < 2:
                continue
            # Same record + same lead; sort by stored time first, then token index.
            order = np.lexsort((token_indices, row_times[token_indices]))
            token_indices = token_indices[order]

            waves = row_data[token_indices]
            shift_corr = adjacent_shift_corr(waves, args.max_shift)
            if len(shift_corr) == 0:
                continue
            high_corr = shift_corr > args.corr_threshold
            if not high_corr.any():
                total_adjacent_pairs += int(len(shift_corr))
                continue

            pair_phys_a = row_phys[token_indices[:-1]]
            pair_phys_b = row_phys[token_indices[1:]]
            same_phys = np.all(pair_phys_a == pair_phys_b, axis=1)

            total_adjacent_pairs += int(len(shift_corr))
            total_high_corr_pairs += int(high_corr.sum())
            high_corr_same_phys_pairs += int((high_corr & same_phys).sum())
            high_corr_diff_phys_pairs += int((high_corr & ~same_phys).sum())

            for name, sid_array in sid_runs.items():
                sids = np.asarray(sid_array[row_idx, token_indices])
                sid_valid = (sids >= 0) & (sids < args.codebook_size)
                if not sid_valid.all():
                    pair_sid_valid = sid_valid[:-1] & sid_valid[1:]
                    update_counts(
                        per_run_counts[name],
                        sids,
                        high_corr & pair_sid_valid,
                        same_phys,
                    )
                else:
                    update_counts(per_run_counts[name], sids, high_corr, same_phys)

        if (row_idx + 1) % 1000 == 0:
            print(
                f"Processed {row_idx + 1}/{data.shape[0]} records; "
                f"high-corr pairs={total_high_corr_pairs}",
                flush=True,
            )

    for name, counts in per_run_counts.items():
        rows.append(
            {
                "run": name,
                "same_phys_pairs": counts["same_phys_pairs"],
                "same_phys_sid_diff": counts["same_phys_sid_diff"],
                "same_phys_sid_disagreement": safe_rate(
                    counts["same_phys_sid_diff"], counts["same_phys_pairs"]
                ),
                "diff_phys_pairs": counts["diff_phys_pairs"],
                "diff_phys_sid_diff": counts["diff_phys_sid_diff"],
                "diff_phys_sid_disagreement": safe_rate(
                    counts["diff_phys_sid_diff"], counts["diff_phys_pairs"]
                ),
                "physiology_discrimination_margin": (
                    safe_rate(counts["diff_phys_sid_diff"], counts["diff_phys_pairs"])
                    - safe_rate(counts["same_phys_sid_diff"], counts["same_phys_pairs"])
                ),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "high_corr_phys_sid_disagreement.csv", index=False)

    metric_rows = []
    v1 = result[result["run"] == args.v1_name].iloc[0]
    v2 = result[result["run"] == args.v2_name].iloc[0]
    for metric, label in [
        ("same_phys_sid_disagreement", "Same physiology SID disagreement"),
        ("diff_phys_sid_disagreement", "Different physiology SID disagreement"),
        ("physiology_discrimination_margin", "Physiology Discrimination Margin"),
    ]:
        metric_rows.append(
            {
                "Metric": label,
                args.v1_name: float(v1[metric]),
                args.v2_name: float(v2[metric]),
                "Delta": float(v2[metric] - v1[metric]),
            }
        )
    table = pd.DataFrame(metric_rows)
    table.to_csv(args.output_dir / "high_corr_phys_sid_disagreement_delta.csv", index=False)

    summary = {
        "split": args.split,
        "max_shift_samples": args.max_shift,
        "corr_threshold": args.corr_threshold,
        "lead_id_filter": args.lead_id if args.lead_id > 0 else None,
        "discrete_physiology_labels": list(DISCRETE_LABELS),
        "total_adjacent_pairs": int(total_adjacent_pairs),
        "total_high_corr_pairs": int(total_high_corr_pairs),
        "high_corr_pair_coverage": safe_rate(total_high_corr_pairs, total_adjacent_pairs),
        "high_corr_same_phys_pairs": int(high_corr_same_phys_pairs),
        "high_corr_diff_phys_pairs": int(high_corr_diff_phys_pairs),
        "runs": rows,
    }
    (args.output_dir / "high_corr_phys_sid_disagreement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== High-corr physiology-stratified SID disagreement ===")
    print(table.to_string(index=False))
    print("\nPair counts:")
    print(result.to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
