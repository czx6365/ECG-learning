#!/usr/bin/env python3
"""Combine NoContinuous mean, mean+max, and attention-pooling probe results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "r2", "omi_r2", "nomi_r2", "control_r2", "healthy_r2",
    "accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean_max_table", type=Path, required=True)
    parser.add_argument("--attention_table", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for row in read_csv(args.mean_max_table):
        source_rep = row["representation"]
        if source_rep not in {"z_e", "z_q", "z_e_mean_max", "z_q_mean_max"}:
            continue
        row = dict(row)
        row["pooling"] = "mean_max" if source_rep.endswith("_mean_max") else "mean"
        row["representation"] = source_rep.replace("_mean_max", "")
        rows.append(row)
    rows.extend(read_csv(args.attention_table))
    order = {("z_e", "mean"): 0, ("z_e", "mean_max"): 1, ("z_e", "attention"): 2,
             ("z_q", "mean"): 3, ("z_q", "mean_max"): 4, ("z_q", "attention"): 5}
    rows.sort(key=lambda row: order[(row["representation"], row["pooling"])])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "representation", "pooling", *METRICS,
                  "regression_best_epoch", "classification_best_epoch"]
    with (args.output_dir / "pooling_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    indexed = {(row["representation"], row["pooling"]): row for row in rows}
    deltas = {}
    for rep in ("z_e", "z_q"):
        attention = indexed[(rep, "attention")]
        deltas[rep] = {}
        for baseline in ("mean", "mean_max"):
            deltas[rep][f"attention_minus_{baseline}"] = {
                metric: float(attention[metric]) - float(indexed[(rep, baseline)][metric])
                for metric in METRICS
            }
    payload = {"rows": rows, "deltas": deltas}
    (args.output_dir / "pooling_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
