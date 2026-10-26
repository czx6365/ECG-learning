#!/usr/bin/env python3
"""Average fixed test predictions from C2.2b seeds and summarize R2."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score


LABELS = ["OMI", "NOMI", "Clinical Control", "Healthy"]


def read_matrix(path: Path, prefix: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([[float(row[f"{prefix}_{name}"]) for name in LABELS] for row in rows])


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    mse = np.mean((pred - target) ** 2, axis=0)
    mae = np.mean(np.abs(pred - target), axis=0)
    per_r2 = [float(r2_score(target[:, i], pred[:, i])) for i in range(4)]
    return {
        "r2": float(np.mean(per_r2)),
        "per_output_r2": dict(zip(LABELS, per_r2)),
        "mse": float(mse.mean()),
        "rmse": float(math.sqrt(mse.mean())),
        "mae": float(mae.mean()),
        "per_output_mse": dict(zip(LABELS, map(float, mse))),
        "per_output_mae": dict(zip(LABELS, map(float, mae))),
        "n_test": int(len(target)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    predictions = [read_matrix(path / "test_outputs.csv", "pred") for path in args.run_dirs]
    targets = [read_matrix(path / "test_targets.csv", "target") for path in args.run_dirs]
    if not all(np.array_equal(targets[0], target) for target in targets[1:]):
        raise RuntimeError("Test targets or ordering differ across seeds")
    summaries = [json.loads((path / "test_regression_summary.json").read_text()) for path in args.run_dirs]
    ensemble = metrics(np.mean(predictions, axis=0), targets[0])
    overall = np.asarray([item["r2"] for item in summaries], dtype=float)
    per_output = {
        name: np.asarray([item["per_output_r2"][name] for item in summaries], dtype=float)
        for name in LABELS
    }
    result = {
        "seeds": [0, 1, 2],
        "run_dirs": [str(path) for path in args.run_dirs],
        "individual": summaries,
        "individual_mean_std": {
            "r2_mean": float(overall.mean()),
            "r2_std": float(overall.std(ddof=0)),
            "per_output_r2_mean": {name: float(values.mean()) for name, values in per_output.items()},
            "per_output_r2_std": {name: float(values.std(ddof=0)) for name, values in per_output.items()},
        },
        "ensemble": ensemble,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multiseed_ensemble_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
