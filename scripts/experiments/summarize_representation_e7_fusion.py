#!/usr/bin/env python3
"""E7 强基线与形态融合三种子、按患者聚类 bootstrap 汇总。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
BASE = ROOT / "results/ecg_representation_comparison/experiment1_e7_meanmax_fusion"
METHODS = ("E7_token_meanmax", "E7_token_meanmax_z_e")


def score(task, frame):
    if task == "stenosis":
        return float(r2_score(frame["target_overall_max_stenosis"], frame["prediction"]))
    logits = frame[[f"logit_{name}" for name in ("OMI", "NOMI", "Clinical Control", "Healthy")]].to_numpy()
    exp = np.exp(logits - logits.max(1, keepdims=True))
    prob = exp / exp.sum(1, keepdims=True)
    y = frame["label_id"].to_numpy()
    return float(np.mean([roc_auc_score(y == k, prob[:, k]) for k in range(4)]))


def main():
    if not (BASE / "COMPLETE").exists():
        raise RuntimeError("E7 对照尚未完成")
    output = {"methods": {}, "paired_entity_bootstrap": {}}
    for task in ("classification", "stenosis"):
        rows = {}
        preds = {}
        key = "macro_auc_ovr" if task == "classification" else "r2"
        for method in METHODS:
            rows[method] = [json.loads((BASE / task / method / f"seed{seed}" / "summary.json").read_text()) for seed in range(3)]
            preds[method] = [pd.read_csv(BASE / task / method / f"seed{seed}" / "test_predictions.csv") for seed in range(3)]
            output["methods"][f"{task}:{method}"] = {
                "input_dim": rows[method][0]["input_dim"], "trainable_parameters": rows[method][0]["trainable_parameters"],
                "test_n": rows[method][0]["test"]["n"], "best_epochs": [r["best_epoch"] for r in rows[method]],
                "val_primary_mean": float(np.mean([r["val"][key] for r in rows[method]])),
                "test_metrics": {field: {"mean": float(np.mean([r["test"][field] for r in rows[method]])),
                                        "sd": float(np.std([r["test"][field] for r in rows[method]], ddof=1))}
                                 for field in (("macro_auc_ovr", "macro_f1", "accuracy", "balanced_accuracy", "omi_nomi_auc")
                                               if task == "classification" else ("r2", "rmse", "mae", "pearson", "spearman"))}}
        anchor = preds[METHODS[0]][0]
        for group in preds.values():
            for frame in group:
                assert frame["entity_id"].equals(anchor["entity_id"])
                assert frame["csv_name"].equals(anchor["csv_name"])
                assert frame["label_id"].equals(anchor["label_id"])
        ids = anchor["entity_id"].astype(str).to_numpy()
        unique = np.unique(ids)
        loc = {name: np.flatnonzero(ids == name) for name in unique}
        rng = np.random.default_rng(20261016)
        point = float(np.mean([score(task, preds[METHODS[1]][seed]) - score(task, preds[METHODS[0]][seed])
                               for seed in range(3)]))
        boot = []
        for _ in range(500):
            sampled = rng.choice(unique, len(unique), replace=True)
            ix = np.concatenate([loc[name] for name in sampled])
            try:
                boot.append(float(np.mean([score(task, preds[METHODS[1]][seed].iloc[ix]) -
                                           score(task, preds[METHODS[0]][seed].iloc[ix]) for seed in range(3)])))
            except ValueError:
                pass
        output["paired_entity_bootstrap"][task] = {"point_delta": point,
                                                    "ci95": np.quantile(boot, [.025, .975]).tolist(),
                                                    "resamples": len(boot), "entities": len(unique)}
    (BASE / "aggregated_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output["paired_entity_bootstrap"]))


if __name__ == "__main__":
    main()
