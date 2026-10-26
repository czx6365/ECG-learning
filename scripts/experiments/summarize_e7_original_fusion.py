#!/usr/bin/env python3
"""原 E7 协议的三种子 CLS token Mean+Max/+z_e 聚类 bootstrap。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
BASE = ROOT / "results/ecg_representation_comparison/experiment1_e7_original_protocol_fusion"
CAP = ROOT / "results/ecg_representation_comparison/experiment1_e7_original_protocol_capacity_control"
CACHE = ROOT / "outputs/curriculum_v1/e4_e3_aggregation_20261011"
META = ROOT / "datasets/ecg_datasets/ECG_DSA/qrs_sid/e3_healthy_patient_replay_20261011"
MODES = ("E7_token_meanmax", "E7_token_meanmax_z_e", "E7_token_meanmax_matched")


def folder(name):
    return CAP if name.endswith("_matched") else BASE


def auc(raw, y):
    exp = np.exp(raw - raw.max(1, keepdims=True))
    prob = exp / exp.sum(1, keepdims=True)
    return float(np.mean([roc_auc_score(y[:, k], prob[:, k]) for k in range(4)]))


def main():
    if not (BASE / "COMPLETE").exists():
        raise RuntimeError("训练未完成")
    y = np.load(CACHE / "test_labels.npy")
    meta = pd.read_csv(META / "test_metadata.csv", low_memory=False)
    ids = meta["entity_id"].astype(str).to_numpy()
    unique = np.unique(ids)
    loc = {name: np.flatnonzero(ids == name) for name in unique}
    raw = {name: [np.load(folder(name) / name / f"seed{seed}" / "test_raw_outputs.npy") for seed in range(3)]
           for name in MODES}
    result = {"methods": {}, "entity_bootstrap": {}}
    for name in MODES:
        rows = [json.loads((folder(name) / name / f"seed{seed}" / "summary.json").read_text()) for seed in range(3)]
        result["methods"][name] = {
            "input_dim": rows[0]["input_dim"], "parameters": rows[0]["parameters"],
            "best_epochs": [r["best_epoch"] for r in rows],
            "val_auc_mean": float(np.mean([r["val"]["macro_roc_auc"] for r in rows])),
            "test_auc_mean": float(np.mean([r["test"]["macro_roc_auc"] for r in rows])),
            "test_auc_sd": float(np.std([r["test"]["macro_roc_auc"] for r in rows], ddof=1)),
            "test_macro_f1_mean": float(np.mean([r["test"]["macro_f1"] for r in rows]))}
    rng = np.random.default_rng(20261016)
    for baseline in (MODES[0], MODES[2]):
        point = float(np.mean([auc(raw[MODES[1]][s], y) - auc(raw[baseline][s], y) for s in range(3)]))
        boot = []
        for _ in range(500):
            sampled = rng.choice(unique, len(unique), replace=True)
            ix = np.concatenate([loc[name] for name in sampled])
            try:
                boot.append(float(np.mean([auc(raw[MODES[1]][s][ix], y[ix]) - auc(raw[baseline][s][ix], y[ix])
                                           for s in range(3)])))
            except ValueError:
                continue
        result["entity_bootstrap"][f"fusion_minus_{baseline}"] = {
            "delta_auc": point, "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "entities": len(unique), "valid_resamples": len(boot)}
    (BASE / "aggregated_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
