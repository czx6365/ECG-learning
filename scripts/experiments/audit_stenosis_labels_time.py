#!/usr/bin/env python3
"""按固定患者划分审计狭窄标签、ECG/DSA 时间差与简单基线。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
META = ROOT / "datasets/ecg_datasets/ECG_DSA/qrs_sid/e3_healthy_patient_replay_20261011"
OUT = ROOT / "results/ecg_representation_comparison/stenosis_audit"
LABELS = ["OMI", "NOMI", "Clinical Control"]


def describe(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    quantiles = np.quantile(a, [0, .1, .25, .5, .75, .9, 1])
    return {"n": len(a), "mean": float(a.mean()), "std": float(a.std()),
            "min_p10_p25_p50_p75_p90_max": quantiles.tolist()}


def score(y, p):
    return {"n": len(y), "r2": float(r2_score(y, p)) if len(y) >= 2 and np.std(y) > 0 else None,
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p))}


def nearest_hours(row):
    dsa = pd.to_datetime(row["冠脉造影时间"], errors="coerce")
    if pd.isna(dsa):
        return np.nan, np.nan, 0
    dates = str(row["心电图时间（日期）"]).split(";")
    times = str(row["心电图时间（时刻）"]).split(";")
    deltas = []
    for index, date in enumerate(dates):
        time = times[index] if index < len(times) else ""
        ecg = pd.to_datetime(f"{date.strip()} {time.strip()}", format="%m-%d-%Y %H:%M:%S", errors="coerce")
        if pd.notna(ecg):
            deltas.append((dsa - ecg).total_seconds() / 3600)
    if not deltas:
        return np.nan, np.nan, 0
    before = [x for x in deltas if x >= 0]
    return min(deltas, key=abs), min(before) if before else np.nan, len(deltas)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(META / f"{split}_metadata.csv", low_memory=False)
        frame = frame.loc[frame["label_id"].isin([0, 1, 2])].copy()
        frame["overall_max_stenosis"] = pd.to_numeric(frame["overall_max_stenosis"], errors="coerce")
        assert frame["overall_max_stenosis"].notna().all()
        frame[["nearest_signed_hours", "nearest_pre_dsa_hours", "n_parsed_ecgs"]] = pd.DataFrame(
            [nearest_hours(row) for _, row in frame.iterrows()], index=frame.index)
        frames[split] = frame
    train_mean = frames["train"]["overall_max_stenosis"].mean()
    train_median = frames["train"]["overall_max_stenosis"].median()
    out = {"train_mean": float(train_mean), "train_median": float(train_median),
           "splits": {}, "time_error_audit_model": "E1-2_CLS_z_e seed0; descriptive only"}
    for split, frame in frames.items():
        y = frame["overall_max_stenosis"].to_numpy(float)
        section = {"records": len(frame), "label_distribution": describe(y),
                   "unique_entities": int(frame["entity_id"].nunique()),
                   "entities_with_conflicting_diagnosis": int((frame.groupby("entity_id")["label_id"].nunique() > 1).sum()),
                   "duplicate_ecg_csv_names": int(frame["csv_name"].duplicated().sum()),
                   "ecg_csv_names_with_conflicting_diagnosis": int((frame.groupby("csv_name")["label_id"].nunique() > 1).sum()),
                   "zero_count": int((y == 0).sum()), "hundred_count": int((y == 100).sum()),
                   "values_ge_90": int((y >= 90).sum()), "distinct_values": int(frame["overall_max_stenosis"].nunique()),
                   "multiple_ecgs": int((frame["n_parsed_ecgs"] > 1).sum()),
                   "nearest_abs_hours": describe(np.abs(frame["nearest_signed_hours"])),
                   "nearest_pre_dsa_hours": describe(frame["nearest_pre_dsa_hours"]),
                   "missing_all_ecg_time": int(frame["nearest_signed_hours"].isna().sum()),
                   "no_pre_dsa_ecg": int(frame["nearest_pre_dsa_hours"].isna().sum()),
                   "baselines": {"train_mean": score(y, np.full_like(y, train_mean)),
                                 "train_median": score(y, np.full_like(y, train_median))},
                   "groups": {}}
        for index, name in enumerate(LABELS):
            selected = frame["label_id"].to_numpy() == index
            section["groups"][name] = {"label": describe(y[selected]),
                                        "train_mean_baseline": score(y[selected], np.full(selected.sum(), train_mean))}
        if split == "test":
            prediction = pd.read_csv(ROOT / "results/ecg_representation_comparison/experiment1_dual_encoder/stenosis/E1-2_CLS_z_e/seed0/test_predictions.csv")
            assert len(prediction) == len(frame)
            # 同一患者可有多条记录；必须按缓存原始行序核对，不能只以 entity_id 关联。
            aligned = frame.reset_index(drop=True)
            assert np.array_equal(aligned["entity_id"].astype(str).to_numpy(), prediction["entity_id"].astype(str).to_numpy())
            assert np.array_equal(aligned["label_id"].to_numpy(), prediction["label_id"].to_numpy())
            errors = (prediction["prediction"].to_numpy() - prediction["target_overall_max_stenosis"].to_numpy())
            intervals = [(-np.inf, 0), (0, 24), (24, 72), (72, 168), (168, np.inf)]
            section["model_error_by_pre_dsa_interval_hours"] = {}
            time = aligned["nearest_pre_dsa_hours"].to_numpy(float)
            for low, high in intervals:
                m = (time > low) & (time <= high)
                key = f"({low},{high}]"
                section["model_error_by_pre_dsa_interval_hours"][key] = {
                    "n": int(m.sum()), "mae": float(np.mean(np.abs(errors[m]))) if m.any() else None}
            section["model_error_by_label"] = {}
            for index, name in enumerate(LABELS):
                m = prediction["label_id"].to_numpy() == index
                section["model_error_by_label"][name] = score(
                    prediction.loc[m, "target_overall_max_stenosis"], prediction.loc[m, "prediction"])
        out["splits"][split] = section
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({split: {"n": out["splits"][split]["records"],
                              "baseline_r2": out["splits"][split]["baselines"]["train_mean"]["r2"],
                              "time_parse_missing": out["splits"][split]["missing_all_ecg_time"]}
                      for split in out["splits"]}))


if __name__ == "__main__":
    main()
