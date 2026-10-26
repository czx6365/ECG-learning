#!/usr/bin/env python3
"""Audit whether H-clean filtering disproportionately removes clinical ECG tokens.

This script compares an original QRS-token dataset with its cleaned copy. A token
is treated as filtered when it was valid in the original arrays but invalid in the
clean arrays. The audit summarizes filtered-rate shifts by source, label,
stenosis, lead, patient, and simple waveform morphology features.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LEAD_NAMES = {
    1: "I",
    2: "II",
    3: "III",
    4: "V1",
    5: "V2",
    6: "V3",
    7: "V4",
    8: "V5",
    9: "V6",
}

CLINICAL_COLUMNS = [
    "label_name",
    "label_id",
    "标签",
    "可疑罪犯血管",
    "patient_sn",
    "gender",
    "年龄",
    "lm_max_stenosis",
    "lad_max_stenosis",
    "lca_max_stenosis",
    "rca_max_stenosis",
    "overall_max_stenosis",
    "llm_has_ge90",
    "是否接受了PCI治疗",
    "最低TIMI分级",
    "诊断结论_1",
    "诊断结论_2",
    "诊断结论_3",
    "诊断结论_4",
    "诊断结论_5",
    "诊断结论_6",
    "出院诊断",
]

TEXT_FEATURES = {
    "dx_ST": "ST",
    "dx_T_wave": "T波",
    "dx_Q_wave": "Q波",
    "dx_MI": "心肌梗死",
    "dx_ischemia": "缺血",
    "dx_injury": "损伤",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit H-clean token filtering bias.")
    parser.add_argument("--raw_data_path", type=Path, required=True)
    parser.add_argument("--clean_data_path", type=Path, required=True)
    parser.add_argument("--label_metadata", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--sample_per_bucket", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-4)
    return parser.parse_args()


def load_metadata(raw_data_path: Path, split: str, label_metadata: Path | None) -> pd.DataFrame:
    meta = pd.read_csv(raw_data_path / f"{split}_metadata.csv")
    meta = meta.reset_index().rename(columns={"index": "sample_id"})
    meta["csv_key"] = meta["csv_name"].astype(str).str.replace(".csv", "", regex=False)

    if label_metadata and label_metadata.exists():
        label_df = pd.read_csv(label_metadata)
        if "csv_name" not in label_df.columns and "csv_name_expected" in label_df.columns:
            label_df["csv_name"] = label_df["csv_name_expected"]
        label_df["csv_key"] = label_df["csv_name"].astype(str).str.replace(".csv", "", regex=False)
        keep_cols = ["csv_key"] + [col for col in CLINICAL_COLUMNS if col in label_df.columns]
        label_df = label_df[keep_cols].drop_duplicates("csv_key")
        meta = meta.merge(label_df, on="csv_key", how="left", suffixes=("", "_label"))

    meta["cohort_group"] = np.where(meta.get("cohort", "").astype(str).eq("healthy"), "healthy", "patient_or_unknown")
    if "label_name" not in meta.columns:
        meta["label_name"] = np.nan
    meta["label_name_filled"] = meta["label_name"].fillna(meta["cohort_group"])

    for col in ["overall_max_stenosis", "lm_max_stenosis", "lad_max_stenosis", "lca_max_stenosis", "rca_max_stenosis"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    if "overall_max_stenosis" in meta.columns:
        meta["stenosis_bin"] = pd.cut(
            meta["overall_max_stenosis"],
            bins=[-math.inf, 0, 50, 70, 90, math.inf],
            labels=["0", "1-49", "50-69", "70-89", ">=90"],
        ).astype(object).fillna("unknown")
    else:
        meta["stenosis_bin"] = "unknown"

    text_cols = [col for col in CLINICAL_COLUMNS if col in meta.columns and col.startswith("诊断结论_")]
    if "出院诊断" in meta.columns:
        text_cols.append("出院诊断")
    text = meta[text_cols].fillna("").astype(str).agg(";".join, axis=1) if text_cols else pd.Series("", index=meta.index)
    for feature, pattern in TEXT_FEATURES.items():
        meta[feature] = text.str.contains(pattern, regex=False).map({True: "yes", False: "no"})
    return meta


def max_flat_run_batch(waves: np.ndarray, eps: float) -> np.ndarray:
    flat = np.abs(np.diff(waves, axis=-1)) <= eps
    flat2 = flat.reshape(-1, flat.shape[-1])
    best = np.ones(flat2.shape[0], dtype=np.int16)
    cur = np.ones(flat2.shape[0], dtype=np.int16)
    for idx in range(flat2.shape[1]):
        cur = np.where(flat2[:, idx], cur + 1, 1)
        best = np.maximum(best, cur)
    return best.reshape(waves.shape[:-1])


def token_features(waves: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    peak_idx = np.argmax(np.abs(waves), axis=-1)
    max_abs = np.max(np.abs(waves), axis=-1)
    width = (np.abs(waves) >= (0.5 * np.maximum(max_abs[..., None], 1e-8))).sum(axis=-1)
    return {
        "peak_idx": peak_idx.astype(np.float32),
        "max_abs": max_abs.astype(np.float32),
        "std": np.std(waves, axis=-1).astype(np.float32),
        "range": (np.max(waves, axis=-1) - np.min(waves, axis=-1)).astype(np.float32),
        "qrs_width_halfmax": width.astype(np.float32),
        "st_level_60_72": np.mean(waves[..., 60:72], axis=-1).astype(np.float32),
        "t_abs_72_95": np.max(np.abs(waves[..., 72:]), axis=-1).astype(np.float32),
        "max_flat_run": max_flat_run_batch(waves, eps).astype(np.float32),
    }


def reason_masks(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "long_flat_run": features["max_flat_run"] >= 50,
        "peak_too_early": features["peak_idx"] < 40,
        "peak_too_late": features["peak_idx"] > 65,
    }


def update_group(rows: list[dict], split: str, name: str, values: np.ndarray, filtered: np.ndarray) -> None:
    frame = pd.DataFrame({"value": values.astype(object), "filtered": filtered.astype(bool)})
    grouped = frame.groupby("value", dropna=False)["filtered"].agg(["count", "sum"]).reset_index()
    for _, row in grouped.iterrows():
        count = int(row["count"])
        filt = int(row["sum"])
        rows.append(
            {
                "split": split,
                "group": name,
                "value": str(row["value"]),
                "valid_tokens": count,
                "filtered_tokens": filt,
                "kept_tokens": count - filt,
                "filtered_fraction": float(filt / count) if count else 0.0,
            }
        )


def sample_add(bucket: list[dict], item: dict, limit: int, rng: np.random.Generator) -> None:
    seen = bucket[0]["_seen"] + 1 if bucket and "_seen" in bucket[0] else len(bucket) + 1
    if len(bucket) < limit:
        item["_seen"] = seen
        bucket.append(item)
        return
    replace_idx = int(rng.integers(0, seen))
    bucket[0]["_seen"] = seen
    if replace_idx < limit:
        item["_seen"] = seen
        bucket[replace_idx] = item


def plot_examples(examples: list[dict], title: str, output_path: Path) -> None:
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
            f"{ex['split']} s{ex['sample_id']} t{ex['token_index']} {ex['lead_name']}\n"
            f"{ex['label']} sten={ex['stenosis_bin']}",
            fontsize=8,
        )
        ax.grid(alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fraction_table(df: pd.DataFrame, group: str, output_path: Path, top_n: int = 30) -> None:
    sub = df[df["group"].eq(group)].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["split", "valid_tokens"], ascending=[True, False]).head(top_n)
    labels = sub["split"] + ":" + sub["value"].astype(str)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(sub))))
    ax.barh(np.arange(len(sub)), sub["filtered_fraction"] * 100, color="#4c78a8")
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Filtered tokens (%)")
    ax.set_title(f"Filtered fraction by {group}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_feature_rows(rows: list[dict], split: str, features: dict[str, np.ndarray], valid: np.ndarray, filtered: np.ndarray) -> None:
    for name, arr in features.items():
        arr = arr[valid]
        filt = filtered[valid]
        for subset_name, mask in [("kept", ~filt), ("filtered", filt)]:
            vals = arr[mask]
            if vals.size == 0:
                continue
            rows.append(
                {
                    "split": split,
                    "feature": name,
                    "subset": subset_name,
                    "count": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p50": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                }
            )


def audit_split(args: argparse.Namespace, split: str, rng: np.random.Generator) -> dict:
    raw_data = np.load(args.raw_data_path / f"{split}_data.npy", mmap_mode="r")
    raw_chans = np.load(args.raw_data_path / f"{split}_data_in_chans.npy", mmap_mode="r")
    raw_times = np.load(args.raw_data_path / f"{split}_data_in_times.npy", mmap_mode="r")
    clean_chans = np.load(args.clean_data_path / f"{split}_data_in_chans.npy", mmap_mode="r")
    clean_times = np.load(args.clean_data_path / f"{split}_data_in_times.npy", mmap_mode="r")
    meta = load_metadata(args.raw_data_path, split, args.label_metadata)

    group_rows: list[dict] = []
    patient_counts = defaultdict(lambda: {"valid_tokens": 0, "filtered_tokens": 0})
    feature_rows: list[dict] = []
    reason_counts = defaultdict(int)
    reason_label_counts = defaultdict(lambda: defaultdict(int))
    samples = defaultdict(list)

    valid_total = 0
    filtered_total = 0
    patient_labeled_valid = 0
    patient_labeled_filtered = 0

    for start in range(0, raw_data.shape[0], args.chunk_size):
        end = min(start + args.chunk_size, raw_data.shape[0])
        waves = np.asarray(raw_data[start:end])
        chans = np.asarray(raw_chans[start:end])
        times = np.asarray(raw_times[start:end])
        clean_valid = (np.asarray(clean_chans[start:end]) > 0) & (np.asarray(clean_times[start:end]) > 0)
        raw_valid = (chans > 0) & (times > 0)
        filtered = raw_valid & ~clean_valid
        features = token_features(waves, args.eps)
        reasons = reason_masks(features)

        valid_total += int(raw_valid.sum())
        filtered_total += int(filtered.sum())

        chunk_meta = meta.iloc[start:end].reset_index(drop=True)
        positions = np.argwhere(raw_valid)
        if positions.size:
            local_sample_idx = positions[:, 0]
            token_idx = positions[:, 1]
            flat_filtered = filtered[raw_valid]
            lead_values = chans[raw_valid].astype(int)
            lead_names = np.array([LEAD_NAMES.get(int(v), str(int(v))) for v in lead_values], dtype=object)
            sample_meta = chunk_meta.iloc[local_sample_idx]

            update_group(group_rows, split, "cohort", sample_meta["cohort_group"].to_numpy(), flat_filtered)
            update_group(group_rows, split, "source_zip", sample_meta["source_zip"].astype(str).to_numpy(), flat_filtered)
            update_group(group_rows, split, "lead", lead_names, flat_filtered)
            update_group(group_rows, split, "label_name", sample_meta["label_name_filled"].to_numpy(), flat_filtered)
            update_group(group_rows, split, "stenosis_bin", sample_meta["stenosis_bin"].to_numpy(), flat_filtered)
            for feature in TEXT_FEATURES:
                if feature in sample_meta.columns:
                    update_group(group_rows, split, feature, sample_meta[feature].to_numpy(), flat_filtered)

            if "patient_sn" in sample_meta.columns:
                for pid, is_filtered in zip(sample_meta["patient_sn"].fillna("unknown").astype(str), flat_filtered):
                    if pid == "unknown" or pid == "nan":
                        continue
                    patient_counts[pid]["valid_tokens"] += 1
                    patient_counts[pid]["filtered_tokens"] += int(is_filtered)

            labeled = sample_meta["label_name"].notna().to_numpy()
            patient_labeled_valid += int(labeled.sum())
            patient_labeled_filtered += int((flat_filtered & labeled).sum())

            summarize_feature_rows(feature_rows, split, features, raw_valid, filtered)

            flat_indices = np.flatnonzero(filtered)
            flat_positions = np.column_stack(np.unravel_index(flat_indices, filtered.shape))
            for local_i, tok_i in flat_positions:
                wave = waves[local_i, tok_i].astype(float)
                abs_sample_id = start + int(local_i)
                row = chunk_meta.iloc[int(local_i)]
                lead_id = int(chans[local_i, tok_i])
                label = str(row.get("label_name_filled", "unknown"))
                item = {
                    "split": split,
                    "sample_id": abs_sample_id,
                    "token_index": int(tok_i),
                    "csv_name": str(row.get("csv_name", "")),
                    "patient_sn": str(row.get("patient_sn", "")),
                    "label": label,
                    "stenosis_bin": str(row.get("stenosis_bin", "unknown")),
                    "lead_id": lead_id,
                    "lead_name": LEAD_NAMES.get(lead_id, str(lead_id)),
                    "peak_idx": int(np.argmax(np.abs(wave))),
                    "max_flat_run": int(max_flat_run_batch(wave[None, :], args.eps)[0]),
                    "wave": wave,
                }
                sample_add(samples["filtered_all"], item, args.sample_per_bucket, rng)
                sample_add(samples[f"filtered_label_{label}"], item, args.sample_per_bucket, rng)
                for reason, mask in reasons.items():
                    if bool(mask[local_i, tok_i]):
                        reason_counts[reason] += 1
                        reason_label_counts[reason][label] += 1
                        sample_add(samples[f"reason_{reason}"], item, args.sample_per_bucket, rng)

        print(f"{split}: scanned {end}/{raw_data.shape[0]}, filtered={filtered_total}/{valid_total}", flush=True)

    patient_df = pd.DataFrame(
        [
            {
                "patient_sn": pid,
                "valid_tokens": vals["valid_tokens"],
                "filtered_tokens": vals["filtered_tokens"],
                "filtered_fraction": vals["filtered_tokens"] / vals["valid_tokens"] if vals["valid_tokens"] else 0.0,
            }
            for pid, vals in patient_counts.items()
        ]
    ).sort_values(["filtered_fraction", "valid_tokens"], ascending=[False, False])

    for key, examples in samples.items():
        for ex in examples:
            ex.pop("_seen", None)
        safe_key = key.replace("/", "_").replace(" ", "_")
        rows = [{k: v for k, v in ex.items() if k != "wave"} for ex in examples]
        if rows:
            pd.DataFrame(rows).to_csv(args.output_dir / f"{split}_{safe_key}_examples.csv", index=False)
            plot_examples(examples, f"{split}: {key}", args.output_dir / f"{split}_{safe_key}_examples.png")

    return {
        "split": split,
        "num_samples": int(raw_data.shape[0]),
        "valid_tokens": int(valid_total),
        "filtered_tokens": int(filtered_total),
        "kept_tokens": int(valid_total - filtered_total),
        "filtered_fraction": float(filtered_total / valid_total) if valid_total else 0.0,
        "patient_labeled_valid_tokens": int(patient_labeled_valid),
        "patient_labeled_filtered_tokens": int(patient_labeled_filtered),
        "patient_labeled_filtered_fraction": float(patient_labeled_filtered / patient_labeled_valid) if patient_labeled_valid else 0.0,
        "reason_counts": dict(reason_counts),
        "reason_label_counts": {reason: dict(vals) for reason, vals in reason_label_counts.items()},
        "group_rows": group_rows,
        "patient_rows": patient_df,
        "feature_rows": feature_rows,
    }


def main() -> None:
    args = parse_args()
    args.raw_data_path = args.raw_data_path.expanduser().resolve()
    args.clean_data_path = args.clean_data_path.expanduser().resolve()
    if args.label_metadata:
        args.label_metadata = args.label_metadata.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summaries = []
    all_group_rows = []
    all_patient_rows = []
    all_feature_rows = []
    for split in args.splits:
        result = audit_split(args, split, rng)
        summaries.append({k: v for k, v in result.items() if not k.endswith("_rows")})
        all_group_rows.extend(result["group_rows"])
        patient_rows = result["patient_rows"]
        if not patient_rows.empty:
            patient_rows.insert(0, "split", split)
            all_patient_rows.append(patient_rows)
        all_feature_rows.extend(result["feature_rows"])

    group_df = pd.DataFrame(all_group_rows)
    if not group_df.empty:
        total_by_split = group_df.groupby(["split", "group"])["filtered_tokens"].transform("sum")
        group_df["share_of_filtered_within_group_type"] = group_df["filtered_tokens"] / total_by_split.replace(0, np.nan)
        group_df.to_csv(args.output_dir / "filter_distribution_by_group.csv", index=False)
        for group in ["label_name", "stenosis_bin", "lead", "source_zip", "dx_ST", "dx_T_wave", "dx_Q_wave"]:
            plot_fraction_table(group_df, group, args.output_dir / f"filtered_fraction_by_{group}.png")

    if all_patient_rows:
        pd.concat(all_patient_rows, ignore_index=True).to_csv(args.output_dir / "filter_distribution_by_patient.csv", index=False)
    feature_df = pd.DataFrame(all_feature_rows)
    if not feature_df.empty:
        feature_df.to_csv(args.output_dir / "waveform_feature_comparison.csv", index=False)

    summary = {
        "raw_data_path": str(args.raw_data_path),
        "clean_data_path": str(args.clean_data_path),
        "label_metadata": str(args.label_metadata) if args.label_metadata else None,
        "splits": summaries,
    }
    with open(args.output_dir / "clean_filter_bias_audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
