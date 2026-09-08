"""ECG-DSA 有标签数据集主构建脚本。

这个脚本属于当前正在使用的 ECG-DSA 主流程，主要负责：
- 将 Excel 标注记录与 ZIP 压缩包中的 CSV 心电文件进行匹配
- 由标准导联推导缺失的增强导联（aVR、aVL、aVF）
- 按患者维度做分层切分，生成 train/val/test
- 将波形重采样到固定长度，并使用训练集统计量做标准化
- 输出按划分保存的 .npy 数组和 metadata CSV

被以下脚本调用：scripts/pipelines/ECG_DSA/run_ecg_dsa_pipeline.py
输出目录：datasets/ecg_datasets/ECG_DSA/waveforms/base/
"""

import argparse
from contextlib import ExitStack
import json
import math
import zipfile
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm import tqdm

# 允许直接以 `python scripts/data_prep/ECG_DSA/make_ecg_dsa_dataset.py` 方式运行。
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import ECG_DSA_RAW_SOURCE_DIR, ECG_DSA_WAVEFORM_DIR
from scripts.data_prep.ECG_DSA.ecg_zip_utils import (
    build_member_table,
    resample_waveforms,
    load_waveform,
    standardize_in_place,
    CSV_LEADS,
)


STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_LABEL_COL = "解剖学狭窄"
DEFAULT_TASK_TYPE = "regression"
TARGET_LENGTH = 1000
NSPLITS = 10


def build_member_map(zip_paths: list[Path]) -> dict[str, dict[str, str]]:
    """Build a map from CSV basename to ZIP member info.

    This is a wrapper around build_member_table that returns a dict instead of DataFrame.
    Used for label matching logic in this script.
    """
    table = build_member_table(zip_paths)
    member_map = {}
    for _, row in table.iterrows():
        member_map[row["csv_name"]] = {
            "member_name": row["member_name"],
            "source_zip": row["source_zip"],
            "source_zip_path": row["source_zip_path"],
        }
    return member_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ECG-DSA train/val/test datasets directly from a ZIP of ECG CSV files."
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        nargs="+",
        default=[ECG_DSA_RAW_SOURCE_DIR / "data.zip"],
        help="One or more ZIP files containing ECG CSV waveforms.",
    )
    parser.add_argument(
        "--label_xlsx",
        type=Path,
        default=ECG_DSA_RAW_SOURCE_DIR / "ECG-DSA-target.xlsx",
        help="Excel file containing ECG labels and file_name references.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ECG_DSA_WAVEFORM_DIR,
        help="Directory to receive train/val/test .npy files.",
    )
    parser.add_argument(
        "--existing_metadata",
        type=Path,
        default=None,
        help=(
            "Optional existing matched_metadata.csv used to preserve current patient-level "
            "split assignments when adding newly available ECG files."
        ),
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default=DEFAULT_LABEL_COL,
        help="Label column used for training.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=DEFAULT_TASK_TYPE,
        choices=["classification", "regression"],
        help="Interpret labels as classification targets or regression values.",
    )
    parser.add_argument(
        "--group_col",
        type=str,
        default="patient_sn",
        help="Column used for grouped splitting to avoid patient leakage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for split generation.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=NSPLITS,
        help="Number of StratifiedGroupKFold folds. Fold 0 is test, fold 1 is val, the rest are train.",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="patient_stratified",
        choices=["patient_stratified", "group_kfold"],
        help=(
            "Split strategy. 'patient_stratified' keeps all records of a patient in one split "
            "and uses explicit train/val/test ratios. 'group_kfold' preserves the old 10-fold behavior."
        ),
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split ratio used by --split_strategy patient_stratified.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio used by --split_strategy patient_stratified.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test split ratio used by --split_strategy patient_stratified.",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=TARGET_LENGTH,
        help="Number of samples after resampling/downsampling.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional cap for quick validation runs. Use full data by default.",
    )
    return parser.parse_args()


def coerce_label_series(series: pd.Series, task_type: str) -> pd.Series:
    if task_type == "regression":
        return pd.to_numeric(series, errors="coerce").astype(np.float32)

    mapped = series.map({True: 1, False: 0})
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = mapped.fillna(numeric)
    return mapped.astype("Int64")


def maybe_limit_records(df: pd.DataFrame, max_records: Optional[int], seed: int) -> pd.DataFrame:
    if max_records is None or max_records >= len(df):
        return df

    sampled = []
    rng = np.random.RandomState(seed)
    label_counts = df["label"].value_counts(normalize=True)
    for label, frac in label_counts.items():
        label_df = df[df["label"] == label]
        sample_count = max(1, int(round(max_records * frac)))
        sample_count = min(sample_count, len(label_df))
        sampled.append(label_df.sample(n=sample_count, random_state=rng.randint(0, 10**9)))

    limited = pd.concat(sampled, axis=0).drop_duplicates(subset=["csv_name"])
    if len(limited) > max_records:
        limited = limited.sample(n=max_records, random_state=seed)
    return limited.reset_index(drop=True)


def maybe_limit_records_for_task(
    df: pd.DataFrame, max_records: Optional[int], seed: int, task_type: str
) -> pd.DataFrame:
    if task_type == "regression":
        if max_records is None or max_records >= len(df):
            return df
        return df.sample(n=max_records, random_state=seed).reset_index(drop=True)
    return maybe_limit_records(df, max_records, seed)




def build_group_label_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    label_nunique = df.groupby(group_col)["label"].nunique()
    inconsistent_groups = label_nunique[label_nunique > 1]
    if not inconsistent_groups.empty:
        preview = inconsistent_groups.head(10).to_dict()
        raise ValueError(
            "Found patients with inconsistent labels across repeated records. "
            f"Preview: {preview}"
        )

    group_df = (
        df.groupby(group_col, as_index=False)
        .agg(label=("label", "first"))
        .reset_index(drop=True)
    )
    return group_df


def build_group_table_for_regression(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return df[[group_col]].drop_duplicates().reset_index(drop=True)


def assign_patient_stratified_splits(df: pd.DataFrame, args) -> pd.DataFrame:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "--train_ratio + --val_ratio + --test_ratio must sum to 1.0 "
            f"(got {ratio_sum:.6f})."
        )
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("Split ratios must be non-negative and train_ratio must be > 0.")
    if args.val_ratio == 0 or args.test_ratio == 0:
        raise ValueError(
            "ECG-DSA fine-tuning expects train/val/test files. "
            "Use non-zero --val_ratio and --test_ratio."
        )

    if args.task_type == "regression":
        group_df = build_group_table_for_regression(df, args.group_col)
    else:
        group_df = build_group_label_table(df, args.group_col)
    heldout_ratio = args.val_ratio + args.test_ratio

    if args.task_type == "regression":
        train_groups, heldout_groups = train_test_split(
            group_df,
            test_size=heldout_ratio,
            random_state=args.seed,
            shuffle=True,
        )
    else:
        train_groups, heldout_groups = train_test_split(
            group_df,
            test_size=heldout_ratio,
            random_state=args.seed,
            stratify=group_df["label"],
        )

    test_fraction_within_heldout = args.test_ratio / heldout_ratio
    if args.task_type == "regression":
        val_groups, test_groups = train_test_split(
            heldout_groups,
            test_size=test_fraction_within_heldout,
            random_state=args.seed + 1,
            shuffle=True,
        )
    else:
        val_groups, test_groups = train_test_split(
            heldout_groups,
            test_size=test_fraction_within_heldout,
            random_state=args.seed + 1,
            stratify=heldout_groups["label"],
        )

    split_map = {}
    split_map.update({group_id: "train" for group_id in train_groups[args.group_col]})
    split_map.update({group_id: "val" for group_id in val_groups[args.group_col]})
    split_map.update({group_id: "test" for group_id in test_groups[args.group_col]})

    split_df = df.copy()
    split_df["split"] = split_df[args.group_col].map(split_map)
    if split_df["split"].isna().any():
        raise RuntimeError("Some ECG-DSA records were not assigned to any split.")
    return split_df


def assign_group_kfold_splits(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.n_splits < 3:
        raise ValueError("--n_splits must be at least 3.")

    if args.task_type == "regression":
        unique_groups = df[args.group_col].drop_duplicates().to_numpy()
        rng = np.random.RandomState(args.seed)
        rng.shuffle(unique_groups)
        fold_chunks = np.array_split(unique_groups, args.n_splits)
        group_to_fold = {
            group_id: fold_id
            for fold_id, groups in enumerate(fold_chunks)
            for group_id in groups.tolist()
        }
        split_df = df.copy()
        split_df["fold_id"] = split_df[args.group_col].map(group_to_fold).astype(np.int64)
        split_df["split"] = np.where(
            split_df["fold_id"] == 0,
            "test",
            np.where(split_df["fold_id"] == 1, "val", "train"),
        )
        return split_df

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    fold_ids = np.full(len(df), -1, dtype=np.int64)
    dummy_x = np.zeros(len(df), dtype=np.float32)
    for fold_id, (_, test_index) in enumerate(
        splitter.split(dummy_x, df["label"], df[args.group_col])
    ):
        fold_ids[test_index] = fold_id

    split_df = df.copy()
    split_df["fold_id"] = fold_ids
    split_df["split"] = np.where(
        split_df["fold_id"] == 0,
        "test",
        np.where(split_df["fold_id"] == 1, "val", "train"),
    )
    return split_df


def assign_splits(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.split_strategy == "patient_stratified":
        return assign_patient_stratified_splits(df, args)
    return assign_group_kfold_splits(df, args)


def load_existing_group_splits(existing_metadata_path: Path, group_col: str) -> dict[str, str]:
    existing_df = pd.read_csv(existing_metadata_path)
    if group_col not in existing_df.columns:
        raise KeyError(
            f"Group column '{group_col}' not found in existing metadata: "
            f"{existing_metadata_path}"
        )
    if "split" not in existing_df.columns:
        raise KeyError(
            f"'split' column not found in existing metadata: {existing_metadata_path}"
        )

    split_nunique = existing_df.groupby(group_col)["split"].nunique()
    inconsistent_groups = split_nunique[split_nunique > 1]
    if not inconsistent_groups.empty:
        preview = inconsistent_groups.head(10).to_dict()
        raise ValueError(
            "Existing metadata contains patients assigned to multiple splits. "
            f"Preview: {preview}"
        )

    return (
        existing_df.groupby(group_col)["split"]
        .first()
        .to_dict()
    )


def assign_preserved_splits(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.existing_metadata is None:
        return assign_splits(df, args)

    group_split_map = load_existing_group_splits(args.existing_metadata, args.group_col)
    split_df = df.copy()
    split_df["split"] = split_df[args.group_col].map(group_split_map)

    unresolved_mask = split_df["split"].isna()
    if not unresolved_mask.any():
        return split_df

    unresolved_df = split_df.loc[unresolved_mask].drop(columns=["split"]).reset_index(drop=True)
    assigned_unresolved = assign_splits(unresolved_df, args)
    split_df.loc[unresolved_mask, "split"] = assigned_unresolved["split"].to_numpy()
    return split_df


def prepare_metadata(args) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    labels_df = pd.read_excel(args.label_xlsx).copy()
    if args.label_col not in labels_df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")
    if args.group_col not in labels_df.columns:
        raise KeyError(f"Group column not found: {args.group_col}")

    labels_df["csv_name"] = (
        labels_df["file_name"].astype(str).str.replace(".xml", ".csv", regex=False)
    )
    labels_df["label"] = coerce_label_series(labels_df[args.label_col], args.task_type)
    labels_df = labels_df[labels_df["label"].notna()].copy()
    if args.task_type == "regression":
        labels_df["label"] = labels_df["label"].astype(np.float32)
    else:
        labels_df["label"] = labels_df["label"].astype(np.int64)

    member_map = build_member_map(args.zip_path)
    labels_df["member_info"] = labels_df["csv_name"].map(member_map)
    labels_df["member_name"] = labels_df["member_info"].map(
        lambda info: info["member_name"] if isinstance(info, dict) else None
    )
    labels_df["source_zip"] = labels_df["member_info"].map(
        lambda info: info["source_zip"] if isinstance(info, dict) else None
    )
    labels_df["source_zip_path"] = labels_df["member_info"].map(
        lambda info: info["source_zip_path"] if isinstance(info, dict) else None
    )
    labels_df = labels_df.drop(columns=["member_info"])

    missing_df = labels_df[labels_df["member_name"].isna()].copy()
    matched_df = labels_df[labels_df["member_name"].notna()].copy().reset_index(drop=True)
    matched_df = maybe_limit_records_for_task(
        matched_df, args.max_records, args.seed, args.task_type
    )

    if len(matched_df) == 0:
        raise RuntimeError("No labeled ECG files matched entries inside the ZIP archive.")

    matched_df = assign_preserved_splits(matched_df, args)

    return matched_df, missing_df, member_map






def create_memmaps(
    output_dir: Path, split_frames: dict[str, pd.DataFrame], target_length: int, task_type: str
):
    data_arrays = {}
    label_arrays = {}
    label_dtype = np.float32 if task_type == "regression" else np.int64
    for split, frame in split_frames.items():
        data_arrays[split] = open_memmap(
            output_dir / f"{split}_data.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(frame), len(CSV_LEADS), target_length),
        )
        label_arrays[split] = open_memmap(
            output_dir / f"{split}_labels.npy",
            mode="w+",
            dtype=label_dtype,
            shape=(len(frame), 1),
        )
    return data_arrays, label_arrays




def write_summary(
    output_dir: Path,
    matched_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    zip_paths: list[Path],
    group_col: str,
    split_strategy: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    label_col: str,
    task_type: str,
    mean: float,
    std: float,
    existing_metadata: Optional[Path],
):
    summary = {
        "label_col": label_col,
        "task_type": task_type,
        "group_col": group_col,
        "split_strategy": split_strategy,
        "source_zips": [path.name for path in zip_paths],
        "source_zip_counts": {
            source_zip: int(count)
            for source_zip, count in matched_df["source_zip"].value_counts().sort_index().items()
        },
        "existing_metadata": str(existing_metadata) if existing_metadata is not None else None,
        "requested_split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "lead_names": CSV_LEADS,
        "matched_records": int(len(matched_df)),
        "missing_records": int(len(missing_df)),
        "unique_groups": int(matched_df[group_col].nunique()),
        "split_sizes": {
            split: int((matched_df["split"] == split).sum())
            for split in ["train", "val", "test"]
        },
        "split_group_sizes": {
            split: int(matched_df.loc[matched_df["split"] == split, group_col].nunique())
            for split in ["train", "val", "test"]
        },
        "global_mean": mean,
        "global_std": std,
    }
    if task_type == "regression":
        summary["label_summary"] = {}
        for split in ["train", "val", "test"]:
            split_values = matched_df.loc[
                matched_df["split"] == split, "label"
            ].to_numpy(dtype=np.float32)
            summary["label_summary"][split] = {
                "count": int(len(split_values)),
                "mean": float(split_values.mean()) if len(split_values) > 0 else None,
                "std": float(split_values.std(ddof=0)) if len(split_values) > 0 else None,
                "min": float(split_values.min()) if len(split_values) > 0 else None,
                "max": float(split_values.max()) if len(split_values) > 0 else None,
            }
    else:
        summary["label_distribution"] = {
            split: {
                str(label): int(count)
                for label, count in matched_df.loc[matched_df["split"] == split, "label"]
                .value_counts()
                .sort_index()
                .items()
            }
            for split in ["train", "val", "test"]
        }
    with open(output_dir / "prepare_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    args.zip_path = [path.resolve() for path in args.zip_path]
    args.label_xlsx = args.label_xlsx.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.existing_metadata is not None:
        args.existing_metadata = args.existing_metadata.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matched_df, missing_df, _ = prepare_metadata(args)
    matched_df.to_csv(args.output_dir / "matched_metadata.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(args.output_dir / "missing_metadata.csv", index=False, encoding="utf-8-sig")

    split_frames = {
        split: matched_df[matched_df["split"] == split].reset_index(drop=True)
        for split in ["train", "val", "test"]
    }
    for split, frame in split_frames.items():
        frame.to_csv(args.output_dir / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")

    data_arrays, label_arrays = create_memmaps(
        args.output_dir, split_frames, args.target_length, args.task_type
    )
    split_indices = {
        split: {row.csv_name: idx for idx, row in frame.iterrows()}
        for split, frame in split_frames.items()
    }
    train_sum = 0.0
    train_sq_sum = 0.0
    train_value_count = 0

    with ExitStack() as stack:
        zip_files = {
            str(zip_path): stack.enter_context(zipfile.ZipFile(zip_path))
            for zip_path in args.zip_path
        }
        for split in ["train", "val", "test"]:
            frame = split_frames[split]
            for row in tqdm(
                frame.itertuples(index=False),
                total=len(frame),
                desc=f"Preparing {split}",
            ):
                array_index = split_indices[split][row.csv_name]
                waveform = load_waveform(
                    zip_files[row.source_zip_path],
                    row.member_name,
                    args.target_length,
                )
                data_arrays[split][array_index] = waveform
                label_arrays[split][array_index, 0] = row.label

                if split == "train":
                    waveform64 = waveform.astype(np.float64, copy=False)
                    train_sum += float(waveform64.sum())
                    train_sq_sum += float(np.square(waveform64).sum())
                    train_value_count += waveform64.size

    for split in ["train", "val", "test"]:
        data_arrays[split].flush()
        label_arrays[split].flush()

    train_mean = train_sum / train_value_count
    variance = max((train_sq_sum / train_value_count) - (train_mean ** 2), 1e-12)
    train_std = math.sqrt(variance)

    for split in ["train", "val", "test"]:
        standardize_in_place(args.output_dir / f"{split}_data.npy", train_mean, train_std)

    write_summary(
        args.output_dir,
        matched_df,
        missing_df,
        args.zip_path,
        args.group_col,
        args.split_strategy,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.label_col,
        args.task_type,
        train_mean,
        train_std,
        args.existing_metadata,
    )
    print(
        f"Prepared ECG-DSA dataset at {args.output_dir} using label '{args.label_col}'.",
        flush=True,
    )


if __name__ == "__main__":
    main()
