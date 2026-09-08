"""ECG-DSA 的过滤版 QRS 数据集构建脚本。

这个脚本会根据通道扫描得到的可疑样本，过滤掉质量异常的记录，
例如平直导联或其他明显有问题的样本。

输入：
- 来自 ECG_DSA_WAVEFORM_DIR 的基础波形数据集
- 来自 RESULTS_ECG_DSA_CHANNEL_SCAN_DIR 的通道扫描结果

输出：
- 过滤后的 *_data.npy、*_labels.npy 数组
- 过滤后的 metadata CSV
- 记录统计信息的 filter_summary.json

适用场景：通过删除问题样本，构建更干净、更适合训练的 QRS 数据集。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

# 允许直接以 `python scripts/data_prep/ECG_DSA/build_filtered_qrs_dataset.py` 方式运行。
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import (
    ECG_DSA_QRS_BASE_DIR,
    ECG_DSA_QRS_NONFLAT_CLEAN_DIR,
    ECG_DSA_WAVEFORM_DIR,
    RESULTS_ECG_DSA_CHANNEL_SCAN_DIR,
)


ARRAY_SUFFIXES = [
    "data.npy",
    "labels.npy",
    "data_in_chans.npy",
    "data_in_times.npy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a filtered ECG-DSA QRS dataset by removing suspicious samples."
    )
    parser.add_argument(
        "--qrs_dir",
        type=Path,
        default=ECG_DSA_QRS_BASE_DIR,
        help="Source QRS dataset directory.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=ECG_DSA_WAVEFORM_DIR,
        help="Directory containing split metadata CSV files.",
    )
    parser.add_argument(
        "--suspicious_csv",
        type=Path,
        default=RESULTS_ECG_DSA_CHANNEL_SCAN_DIR / "top_nonflat_suspicious_channels.csv",
        help="CSV report containing suspicious sample-channel rows.",
    )
    parser.add_argument(
        "--match_column",
        type=str,
        default="csv_name",
        help="Column used to match suspicious rows against split metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ECG_DSA_QRS_NONFLAT_CLEAN_DIR,
        help="Destination directory for the filtered dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to filter.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="How many kept rows to copy into each memmap write chunk.",
    )
    return parser.parse_args()


def load_removed_value_map(
    suspicious_csv: Path,
    splits: list[str],
    match_column: str,
) -> dict[str, np.ndarray]:
    suspicious = pd.read_csv(suspicious_csv)
    if match_column not in suspicious.columns:
        raise KeyError(f"Column '{match_column}' not found in {suspicious_csv}")

    removed_value_map: dict[str, np.ndarray] = {}
    for split in splits:
        split_rows = suspicious[suspicious["split"] == split]
        removed_values = np.sort(split_rows[match_column].astype(str).unique())
        removed_value_map[split] = removed_values
    return removed_value_map


def copy_filtered_array(
    source_path: Path,
    destination_path: Path,
    keep_indices: np.ndarray,
    chunk_size: int,
) -> tuple[int, ...]:
    source = np.load(source_path, mmap_mode="r")
    destination_shape = (len(keep_indices),) + source.shape[1:]
    destination = open_memmap(
        destination_path,
        mode="w+",
        dtype=source.dtype,
        shape=destination_shape,
    )

    for start in range(0, len(keep_indices), chunk_size):
        end = min(start + chunk_size, len(keep_indices))
        destination[start:end] = source[keep_indices[start:end]]

    destination.flush()
    return destination_shape


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    removed_value_map = load_removed_value_map(
        args.suspicious_csv,
        args.splits,
        args.match_column,
    )
    summary: dict[str, dict[str, object]] = {}

    for split in args.splits:
        metadata_path = args.raw_dir / f"{split}_metadata.csv"
        metadata = pd.read_csv(metadata_path).reset_index().rename(
            columns={"index": "original_sample_index"}
        )
        if args.match_column not in metadata.columns:
            raise KeyError(f"Column '{args.match_column}' not found in {metadata_path}")

        removed_values = removed_value_map[split]
        metadata_match_values = metadata[args.match_column].astype(str)
        removed_mask = metadata_match_values.isin(removed_values).to_numpy()

        keep_mask = ~removed_mask
        keep_indices = np.flatnonzero(keep_mask)
        removed_indices = np.flatnonzero(removed_mask)
        matched_removed_values = np.sort(metadata_match_values[removed_mask].unique())
        missing_removed_values = np.setdiff1d(removed_values, matched_removed_values)

        filtered_metadata = metadata.iloc[keep_indices].copy().reset_index(drop=True)
        filtered_metadata.to_csv(
            args.output_dir / f"{split}_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )

        array_shapes: dict[str, tuple[int, ...]] = {}
        for suffix in ARRAY_SUFFIXES:
            source_path = args.qrs_dir / f"{split}_{suffix}"
            destination_path = args.output_dir / f"{split}_{suffix}"
            array_shapes[f"{split}_{suffix}"] = copy_filtered_array(
                source_path=source_path,
                destination_path=destination_path,
                keep_indices=keep_indices,
                chunk_size=args.chunk_size,
            )

        removed_manifest = metadata.iloc[removed_indices].copy().reset_index(drop=True)
        removed_manifest.to_csv(
            args.output_dir / f"{split}_removed_samples.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary[split] = {
            "source_size": int(len(metadata)),
            "removed_size": int(len(removed_indices)),
            "kept_size": int(len(keep_indices)),
            "removed_indices": removed_indices.tolist(),
            "matched_removed_values": matched_removed_values.tolist(),
            "missing_removed_values": missing_removed_values.tolist(),
            "array_shapes": {name: list(shape) for name, shape in array_shapes.items()},
        }

    with open(args.output_dir / "filter_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "qrs_dir": str(args.qrs_dir),
                "raw_dir": str(args.raw_dir),
                "suspicious_csv": str(args.suspicious_csv),
                "match_column": args.match_column,
                "splits": args.splits,
                "summary": summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    for split, split_summary in summary.items():
        print(
            f"{split}: kept {split_summary['kept_size']} / {split_summary['source_size']} "
            f"(removed {split_summary['removed_size']})"
        )
    print(f"Filtered dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
