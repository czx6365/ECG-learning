"""无标签 ECG 波形数据集构建脚本。

这是一个辅助脚本，用于在没有标签的情况下临时抽取波形数据。
它会处理 ZIP 文件中的 ECG CSV，并输出一个合并后的单一数据集。

输出内容：
- all_data.npy：包含全部波形的单个数组（不区分 train/val/test）
- all_metadata.csv：每条记录的元数据
- prepare_summary.json：数据集统计信息

适用场景：用于探索性分析或无监督学习时快速抽取波形。
它不属于主 ECG-DSA 流程。
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm

# 允许直接以 `python scripts/data_prep/ECG_DSA/prepare_ecg_zip_waveforms.py` 方式运行。
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_prep.ECG_DSA.ecg_zip_utils import (
    build_member_table,
    load_waveform,
    standardize_in_place,
    CSV_LEADS,
)


TARGET_LENGTH = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare waveform-only ECG datasets from one or more ZIP files of CSV waveforms."
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        nargs="+",
        required=True,
        help="One or more ZIP files containing ECG CSV waveforms.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to receive all_data.npy and metadata files.",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=TARGET_LENGTH,
        help="Number of samples after resampling/downsampling.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=None,
        help="Optional external mean used for standardization.",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=None,
        help="Optional external std used for standardization.",
    )
    return parser.parse_args()








def write_summary(
    output_dir: Path,
    metadata: pd.DataFrame,
    zip_paths: list[Path],
    target_length: int,
    mean: float,
    std: float,
    standardization_source: str,
):
    summary = {
        "source_zips": [path.name for path in zip_paths],
        "record_count": int(len(metadata)),
        "lead_names": CSV_LEADS,
        "target_length": target_length,
        "source_zip_counts": {
            source_zip: int(count)
            for source_zip, count in metadata["source_zip"].value_counts().sort_index().items()
        },
        "global_mean": mean,
        "global_std": std,
        "standardization_source": standardization_source,
    }
    with open(output_dir / "prepare_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    args.zip_path = [path.resolve() for path in args.zip_path]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if (args.mean is None) ^ (args.std is None):
        raise ValueError("--mean and --std must be provided together.")

    metadata = build_member_table(args.zip_path)
    metadata.to_csv(args.output_dir / "all_metadata.csv", index=False, encoding="utf-8-sig")

    data_array = open_memmap(
        args.output_dir / "all_data.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(metadata), len(CSV_LEADS), args.target_length),
    )

    running_sum = 0.0
    running_sq_sum = 0.0
    running_value_count = 0

    with ExitStack() as stack:
        zip_files = {
            str(zip_path): stack.enter_context(zipfile.ZipFile(zip_path))
            for zip_path in args.zip_path
        }
        for array_index, row in enumerate(
            tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Preparing all")
        ):
            waveform = load_waveform(
                zip_files[row.source_zip_path],
                row.member_name,
                args.target_length,
            )
            data_array[array_index] = waveform

            if args.mean is None:
                waveform64 = waveform.astype(np.float64, copy=False)
                running_sum += float(waveform64.sum())
                running_sq_sum += float(np.square(waveform64).sum())
                running_value_count += waveform64.size

    data_array.flush()

    if args.mean is None:
        mean = running_sum / running_value_count
        variance = max((running_sq_sum / running_value_count) - (mean ** 2), 1e-12)
        std = np.sqrt(variance)
        standardization_source = "computed_from_all_records"
    else:
        mean = args.mean
        std = args.std
        standardization_source = "external_arguments"

    standardize_in_place(args.output_dir / "all_data.npy", mean, std)
    write_summary(
        args.output_dir,
        metadata,
        args.zip_path,
        args.target_length,
        mean,
        std,
        standardization_source,
    )
    print(f"Prepared waveform-only ECG dataset at {args.output_dir}.", flush=True)


if __name__ == "__main__":
    main()
