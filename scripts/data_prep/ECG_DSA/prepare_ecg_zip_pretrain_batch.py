"""无标签 ECG 预训练批次构建脚本（含 train/val 划分）。

这个脚本会从包含 ECG CSV 的 ZIP 文件中构建预训练数据集。
与 prepare_ecg_zip_waveforms.py 不同，这个脚本会进一步划分 train/val。

输出内容：
- train_data.npy：训练集波形
- val_data.npy：验证集波形
- train_metadata.csv：训练集元数据
- val_metadata.csv：验证集元数据
- prepare_summary.json：数据集统计信息

适用场景：为自监督预训练准备带有 train/val 划分的数据。
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

# 允许直接以 `python scripts/data_prep/ECG_DSA/prepare_ecg_zip_pretrain_batch.py` 方式运行。
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
        description="Prepare an unlabeled ECG pretraining batch from one or more ZIP files of CSV waveforms."
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
        help="Directory to receive train_data.npy and val_data.npy.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio for the unlabeled pretraining batch.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio for the unlabeled pretraining batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the train/val split.",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=TARGET_LENGTH,
        help="Number of samples after resampling/downsampling.",
    )
    return parser.parse_args()




def assign_splits(metadata: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int) -> pd.DataFrame:
    ratio_sum = train_ratio + val_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "--train_ratio + --val_ratio must sum to 1.0 "
            f"(got {ratio_sum:.6f})."
        )
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("--train_ratio and --val_ratio must both be > 0.")

    rng = np.random.RandomState(seed)
    shuffled_indices = np.arange(len(metadata))
    rng.shuffle(shuffled_indices)

    train_count = int(round(len(metadata) * train_ratio))
    train_count = min(max(train_count, 1), len(metadata) - 1)

    split = np.empty(len(metadata), dtype=object)
    split[shuffled_indices[:train_count]] = "train"
    split[shuffled_indices[train_count:]] = "val"

    split_metadata = metadata.copy()
    split_metadata["split"] = split
    return split_metadata




def create_memmaps(output_dir: Path, split_frames: dict[str, pd.DataFrame], target_length: int):
    data_arrays = {}
    for split, frame in split_frames.items():
        data_arrays[split] = open_memmap(
            output_dir / f"{split}_data.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(frame), len(CSV_LEADS), target_length),
        )
    return data_arrays




def write_summary(
    output_dir: Path,
    metadata: pd.DataFrame,
    zip_paths: list[Path],
    target_length: int,
    train_ratio: float,
    val_ratio: float,
    mean: float,
    std: float,
):
    summary = {
        "source_zips": [path.name for path in zip_paths],
        "record_count": int(len(metadata)),
        "lead_names": CSV_LEADS,
        "target_length": target_length,
        "requested_split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
        },
        "split_sizes": {
            split: int((metadata["split"] == split).sum())
            for split in ["train", "val"]
        },
        "source_zip_counts": {
            source_zip: int(count)
            for source_zip, count in metadata["source_zip"].value_counts().sort_index().items()
        },
        "global_mean": mean,
        "global_std": std,
    }
    with open(output_dir / "prepare_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    args.zip_path = [path.resolve() for path in args.zip_path]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_member_table(args.zip_path)
    metadata = assign_splits(metadata, args.train_ratio, args.val_ratio, args.seed)
    metadata.to_csv(args.output_dir / "all_metadata.csv", index=False, encoding="utf-8-sig")

    split_frames = {
        split: metadata[metadata["split"] == split].reset_index(drop=True)
        for split in ["train", "val"]
    }
    for split, frame in split_frames.items():
        frame.to_csv(args.output_dir / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")

    data_arrays = create_memmaps(args.output_dir, split_frames, args.target_length)

    train_sum = 0.0
    train_sq_sum = 0.0
    train_value_count = 0

    with ExitStack() as stack:
        zip_files = {
            str(zip_path): stack.enter_context(zipfile.ZipFile(zip_path))
            for zip_path in args.zip_path
        }

        for split in ["train", "val"]:
            frame = split_frames[split]
            for array_index, row in enumerate(
                tqdm(frame.itertuples(index=False), total=len(frame), desc=f"Preparing {split}")
            ):
                waveform = load_waveform(
                    zip_files[row.source_zip_path],
                    row.member_name,
                    args.target_length,
                )
                data_arrays[split][array_index] = waveform

                if split == "train":
                    waveform64 = waveform.astype(np.float64, copy=False)
                    train_sum += float(waveform64.sum())
                    train_sq_sum += float(np.square(waveform64).sum())
                    train_value_count += waveform64.size

    for split in ["train", "val"]:
        data_arrays[split].flush()

    train_mean = train_sum / train_value_count
    variance = max((train_sq_sum / train_value_count) - (train_mean ** 2), 1e-12)
    train_std = np.sqrt(variance)

    for split in ["train", "val"]:
        standardize_in_place(args.output_dir / f"{split}_data.npy", train_mean, train_std)

    write_summary(
        args.output_dir,
        metadata,
        args.zip_path,
        args.target_length,
        args.train_ratio,
        args.val_ratio,
        train_mean,
        train_std,
    )
    print(f"Prepared unlabeled ECG pretraining batch at {args.output_dir}.", flush=True)


if __name__ == "__main__":
    main()
