"""ECG-DSA 的掩码版 QRS 数据集构建脚本。

这个脚本会根据通道扫描得到的可疑 sample-channel token，
将对应位置置零进行掩码，而不是直接删除整条记录。

输入：
- 来自 ECG_DSA_WAVEFORM_DIR 的基础波形数据集
- 来自 RESULTS_ECG_DSA_CHANNEL_SCAN_DIR 的通道扫描结果

输出：
- 掩码后的 *_data.npy、*_labels.npy 数组（可疑 token 会被置零）
- 包含掩码统计信息的 metadata CSV
- 记录详细掩码统计的 mask_summary.json

适用场景：在处理质量问题时尽量保留数据量，只屏蔽有问题的 token。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

# 允许直接以 `python scripts/data_prep/ECG_DSA/build_masked_qrs_dataset.py` 方式运行。
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import (
    ECG_DSA_QRS_ALLNOISE_MASKED_DIR,
    ECG_DSA_QRS_BASE_DIR,
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
        description="Build a masked ECG-DSA QRS dataset by zeroing suspicious sample-channel tokens."
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
        default=RESULTS_ECG_DSA_CHANNEL_SCAN_DIR / "threshold_hits.csv",
        help="CSV report containing suspicious sample-channel rows.",
    )
    parser.add_argument(
        "--match_column",
        type=str,
        default="csv_name",
        help="Metadata column used to match suspicious rows to records.",
    )
    parser.add_argument(
        "--channel_column",
        type=str,
        default="channel_id",
        help="CSV column containing the channel id to mask.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ECG_DSA_QRS_ALLNOISE_MASKED_DIR,
        help="Destination directory for the masked dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="How many samples to copy per memmap write chunk.",
    )
    return parser.parse_args()


def load_suspicious_map(
    suspicious_csv: Path,
    splits: list[str],
    match_column: str,
    channel_column: str,
) -> dict[str, dict[str, set[int]]]:
    suspicious = pd.read_csv(suspicious_csv)
    required = {"split", match_column, channel_column}
    missing = required.difference(suspicious.columns)
    if missing:
        raise KeyError(f"Missing required columns in {suspicious_csv}: {sorted(missing)}")

    suspicious_map: dict[str, dict[str, set[int]]] = {}
    for split in splits:
        split_rows = suspicious[suspicious["split"] == split]
        split_map: dict[str, set[int]] = defaultdict(set)
        for _, row in split_rows.iterrows():
            split_map[str(row[match_column])].add(int(row[channel_column]))
        suspicious_map[split] = dict(split_map)
    return suspicious_map


def copy_array_verbatim(
    source_path: Path,
    destination_path: Path,
    chunk_size: int,
) -> tuple[int, ...]:
    source = np.load(source_path, mmap_mode="r")
    destination = open_memmap(
        destination_path,
        mode="w+",
        dtype=source.dtype,
        shape=source.shape,
    )
    for start in range(0, source.shape[0], chunk_size):
        end = min(start + chunk_size, source.shape[0])
        destination[start:end] = source[start:end]
    destination.flush()
    return source.shape


def build_masked_split(
    split: str,
    qrs_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    suspicious_map: dict[str, set[int]],
    match_column: str,
    chunk_size: int,
) -> dict[str, object]:
    metadata = pd.read_csv(raw_dir / f"{split}_metadata.csv").reset_index().rename(
        columns={"index": "sample_index"}
    )
    if match_column not in metadata.columns:
        raise KeyError(f"Column '{match_column}' not found in {raw_dir / f'{split}_metadata.csv'}")

    data = np.load(qrs_dir / f"{split}_data.npy", mmap_mode="r")
    labels_shape = copy_array_verbatim(
        qrs_dir / f"{split}_labels.npy",
        output_dir / f"{split}_labels.npy",
        chunk_size=chunk_size,
    )

    in_chans = np.load(qrs_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    in_times = np.load(qrs_dir / f"{split}_data_in_times.npy", mmap_mode="r")

    masked_data = open_memmap(
        output_dir / f"{split}_data.npy",
        mode="w+",
        dtype=data.dtype,
        shape=data.shape,
    )
    masked_in_chans = open_memmap(
        output_dir / f"{split}_data_in_chans.npy",
        mode="w+",
        dtype=in_chans.dtype,
        shape=in_chans.shape,
    )
    masked_in_times = open_memmap(
        output_dir / f"{split}_data_in_times.npy",
        mode="w+",
        dtype=in_times.dtype,
        shape=in_times.shape,
    )

    masked_sample_rows = []
    masked_samples = 0
    masked_tokens = 0

    key_values = metadata[match_column].astype(str).to_numpy()
    for start in range(0, len(metadata), chunk_size):
        end = min(start + chunk_size, len(metadata))
        chunk_data = np.array(data[start:end], copy=True)
        chunk_chans = np.array(in_chans[start:end], copy=True)
        chunk_times = np.array(in_times[start:end], copy=True)

        for local_idx, record_key in enumerate(key_values[start:end]):
            suspicious_channels = suspicious_map.get(record_key)
            if not suspicious_channels:
                continue

            token_mask = np.isin(chunk_chans[local_idx], list(suspicious_channels))
            if not np.any(token_mask):
                continue

            chunk_data[local_idx, token_mask] = 0
            chunk_chans[local_idx, token_mask] = 0
            chunk_times[local_idx, token_mask] = 0

            absolute_idx = start + local_idx
            masked_samples += 1
            masked_tokens += int(token_mask.sum())
            masked_sample_rows.append(
                {
                    "split": split,
                    "sample_index": absolute_idx,
                    match_column: record_key,
                    "masked_channel_ids": sorted(int(x) for x in suspicious_channels),
                    "masked_token_count": int(token_mask.sum()),
                }
            )

        masked_data[start:end] = chunk_data
        masked_in_chans[start:end] = chunk_chans
        masked_in_times[start:end] = chunk_times

    masked_data.flush()
    masked_in_chans.flush()
    masked_in_times.flush()

    metadata.to_csv(output_dir / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")
    masked_manifest = pd.DataFrame(masked_sample_rows)
    masked_manifest.to_csv(
        output_dir / f"{split}_masked_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "source_size": int(len(metadata)),
        "masked_sample_size": int(masked_samples),
        "masked_token_count": int(masked_tokens),
        "label_shape": list(labels_shape),
        "data_shape": list(data.shape),
        "channels_shape": list(in_chans.shape),
        "times_shape": list(in_times.shape),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suspicious_map = load_suspicious_map(
        args.suspicious_csv,
        args.splits,
        args.match_column,
        args.channel_column,
    )

    summary: dict[str, dict[str, object]] = {}
    for split in args.splits:
        summary[split] = build_masked_split(
            split=split,
            qrs_dir=args.qrs_dir,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            suspicious_map=suspicious_map[split],
            match_column=args.match_column,
            chunk_size=args.chunk_size,
        )
        split_summary = summary[split]
        print(
            f"{split}: masked samples {split_summary['masked_sample_size']} / {split_summary['source_size']}, "
            f"masked tokens={split_summary['masked_token_count']}"
        )

    with open(args.output_dir / "mask_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "qrs_dir": str(args.qrs_dir),
                "raw_dir": str(args.raw_dir),
                "suspicious_csv": str(args.suspicious_csv),
                "match_column": args.match_column,
                "channel_column": args.channel_column,
                "splits": args.splits,
                "summary": summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Masked dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
