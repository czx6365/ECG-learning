from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 允许直接以 `python scripts/analysis/ECG_DSA/visualize_random_qrs_samples.py` 方式运行。
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import (
    ECG_DSA_QRS_BASE_DIR,
    ECG_DSA_WAVEFORM_DIR,
    RESULTS_ECG_DSA_RANDOM_QRS_SAMPLES_DIR,
)


CHANNEL_IDS = [1, 2, 3, 7, 8, 9, 10, 11, 12]
CHANNEL_ID_TO_NAME = {
    1: "I",
    2: "II",
    3: "III",
    7: "V1",
    8: "V2",
    9: "V3",
    10: "V4",
    11: "V5",
    12: "V6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample ECG-DSA records and visualize every QRS segment tile."
    )
    parser.add_argument(
        "--qrs_dir",
        type=Path,
        default=ECG_DSA_QRS_BASE_DIR,
        help="Directory containing *_data.npy and *_data_in_chans.npy.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=ECG_DSA_WAVEFORM_DIR,
        help="Directory containing *_metadata.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RESULTS_ECG_DSA_RANDOM_QRS_SAMPLES_DIR,
        help="Directory to receive exported sample figures.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=30,
        help="Number of ECG records to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits eligible for sampling.",
    )
    return parser.parse_args()


def load_split_table(raw_dir: Path, split: str) -> pd.DataFrame:
    metadata = pd.read_csv(raw_dir / f"{split}_metadata.csv")
    metadata = metadata.reset_index().rename(columns={"index": "sample_index"})
    metadata["split"] = split
    return metadata


def sample_rows(metadata: pd.DataFrame, sample_count: int, seed: int) -> pd.DataFrame:
    sample_count = min(sample_count, len(metadata))
    return metadata.sample(n=sample_count, random_state=seed).sort_values(
        ["split", "sample_index"]
    )


def build_grid_axes(row_count: int, col_count: int, fig_width: float, fig_height: float):
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes


def plot_sample(
    qrs_data: np.ndarray,
    chan_data: np.ndarray,
    sample_row: pd.Series,
    output_path: Path,
) -> None:
    split = str(sample_row["split"])
    sample_index = int(sample_row["sample_index"])
    sample_qrs = qrs_data[sample_index]
    sample_chans = chan_data[sample_index]

    segments_by_channel = {
        channel_id: sample_qrs[sample_chans == channel_id] for channel_id in CHANNEL_IDS
    }
    max_segments = max(len(segments) for segments in segments_by_channel.values())
    col_count = max(max_segments, 1)

    fig_width = max(14.0, col_count * 1.1)
    fig_height = len(CHANNEL_IDS) * 1.25
    fig, axes = build_grid_axes(len(CHANNEL_IDS), col_count, fig_width, fig_height)

    title_bits = [
        f"{split} sample {sample_index}",
        str(sample_row.get("csv_name", "")),
    ]
    if "label" in sample_row:
        title_bits.append(f"label={sample_row['label']}")
    fig.suptitle(" | ".join(bit for bit in title_bits if bit), fontsize=12)

    for row_idx, channel_id in enumerate(CHANNEL_IDS):
        channel_name = CHANNEL_ID_TO_NAME[channel_id]
        segments = segments_by_channel[channel_id]

        for col_idx in range(col_count):
            axis = axes[row_idx, col_idx]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

            if col_idx < len(segments):
                segment = segments[col_idx]
                axis.plot(segment, color="tab:blue", linewidth=1.0)
                axis.axhline(0.0, color="#cccccc", linewidth=0.6)
                if row_idx == 0:
                    axis.set_title(f"b{col_idx + 1}", fontsize=8)
            else:
                axis.axis("off")

            if col_idx == 0:
                axis.set_ylabel(channel_name, rotation=0, labelpad=24, fontsize=9)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.concat(
        [load_split_table(args.raw_dir.resolve(), split) for split in args.splits],
        ignore_index=True,
    )
    sampled = sample_rows(metadata, args.sample_count, args.seed).reset_index(drop=True)
    sampled.to_csv(
        args.output_dir / "sample_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, sample_row in sampled.iterrows():
        split = str(sample_row["split"])
        if split not in split_cache:
            split_cache[split] = (
                np.load(args.qrs_dir.resolve() / f"{split}_data.npy", mmap_mode="r"),
                np.load(
                    args.qrs_dir.resolve() / f"{split}_data_in_chans.npy",
                    mmap_mode="r",
                ),
            )

        qrs_data, chan_data = split_cache[split]
        output_name = (
            f"{split}_idx{int(sample_row['sample_index']):05d}_"
            f"{str(sample_row.get('csv_name', 'sample')).replace('/', '_')}.png"
        )
        plot_sample(qrs_data, chan_data, sample_row, args.output_dir / output_name)

    print(f"Saved {len(sampled)} sampled ECG figures to {args.output_dir}")
    print(f"Sample index written to {args.output_dir / 'sample_index.csv'}")


if __name__ == "__main__":
    main()
