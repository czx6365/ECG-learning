from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wfdb import processing

# 允许直接以 `python scripts/analysis/ECG_DSA/visualize_trimmed_qrs_comparison.py` 方式运行。
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import ECG_DSA_WAVEFORM_DIR, RESULTS_QRS_TRIM_DEBUG_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export side-by-side plots showing QRS detection before and after trimming edge beats."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=ECG_DSA_WAVEFORM_DIR,
        help="Directory containing raw ECG *_data.npy and *_metadata.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RESULTS_QRS_TRIM_DEBUG_DIR,
        help="Directory to receive exported comparison figures.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=10,
        help="Number of ECG records to visualize.",
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
    parser.add_argument(
        "--lead_index",
        type=int,
        default=0,
        help="Lead index used for QRS detection and waveform overlay.",
    )
    parser.add_argument(
        "--token_len",
        type=int,
        default=96,
        help="Token length used when rebuilding kept QRS segments.",
    )
    parser.add_argument(
        "--max_segments_to_plot",
        type=int,
        default=12,
        help="Maximum number of kept QRS segments shown in the lower panel.",
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


def trim_edge_qrs(qrs_inds: np.ndarray) -> np.ndarray:
    if len(qrs_inds) <= 2:
        return qrs_inds[:0]
    return qrs_inds[1:-1]


def extract_qrs_segments(
    lead_signal: np.ndarray, qrs_inds: np.ndarray, token_len: int
) -> np.ndarray:
    segments: list[np.ndarray] = []
    for idx in range(1, len(qrs_inds) - 1):
        center = int(qrs_inds[idx])
        start = int((qrs_inds[idx] + qrs_inds[idx - 1]) // 2)
        end = int((qrs_inds[idx] + qrs_inds[idx + 1]) // 2)

        start = max(start, 0)
        end = min(end, len(lead_signal))
        actual_len = end - start

        if actual_len > token_len:
            start = max(center - token_len // 2, 0)
            end = min(start + token_len, len(lead_signal))

        segment = np.zeros(token_len, dtype=np.float32)
        segment_start = max(token_len // 2 - (center - start), 0)
        segment_end = segment_start + (end - start)

        if segment_end > token_len:
            end -= segment_end - token_len
            segment_end = token_len

        segment[segment_start:segment_end] = lead_signal[start:end]
        segments.append(segment)

    if not segments:
        return np.zeros((0, token_len), dtype=np.float32)
    return np.stack(segments)


def plot_sample(
    lead_signal: np.ndarray,
    qrs_inds: np.ndarray,
    kept_qrs_inds: np.ndarray,
    kept_segments: np.ndarray,
    sample_row: pd.Series,
    output_path: Path,
    max_segments_to_plot: int,
) -> dict[str, object]:
    dropped = (
        np.array([qrs_inds[0], qrs_inds[-1]], dtype=np.int32)
        if len(qrs_inds) > 1
        else np.array(qrs_inds[:1], dtype=np.int32)
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    top_axis = axes[0]
    top_axis.plot(lead_signal, color="tab:blue", linewidth=1.0, label="Lead I raw")
    if len(qrs_inds):
        top_axis.scatter(
            qrs_inds,
            lead_signal[qrs_inds],
            s=28,
            color="tab:red",
            label="All detected QRS",
            zorder=3,
        )
    if len(kept_qrs_inds):
        top_axis.scatter(
            kept_qrs_inds,
            lead_signal[kept_qrs_inds],
            s=34,
            color="tab:green",
            label="Kept after trimming",
            zorder=4,
        )
    if len(dropped):
        top_axis.scatter(
            dropped,
            lead_signal[dropped],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Dropped head/tail",
            zorder=5,
        )

    top_axis.set_title(
        (
            "QRS detection and edge trimming"
            f" | {sample_row['split']} idx={int(sample_row['sample_index'])}"
            f" | {sample_row.get('csv_name', '')}"
        ),
        fontsize=12,
    )
    top_axis.set_xlabel("Sample point")
    top_axis.set_ylabel("Amplitude")
    top_axis.legend(loc="upper right")

    bottom_axis = axes[1]
    plotted_segments = kept_segments[:max_segments_to_plot]
    offset = 0
    for segment_idx, segment in enumerate(plotted_segments, start=1):
        xs = np.arange(offset, offset + len(segment))
        bottom_axis.plot(xs, segment, linewidth=1.0)
        bottom_axis.axvline(offset, color="#cccccc", linewidth=0.6)
        label_y = float(np.max(segment))
        bottom_axis.text(offset + 4, label_y if label_y > 0 else 0.1, f"b{segment_idx}", fontsize=8)
        offset += len(segment)
    if len(plotted_segments):
        bottom_axis.axvline(offset, color="#cccccc", linewidth=0.6)

    bottom_axis.set_title(
        (
            "Kept QRS segments only"
            f" | kept={len(kept_qrs_inds)}"
            f" | dropped={max(len(qrs_inds) - len(kept_qrs_inds), 0)}"
        ),
        fontsize=12,
    )
    bottom_axis.set_xlabel("Concatenated token samples")
    bottom_axis.set_ylabel("Amplitude")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "split": sample_row["split"],
        "sample_index": int(sample_row["sample_index"]),
        "csv_name": sample_row.get("csv_name", ""),
        "all_qrs": int(len(qrs_inds)),
        "kept_qrs": int(len(kept_qrs_inds)),
        "dropped_qrs": int(max(len(qrs_inds) - len(kept_qrs_inds), 0)),
        "dropped_positions": ",".join(str(int(x)) for x in dropped),
        "output_file": output_path.name,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.concat(
        [load_split_table(args.raw_dir.resolve(), split) for split in args.splits],
        ignore_index=True,
    )
    sampled = sample_rows(metadata, args.sample_count, args.seed).reset_index(drop=True)

    split_cache: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []

    for _, sample_row in sampled.iterrows():
        split = str(sample_row["split"])
        if split not in split_cache:
            split_cache[split] = np.load(
                args.raw_dir.resolve() / f"{split}_data.npy",
                mmap_mode="r",
            )

        raw = split_cache[split]
        sample_index = int(sample_row["sample_index"])
        lead_signal = np.asarray(raw[sample_index][args.lead_index], dtype=np.float32)
        qrs_inds = processing.xqrs_detect(sig=lead_signal, fs=100, verbose=False)
        kept_qrs_inds = trim_edge_qrs(qrs_inds)
        kept_segments = extract_qrs_segments(lead_signal, qrs_inds, args.token_len)

        output_path = args.output_dir / f"{split}_idx{sample_index:05d}_trimmed_qrs.png"
        summary_rows.append(
            plot_sample(
                lead_signal=lead_signal,
                qrs_inds=qrs_inds,
                kept_qrs_inds=kept_qrs_inds,
                kept_segments=kept_segments,
                sample_row=sample_row,
                output_path=output_path,
                max_segments_to_plot=args.max_segments_to_plot,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        args.output_dir / "trimmed_qrs_index.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved {len(summary)} trimmed-QRS comparison figures to {args.output_dir}")
    print(f"Summary written to {args.output_dir / 'trimmed_qrs_index.csv'}")


if __name__ == "__main__":
    main()
