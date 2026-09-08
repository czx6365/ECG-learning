from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 允许直接以 `python scripts/analysis/ECG_DSA/find_suspicious_channels.py` 方式运行。
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import (
    ECG_DSA_QRS_BASE_DIR,
    ECG_DSA_WAVEFORM_DIR,
    RESULTS_ECG_DSA_CHANNEL_SCAN_DIR,
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
CHANNEL_ID_TO_RAW_INDEX = {channel_id: idx for idx, channel_id in enumerate(CHANNEL_IDS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan ECG-DSA QRS tensors and flag suspicious sample-channel pairs."
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
        help="Directory containing raw ECG *_data.npy and *_metadata.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RESULTS_ECG_DSA_CHANNEL_SCAN_DIR,
        help="Directory to receive CSV reports and preview plots.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to scan.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=500,
        help="Number of highest-scoring suspicious channels to save separately.",
    )
    parser.add_argument(
        "--preview_count",
        type=int,
        default=30,
        help="Number of preview plots to export for the most suspicious channels.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=6.0,
        help="Channels with suspicion_score >= threshold are written to threshold_hits.csv.",
    )
    parser.add_argument(
        "--nonflat_min_ptp",
        type=float,
        default=0.5,
        help="Minimum raw peak-to-peak amplitude for the non-flat suspicious report.",
    )
    return parser.parse_args()


def corr_to_template(segments: np.ndarray, template: np.ndarray) -> np.ndarray:
    centered_template = template - template.mean()
    template_norm = np.linalg.norm(centered_template)
    corrs = np.zeros(len(segments), dtype=np.float32)
    if template_norm <= 1e-8:
        return corrs

    for idx, segment in enumerate(segments):
        centered_segment = segment - segment.mean()
        segment_norm = np.linalg.norm(centered_segment)
        if segment_norm <= 1e-8:
            corrs[idx] = 0.0
        else:
            corrs[idx] = float(
                np.dot(centered_segment, centered_template)
                / (segment_norm * template_norm)
            )
    return corrs


def compute_channel_metrics(
    segments: np.ndarray, raw_signal: np.ndarray
) -> dict[str, float | int]:
    template = np.median(segments, axis=0)
    corrs = corr_to_template(segments, template)

    residual_rms = np.sqrt(np.mean((segments - template) ** 2, axis=1))
    hf_rms = np.sqrt(np.mean(np.diff(segments, n=2, axis=1) ** 2, axis=1))
    beat_std = np.std(segments, axis=1)

    template_ptp = float(np.ptp(template))
    template_scale = max(template_ptp, 1e-6)
    beat_scale = np.maximum(beat_std, 1e-6)

    raw_hf_rms = float(np.sqrt(np.mean(np.diff(raw_signal, n=2) ** 2)))
    raw_std = float(np.std(raw_signal))
    raw_ptp = float(np.ptp(raw_signal))
    raw_jump_p95 = float(np.percentile(np.abs(np.diff(raw_signal)), 95))

    return {
        "beat_count": int(len(segments)),
        "median_corr": float(np.median(corrs)),
        "mean_corr": float(np.mean(corrs)),
        "template_ptp": template_ptp,
        "residual_rms": float(np.mean(residual_rms)),
        "residual_ratio": float(np.mean(residual_rms / template_scale)),
        "hf_rms": float(np.mean(hf_rms)),
        "hf_ratio": float(np.mean(hf_rms / beat_scale)),
        "raw_std": raw_std,
        "raw_ptp": raw_ptp,
        "raw_hf_rms": raw_hf_rms,
        "raw_jump_p95": raw_jump_p95,
    }


def robust_upper_score(values: pd.Series) -> pd.Series:
    median = float(values.median())
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    if iqr < 1e-6:
        iqr = max(float(values.std()), 1e-6)
    return ((values - median) / iqr).clip(lower=0.0)


def robust_lower_score(values: pd.Series) -> pd.Series:
    median = float(values.median())
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    if iqr < 1e-6:
        iqr = max(float(values.std()), 1e-6)
    return ((median - values) / iqr).clip(lower=0.0)


def load_metadata(raw_dir: Path, split: str) -> pd.DataFrame:
    metadata_path = raw_dir / f"{split}_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    keep_cols = [
        column
        for column in ["patient_sn", "file_name", "csv_name", "label", "split"]
        if column in metadata.columns
    ]
    return metadata[keep_cols].copy()


def scan_split(qrs_dir: Path, raw_dir: Path, split: str) -> pd.DataFrame:
    qrs = np.load(qrs_dir / f"{split}_data.npy", mmap_mode="r")
    in_chans = np.load(qrs_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    raw = np.load(raw_dir / f"{split}_data.npy", mmap_mode="r")
    metadata = load_metadata(raw_dir, split)

    rows: list[dict[str, object]] = []
    for sample_index in range(len(qrs)):
        channel_vector = in_chans[sample_index]
        present_channel_ids = [
            int(channel_id) for channel_id in np.unique(channel_vector) if channel_id != 0
        ]
        meta_row = metadata.iloc[sample_index].to_dict()

        for channel_id in present_channel_ids:
            segments = qrs[sample_index][channel_vector == channel_id]
            raw_signal = raw[sample_index][CHANNEL_ID_TO_RAW_INDEX[channel_id]]
            metrics = compute_channel_metrics(segments, raw_signal)
            rows.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "channel_id": channel_id,
                    "channel_name": CHANNEL_ID_TO_NAME.get(channel_id, str(channel_id)),
                    **meta_row,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def add_scores(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    grouped = dataframe.groupby("channel_name", sort=False)

    dataframe["low_corr_score"] = grouped["median_corr"].transform(robust_lower_score)
    dataframe["high_residual_score"] = grouped["residual_ratio"].transform(
        robust_upper_score
    )
    dataframe["high_hf_score"] = grouped["hf_ratio"].transform(robust_upper_score)
    dataframe["raw_hf_score"] = grouped["raw_hf_rms"].transform(robust_upper_score)

    dataframe["suspicion_score"] = (
        dataframe["low_corr_score"]
        + dataframe["high_residual_score"]
        + dataframe["high_hf_score"]
        + 0.5 * dataframe["raw_hf_score"]
    )
    return dataframe.sort_values("suspicion_score", ascending=False).reset_index(
        drop=True
    )


def save_preview_plots(
    dataframe: pd.DataFrame,
    qrs_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    preview_count: int,
    subdir_name: str = "previews",
) -> None:
    if preview_count <= 0 or dataframe.empty:
        return

    preview_dir = output_dir / subdir_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    split_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for _, row in dataframe.head(preview_count).iterrows():
        split = str(row["split"])
        if split not in split_cache:
            split_cache[split] = (
                np.load(qrs_dir / f"{split}_data.npy", mmap_mode="r"),
                np.load(qrs_dir / f"{split}_data_in_chans.npy", mmap_mode="r"),
                np.load(raw_dir / f"{split}_data.npy", mmap_mode="r"),
            )

        qrs, in_chans, raw = split_cache[split]
        sample_index = int(row["sample_index"])
        channel_id = int(row["channel_id"])
        channel_name = str(row["channel_name"])
        qrs_segments = qrs[sample_index][in_chans[sample_index] == channel_id]
        raw_signal = raw[sample_index][CHANNEL_ID_TO_RAW_INDEX[channel_id]]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
        axes[0].plot(raw_signal, color="tab:blue", linewidth=1.0)
        axes[0].set_title(
            f"{split} sample {sample_index} {channel_name} raw waveform"
        )
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("Amplitude")

        offset = 0
        for segment in qrs_segments:
            axes[1].plot(
                np.arange(offset, offset + len(segment)),
                segment,
                color="tab:orange",
                linewidth=1.0,
            )
            offset += len(segment)
        axes[1].set_title(
            f"QRS segments | score={row['suspicion_score']:.2f} | corr={row['median_corr']:.3f}"
        )
        axes[1].set_xlabel("Concatenated QRS samples")
        axes[1].set_ylabel("Amplitude")

        safe_csv_name = str(row.get("csv_name", f"{sample_index}")).replace("\\", "_")
        plot_path = preview_dir / (
            f"{split}_idx{sample_index:05d}_{channel_name}_{safe_csv_name}.png"
        )
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        scan_split(args.qrs_dir.resolve(), args.raw_dir.resolve(), split)
        for split in args.splits
    ]
    metrics = add_scores(pd.concat(frames, ignore_index=True))

    all_metrics_path = args.output_dir / "all_channel_metrics.csv"
    suspicious_path = args.output_dir / "top_suspicious_channels.csv"
    threshold_path = args.output_dir / "threshold_hits.csv"
    nonflat_path = args.output_dir / "top_nonflat_suspicious_channels.csv"

    metrics.to_csv(all_metrics_path, index=False, encoding="utf-8-sig")
    metrics.head(args.top_k).to_csv(suspicious_path, index=False, encoding="utf-8-sig")
    metrics[metrics["suspicion_score"] >= args.score_threshold].to_csv(
        threshold_path, index=False, encoding="utf-8-sig"
    )
    nonflat_metrics = metrics[metrics["raw_ptp"] >= args.nonflat_min_ptp].copy()
    nonflat_metrics.head(args.top_k).to_csv(
        nonflat_path, index=False, encoding="utf-8-sig"
    )

    save_preview_plots(
        dataframe=metrics,
        qrs_dir=args.qrs_dir.resolve(),
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        preview_count=args.preview_count,
        subdir_name="previews",
    )
    save_preview_plots(
        dataframe=nonflat_metrics,
        qrs_dir=args.qrs_dir.resolve(),
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        preview_count=args.preview_count,
        subdir_name="previews_nonflat",
    )

    print(f"Saved all metrics to {all_metrics_path}")
    print(f"Saved top suspicious channels to {suspicious_path}")
    print(f"Saved threshold hits to {threshold_path}")
    print(f"Saved top non-flat suspicious channels to {nonflat_path}")
    print(
        "Top 10 suspicious channels:",
        metrics[
            [
                "split",
                "sample_index",
                "channel_name",
                "csv_name",
                "suspicion_score",
                "median_corr",
                "residual_ratio",
                "hf_ratio",
            ]
        ]
        .head(10)
        .to_string(index=False),
        sep="\n",
    )
    if not nonflat_metrics.empty:
        print(
            "Top 10 non-flat suspicious channels:",
            nonflat_metrics[
                [
                    "split",
                    "sample_index",
                    "channel_name",
                    "csv_name",
                    "suspicion_score",
                    "median_corr",
                    "residual_ratio",
                    "hf_ratio",
                    "raw_ptp",
                ]
            ]
            .head(10)
            .to_string(index=False),
            sep="\n",
        )


if __name__ == "__main__":
    main()
