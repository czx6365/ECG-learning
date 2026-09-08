#!/usr/bin/env python3
"""Rebuild 10-second ECG waveforms from reconstructed QRS tokens.

The QRS tokenizer stores fixed-length heartbeat patches but not the original
sample windows. This script recomputes the same R-peak windows from the saved
waveform split, decodes VQ-HBR tokens, stitches each lead back onto the 1000
point ECG timeline, and compares the stitched signal with the waveform input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wfdb import processing

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_codebook import choose_device, iter_batches, load_tokenizer, reconstruct_batch


ECG_DSA_LEADS = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_DSA_LEAD_IDS = [1, 2, 3, 7, 8, 9, 10, 11, 12]
LEAD_ID_TO_WAVE_INDEX = {lead_id: idx for idx, lead_id in enumerate(ECG_DSA_LEAD_IDS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch reconstructed heartbeat tokens back to raw ECG timelines."
    )
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--qrs_path", type=Path, required=True)
    parser.add_argument("--waveform_path", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--codebook_size", type=int, default=8192)
    parser.add_argument("--codebook_dim", type=int, default=128)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--token_len", type=int, default=96)
    parser.add_argument("--plot_samples", type=int, default=8)
    return parser.parse_args()


def load_split_arrays(root: Path, split: str):
    data = np.load(root / f"{split}_data.npy", mmap_mode="r")
    chans_path = root / f"{split}_data_in_chans.npy"
    times_path = root / f"{split}_data_in_times.npy"
    if chans_path.exists() and times_path.exists():
        return data, np.load(chans_path, mmap_mode="r"), np.load(times_path, mmap_mode="r")
    return data, None, None


def qrs_windows(ecg_signal: np.ndarray, fs: int, token_len: int) -> list[dict]:
    qrs_inds = processing.xqrs_detect(sig=ecg_signal[0], fs=fs, verbose=False)
    windows = []
    signal_len = ecg_signal.shape[-1]
    for i in range(1, len(qrs_inds) - 1):
        center = int(qrs_inds[i])
        start = int((qrs_inds[i] + qrs_inds[i - 1]) // 2)
        end = int((qrs_inds[i] + qrs_inds[i + 1]) // 2)
        start = max(start, 0)
        end = min(end, signal_len)
        actual_len = end - start

        if actual_len > token_len:
            start = max(center - token_len // 2, 0)
            end = min(start + token_len, signal_len)

        segment_start = max(token_len // 2 - (center - start), 0)
        segment_end = segment_start + (end - start)
        if segment_end > token_len:
            end -= segment_end - token_len
            segment_end = token_len

        if end > start and segment_end > segment_start:
            windows.append(
                {
                    "center": center,
                    "start": int(start),
                    "end": int(end),
                    "segment_start": int(segment_start),
                    "segment_end": int(segment_end),
                    "time_block": int(center // 100) + 1,
                }
            )
    return windows


def stitch_tokens(
    tokens: np.ndarray,
    chans: np.ndarray,
    times: np.ndarray,
    waveform: np.ndarray,
    fs: int,
    token_len: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    reconstructed = np.zeros_like(waveform, dtype=np.float32)
    coverage = np.zeros_like(waveform, dtype=np.float32)
    windows = qrs_windows(waveform, fs=fs, token_len=token_len)
    mismatched_channel = 0
    mismatched_time = 0
    placed_tokens = 0
    truncated_tokens = 0

    for lead_offset, lead_id in enumerate(ECG_DSA_LEAD_IDS):
        wave_idx = LEAD_ID_TO_WAVE_INDEX[lead_id]
        for beat_idx, window in enumerate(windows):
            token_idx = lead_offset * len(windows) + beat_idx
            if token_idx >= len(tokens):
                truncated_tokens += 1
                continue
            if int(chans[token_idx]) <= 0:
                truncated_tokens += 1
                continue
            if int(chans[token_idx]) != lead_id:
                mismatched_channel += 1
            if int(times[token_idx]) != int(window["time_block"]):
                mismatched_time += 1

            start = window["start"]
            end = window["end"]
            seg_start = window["segment_start"]
            seg_end = window["segment_end"]
            values = tokens[token_idx, seg_start:seg_end]
            expected_len = end - start
            if len(values) != expected_len:
                clipped = min(len(values), expected_len)
                values = values[:clipped]
                end = start + clipped
            if end <= start:
                continue
            reconstructed[wave_idx, start:end] += values.astype(np.float32, copy=False)
            coverage[wave_idx, start:end] += 1.0
            placed_tokens += 1

    mask = coverage > 0
    reconstructed[mask] /= coverage[mask]
    return reconstructed, mask, {
        "detected_qrs": int(len(windows) + 2) if windows else 0,
        "kept_beats": int(len(windows)),
        "placed_tokens": int(placed_tokens),
        "truncated_or_padding_tokens": int(truncated_tokens),
        "mismatched_channel_tokens": int(mismatched_channel),
        "mismatched_time_tokens": int(mismatched_time),
    }


def metric_block(original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> dict:
    count = int(mask.sum())
    if count <= 0:
        return {
            "points": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "prd": float("nan"),
            "snr_db": float("nan"),
        }
    err = reconstructed[mask].astype(np.float64) - original[mask].astype(np.float64)
    ref = original[mask].astype(np.float64)
    sse = float(np.square(err).sum())
    ref_energy = float(np.square(ref).sum())
    mse = sse / count
    mae = float(np.abs(err).mean())
    return {
        "points": count,
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "prd": float(100.0 * math.sqrt(sse / ref_energy)) if ref_energy > 0 else float("nan"),
        "snr_db": float(10.0 * math.log10(ref_energy / sse)) if sse > 0 and ref_energy > 0 else float("inf"),
    }


def add_sums(sums: dict, original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray, prefix: str) -> None:
    if not mask.any():
        return
    err = reconstructed[mask].astype(np.float64) - original[mask].astype(np.float64)
    ref = original[mask].astype(np.float64)
    sums[f"{prefix}_points"] += int(mask.sum())
    sums[f"{prefix}_sse"] += float(np.square(err).sum())
    sums[f"{prefix}_sae"] += float(np.abs(err).sum())
    sums[f"{prefix}_ref_energy"] += float(np.square(ref).sum())


def summarize_sums(sums: dict, prefix: str) -> dict:
    points = int(sums[f"{prefix}_points"])
    if points <= 0:
        return metric_block(np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool))
    mse = sums[f"{prefix}_sse"] / points
    mae = sums[f"{prefix}_sae"] / points
    sse = sums[f"{prefix}_sse"]
    ref_energy = sums[f"{prefix}_ref_energy"]
    return {
        "points": points,
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mae),
        "prd": float(100.0 * math.sqrt(sse / ref_energy)) if ref_energy > 0 else float("nan"),
        "snr_db": float(10.0 * math.log10(ref_energy / sse)) if sse > 0 and ref_energy > 0 else float("inf"),
    }


def plot_overlay(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    sample_id: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    time = np.arange(original.shape[-1])
    for axis, lead_idx in zip(axes, [0, 1, 3]):
        lead_name = ECG_DSA_LEADS[lead_idx]
        axis.plot(time, original[lead_idx], label="waveform input", linewidth=1.0)
        axis.plot(time, reconstructed[lead_idx], label="stitched decoder", linewidth=1.0, alpha=0.85)
        uncovered = ~mask[lead_idx]
        if uncovered.any():
            axis.fill_between(
                time,
                np.nanmin(original[lead_idx]),
                np.nanmax(original[lead_idx]),
                where=uncovered,
                color="lightgray",
                alpha=0.25,
                label="not covered by QRS tokens",
            )
        axis.set_ylabel(lead_name)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Sample index at 100 Hz")
    fig.suptitle(f"Sample {sample_id}: stitched raw ECG reconstruction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qrs_data, qrs_chans, qrs_times = load_split_arrays(args.qrs_path, args.split)
    waveform_data, _, _ = load_split_arrays(args.waveform_path, args.split)
    if qrs_chans is None or qrs_times is None:
        raise FileNotFoundError("QRS channel/time matrices are required for raw stitching.")
    if len(qrs_data) != len(waveform_data):
        raise ValueError(f"Split sample mismatch: qrs={len(qrs_data)} waveform={len(waveform_data)}")

    max_samples = len(qrs_data) if args.max_samples <= 0 else min(args.max_samples, len(qrs_data))
    device = choose_device(args.device)
    model, checkpoint_path = load_tokenizer(
        args.checkpoint_path,
        args.codebook_size,
        args.codebook_dim,
        device,
    )

    sums = {
        "covered_points": 0,
        "covered_sse": 0.0,
        "covered_sae": 0.0,
        "covered_ref_energy": 0.0,
        "full_zero_points": 0,
        "full_zero_sse": 0.0,
        "full_zero_sae": 0.0,
        "full_zero_ref_energy": 0.0,
        "tokenizer_covered_points": 0,
        "tokenizer_covered_sse": 0.0,
        "tokenizer_covered_sae": 0.0,
        "tokenizer_covered_ref_energy": 0.0,
        "tokenizer_full_zero_points": 0,
        "tokenizer_full_zero_sse": 0.0,
        "tokenizer_full_zero_sae": 0.0,
        "tokenizer_full_zero_ref_energy": 0.0,
    }
    sample_rows = []
    lead_rows = []
    align_rows = []
    plot_count = 0

    for start, end in iter_batches(max_samples, args.batch_size):
        batch_qrs = np.asarray(qrs_data[start:end])
        batch_chans = np.asarray(qrs_chans[start:end])
        batch_times = np.asarray(qrs_times[start:end])
        _, batch_rec = reconstruct_batch(model, batch_qrs, batch_chans, batch_times, device)

        for local_idx in range(end - start):
            sample_id = start + local_idx
            waveform = np.asarray(waveform_data[sample_id], dtype=np.float32)
            rec_wave, rec_mask, rec_info = stitch_tokens(
                batch_rec[local_idx],
                batch_chans[local_idx],
                batch_times[local_idx],
                waveform,
                fs=args.fs,
                token_len=args.token_len,
            )
            token_wave, token_mask, token_info = stitch_tokens(
                batch_qrs[local_idx],
                batch_chans[local_idx],
                batch_times[local_idx],
                waveform,
                fs=args.fs,
                token_len=args.token_len,
            )
            full_mask = np.ones_like(rec_mask, dtype=bool)
            covered = metric_block(waveform, rec_wave, rec_mask)
            full_zero = metric_block(waveform, rec_wave, full_mask)
            tokenizer_covered = metric_block(waveform, token_wave, token_mask)
            tokenizer_full_zero = metric_block(waveform, token_wave, full_mask)

            add_sums(sums, waveform, rec_wave, rec_mask, "covered")
            add_sums(sums, waveform, rec_wave, full_mask, "full_zero")
            add_sums(sums, waveform, token_wave, token_mask, "tokenizer_covered")
            add_sums(sums, waveform, token_wave, full_mask, "tokenizer_full_zero")

            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "coverage_ratio": float(rec_mask.mean()),
                    "covered_mse": covered["mse"],
                    "covered_rmse": covered["rmse"],
                    "covered_mae": covered["mae"],
                    "covered_prd": covered["prd"],
                    "covered_snr_db": covered["snr_db"],
                    "full_zero_mse": full_zero["mse"],
                    "full_zero_rmse": full_zero["rmse"],
                    "full_zero_mae": full_zero["mae"],
                    "full_zero_prd": full_zero["prd"],
                    "full_zero_snr_db": full_zero["snr_db"],
                    "tokenizer_covered_mse": tokenizer_covered["mse"],
                    "tokenizer_full_zero_mse": tokenizer_full_zero["mse"],
                    **{f"recon_{key}": value for key, value in rec_info.items()},
                    **{f"tokenizer_{key}": value for key, value in token_info.items()},
                }
            )

            for lead_idx, lead_name in enumerate(ECG_DSA_LEADS):
                lead_mask = rec_mask[lead_idx]
                lead_full_mask = np.ones_like(lead_mask, dtype=bool)
                lead_covered = metric_block(waveform[lead_idx], rec_wave[lead_idx], lead_mask)
                lead_full = metric_block(waveform[lead_idx], rec_wave[lead_idx], lead_full_mask)
                lead_rows.append(
                    {
                        "sample_id": sample_id,
                        "lead_name": lead_name,
                        "coverage_ratio": float(lead_mask.mean()),
                        "covered_mse": lead_covered["mse"],
                        "covered_rmse": lead_covered["rmse"],
                        "covered_mae": lead_covered["mae"],
                        "covered_prd": lead_covered["prd"],
                        "covered_snr_db": lead_covered["snr_db"],
                        "full_zero_mse": lead_full["mse"],
                        "full_zero_rmse": lead_full["rmmse"] if "rmmse" in lead_full else lead_full["rmse"],
                        "full_zero_mae": lead_full["mae"],
                        "full_zero_prd": lead_full["prd"],
                        "full_zero_snr_db": lead_full["snr_db"],
                    }
                )

            align_rows.append(
                {
                    "sample_id": sample_id,
                    "qrs_tokenizer_kept_beats": int(token_info["kept_beats"]),
                    "qrs_tokenizer_placed_tokens": int(token_info["placed_tokens"]),
                    "qrs_tokenizer_mismatched_channel_tokens": int(token_info["mismatched_channel_tokens"]),
                    "qrs_tokenizer_mismatched_time_tokens": int(token_info["mismatched_time_tokens"]),
                    "tokenizer_covered_mse": tokenizer_covered["mse"],
                    "tokenizer_full_zero_mse": tokenizer_full_zero["mse"],
                }
            )

            if plot_count < args.plot_samples:
                plot_overlay(
                    waveform,
                    rec_wave,
                    rec_mask,
                    sample_id,
                    args.output_dir / f"sample_{sample_id:05d}_raw_overlay.png",
                )
                plot_count += 1

        print(f"Raw ECG stitched samples {start}:{end}")

    sample_df = pd.DataFrame(sample_rows)
    lead_df = pd.DataFrame(lead_rows)
    align_df = pd.DataFrame(align_rows)
    sample_df.to_csv(args.output_dir / "raw_reconstruction_by_sample.csv", index=False)
    lead_df.to_csv(args.output_dir / "raw_reconstruction_by_sample_lead.csv", index=False)
    align_df.to_csv(args.output_dir / "raw_stitch_alignment.csv", index=False)

    lead_summary = (
        lead_df.groupby("lead_name", as_index=False)
        .agg(
            coverage_ratio=("coverage_ratio", "mean"),
            covered_mse=("covered_mse", "mean"),
            covered_rmse=("covered_rmse", "mean"),
            covered_mae=("covered_mae", "mean"),
            covered_prd=("covered_prd", "mean"),
            covered_snr_db=("covered_snr_db", "mean"),
            full_zero_mse=("full_zero_mse", "mean"),
            full_zero_rmse=("full_zero_rmse", "mean"),
            full_zero_mae=("full_zero_mae", "mean"),
        )
        .sort_values("lead_name")
    )
    lead_summary.to_csv(args.output_dir / "raw_reconstruction_by_lead.csv", index=False)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "evaluated_samples": int(max_samples),
        "waveform_shape": [int(v) for v in waveform_data.shape],
        "qrs_shape": [int(v) for v in qrs_data.shape],
        "fs": int(args.fs),
        "token_len": int(args.token_len),
        "coverage_ratio_mean": float(sample_df["coverage_ratio"].mean()) if not sample_df.empty else float("nan"),
        "coverage_ratio_p50": float(sample_df["coverage_ratio"].quantile(0.5)) if not sample_df.empty else float("nan"),
        "coverage_ratio_p05": float(sample_df["coverage_ratio"].quantile(0.05)) if not sample_df.empty else float("nan"),
        "decoder_vs_waveform_covered": summarize_sums(sums, "covered"),
        "decoder_vs_waveform_full_zero_fill": summarize_sums(sums, "full_zero"),
        "tokenizer_vs_waveform_covered": summarize_sums(sums, "tokenizer_covered"),
        "tokenizer_vs_waveform_full_zero_fill": summarize_sums(sums, "tokenizer_full_zero"),
        "sample_covered_mse_p50": float(sample_df["covered_mse"].quantile(0.5)) if not sample_df.empty else float("nan"),
        "sample_covered_mse_p90": float(sample_df["covered_mse"].quantile(0.9)) if not sample_df.empty else float("nan"),
        "sample_covered_mse_p95": float(sample_df["covered_mse"].quantile(0.95)) if not sample_df.empty else float("nan"),
        "sample_full_zero_mse_p50": float(sample_df["full_zero_mse"].quantile(0.5)) if not sample_df.empty else float("nan"),
        "sample_full_zero_mse_p90": float(sample_df["full_zero_mse"].quantile(0.9)) if not sample_df.empty else float("nan"),
        "sample_full_zero_mse_p95": float(sample_df["full_zero_mse"].quantile(0.95)) if not sample_df.empty else float("nan"),
        "total_mismatched_channel_tokens": int(sample_df["recon_mismatched_channel_tokens"].sum()) if not sample_df.empty else 0,
        "total_mismatched_time_tokens": int(sample_df["recon_mismatched_time_tokens"].sum()) if not sample_df.empty else 0,
    }
    with (args.output_dir / "raw_reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sample_df["covered_mse"].dropna().to_numpy(), bins=80, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Covered-region MSE")
    ax.set_ylabel("Sample count")
    ax.set_title("Raw ECG reconstruction error from stitched tokens")
    fig.tight_layout()
    fig.savefig(args.output_dir / "raw_reconstruction_covered_mse_hist.png", dpi=200)
    plt.close(fig)

    print("\nRaw ECG reconstruction summary")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  split: {args.split}, samples: {max_samples}")
    print(f"  coverage_ratio_mean: {summary['coverage_ratio_mean']:.6f}")
    print(f"  covered_mse: {summary['decoder_vs_waveform_covered']['mse']:.6f}")
    print(f"  covered_rmse: {summary['decoder_vs_waveform_covered']['rmse']:.6f}")
    print(f"  covered_mae: {summary['decoder_vs_waveform_covered']['mae']:.6f}")
    print(f"  covered_prd: {summary['decoder_vs_waveform_covered']['prd']:.3f}")
    print(f"  covered_snr_db: {summary['decoder_vs_waveform_covered']['snr_db']:.3f}")
    print(f"  full_zero_fill_mse: {summary['decoder_vs_waveform_full_zero_fill']['mse']:.6f}")
    print(f"  tokenizer_covered_mse: {summary['tokenizer_vs_waveform_covered']['mse']:.6g}")
    print(f"  outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
