#!/usr/bin/env python3
"""Extract interpretable physiological features from segmented ECG heartbeats.

The output is designed for SID-quality evaluation and optional auxiliary
supervision. Invalid/padded tokens remain masked and are never assigned a
physiology label.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np


IGNORE_INDEX = -100

CONTINUOUS_FEATURES = [
    "r_amplitude",
    "s_amplitude",
    "r_s_ratio",
    "qrs_duration_ms",
    "st_deviation",
    "st_slope_per_s",
    "t_amplitude",
    "t_peak_time_ms",
    "rr_interval_ms",
]

DISCRETE_FEATURES = {
    "r_polarity": ["positive", "negative", "absent"],
    "t_polarity": ["positive", "inverted", "flat", "biphasic"],
    "st_state": ["elevation", "depression", "normal"],
    "qrs_type": ["R", "rS", "QS", "qR", "wide"],
}

LANDMARKS = ["r_peak_index", "qrs_onset_index", "qrs_offset_index", "t_peak_index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Extract interpretable ECG heartbeat physiology features")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--chunk_records", type=int, default=512)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--qrs_search_start_frac", type=float, default=0.20)
    parser.add_argument("--qrs_search_end_frac", type=float, default=0.70)
    parser.add_argument("--qrs_threshold_frac", type=float, default=0.15)
    parser.add_argument("--qrs_wide_ms", type=float, default=120.0)
    parser.add_argument("--deflection_abs_threshold", type=float, default=0.02)
    parser.add_argument("--deflection_relative_threshold", type=float, default=0.05)
    parser.add_argument("--st_threshold", type=float, default=0.10)
    parser.add_argument("--t_abs_threshold", type=float, default=0.02)
    parser.add_argument("--t_relative_threshold", type=float, default=0.05)
    parser.add_argument("--time_block_samples", type=int, default=100)
    parser.add_argument("--rr_min_ms", type=float, default=300.0)
    parser.add_argument("--rr_max_ms", type=float, default=2500.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def masked_extreme(
    signal: np.ndarray, mask: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, length = signal.shape
    idx = np.arange(length, dtype=np.int16)[None, :]
    valid = mask.any(axis=1)
    if mode == "max":
        values = np.where(mask, signal, -np.inf)
        positions = values.argmax(axis=1)
        extrema = values[np.arange(n), positions]
    else:
        values = np.where(mask, signal, np.inf)
        positions = values.argmin(axis=1)
        extrema = values[np.arange(n), positions]
    extrema = extrema.astype(np.float32)
    positions = positions.astype(np.int16)
    extrema[~valid] = 0.0
    positions[~valid] = -1
    return extrema, positions, valid


def masked_mean(signal: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = mask.sum(axis=1)
    valid = count > 0
    out = np.zeros(signal.shape[0], dtype=np.float32)
    out[valid] = (np.where(mask, signal, 0.0).sum(axis=1)[valid] / count[valid]).astype(np.float32)
    return out, valid


def extract_wave_features(waves: np.ndarray, args: argparse.Namespace):
    waves = waves.astype(np.float32, copy=False)
    n, length = waves.shape
    idx = np.arange(length, dtype=np.int16)[None, :]
    edge_n = min(10, max(1, length // 4))
    edges = np.concatenate([waves[:, :edge_n], waves[:, -edge_n:]], axis=1)
    baseline = np.median(edges, axis=1).astype(np.float32)
    signal = waves - baseline[:, None]
    abs_signal = np.abs(signal)

    edge_centered = edges - baseline[:, None]
    edge_median = np.median(edge_centered, axis=1)
    mad = np.median(np.abs(edge_centered - edge_median[:, None]), axis=1)
    noise_floor = (3.0 * 1.4826 * mad).astype(np.float32)

    search_start = max(0, min(length - 1, int(round(args.qrs_search_start_frac * length))))
    search_end = max(search_start + 1, min(length, int(round(args.qrs_search_end_frac * length))))
    r_idx = (search_start + np.argmax(abs_signal[:, search_start:search_end], axis=1)).astype(np.int16)
    row = np.arange(n)
    dominant_amp = signal[row, r_idx].astype(np.float32)
    dominant_abs = np.abs(dominant_amp)

    qrs_threshold = np.maximum.reduce(
        [
            args.qrs_threshold_frac * dominant_abs,
            noise_floor,
            np.full(n, 1e-6, dtype=np.float32),
        ]
    )
    qrs_radius = int(round(0.24 * args.fs))
    left_limit = np.maximum(0, r_idx.astype(np.int32) - qrs_radius)
    right_limit = np.minimum(length - 1, r_idx.astype(np.int32) + qrs_radius)
    above = abs_signal >= qrs_threshold[:, None]
    left_invalid = (~above) & (idx < r_idx[:, None]) & (idx >= left_limit[:, None])
    last_left = np.where(left_invalid, idx, -1).max(axis=1)
    qrs_onset = np.where(last_left >= 0, last_left + 1, left_limit).astype(np.int16)
    right_invalid = (~above) & (idx > r_idx[:, None]) & (idx <= right_limit[:, None])
    first_right = np.where(right_invalid, idx, length).min(axis=1)
    qrs_offset = np.where(first_right < length, first_right - 1, right_limit).astype(np.int16)
    qrs_duration = ((qrs_offset - qrs_onset + 1).astype(np.float32) / args.fs * 1000.0)

    qrs_mask = (idx >= qrs_onset[:, None]) & (idx <= qrs_offset[:, None])
    pos_amp, pos_idx, pos_valid = masked_extreme(signal, qrs_mask, "max")
    neg_amp, neg_idx, neg_valid = masked_extreme(signal, qrs_mask, "min")
    sig_threshold = np.maximum.reduce(
        [
            np.full(n, args.deflection_abs_threshold, dtype=np.float32),
            args.deflection_relative_threshold * np.maximum(dominant_abs, 1e-6),
            noise_floor,
        ]
    )
    positive_present = pos_valid & (pos_amp >= sig_threshold)
    negative_present = neg_valid & (-neg_amp >= sig_threshold)
    r_present = dominant_abs >= sig_threshold

    # Signed dominant QRS deflection; this matches the polarity target.
    r_amplitude = dominant_amp
    r_polarity = np.full(n, 2, dtype=np.int16)  # absent
    r_polarity[r_present & (dominant_amp >= 0)] = 0
    r_polarity[r_present & (dominant_amp < 0)] = 1

    after_r = qrs_mask & (idx > pos_idx[:, None])
    s_amplitude, _s_idx, s_valid = masked_extreme(signal, after_r, "min")
    fallback_s = ~s_valid
    s_amplitude[fallback_s] = neg_amp[fallback_s]
    s_valid |= neg_valid
    s_significant = s_valid & (-s_amplitude >= sig_threshold)
    r_s_ratio = np.zeros(n, dtype=np.float32)
    ratio_valid = positive_present & s_significant
    r_s_ratio[ratio_valid] = pos_amp[ratio_valid] / np.maximum(-s_amplitude[ratio_valid], 1e-6)

    before_pos = qrs_mask & (idx < pos_idx[:, None])
    q_amp, _q_idx, q_valid = masked_extreme(signal, before_pos, "min")
    q_significant = q_valid & (-q_amp >= sig_threshold)

    qrs_type = np.zeros(n, dtype=np.int16)  # R
    qrs_type[(positive_present & s_significant) & (pos_amp < -s_amplitude)] = 1  # rS
    qrs_type[(~positive_present) & negative_present] = 2  # QS
    qrs_type[q_significant & positive_present & (pos_amp >= np.maximum(-s_amplitude, 0.0))] = 3  # qR
    qrs_type[qrs_duration >= args.qrs_wide_ms] = 4  # wide overrides shape

    st_start = qrs_offset.astype(np.int32) + int(round(0.06 * args.fs))
    st_end = qrs_offset.astype(np.int32) + int(round(0.12 * args.fs))
    st_mask = (idx >= st_start[:, None]) & (idx < st_end[:, None])
    st_deviation, st_valid = masked_mean(signal, st_mask)
    st_count = st_mask.sum(axis=1)
    st_x = (idx.astype(np.float32) - st_start[:, None].astype(np.float32)) / float(args.fs)
    st_x_mean = np.divide(
        np.where(st_mask, st_x, 0.0).sum(axis=1), np.maximum(st_count, 1), dtype=np.float32
    )
    st_y_mean = st_deviation
    x_centered = st_x - st_x_mean[:, None]
    y_centered = signal - st_y_mean[:, None]
    slope_num = np.where(st_mask, x_centered * y_centered, 0.0).sum(axis=1)
    slope_den = np.where(st_mask, x_centered * x_centered, 0.0).sum(axis=1)
    slope_valid = st_count >= 2
    st_slope = np.zeros(n, dtype=np.float32)
    st_slope[slope_valid] = (slope_num[slope_valid] / np.maximum(slope_den[slope_valid], 1e-8)).astype(np.float32)
    st_state = np.full(n, 2, dtype=np.int16)  # normal
    st_state[st_deviation >= args.st_threshold] = 0
    st_state[st_deviation <= -args.st_threshold] = 1

    t_start = r_idx.astype(np.int32) + int(round(0.16 * args.fs))
    t_end = r_idx.astype(np.int32) + int(round(0.40 * args.fs))
    t_mask = (idx >= t_start[:, None]) & (idx < t_end[:, None])
    t_pos, t_pos_idx, t_pos_valid = masked_extreme(signal, t_mask, "max")
    t_neg, t_neg_idx, t_neg_valid = masked_extreme(signal, t_mask, "min")
    choose_pos = t_pos >= -t_neg
    t_amplitude = np.where(choose_pos, t_pos, t_neg).astype(np.float32)
    t_peak_idx = np.where(choose_pos, t_pos_idx, t_neg_idx).astype(np.int16)
    t_valid = t_pos_valid | t_neg_valid
    t_threshold = np.maximum(
        args.t_abs_threshold,
        args.t_relative_threshold * np.maximum(dominant_abs, 1e-6),
    )
    t_pos_present = t_pos >= t_threshold
    t_neg_present = -t_neg >= t_threshold
    t_polarity = np.full(n, 2, dtype=np.int16)  # flat
    t_polarity[t_pos_present & ~t_neg_present] = 0
    t_polarity[t_neg_present & ~t_pos_present] = 1
    t_polarity[t_pos_present & t_neg_present] = 3
    t_peak_time = ((t_peak_idx.astype(np.float32) - r_idx.astype(np.float32)) / args.fs * 1000.0)
    t_time_valid = t_valid & (t_polarity != 2)

    continuous = np.stack(
        [
            r_amplitude,
            s_amplitude,
            r_s_ratio,
            qrs_duration,
            st_deviation,
            st_slope,
            t_amplitude,
            t_peak_time,
            np.zeros(n, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    continuous_mask = np.stack(
        [
            r_present,
            s_significant,
            ratio_valid,
            np.ones(n, dtype=bool),
            st_valid,
            slope_valid,
            t_valid,
            t_time_valid,
            np.zeros(n, dtype=bool),
        ],
        axis=1,
    )
    discrete = np.stack([r_polarity, t_polarity, st_state, qrs_type], axis=1).astype(np.int16)
    landmarks = np.stack([r_idx, qrs_onset, qrs_offset, t_peak_idx], axis=1).astype(np.int16)
    return continuous, continuous_mask, discrete, landmarks


def estimate_rr(chans: np.ndarray, times: np.ndarray, args: argparse.Namespace):
    rr = np.zeros(chans.shape, dtype=np.float32)
    mask = np.zeros(chans.shape, dtype=bool)
    block_ms = args.time_block_samples / args.fs * 1000.0
    for row in range(chans.shape[0]):
        valid = (chans[row] > 0) & (times[row] > 0)
        for lead in np.unique(chans[row, valid]):
            positions = np.flatnonzero(valid & (chans[row] == lead))
            if positions.size < 2:
                continue
            order = np.lexsort((positions, times[row, positions]))
            positions = positions[order]
            delta = np.diff(times[row, positions].astype(np.float32)) * block_ms
            plausible = (delta >= args.rr_min_ms) & (delta <= args.rr_max_ms)
            rr[row, positions[1:]] = delta
            mask[row, positions[1:]] = plausible
    return rr, mask


class RunningStats:
    def __init__(self, size: int):
        self.count = np.zeros(size, dtype=np.int64)
        self.sum = np.zeros(size, dtype=np.float64)
        self.sum_sq = np.zeros(size, dtype=np.float64)

    def update(self, values: np.ndarray, mask: np.ndarray):
        for col in range(values.shape[-1]):
            x = values[..., col][mask[..., col]].astype(np.float64)
            if x.size:
                self.count[col] += x.size
                self.sum[col] += x.sum()
                self.sum_sq[col] += np.square(x).sum()

    def summary(self):
        mean = np.divide(self.sum, np.maximum(self.count, 1))
        var = np.divide(self.sum_sq, np.maximum(self.count, 1)) - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        return {
            name: {"count": int(self.count[i]), "mean": float(mean[i]), "std": float(std[i])}
            for i, name in enumerate(CONTINUOUS_FEATURES)
        }


def output_files(output_dir: Path, split: str):
    return {
        "continuous": output_dir / f"{split}_phys_continuous.npy",
        "continuous_mask": output_dir / f"{split}_phys_continuous_mask.npy",
        "discrete": output_dir / f"{split}_phys_discrete.npy",
        "landmarks": output_dir / f"{split}_phys_landmarks.npy",
    }


def process_split(args: argparse.Namespace, split: str):
    data_path = args.data_dir / f"{split}_data.npy"
    chans_path = args.data_dir / f"{split}_data_in_chans.npy"
    times_path = args.data_dir / f"{split}_data_in_times.npy"
    if not data_path.exists():
        print(f"Skip {split}: missing {data_path}", flush=True)
        return {"split": split, "skipped": True}
    data = np.load(data_path, mmap_mode="r")
    chans = np.load(chans_path, mmap_mode="r")
    times = np.load(times_path, mmap_mode="r")
    n = min(len(data), args.max_records if args.max_records > 0 else len(data))
    tokens, length = data.shape[1], data.shape[2]
    files = output_files(args.output_dir, split)
    if not args.overwrite and all(path.exists() for path in files.values()):
        print(f"Reuse completed split {split}", flush=True)
        return {"split": split, "records": n, "reused": True}

    continuous = np.lib.format.open_memmap(
        files["continuous"], mode="w+", dtype=np.float32,
        shape=(n, tokens, len(CONTINUOUS_FEATURES)),
    )
    continuous_mask = np.lib.format.open_memmap(
        files["continuous_mask"], mode="w+", dtype=bool,
        shape=(n, tokens, len(CONTINUOUS_FEATURES)),
    )
    discrete = np.lib.format.open_memmap(
        files["discrete"], mode="w+", dtype=np.int16,
        shape=(n, tokens, len(DISCRETE_FEATURES)),
    )
    landmarks = np.lib.format.open_memmap(
        files["landmarks"], mode="w+", dtype=np.int16,
        shape=(n, tokens, len(LANDMARKS)),
    )
    continuous[:] = 0.0
    continuous_mask[:] = False
    discrete[:] = IGNORE_INDEX
    landmarks[:] = -1
    stats = RunningStats(len(CONTINUOUS_FEATURES))
    discrete_counts = {
        feature: np.zeros(len(categories), dtype=np.int64)
        for feature, categories in DISCRETE_FEATURES.items()
    }
    valid_total = 0

    for start in range(0, n, args.chunk_records):
        end = min(start + args.chunk_records, n)
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        valid = (
            (batch_chans > 0)
            & (batch_times > 0)
            & (np.abs(batch_data).max(axis=-1) > 1e-8)
        )
        cont_chunk = np.zeros((end - start, tokens, len(CONTINUOUS_FEATURES)), dtype=np.float32)
        mask_chunk = np.zeros_like(cont_chunk, dtype=bool)
        disc_chunk = np.full((end - start, tokens, len(DISCRETE_FEATURES)), IGNORE_INDEX, dtype=np.int16)
        landmark_chunk = np.full((end - start, tokens, len(LANDMARKS)), -1, dtype=np.int16)
        if valid.any():
            c, m, d, lm = extract_wave_features(batch_data[valid], args)
            cont_chunk[valid] = c
            mask_chunk[valid] = m
            disc_chunk[valid] = d
            landmark_chunk[valid] = lm
        rr, rr_mask = estimate_rr(batch_chans, batch_times, args)
        rr_mask &= valid
        cont_chunk[..., CONTINUOUS_FEATURES.index("rr_interval_ms")] = rr
        mask_chunk[..., CONTINUOUS_FEATURES.index("rr_interval_ms")] = rr_mask
        continuous[start:end] = cont_chunk
        continuous_mask[start:end] = mask_chunk
        discrete[start:end] = disc_chunk
        landmarks[start:end] = landmark_chunk
        stats.update(cont_chunk, mask_chunk)
        for feature_idx, (feature, categories) in enumerate(DISCRETE_FEATURES.items()):
            labels = disc_chunk[..., feature_idx][valid]
            discrete_counts[feature] += np.bincount(labels, minlength=len(categories))[: len(categories)]
        valid_total += int(valid.sum())
        if start == 0 or end == n or (start // args.chunk_records) % 20 == 0:
            print(f"{split}: {end}/{n} records, valid heartbeat tokens={valid_total}", flush=True)

    for array in (continuous, continuous_mask, discrete, landmarks):
        array.flush()
    summary = {
        "split": split,
        "records": n,
        "valid_heartbeat_tokens": valid_total,
        "continuous_stats": stats.summary(),
        "discrete_counts": {
            feature: {categories[i]: int(counts[i]) for i in range(len(categories))}
            for (feature, categories), counts in zip(DISCRETE_FEATURES.items(), discrete_counts.values())
        },
        "files": {key: str(path) for key, path in files.items()},
    }
    (args.output_dir / f"{split}_phys_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    args = parse_args()
    args.data_dir = resolve(args.data_dir)
    args.output_dir = resolve(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [process_split(args, split) for split in args.splits]
    config = {
        "schema": "ecg_physiology_expanded_v1",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "continuous_features": CONTINUOUS_FEATURES,
        "continuous_units": {
            "r_amplitude": "preprocessed_ecg_unit",
            "s_amplitude": "preprocessed_ecg_unit_signed",
            "r_s_ratio": "absolute_ratio",
            "qrs_duration_ms": "ms",
            "st_deviation": "preprocessed_ecg_unit",
            "st_slope_per_s": "preprocessed_ecg_unit_per_second",
            "t_amplitude": "preprocessed_ecg_unit_signed",
            "t_peak_time_ms": "ms_after_detected_r_peak",
            "rr_interval_ms": "ms_approx_from_time_bucket",
        },
        "discrete_features": DISCRETE_FEATURES,
        "landmarks": LANDMARKS,
        "ignore_index": IGNORE_INDEX,
        "settings": {
            key: value for key, value in vars(args).items()
            if key not in {"data_dir", "output_dir", "splits", "overwrite"}
        },
        "rr_limitation": (
            "RR uses differences between stored time buckets. It is masked when the delta is zero or outside "
            "the configured physiological range; exact R-peak timestamps are required for clinical-grade RR."
        ),
        "qrs_type_priority": "wide overrides R/rS/QS/qR",
        "splits": summaries,
        "training_ready": (
            "Use *_phys_continuous.npy with *_phys_continuous_mask.npy for regression heads and "
            "*_phys_discrete.npy with ignore_index=-100 for classification heads."
        ),
    }
    (args.output_dir / "physiology_feature_schema.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
