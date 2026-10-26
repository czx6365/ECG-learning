#!/usr/bin/env python3
"""Train Morphology-V2 physiology-aware ECG VQ codebook.

This keeps the morphology-only principle: SID assignment is determined only by
the current 96-point heartbeat waveform. Lead/time matrices are used only to
filter padding tokens, not as encoder context.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import heartlang.utils.utils as utils
from heartlang.utils.norm_ema_quantizer import NormEMAVectorQuantizer
from heartlang.utils.optim_factory import create_optimizer
from train_morphology_only_vq import MorphologyDecoder, MorphologyEncoder


IGNORE_INDEX = -100

DISCRETE_FEATURES = {
    "r_polarity": ["positive", "negative"],
    "t_polarity": ["positive", "inverted", "biphasic"],
    "st_state": ["elevation", "normal", "depression"],
    "qrs_type": ["normal", "wide"],
}

CONTINUOUS_FEATURES = [
    "qrs_duration_ms",
    "r_peak_amplitude",
    "st_deviation",
    "t_wave_amplitude",
]

LEGACY_DISCRETE_FEATURES = dict(DISCRETE_FEATURES)
LEGACY_CONTINUOUS_FEATURES = list(CONTINUOUS_FEATURES)
CONTINUOUS_CACHE_INDICES = list(range(len(CONTINUOUS_FEATURES)))


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Morphology-V2 physiology-aware ECG VQ training")
    parser.add_argument("--pretrain_data_dir", type=Path, required=True)
    parser.add_argument("--phys_cache_dir", type=Path, default=None)
    parser.add_argument(
        "--phys_schema",
        choices=("legacy_v2", "expanded_v1"),
        default="legacy_v2",
        help="Use the original four-feature cache or an expanded cache produced by extract_ecg_physiology_features.py.",
    )
    parser.add_argument(
        "--exclude_continuous_features",
        nargs="*",
        default=[],
        help="Continuous cache features to exclude from the auxiliary head and loss.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_ckpt_freq", type=int, default=50)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--codebook_n_emd", type=int, default=1024)
    parser.add_argument("--codebook_emd_dim", type=int, default=64)
    parser.add_argument("--token_len", type=int, default=96)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--quantize_kmeans_init", action="store_true")
    parser.add_argument("--phys_head_source", choices=("z", "z_q"), default="z_q")
    parser.add_argument("--lambda_waveform", type=float, default=1.0)
    parser.add_argument("--lambda_corr", type=float, default=0.2)
    parser.add_argument("--lambda_deriv", type=float, default=0.1)
    parser.add_argument("--lambda_vq", type=float, default=1.0)
    parser.add_argument("--lambda_phys", type=float, default=0.2)
    parser.add_argument("--lambda_feature_contrast", type=float, default=0.0)
    parser.add_argument("--lambda_usage_diversity", type=float, default=0.0)
    parser.add_argument(
        "--discrete_alpha",
        type=float,
        default=1.0,
        help="Multiplier for discrete morphology classification inside the physiology loss; use 0 for numeric-only evidence.",
    )
    parser.add_argument(
        "--loss_schedule",
        choices=("static", "curriculum_v2", "numeric_early", "feature_aware_early"),
        default="static",
        help="Use fixed CLI loss weights or the four-stage Curriculum-V2 schedule.",
    )
    parser.add_argument("--continuous_alpha", type=float, default=0.25)
    parser.add_argument(
        "--continuous_loss_reduction",
        choices=("element", "feature"),
        default="element",
        help="Average all valid elements or average each continuous feature loss equally.",
    )
    parser.add_argument(
        "--continuous_clip_z",
        type=float,
        default=0.0,
        help="If >0, clip standardized continuous targets to +/- this value before auxiliary loss.",
    )
    parser.add_argument("--feature_contrast_samples", type=int, default=512)
    parser.add_argument("--feature_contrast_k", type=int, default=4)
    parser.add_argument("--feature_contrast_temperature", type=float, default=0.1)
    parser.add_argument("--usage_diversity_samples", type=int, default=512)
    parser.add_argument("--usage_diversity_temperature", type=float, default=0.1)
    parser.add_argument(
        "--restart_unused_codes_every",
        type=int,
        default=0,
        help="If >0, recycle unused code vectors after every N training epochs.",
    )
    parser.add_argument(
        "--restart_unused_codes_until_epoch",
        type=int,
        default=0,
        help="Stop unused-code recycling after this epoch; 0 means no epoch limit.",
    )
    parser.add_argument(
        "--restart_unused_codes_max",
        type=int,
        default=0,
        help="Maximum unused codes recycled per eligible epoch; 0 means all unused codes.",
    )
    parser.add_argument("--qrs_onset_threshold_frac", type=float, default=0.15)
    parser.add_argument("--qrs_wide_ms", type=float, default=120.0)
    parser.add_argument("--st_threshold", type=float, default=0.1)
    parser.add_argument("--t_relative_threshold", type=float, default=0.05)
    parser.add_argument("--t_abs_threshold", type=float, default=0.02)
    parser.add_argument("--cache_chunk_records", type=int, default=512)
    parser.add_argument("--rebuild_phys_cache", action="store_true")
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=None, type=float, nargs="+")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decay_end", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--write_assignments", action="store_true")
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    parser.add_argument("--max_train_records", type=int, default=0)
    parser.add_argument("--max_val_records", type=int, default=0)
    parser.add_argument(
        "--lead_id",
        type=int,
        default=0,
        help="If >0, train/evaluate only heartbeat tokens from this channel id.",
    )
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def robust_std(std: np.ndarray) -> np.ndarray:
    std = std.astype(np.float32, copy=True)
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-6] = 1.0
    return std


def masked_mean(signal: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, length = signal.shape
    idx = np.arange(length, dtype=np.int32)[None, :]
    starts = np.clip(starts, 0, length).astype(np.int32)
    ends = np.clip(ends, 0, length).astype(np.int32)
    mask = (idx >= starts[:, None]) & (idx < ends[:, None])
    counts = mask.sum(axis=1)
    values = np.where(mask, signal, 0.0).sum(axis=1)
    mean = np.full(n, np.nan, dtype=np.float32)
    valid = counts > 0
    mean[valid] = (values[valid] / counts[valid]).astype(np.float32)
    return mean, valid


def masked_min_max(signal: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, length = signal.shape
    idx = np.arange(length, dtype=np.int32)[None, :]
    starts = np.clip(starts, 0, length).astype(np.int32)
    ends = np.clip(ends, 0, length).astype(np.int32)
    mask = (idx >= starts[:, None]) & (idx < ends[:, None])
    valid = mask.any(axis=1)
    max_val = np.where(mask, signal, -np.inf).max(axis=1).astype(np.float32)
    min_val = np.where(mask, signal, np.inf).min(axis=1).astype(np.float32)
    max_val[~valid] = 0.0
    min_val[~valid] = 0.0
    return min_val, max_val, valid


def extract_physiology_vectorized(waves: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized copy of the current physiology-purity feature rules."""
    waves = waves.astype(np.float32, copy=False)
    n, length = waves.shape
    discrete = np.full((n, len(DISCRETE_FEATURES)), IGNORE_INDEX, dtype=np.int16)
    continuous = np.zeros((n, len(CONTINUOUS_FEATURES)), dtype=np.float32)
    continuous_mask = np.zeros((n, len(CONTINUOUS_FEATURES)), dtype=bool)
    if n == 0:
        return discrete, continuous, continuous_mask

    edge_n = min(10, length // 2)
    edges = np.concatenate([waves[:, :edge_n], waves[:, -edge_n:]], axis=1)
    baseline = np.median(edges, axis=1).astype(np.float32)
    signal = waves - baseline[:, None]
    abs_signal = np.abs(signal)
    row_idx = np.arange(n)
    r_idx = np.argmax(abs_signal, axis=1).astype(np.int32)
    r_amp = signal[row_idx, r_idx].astype(np.float32)
    r_abs = np.abs(r_amp)

    edge_median = np.median(edges - baseline[:, None], axis=1)
    mad = np.median(np.abs((edges - baseline[:, None]) - edge_median[:, None]), axis=1)
    noise_floor = (3.0 * 1.4826 * mad).astype(np.float32)
    threshold = np.maximum.reduce(
        [
            np.asarray(args.qrs_onset_threshold_frac * r_abs, dtype=np.float32),
            noise_floor,
            np.full(n, 1e-6, dtype=np.float32),
        ]
    )
    idx = np.arange(length, dtype=np.int32)
    qrs_radius = int(round(0.24 * args.fs))
    left_limit = np.maximum(0, r_idx - qrs_radius)
    right_limit = np.minimum(length - 1, r_idx + qrs_radius)
    above = abs_signal >= threshold[:, None]
    left_invalid = (~above) & (idx[None, :] < r_idx[:, None]) & (idx[None, :] >= left_limit[:, None])
    last_invalid_left = np.where(left_invalid, idx[None, :], -1).max(axis=1)
    qrs_onset = np.where(last_invalid_left >= 0, last_invalid_left + 1, left_limit).astype(np.int32)
    right_invalid = (~above) & (idx[None, :] > r_idx[:, None]) & (idx[None, :] <= right_limit[:, None])
    first_invalid_right = np.where(right_invalid, idx[None, :], length).min(axis=1)
    qrs_offset = np.where(first_invalid_right < length, first_invalid_right - 1, right_limit).astype(np.int32)
    zero_peak = r_abs <= 1e-8
    qrs_onset[zero_peak] = r_idx[zero_peak]
    qrs_offset[zero_peak] = r_idx[zero_peak]
    qrs_duration_ms = ((qrs_offset - qrs_onset + 1).astype(np.float32) / args.fs * 1000.0).astype(np.float32)

    st_start = qrs_offset + int(round(0.06 * args.fs))
    st_end = qrs_offset + int(round(0.12 * args.fs))
    fallback_st = st_start >= (length - 2)
    st_start[fallback_st] = r_idx[fallback_st] + int(round(0.12 * args.fs))
    st_end[fallback_st] = r_idx[fallback_st] + int(round(0.20 * args.fs))
    st_deviation, st_valid = masked_mean(signal, st_start, st_end)

    t_start = r_idx + int(round(0.16 * args.fs))
    t_end = r_idx + int(round(0.36 * args.fs))
    fallback_t = t_start >= (length - 2)
    t_start[fallback_t] = qrs_offset[fallback_t] + int(round(0.08 * args.fs))
    t_end[fallback_t] = length
    t_min, t_max, t_valid = masked_min_max(signal, t_start, t_end)
    t_pos = t_max
    t_neg = -t_min
    t_amp = np.where(t_pos >= t_neg, t_pos, -t_neg).astype(np.float32)

    t_threshold = np.maximum(
        float(args.t_abs_threshold),
        float(args.t_relative_threshold) * np.maximum(r_abs, 1e-6),
    ).astype(np.float32)

    discrete[:, 0] = np.where(r_amp >= 0.0, 0, 1).astype(np.int16)
    t_labels = np.where(t_amp >= 0.0, 0, 1).astype(np.int16)
    t_labels[(t_pos >= t_threshold) & (t_neg >= t_threshold)] = 2
    discrete[:, 1] = t_labels
    st_labels = np.full(n, 1, dtype=np.int16)
    st_labels[st_deviation >= args.st_threshold] = 0
    st_labels[st_deviation <= -args.st_threshold] = 2
    discrete[:, 2] = st_labels
    discrete[:, 3] = np.where(qrs_duration_ms >= args.qrs_wide_ms, 1, 0).astype(np.int16)

    values = [qrs_duration_ms, r_amp, st_deviation, t_amp]
    valids = [
        np.isfinite(qrs_duration_ms),
        np.isfinite(r_amp),
        st_valid & np.isfinite(st_deviation),
        t_valid & np.isfinite(t_amp),
    ]
    for feature_idx, (value, valid) in enumerate(zip(values, valids)):
        continuous[:, feature_idx] = np.nan_to_num(value, nan=0.0).astype(np.float32)
        continuous_mask[:, feature_idx] = valid
    return discrete, continuous, continuous_mask


def expected_cache_meta(data_dir: Path, args: argparse.Namespace) -> dict:
    return {
        "data_dir": str(data_dir),
        "token_len": int(args.token_len),
        "fs": float(args.fs),
        "qrs_onset_threshold_frac": float(args.qrs_onset_threshold_frac),
        "qrs_wide_ms": float(args.qrs_wide_ms),
        "st_threshold": float(args.st_threshold),
        "t_relative_threshold": float(args.t_relative_threshold),
        "t_abs_threshold": float(args.t_abs_threshold),
        "discrete_features": DISCRETE_FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
    }


def cache_files(cache_dir: Path, split: str) -> dict[str, Path]:
    return {
        "discrete": cache_dir / f"{split}_phys_discrete.npy",
        "continuous": cache_dir / f"{split}_phys_continuous.npy",
        "continuous_mask": cache_dir / f"{split}_phys_continuous_mask.npy",
    }


def precompute_split_targets(data_dir: Path, cache_dir: Path, split: str, args: argparse.Namespace) -> None:
    data = np.load(data_dir / f"{split}_data.npy", mmap_mode="r")
    chans = np.load(data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_dir / f"{split}_data_in_times.npy", mmap_mode="r")
    n_records, n_tokens, token_len = data.shape
    if token_len != args.token_len:
        raise ValueError(f"{split} token_len={token_len}, expected {args.token_len}")
    files = cache_files(cache_dir, split)
    discrete = np.lib.format.open_memmap(
        files["discrete"],
        mode="w+",
        dtype=np.int16,
        shape=(n_records, n_tokens, len(DISCRETE_FEATURES)),
    )
    continuous = np.lib.format.open_memmap(
        files["continuous"],
        mode="w+",
        dtype=np.float32,
        shape=(n_records, n_tokens, len(CONTINUOUS_FEATURES)),
    )
    continuous_mask = np.lib.format.open_memmap(
        files["continuous_mask"],
        mode="w+",
        dtype=bool,
        shape=(n_records, n_tokens, len(CONTINUOUS_FEATURES)),
    )
    discrete[:] = IGNORE_INDEX
    continuous[:] = 0.0
    continuous_mask[:] = False

    chunk = max(1, int(args.cache_chunk_records))
    for start in range(0, n_records, chunk):
        end = min(start + chunk, n_records)
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        valid = (batch_chans > 0) & (batch_times > 0) & (np.abs(batch_data).max(axis=-1) > 1e-8)
        disc_chunk = np.full((end - start, n_tokens, len(DISCRETE_FEATURES)), IGNORE_INDEX, dtype=np.int16)
        cont_chunk = np.zeros((end - start, n_tokens, len(CONTINUOUS_FEATURES)), dtype=np.float32)
        cont_mask_chunk = np.zeros((end - start, n_tokens, len(CONTINUOUS_FEATURES)), dtype=bool)
        waves = batch_data[valid]
        if waves.size:
            disc_valid, cont_valid, cont_mask_valid = extract_physiology_vectorized(waves, args)
            disc_chunk[valid] = disc_valid
            cont_chunk[valid] = cont_valid
            cont_mask_chunk[valid] = cont_mask_valid
        discrete[start:end] = disc_chunk
        continuous[start:end] = cont_chunk
        continuous_mask[start:end] = cont_mask_chunk
        if start == 0 or end == n_records or (start // chunk) % 20 == 0:
            print(
                f"Precomputed {split} physiology targets {end}/{n_records}; "
                f"valid={int(valid.sum())}",
                flush=True,
            )
    discrete.flush()
    continuous.flush()
    continuous_mask.flush()


def compute_continuous_stats(cache_dir: Path, split: str, chunk_records: int = 1024) -> dict:
    files = cache_files(cache_dir, split)
    continuous = np.load(files["continuous"], mmap_mode="r")
    continuous_mask = np.load(files["continuous_mask"], mmap_mode="r")
    sums = np.zeros(len(CONTINUOUS_FEATURES), dtype=np.float64)
    sums_sq = np.zeros(len(CONTINUOUS_FEATURES), dtype=np.float64)
    counts = np.zeros(len(CONTINUOUS_FEATURES), dtype=np.int64)
    for start in range(0, continuous.shape[0], chunk_records):
        end = min(start + chunk_records, continuous.shape[0])
        values = np.asarray(continuous[start:end], dtype=np.float64)
        mask = np.asarray(continuous_mask[start:end], dtype=bool)
        for feature_idx in range(len(CONTINUOUS_FEATURES)):
            feature_mask = mask[..., feature_idx]
            if not feature_mask.any():
                continue
            vals = values[..., feature_idx][feature_mask]
            sums[feature_idx] += vals.sum()
            sums_sq[feature_idx] += np.square(vals).sum()
            counts[feature_idx] += vals.size
    means = np.divide(sums, np.maximum(counts, 1), out=np.zeros_like(sums), where=counts > 0)
    var = np.divide(sums_sq, np.maximum(counts, 1), out=np.zeros_like(sums_sq), where=counts > 0) - np.square(means)
    std = robust_std(np.sqrt(np.maximum(var, 0.0)))
    return {
        "features": CONTINUOUS_FEATURES,
        "mean": means.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "count": counts.astype(int).tolist(),
    }


def ensure_phys_cache(data_dir: Path, args: argparse.Namespace) -> tuple[Path, dict]:
    if args.phys_schema == "expanded_v1":
        return load_expanded_phys_cache(data_dir, args)

    set_physiology_schema(LEGACY_DISCRETE_FEATURES, LEGACY_CONTINUOUS_FEATURES)
    cache_dir = args.phys_cache_dir or (ROOT / "results" / "morphology_v2_phys_cache" / data_dir.name)
    cache_dir = resolve_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"
    stats_path = cache_dir / "continuous_train_stats.json"
    expected = expected_cache_meta(data_dir, args)
    can_reuse = False
    if not args.rebuild_phys_cache and meta_path.is_file() and stats_path.is_file():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            can_reuse = all(cache_files(cache_dir, split)[key].is_file() for split in ("train", "val") for key in cache_files(cache_dir, split)) and existing.get("settings") == expected
        except Exception:
            can_reuse = False
    if not can_reuse:
        print(f"Building physiology target cache in {cache_dir}", flush=True)
        for split in ("train", "val"):
            precompute_split_targets(data_dir, cache_dir, split, args)
        stats = compute_continuous_stats(cache_dir, "train", chunk_records=max(1, args.cache_chunk_records))
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {
            "settings": expected,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "continuous_train_stats": stats,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return cache_dir, stats


def set_physiology_schema(
    discrete_features: dict,
    continuous_features: list[str],
    continuous_cache_indices: list[int] | None = None,
) -> None:
    global DISCRETE_FEATURES, CONTINUOUS_FEATURES, CONTINUOUS_CACHE_INDICES
    DISCRETE_FEATURES = {
        str(feature): [str(category) for category in categories]
        for feature, categories in discrete_features.items()
    }
    CONTINUOUS_FEATURES = [str(feature) for feature in continuous_features]
    CONTINUOUS_CACHE_INDICES = (
        list(range(len(CONTINUOUS_FEATURES)))
        if continuous_cache_indices is None
        else [int(index) for index in continuous_cache_indices]
    )


def load_expanded_phys_cache(data_dir: Path, args: argparse.Namespace) -> tuple[Path, dict]:
    if args.phys_cache_dir is None:
        raise ValueError("--phys_cache_dir is required when --phys_schema expanded_v1 is selected")
    if args.rebuild_phys_cache:
        raise ValueError(
            "Expanded targets are built by extract_ecg_physiology_features.py; "
            "run that script explicitly instead of --rebuild_phys_cache."
        )
    cache_dir = resolve_path(args.phys_cache_dir)
    schema_path = cache_dir / "physiology_feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Expanded physiology schema is missing: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema") != "ecg_physiology_expanded_v1":
        raise ValueError(f"Unsupported expanded physiology schema: {schema.get('schema')!r}")
    cache_continuous_features = [str(name) for name in schema["continuous_features"]]
    excluded = {str(name) for name in args.exclude_continuous_features}
    unknown = excluded.difference(cache_continuous_features)
    if unknown:
        raise ValueError(f"Cannot exclude unknown continuous features: {sorted(unknown)}")
    selected_indices = [
        index for index, name in enumerate(cache_continuous_features) if name not in excluded
    ]
    selected_features = [cache_continuous_features[index] for index in selected_indices]
    if not selected_features:
        raise ValueError("At least one continuous physiology feature must remain enabled")
    set_physiology_schema(schema["discrete_features"], selected_features, selected_indices)

    for split in ("train", "val"):
        source = np.load(data_dir / f"{split}_data.npy", mmap_mode="r")
        files = cache_files(cache_dir, split)
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing expanded physiology cache files: {missing}")
        discrete = np.load(files["discrete"], mmap_mode="r")
        continuous = np.load(files["continuous"], mmap_mode="r")
        continuous_mask = np.load(files["continuous_mask"], mmap_mode="r")
        expected_prefix = tuple(source.shape[:2])
        if discrete.shape != (*expected_prefix, len(DISCRETE_FEATURES)):
            raise ValueError(f"{split} discrete shape {discrete.shape}, expected {(*expected_prefix, len(DISCRETE_FEATURES))}")
        expected_continuous = (*expected_prefix, len(cache_continuous_features))
        if continuous.shape != expected_continuous or continuous_mask.shape != expected_continuous:
            raise ValueError(
                f"{split} continuous shapes {continuous.shape}/{continuous_mask.shape}, "
                f"expected {expected_continuous}"
            )

    train_summary = next(
        (item for item in schema.get("splits", []) if item.get("split") == "train"),
        None,
    )
    if train_summary is None:
        raise ValueError("Expanded physiology schema has no train summary")
    by_feature = train_summary["continuous_stats"]
    stats = {
        "features": list(CONTINUOUS_FEATURES),
        "mean": [float(by_feature[name]["mean"]) for name in CONTINUOUS_FEATURES],
        "std": robust_std(
            np.asarray([by_feature[name]["std"] for name in CONTINUOUS_FEATURES], dtype=np.float32)
        ).astype(float).tolist(),
        "count": [int(by_feature[name]["count"]) for name in CONTINUOUS_FEATURES],
    }
    print(
        f"Using expanded physiology cache {cache_dir}: "
        f"{len(CONTINUOUS_FEATURES)} continuous + {len(DISCRETE_FEATURES)} discrete targets",
        flush=True,
    )
    return cache_dir, stats


class PhysTargetDataset(Dataset):
    def __init__(self, data_dir: Path, cache_dir: Path, split: str, max_records: int = 0, lead_id: int = 0):
        self.data = np.load(data_dir / f"{split}_data.npy", mmap_mode="r")
        self.chans = np.load(data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
        self.times = np.load(data_dir / f"{split}_data_in_times.npy", mmap_mode="r")
        files = cache_files(cache_dir, split)
        self.discrete = np.load(files["discrete"], mmap_mode="r")
        self.continuous = np.load(files["continuous"], mmap_mode="r")
        self.continuous_mask = np.load(files["continuous_mask"], mmap_mode="r")
        self.length = min(len(self.data), len(self.discrete))
        self.lead_id = int(lead_id)
        if max_records and max_records > 0:
            self.length = min(self.length, int(max_records))

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        chans = np.array(self.chans[idx], copy=True)
        times = np.array(self.times[idx], copy=True)
        if self.lead_id > 0:
            selected = chans == self.lead_id
            chans[~selected] = 0
            times[~selected] = 0
        return (
            np.array(self.data[idx], copy=True),
            chans,
            times,
            np.array(self.discrete[idx], copy=True),
            np.array(self.continuous[idx], copy=True)[..., CONTINUOUS_CACHE_INDICES],
            np.array(self.continuous_mask[idx], copy=True)[..., CONTINUOUS_CACHE_INDICES],
        )


class PhysiologyHeads(nn.Module):
    def __init__(self, code_dim: int):
        super().__init__()
        self.discrete_heads = nn.ModuleDict(
            {
                feature: nn.Sequential(
                    nn.Linear(code_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, len(categories)),
                )
                for feature, categories in DISCRETE_FEATURES.items()
            }
        )
        self.continuous_head = nn.Sequential(
            nn.Linear(code_dim, 128),
            nn.GELU(),
            nn.Linear(128, len(CONTINUOUS_FEATURES)),
        )

    def forward(
        self, z: torch.Tensor, include_discrete: bool = True
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        logits = (
            {feature: head(z) for feature, head in self.discrete_heads.items()}
            if include_discrete
            else {}
        )
        return logits, self.continuous_head(z)


class MorphologyV2PhysAwareVQ(nn.Module):
    """Morphology-only VQ with waveform, correlation, derivative and physiology losses."""

    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 64,
        token_len: int = 96,
        ema_decay: float = 0.99,
        quantize_kmeans_init: bool = False,
        phys_head_source: str = "z_q",
        continuous_mean: list[float] | None = None,
        continuous_std: list[float] | None = None,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.token_len = int(token_len)
        self.phys_head_source = str(phys_head_source)
        self.encoder = MorphologyEncoder(token_len=token_len, code_dim=codebook_dim)
        self.decoder = MorphologyDecoder(token_len=token_len, code_dim=codebook_dim)
        self.quantize = NormEMAVectorQuantizer(
            n_embed=codebook_size,
            embedding_dim=codebook_dim,
            beta=1.0,
            decay=ema_decay,
            kmeans_init=quantize_kmeans_init,
        )
        self.phys_heads = PhysiologyHeads(code_dim=codebook_dim)
        mean = torch.as_tensor(continuous_mean if continuous_mean is not None else [0.0] * len(CONTINUOUS_FEATURES), dtype=torch.float32)
        std = torch.as_tensor(continuous_std if continuous_std is not None else [1.0] * len(CONTINUOUS_FEATURES), dtype=torch.float32)
        std = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)
        self.register_buffer("continuous_mean", mean)
        self.register_buffer("continuous_std", std)

    @staticmethod
    def valid_mask(x: torch.Tensor, in_chan_matrix=None, in_time_matrix=None) -> torch.Tensor:
        valid = torch.abs(x).amax(dim=-1) > 1e-8
        if in_chan_matrix is not None:
            valid = valid & (in_chan_matrix > 0)
        if in_time_matrix is not None:
            valid = valid & (in_time_matrix > 0)
        return valid

    def encode_to_quantizer_features(self, x, in_chan_matrix=None, in_time_matrix=None):
        valid = self.valid_mask(x, in_chan_matrix, in_time_matrix)
        z_full = torch.zeros(*x.shape[:2], self.codebook_dim, device=x.device, dtype=x.dtype)
        if valid.any():
            z_full[valid] = self.encoder(x[valid])
        return z_full

    def encode_valid(self, x, in_chan_matrix=None, in_time_matrix=None):
        valid = self.valid_mask(x, in_chan_matrix, in_time_matrix)
        if not valid.any():
            empty_z = torch.empty(0, self.codebook_dim, device=x.device, dtype=x.dtype)
            empty_beats = torch.empty(0, self.token_len, device=x.device, dtype=x.dtype)
            empty_ids = torch.empty(0, dtype=torch.long, device=x.device)
            return valid, empty_beats, empty_z, empty_z, empty_ids, x.sum() * 0.0
        beats = x[valid]
        z = self.encoder(beats)
        z_q, quant_loss, embed_ind = self.quantize(z)
        return valid, beats, z, z_q, embed_ind, quant_loss

    @staticmethod
    def correlation_loss(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        target_centered = target - target.mean(dim=-1, keepdim=True)
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        numerator = torch.sum(target_centered * pred_centered, dim=-1)
        denominator = torch.linalg.norm(target_centered, dim=-1) * torch.linalg.norm(pred_centered, dim=-1)
        corr = numerator / denominator.clamp_min(eps)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        return 1.0 - corr.mean()

    def physiology_loss(
        self,
        source: torch.Tensor,
        discrete_targets: torch.Tensor | None,
        continuous_targets: torch.Tensor | None,
        continuous_mask: torch.Tensor | None,
        continuous_clip_z: float = 0.0,
        continuous_loss_reduction: str = "element",
        enable_discrete: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        log: dict[str, torch.Tensor] = {}
        zero = source.sum() * 0.0
        discrete_losses = []
        logits_by_feature, pred_continuous = self.phys_heads(
            source, include_discrete=enable_discrete
        )
        if enable_discrete and discrete_targets is not None and discrete_targets.numel() > 0:
            discrete_targets = discrete_targets.long()
            for feature_idx, feature in enumerate(DISCRETE_FEATURES):
                labels = discrete_targets[:, feature_idx]
                valid = labels != IGNORE_INDEX
                if valid.any():
                    logits = logits_by_feature[feature][valid].float()
                    target = labels[valid]
                    loss = F.cross_entropy(logits, target)
                    acc = (torch.argmax(logits, dim=-1) == target).float().mean()
                    discrete_losses.append(loss)
                    log[f"phys_{feature}_loss"] = loss.detach()
                    log[f"phys_{feature}_acc"] = acc.detach()
                else:
                    log[f"phys_{feature}_loss"] = zero.detach()
                    log[f"phys_{feature}_acc"] = zero.detach()
        if discrete_losses:
            discrete_loss = torch.stack(discrete_losses).mean()
        else:
            discrete_loss = zero

        continuous_loss = zero
        if continuous_targets is not None and continuous_mask is not None and continuous_targets.numel() > 0:
            continuous_targets = continuous_targets.float()
            continuous_mask = continuous_mask.bool()
            norm_targets = (continuous_targets - self.continuous_mean) / self.continuous_std
            if continuous_clip_z > 0:
                norm_targets = norm_targets.clamp(-continuous_clip_z, continuous_clip_z)
            finite = torch.isfinite(norm_targets) & continuous_mask
            if finite.any():
                if continuous_loss_reduction == "feature":
                    feature_losses = []
                    for feature_idx in range(len(CONTINUOUS_FEATURES)):
                        feature_mask = finite[:, feature_idx]
                        if feature_mask.any():
                            feature_losses.append(
                                F.smooth_l1_loss(
                                    pred_continuous[:, feature_idx].float()[feature_mask],
                                    norm_targets[:, feature_idx].float()[feature_mask],
                                )
                            )
                    if feature_losses:
                        continuous_loss = torch.stack(feature_losses).mean()
                else:
                    continuous_loss = F.smooth_l1_loss(
                        pred_continuous.float()[finite], norm_targets.float()[finite]
                    )
            pred_raw = pred_continuous.float() * self.continuous_std + self.continuous_mean
            for feature_idx, feature in enumerate(CONTINUOUS_FEATURES):
                mask = finite[:, feature_idx]
                if mask.any():
                    mae = torch.abs(pred_raw[:, feature_idx][mask] - continuous_targets[:, feature_idx][mask]).mean()
                    log[f"phys_{feature}_mae"] = mae.detach()
                else:
                    log[f"phys_{feature}_mae"] = zero.detach()
        log["phys_discrete_loss"] = discrete_loss.detach()
        log["phys_continuous_loss"] = continuous_loss.detach()
        return discrete_loss, continuous_loss, log

    def feature_contrastive_loss(
        self,
        z: torch.Tensor,
        continuous_targets: torch.Tensor | None,
        continuous_mask: torch.Tensor | None,
        max_samples: int,
        positive_k: int,
        temperature: float,
        clip_z: float,
    ) -> torch.Tensor:
        zero = z.sum() * 0.0
        if continuous_targets is None or continuous_mask is None or z.shape[0] < 2:
            return zero
        sample_count = min(int(max_samples), z.shape[0])
        if sample_count < 2:
            return zero
        if sample_count < z.shape[0]:
            selected = (
                torch.randperm(z.shape[0], device=z.device)[:sample_count]
                if self.training
                else torch.arange(sample_count, device=z.device)
            )
            z = z[selected]
            continuous_targets = continuous_targets[selected]
            continuous_mask = continuous_mask[selected]
        targets = (continuous_targets.float() - self.continuous_mean) / self.continuous_std
        if clip_z > 0:
            targets = targets.clamp(-clip_z, clip_z)
        target_mask = continuous_mask.bool() & torch.isfinite(targets)
        targets = torch.nan_to_num(targets)

        joint = target_mask[:, None, :] & target_mask[None, :, :]
        shared = joint.sum(dim=-1)
        feature_distance = (
            torch.where(joint, (targets[:, None, :] - targets[None, :, :]).square(), 0.0).sum(dim=-1)
            / shared.clamp_min(1)
        )
        pair_valid = shared >= min(2, targets.shape[-1])
        diagonal = torch.eye(sample_count, dtype=torch.bool, device=z.device)
        pair_valid &= ~diagonal
        feature_distance = feature_distance.masked_fill(~pair_valid, float("inf"))
        valid_anchor = pair_valid.any(dim=-1)
        if not valid_anchor.any():
            return zero

        k = min(max(1, int(positive_k)), sample_count - 1)
        nearest_distance, nearest = torch.topk(feature_distance, k=k, largest=False, dim=-1)
        positive_valid = torch.isfinite(nearest_distance)
        z_norm = F.normalize(z.float(), dim=-1)
        logits = (z_norm @ z_norm.T) / max(float(temperature), 1e-6)
        logits = logits.masked_fill(diagonal, -torch.inf)
        log_prob = F.log_softmax(logits, dim=-1)
        positive_log_prob = torch.gather(log_prob, 1, nearest)
        per_anchor = -(
            torch.where(positive_valid, positive_log_prob, 0.0).sum(dim=-1)
            / positive_valid.sum(dim=-1).clamp_min(1)
        )
        return per_anchor[valid_anchor].mean()

    def usage_diversity_loss(
        self,
        z: torch.Tensor,
        max_samples: int,
        temperature: float,
    ) -> torch.Tensor:
        zero = z.sum() * 0.0
        if z.shape[0] < 2 or self.codebook_size < 2:
            return zero
        sample_count = min(int(max_samples), z.shape[0])
        if sample_count < z.shape[0]:
            selected = (
                torch.randperm(z.shape[0], device=z.device)[:sample_count]
                if self.training
                else torch.arange(sample_count, device=z.device)
            )
            z = z[selected]
        z_norm = F.normalize(z.float(), dim=-1)
        codebook = F.normalize(self.quantize.embedding.weight.float(), dim=-1)
        probabilities = F.softmax(
            (z_norm @ codebook.T) / max(float(temperature), 1e-6), dim=-1
        )
        eps = 1e-8
        conditional_entropy = -(
            probabilities * torch.log(probabilities.clamp_min(eps))
        ).sum(dim=-1).mean()
        marginal = probabilities.mean(dim=0)
        marginal_entropy = -(marginal * torch.log(marginal.clamp_min(eps))).sum()
        normalizer = math.log(float(self.codebook_size))
        return 1.0 + (conditional_entropy - marginal_entropy) / normalizer

    def forward(
        self,
        x,
        in_chan_matrix=None,
        in_time_matrix=None,
        discrete_targets=None,
        continuous_targets=None,
        continuous_mask=None,
        loss_weights: dict | None = None,
    ):
        loss_weights = loss_weights or {}
        # SID assignment 前只使用 heartbeat waveform；lead/time 只过滤 padding。
        valid, beats, z, z_q, _embed_ind, quant_loss = self.encode_valid(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )
        split = "train" if self.training else "val"
        if beats.numel() == 0:
            zero = x.sum() * 0.0
            log = {
                f"{split}/waveform_loss": zero.detach(),
                f"{split}/corr_loss": zero.detach(),
                f"{split}/deriv_loss": zero.detach(),
                f"{split}/quant_loss": zero.detach(),
                f"{split}/phys_loss": zero.detach(),
                f"{split}/feature_contrast_loss": zero.detach(),
                f"{split}/usage_diversity_loss": zero.detach(),
                f"{split}/total_loss": zero.detach(),
                f"{split}/valid_tokens": torch.zeros((), device=x.device),
            }
            return zero, log

        rec = self.decoder(z_q)
        waveform_loss = F.mse_loss(rec, beats)
        corr_loss = self.correlation_loss(beats.float(), rec.float())
        deriv_loss = F.smooth_l1_loss(torch.diff(rec.float(), dim=-1), torch.diff(beats.float(), dim=-1))

        disc_valid = discrete_targets[valid] if discrete_targets is not None else None
        cont_valid = continuous_targets[valid] if continuous_targets is not None else None
        cont_mask_valid = continuous_mask[valid] if continuous_mask is not None else None
        phys_source = z_q if self.phys_head_source == "z_q" else z
        if float(loss_weights.get("lambda_phys", 0.2)) != 0.0:
            discrete_loss, continuous_loss, phys_log = self.physiology_loss(
                phys_source,
                disc_valid,
                cont_valid,
                cont_mask_valid,
                continuous_clip_z=float(loss_weights.get("continuous_clip_z", 0.0)),
                continuous_loss_reduction=str(loss_weights.get("continuous_loss_reduction", "element")),
                enable_discrete=float(loss_weights.get("discrete_alpha", 1.0)) != 0.0,
            )
        else:
            discrete_loss = zero = phys_source.sum() * 0.0
            continuous_loss = zero
            phys_log = {
                "phys_discrete_loss": zero.detach(),
                "phys_continuous_loss": zero.detach(),
            }
        phys_loss = (
            float(loss_weights.get("discrete_alpha", 1.0)) * discrete_loss
            + float(loss_weights.get("continuous_alpha", 0.25)) * continuous_loss
        )
        raw_phys_loss = discrete_loss + continuous_loss
        feature_contrast_loss = (
            self.feature_contrastive_loss(
                z,
                cont_valid,
                cont_mask_valid,
                max_samples=int(loss_weights.get("feature_contrast_samples", 512)),
                positive_k=int(loss_weights.get("feature_contrast_k", 4)),
                temperature=float(loss_weights.get("feature_contrast_temperature", 0.1)),
                clip_z=float(loss_weights.get("continuous_clip_z", 5.0)),
            )
            if float(loss_weights.get("lambda_feature_contrast", 0.0)) != 0.0
            else z.sum() * 0.0
        )
        usage_diversity_loss = (
            self.usage_diversity_loss(
                z,
                max_samples=int(loss_weights.get("usage_diversity_samples", 512)),
                temperature=float(loss_weights.get("usage_diversity_temperature", 0.1)),
            )
            if float(loss_weights.get("lambda_usage_diversity", 0.0)) != 0.0
            else z.sum() * 0.0
        )

        total = (
            float(loss_weights.get("lambda_waveform", 1.0)) * waveform_loss
            + float(loss_weights.get("lambda_corr", 0.2)) * corr_loss
            + float(loss_weights.get("lambda_deriv", 0.1)) * deriv_loss
            + float(loss_weights.get("lambda_vq", 1.0)) * quant_loss
            + float(loss_weights.get("lambda_phys", 0.2)) * phys_loss
            + float(loss_weights.get("lambda_feature_contrast", 0.0)) * feature_contrast_loss
            + float(loss_weights.get("lambda_usage_diversity", 0.0)) * usage_diversity_loss
        )
        log = {
            f"{split}/waveform_loss": waveform_loss.detach(),
            f"{split}/corr_loss": corr_loss.detach(),
            f"{split}/deriv_loss": deriv_loss.detach(),
            f"{split}/quant_loss": quant_loss.detach(),
            f"{split}/phys_loss": phys_loss.detach(),
            f"{split}/phys_raw_loss": raw_phys_loss.detach(),
            f"{split}/feature_contrast_loss": feature_contrast_loss.detach(),
            f"{split}/usage_diversity_loss": usage_diversity_loss.detach(),
            f"{split}/total_loss": total.detach(),
            f"{split}/valid_tokens": torch.as_tensor(float(beats.shape[0]), device=x.device),
        }
        for key, value in phys_log.items():
            log[f"{split}/{key}"] = value.detach()
        return total, log

    @torch.no_grad()
    def reconstruct(self, x, in_chan_matrix=None, in_time_matrix=None):
        valid, beats, _z, z_q, embed_ind, _quant_loss = self.encode_valid(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )
        rec_full = torch.zeros_like(x)
        if beats.numel() > 0:
            rec_full[valid] = self.decoder(z_q).to(rec_full.dtype)
        ids_full = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        if embed_ind.numel() > 0:
            ids_full[valid] = embed_ind
        return rec_full, ids_full

    @torch.no_grad()
    def get_codebook_indices(self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        _rec, ids = self.reconstruct(x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix)
        return ids

    @torch.no_grad()
    def get_codebook_quantize(self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        return self.encode_to_quantizer_features(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )


def build_loaders(args, data_dir: Path, cache_dir: Path):
    dataset_train = PhysTargetDataset(
        data_dir, cache_dir, "train", max_records=args.max_train_records, lead_id=args.lead_id
    )
    dataset_val = PhysTargetDataset(
        data_dir, cache_dir, "val", max_records=args.max_val_records, lead_id=args.lead_id
    )
    sampler_train = DistributedSampler(dataset_train, num_replicas=1, rank=0, shuffle=True)
    loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return dataset_train, dataset_val, sampler_train, loader_train, loader_val


def build_model(args, continuous_stats: dict) -> MorphologyV2PhysAwareVQ:
    return MorphologyV2PhysAwareVQ(
        codebook_size=args.codebook_n_emd,
        codebook_dim=args.codebook_emd_dim,
        token_len=args.token_len,
        ema_decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
        phys_head_source=args.phys_head_source,
        continuous_mean=continuous_stats["mean"],
        continuous_std=continuous_stats["std"],
    )


def loss_weights(args, epoch: int | None = None) -> dict:
    weights = {
        "lambda_waveform": args.lambda_waveform,
        "lambda_corr": args.lambda_corr,
        "lambda_deriv": args.lambda_deriv,
        "lambda_vq": args.lambda_vq,
        "lambda_phys": args.lambda_phys,
        "discrete_alpha": args.discrete_alpha,
        "continuous_alpha": args.continuous_alpha,
        "continuous_clip_z": args.continuous_clip_z,
        "continuous_loss_reduction": args.continuous_loss_reduction,
        "lambda_feature_contrast": args.lambda_feature_contrast,
        "lambda_usage_diversity": args.lambda_usage_diversity,
        "feature_contrast_samples": args.feature_contrast_samples,
        "feature_contrast_k": args.feature_contrast_k,
        "feature_contrast_temperature": args.feature_contrast_temperature,
        "usage_diversity_samples": args.usage_diversity_samples,
        "usage_diversity_temperature": args.usage_diversity_temperature,
    }
    if args.loss_schedule == "static":
        weights["stage"] = "static"
        return weights
    if args.loss_schedule == "numeric_early":
        if epoch is None:
            weights["stage"] = "numeric_early"
        elif epoch <= 5:
            weights["lambda_phys"] = min(float(args.lambda_phys), 0.01)
            weights["stage"] = "numeric_gentle_start"
        elif epoch <= 10:
            start = min(float(args.lambda_phys), 0.01)
            weights["lambda_phys"] = start + (float(args.lambda_phys) - start) * (epoch - 5) / 5.0
            weights["stage"] = "numeric_ramp"
        else:
            weights["stage"] = "numeric_full"
        return weights
    if args.loss_schedule == "feature_aware_early":
        if epoch is None:
            weights["stage"] = "feature_aware_early"
            return weights
        if epoch <= 5:
            factor = 0.2
            weights["stage"] = "feature_gentle_start"
        elif epoch <= 10:
            factor = 0.2 + 0.8 * (epoch - 5) / 5.0
            weights["stage"] = "feature_ramp"
        else:
            factor = 1.0
            weights["stage"] = "feature_full"
        weights["lambda_feature_contrast"] *= factor
        weights["lambda_usage_diversity"] *= factor
        return weights
    # curriculum_v2
    if epoch is None:
        weights["stage"] = "curriculum_v2"
        return weights
    if epoch <= 30:
        weights.update(lambda_corr=0.0, lambda_deriv=0.0, lambda_phys=0.0)
        weights["stage"] = "waveform_foundation"
    elif epoch <= 60:
        weights.update(lambda_corr=0.2, lambda_deriv=0.1, lambda_phys=0.0)
        weights["stage"] = "morphology_refinement"
    elif epoch <= 80:
        weights.update(
            lambda_corr=0.2,
            lambda_deriv=0.1,
            lambda_phys=0.2 * (epoch - 60) / 20.0,
        )
        weights["stage"] = "physiology_ramp_up"
    else:
        weights.update(lambda_corr=0.2, lambda_deriv=0.1, lambda_phys=0.2)
        weights["stage"] = "full_physiology_aware"
    return weights


def save_checkpoint(model, optimizer, args, epoch, tag=None) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{epoch}.pth" if tag is None else f"checkpoint-{tag}.pth"
    path = args.output_dir / name
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "args": vars(args),
        "model_type": "morphology_v2_physaware_vq",
        "discrete_features": DISCRETE_FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
        "continuous_mean": model.continuous_mean.detach().cpu().tolist(),
        "continuous_std": model.continuous_std.detach().cpu().tolist(),
    }
    torch.save(payload, path)
    torch.save(payload, args.output_dir / "checkpoint.pth")
    np.save(args.output_dir / "codebook_vectors.npy", model.quantize.embedding.weight.detach().cpu().numpy())
    return path


def load_checkpoint(model, optimizer, resume: Path, device: torch.device):
    patch_checkpoint_paths()
    ckpt = torch.load(resolve_path(resume), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


class MetricAverager:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, key: str, value) -> None:
        value = float(value)
        if math.isfinite(value):
            self.sums[key] += value
            self.counts[key] += 1

    def mean_dict(self) -> dict:
        return {key: self.sums[key] / max(self.counts[key], 1) for key in sorted(self.sums)}


def move_batch_to_device(batch, device):
    ecg = batch[0].float().to(device, non_blocking=True)
    chans = batch[1].to(device, non_blocking=True)
    times = batch[2].to(device, non_blocking=True)
    discrete = batch[3].long().to(device, non_blocking=True)
    continuous = batch[4].float().to(device, non_blocking=True)
    continuous_mask = batch[5].bool().to(device, non_blocking=True)
    return ecg, chans, times, discrete, continuous, continuous_mask


def train_one_epoch(model, loader, sampler, optimizer, device, epoch, args, lr_schedule, start_step, weights):
    model.train()
    sampler.set_epoch(epoch - 1)
    model.quantize.reset_cluster_size(device)
    meters = MetricAverager()
    step_count = 0
    for step, batch in enumerate(loader):
        if args.max_train_steps_per_epoch and step >= args.max_train_steps_per_epoch:
            break
        global_step = start_step + step
        if global_step < len(lr_schedule):
            set_lr(optimizer, lr_schedule[global_step])
        ecg, chans, times, discrete, continuous, continuous_mask = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss, log = model(
                ecg,
                in_chan_matrix=chans,
                in_time_matrix=times,
                discrete_targets=discrete,
                continuous_targets=continuous,
                continuous_mask=continuous_mask,
                loss_weights=weights,
            )
        loss.backward()
        if args.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        else:
            grad_norm = utils.get_grad_norm_(model.parameters())
        optimizer.step()
        meters.add("loss", loss.item())
        meters.add("grad_norm", grad_norm)
        meters.add("lr", optimizer.param_groups[0]["lr"])
        for key, value in log.items():
            meters.add(key.split("/")[-1], value.detach().item())
        step_count += 1
        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"Epoch {epoch} step {step + 1}/{len(loader)} "
                f"loss={loss.item():.6f} wave={log['train/waveform_loss'].item():.6f} "
                f"corr={log['train/corr_loss'].item():.6f} deriv={log['train/deriv_loss'].item():.6f} "
                f"vq={log['train/quant_loss'].item():.6f} phys={log['train/phys_loss'].item():.6f} "
                f"feat={log['train/feature_contrast_loss'].item():.6f} "
                f"usage={log['train/usage_diversity_loss'].item():.6f} "
                f"valid={int(log['train/valid_tokens'].item())}",
                flush=True,
            )
    stats = meters.mean_dict()
    usage = model.quantize.cluster_size.detach().float()
    total_usage = usage.sum()
    probs = usage[usage > 0] / total_usage if total_usage > 0 else usage.new_empty(0)
    stats["active_code"] = int((usage > 0).sum().item())
    stats["unused_code"] = int((usage <= 0).sum().item())
    stats["perplexity"] = float(torch.exp(-(probs * torch.log(probs)).sum()).item()) if probs.numel() else 0.0
    should_restart = (
        args.restart_unused_codes_every > 0
        and epoch % args.restart_unused_codes_every == 0
        and (
            args.restart_unused_codes_until_epoch <= 0
            or epoch <= args.restart_unused_codes_until_epoch
        )
    )
    stats["restarted_code"] = (
        model.quantize.restart_unused_codes(
            max_codes=(args.restart_unused_codes_max or None)
        )
        if should_restart
        else 0
    )
    return stats, step_count


@torch.no_grad()
def evaluate(model, loader, device, args, weights):
    model.eval()
    model.quantize.reset_cluster_size(device)
    meters = MetricAverager()
    total_valid = 0
    step_count = 0
    for step, batch in enumerate(loader):
        if args.max_eval_steps and step >= args.max_eval_steps:
            break
        ecg, chans, times, discrete, continuous, continuous_mask = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss, log = model(
                ecg,
                in_chan_matrix=chans,
                in_time_matrix=times,
                discrete_targets=discrete,
                continuous_targets=continuous,
                continuous_mask=continuous_mask,
                loss_weights=weights,
            )
        meters.add("loss", loss.item())
        for key, value in log.items():
            meters.add(key.split("/")[-1], value.detach().item())
        total_valid += int(log["val/valid_tokens"].item())
        step_count += 1
    stats = meters.mean_dict()
    usage = model.quantize.cluster_size.detach().cpu().numpy()
    total_usage = usage.sum()
    probs = usage[usage > 0] / total_usage if total_usage > 0 else np.array([])
    stats["valid_tokens"] = total_valid
    stats["active_code"] = int((usage > 0).sum())
    stats["unused_code"] = int((usage <= 0).sum())
    stats["perplexity"] = float(np.exp(-(probs * np.log(probs)).sum())) if probs.size else 0.0
    return stats


def write_history(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def write_assignments(model, split: str, loader, device, args):
    out = np.lib.format.open_memmap(
        args.output_dir / f"{split}_sid_assignments.npy",
        mode="w+",
        dtype=np.int32,
        shape=(len(loader.dataset), 256),
    )
    offset = 0
    for step, batch in enumerate(loader):
        ecg = batch[0].float().to(device, non_blocking=True)
        chans = batch[1].to(device, non_blocking=True)
        times = batch[2].to(device, non_blocking=True)
        ids = model.get_codebook_indices(ecg, in_chan_matrix=chans, in_time_matrix=times)
        bs = ids.shape[0]
        out[offset : offset + bs] = ids.cpu().numpy().astype(np.int32, copy=False)
        offset += bs
        if step == 0 or (step + 1) % 20 == 0:
            print(f"Wrote {split} SID assignments: {offset}/{len(loader.dataset)}", flush=True)
    out.flush()


def write_config(args, data_dir: Path, cache_dir: Path, train_len: int, val_len: int, continuous_stats: dict, base_lr: float):
    config = {
        "experiment": "morphology_v2_physaware_vq",
        "data_dir": str(data_dir),
        "phys_cache_dir": str(cache_dir),
        "train_records": int(train_len),
        "val_records": int(val_len),
        "codebook_size": args.codebook_n_emd,
        "codebook_dim": args.codebook_emd_dim,
        "token_len": args.token_len,
        "fs": args.fs,
        "batch_size_records": args.batch_size,
        "lead_id_filter": args.lead_id if args.lead_id > 0 else None,
        "base_lr_before_scaling": base_lr,
        "actual_lr_after_batch_scaling": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "epochs": args.epochs,
        "phys_head_source": args.phys_head_source,
        "loss_schedule": args.loss_schedule,
        "loss_weights": loss_weights(args),
        "curriculum_v2_stages": [
            {"epochs": "1-30", "lambda_corr": 0.0, "lambda_deriv": 0.0, "lambda_phys": 0.0},
            {"epochs": "31-60", "lambda_corr": 0.2, "lambda_deriv": 0.1, "lambda_phys": 0.0},
            {"epochs": "61-80", "lambda_corr": 0.2, "lambda_deriv": 0.1, "lambda_phys": "0.2 * (epoch - 60) / 20"},
            {"epochs": "81-100", "lambda_corr": 0.2, "lambda_deriv": 0.1, "lambda_phys": 0.2},
        ] if args.loss_schedule == "curriculum_v2" else None,
        "numeric_early_stages": [
            {"epochs": "1-5", "lambda_phys": "min(configured, 0.01)"},
            {"epochs": "6-10", "lambda_phys": "linear ramp to configured value"},
            {"epochs": "11-end", "lambda_phys": "configured value"},
        ] if args.loss_schedule == "numeric_early" else None,
        "feature_aware_early_stages": [
            {"epochs": "1-5", "feature_and_usage_weight_factor": 0.2},
            {"epochs": "6-10", "feature_and_usage_weight_factor": "linear ramp from 0.2 to 1.0"},
            {"epochs": "11-end", "feature_and_usage_weight_factor": 1.0},
        ] if args.loss_schedule == "feature_aware_early" else None,
        "unused_code_recycling": {
            "every_epochs": args.restart_unused_codes_every,
            "until_epoch": args.restart_unused_codes_until_epoch,
            "max_codes_per_restart": args.restart_unused_codes_max,
            "source": "recent normalized encoder vectors",
        },
        "discrete_features": DISCRETE_FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
        "continuous_train_stats": continuous_stats,
        "main_control_variable": (
            "four-stage loss schedule only; model, data, optimizer, codebook and physiology heads unchanged"
            if args.loss_schedule == "curriculum_v2"
            else (
                "add early low-weight continuous numeric evidence to morphology-only VQ"
                if args.loss_schedule == "numeric_early"
                else (
                    "shape pre-quantization embeddings with Stable4 feature similarity and preserve codebook diversity"
                    if args.loss_schedule == "feature_aware_early"
                    else "add corr/derivative/physiology-aware objectives to morphology-only VQ"
                )
            )
        ),
        "kept_morphology_only_principle": "SID uses current heartbeat waveform only; no spatial/temporal/position embedding before VQ.",
        "prohibited_context": [
            "lead embedding",
            "time embedding",
            "spatial embedding",
            "lead-region consistency",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


def main():
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    if args.log_dir is not None:
        args.log_dir = resolve_path(args.log_dir)
        args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    utils.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    data_dir = resolve_path(args.pretrain_data_dir)
    cache_dir, continuous_stats = ensure_phys_cache(data_dir, args)
    dataset_train, dataset_val, sampler_train, loader_train, loader_val = build_loaders(args, data_dir, cache_dir)
    model = build_model(args, continuous_stats).to(device)

    total_batch_size = args.batch_size
    base_lr = args.lr
    args.lr = total_batch_size / 128 * args.lr
    optimizer = create_optimizer(args, model)
    steps_per_epoch = len(loader_train)
    if args.max_train_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps_per_epoch)
    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    start_epoch = 1
    if args.resume:
        loaded_epoch = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch = loaded_epoch + 1
        print(f"Resumed from {args.resume}, next epoch={start_epoch}")

    write_config(args, data_dir, cache_dir, len(dataset_train), len(dataset_val), continuous_stats, base_lr)
    print(model, flush=True)

    if args.eval:
        stats = evaluate(model, loader_val, device, args, loss_weights(args, args.epochs))
        print(json.dumps({"val": stats}, ensure_ascii=False, indent=2), flush=True)
        if args.write_assignments:
            write_assignments(model, "val", loader_val, device, args)
        return

    start_time = time.time()
    history = []
    best_val = math.inf
    best_epoch = None
    for epoch in range(start_epoch, args.epochs + 1):
        weights = loss_weights(args, epoch)
        print(
            f"Epoch {epoch} loss schedule: stage={weights['stage']} "
            f"wave={weights['lambda_waveform']:.4f} corr={weights['lambda_corr']:.4f} "
            f"deriv={weights['lambda_deriv']:.4f} vq={weights['lambda_vq']:.4f} "
            f"phys={weights['lambda_phys']:.4f} "
            f"feature={weights['lambda_feature_contrast']:.4f} "
            f"usage={weights['lambda_usage_diversity']:.4f}",
            flush=True,
        )
        train_stats, _train_steps = train_one_epoch(
            model,
            loader_train,
            sampler_train,
            optimizer,
            device,
            epoch,
            args,
            lr_schedule,
            (epoch - 1) * steps_per_epoch,
            weights,
        )
        row = {
            "epoch": epoch,
            "stage": weights["stage"],
            **{key: value for key, value in weights.items() if key.startswith("lambda_")},
            **{f"train_{k}": v for k, v in train_stats.items()},
        }
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            val_stats = evaluate(model, loader_val, device, args, weights)
            row.update({f"val_{k}": v for k, v in val_stats.items()})
            print("Validation:", json.dumps(val_stats, ensure_ascii=False), flush=True)
            eligible_for_best = args.loss_schedule != "curriculum_v2" or epoch >= 81
            if eligible_for_best and val_stats.get("loss", math.inf) < best_val:
                best_val = val_stats["loss"]
                best_epoch = epoch
                best_path = save_checkpoint(model, optimizer, args, epoch, tag="best")
                print(f"Saved best checkpoint: {best_path}", flush=True)
        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs:
            ckpt_path = save_checkpoint(model, optimizer, args, epoch)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)
        history.append(row)
        write_history(args.output_dir, history)
        with (args.output_dir / "log.txt").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = str(dt.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}; best_epoch={best_epoch}", flush=True)
    write_assignments(model, "val", loader_val, device, args)


if __name__ == "__main__":
    main()
