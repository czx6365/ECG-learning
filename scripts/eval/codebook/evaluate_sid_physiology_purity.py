#!/usr/bin/env python3
"""Evaluate SID physiological purity for VQ-HBR assignments.

The script reuses the trained VQ-HBR tokenizer to assign each valid heartbeat
token to a SID, then measures whether each SID is internally consistent in
simple ECG physiology descriptors, not just waveform correlation.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timm.models import create_model


ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models import modeling_vqhbr as modeling_vqhbr  # noqa: F401


CONTINUOUS_FEATURES = [
    "qrs_duration_ms",
    "r_peak_amplitude",
    "st_deviation",
    "t_wave_amplitude",
    "rr_interval_ms",
]

DISCRETE_FEATURES = {
    "r_polarity": ["positive", "negative"],
    "t_polarity": ["positive", "inverted", "biphasic"],
    "st_state": ["elevation", "normal", "depression"],
    "qrs_type": ["normal", "wide"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate physiological purity of ECG SID vocabulary.")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/sid_physiology_purity"))
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means the whole split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--min_sid_count", type=int, default=20)
    parser.add_argument(
        "--record_length_samples",
        type=int,
        default=1000,
        help="Used only as fallback for approximate RR when exact R indices are not saved.",
    )
    parser.add_argument(
        "--time_block_samples",
        type=int,
        default=100,
        help="Tokenizer stores time_id = floor(r_peak_index / interval_length) + 1.",
    )
    parser.add_argument(
        "--qrs_onset_threshold_frac",
        type=float,
        default=0.15,
        help="Relative threshold of R deflection used for approximate QRS width.",
    )
    parser.add_argument("--qrs_wide_ms", type=float, default=120.0)
    parser.add_argument(
        "--st_threshold",
        type=float,
        default=0.1,
        help="Amplitude threshold in the current preprocessed ECG units.",
    )
    parser.add_argument(
        "--t_relative_threshold",
        type=float,
        default=0.05,
        help="Relative-to-R threshold for biphasic T-wave detection.",
    )
    parser.add_argument(
        "--t_abs_threshold",
        type=float,
        default=0.02,
        help="Minimum absolute T deflection threshold in current preprocessed ECG units.",
    )
    return parser.parse_args()


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = sorted(
        path.glob("checkpoint-*.pth"),
        key=lambda p: int(p.stem.split("-")[-1]) if p.stem.split("-")[-1].isdigit() else -1,
    )
    if not candidates:
        candidates = sorted(path.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {path}")
    return candidates[-1]


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def load_tokenizer(checkpoint_path: Path, codebook_size: int, codebook_dim: int, device: torch.device):
    patch_checkpoint_paths()
    resolved = resolve_checkpoint(checkpoint_path)
    model = create_model(
        "vqhbr",
        pretrained=True,
        pretrained_weight=str(resolved),
        as_tokenzer=True,
        n_code=codebook_size,
        code_dim=codebook_dim,
        quantize_kmeans_init=False,
    )
    model.to(device)
    model.eval()
    return model, resolved


def load_arrays(data_path: Path, split: str):
    data_path = data_path.expanduser().resolve()
    data = np.load(data_path / f"{split}_data.npy", mmap_mode="r")
    chans = np.load(data_path / f"{split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_path / f"{split}_data_in_times.npy", mmap_mode="r")
    return data, chans, times


@torch.no_grad()
def encode_batch(model, data: np.ndarray, chans: np.ndarray, times: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).to(device, non_blocking=True)
    in_chans = torch.from_numpy(np.array(chans, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    in_times = torch.from_numpy(np.array(times, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        code_ids = model.get_codebook_indices(x, in_chan_matrix=in_chans, in_time_matrix=in_times)
    return code_ids.detach().cpu().numpy().astype(np.int64, copy=False)


def robust_baseline(wave: np.ndarray, edge_n: int = 10) -> float:
    if len(wave) <= edge_n * 2:
        return float(np.median(wave))
    edges = np.concatenate([wave[:edge_n], wave[-edge_n:]])
    return float(np.median(edges))


def longest_true_span(mask: np.ndarray, center: int, left_limit: int, right_limit: int) -> tuple[int, int]:
    left = center
    while left > left_limit and mask[left - 1]:
        left -= 1
    right = center
    while right < right_limit and mask[right + 1]:
        right += 1
    return left, right


def estimate_qrs_bounds(signal: np.ndarray, r_idx: int, fs: float, threshold_frac: float) -> tuple[int, int]:
    abs_sig = np.abs(signal)
    peak_abs = float(abs_sig[r_idx])
    if peak_abs <= 1e-8:
        return r_idx, r_idx
    edge = np.concatenate([abs_sig[:10], abs_sig[-10:]]) if len(signal) >= 20 else abs_sig
    mad = float(np.median(np.abs(edge - np.median(edge))))
    noise_floor = 3.0 * 1.4826 * mad
    threshold = max(float(threshold_frac) * peak_abs, noise_floor, 1e-6)
    left_limit = max(0, r_idx - int(round(0.24 * fs)))
    right_limit = min(len(signal) - 1, r_idx + int(round(0.24 * fs)))
    mask = abs_sig >= threshold
    return longest_true_span(mask, r_idx, left_limit, right_limit)


def safe_window(signal: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, min(len(signal), start))
    end = max(start, min(len(signal), end))
    return signal[start:end]


def extract_wave_features(wave: np.ndarray, args: argparse.Namespace) -> dict:
    wave = wave.astype(np.float32, copy=False)
    baseline = robust_baseline(wave)
    signal = wave - baseline
    r_idx = int(np.argmax(np.abs(signal)))
    r_amp = float(signal[r_idx])
    r_abs = abs(r_amp)

    qrs_onset, qrs_offset = estimate_qrs_bounds(
        signal, r_idx=r_idx, fs=args.fs, threshold_frac=args.qrs_onset_threshold_frac
    )
    qrs_duration_ms = float((qrs_offset - qrs_onset + 1) / args.fs * 1000.0)

    st_start = qrs_offset + int(round(0.06 * args.fs))
    st_end = qrs_offset + int(round(0.12 * args.fs))
    if st_start >= len(signal) - 2:
        st_start = r_idx + int(round(0.12 * args.fs))
        st_end = r_idx + int(round(0.20 * args.fs))
    st_segment = safe_window(signal, st_start, st_end)
    st_deviation = float(np.mean(st_segment)) if st_segment.size else math.nan

    t_start = r_idx + int(round(0.16 * args.fs))
    t_end = r_idx + int(round(0.36 * args.fs))
    if t_start >= len(signal) - 2:
        t_start = qrs_offset + int(round(0.08 * args.fs))
        t_end = len(signal)
    t_segment = safe_window(signal, t_start, t_end)
    if t_segment.size:
        t_pos = float(np.max(t_segment))
        t_neg = float(-np.min(t_segment))
        t_amp = t_pos if t_pos >= t_neg else -t_neg
    else:
        t_pos = 0.0
        t_neg = 0.0
        t_amp = math.nan

    t_threshold = max(float(args.t_abs_threshold), float(args.t_relative_threshold) * max(r_abs, 1e-6))
    if t_pos >= t_threshold and t_neg >= t_threshold:
        t_polarity = "biphasic"
    elif t_amp >= 0:
        t_polarity = "positive"
    else:
        t_polarity = "inverted"

    if st_deviation >= args.st_threshold:
        st_state = "elevation"
    elif st_deviation <= -args.st_threshold:
        st_state = "depression"
    else:
        st_state = "normal"

    return {
        "qrs_duration_ms": qrs_duration_ms,
        "r_peak_amplitude": r_amp,
        "st_deviation": st_deviation,
        "t_wave_amplitude": t_amp,
        "r_polarity": "positive" if r_amp >= 0 else "negative",
        "t_polarity": t_polarity,
        "st_state": st_state,
        "qrs_type": "wide" if qrs_duration_ms >= args.qrs_wide_ms else "normal",
        "r_peak_index": r_idx,
        "qrs_onset_index": qrs_onset,
        "qrs_offset_index": qrs_offset,
    }


def estimate_rr_for_batch(chans: np.ndarray, times: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    rr = np.full(chans.shape, np.nan, dtype=np.float32)
    valid = (chans > 0) & (times > 0)
    time_block_ms = float(args.time_block_samples / args.fs * 1000.0)
    record_ms = float(args.record_length_samples / args.fs * 1000.0)

    for row_idx in range(chans.shape[0]):
        row_valid = valid[row_idx]
        for lead_id in np.unique(chans[row_idx, row_valid]):
            positions = np.where(row_valid & (chans[row_idx] == lead_id))[0]
            if positions.size == 0:
                continue
            fallback_rr = record_ms / max(float(positions.size), 1.0)
            if positions.size == 1:
                rr[row_idx, positions] = fallback_rr
                continue
            deltas = np.diff(times[row_idx, positions].astype(np.float32)) * time_block_ms
            positive_deltas = deltas[deltas > 0]
            typical_rr = float(np.median(positive_deltas)) if positive_deltas.size else fallback_rr
            rr[row_idx, positions[0]] = typical_rr
            rr_vals = np.where(deltas > 0, deltas, typical_rr)
            rr[row_idx, positions[1:]] = rr_vals
    return rr


class SidAccumulator:
    def __init__(self, codebook_size: int):
        self.counts = np.zeros(codebook_size, dtype=np.int64)
        self.continuous = {
            sid: {feature: [] for feature in CONTINUOUS_FEATURES} for sid in range(codebook_size)
        }
        self.discrete = {
            sid: {feature: Counter() for feature in DISCRETE_FEATURES} for sid in range(codebook_size)
        }

    def add(self, sid: int, features: dict) -> None:
        self.counts[sid] += 1
        for feature in CONTINUOUS_FEATURES:
            value = features.get(feature, math.nan)
            if value is not None and np.isfinite(value):
                self.continuous[sid][feature].append(float(value))
        for feature in DISCRETE_FEATURES:
            value = features.get(feature)
            if value is not None:
                self.discrete[sid][feature][str(value)] += 1


def continuous_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "iqr": math.nan,
            "cv": math.nan,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "count": int(arr.size),
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "cv": float(std / max(abs(mean), 1e-8)),
    }


def summarize_sid_metrics(acc: SidAccumulator, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for sid in range(len(acc.counts)):
        count = int(acc.counts[sid])
        row = {
            "sid": sid,
            "count": count,
            "low_count": bool(count < args.min_sid_count),
        }
        for feature in CONTINUOUS_FEATURES:
            stats = continuous_stats(acc.continuous[sid][feature])
            for name, value in stats.items():
                row[f"{feature}_{name}"] = value

        discrete_purities = []
        for feature, categories in DISCRETE_FEATURES.items():
            counter = acc.discrete[sid][feature]
            total = sum(counter.values())
            dominant = counter.most_common(1)[0][0] if total else ""
            purity = float(counter[dominant] / total) if total else math.nan
            row[f"{feature}_dominant"] = dominant
            row[f"{feature}_purity"] = purity
            row[f"{feature}_count"] = int(total)
            if np.isfinite(purity):
                discrete_purities.append(purity)
            for category in categories:
                n = int(counter.get(category, 0))
                row[f"{feature}_{category}_n"] = n
                row[f"{feature}_{category}_prop"] = float(n / total) if total else math.nan
            extra_categories = sorted(set(counter) - set(categories))
            for category in extra_categories:
                n = int(counter.get(category, 0))
                safe_category = category.replace(" ", "_")
                row[f"{feature}_{safe_category}_n"] = n
                row[f"{feature}_{safe_category}_prop"] = float(n / total) if total else math.nan

        row["physiology_purity_mean"] = float(np.mean(discrete_purities)) if discrete_purities else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "count") -> float:
    valid = df[value_col].notna() & df[weight_col].gt(0)
    if not valid.any():
        return math.nan
    return float(np.average(df.loc[valid, value_col], weights=df.loc[valid, weight_col]))


def vocabulary_summary(metrics: pd.DataFrame, args: argparse.Namespace, checkpoint: Path, data_path: Path) -> dict:
    eligible = metrics[(metrics["count"] >= args.min_sid_count) & metrics["physiology_purity_mean"].notna()]
    summary = {
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "split": args.split,
        "fs": float(args.fs),
        "codebook_size": int(args.codebook_size),
        "codebook_dim": int(args.codebook_dim),
        "min_sid_count": int(args.min_sid_count),
        "total_valid_tokens": int(metrics["count"].sum()),
        "active_sids": int((metrics["count"] > 0).sum()),
        "eligible_sids": int(len(eligible)),
        "low_count_sids": int(((metrics["count"] > 0) & (metrics["count"] < args.min_sid_count)).sum()),
        "weighted_average_physiology_purity": weighted_mean(eligible, "physiology_purity_mean"),
        "macro_average_physiology_purity": float(eligible["physiology_purity_mean"].mean()) if len(eligible) else math.nan,
        "feature_extraction": {
            "rr_interval_ms": "approximate; derived from saved time_id blocks because exact R peak indices are not stored in current QRS arrays",
            "qrs_duration_ms": f"threshold crossing around R peak at {args.qrs_onset_threshold_frac:g} of R deflection",
            "qrs_type_wide_threshold_ms": float(args.qrs_wide_ms),
            "st_threshold": float(args.st_threshold),
            "t_relative_threshold": float(args.t_relative_threshold),
            "t_abs_threshold": float(args.t_abs_threshold),
        },
    }
    summary["discrete_purity_weighted"] = {
        feature: weighted_mean(eligible, f"{feature}_purity") for feature in DISCRETE_FEATURES
    }
    summary["continuous_weighted_iqr"] = {
        feature: weighted_mean(eligible, f"{feature}_iqr") for feature in CONTINUOUS_FEATURES
    }
    summary["continuous_weighted_cv"] = {
        feature: weighted_mean(eligible, f"{feature}_cv") for feature in CONTINUOUS_FEATURES
    }
    summary["continuous_macro_iqr"] = {
        feature: float(eligible[f"{feature}_iqr"].mean()) if len(eligible) else math.nan
        for feature in CONTINUOUS_FEATURES
    }
    summary["continuous_macro_cv"] = {
        feature: float(eligible[f"{feature}_cv"].mean()) if len(eligible) else math.nan
        for feature in CONTINUOUS_FEATURES
    }
    return summary


def write_readme(output_dir: Path, args: argparse.Namespace, checkpoint: Path) -> None:
    text = f"""# SID Physiological Purity Evaluation

This folder contains physiological consistency metrics for VQ-HBR SID assignments.

## What It Measures

Correlation / shift-correlation answers:

> Do heartbeat waveforms assigned to the same SID look similar as signals?

Physiological purity answers:

> Are heartbeat tokens assigned to the same SID consistent in ECG morphology, such as QRS width, R polarity, ST state, and T-wave polarity?

These two views are complementary. A SID can have good waveform correlation but still mix medically different morphologies, such as positive and inverted T waves, elevated and depressed ST segments, or normal and wide QRS complexes.

## Inputs

- checkpoint: `{checkpoint}`
- data_path: `{args.data_path}`
- split: `{args.split}`
- fs: `{args.fs:g} Hz`
- codebook size: `{args.codebook_size}`
- codebook dim: `{args.codebook_dim}`

## Feature Definitions

Continuous features:

- `qrs_duration_ms`: approximate QRS width around the dominant R/QRS deflection.
- `r_peak_amplitude`: signed amplitude of the dominant R/QRS deflection after local baseline subtraction.
- `st_deviation`: mean ST-segment deviation after QRS offset, in current preprocessed ECG units.
- `t_wave_amplitude`: signed dominant T-wave deflection after the R peak, in current preprocessed ECG units.
- `rr_interval_ms`: approximate RR interval from saved `time_id` blocks. The current QRS arrays do not store exact R peak indices, so this is a coarse estimate.

Discrete features:

- `r_polarity`: `positive` / `negative`.
- `t_polarity`: `positive` / `inverted` / `biphasic`.
- `st_state`: `elevation` / `normal` / `depression`.
- `qrs_type`: `normal` / `wide`, with wide threshold `{args.qrs_wide_ms:g} ms`.

## Important Notes

- Padding and all-zero invalid heartbeat tokens are ignored.
- No token is removed because it has ST abnormality, T inversion, wide QRS, or other pathological morphology.
- SIDs with `count < {args.min_sid_count}` are marked as low-count and excluded from main vocabulary-level summary.
- ST/T thresholds are in the current preprocessed signal units. If future data are calibrated in mV, set `--st_threshold` and T thresholds to clinically calibrated values.
- For exact RR interval analysis, the tokenizer should save per-token original R peak indices in a future dataset version.

## Outputs

- `sid_physiology_metrics.csv`: full per-SID metrics.
- `vocabulary_summary.json`: vocabulary-level weighted and macro summaries.
- `low_purity_sids.csv`: Top20 eligible SIDs with lowest mean physiological purity.
- `README.md`: this explanation.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def iter_batches(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model, resolved_checkpoint = load_tokenizer(
        args.checkpoint_path, args.codebook_size, args.codebook_dim, device
    )
    data, chans, times = load_arrays(args.data_path, args.split)
    num_samples = min(len(data), args.max_samples if args.max_samples > 0 else len(data))
    acc = SidAccumulator(args.codebook_size)

    for start, end in iter_batches(num_samples, args.batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        batch_codes = encode_batch(model, batch_data, batch_chans, batch_times, device)
        batch_rr = estimate_rr_for_batch(batch_chans, batch_times, args)

        valid = (batch_chans > 0) & (batch_times > 0) & ~(batch_data == 0).all(axis=-1)
        positions = np.argwhere(valid)
        for local_idx, token_idx in positions:
            sid = int(batch_codes[local_idx, token_idx])
            if not 0 <= sid < args.codebook_size:
                continue
            features = extract_wave_features(batch_data[local_idx, token_idx], args)
            features["rr_interval_ms"] = float(batch_rr[local_idx, token_idx])
            acc.add(sid, features)

        if start == 0 or end == num_samples or (start // args.batch_size) % 20 == 0:
            print(f"Evaluated {end}/{num_samples}; valid tokens={int(acc.counts.sum())}", flush=True)

    metrics = summarize_sid_metrics(acc, args)
    metrics.sort_values(["count", "sid"], ascending=[False, True]).to_csv(
        args.output_dir / "sid_physiology_metrics.csv", index=False
    )

    eligible = metrics[(metrics["count"] >= args.min_sid_count) & metrics["physiology_purity_mean"].notna()]
    low_purity = eligible.sort_values(["physiology_purity_mean", "count"], ascending=[True, False]).head(20)
    low_purity.to_csv(args.output_dir / "low_purity_sids.csv", index=False)

    summary = vocabulary_summary(metrics, args, resolved_checkpoint, args.data_path)
    with open(args.output_dir / "vocabulary_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_readme(args.output_dir, args, resolved_checkpoint)

    print("\n=== SID Physiological Purity Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote outputs to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
