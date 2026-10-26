#!/usr/bin/env python3
"""Unified comparison for contextual H clean, Morphology-only and Morphology-V2 VQ."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from timm.models import create_model


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "scripts" / "eval" / "codebook"))

from heartlang.models import modeling_vqhbr as modeling_vqhbr  # noqa: F401
from train_morphology_only_vq import MorphologyOnlyVQ
from train_morphology_v2_physaware_vq import MorphologyV2PhysAwareVQ, set_physiology_schema
from evaluate_sid_physiology_purity import (
    CONTINUOUS_FEATURES,
    DISCRETE_FEATURES,
    SidAccumulator,
    estimate_rr_for_batch,
    extract_wave_features,
    summarize_sid_metrics,
    vocabulary_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare H clean, morphology-only and Morphology-V2 codebooks.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: NAME,TYPE,CHECKPOINT. TYPE is contextual_vqhbr, morphology_only, or morphology_v2.",
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--reservoir_per_sid", type=int, default=256)
    parser.add_argument("--min_count_for_distance", type=int, default=50)
    parser.add_argument("--intra_pairs_per_sid", type=int, default=256)
    parser.add_argument("--inter_pairs", type=int, default=200000)
    parser.add_argument("--max_shift", type=int, default=4)
    parser.add_argument("--corr_threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--min_sid_count", type=int, default=20)
    parser.add_argument("--lead_id", type=int, default=0, help="If >0, evaluate only this channel id.")
    parser.add_argument("--record_length_samples", type=int, default=1000)
    parser.add_argument("--time_block_samples", type=int, default=100)
    parser.add_argument("--qrs_onset_threshold_frac", type=float, default=0.15)
    parser.add_argument("--qrs_wide_ms", type=float, default=120.0)
    parser.add_argument("--st_threshold", type=float, default=0.1)
    parser.add_argument("--t_relative_threshold", type=float, default=0.05)
    parser.add_argument("--t_abs_threshold", type=float, default=0.02)
    return parser.parse_args()


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def resolve_checkpoint(path: Path) -> Path:
    path = resolve_path(path)
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


def parse_run_spec(spec: str) -> dict:
    parts = [part.strip() for part in spec.split(",", 2)]
    if len(parts) != 3:
        raise ValueError("--run must be formatted as NAME,TYPE,CHECKPOINT")
    name, model_type, checkpoint = parts
    if model_type not in {"contextual_vqhbr", "morphology_only", "morphology_v2"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    return {"name": name, "model_type": model_type, "checkpoint": Path(checkpoint)}


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def load_arrays(data_path: Path, split: str):
    data_path = resolve_path(data_path)
    data = np.load(data_path / f"{split}_data.npy", mmap_mode="r")
    chans = np.load(data_path / f"{split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_path / f"{split}_data_in_times.npy", mmap_mode="r")
    return data_path, data, chans, times


def load_model(run: dict, args: argparse.Namespace, device: torch.device):
    patch_checkpoint_paths()
    checkpoint = resolve_checkpoint(run["checkpoint"])
    if run["model_type"] == "contextual_vqhbr":
        model = create_model(
            "vqhbr",
            pretrained=True,
            pretrained_weight=str(checkpoint),
            as_tokenzer=True,
            n_code=args.codebook_size,
            code_dim=args.codebook_dim,
            quantize_kmeans_init=False,
        )
    elif run["model_type"] == "morphology_only":
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = MorphologyOnlyVQ(
            codebook_size=args.codebook_size,
            codebook_dim=args.codebook_dim,
            token_len=96,
            ema_decay=0.99,
            quantize_kmeans_init=False,
        )
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_discrete_features = ckpt.get("discrete_features")
        checkpoint_continuous_features = ckpt.get("continuous_features")
        if checkpoint_discrete_features and checkpoint_continuous_features:
            set_physiology_schema(checkpoint_discrete_features, checkpoint_continuous_features)
        continuous_mean = ckpt.get("continuous_mean")
        continuous_std = ckpt.get("continuous_std")
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        model = MorphologyV2PhysAwareVQ(
            codebook_size=args.codebook_size,
            codebook_dim=args.codebook_dim,
            token_len=int(ckpt_args.get("token_len", 96)),
            ema_decay=float(ckpt_args.get("ema_decay", 0.99)),
            quantize_kmeans_init=False,
            phys_head_source=str(ckpt_args.get("phys_head_source", "z_q")),
            continuous_mean=continuous_mean,
            continuous_std=continuous_std,
        )
        model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def valid_token_mask(data: np.ndarray, chans: np.ndarray, times: np.ndarray, lead_id: int = 0) -> np.ndarray:
    valid = (chans > 0) & (times > 0) & (np.abs(data).max(axis=-1) > 1e-8)
    if lead_id > 0:
        valid &= chans == lead_id
    return valid


@torch.no_grad()
def infer_batch(
    model, model_type: str, data: np.ndarray, chans: np.ndarray, times: np.ndarray,
    device: torch.device, lead_id: int = 0,
):
    x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).to(device, non_blocking=True)
    in_chans = torch.from_numpy(np.array(chans, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    in_times = torch.from_numpy(np.array(times, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    if lead_id > 0:
        selected = in_chans == lead_id
        in_chans = torch.where(selected, in_chans, torch.zeros_like(in_chans))
        in_times = torch.where(selected, in_times, torch.zeros_like(in_times))
    if model_type == "contextual_vqhbr":
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            valid = (in_chans > 0) & (in_times > 0) & (torch.abs(x).amax(dim=-1) > 1e-8)
            quantize, embed_ind, _loss = model.encode(
                x,
                in_chan_matrix=in_chans,
                in_time_matrix=in_times,
                valid_token_mask=valid,
            )
            rec = model.decode(quantize, in_chan_matrix=in_chans, in_time_matrix=in_times)
            ids = embed_ind.view(x.shape[0], -1)
    else:
        # Morphology-only reconstruction writes decoded valid tokens back into
        # a full fp32 tensor, so keep this path in fp32 during evaluation.
        rec, ids = model.reconstruct(x, in_chan_matrix=in_chans, in_time_matrix=in_times)
    return rec.detach().float().cpu().numpy(), ids.detach().cpu().numpy().astype(np.int64, copy=False)


def iter_batches(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


def zscore_one(wave: np.ndarray) -> np.ndarray:
    wave = wave.astype(np.float32, copy=False)
    return (wave - wave.mean()) / max(float(wave.std()), 1e-6)


def overlap_for_shift(a: np.ndarray, b: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    if shift == 0:
        return a, b
    if shift > 0:
        return a[shift:], b[:-shift]
    shift_abs = -shift
    return a[:-shift_abs], b[shift_abs:]


def corr_distance_pre_zscored(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def shift_invariant_corr_distance(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[float, int]:
    best_distance = math.inf
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        aa, bb = overlap_for_shift(a, b, shift)
        if len(aa) < 8:
            continue
        distance = corr_distance_pre_zscored(aa, bb)
        if distance < best_distance:
            best_distance = distance
            best_shift = shift
    return float(best_distance), int(best_shift)


def update_reservoir(reservoirs, seen_counts, sid: int, wave: np.ndarray, capacity: int, rng: random.Random):
    seen_counts[sid] += 1
    seen = int(seen_counts[sid])
    bucket = reservoirs[sid]
    wave_z = zscore_one(wave).astype(np.float32, copy=True)
    if len(bucket) < capacity:
        bucket.append(wave_z)
        return
    replace_idx = rng.randrange(seen)
    if replace_idx < capacity:
        bucket[replace_idx] = wave_z


def rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    corr = np.zeros(a.shape[0], dtype=np.float32)
    valid = denom > 1e-8
    corr[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return np.clip(corr, -1.0, 1.0)


def adjacent_shift_corr(waves: np.ndarray, max_shift: int) -> np.ndarray:
    if len(waves) < 2:
        return np.empty(0, dtype=np.float32)
    a = waves[:-1].astype(np.float32, copy=False)
    b = waves[1:].astype(np.float32, copy=False)
    best = np.full(a.shape[0], -np.inf, dtype=np.float32)
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            aa, bb = a, b
        elif shift > 0:
            aa, bb = a[:, shift:], b[:, :-shift]
        else:
            shift_abs = -shift
            aa, bb = a[:, :-shift_abs], b[:, shift_abs:]
        if aa.shape[1] < 8:
            continue
        best = np.maximum(best, rowwise_corr(aa, bb))
    best[~np.isfinite(best)] = 0.0
    return best


def transition_entropy(sids: np.ndarray) -> float:
    if len(sids) < 2:
        return math.nan
    by_source = defaultdict(Counter)
    for src, dst in zip(sids[:-1], sids[1:]):
        by_source[int(src)][int(dst)] += 1
    total = len(sids) - 1
    entropy = 0.0
    for counter in by_source.values():
        src_total = sum(counter.values())
        probs = np.asarray(list(counter.values()), dtype=np.float64) / src_total
        entropy += (src_total / total) * float(-(probs * np.log2(probs)).sum())
    return float(entropy)


def global_transition_entropy(global_transitions: dict[int, Counter]) -> float:
    total = sum(sum(counter.values()) for counter in global_transitions.values())
    if total == 0:
        return math.nan
    entropy = 0.0
    for counter in global_transitions.values():
        src_total = sum(counter.values())
        probs = np.asarray(list(counter.values()), dtype=np.float64) / src_total
        entropy += (src_total / total) * float(-(probs * np.log2(probs)).sum())
    return float(entropy)


def summarize_values(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "q1": math.nan, "q3": math.nan, "iqr": math.nan, "n": 0}
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "mean": float(arr.mean()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "n": int(arr.size),
    }


class RunStats:
    def __init__(self, name: str, codebook_size: int, reservoir_per_sid: int, seed: int):
        self.name = name
        self.codebook_size = codebook_size
        self.counts = np.zeros(codebook_size, dtype=np.int64)
        self.reservoirs = defaultdict(list)
        self.seen_counts = np.zeros(codebook_size, dtype=np.int64)
        self.rng = random.Random(seed)
        self.phys = SidAccumulator(codebook_size)
        self.rec_sse = 0.0
        self.rec_abs = 0.0
        self.rec_points = 0
        self.rec_corr_sum = 0.0
        self.rec_corr_n = 0
        self.temporal = defaultdict(list)
        self.temporal_totals = Counter()
        self.global_transitions = defaultdict(Counter)
        self.reservoir_per_sid = reservoir_per_sid

    def update_reconstruction(self, data: np.ndarray, rec: np.ndarray, valid: np.ndarray):
        y = data[valid].astype(np.float32, copy=False)
        yhat = rec[valid].astype(np.float32, copy=False)
        if y.size == 0:
            return
        diff = yhat - y
        self.rec_sse += float(np.sum(diff * diff))
        self.rec_abs += float(np.sum(np.abs(diff)))
        self.rec_points += int(diff.size)
        self.rec_corr_sum += float(np.sum(rowwise_corr(y, yhat)))
        self.rec_corr_n += int(y.shape[0])

    def update_usage_and_reservoir(self, data: np.ndarray, ids: np.ndarray, valid: np.ndarray):
        valid_ids = ids[valid]
        valid_ids = valid_ids[(valid_ids >= 0) & (valid_ids < self.codebook_size)]
        self.counts += np.bincount(valid_ids, minlength=self.codebook_size)[: self.codebook_size]
        positions = np.argwhere(valid)
        for row, token in positions:
            sid = int(ids[row, token])
            if 0 <= sid < self.codebook_size:
                update_reservoir(
                    self.reservoirs,
                    self.seen_counts,
                    sid,
                    data[row, token],
                    self.reservoir_per_sid,
                    self.rng,
                )

    def update_physiology(self, data: np.ndarray, ids: np.ndarray, chans: np.ndarray, times: np.ndarray, valid: np.ndarray, phys_args):
        rr = estimate_rr_for_batch(chans, times, phys_args)
        positions = np.argwhere(valid)
        for row, token in positions:
            sid = int(ids[row, token])
            if not (0 <= sid < self.codebook_size):
                continue
            features = extract_wave_features(data[row, token], phys_args)
            features["rr_interval_ms"] = float(rr[row, token])
            self.phys.add(sid, features)

    def update_temporal(self, data: np.ndarray, ids: np.ndarray, chans: np.ndarray, valid: np.ndarray, max_shift: int, corr_threshold: float):
        for row_idx in range(data.shape[0]):
            row_valid = valid[row_idx]
            if not row_valid.any():
                continue
            for lead in sorted(np.unique(chans[row_idx, row_valid]).astype(int).tolist()):
                token_indices = np.flatnonzero(row_valid & (chans[row_idx] == lead))
                if len(token_indices) < 2:
                    continue
                sids = ids[row_idx, token_indices].astype(np.int64, copy=False)
                sid_mask = (sids >= 0) & (sids < self.codebook_size)
                sids = sids[sid_mask]
                token_indices = token_indices[sid_mask]
                if len(sids) < 2:
                    continue
                waves = data[row_idx, token_indices].astype(np.float32, copy=False)
                adjacent_equal = sids[:-1] == sids[1:]
                adjacent_pairs = len(sids) - 1
                sid_counter = Counter(int(x) for x in sids)
                shift_corr = adjacent_shift_corr(waves, max_shift)
                high_corr = shift_corr >= corr_threshold
                high_corr_pairs = int(high_corr.sum())
                high_corr_sid_diff = int(((sids[:-1] != sids[1:]) & high_corr).sum())
                disagreement = float(high_corr_sid_diff / high_corr_pairs) if high_corr_pairs else math.nan

                self.temporal["adjacent_same_sid_rate"].append(float(adjacent_equal.mean()))
                self.temporal["modal_sid_ratio"].append(float(sid_counter.most_common(1)[0][1] / len(sids)))
                self.temporal["unique_sid_beat_ratio"].append(float(len(sid_counter) / len(sids)))
                self.temporal["transition_entropy_bits"].append(transition_entropy(sids))
                self.temporal["morphology_consistent_sid_disagreement"].append(disagreement)
                self.temporal["adjacent_shift_corr_mean"].append(float(np.mean(shift_corr)) if len(shift_corr) else math.nan)
                self.temporal["high_corr_pair_ratio"].append(float(high_corr_pairs / adjacent_pairs))
                self.temporal_totals["record_lead_units"] += 1
                self.temporal_totals["beats"] += int(len(sids))
                self.temporal_totals["adjacent_pairs"] += int(adjacent_pairs)
                self.temporal_totals["adjacent_same"] += int(adjacent_equal.sum())
                self.temporal_totals["high_corr_pairs"] += int(high_corr_pairs)
                self.temporal_totals["high_corr_sid_diff"] += int(high_corr_sid_diff)
                for src, dst in zip(sids[:-1], sids[1:]):
                    self.global_transitions[int(src)][int(dst)] += 1

    def reconstruction_summary(self) -> dict:
        mse = self.rec_sse / self.rec_points if self.rec_points else math.nan
        return {
            "reconstruction_mse": float(mse),
            "reconstruction_rmse": float(math.sqrt(mse)) if np.isfinite(mse) else math.nan,
            "reconstruction_mae": float(self.rec_abs / self.rec_points) if self.rec_points else math.nan,
            "reconstruction_corr": float(self.rec_corr_sum / self.rec_corr_n) if self.rec_corr_n else math.nan,
        }

    def usage_summary(self) -> dict:
        total = int(self.counts.sum())
        active = int((self.counts > 0).sum())
        probs = self.counts[self.counts > 0].astype(np.float64) / total if total else np.array([])
        entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
        sorted_counts = np.sort(self.counts)[::-1]

        def top_fraction(frac: float) -> float:
            if total == 0:
                return 0.0
            k = max(1, int(math.ceil(self.codebook_size * frac)))
            return float(sorted_counts[:k].sum() / total)

        return {
            "total_tokens": total,
            "active_sid": active,
            "dead_sid": int(self.codebook_size - active),
            "perplexity": float(np.exp(entropy)) if probs.size else 0.0,
            "top1pct_usage": top_fraction(0.01),
            "top5pct_usage": top_fraction(0.05),
            "top10pct_usage": top_fraction(0.10),
        }

    def distance_summary(self, args: argparse.Namespace, output_dir: Path) -> dict:
        rng = np.random.default_rng(args.seed)
        eligible = [
            int(sid)
            for sid in np.where(self.counts >= args.min_count_for_distance)[0]
            if len(self.reservoirs[int(sid)]) >= 2
        ]
        per_sid_rows = []
        all_intra = []
        for sid in eligible:
            waves = np.asarray(self.reservoirs[sid], dtype=np.float32)
            max_pairs = len(waves) * (len(waves) - 1) // 2
            pair_count = min(args.intra_pairs_per_sid, max_pairs)
            distances = []
            for _ in range(pair_count):
                i, j = rng.choice(len(waves), size=2, replace=False)
                distance, _shift = shift_invariant_corr_distance(waves[i], waves[j], args.max_shift)
                distances.append(distance)
            if distances:
                all_intra.extend(distances)
                per_sid_rows.append(
                    {
                        "sid": sid,
                        "count": int(self.counts[sid]),
                        "intra_shift_corr_dist_mean": float(np.mean(distances)),
                        "intra_shift_corr_dist_p90": float(np.percentile(distances, 90)),
                    }
                )
        pd.DataFrame(per_sid_rows).to_csv(output_dir / f"{self.name}_per_sid_shift_distance.csv", index=False)

        inter = []
        if len(eligible) >= 2:
            for _ in range(args.inter_pairs):
                sid_a, sid_b = rng.choice(eligible, size=2, replace=False)
                wa = self.reservoirs[int(sid_a)]
                wb = self.reservoirs[int(sid_b)]
                if not wa or not wb:
                    continue
                a = wa[int(rng.integers(0, len(wa)))]
                b = wb[int(rng.integers(0, len(wb)))]
                distance, _shift = shift_invariant_corr_distance(a, b, args.max_shift)
                inter.append(distance)

        intra_mean = float(np.mean(all_intra)) if all_intra else math.nan
        inter_mean = float(np.mean(inter)) if inter else math.nan
        return {
            "eligible_sids_distance": int(len(eligible)),
            "macro_intra_shift_corr_distance": intra_mean,
            "macro_intra_shift_corr_similarity": float(1.0 - intra_mean) if np.isfinite(intra_mean) else math.nan,
            "intra_shift_corr_distance_p90": float(np.percentile(all_intra, 90)) if all_intra else math.nan,
            "overall_inter_shift_corr_distance": inter_mean,
            "inter_over_intra_shift_corr": float(inter_mean / intra_mean) if intra_mean and np.isfinite(inter_mean) else math.nan,
        }

    def temporal_summary(self) -> dict:
        summary = {}
        for key, values in self.temporal.items():
            stats = summarize_values(values)
            for stat_name, value in stats.items():
                summary[f"{key}_{stat_name}"] = value
        pairs = self.temporal_totals["adjacent_pairs"]
        high_pairs = self.temporal_totals["high_corr_pairs"]
        summary.update(
            {
                "record_lead_units": int(self.temporal_totals["record_lead_units"]),
                "total_adjacent_pairs": int(pairs),
                "global_adjacent_same_sid_rate": float(self.temporal_totals["adjacent_same"] / pairs) if pairs else math.nan,
                "global_transition_entropy_bits": global_transition_entropy(self.global_transitions),
                "total_high_corr_pairs": int(high_pairs),
                "high_corr_pair_coverage": float(high_pairs / pairs) if pairs else math.nan,
                "global_morphology_consistent_sid_disagreement": float(self.temporal_totals["high_corr_sid_diff"] / high_pairs) if high_pairs else math.nan,
            }
        )
        return summary


def make_phys_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        fs=args.fs,
        min_sid_count=args.min_sid_count,
        split=args.split,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        record_length_samples=args.record_length_samples,
        time_block_samples=args.time_block_samples,
        qrs_onset_threshold_frac=args.qrs_onset_threshold_frac,
        qrs_wide_ms=args.qrs_wide_ms,
        st_threshold=args.st_threshold,
        t_relative_threshold=args.t_relative_threshold,
        t_abs_threshold=args.t_abs_threshold,
    )


def evaluate_one_run(run: dict, args: argparse.Namespace, data_path: Path, data, chans, times, device: torch.device) -> dict:
    model, checkpoint = load_model(run, args, device)
    num_samples = min(len(data), args.max_samples if args.max_samples > 0 else len(data))
    out_dir = args.output_dir / run["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments = np.lib.format.open_memmap(
        out_dir / f"{args.split}_sid_assignments.npy",
        mode="w+",
        dtype=np.int32,
        shape=(num_samples, data.shape[1]),
    )
    stats = RunStats(run["name"], args.codebook_size, args.reservoir_per_sid, args.seed)
    phys_args = make_phys_args(args)

    for start, end in iter_batches(num_samples, args.batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        valid = valid_token_mask(batch_data, batch_chans, batch_times, args.lead_id)
        rec, ids = infer_batch(
            model, run["model_type"], batch_data, batch_chans, batch_times, device, args.lead_id
        )
        assignments[start:end] = ids.astype(np.int32, copy=False)
        stats.update_reconstruction(batch_data, rec, valid)
        stats.update_usage_and_reservoir(batch_data, ids, valid)
        stats.update_physiology(batch_data, ids, batch_chans, batch_times, valid, phys_args)
        stats.update_temporal(batch_data, ids, batch_chans, valid, args.max_shift, args.corr_threshold)

        if start == 0 or end == num_samples or (start // args.batch_size) % 20 == 0:
            print(
                f"{run['name']}: evaluated {end}/{num_samples}; "
                f"valid={int(stats.counts.sum())}; active={(stats.counts > 0).sum()}",
                flush=True,
            )

    assignments.flush()
    np.save(out_dir / "codebook_vectors.npy", model.quantize.embedding.weight.detach().cpu().numpy())
    usage_df = pd.DataFrame({"sid": np.arange(args.codebook_size), "count": stats.counts})
    usage_df.to_csv(out_dir / "sid_usage_summary.csv", index=False)

    phys_metrics = summarize_sid_metrics(stats.phys, phys_args)
    phys_metrics.to_csv(out_dir / "sid_physiology_metrics.csv", index=False)
    eligible_phys = phys_metrics[
        (phys_metrics["count"] >= args.min_sid_count) & phys_metrics["physiology_purity_mean"].notna()
    ]
    eligible_phys.sort_values("physiology_purity_mean").head(20).to_csv(
        out_dir / "low_purity_sids.csv", index=False
    )
    phys_summary = vocabulary_summary(phys_metrics, phys_args, checkpoint, data_path)

    summary = {
        "run": run["name"],
        "model_type": run["model_type"],
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "split": args.split,
        "num_records": int(num_samples),
        "lead_id_filter": args.lead_id if args.lead_id > 0 else None,
        **stats.reconstruction_summary(),
        **stats.usage_summary(),
        **stats.distance_summary(args, out_dir),
        "physiology": phys_summary,
        **stats.temporal_summary(),
        "assignment_file": str(out_dir / f"{args.split}_sid_assignments.npy"),
        "codebook_vectors_file": str(out_dir / "codebook_vectors.npy"),
        "sid_usage_file": str(out_dir / "sid_usage_summary.csv"),
        "sid_physiology_file": str(out_dir / "sid_physiology_metrics.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def flatten_summary(summary: dict) -> dict:
    phys = summary["physiology"]
    row = {
        "run": summary["run"],
        "model_type": summary["model_type"],
        "reconstruction_rmse": summary["reconstruction_rmse"],
        "reconstruction_mae": summary["reconstruction_mae"],
        "reconstruction_corr": summary["reconstruction_corr"],
        "active_sid": summary["active_sid"],
        "perplexity": summary["perplexity"],
        "top1pct_usage": summary["top1pct_usage"],
        "top5pct_usage": summary["top5pct_usage"],
        "top10pct_usage": summary["top10pct_usage"],
        "macro_intra_shift_corr_similarity": summary["macro_intra_shift_corr_similarity"],
        "macro_intra_shift_corr_distance": summary["macro_intra_shift_corr_distance"],
        "inter_over_intra_shift_corr": summary["inter_over_intra_shift_corr"],
        "weighted_phys_purity": phys["weighted_average_physiology_purity"],
        "macro_phys_purity": phys["macro_average_physiology_purity"],
        "r_polarity_purity": phys["discrete_purity_weighted"]["r_polarity"],
        "t_polarity_purity": phys["discrete_purity_weighted"]["t_polarity"],
        "st_state_purity": phys["discrete_purity_weighted"]["st_state"],
        "qrs_type_purity": phys["discrete_purity_weighted"]["qrs_type"],
        "adjacent_same_sid_rate": summary["global_adjacent_same_sid_rate"],
        "modal_sid_ratio_mean": summary["modal_sid_ratio_mean"],
        "unique_sid_beat_ratio_mean": summary["unique_sid_beat_ratio_mean"],
        "global_transition_entropy_bits": summary["global_transition_entropy_bits"],
        "high_corr_sid_disagreement": summary["global_morphology_consistent_sid_disagreement"],
    }
    for feature in CONTINUOUS_FEATURES:
        row[f"{feature}_weighted_iqr"] = phys["continuous_weighted_iqr"][feature]
        row[f"{feature}_weighted_cv"] = phys["continuous_weighted_cv"][feature]
    return row


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    runs = [parse_run_spec(spec) for spec in args.run]
    device = choose_device(args.device)
    data_path, data, chans, times = load_arrays(args.data_path, args.split)

    summaries = []
    for run in runs:
        summaries.append(evaluate_one_run(run, args, data_path, data, chans, times, device))

    with (args.output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_path": str(data_path),
                "split": args.split,
                "codebook_size": args.codebook_size,
                "codebook_dim": args.codebook_dim,
                "max_shift": args.max_shift,
                "corr_threshold": args.corr_threshold,
                "lead_id_filter": args.lead_id if args.lead_id > 0 else None,
                "runs": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    table = pd.DataFrame([flatten_summary(summary) for summary in summaries])
    table.to_csv(args.output_dir / "comparison_table.csv", index=False)
    if len(table) == 2:
        v1 = table.iloc[0]
        v2 = table.iloc[1]
        delta_rows = []
        for metric in table.columns:
            if metric in {"run", "model_type"}:
                continue
            delta = v2[metric] - v1[metric] if pd.notna(v1[metric]) and pd.notna(v2[metric]) else math.nan
            delta_rows.append({"metric": metric, "V1": v1[metric], "V2": v2[metric], "delta": delta})
        pd.DataFrame(delta_rows).to_csv(args.output_dir / "comparison_delta_table.csv", index=False)
    print("\n=== Unified Comparison Table ===")
    print(table.to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
