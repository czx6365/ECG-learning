#!/usr/bin/env python3
"""Evaluate SID morphology similarity with local shift-invariant correlation distance."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shift-invariant SID morphology distance evaluation.")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means whole split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reservoir_per_sid", type=int, default=256)
    parser.add_argument("--min_count_for_distance", type=int, default=50)
    parser.add_argument("--intra_pairs_per_sid", type=int, default=256)
    parser.add_argument("--inter_pairs", type=int, default=200000)
    parser.add_argument("--max_shift", type=int, default=6)
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


def zscore_one(wave: np.ndarray) -> np.ndarray:
    wave = wave.astype(np.float32, copy=False)
    return (wave - wave.mean()) / max(float(wave.std()), 1e-6)


def corr_distance_pre_zscored(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def overlap_for_shift(a: np.ndarray, b: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping segments after shifting b by shift samples."""
    if shift == 0:
        return a, b
    if shift > 0:
        return a[shift:], b[:-shift]
    shift_abs = -shift
    return a[:-shift_abs], b[shift_abs:]


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


def update_reservoir(
    reservoirs: dict[int, list[np.ndarray]],
    seen_counts: np.ndarray,
    code_id: int,
    wave: np.ndarray,
    capacity: int,
) -> None:
    seen_counts[code_id] += 1
    seen = int(seen_counts[code_id])
    bucket = reservoirs[code_id]
    if len(bucket) < capacity:
        bucket.append(zscore_one(wave).astype(np.float32, copy=True))
        return
    replace_idx = random.randrange(seen)
    if replace_idx < capacity:
        bucket[replace_idx] = zscore_one(wave).astype(np.float32, copy=True)


def iter_batches(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


def collect_sid_samples(args, model, data, chans, times, device):
    num_samples = min(len(data), args.max_samples if args.max_samples > 0 else len(data))
    counts = np.zeros(args.codebook_size, dtype=np.int64)
    reservoirs: dict[int, list[np.ndarray]] = defaultdict(list)

    for start, end in iter_batches(num_samples, args.batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        batch_codes = encode_batch(model, batch_data, batch_chans, batch_times, device)
        valid = (batch_chans > 0) & (batch_times > 0) & ~(np.isclose(batch_data, 0.0).all(axis=-1))
        positions = np.argwhere(valid)
        for local_idx, token_idx in positions:
            code_id = int(batch_codes[local_idx, token_idx])
            if 0 <= code_id < args.codebook_size:
                update_reservoir(
                    reservoirs,
                    counts,
                    code_id,
                    batch_data[local_idx, token_idx],
                    args.reservoir_per_sid,
                )
        if start == 0 or end == num_samples or (start // args.batch_size) % 20 == 0:
            print(f"Encoded {end}/{num_samples} samples; valid tokens={int(counts.sum())}", flush=True)

    return num_samples, counts, reservoirs


def summarize_usage(counts: np.ndarray) -> dict:
    total = int(counts.sum())
    active = int((counts > 0).sum())
    probs = counts[counts > 0].astype(np.float64) / total if total else np.array([])
    entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
    sorted_counts = np.sort(counts)[::-1]
    return {
        "total_tokens": total,
        "active_sids": active,
        "dead_sids": int(len(counts) - active),
        "active_ratio": float(active / len(counts)),
        "perplexity": float(np.exp(entropy)) if probs.size else 0.0,
        "top1_frequency": float(sorted_counts[0] / total) if total else 0.0,
        "top5_frequency": float(sorted_counts[:5].sum() / total) if total else 0.0,
        "top1pct_frequency": float(sorted_counts[: max(1, math.ceil(len(counts) * 0.01))].sum() / total)
        if total
        else 0.0,
    }


def compute_shift_invariant_distances(
    counts: np.ndarray,
    reservoirs: dict[int, list[np.ndarray]],
    args,
    output_dir: Path,
) -> dict:
    rng = np.random.default_rng(args.seed)
    eligible = [
        int(code_id)
        for code_id in np.where(counts >= args.min_count_for_distance)[0]
        if len(reservoirs[int(code_id)]) >= 2
    ]

    per_sid_rows = []
    all_intra = []
    all_intra_abs_shift = []
    for code_id in eligible:
        waves = np.asarray(reservoirs[code_id], dtype=np.float32)
        pair_count = min(args.intra_pairs_per_sid, len(waves) * (len(waves) - 1) // 2)
        distances = []
        shifts = []
        for _ in range(pair_count):
            i, j = rng.choice(len(waves), size=2, replace=False)
            distance, shift = shift_invariant_corr_distance(waves[i], waves[j], args.max_shift)
            distances.append(distance)
            shifts.append(shift)
        if not distances:
            continue
        all_intra.extend(distances)
        all_intra_abs_shift.extend(abs(s) for s in shifts)
        per_sid_rows.append(
            {
                "sid": code_id,
                "count": int(counts[code_id]),
                "sampled": int(len(waves)),
                "intra_pairs": int(pair_count),
                "intra_shift_corr_dist_mean": float(np.mean(distances)),
                "intra_shift_corr_dist_p50": float(np.percentile(distances, 50)),
                "intra_shift_corr_dist_p90": float(np.percentile(distances, 90)),
                "best_abs_shift_mean": float(np.mean(np.abs(shifts))),
                "best_abs_shift_p50": float(np.percentile(np.abs(shifts), 50)),
                "best_abs_shift_p90": float(np.percentile(np.abs(shifts), 90)),
            }
        )

    per_sid = pd.DataFrame(per_sid_rows).sort_values(
        ["intra_shift_corr_dist_mean", "count"], ascending=[True, False]
    )
    per_sid.to_csv(output_dir / "per_sid_intra_shift_corr.csv", index=False)

    arrays = []
    labels = []
    for code_id in eligible:
        waves = np.asarray(reservoirs[code_id], dtype=np.float32)
        arrays.append(waves)
        labels.extend([code_id] * len(waves))
    if arrays:
        X = np.vstack(arrays)
        y = np.asarray(labels)
    else:
        X = np.empty((0, 0), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)

    inter = []
    inter_abs_shift = []
    if len(X) >= 2 and len(np.unique(y)) >= 2:
        for _ in range(args.inter_pairs):
            for _attempt in range(100):
                i, j = rng.choice(len(X), size=2, replace=False)
                if y[i] != y[j]:
                    break
            if y[i] == y[j]:
                continue
            distance, shift = shift_invariant_corr_distance(X[i], X[j], args.max_shift)
            inter.append(distance)
            inter_abs_shift.append(abs(shift))

    summary = {
        "distance_mode": "local_shift_invariant_correlation",
        "max_shift": int(args.max_shift),
        "eligible_sids": int(len(eligible)),
        "intra_pairs": int(len(all_intra)),
        "inter_pairs": int(len(inter)),
        "intra_shift_corr_dist_mean": float(np.mean(all_intra)) if all_intra else None,
        "intra_shift_corr_dist_p50": float(np.percentile(all_intra, 50)) if all_intra else None,
        "intra_shift_corr_dist_p90": float(np.percentile(all_intra, 90)) if all_intra else None,
        "inter_shift_corr_dist_mean": float(np.mean(inter)) if inter else None,
        "inter_shift_corr_dist_p50": float(np.percentile(inter, 50)) if inter else None,
        "inter_shift_corr_dist_p90": float(np.percentile(inter, 90)) if inter else None,
        "inter_over_intra_shift_corr_ratio": float(np.mean(inter) / np.mean(all_intra))
        if inter and all_intra
        else None,
        "intra_best_abs_shift_mean": float(np.mean(all_intra_abs_shift)) if all_intra_abs_shift else None,
        "inter_best_abs_shift_mean": float(np.mean(inter_abs_shift)) if inter_abs_shift else None,
    }
    with open(output_dir / "shift_invariant_distance_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model, resolved_checkpoint = load_tokenizer(args.checkpoint_path, args.codebook_size, args.codebook_dim, device)
    data, chans, times = load_arrays(args.data_path, args.split)
    num_samples, counts, reservoirs = collect_sid_samples(args, model, data, chans, times, device)

    usage = summarize_usage(counts)
    usage.update(
        {
            "checkpoint": str(resolved_checkpoint),
            "data_path": str(args.data_path),
            "split": args.split,
            "evaluated_samples": int(num_samples),
            "codebook_size": int(args.codebook_size),
            "codebook_dim": int(args.codebook_dim),
            "reservoir_per_sid": int(args.reservoir_per_sid),
        }
    )
    with open(args.output_dir / "usage_summary.json", "w", encoding="utf-8") as handle:
        json.dump(usage, handle, indent=2, ensure_ascii=False)

    sid_summary = pd.DataFrame(
        {
            "sid": np.arange(args.codebook_size, dtype=np.int64),
            "count": counts,
            "frequency": counts / max(1, counts.sum()),
            "sampled": [len(reservoirs.get(int(idx), [])) for idx in range(args.codebook_size)],
        }
    )
    sid_summary.sort_values("count", ascending=False).to_csv(args.output_dir / "sid_usage_summary.csv", index=False)

    distance = compute_shift_invariant_distances(counts, reservoirs, args, args.output_dir)
    final = {"usage": usage, "distance": distance}
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, ensure_ascii=False)
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
