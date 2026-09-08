#!/usr/bin/env python3
"""Evaluate whether VQ-HBR SID/codebook ids represent similar ECG segments."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from timm.models import create_model


ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models import modeling_vqhbr as modeling_vqhbr  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SID visualization and intra/inter distance evaluation.")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means the whole split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reservoir_per_sid", type=int, default=160)
    parser.add_argument("--overlay_per_sid", type=int, default=50)
    parser.add_argument("--plot_top", type=int, default=12)
    parser.add_argument("--plot_mid", type=int, default=12)
    parser.add_argument("--plot_low", type=int, default=12)
    parser.add_argument("--min_count_for_distance", type=int, default=20)
    parser.add_argument("--intra_pairs_per_sid", type=int, default=300)
    parser.add_argument("--inter_pairs", type=int, default=20000)
    parser.add_argument("--silhouette_samples", type=int, default=5000)
    parser.add_argument("--max_flat_run", type=int, default=0, help="Keep tokens with max flat run < this value. 0 disables.")
    parser.add_argument("--drop_prefix_flat12", action="store_true")
    parser.add_argument("--drop_suffix_flat12", action="store_true")
    parser.add_argument("--peak_idx_min", type=int, default=-1)
    parser.add_argument("--peak_idx_max", type=int, default=-1)
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
        print("CUDA requested but unavailable; falling back to CPU.")
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


def zscore(waves: np.ndarray) -> np.ndarray:
    waves = waves.astype(np.float32, copy=False)
    mean = waves.mean(axis=1, keepdims=True)
    std = waves.std(axis=1, keepdims=True)
    return (waves - mean) / np.maximum(std, 1e-6)


def shift_with_edge(wave: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return wave.copy()
    out = np.empty_like(wave)
    if shift > 0:
        out[:shift] = wave[0]
        out[shift:] = wave[:-shift]
    else:
        shift_abs = -shift
        out[-shift_abs:] = wave[-1]
        out[:-shift_abs] = wave[shift_abs:]
    return out


def align_r_peak(waves: np.ndarray) -> np.ndarray:
    if waves.size == 0:
        return waves
    center = waves.shape[1] // 2
    peak_indices = np.argmax(np.abs(waves), axis=1)
    aligned = np.empty_like(waves)
    for idx, wave in enumerate(waves):
        aligned[idx] = shift_with_edge(wave, center - int(peak_indices[idx]))
    return aligned


def rpeak_aligned_zscore_waves(waves: np.ndarray) -> np.ndarray:
    return align_r_peak(zscore(waves))


def raw_zscore_waves(waves: np.ndarray) -> np.ndarray:
    return zscore(waves)


def transform_waves_for_distance(waves: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw_zscore":
        return raw_zscore_waves(waves)
    if mode == "rpeak_aligned_zscore":
        return rpeak_aligned_zscore_waves(waves)
    raise ValueError(f"Unknown distance mode: {mode}")


def max_flat_run(wave: np.ndarray, eps: float = 1e-4) -> int:
    diffs = np.abs(np.diff(wave))
    best = 1
    cur = 1
    for diff in diffs:
        if diff <= eps:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def is_flat_edge(wave: np.ndarray, prefix: bool, n: int = 12, eps: float = 1e-4) -> bool:
    edge = wave[:n] if prefix else wave[-n:]
    return bool(np.max(np.abs(np.diff(edge))) <= eps)


def passes_artifact_filter(wave: np.ndarray, args) -> bool:
    if args.max_flat_run > 0 and max_flat_run(wave) >= args.max_flat_run:
        return False
    if args.drop_prefix_flat12 and is_flat_edge(wave, prefix=True):
        return False
    if args.drop_suffix_flat12 and is_flat_edge(wave, prefix=False):
        return False
    if args.peak_idx_min >= 0 or args.peak_idx_max >= 0:
        peak_idx = int(np.argmax(np.abs(wave)))
        if args.peak_idx_min >= 0 and peak_idx < args.peak_idx_min:
            return False
        if args.peak_idx_max >= 0 and peak_idx > args.peak_idx_max:
            return False
    return True


def corr_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def rmse_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def update_reservoir(
    reservoirs: dict[int, list[np.ndarray]],
    examples: dict[int, list[dict]],
    seen_counts: np.ndarray,
    code_id: int,
    wave: np.ndarray,
    meta: dict,
    capacity: int,
) -> None:
    seen_counts[code_id] += 1
    seen = int(seen_counts[code_id])
    bucket = reservoirs[code_id]
    meta_bucket = examples[code_id]
    if len(bucket) < capacity:
        bucket.append(wave.astype(np.float32, copy=True))
        meta_bucket.append(meta)
        return
    replace_idx = random.randrange(seen)
    if replace_idx < capacity:
        bucket[replace_idx] = wave.astype(np.float32, copy=True)
        meta_bucket[replace_idx] = meta


def iter_batches(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


def collect_sid_samples(args, model, data, chans, times, device):
    num_samples = min(len(data), args.max_samples if args.max_samples > 0 else len(data))
    counts = np.zeros(args.codebook_size, dtype=np.int64)
    reservoirs: dict[int, list[np.ndarray]] = defaultdict(list)
    examples: dict[int, list[dict]] = defaultdict(list)
    filter_stats = {"valid_before_filter": 0, "kept_after_filter": 0}

    for start, end in iter_batches(num_samples, args.batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        batch_codes = encode_batch(model, batch_data, batch_chans, batch_times, device)
        valid = (batch_chans > 0) & (batch_times > 0)
        positions = np.argwhere(valid)
        for local_idx, token_idx in positions:
            filter_stats["valid_before_filter"] += 1
            wave = batch_data[local_idx, token_idx]
            if not passes_artifact_filter(wave, args):
                continue
            filter_stats["kept_after_filter"] += 1
            code_id = int(batch_codes[local_idx, token_idx])
            if not 0 <= code_id < args.codebook_size:
                continue
            update_reservoir(
                reservoirs,
                examples,
                counts,
                code_id,
                wave,
                {
                    "sample_id": int(start + local_idx),
                    "token_index": int(token_idx),
                    "lead_id": int(batch_chans[local_idx, token_idx]),
                    "time_id": int(batch_times[local_idx, token_idx]),
                },
                args.reservoir_per_sid,
            )
        if start == 0 or end == num_samples or (start // args.batch_size) % 20 == 0:
            print(f"Encoded {end}/{num_samples} samples; valid tokens={int(counts.sum())}", flush=True)

    return num_samples, counts, reservoirs, examples, filter_stats


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


def select_plot_sids(counts: np.ndarray, args) -> pd.DataFrame:
    active_ids = np.where(counts > 0)[0]
    ranked = active_ids[np.argsort(counts[active_ids])[::-1]]
    rows = []
    for code_id in ranked[: args.plot_top]:
        rows.append({"sid": int(code_id), "group": "high", "count": int(counts[code_id])})

    if len(ranked) > 0 and args.plot_mid > 0:
        mid_start = max(0, int(len(ranked) * 0.45) - args.plot_mid // 2)
        mid_ids = ranked[mid_start : mid_start + args.plot_mid]
        for code_id in mid_ids:
            rows.append({"sid": int(code_id), "group": "mid", "count": int(counts[code_id])})

    low_pool = ranked[::-1]
    low_added = 0
    for code_id in low_pool:
        if counts[code_id] < 2:
            continue
        rows.append({"sid": int(code_id), "group": "low", "count": int(counts[code_id])})
        low_added += 1
        if low_added >= args.plot_low:
            break

    selected = pd.DataFrame(rows).drop_duplicates("sid")
    return selected.sort_values(["group", "count"], ascending=[True, False]).reset_index(drop=True)


def plot_usage(counts: np.ndarray, path: Path) -> None:
    sorted_counts = np.sort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(sorted_counts)), sorted_counts, width=1.0, color="#3b6ea8")
    ax.set_yscale("log")
    ax.set_xlabel("SID rank by frequency")
    ax.set_ylabel("Token count, log scale")
    ax.set_title("SID usage distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sid_page(code_id: int, group: str, count: int, waves: np.ndarray, examples: list[dict], output_dir: Path, overlay_n: int) -> None:
    if len(waves) == 0:
        return
    rng = np.random.default_rng(0)
    if len(waves) > overlay_n:
        idx = rng.choice(len(waves), size=overlay_n, replace=False)
        plot_waves = waves[idx]
    else:
        plot_waves = waves
    aligned = rpeak_aligned_zscore_waves(plot_waves)
    all_aligned = rpeak_aligned_zscore_waves(waves)
    mean_wave = all_aligned.mean(axis=0)
    q10 = np.percentile(all_aligned, 10, axis=0)
    q90 = np.percentile(all_aligned, 90, axis=0)
    dists = np.sqrt(np.mean((all_aligned - mean_wave[None, :]) ** 2, axis=1))
    order = np.argsort(dists)
    typical = all_aligned[order[: min(3, len(order))]]
    atypical = all_aligned[order[-min(3, len(order)) :]]
    x = np.arange(waves.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    for wave in plot_waves:
        ax.plot(x, wave, color="#376795", alpha=0.18, linewidth=0.9)
    ax.plot(x, plot_waves.mean(axis=0), color="#d1495b", linewidth=2)
    ax.set_title("Raw overlays")

    ax = axes[0, 1]
    for wave in aligned:
        ax.plot(x, wave, color="#4f772d", alpha=0.18, linewidth=0.9)
    ax.plot(x, mean_wave, color="#d1495b", linewidth=2)
    ax.axvline(waves.shape[1] // 2, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Z-score + R-peak aligned overlays")

    ax = axes[1, 0]
    ax.fill_between(x, q10, q90, color="#80b1d3", alpha=0.35, label="10-90%")
    ax.plot(x, mean_wave, color="#1f4e79", linewidth=2, label="mean")
    ax.set_title("Mean waveform with 10-90% band")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    for wave in typical:
        ax.plot(x, wave, color="#2a9d8f", alpha=0.85, linewidth=1.5)
    for wave in atypical:
        ax.plot(x, wave, color="#e76f51", alpha=0.85, linewidth=1.5)
    ax.plot(x, mean_wave, color="black", linewidth=2, alpha=0.7)
    ax.set_title("Typical green vs atypical red")

    for ax in axes.ravel():
        ax.set_xlabel("time index")
        ax.grid(alpha=0.18)
    fig.suptitle(f"SID {code_id} | {group} frequency | total count={count} | sampled={len(waves)}", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / f"sid_{code_id:04d}_{group}_count_{count}.png", dpi=180)
    plt.close(fig)

    meta = pd.DataFrame(examples)
    if not meta.empty:
        meta.to_csv(output_dir / f"sid_{code_id:04d}_{group}_examples.csv", index=False)


def pair_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return rmse_distance(a, b), corr_distance(a, b)


def compute_distances_for_mode(
    counts: np.ndarray,
    reservoirs: dict[int, list[np.ndarray]],
    args,
    output_dir: Path,
    mode: str,
) -> dict:
    rng = np.random.default_rng(args.seed)
    per_sid_rows = []
    all_intra_rmse = []
    all_intra_corr = []

    eligible = [int(code_id) for code_id in np.where(counts >= args.min_count_for_distance)[0] if len(reservoirs[int(code_id)]) >= 2]
    for code_id in eligible:
        waves = transform_waves_for_distance(np.asarray(reservoirs[code_id], dtype=np.float32), mode)
        pair_count = min(args.intra_pairs_per_sid, len(waves) * (len(waves) - 1) // 2)
        rmse_vals = []
        corr_vals = []
        for _ in range(pair_count):
            i, j = rng.choice(len(waves), size=2, replace=False)
            rmse, corr = pair_metrics(waves[i], waves[j])
            rmse_vals.append(rmse)
            corr_vals.append(corr)
        if not rmse_vals:
            continue
        all_intra_rmse.extend(rmse_vals)
        all_intra_corr.extend(corr_vals)
        per_sid_rows.append(
            {
                "sid": code_id,
                "count": int(counts[code_id]),
                "sampled": int(len(waves)),
                "intra_pairs": int(pair_count),
                "intra_rmse_mean": float(np.mean(rmse_vals)),
                "intra_rmse_p50": float(np.percentile(rmse_vals, 50)),
                "intra_rmse_p90": float(np.percentile(rmse_vals, 90)),
                "intra_corr_dist_mean": float(np.mean(corr_vals)),
                "intra_corr_dist_p50": float(np.percentile(corr_vals, 50)),
                "intra_corr_dist_p90": float(np.percentile(corr_vals, 90)),
            }
        )

    per_sid = pd.DataFrame(per_sid_rows).sort_values(["intra_corr_dist_mean", "count"], ascending=[True, False])
    per_sid.to_csv(output_dir / f"sid_intra_distance_by_sid_{mode}.csv", index=False)
    if mode == "rpeak_aligned_zscore":
        per_sid.to_csv(output_dir / "sid_intra_distance_by_sid.csv", index=False)

    arrays = []
    labels = []
    for code_id in eligible:
        waves = transform_waves_for_distance(np.asarray(reservoirs[code_id], dtype=np.float32), mode)
        arrays.append(waves)
        labels.extend([code_id] * len(waves))
    if arrays:
        X = np.vstack(arrays)
        y = np.asarray(labels)
    else:
        X = np.empty((0, 0), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)

    inter_rmse = []
    inter_corr = []
    if len(X) >= 2 and len(np.unique(y)) >= 2:
        for _ in range(args.inter_pairs):
            for _attempt in range(100):
                i, j = rng.choice(len(X), size=2, replace=False)
                if y[i] != y[j]:
                    break
            if y[i] == y[j]:
                continue
            rmse, corr = pair_metrics(X[i], X[j])
            inter_rmse.append(rmse)
            inter_corr.append(corr)

    silhouette = None
    if len(X) >= 10 and len(np.unique(y)) >= 2 and args.silhouette_samples > 0:
        sample_n = min(args.silhouette_samples, len(X))
        sample_idx = rng.choice(len(X), size=sample_n, replace=False)
        unique_labels = np.unique(y[sample_idx])
        if len(unique_labels) >= 2:
            silhouette = float(silhouette_score(X[sample_idx], y[sample_idx], metric="euclidean"))

    summary = {
        "distance_mode": mode,
        "eligible_sids": int(len(eligible)),
        "intra_pairs": int(len(all_intra_rmse)),
        "inter_pairs": int(len(inter_rmse)),
        "intra_rmse_mean": float(np.mean(all_intra_rmse)) if all_intra_rmse else None,
        "inter_rmse_mean": float(np.mean(inter_rmse)) if inter_rmse else None,
        "rmse_separation_ratio_inter_over_intra": float(np.mean(inter_rmse) / np.mean(all_intra_rmse))
        if inter_rmse and all_intra_rmse
        else None,
        "intra_corr_dist_mean": float(np.mean(all_intra_corr)) if all_intra_corr else None,
        "inter_corr_dist_mean": float(np.mean(inter_corr)) if inter_corr else None,
        "corr_dist_separation_ratio_inter_over_intra": float(np.mean(inter_corr) / np.mean(all_intra_corr))
        if inter_corr and all_intra_corr
        else None,
        "silhouette_euclidean": silhouette,
    }

    with open(output_dir / f"distance_summary_{mode}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    if mode == "rpeak_aligned_zscore":
        with open(output_dir / "distance_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    if per_sid_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(per_sid["count"], per_sid["intra_corr_dist_mean"], s=12, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("SID count, log scale")
        ax.set_ylabel("Mean intra-SID correlation distance")
        ax.set_title("Per-SID compactness")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"sid_intra_corr_distance_vs_count_{mode}.png", dpi=180)
        if mode == "rpeak_aligned_zscore":
            fig.savefig(output_dir / "sid_intra_corr_distance_vs_count.png", dpi=180)
        plt.close(fig)

    return summary


def compute_distances(counts: np.ndarray, reservoirs: dict[int, list[np.ndarray]], args, output_dir: Path) -> dict:
    raw_summary = compute_distances_for_mode(counts, reservoirs, args, output_dir, "raw_zscore")
    aligned_summary = compute_distances_for_mode(counts, reservoirs, args, output_dir, "rpeak_aligned_zscore")
    comparison = {
        "raw_zscore": raw_summary,
        "rpeak_aligned_zscore": aligned_summary,
        "note": "distance_summary.json keeps rpeak_aligned_zscore for backward compatibility.",
    }
    with open(output_dir / "distance_comparison_raw_vs_rpeak_aligned.json", "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)
    return aligned_summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "sid_visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model, resolved_checkpoint = load_tokenizer(args.checkpoint_path, args.codebook_size, args.codebook_dim, device)
    data, chans, times = load_arrays(args.data_path, args.split)

    num_samples, counts, reservoirs, examples, filter_stats = collect_sid_samples(args, model, data, chans, times, device)

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
            "filter": {
                **filter_stats,
                "kept_fraction": float(filter_stats["kept_after_filter"] / filter_stats["valid_before_filter"])
                if filter_stats["valid_before_filter"]
                else 0.0,
                "max_flat_run": int(args.max_flat_run),
                "drop_prefix_flat12": bool(args.drop_prefix_flat12),
                "drop_suffix_flat12": bool(args.drop_suffix_flat12),
                "peak_idx_min": int(args.peak_idx_min),
                "peak_idx_max": int(args.peak_idx_max),
            },
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
    sid_summary.sort_values("count", ascending=False).to_csv(args.output_dir / "sid_summary.csv", index=False)
    plot_usage(counts, args.output_dir / "sid_usage_rank.png")

    selected = select_plot_sids(counts, args)
    selected.to_csv(args.output_dir / "selected_visualized_sids.csv", index=False)
    for row in selected.to_dict(orient="records"):
        code_id = int(row["sid"])
        waves = np.asarray(reservoirs[code_id], dtype=np.float32)
        plot_sid_page(
            code_id,
            str(row["group"]),
            int(row["count"]),
            waves,
            examples[code_id],
            plot_dir,
            args.overlay_per_sid,
        )
    print(f"Wrote {len(selected)} SID visualization pages to {plot_dir}", flush=True)

    distance_summary = compute_distances(counts, reservoirs, args, args.output_dir)
    final_summary = {"usage": usage, "distance": distance_summary}
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(final_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
