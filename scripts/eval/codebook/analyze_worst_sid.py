#!/usr/bin/env python3
"""Inspect high intra-distance SID groups for preprocessing artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from timm.models import create_model


ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_sid_quality import align_r_peak, plot_sid_page, zscore  # noqa: E402
from heartlang.models import modeling_vqhbr as modeling_vqhbr  # noqa: F401,E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze worst SID groups.")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--sid_distance_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--worst_n", type=int, default=20)
    parser.add_argument("--select", choices=("worst", "best"), default="worst")
    parser.add_argument("--max_store_per_sid", type=int, default=500)
    parser.add_argument("--overlay_per_sid", type=int, default=80)
    return parser.parse_args()


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def load_tokenizer(args, device):
    patch_checkpoint_paths()
    model = create_model(
        "vqhbr",
        pretrained=True,
        pretrained_weight=str(args.checkpoint_path),
        as_tokenzer=True,
        n_code=args.codebook_size,
        code_dim=args.codebook_dim,
        quantize_kmeans_init=False,
    )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_batch(model, data, chans, times, device):
    x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).to(device)
    in_chans = torch.from_numpy(np.array(chans, dtype=np.int64, copy=True)).to(device)
    in_times = torch.from_numpy(np.array(times, dtype=np.int64, copy=True)).to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        out = model.get_codebook_indices(x, in_chan_matrix=in_chans, in_time_matrix=in_times)
    return out.detach().cpu().numpy().astype(np.int64, copy=False)


def max_flat_run(wave: np.ndarray, eps: float = 1e-4) -> int:
    diffs = np.abs(np.diff(wave))
    best = 1
    cur = 1
    for d in diffs:
        if d <= eps:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def flat_edge(wave: np.ndarray, n: int = 12, eps: float = 1e-4) -> tuple[bool, bool]:
    return (
        bool(np.max(np.abs(np.diff(wave[:n]))) <= eps),
        bool(np.max(np.abs(np.diff(wave[-n:]))) <= eps),
    )


def wave_artifact_features(wave: np.ndarray) -> dict:
    prefix_flat, suffix_flat = flat_edge(wave)
    peak_idx = int(np.argmax(np.abs(wave)))
    return {
        "mean": float(np.mean(wave)),
        "std": float(np.std(wave)),
        "min": float(np.min(wave)),
        "max": float(np.max(wave)),
        "range": float(np.max(wave) - np.min(wave)),
        "peak_idx": peak_idx,
        "peak_abs_amp": float(np.max(np.abs(wave))),
        "max_flat_run": int(max_flat_run(wave)),
        "prefix_flat12": prefix_flat,
        "suffix_flat12": suffix_flat,
        "near_zero_fraction": float(np.mean(np.abs(wave) < 1e-6)),
        "first12_std": float(np.std(wave[:12])),
        "last12_std": float(np.std(wave[-12:])),
    }


def iter_batches(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def plot_peak_time_hist(code_id: int, rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].hist(df["peak_idx"], bins=np.arange(98) - 0.5, color="#4c78a8")
    axes[0].set_title("R-peak index")
    axes[0].set_xlabel("index")
    axes[1].hist(df["time_id"], bins=40, color="#59a14f")
    axes[1].set_title("time_id")
    axes[1].set_xlabel("time_id")
    axes[2].hist(df["max_flat_run"], bins=40, color="#e15759")
    axes[2].set_title("max flat run")
    axes[2].set_xlabel("samples")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"SID {code_id} artifact diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / f"sid_{code_id:04d}_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "worst_sid_visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)

    distance = pd.read_csv(args.sid_distance_csv)
    selected = (
        distance.sort_values("intra_corr_dist_mean", ascending=args.select == "best")
        .head(args.worst_n)["sid"]
        .astype(int)
        .tolist()
    )
    target = set(selected)
    print(f"{args.select.title()} SIDs:", selected, flush=True)

    data_path = args.data_path.expanduser().resolve()
    data = np.load(data_path / f"{args.split}_data.npy", mmap_mode="r")
    chans = np.load(data_path / f"{args.split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_path / f"{args.split}_data_in_times.npy", mmap_mode="r")

    device = choose_device(args.device)
    model = load_tokenizer(args, device)

    counts = Counter()
    lead_counts = defaultdict(Counter)
    time_counts = defaultdict(Counter)
    rows_by_sid = defaultdict(list)
    waves_by_sid = defaultdict(list)

    for start, end in iter_batches(len(data), args.batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        batch_codes = encode_batch(model, batch_data, batch_chans, batch_times, device)
        valid = (batch_chans > 0) & (batch_times > 0)
        positions = np.argwhere(valid)
        for local_idx, token_idx in positions:
            sid = int(batch_codes[local_idx, token_idx])
            if sid not in target:
                continue
            wave = batch_data[local_idx, token_idx].astype(np.float32, copy=True)
            features = wave_artifact_features(wave)
            row = {
                "sid": sid,
                "sample_id": int(start + local_idx),
                "token_index": int(token_idx),
                "lead_id": int(batch_chans[local_idx, token_idx]),
                "time_id": int(batch_times[local_idx, token_idx]),
                **features,
            }
            counts[sid] += 1
            lead_counts[sid][row["lead_id"]] += 1
            time_counts[sid][row["time_id"]] += 1
            if len(waves_by_sid[sid]) < args.max_store_per_sid:
                waves_by_sid[sid].append(wave)
                rows_by_sid[sid].append(row)
        if start == 0 or end == len(data) or (start // args.batch_size) % 20 == 0:
            print(f"Scanned {end}/{len(data)} samples", flush=True)

    summary_rows = []
    for sid in selected:
        rows = rows_by_sid[sid]
        all_rows_df = pd.DataFrame(rows)
        if all_rows_df.empty:
            continue
        waves = np.asarray(waves_by_sid[sid], dtype=np.float32)
        norm = align_r_peak(zscore(waves))
        mean_wave = norm.mean(axis=0)
        dist_to_mean = np.sqrt(np.mean((norm - mean_wave[None, :]) ** 2, axis=1))
        lead_total = sum(lead_counts[sid].values())
        lead_top = lead_counts[sid].most_common(5)
        time_top = time_counts[sid].most_common(5)
        plot_sid_page(sid, "worst", int(counts[sid]), waves, rows, plot_dir, args.overlay_per_sid)
        plot_peak_time_hist(sid, rows, plot_dir)
        all_rows_df.to_csv(plot_dir / f"sid_{sid:04d}_stored_examples_features.csv", index=False)
        summary_rows.append(
            {
                "sid": sid,
                "total_count": int(counts[sid]),
                "stored": int(len(rows)),
                "lead_top5": json.dumps(lead_top, ensure_ascii=False),
                "lead_top1_fraction": float(lead_top[0][1] / lead_total) if lead_total else np.nan,
                "time_top5": json.dumps(time_top, ensure_ascii=False),
                "stored_mean_std": float(all_rows_df["std"].mean()),
                "stored_median_range": float(all_rows_df["range"].median()),
                "stored_prefix_flat12_fraction": float(all_rows_df["prefix_flat12"].mean()),
                "stored_suffix_flat12_fraction": float(all_rows_df["suffix_flat12"].mean()),
                "stored_long_flat_run_ge12_fraction": float((all_rows_df["max_flat_run"] >= 12).mean()),
                "stored_peak_idx_p10": float(all_rows_df["peak_idx"].quantile(0.10)),
                "stored_peak_idx_p50": float(all_rows_df["peak_idx"].quantile(0.50)),
                "stored_peak_idx_p90": float(all_rows_df["peak_idx"].quantile(0.90)),
                "stored_rmse_to_sid_mean_p50": float(np.percentile(dist_to_mean, 50)),
                "stored_rmse_to_sid_mean_p90": float(np.percentile(dist_to_mean, 90)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "worst_sid_artifact_summary.csv", index=False)
    with open(args.output_dir / f"{args.select}_sids.json", "w", encoding="utf-8") as handle:
        json.dump({f"{args.select}_sids": selected}, handle, indent=2, ensure_ascii=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
