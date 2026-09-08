#!/usr/bin/env python3
"""Build a QRS dataset variant that removes ECG tokens matching artifact rules.

This script preserves ECG-sentence rows. Bad token positions are set to zero and
marked invalid in channel/time matrices. Use it together with a training path
that masks invalid positions, or use the summary to decide whether to build a
packed-token dataset instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a clean QRS-token dataset.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--max_flat_run", type=int, default=50)
    parser.add_argument("--peak_idx_min", type=int, default=40)
    parser.add_argument("--peak_idx_max", type=int, default=65)
    return parser.parse_args()


def max_flat_run_batch(waves: np.ndarray, eps: float) -> np.ndarray:
    flat = np.abs(np.diff(waves, axis=-1)) <= eps
    flat2 = flat.reshape(-1, flat.shape[-1])
    best = np.ones(flat2.shape[0], dtype=np.int16)
    cur = np.ones(flat2.shape[0], dtype=np.int16)
    for idx in range(flat2.shape[1]):
        cur = np.where(flat2[:, idx], cur + 1, 1)
        best = np.maximum(best, cur)
    return best.reshape(waves.shape[:-1])


def artifact_mask(data: np.ndarray, chans: np.ndarray, times: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    valid = (chans > 0) & (times > 0)
    peak_idx = np.argmax(np.abs(data), axis=-1)
    flat_run = max_flat_run_batch(data, args.eps)
    bad = (
        valid
        & (
            (flat_run >= args.max_flat_run)
            | (peak_idx < args.peak_idx_min)
            | (peak_idx > args.peak_idx_max)
        )
    )
    return bad


def copy_metadata(src: Path, dst: Path) -> None:
    for path in src.iterdir():
        if path.suffix.lower() in {".csv", ".json", ".txt", ".xlsx"}:
            shutil.copy2(path, dst / path.name)


def process_split(split: str, args: argparse.Namespace) -> dict:
    src_data_path = args.src / f"{split}_data.npy"
    src_chans_path = args.src / f"{split}_data_in_chans.npy"
    src_times_path = args.src / f"{split}_data_in_times.npy"
    src_data = np.load(src_data_path, mmap_mode="r")
    src_chans = np.load(src_chans_path, mmap_mode="r")
    src_times = np.load(src_times_path, mmap_mode="r")

    dst_data = np.lib.format.open_memmap(
        args.dst / f"{split}_data.npy",
        mode="w+",
        dtype=src_data.dtype,
        shape=src_data.shape,
    )
    dst_chans = np.lib.format.open_memmap(
        args.dst / f"{split}_data_in_chans.npy",
        mode="w+",
        dtype=src_chans.dtype,
        shape=src_chans.shape,
    )
    dst_times = np.lib.format.open_memmap(
        args.dst / f"{split}_data_in_times.npy",
        mode="w+",
        dtype=src_times.dtype,
        shape=src_times.shape,
    )

    valid_before = 0
    filtered = 0
    for start in range(0, src_data.shape[0], args.block_size):
        end = min(start + args.block_size, src_data.shape[0])
        data_block = np.array(src_data[start:end], copy=True)
        chans_block = np.array(src_chans[start:end], copy=True)
        times_block = np.array(src_times[start:end], copy=True)
        valid = (chans_block > 0) & (times_block > 0)
        bad = artifact_mask(data_block, chans_block, times_block, args)
        valid_before += int(valid.sum())
        filtered += int(bad.sum())
        data_block[bad] = 0
        chans_block[bad] = 0
        times_block[bad] = 0
        dst_data[start:end] = data_block
        dst_chans[start:end] = chans_block
        dst_times[start:end] = times_block
        if start == 0 or end == src_data.shape[0] or (start // args.block_size) % 20 == 0:
            print(
                f"{split}: wrote {end}/{src_data.shape[0]} samples; "
                f"filtered={filtered}/{valid_before}",
                flush=True,
            )

    dst_data.flush()
    dst_chans.flush()
    dst_times.flush()
    kept = valid_before - filtered
    return {
        "split": split,
        "samples": int(src_data.shape[0]),
        "valid_before": int(valid_before),
        "filtered": int(filtered),
        "kept": int(kept),
        "filtered_fraction": float(filtered / valid_before) if valid_before else 0.0,
        "kept_fraction": float(kept / valid_before) if valid_before else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.src = args.src.expanduser().resolve()
    args.dst.mkdir(parents=True, exist_ok=True)
    copy_metadata(args.src, args.dst)

    summaries = [process_split(split, args) for split in args.splits]
    summary = {
        "src": str(args.src),
        "dst": str(args.dst.resolve()),
        "filter": {
            "max_flat_run": int(args.max_flat_run),
            "peak_idx_min": int(args.peak_idx_min),
            "peak_idx_max": int(args.peak_idx_max),
            "eps": float(args.eps),
        },
        "splits": summaries,
    }
    with open(args.dst / "clean_filter_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
