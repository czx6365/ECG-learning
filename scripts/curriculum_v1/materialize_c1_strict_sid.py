#!/usr/bin/env python3
"""Materialize the strict C1 SID train/val view from full source SID arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import shutil


SOURCE_SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_sid_dir", type=Path, required=True)
    parser.add_argument("--source_data_dir", type=Path, required=True)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def copy_indexed(out: np.memmap, sources: dict[str, np.ndarray], codes: np.ndarray, indices: np.ndarray) -> None:
    for code, name in enumerate(SOURCE_SPLITS):
        positions = np.flatnonzero(codes == code)
        if positions.size:
            out[positions] = sources[name][indices[positions]]
    out.flush()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"splits": {}, "source_splits": list(SOURCE_SPLITS)}

    sid_sources = {
        name: np.load(args.full_sid_dir / f"{name}_sid.npy", mmap_mode="r")
        for name in SOURCE_SPLITS
    }
    chan_sources = {
        name: np.load(args.source_data_dir / f"{name}_data_in_chans.npy", mmap_mode="r")
        for name in SOURCE_SPLITS
    }
    time_sources = {
        name: np.load(args.source_data_dir / f"{name}_data_in_times.npy", mmap_mode="r")
        for name in SOURCE_SPLITS
    }

    for split in SOURCE_SPLITS:
        codes = np.load(args.index_dir / f"{split}_source_code.npy")
        indices = np.load(args.index_dir / f"{split}_source_index.npy")
        n = len(indices)
        if len(codes) != n:
            raise RuntimeError(f"Index length mismatch for {split}")

        specs = (
            ("sid", sid_sources, np.int16),
            ("data_in_chans", chan_sources, np.int32),
            ("data_in_times", time_sources, np.int32),
        )
        for suffix, sources, dtype in specs:
            shape = (n, sources["train"].shape[1])
            out = np.lib.format.open_memmap(
                args.output_dir / f"{split}_{suffix}.npy",
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
            copy_indexed(out, sources, codes, indices)
            del out

        shutil.copy2(args.index_dir / f"{split}_metadata.csv", args.output_dir / f"{split}_metadata.csv")
        sid = np.load(args.output_dir / f"{split}_sid.npy", mmap_mode="r")
        valid = (sid >= 0) & (sid < 1024)
        summary["splits"][split] = {
            "records": n,
            "valid_tokens": int(valid.sum()),
            "pad_tokens": int((sid == 1024).sum()),
            "active_sid": int(np.unique(sid[valid]).size),
        }

    index_summary = json.loads((args.index_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "full_sid_dir": str(args.full_sid_dir),
            "source_data_dir": str(args.source_data_dir),
            "index_dir": str(args.index_dir),
            "index_manifest_sha256": index_summary["assignment_manifest_sha256"],
        }
    )
    summary_path = args.output_dir / "strict_sid_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
