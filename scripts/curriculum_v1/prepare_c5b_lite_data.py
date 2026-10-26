#!/usr/bin/env python3
"""Audit raw-QRS/fixed-SID alignment and cache C2.1 discrete morphology targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import train_morphology_v2_physaware_vq as morphology


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def equal_arrays(left: Path, right: Path, chunk_rows: int = 512) -> bool:
    a = np.load(left, mmap_mode="r", allow_pickle=False)
    b = np.load(right, mmap_mode="r", allow_pickle=False)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    for start in range(0, len(a), chunk_rows):
        end = min(start + chunk_rows, len(a))
        if not np.array_equal(np.asarray(a[start:end]), np.asarray(b[start:end])):
            return False
    return True


def build_discrete_cache(raw_dir: Path, out_dir: Path, split: str, chunk_records: int) -> dict:
    data = np.load(raw_dir / f"{split}_data.npy", mmap_mode="r")
    chans = np.load(raw_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(raw_dir / f"{split}_data_in_times.npy", mmap_mode="r")
    target_path = out_dir / f"{split}_phys_discrete.npy"
    target = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.int16,
        shape=(data.shape[0], data.shape[1], len(morphology.DISCRETE_FEATURES)),
    )
    target[:] = np.int16(morphology.IGNORE_INDEX)
    config = SimpleNamespace(
        fs=100.0,
        qrs_onset_threshold_frac=0.15,
        qrs_wide_ms=120.0,
        st_threshold=0.1,
        t_relative_threshold=0.05,
        t_abs_threshold=0.02,
    )
    valid_total = 0
    counts = {
        name: np.zeros(len(categories), dtype=np.int64)
        for name, categories in morphology.DISCRETE_FEATURES.items()
    }
    for start in range(0, len(data), chunk_records):
        end = min(start + chunk_records, len(data))
        x = np.asarray(data[start:end], dtype=np.float32)
        c = np.asarray(chans[start:end])
        t = np.asarray(times[start:end])
        valid = (c > 0) & (t > 0) & (np.abs(x).max(axis=-1) > 1e-8)
        discrete, _, _ = morphology.extract_physiology_vectorized(x[valid], config)
        block = np.full(
            (end - start, data.shape[1], len(morphology.DISCRETE_FEATURES)),
            morphology.IGNORE_INDEX,
            dtype=np.int16,
        )
        block[valid] = discrete
        target[start:end] = block
        valid_total += int(valid.sum())
        for feature_idx, (name, categories) in enumerate(morphology.DISCRETE_FEATURES.items()):
            labels = discrete[:, feature_idx]
            ok = (labels >= 0) & (labels < len(categories))
            counts[name] += np.bincount(labels[ok], minlength=len(categories))[: len(categories)]
        if start == 0 or end == len(data) or (start // chunk_records) % 20 == 0:
            print(f"{split}: morphology cache {end}/{len(data)}; valid={valid_total}", flush=True)
    target.flush()
    return {
        "path": str(target_path),
        "shape": list(target.shape),
        "valid_tokens": valid_total,
        "counts": {key: value.tolist() for key, value in counts.items()},
        "sha256": sha256(target_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=Path, required=True)
    parser.add_argument("--fixed_sid_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--chunk_records", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raw_dir = resolve(args.raw_data_dir)
    sid_dir = resolve(args.fixed_sid_dir)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "alignment_audit.json"
    if audit_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {audit_path}")

    report = {
        "raw_data_dir": str(raw_dir),
        "fixed_sid_dir": str(sid_dir),
        "splits": {},
        "all_checks_passed": True,
    }
    for split in ("train", "val", "test"):
        raw = np.load(raw_dir / f"{split}_data.npy", mmap_mode="r")
        sid = np.load(sid_dir / f"{split}_sid.npy", mmap_mode="r")
        labels_raw = np.load(raw_dir / f"{split}_labels.npy", mmap_mode="r")
        labels_sid = np.load(sid_dir / f"{split}_labels.npy", mmap_mode="r")
        chans_raw_path = raw_dir / f"{split}_data_in_chans.npy"
        chans_sid_path = sid_dir / f"{split}_data_in_chans.npy"
        times_raw_path = raw_dir / f"{split}_data_in_times.npy"
        times_sid_path = sid_dir / f"{split}_data_in_times.npy"
        chans_raw = np.load(chans_raw_path, mmap_mode="r")
        times_raw = np.load(times_raw_path, mmap_mode="r")
        valid_wave = np.zeros(raw.shape[:2], dtype=bool)
        for start in range(0, len(raw), args.chunk_records):
            end = min(start + args.chunk_records, len(raw))
            valid_wave[start:end] = (
                (np.asarray(chans_raw[start:end]) > 0)
                & (np.asarray(times_raw[start:end]) > 0)
                & (np.abs(np.asarray(raw[start:end])).max(axis=-1) > 1e-8)
            )
        valid_sid = (sid >= 0) & (sid < 1024)
        checks = {
            "records_match": int(raw.shape[0]) == int(sid.shape[0]),
            "tokens_match": tuple(raw.shape[:2]) == tuple(sid.shape),
            "token_len_96": int(raw.shape[-1]) == 96,
            "labels_equal": np.array_equal(labels_raw, labels_sid),
            "channels_equal": equal_arrays(chans_raw_path, chans_sid_path),
            "times_equal": equal_arrays(times_raw_path, times_sid_path),
            "valid_masks_equal": np.array_equal(valid_wave, valid_sid),
        }
        if not all(checks.values()):
            report["all_checks_passed"] = False
        report["splits"][split] = {
            "raw_shape": list(raw.shape),
            "sid_shape": list(sid.shape),
            "label_shape": list(labels_raw.shape),
            "valid_tokens": int(valid_wave.sum()),
            "checks": checks,
            "morphology_cache": build_discrete_cache(raw_dir, out_dir, split, args.chunk_records),
        }
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise RuntimeError("C5b raw/SID alignment audit failed")


if __name__ == "__main__":
    main()
