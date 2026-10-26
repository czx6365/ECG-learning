#!/usr/bin/env python3
"""Evaluate temporal consistency of VQ-HBR SID assignments within each ECG lead."""

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


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models import modeling_vqhbr as modeling_vqhbr  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare within-record, within-lead SID temporal consistency."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec formatted as NAME=/path/to/checkpoint.pth. Can be repeated.",
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means the whole split.")
    parser.add_argument("--max_shift", type=int, default=4)
    parser.add_argument("--corr_threshold", type=float, default=0.9)
    parser.add_argument("--min_beats", type=int, default=2)
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--run must be NAME=/path/to/checkpoint.pth, got: {spec}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty run name in: {spec}")
    return name, Path(path.strip())


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
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
    data_path = data_path.expanduser()
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()
    else:
        data_path = data_path.resolve()
    data = np.load(data_path / f"{split}_data.npy", mmap_mode="r")
    chans = np.load(data_path / f"{split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_path / f"{split}_data_in_times.npy", mmap_mode="r")
    metadata_path = data_path / f"{split}_metadata.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else None
    return data_path, data, chans, times, metadata


@torch.no_grad()
def encode_batch(model, data: np.ndarray, chans: np.ndarray, times: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).to(device, non_blocking=True)
    in_chans = torch.from_numpy(np.array(chans, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    in_times = torch.from_numpy(np.array(times, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        code_ids = model.get_codebook_indices(x, in_chan_matrix=in_chans, in_time_matrix=in_times)
    return code_ids.detach().cpu().numpy().astype(np.int64, copy=False)


def iter_batches(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


def zscore_rows(waves: np.ndarray) -> np.ndarray:
    waves = waves.astype(np.float32, copy=False)
    mean = waves.mean(axis=1, keepdims=True)
    std = waves.std(axis=1, keepdims=True)
    return (waves - mean) / np.maximum(std, 1e-6)


def overlap_for_shift(a: np.ndarray, b: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    if shift == 0:
        return a, b
    if shift > 0:
        return a[:, shift:], b[:, :-shift]
    shift_abs = -shift
    return a[:, :-shift_abs], b[:, shift_abs:]


def rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 1e-8
    corr = np.zeros(a.shape[0], dtype=np.float32)
    corr[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return np.clip(corr, -1.0, 1.0)


def adjacent_shift_corr(waves: np.ndarray, max_shift: int) -> np.ndarray:
    if len(waves) < 2:
        return np.empty(0, dtype=np.float32)
    a = zscore_rows(waves[:-1])
    b = zscore_rows(waves[1:])
    best = np.full(a.shape[0], -np.inf, dtype=np.float32)
    for shift in range(-max_shift, max_shift + 1):
        aa, bb = overlap_for_shift(a, b, shift)
        if aa.shape[1] < 8:
            continue
        best = np.maximum(best, rowwise_corr(aa, bb))
    best[~np.isfinite(best)] = 0.0
    return best


def conditional_transition_entropy(sids: np.ndarray) -> float:
    if len(sids) < 2:
        return math.nan
    by_source: dict[int, Counter] = defaultdict(Counter)
    for src, dst in zip(sids[:-1], sids[1:]):
        by_source[int(src)][int(dst)] += 1
    total = len(sids) - 1
    entropy = 0.0
    for dst_counts in by_source.values():
        src_total = sum(dst_counts.values())
        probs = np.asarray(list(dst_counts.values()), dtype=np.float64) / src_total
        entropy += (src_total / total) * float(-(probs * np.log2(probs)).sum())
    return float(entropy)


def update_global_transition_counts(global_transitions: dict[int, Counter], sids: np.ndarray) -> None:
    for src, dst in zip(sids[:-1], sids[1:]):
        global_transitions[int(src)][int(dst)] += 1


def transition_entropy_from_counts(global_transitions: dict[int, Counter]) -> float:
    total = sum(sum(counter.values()) for counter in global_transitions.values())
    if total == 0:
        return math.nan
    entropy = 0.0
    for counter in global_transitions.values():
        src_total = sum(counter.values())
        probs = np.asarray(list(counter.values()), dtype=np.float64) / src_total
        entropy += (src_total / total) * float(-(probs * np.log2(probs)).sum())
    return float(entropy)


def safe_metadata_value(metadata: pd.DataFrame | None, index: int, columns: list[str]) -> str:
    if metadata is None or index >= len(metadata):
        return ""
    row = metadata.iloc[index]
    for column in columns:
        if column in metadata.columns and pd.notna(row[column]):
            return str(row[column])
    return ""


def summarize_series(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": math.nan, "median": math.nan, "q1": math.nan, "q3": math.nan, "iqr": math.nan, "n": 0}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "mean": float(values.mean()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "n": int(values.size),
    }


def summarize_usage(counts: np.ndarray) -> dict:
    total = int(counts.sum())
    active = int((counts > 0).sum())
    probs = counts[counts > 0].astype(np.float64) / total if total else np.array([], dtype=np.float64)
    entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
    return {
        "total_tokens": total,
        "active_sids": active,
        "dead_sids": int(len(counts) - active),
        "perplexity": float(np.exp(entropy)) if probs.size else 0.0,
    }


def evaluate_run(
    name: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    data: np.ndarray,
    chans: np.ndarray,
    times: np.ndarray,
    metadata: pd.DataFrame | None,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    model, resolved_checkpoint = load_tokenizer(
        checkpoint_path, args.codebook_size, args.codebook_dim, device
    )
    num_samples = min(len(data), args.max_samples if args.max_samples > 0 else len(data))
    rows = []
    sid_counts = np.zeros(args.codebook_size, dtype=np.int64)
    global_transitions: dict[int, Counter] = defaultdict(Counter)
    totals = Counter()

    for start, end in iter_batches(num_samples, args.batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        batch_codes = encode_batch(model, batch_data, batch_chans, batch_times, device)
        valid = (batch_chans > 0) & (batch_times > 0) & (np.abs(batch_data).max(axis=-1) > 1e-8)

        valid_codes = batch_codes[valid]
        valid_codes = valid_codes[(valid_codes >= 0) & (valid_codes < args.codebook_size)]
        sid_counts += np.bincount(valid_codes, minlength=args.codebook_size)[: args.codebook_size]

        for local_idx in range(end - start):
            record_index = start + local_idx
            valid_row = valid[local_idx]
            if not valid_row.any():
                continue
            record_id = safe_metadata_value(metadata, record_index, ["csv_name", "member_name", "file_name"])
            patient_id = safe_metadata_value(metadata, record_index, ["patient_sn", "csv_name"])
            label_name = safe_metadata_value(metadata, record_index, ["label_name", "标签", "cohort"])
            for lead in sorted(np.unique(batch_chans[local_idx][valid_row]).astype(int).tolist()):
                token_indices = np.flatnonzero(valid_row & (batch_chans[local_idx] == lead))
                if len(token_indices) < args.min_beats:
                    continue
                sids = batch_codes[local_idx, token_indices].astype(np.int64, copy=False)
                waves = batch_data[local_idx, token_indices].astype(np.float32, copy=False)
                valid_sid_mask = (sids >= 0) & (sids < args.codebook_size)
                sids = sids[valid_sid_mask]
                waves = waves[valid_sid_mask]
                if len(sids) < args.min_beats:
                    continue

                adjacent_equal = sids[:-1] == sids[1:]
                adjacent_pairs = len(sids) - 1
                same_count = int(adjacent_equal.sum())
                sid_counter = Counter(int(x) for x in sids)
                transition_entropy = conditional_transition_entropy(sids)
                shift_corr = adjacent_shift_corr(waves, args.max_shift)
                high_corr = shift_corr >= args.corr_threshold
                high_corr_pairs = int(high_corr.sum())
                high_corr_sid_diff = int(((sids[:-1] != sids[1:]) & high_corr).sum())
                disagreement = (
                    float(high_corr_sid_diff / high_corr_pairs) if high_corr_pairs > 0 else math.nan
                )

                totals["record_lead_units"] += 1
                totals["beats"] += int(len(sids))
                totals["adjacent_pairs"] += int(adjacent_pairs)
                totals["adjacent_same"] += same_count
                totals["high_corr_pairs"] += high_corr_pairs
                totals["high_corr_sid_diff"] += high_corr_sid_diff
                update_global_transition_counts(global_transitions, sids)

                rows.append(
                    {
                        "run": name,
                        "record_index": record_index,
                        "record_id": record_id,
                        "patient_id": patient_id,
                        "label_name": label_name,
                        "lead": lead,
                        "n_beats": int(len(sids)),
                        "adjacent_pairs": int(adjacent_pairs),
                        "adjacent_same_sid_rate": float(same_count / adjacent_pairs),
                        "modal_sid": int(sid_counter.most_common(1)[0][0]),
                        "modal_sid_ratio": float(sid_counter.most_common(1)[0][1] / len(sids)),
                        "unique_sid_count": int(len(sid_counter)),
                        "unique_sid_beat_ratio": float(len(sid_counter) / len(sids)),
                        "transition_entropy_bits": transition_entropy,
                        "adjacent_shift_corr_mean": float(np.mean(shift_corr)) if len(shift_corr) else math.nan,
                        "adjacent_shift_corr_median": float(np.median(shift_corr)) if len(shift_corr) else math.nan,
                        "high_corr_pairs": high_corr_pairs,
                        "high_corr_pair_ratio": float(high_corr_pairs / adjacent_pairs),
                        "morphology_consistent_sid_disagreement": disagreement,
                    }
                )

        if start == 0 or end == num_samples or (start // args.batch_size) % 20 == 0:
            print(
                f"{name}: encoded {end}/{num_samples}; "
                f"record-lead units={totals['record_lead_units']}; "
                f"adjacent pairs={totals['adjacent_pairs']}",
                flush=True,
            )

    per_unit = pd.DataFrame(rows)
    usage = summarize_usage(sid_counts)
    summary = {
        "run": name,
        "checkpoint": str(resolved_checkpoint),
        "num_records": int(num_samples),
        "record_lead_units": int(totals["record_lead_units"]),
        "total_beats": int(totals["beats"]),
        "total_adjacent_pairs": int(totals["adjacent_pairs"]),
        "total_high_corr_pairs": int(totals["high_corr_pairs"]),
        "high_corr_pair_coverage": float(totals["high_corr_pairs"] / totals["adjacent_pairs"])
        if totals["adjacent_pairs"]
        else math.nan,
        "global_adjacent_same_sid_rate": float(totals["adjacent_same"] / totals["adjacent_pairs"])
        if totals["adjacent_pairs"]
        else math.nan,
        "global_morphology_consistent_sid_disagreement": float(
            totals["high_corr_sid_diff"] / totals["high_corr_pairs"]
        )
        if totals["high_corr_pairs"]
        else math.nan,
        "global_transition_entropy_bits": transition_entropy_from_counts(global_transitions),
        **usage,
    }

    metric_columns = [
        "adjacent_same_sid_rate",
        "modal_sid_ratio",
        "unique_sid_beat_ratio",
        "transition_entropy_bits",
        "adjacent_shift_corr_mean",
        "high_corr_pair_ratio",
        "morphology_consistent_sid_disagreement",
    ]
    for column in metric_columns:
        stats = summarize_series(per_unit[column].to_numpy() if column in per_unit else np.array([]))
        for key, value in stats.items():
            summary[f"{column}_{key}"] = value

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, per_unit


def main() -> None:
    args = parse_args()
    runs = [parse_run_spec(spec) for spec in args.run]
    device = choose_device(args.device)
    data_path, data, chans, times, metadata = load_arrays(args.data_path, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name, checkpoint_path in runs:
        summary, per_unit = evaluate_run(name, checkpoint_path, args, data, chans, times, metadata, device)
        per_unit_path = args.output_dir / f"{name}_per_record_lead.csv"
        per_unit.to_csv(per_unit_path, index=False)
        summary["per_record_lead_csv"] = str(per_unit_path)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "temporal_consistency_summary.csv", index=False)
    with open(args.output_dir / "temporal_consistency_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(data_path),
                "split": args.split,
                "codebook_size": args.codebook_size,
                "codebook_dim": args.codebook_dim,
                "max_shift": args.max_shift,
                "corr_threshold": args.corr_threshold,
                "unit": "one 10-second ECG record x one lead",
                "summaries": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    columns = [
        "run",
        "active_sids",
        "perplexity",
        "global_adjacent_same_sid_rate",
        "adjacent_same_sid_rate_mean",
        "modal_sid_ratio_mean",
        "unique_sid_beat_ratio_mean",
        "global_transition_entropy_bits",
        "transition_entropy_bits_mean",
        "adjacent_shift_corr_mean_mean",
        "high_corr_pair_coverage",
        "global_morphology_consistent_sid_disagreement",
        "morphology_consistent_sid_disagreement_mean",
    ]
    print("\n=== Temporal Consistency Summary ===")
    print(summary_df[columns].to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
