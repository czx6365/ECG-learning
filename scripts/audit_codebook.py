#!/usr/bin/env python3
"""Audit VQ-HBR codebook assignments for QRS-tokenized ECG datasets.

This script is intentionally standalone: it loads an existing VQ-HBR checkpoint,
maps QRS heartbeat tokens to codebook ids, and writes statistical/clinical
diagnostics without changing any training code or model architecture.

Example:
    python scripts/audit_codebook.py \
      --checkpoint_path checkpoints/vqhbr/ECG_DSA_all_raw_v1 \
      --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/target_raw_v1 \
      --metadata_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/target_raw_v1/val_metadata.csv \
      --split val \
      --output_dir outputs/codebook_audit \
      --max_samples 2000 \
      --codebook_size 8192 \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from timm.models import create_model


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models import modeling_vqhbr as modeling_vqhbr  # noqa: F401


LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ID_TO_NAME = {idx + 1: name for idx, name in enumerate(LEAD_ORDER)}
LEAD_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(LEAD_ORDER)}
STENOSIS_COLS = [
    "overall_max_stenosis",
    "lm_max_stenosis",
    "lad_max_stenosis",
    "lca_max_stenosis",
    "rca_max_stenosis",
]
SAME_GROUP_PAIRS = [
    ("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"),
    ("V5", "V6"), ("I", "aVL"), ("II", "III"), ("III", "aVF"),
]
DIFF_GROUP_PAIRS = [
    ("V1", "II"), ("V2", "aVF"), ("V3", "III"),
    ("V5", "aVR"), ("aVL", "aVF"), ("I", "III"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit VQ-HBR codebook usage on QRS-tokenized ECG samples."
    )
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/codebook_audit"))
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--codebook_size", type=int, default=8192)
    parser.add_argument("--codebook_dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid_eps", type=float, default=1e-8)
    parser.add_argument(
        "--valid_by_waveform",
        action="store_true",
        help="Also require a nonzero waveform when filtering valid tokens. Default uses channel/time ids only.",
    )
    parser.add_argument("--top_heatmap_codes", type=int, default=200)
    parser.add_argument("--top_prototype_codes", type=int, default=30)
    parser.add_argument("--top_associated_codes", type=int, default=20)
    parser.add_argument("--prototype_overlays", type=int, default=20)
    parser.add_argument("--topk_overlap", type=int, default=20)
    parser.add_argument(
        "--skip_reconstruction",
        action="store_true",
        help="Skip reconstruction-quality metrics.",
    )
    parser.add_argument(
        "--reconstruction_max_samples",
        type=int,
        default=0,
        help="Samples used for reconstruction metrics. 0 means reuse --max_samples.",
    )
    parser.add_argument(
        "--pairwise_sample_size",
        type=int,
        default=2048,
        help="Number of codebook vectors sampled for approximate pairwise geometry metrics.",
    )
    parser.add_argument(
        "--skip_full_heatmap",
        action="store_true",
        help="Skip the full lead x codebook heatmap if codebook_size is very large.",
    )
    return parser.parse_args()


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
        raise FileNotFoundError(f"No .pth checkpoint found under {path}")
    return candidates[-1]


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def patch_cross_platform_checkpoint_paths() -> None:
    """Allow checkpoints saved with WindowsPath args to load on POSIX systems."""
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


def load_tokenizer(checkpoint_path: Path, codebook_size: int, codebook_dim: int, device: torch.device):
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    patch_cross_platform_checkpoint_paths()
    model = create_model(
        "vqhbr",
        pretrained=True,
        pretrained_weight=str(checkpoint_path),
        as_tokenzer=True,
        n_code=codebook_size,
        code_dim=codebook_dim,
        quantize_kmeans_init=False,
    )
    model.to(device)
    model.eval()
    return model, checkpoint_path


def load_split_arrays(data_path: Path, split: str):
    data_path = data_path.expanduser().resolve()
    data = np.load(data_path / f"{split}_data.npy", mmap_mode="r")
    in_chans = np.load(data_path / f"{split}_data_in_chans.npy", mmap_mode="r")
    in_times = np.load(data_path / f"{split}_data_in_times.npy", mmap_mode="r")
    if data.shape[:2] != in_chans.shape or data.shape[:2] != in_times.shape:
        raise ValueError(
            "Data and channel/time matrices are misaligned: "
            f"data={data.shape}, in_chans={in_chans.shape}, in_times={in_times.shape}"
        )
    return data, in_chans, in_times


def infer_metadata_path(data_path: Path, split: str, metadata_path: Optional[Path]) -> Optional[Path]:
    if metadata_path is not None:
        return metadata_path.expanduser().resolve()
    split_path = data_path / f"{split}_metadata.csv"
    if split_path.exists():
        return split_path
    all_path = data_path / "all_metadata.csv"
    if all_path.exists():
        return all_path
    return None


def load_metadata(data_path: Path, split: str, metadata_path: Optional[Path], sample_count: int) -> pd.DataFrame:
    path = infer_metadata_path(data_path, split, metadata_path)
    if path is None or not path.exists():
        print("No metadata CSV found; stenosis label columns will be omitted.")
        return pd.DataFrame(index=np.arange(sample_count))

    metadata = pd.read_csv(path, encoding="utf-8-sig")
    if "split" in metadata.columns:
        split_metadata = metadata[metadata["split"].astype(str) == split].reset_index(drop=True)
        if len(split_metadata) > 0:
            metadata = split_metadata

    if len(metadata) != sample_count:
        print(
            f"Metadata row count {len(metadata)} does not match split sample count {sample_count}. "
            "Sample-level labels will be attached only where row indices overlap."
        )
    return metadata.reset_index(drop=True)


def load_aux_fallback(data_path: Path, split: str, sample_count: int) -> pd.DataFrame:
    aux_path = data_path / f"{split}_aux_regression_labels.npy"
    if not aux_path.exists():
        return pd.DataFrame(index=np.arange(sample_count))
    aux = np.load(aux_path, mmap_mode="r")
    if aux.ndim == 1:
        aux = aux.reshape(-1, 1)
    if aux.shape[0] != sample_count:
        return pd.DataFrame(index=np.arange(sample_count))
    cols = STENOSIS_COLS[: aux.shape[1]]
    return pd.DataFrame(np.asarray(aux[:, : len(cols)]), columns=cols)


def get_label_frame(data_path: Path, split: str, metadata: pd.DataFrame, sample_count: int) -> tuple[pd.DataFrame, list[str]]:
    labels = pd.DataFrame(index=np.arange(sample_count))
    for col in STENOSIS_COLS:
        if col in metadata.columns:
            labels[col] = pd.to_numeric(metadata[col], errors="coerce").reindex(labels.index)

    missing = [col for col in STENOSIS_COLS if col not in labels.columns]
    if missing:
        aux = load_aux_fallback(data_path, split, sample_count)
        for col in missing:
            if col in aux.columns:
                labels[col] = pd.to_numeric(aux[col], errors="coerce").reindex(labels.index)

    present_cols = [col for col in STENOSIS_COLS if col in labels.columns]
    return labels, present_cols


def valid_token_mask(
    batch_data: np.ndarray,
    batch_chans: np.ndarray,
    batch_times: np.ndarray,
    eps: float,
    valid_by_waveform: bool = False,
) -> np.ndarray:
    valid = (batch_chans > 0) & (batch_times > 0)
    if valid_by_waveform:
        waveform_nonzero = np.any(np.abs(batch_data) > eps, axis=-1)
        valid = valid & waveform_nonzero
    return valid


def iter_batches(num_samples: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


@torch.no_grad()
def encode_batch(model, data: np.ndarray, chans: np.ndarray, times: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).to(device, non_blocking=True)
    in_chans = torch.from_numpy(np.array(chans, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    in_times = torch.from_numpy(np.array(times, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        code_ids = model.get_codebook_indices(x, in_chan_matrix=in_chans, in_time_matrix=in_times)
    return code_ids.detach().cpu().numpy().astype(np.int64, copy=False)


@torch.no_grad()
def reconstruct_batch(
    model,
    data: np.ndarray,
    chans: np.ndarray,
    times: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).to(device, non_blocking=True)
    in_chans = torch.from_numpy(np.array(chans, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    in_times = torch.from_numpy(np.array(times, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        quantize, embed_ind, _ = model.encode(
            x,
            in_chan_matrix=in_chans,
            in_time_matrix=in_times,
        )
        rec = model.decode(
            quantize,
            in_chan_matrix=in_chans,
            in_time_matrix=in_times,
        )
    return (
        embed_ind.view(x.shape[0], -1).detach().cpu().numpy().astype(np.int64, copy=False),
        rec.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def extract_assignments(
    model,
    data,
    in_chans,
    in_times,
    labels: pd.DataFrame,
    label_cols: list[str],
    max_samples: int,
    batch_size: int,
    codebook_size: int,
    device: torch.device,
    valid_eps: float,
    valid_by_waveform: bool,
):
    num_samples = min(len(data), max_samples if max_samples > 0 else len(data))
    rows = []
    code_counts = np.zeros(codebook_size, dtype=np.int64)
    lead_code_counts = np.zeros((len(LEAD_ORDER), codebook_size), dtype=np.int64)

    for start, end in iter_batches(num_samples, batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(in_chans[start:end])
        batch_times = np.asarray(in_times[start:end])
        batch_codes = encode_batch(model, batch_data, batch_chans, batch_times, device)
        batch_valid = valid_token_mask(
            batch_data,
            batch_chans,
            batch_times,
            valid_eps,
            valid_by_waveform=valid_by_waveform,
        )

        valid_positions = np.argwhere(batch_valid)
        for local_sample_idx, beat_index in valid_positions:
            sample_id = start + int(local_sample_idx)
            lead_id = int(batch_chans[local_sample_idx, beat_index])
            code_id = int(batch_codes[local_sample_idx, beat_index])
            if code_id < 0 or code_id >= codebook_size:
                continue
            lead_name = LEAD_ID_TO_NAME.get(lead_id, f"lead_{lead_id}")
            code_counts[code_id] += 1
            if lead_name in LEAD_NAME_TO_ID:
                lead_code_counts[LEAD_NAME_TO_ID[lead_name] - 1, code_id] += 1

            row = {
                "sample_id": sample_id,
                "beat_index": int(beat_index),
                "lead_id": lead_id,
                "lead_name": lead_name,
                "code_id": code_id,
                "is_valid": True,
            }
            for col in label_cols:
                row[col] = labels.at[sample_id, col] if sample_id in labels.index else np.nan
            rows.append(row)

        print(f"Encoded samples {start}:{end}; valid tokens so far={int(code_counts.sum())}")

    assignments = pd.DataFrame(rows)
    return assignments, code_counts, lead_code_counts, num_samples


def usage_summary(code_counts: np.ndarray, codebook_size: int) -> dict:
    total_tokens = int(code_counts.sum())
    active_codes = int((code_counts > 0).sum())
    probs = code_counts.astype(np.float64) / total_tokens if total_tokens else np.zeros_like(code_counts, dtype=np.float64)
    active_probs = probs[probs > 0]
    entropy = float(-(active_probs * np.log(active_probs)).sum()) if active_probs.size else 0.0
    perplexity = float(np.exp(entropy)) if active_probs.size else 0.0
    top_n = max(1, int(math.ceil(codebook_size * 0.01)))
    top_usage = float(np.sort(code_counts)[::-1][:top_n].sum() / total_tokens) if total_tokens else 0.0
    top_indices = np.argsort(code_counts)[::-1][:10]
    top_10 = [
        {
            "code_id": int(idx),
            "count": int(code_counts[idx]),
            "frequency": float(code_counts[idx] / total_tokens) if total_tokens else 0.0,
        }
        for idx in top_indices
    ]
    return {
        "total_tokens": total_tokens,
        "codebook_size": int(codebook_size),
        "active_codes": active_codes,
        "active_ratio": float(active_codes / codebook_size) if codebook_size else 0.0,
        "dead_codes": int(codebook_size - active_codes),
        "dead_ratio": float((codebook_size - active_codes) / codebook_size) if codebook_size else 0.0,
        "entropy": entropy,
        "perplexity": perplexity,
        "top_1_percent_usage": top_usage,
        "top_10_codes": top_10,
    }


def reconstruction_summary_from_sums(total_count: int, mse_sum: float, mae_sum: float) -> dict:
    if total_count <= 0:
        return {
            "valid_tokens": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
    mse = float(mse_sum / total_count)
    mae = float(mae_sum / total_count)
    return {
        "valid_tokens": int(total_count),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
    }


def reconstruction_quality(
    model,
    data,
    in_chans,
    in_times,
    max_samples: int,
    batch_size: int,
    codebook_size: int,
    device: torch.device,
    valid_eps: float,
    valid_by_waveform: bool,
    output_dir: Path,
) -> dict:
    num_samples = min(len(data), max_samples if max_samples > 0 else len(data))
    total_count = 0
    mse_sum = 0.0
    mae_sum = 0.0
    sample_rows = []

    code_count = np.zeros(codebook_size, dtype=np.int64)
    code_mse_sum = np.zeros(codebook_size, dtype=np.float64)
    code_mae_sum = np.zeros(codebook_size, dtype=np.float64)
    lead_count = np.zeros(len(LEAD_ORDER), dtype=np.int64)
    lead_mse_sum = np.zeros(len(LEAD_ORDER), dtype=np.float64)
    lead_mae_sum = np.zeros(len(LEAD_ORDER), dtype=np.float64)

    for start, end in iter_batches(num_samples, batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(in_chans[start:end])
        batch_times = np.asarray(in_times[start:end])
        batch_codes, batch_rec = reconstruct_batch(model, batch_data, batch_chans, batch_times, device)
        batch_valid = valid_token_mask(
            batch_data,
            batch_chans,
            batch_times,
            valid_eps,
            valid_by_waveform=valid_by_waveform,
        )

        token_mse = ((batch_rec - batch_data) ** 2).mean(axis=-1)
        token_mae = np.abs(batch_rec - batch_data).mean(axis=-1)
        valid_positions = np.argwhere(batch_valid)

        for local_sample_idx, beat_index in valid_positions:
            sample_id = start + int(local_sample_idx)
            code_id = int(batch_codes[local_sample_idx, beat_index])
            lead_id = int(batch_chans[local_sample_idx, beat_index])
            mse = float(token_mse[local_sample_idx, beat_index])
            mae = float(token_mae[local_sample_idx, beat_index])
            total_count += 1
            mse_sum += mse
            mae_sum += mae
            if 0 <= code_id < codebook_size:
                code_count[code_id] += 1
                code_mse_sum[code_id] += mse
                code_mae_sum[code_id] += mae
            lead_name = LEAD_ID_TO_NAME.get(lead_id)
            if lead_name in LEAD_NAME_TO_ID:
                lead_idx = LEAD_NAME_TO_ID[lead_name] - 1
                lead_count[lead_idx] += 1
                lead_mse_sum[lead_idx] += mse
                lead_mae_sum[lead_idx] += mae

        for local_sample_idx in range(end - start):
            valid = batch_valid[local_sample_idx]
            valid_count = int(valid.sum())
            if valid_count == 0:
                continue
            sample_mse = float(token_mse[local_sample_idx][valid].mean())
            sample_mae = float(token_mae[local_sample_idx][valid].mean())
            sample_rows.append(
                {
                    "sample_id": start + local_sample_idx,
                    "valid_tokens": valid_count,
                    "mse": sample_mse,
                    "rmse": math.sqrt(sample_mse),
                    "mae": sample_mae,
                }
            )

        print(f"Reconstructed samples {start}:{end}; valid tokens so far={total_count}")

    summary = reconstruction_summary_from_sums(total_count, mse_sum, mae_sum)
    summary["evaluated_samples"] = int(num_samples)

    by_code_rows = []
    for code_id in np.where(code_count > 0)[0]:
        mse = float(code_mse_sum[code_id] / code_count[code_id])
        by_code_rows.append(
            {
                "code_id": int(code_id),
                "count": int(code_count[code_id]),
                "mse": mse,
                "rmse": float(math.sqrt(mse)),
                "mae": float(code_mae_sum[code_id] / code_count[code_id]),
            }
        )
    by_code = pd.DataFrame(by_code_rows)
    if not by_code.empty:
        by_code = by_code.sort_values(["mse", "count"], ascending=[False, False])
    by_code.to_csv(output_dir / "reconstruction_by_code.csv", index=False)

    by_lead_rows = []
    for lead_idx, lead_name in enumerate(LEAD_ORDER):
        if lead_count[lead_idx] == 0:
            continue
        mse = float(lead_mse_sum[lead_idx] / lead_count[lead_idx])
        by_lead_rows.append(
            {
                "lead_name": lead_name,
                "count": int(lead_count[lead_idx]),
                "mse": mse,
                "rmse": float(math.sqrt(mse)),
                "mae": float(lead_mae_sum[lead_idx] / lead_count[lead_idx]),
            }
        )
    pd.DataFrame(by_lead_rows).to_csv(output_dir / "reconstruction_by_lead.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output_dir / "reconstruction_by_sample.csv", index=False)

    if sample_rows:
        sample_mse = np.asarray([row["mse"] for row in sample_rows], dtype=np.float64)
        summary["sample_mse_p50"] = float(np.percentile(sample_mse, 50))
        summary["sample_mse_p90"] = float(np.percentile(sample_mse, 90))
        summary["sample_mse_p95"] = float(np.percentile(sample_mse, 95))

    if not by_code.empty:
        summary["worst_10_codes_by_mse"] = by_code.head(10).to_dict(orient="records")
        summary["best_10_codes_by_mse_min_count_10"] = (
            by_code[by_code["count"] >= 10]
            .sort_values(["mse", "count"], ascending=[True, False])
            .head(10)
            .to_dict(orient="records")
        )

    with open(output_dir / "reconstruction_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if sample_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sample_mse, bins=80, color="tab:blue", alpha=0.85)
        ax.set_xlabel("Sample MSE")
        ax.set_ylabel("Count")
        ax.set_title("Reconstruction MSE by ECG sentence")
        fig.tight_layout()
        fig.savefig(output_dir / "reconstruction_sample_mse_hist.png", dpi=200)
        plt.close(fig)

    return summary


def row_normalize(counts: np.ndarray) -> np.ndarray:
    denom = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, denom, out=np.zeros_like(counts, dtype=np.float64), where=denom > 0)


def plot_usage_bar(code_counts: np.ndarray, output_path: Path):
    sorted_counts = np.sort(code_counts)[::-1]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(len(sorted_counts)), sorted_counts, width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Code rank by frequency")
    ax.set_ylabel("Token count (log scale)")
    ax.set_title("VQ-HBR codebook usage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lead_code_heatmap(matrix: np.ndarray, code_ids: np.ndarray, output_path: Path, title: str):
    fig_width = max(10, min(28, len(code_ids) / 10))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(np.arange(len(LEAD_ORDER)))
    ax.set_yticklabels(LEAD_ORDER)
    if len(code_ids) <= 250:
        tick_count = min(20, len(code_ids))
        tick_pos = np.linspace(0, len(code_ids) - 1, tick_count, dtype=int)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(int(code_ids[pos])) for pos in tick_pos], rotation=90)
    else:
        ax.set_xticks([])
    ax.set_xlabel("Code id" if len(code_ids) <= 250 else "Code id (full codebook order)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="P(code | lead)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm > 0)
    return normalized @ normalized.T


def plot_similarity(similarity: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(similarity, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(np.arange(len(LEAD_ORDER)))
    ax.set_yticks(np.arange(len(LEAD_ORDER)))
    ax.set_xticklabels(LEAD_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(LEAD_ORDER)
    ax.set_title("Lead code-distribution cosine similarity")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def ltcs_report(similarity: np.ndarray) -> dict:
    def pair_score(pair):
        a, b = pair
        return float(similarity[LEAD_NAME_TO_ID[a] - 1, LEAD_NAME_TO_ID[b] - 1])

    same_scores = {f"{a}-{b}": pair_score((a, b)) for a, b in SAME_GROUP_PAIRS}
    diff_scores = {f"{a}-{b}": pair_score((a, b)) for a, b in DIFF_GROUP_PAIRS}
    same_mean = float(np.mean(list(same_scores.values()))) if same_scores else 0.0
    diff_mean = float(np.mean(list(diff_scores.values()))) if diff_scores else 0.0
    return {
        "same_group_mean": same_mean,
        "diff_group_mean": diff_mean,
        "LTCS": float(same_mean - diff_mean),
        "same_group_pair_scores": same_scores,
        "diff_group_pair_scores": diff_scores,
    }


def lead_code_counts_from_assignments(
    assignments: pd.DataFrame,
    codebook_size: int,
    sample_ids: Optional[set[int]] = None,
) -> np.ndarray:
    counts = np.zeros((len(LEAD_ORDER), codebook_size), dtype=np.int64)
    if assignments.empty:
        return counts
    frame = assignments
    if sample_ids is not None:
        frame = frame[frame["sample_id"].isin(sample_ids)]
    for row in frame[["lead_name", "code_id"]].itertuples(index=False):
        lead_name = row.lead_name
        code_id = int(row.code_id)
        if lead_name not in LEAD_NAME_TO_ID or code_id < 0 or code_id >= codebook_size:
            continue
        counts[LEAD_NAME_TO_ID[lead_name] - 1, code_id] += 1
    return counts


def high_stenosis_sample_ids(labels: pd.DataFrame, label_cols: list[str]) -> Optional[set[int]]:
    if not label_cols:
        return None
    if "overall_max_stenosis" in labels.columns:
        values = pd.to_numeric(labels["overall_max_stenosis"], errors="coerce")
        high = values >= 50
    else:
        available = [col for col in label_cols if col in labels.columns]
        if not available:
            return None
        numeric = labels[available].apply(pd.to_numeric, errors="coerce")
        high = numeric.ge(50).any(axis=1)
    sample_ids = set(int(idx) for idx in labels.index[high.fillna(False)])
    return sample_ids if sample_ids else None


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def kl_div(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())

    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def save_lead_similarity_tables(lead_distribution: np.ndarray, output_dir: Path, top_k: int):
    js_rows = []
    overlap_rows = []
    top_sets = []
    for lead_idx in range(len(LEAD_ORDER)):
        if lead_distribution[lead_idx].sum() <= 0:
            top_sets.append(set())
        else:
            top_ids = np.argsort(lead_distribution[lead_idx])[::-1][:top_k]
            top_sets.append(set(int(code_id) for code_id in top_ids))

    for i, lead_a in enumerate(LEAD_ORDER):
        for j, lead_b in enumerate(LEAD_ORDER):
            js_rows.append(
                {
                    "lead_a": lead_a,
                    "lead_b": lead_b,
                    "js_divergence": js_divergence(lead_distribution[i], lead_distribution[j]),
                }
            )
            intersection = top_sets[i] & top_sets[j]
            union = top_sets[i] | top_sets[j]
            overlap_rows.append(
                {
                    "lead_a": lead_a,
                    "lead_b": lead_b,
                    "top_k": int(top_k),
                    "overlap_count": int(len(intersection)),
                    "overlap_ratio": float(len(intersection) / top_k) if top_k else 0.0,
                    "jaccard": float(len(intersection) / len(union)) if union else 0.0,
                }
            )

    pd.DataFrame(js_rows).to_csv(output_dir / "lead_similarity_js_divergence.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output_dir / "lead_topk_code_overlap.csv", index=False)


def save_top_codes_per_lead(lead_code_counts: np.ndarray, output_path: Path):
    rows = []
    for lead_idx, lead_name in enumerate(LEAD_ORDER):
        counts = lead_code_counts[lead_idx]
        total = int(counts.sum())
        top_ids = np.argsort(counts)[::-1][:20]
        for rank, code_id in enumerate(top_ids, start=1):
            rows.append(
                {
                    "lead_name": lead_name,
                    "rank": rank,
                    "code_id": int(code_id),
                    "count": int(counts[code_id]),
                    "frequency": float(counts[code_id] / total) if total else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def collect_prototypes(
    model,
    data,
    in_chans,
    in_times,
    target_codes: set[int],
    max_samples: int,
    batch_size: int,
    device: torch.device,
    valid_eps: float,
    valid_by_waveform: bool,
    overlay_limit: int,
    seed: int,
):
    rng = random.Random(seed)
    stats = {
        code_id: {"count": 0, "sum": None, "sumsq": None, "examples": []}
        for code_id in target_codes
    }
    num_samples = min(len(data), max_samples if max_samples > 0 else len(data))
    for start, end in iter_batches(num_samples, batch_size):
        batch_data = np.asarray(data[start:end])
        batch_chans = np.asarray(in_chans[start:end])
        batch_times = np.asarray(in_times[start:end])
        batch_codes = encode_batch(model, batch_data, batch_chans, batch_times, device)
        batch_valid = valid_token_mask(
            batch_data,
            batch_chans,
            batch_times,
            valid_eps,
            valid_by_waveform=valid_by_waveform,
        )
        for local_sample_idx, beat_index in np.argwhere(batch_valid):
            code_id = int(batch_codes[local_sample_idx, beat_index])
            if code_id not in stats:
                continue
            waveform = np.asarray(batch_data[local_sample_idx, beat_index], dtype=np.float64)
            item = stats[code_id]
            item["count"] += 1
            item["sum"] = waveform.copy() if item["sum"] is None else item["sum"] + waveform
            item["sumsq"] = waveform ** 2 if item["sumsq"] is None else item["sumsq"] + waveform ** 2
            examples = item["examples"]
            if len(examples) < overlay_limit:
                examples.append(waveform.astype(np.float32))
            else:
                replace_idx = rng.randrange(item["count"])
                if replace_idx < overlay_limit:
                    examples[replace_idx] = waveform.astype(np.float32)
    return stats


def sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def plot_single_prototype(code_id: int, item: dict, output_path: Path, title_prefix: str = "Code"):
    count = item["count"]
    if count <= 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean = item["sum"] / count
    variance = np.maximum(item["sumsq"] / count - mean ** 2, 0.0)
    std = np.sqrt(variance)
    x = np.arange(mean.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4))
    for waveform in item["examples"]:
        ax.plot(x, waveform, color="tab:gray", alpha=0.25, linewidth=0.8)
    ax.plot(x, mean, color="tab:blue", linewidth=2.0, label="mean")
    ax.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.2, label="mean +/- std")
    ax.set_title(f"{title_prefix} {code_id} prototype waveform (n={count})")
    ax.set_xlabel("Token sample index")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_frequent_prototypes(prototype_stats: dict[int, dict], code_ids: Iterable[int], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for code_id in code_ids:
        if int(code_id) not in prototype_stats:
            continue
        plot_single_prototype(
            int(code_id),
            prototype_stats[int(code_id)],
            output_dir / f"code_{int(code_id)}.png",
        )


def associated_code_requests(sample_assoc: pd.DataFrame, top_n: int) -> list[tuple[str, int]]:
    if sample_assoc.empty:
        return []
    requests = []
    for target, group in sample_assoc.groupby("target_name", sort=False):
        top_group = group[group["diff"] > 0].sort_values("diff", ascending=False).head(top_n)
        for code_id in top_group["code_id"].tolist():
            requests.append((str(target), int(code_id)))
    return requests


def plot_associated_prototypes(
    prototype_stats: dict[int, dict],
    requests: list[tuple[str, int]],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, code_id in requests:
        if code_id not in prototype_stats:
            continue
        plot_single_prototype(
            code_id,
            prototype_stats[code_id],
            output_dir / f"{sanitize_filename(target)}_code_{code_id}.png",
            title_prefix=f"{target} associated code",
        )


def code_label_association(assignments: pd.DataFrame, label_cols: list[str], codebook_size: int) -> pd.DataFrame:
    columns = [
        "target_name",
        "code_id",
        "log_odds",
        "count_high",
        "count_low",
        "freq_high",
        "freq_low",
    ]
    rows = []
    if assignments.empty:
        return pd.DataFrame(columns=columns)
    code_ids = assignments["code_id"].to_numpy(dtype=np.int64)
    for target in label_cols:
        if target not in assignments.columns:
            continue
        values = pd.to_numeric(assignments[target], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        high = valid & (values >= 50)
        low = valid & (values < 50)
        high_total = int(high.sum())
        low_total = int(low.sum())
        if high_total == 0 or low_total == 0:
            continue
        high_counts = np.bincount(code_ids[high], minlength=codebook_size).astype(np.float64)
        low_counts = np.bincount(code_ids[low], minlength=codebook_size).astype(np.float64)
        for code_id in np.where((high_counts + low_counts) > 0)[0]:
            count_high = high_counts[code_id]
            count_low = low_counts[code_id]
            log_odds = math.log((count_high + 0.5) / (high_total - count_high + 0.5)) - math.log(
                (count_low + 0.5) / (low_total - count_low + 0.5)
            )
            rows.append(
                {
                    "target_name": target,
                    "code_id": int(code_id),
                    "log_odds": float(log_odds),
                    "count_high": int(count_high),
                    "count_low": int(count_low),
                    "freq_high": float(count_high / high_total),
                    "freq_low": float(count_low / low_total),
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(["target_name", "log_odds"], ascending=[True, False])
    return result


def code_label_association_sample_level(
    assignments: pd.DataFrame,
    labels: pd.DataFrame,
    label_cols: list[str],
    codebook_size: int,
) -> pd.DataFrame:
    columns = [
        "target_name",
        "code_id",
        "mean_freq_high",
        "mean_freq_low",
        "diff",
        "log_ratio",
        "n_high_samples",
        "n_low_samples",
    ]
    if assignments.empty or not label_cols:
        return pd.DataFrame(columns=columns)

    sample_ids = np.array(sorted(assignments["sample_id"].unique()), dtype=np.int64)
    if sample_ids.size == 0:
        return pd.DataFrame(columns=columns)

    sample_to_pos = {int(sample_id): pos for pos, sample_id in enumerate(sample_ids)}
    hist = np.zeros((len(sample_ids), codebook_size), dtype=np.float32)
    token_counts = np.zeros(len(sample_ids), dtype=np.float32)

    for sample_id, code_id in assignments[["sample_id", "code_id"]].itertuples(index=False):
        pos = sample_to_pos[int(sample_id)]
        code_id = int(code_id)
        if 0 <= code_id < codebook_size:
            hist[pos, code_id] += 1.0
            token_counts[pos] += 1.0

    valid_samples = token_counts > 0
    hist[valid_samples] = hist[valid_samples] / token_counts[valid_samples, None]

    rows = []
    eps = 1e-12
    label_subset = labels.reindex(sample_ids)
    for target in label_cols:
        if target not in label_subset.columns:
            continue
        values = pd.to_numeric(label_subset[target], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values) & valid_samples
        high = finite & (values >= 50)
        low = finite & (values < 50)
        n_high = int(high.sum())
        n_low = int(low.sum())
        if n_high == 0 or n_low == 0:
            continue

        mean_high = hist[high].mean(axis=0)
        mean_low = hist[low].mean(axis=0)
        for code_id in range(codebook_size):
            high_freq = float(mean_high[code_id])
            low_freq = float(mean_low[code_id])
            rows.append(
                {
                    "target_name": target,
                    "code_id": int(code_id),
                    "mean_freq_high": high_freq,
                    "mean_freq_low": low_freq,
                    "diff": float(high_freq - low_freq),
                    "log_ratio": float(math.log((high_freq + eps) / (low_freq + eps))),
                    "n_high_samples": n_high,
                    "n_low_samples": n_low,
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(["target_name", "diff"], ascending=[True, False])
    return result


def get_codebook_weights(model) -> Optional[np.ndarray]:
    weight = None
    if hasattr(model, "quantize") and hasattr(model.quantize, "embedding"):
        embedding = model.quantize.embedding
        if hasattr(embedding, "weight"):
            weight = embedding.weight
    if weight is None:
        return None
    return weight.detach().cpu().numpy()


def codebook_geometry_summary(
    weights: np.ndarray,
    code_counts: np.ndarray,
    output_dir: Path,
    pairwise_sample_size: int,
    seed: int,
) -> dict:
    if weights is None or len(weights) == 0:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "codebook_vectors.npy", weights.astype(np.float32, copy=False))

    norms = np.linalg.norm(weights, axis=1)
    summary = {
        "num_codes": int(weights.shape[0]),
        "code_dim": int(weights.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }

    rng = np.random.default_rng(seed)
    sample_size = min(int(pairwise_sample_size), int(weights.shape[0]))
    if sample_size >= 2:
        active_ids = np.where(code_counts > 0)[0]
        if active_ids.size >= sample_size:
            sample_ids = rng.choice(active_ids, size=sample_size, replace=False)
            sample_source = "active_codes"
        else:
            sample_ids = rng.choice(weights.shape[0], size=sample_size, replace=False)
            sample_source = "all_codes"
        sample = weights[sample_ids].astype(np.float64)
        sample = sample / np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12)
        sim = sample @ sample.T
        tri = sim[np.triu_indices(sample_size, k=1)]
        summary.update(
            {
                "pairwise_sample_size": int(sample_size),
                "pairwise_sample_source": sample_source,
                "cosine_mean": float(tri.mean()),
                "cosine_std": float(tri.std()),
                "cosine_p01": float(np.percentile(tri, 1)),
                "cosine_p50": float(np.percentile(tri, 50)),
                "cosine_p95": float(np.percentile(tri, 95)),
                "cosine_p99": float(np.percentile(tri, 99)),
                "cosine_max": float(tri.max()),
            }
        )

    with open(output_dir / "codebook_geometry.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def plot_codebook_pca(weights: np.ndarray, code_counts: np.ndarray, lead_code_counts: np.ndarray, output_path: Path):
    if weights is None or len(weights) == 0:
        return
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(weights)
    dominant_lead_idx = np.argmax(lead_code_counts, axis=0)
    used = code_counts > 0

    fig, ax = plt.subplots(figsize=(7, 6))
    sizes = 8 + 40 * np.log1p(code_counts) / max(float(np.log1p(code_counts.max())), 1.0)
    ax.scatter(
        coords[~used, 0],
        coords[~used, 1],
        c="lightgray",
        s=6,
        alpha=0.15,
        linewidths=0,
        label="unused",
    )
    scatter = None
    if used.any():
        scatter = ax.scatter(
            coords[used, 0],
            coords[used, 1],
            c=dominant_lead_idx[used],
            s=sizes[used],
            alpha=0.75,
            cmap="tab20",
            linewidths=0,
            label="used",
        )
    ax.set_title("Codebook embedding PCA (color = dominant lead, size = usage)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(len(LEAD_ORDER)))
        cbar.ax.set_yticklabels(LEAD_ORDER)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_run_readme(output_dir: Path, args: argparse.Namespace, checkpoint_path: Path):
    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Codebook Audit Outputs",
                "",
                "Generated by `scripts/audit_codebook.py`.",
                "",
                "## Re-run",
                "",
                "```bash",
                "python scripts/audit_codebook.py \\",
                f"  --checkpoint_path {checkpoint_path} \\",
                f"  --data_path {args.data_path} \\",
                f"  --metadata_path {args.metadata_path if args.metadata_path else args.data_path / (args.split + '_metadata.csv')} \\",
                f"  --split {args.split} \\",
                f"  --output_dir {args.output_dir} \\",
                f"  --max_samples {args.max_samples} \\",
                f"  --codebook_size {args.codebook_size} \\",
                f"  --device {args.device}",
                "```",
                "",
                "Key files: `token_assignments.csv`, `codebook_usage.json`, heatmaps,",
                "`ltcs_report_all.json`, optional `ltcs_report_high_stenosis.json`,",
                "`code_label_association_token_level.csv`,",
                "`code_label_association_sample_level.csv`, `top_codes_per_lead.csv`,",
                "`reconstruction_summary.json`, `reconstruction_by_code.csv`,",
                "`reconstruction_by_lead.csv`, prototype waveforms,",
                "`codebook_geometry.json`, `codebook_vectors.npy`, and `codebook_pca.png`.",
            ]
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model, checkpoint_path = load_tokenizer(
        args.checkpoint_path,
        args.codebook_size,
        args.codebook_dim,
        device,
    )
    data, in_chans, in_times = load_split_arrays(args.data_path, args.split)
    metadata = load_metadata(args.data_path, args.split, args.metadata_path, len(data))
    labels, label_cols = get_label_frame(args.data_path, args.split, metadata, len(data))

    assignments, code_counts, lead_code_counts, used_samples = extract_assignments(
        model=model,
        data=data,
        in_chans=in_chans,
        in_times=in_times,
        labels=labels,
        label_cols=label_cols,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        codebook_size=args.codebook_size,
        device=device,
        valid_eps=args.valid_eps,
        valid_by_waveform=args.valid_by_waveform,
    )
    assignments.to_csv(args.output_dir / "token_assignments.csv", index=False)

    usage = usage_summary(code_counts, args.codebook_size)
    usage["checkpoint_path"] = str(checkpoint_path)
    usage["data_path"] = str(args.data_path)
    usage["split"] = args.split
    usage["used_samples"] = int(used_samples)
    with open(args.output_dir / "codebook_usage.json", "w", encoding="utf-8") as handle:
        json.dump(usage, handle, indent=2, ensure_ascii=False)

    reconstruction = None
    if not args.skip_reconstruction:
        reconstruction_max_samples = (
            args.reconstruction_max_samples
            if args.reconstruction_max_samples > 0
            else args.max_samples
        )
        reconstruction = reconstruction_quality(
            model=model,
            data=data,
            in_chans=in_chans,
            in_times=in_times,
            max_samples=reconstruction_max_samples,
            batch_size=args.batch_size,
            codebook_size=args.codebook_size,
            device=device,
            valid_eps=args.valid_eps,
            valid_by_waveform=args.valid_by_waveform,
            output_dir=args.output_dir,
        )

    plot_usage_bar(code_counts, args.output_dir / "codebook_usage_bar.png")

    lead_distribution = row_normalize(lead_code_counts)
    if not args.skip_full_heatmap:
        plot_lead_code_heatmap(
            lead_distribution,
            np.arange(args.codebook_size),
            args.output_dir / "lead_code_heatmap.png",
            "Lead-code distribution, full codebook",
        )

    top_code_ids = np.argsort(code_counts)[::-1][: min(args.top_heatmap_codes, args.codebook_size)]
    plot_lead_code_heatmap(
        lead_distribution[:, top_code_ids],
        top_code_ids,
        args.output_dir / "lead_code_heatmap_top200.png",
        f"Lead-code distribution, top {len(top_code_ids)} codes",
    )

    similarity = cosine_similarity_matrix(lead_distribution)
    plot_similarity(similarity, args.output_dir / "lead_similarity_heatmap.png")
    save_lead_similarity_tables(lead_distribution, args.output_dir, args.topk_overlap)

    ltcs = ltcs_report(similarity)
    with open(args.output_dir / "ltcs_report_all.json", "w", encoding="utf-8") as handle:
        json.dump(ltcs, handle, indent=2, ensure_ascii=False)
    with open(args.output_dir / "ltcs_report.json", "w", encoding="utf-8") as handle:
        json.dump(ltcs, handle, indent=2, ensure_ascii=False)

    high_ltcs = None
    high_sample_ids = high_stenosis_sample_ids(labels, label_cols)
    if high_sample_ids:
        assignment_sample_ids = set(int(sample_id) for sample_id in assignments["sample_id"].unique())
        high_sample_ids = high_sample_ids & assignment_sample_ids
    if high_sample_ids:
        high_lead_code_counts = lead_code_counts_from_assignments(
            assignments,
            args.codebook_size,
            sample_ids=high_sample_ids,
        )
        high_distribution = row_normalize(high_lead_code_counts)
        high_ltcs = ltcs_report(cosine_similarity_matrix(high_distribution))
        high_ltcs["high_stenosis_sample_count"] = int(len(high_sample_ids))
        with open(args.output_dir / "ltcs_report_high_stenosis.json", "w", encoding="utf-8") as handle:
            json.dump(high_ltcs, handle, indent=2, ensure_ascii=False)
    else:
        print("No high-stenosis labels available; skipping ltcs_report_high_stenosis.json.")

    save_top_codes_per_lead(lead_code_counts, args.output_dir / "top_codes_per_lead.csv")

    token_association = code_label_association(assignments, label_cols, args.codebook_size)
    token_association.to_csv(args.output_dir / "code_label_association_token_level.csv", index=False)

    sample_association = code_label_association_sample_level(
        assignments,
        labels,
        label_cols,
        args.codebook_size,
    )
    sample_association.to_csv(args.output_dir / "code_label_association_sample_level.csv", index=False)

    frequent_code_ids = [
        int(code_id)
        for code_id in np.argsort(code_counts)[::-1][: min(args.top_prototype_codes, args.codebook_size)]
        if code_counts[code_id] > 0
    ]
    associated_requests = associated_code_requests(sample_association, args.top_associated_codes)
    prototype_code_ids = set(frequent_code_ids)
    prototype_code_ids.update(code_id for _, code_id in associated_requests)
    if prototype_code_ids:
        prototype_stats = collect_prototypes(
            model=model,
            data=data,
            in_chans=in_chans,
            in_times=in_times,
            target_codes=prototype_code_ids,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device=device,
            valid_eps=args.valid_eps,
            valid_by_waveform=args.valid_by_waveform,
            overlay_limit=args.prototype_overlays,
            seed=args.seed,
        )
        plot_frequent_prototypes(
            prototype_stats,
            frequent_code_ids,
            args.output_dir / "prototype_waveforms" / "frequent",
        )
        plot_associated_prototypes(
            prototype_stats,
            associated_requests,
            args.output_dir / "prototype_waveforms" / "associated",
        )

    weights = get_codebook_weights(model)
    geometry = None
    if weights is not None:
        geometry = codebook_geometry_summary(
            weights,
            code_counts,
            args.output_dir,
            pairwise_sample_size=args.pairwise_sample_size,
            seed=args.seed,
        )
        plot_codebook_pca(weights, code_counts, lead_code_counts, args.output_dir / "codebook_pca.png")

    write_run_readme(args.output_dir, args, checkpoint_path)

    collapse_warning = usage["active_ratio"] < 0.05 or usage["top_1_percent_usage"] > 0.8
    print("\nCodebook audit summary")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  split: {args.split}, samples: {used_samples}, valid tokens: {usage['total_tokens']}")
    print(f"  active_ratio: {usage['active_ratio']:.6f}")
    print(f"  dead_ratio: {usage['dead_ratio']:.6f}")
    print(f"  perplexity: {usage['perplexity']:.3f}")
    print(f"  top_1_percent_usage: {usage['top_1_percent_usage']:.6f}")
    if reconstruction is not None:
        print(f"  reconstruction_mse: {reconstruction['mse']:.6f}")
        print(f"  reconstruction_rmse: {reconstruction['rmse']:.6f}")
        print(f"  reconstruction_mae: {reconstruction['mae']:.6f}")
    if geometry:
        print(f"  codebook_norm_mean: {geometry['norm_mean']:.6f}")
        if "cosine_p95" in geometry:
            print(f"  codebook_cosine_p95: {geometry['cosine_p95']:.6f}")
    print(f"  LTCS_all: {ltcs['LTCS']:.6f}")
    if high_ltcs is not None:
        print(f"  LTCS_high_stenosis: {high_ltcs['LTCS']:.6f}")
    if collapse_warning:
        print("  WARNING: possible codebook collapse detected.")
    if usage["active_ratio"] < 0.05:
        print("  WARNING: active_ratio < 0.05.")
    if usage["top_1_percent_usage"] > 0.8:
        print("  WARNING: top_1_percent_usage > 0.8.")
    if ltcs["LTCS"] <= 0:
        print("  WARNING: LTCS_all <= 0; lead-territory consistency is not positive.")
    if high_ltcs is not None and high_ltcs["LTCS"] <= 0:
        print("  WARNING: LTCS_high_stenosis <= 0; high-stenosis lead-territory consistency is not positive.")
    if sample_association.empty:
        print("  WARNING: sample-level association file is empty.")
    print(f"  outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
