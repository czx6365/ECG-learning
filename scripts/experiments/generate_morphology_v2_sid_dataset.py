#!/usr/bin/env python3
"""Generate Morphology-V2 SID arrays for pretraining or downstream datasets."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from train_morphology_v2_physaware_vq import MorphologyV2PhysAwareVQ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate SID dataset with frozen Morphology-V2 tokenizer.")
    parser.add_argument("--source_data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tokenizer_checkpoint", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--token_len", type=int, default=96)
    parser.add_argument("--pad_sid", type=int, default=1024)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy_context", action="store_true", default=True)
    parser.add_argument("--symlink_context", action="store_true")
    return parser.parse_args()


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def copy_or_link(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def load_tokenizer(args: argparse.Namespace, device: torch.device):
    patch_checkpoint_paths()
    ckpt_path = resolve_path(args.tokenizer_checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = MorphologyV2PhysAwareVQ(
        codebook_size=int(ckpt_args.get("codebook_n_emd", args.codebook_size)),
        codebook_dim=int(ckpt_args.get("codebook_emd_dim", args.codebook_dim)),
        token_len=int(ckpt_args.get("token_len", args.token_len)),
        ema_decay=float(ckpt_args.get("ema_decay", 0.99)),
        quantize_kmeans_init=False,
        phys_head_source=str(ckpt_args.get("phys_head_source", "z_q")),
        continuous_mean=ckpt.get("continuous_mean"),
        continuous_std=ckpt.get("continuous_std"),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    actual_size = model.quantize.num_tokens
    actual_dim = model.quantize.codebook_dim
    if actual_size != args.codebook_size or actual_dim != args.codebook_dim:
        raise ValueError(
            f"Tokenizer checkpoint has codebook {actual_size}x{actual_dim}, "
            f"expected {args.codebook_size}x{args.codebook_dim}."
        )
    print(f"Loaded Morphology-V2 tokenizer from {ckpt_path}")
    print(f"Tokenizer codebook_size={actual_size}, codebook_dim={actual_dim}")
    return model, ckpt_path


def generate_split(args, model, source_dir: Path, output_dir: Path, split: str, device: torch.device) -> dict:
    data_path = source_dir / f"{split}_data.npy"
    chans_path = source_dir / f"{split}_data_in_chans.npy"
    times_path = source_dir / f"{split}_data_in_times.npy"
    if not data_path.exists():
        print(f"Skip split={split}: missing {data_path}")
        return {"split": split, "skipped": True}
    data = np.load(data_path, mmap_mode="r")
    chans = np.load(chans_path, mmap_mode="r")
    times = np.load(times_path, mmap_mode="r")
    n_records = data.shape[0]
    if args.max_records > 0:
        n_records = min(n_records, args.max_records)
    if data.shape[-1] != args.token_len:
        raise ValueError(f"{data_path} token_len={data.shape[-1]}, expected {args.token_len}.")

    sid_path = output_dir / f"{split}_sid.npy"
    if sid_path.exists() and not args.overwrite:
        print(f"Reuse existing {sid_path}")
    else:
        sid = np.lib.format.open_memmap(
            sid_path,
            mode="w+",
            dtype=np.int16,
            shape=(n_records, data.shape[1]),
        )
        total_valid = 0
        for start in range(0, n_records, args.batch_size):
            end = min(start + args.batch_size, n_records)
            batch_data = np.asarray(data[start:end], dtype=np.float32)
            batch_chans = np.asarray(chans[start:end])
            batch_times = np.asarray(times[start:end])
            valid = (
                (batch_chans > 0)
                & (batch_times > 0)
                & (np.abs(batch_data).max(axis=-1) > 1e-8)
            )
            x = torch.from_numpy(batch_data.copy()).to(device, non_blocking=True)
            in_chans = torch.from_numpy(batch_chans.copy()).to(device, non_blocking=True)
            in_times = torch.from_numpy(batch_times.copy()).to(device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    ids = model.get_codebook_indices(
                        x,
                        in_chan_matrix=in_chans,
                        in_time_matrix=in_times,
                    )
            ids_np = ids.detach().cpu().numpy().astype(np.int16, copy=False)
            ids_np[~valid] = np.int16(args.pad_sid)
            sid[start:end] = ids_np
            total_valid += int(valid.sum())
            if start == 0 or end == n_records or (start // args.batch_size) % 20 == 0:
                print(f"{split}: generated {end}/{n_records}; valid_tokens={total_valid}", flush=True)
        sid.flush()

    linked_files = []
    if args.copy_context:
        for suffix in (
            "data_in_chans.npy",
            "data_in_times.npy",
            "labels.npy",
            "aux_regression_labels.npy",
            "metadata.csv",
        ):
            src = source_dir / f"{split}_{suffix}"
            if src.exists():
                dst = output_dir / src.name
                copy_or_link(src, dst, symlink=args.symlink_context)
                linked_files.append(dst.name)

    sid_arr = np.load(sid_path, mmap_mode="r")
    valid_sid = (sid_arr >= 0) & (sid_arr < args.codebook_size)
    return {
        "split": split,
        "sid_file": str(sid_path),
        "shape": list(sid_arr.shape),
        "valid_sid_tokens": int(valid_sid.sum()),
        "pad_tokens": int((sid_arr == args.pad_sid).sum()),
        "active_sid": int(np.unique(sid_arr[valid_sid]).size) if valid_sid.any() else 0,
        "context_files": linked_files,
    }


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model, ckpt_path = load_tokenizer(args, device)

    summaries = []
    for split in args.splits:
        summaries.append(generate_split(args, model, source_dir, output_dir, split, device))

    config = {
        "source_data_dir": str(source_dir),
        "output_dir": str(output_dir),
        "tokenizer_checkpoint": str(ckpt_path),
        "codebook_size": args.codebook_size,
        "codebook_dim": args.codebook_dim,
        "pad_sid": args.pad_sid,
        "splits": summaries,
        "principle": "Morphology-V2 SID assignment uses heartbeat waveform only; lead/time are padding masks only.",
    }
    (output_dir / "sid_dataset_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
