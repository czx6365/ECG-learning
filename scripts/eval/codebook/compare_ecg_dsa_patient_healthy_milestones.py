#!/usr/bin/env python3
"""Collect comparable codebook metrics for selected VQ-HBR checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare VQ-HBR codebook milestones.")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v1_milestones"),
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v1"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/codebook_eval/ECG_DSA_patient_healthy_raw_v1_milestones"),
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=3832)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_eval(args: argparse.Namespace, epoch: int, checkpoint_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (output_dir / "codebook_usage.json").exists():
        return
    cmd = [
        args.python_bin,
        "scripts/audit_codebook.py",
        "--checkpoint_path",
        str(checkpoint_path),
        "--data_path",
        str(args.data_path),
        "--split",
        args.split,
        "--output_dir",
        str(output_dir),
        "--max_samples",
        str(args.max_samples),
        "--reconstruction_max_samples",
        str(args.max_samples),
        "--batch_size",
        str(args.batch_size),
        "--codebook_size",
        "8192",
        "--codebook_dim",
        "128",
        "--device",
        args.device,
        "--top_heatmap_codes",
        "200",
        "--top_prototype_codes",
        "20",
        "--top_associated_codes",
        "20",
        "--prototype_overlays",
        "20",
        "--pairwise_sample_size",
        "2048",
    ]
    print(f"\n>>> Evaluating epoch {epoch}: {checkpoint_path}")
    subprocess.run(cmd, check=True)


def summarize_epoch(epoch: int, output_dir: Path) -> dict:
    usage = load_json(output_dir / "codebook_usage.json")
    rec = load_json(output_dir / "reconstruction_summary.json")
    geom = load_json(output_dir / "codebook_geometry.json")
    ltcs = load_json(output_dir / "ltcs_report_all.json")
    return {
        "epoch": epoch,
        "valid_tokens": usage["total_tokens"],
        "active_codes": usage["active_codes"],
        "active_ratio": usage["active_ratio"],
        "dead_codes": usage["dead_codes"],
        "dead_ratio": usage["dead_ratio"],
        "perplexity": usage["perplexity"],
        "top_1_percent_usage": usage["top_1_percent_usage"],
        "reconstruction_mse": rec["mse"],
        "reconstruction_rmse": rec["rmse"],
        "reconstruction_mae": rec["mae"],
        "sample_mse_p50": rec.get("sample_mse_p50"),
        "sample_mse_p90": rec.get("sample_mse_p90"),
        "sample_mse_p95": rec.get("sample_mse_p95"),
        "ltcs_all": ltcs["LTCS"],
        "codebook_norm_mean": geom["norm_mean"],
        "codebook_cosine_mean": geom.get("cosine_mean"),
        "codebook_cosine_p95": geom.get("cosine_p95"),
        "codebook_cosine_max": geom.get("cosine_max"),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []

    for epoch in args.epochs:
        checkpoint_path = args.checkpoint_dir / f"checkpoint-{epoch}.pth"
        output_dir = args.output_root / f"epoch{epoch}" / args.split
        if not checkpoint_path.exists():
            missing.append(str(checkpoint_path))
            continue
        run_eval(args, epoch, checkpoint_path, output_dir)
        rows.append(summarize_epoch(epoch, output_dir))

    if rows:
        summary = pd.DataFrame(rows).sort_values("epoch")
        summary_path = args.output_root / f"milestone_summary_{args.split}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nWrote {summary_path}")
        print(summary.to_string(index=False))

    if missing:
        print("\nMissing checkpoints:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
