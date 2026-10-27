#!/usr/bin/env python3
"""Load the published E3 + E12-A pair and extract trainable ECG features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments")]

from heartlang.models.modeling_e12a import load_e12a
from train_morphology_v2_physaware_vq import MorphologyV2PhysAwareVQ


DEFAULT_E3 = ROOT / "checkpoints/vqhbr/E3_morphology_encoder_codebook/checkpoint-best.pth"
DEFAULT_E12 = ROOT / "checkpoints/pretrain/E12-A_continuous_ze_sid_mlm/checkpoint-best.pth"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_e3(path: Path, device: torch.device) -> MorphologyV2PhysAwareVQ:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    model = MorphologyV2PhysAwareVQ(
        codebook_size=state["quantize.embedding.weight"].shape[0],
        codebook_dim=64,
        token_len=96,
        ema_decay=0.99,
        phys_head_source="z_q",
        continuous_mean=payload["continuous_mean"],
        continuous_std=payload["continuous_std"],
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_input(path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path is None:
        # A structural smoke input: eight valid beats followed by padding.
        rng = np.random.default_rng(0)
        wave = np.zeros((2, 256, 96), dtype=np.float32)
        lead = np.zeros((2, 256), dtype=np.int64)
        time = np.zeros((2, 256), dtype=np.int64)
        wave[:, :8] = rng.normal(size=(2, 8, 96)).astype(np.float32)
        lead[:, :8] = np.arange(1, 9)
        time[:, :8] = np.arange(1, 9)
        return wave, lead, time
    arrays = np.load(path)
    return arrays["waveforms"], arrays["lead_ids"], arrays["time_ids"]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="NPZ containing waveforms, lead_ids, time_ids")
    parser.add_argument("--output", type=Path, default=Path("e12a_features.npy"))
    parser.add_argument("--e3-checkpoint", type=Path, default=DEFAULT_E3)
    parser.add_argument("--e12-checkpoint", type=Path, default=DEFAULT_E12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    e3 = load_e3(args.e3_checkpoint, device)
    e12 = load_e12a(args.e12_checkpoint, device)
    wave, lead, time = load_input(args.input)
    if wave.shape[1:] != (256, 96) or lead.shape != wave.shape[:2] or time.shape != wave.shape[:2]:
        raise ValueError("Expected waveforms [N,256,96], lead_ids/time_ids [N,256]")

    outputs = []
    valid_counts = []
    for start in range(0, len(wave), args.batch_size):
        end = min(start + args.batch_size, len(wave))
        x = torch.from_numpy(np.asarray(wave[start:end])).float().to(device)
        c = torch.from_numpy(np.asarray(lead[start:end])).long().to(device)
        t = torch.from_numpy(np.asarray(time[start:end])).long().to(device)
        valid = e3.valid_mask(x, c, t)
        ze = e3.encode_to_quantizer_features(x, c, t).float()
        outputs.append(e12.pooled_mean_max(ze, valid).cpu().numpy())
        valid_counts.extend(valid.sum(1).cpu().tolist())

    features = np.concatenate(outputs).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features)
    report = {
        "records": len(features),
        "feature_shape": list(features.shape),
        "valid_heartbeats_min_max": [int(min(valid_counts)), int(max(valid_counts))],
        "finite": bool(np.isfinite(features).all()),
        "e3_sha256": sha256(args.e3_checkpoint),
        "e12_sha256": sha256(args.e12_checkpoint),
        "output": str(args.output),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
