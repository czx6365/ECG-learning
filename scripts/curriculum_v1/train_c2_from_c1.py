#!/usr/bin/env python3
"""Curriculum V1 C2: refine a frozen C1 starting point with morphology patterns."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import train_morphology_v2_physaware_vq as base


INDEX_DIR = Path(os.environ["C2_INDEX_DIR"])
INIT_CHECKPOINT = Path(os.environ["C2_INIT_CHECKPOINT"])
_original_build_model = base.build_model
_original_write_config = base.write_config


class IndexedPhysTargetDataset(Dataset):
    SOURCE_SPLITS = ("train", "val")

    def __init__(self, data_dir: Path, cache_dir: Path, index_dir: Path, split: str, lead_id: int = 0):
        self.data = {name: np.load(data_dir / f"{name}_data.npy", mmap_mode="r") for name in self.SOURCE_SPLITS}
        self.chans = {name: np.load(data_dir / f"{name}_data_in_chans.npy", mmap_mode="r") for name in self.SOURCE_SPLITS}
        self.times = {name: np.load(data_dir / f"{name}_data_in_times.npy", mmap_mode="r") for name in self.SOURCE_SPLITS}
        self.discrete = {
            name: np.load(cache_dir / f"{name}_phys_discrete.npy", mmap_mode="r")
            for name in self.SOURCE_SPLITS
        }
        self.continuous = {
            name: np.load(cache_dir / f"{name}_phys_continuous.npy", mmap_mode="r")
            for name in self.SOURCE_SPLITS
        }
        self.continuous_mask = {
            name: np.load(cache_dir / f"{name}_phys_continuous_mask.npy", mmap_mode="r")
            for name in self.SOURCE_SPLITS
        }
        self.source_code = np.load(index_dir / f"{split}_source_code.npy", mmap_mode="r")
        self.source_index = np.load(index_dir / f"{split}_source_index.npy", mmap_mode="r")
        self.lead_id = int(lead_id)
        if len(self.source_code) != len(self.source_index):
            raise RuntimeError(f"C2 index length mismatch for {split}")

    def __len__(self):
        return len(self.source_index)

    def __getitem__(self, idx: int):
        source = self.SOURCE_SPLITS[int(self.source_code[idx])]
        row = int(self.source_index[idx])
        chans = np.array(self.chans[source][row], copy=True)
        times = np.array(self.times[source][row], copy=True)
        if self.lead_id > 0:
            selected = chans == self.lead_id
            chans[~selected] = 0
            times[~selected] = 0
        return (
            np.array(self.data[source][row], copy=True),
            chans,
            times,
            np.array(self.discrete[source][row], copy=True),
            np.array(self.continuous[source][row], copy=True)[..., base.CONTINUOUS_CACHE_INDICES],
            np.array(self.continuous_mask[source][row], copy=True)[..., base.CONTINUOUS_CACHE_INDICES],
        )


def build_loaders(args, data_dir: Path, cache_dir: Path):
    index_dir = base.resolve_path(INDEX_DIR)
    train = IndexedPhysTargetDataset(data_dir, cache_dir, index_dir, "train", lead_id=args.lead_id)
    val = IndexedPhysTargetDataset(data_dir, cache_dir, index_dir, "val", lead_id=args.lead_id)
    sampler = torch.utils.data.DistributedSampler(train, num_replicas=1, rank=0, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val,
        sampler=torch.utils.data.SequentialSampler(val),
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return train, val, sampler, train_loader, val_loader


def build_model(args, continuous_stats: dict):
    model = _original_build_model(args, continuous_stats)
    checkpoint_path = base.resolve_path(INIT_CHECKPOINT)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model"]
    target = model.state_dict()
    shared = {
        key: value
        for key, value in source.items()
        if key in target
        and target[key].shape == value.shape
        and key.startswith(("encoder.", "decoder.", "quantize."))
    }
    result = model.load_state_dict(shared, strict=False)
    required_prefixes = ("encoder.", "decoder.", "quantize.")
    missing_shared = [key for key in target if key.startswith(required_prefixes) and key not in shared]
    if missing_shared:
        raise RuntimeError(f"C1 initialization missed shared parameters: {missing_shared[:10]}")
    print(
        f"Initialized C2 from C1 checkpoint {checkpoint_path}; "
        f"loaded_keys={len(shared)} fresh_keys={len(result.missing_keys)}",
        flush=True,
    )
    return model


def loss_weights(args, epoch: int | None = None) -> dict:
    weights = {
        "lambda_waveform": float(args.lambda_waveform),
        "lambda_corr": float(args.lambda_corr),
        "lambda_deriv": float(args.lambda_deriv),
        "lambda_vq": float(args.lambda_vq),
        "lambda_phys": float(args.lambda_phys),
        "discrete_alpha": float(args.discrete_alpha),
        "continuous_alpha": float(args.continuous_alpha),
        "continuous_clip_z": float(args.continuous_clip_z),
        "continuous_loss_reduction": args.continuous_loss_reduction,
        "lambda_feature_contrast": 0.0,
        "lambda_usage_diversity": 0.0,
        "feature_contrast_samples": args.feature_contrast_samples,
        "feature_contrast_k": args.feature_contrast_k,
        "feature_contrast_temperature": args.feature_contrast_temperature,
        "usage_diversity_samples": args.usage_diversity_samples,
        "usage_diversity_temperature": args.usage_diversity_temperature,
    }
    if epoch is None:
        weights["stage"] = "c2_from_c1"
    elif epoch <= 5:
        factor = epoch / 5.0
        weights["lambda_corr"] *= factor
        weights["lambda_deriv"] *= factor
        weights["lambda_phys"] = 0.0
        weights["stage"] = (
            "shape_ramp"
            if float(args.lambda_corr) != 0.0 or float(args.lambda_deriv) != 0.0
            else "base_hold"
        )
    elif epoch <= 15:
        weights["lambda_phys"] *= (epoch - 5) / 10.0
        weights["stage"] = "pattern_ramp"
    else:
        weights["stage"] = "pattern_full"
    return weights


def write_config(args, data_dir, cache_dir, train_len, val_len, continuous_stats, base_lr):
    _original_write_config(args, data_dir, cache_dir, train_len, val_len, continuous_stats, base_lr)
    path = args.output_dir / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    uses_shape_losses = float(args.lambda_corr) != 0.0 or float(args.lambda_deriv) != 0.0
    config.update(
        {
            "experiment": "curriculum_v1_c2_from_c1",
            "stage": "C2",
            "init_c1_checkpoint": str(base.resolve_path(INIT_CHECKPOINT)),
            "split_index_dir": str(base.resolve_path(INDEX_DIR)),
            "loss_schedule": "c2_from_c1",
            "c2_stages": [
                {
                    "epochs": "1-5",
                    "waveform_vq": "full",
                    "corr_deriv": "linear ramp" if uses_shape_losses else 0.0,
                    "discrete_pattern": 0.0,
                },
                {
                    "epochs": "6-15",
                    "waveform_vq": "full",
                    "corr_deriv": "full" if uses_shape_losses else 0.0,
                    "discrete_pattern": "linear ramp",
                },
                {
                    "epochs": "16-50",
                    "waveform_vq": "full",
                    "corr_deriv": "full" if uses_shape_losses else 0.0,
                    "discrete_pattern": "full",
                },
            ],
            "pattern_targets": list(base.DISCRETE_FEATURES),
            "continuous_pattern_supervision": False,
            "main_control_variable": (
                "continue C1 with waveform + VQ + discrete heartbeat morphology patterns only"
                if not uses_shape_losses
                else "continue C1 and add waveform shape plus discrete heartbeat morphology patterns"
            ),
        }
    )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


base.build_loaders = build_loaders
base.build_model = build_model
base.loss_weights = loss_weights
base.write_config = write_config


if __name__ == "__main__":
    base.main()
