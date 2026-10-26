#!/usr/bin/env python3
"""C3 lead-aware SID MLM initialized from the pattern-only C2 sentence model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import train_heartlang_sid_pretraining as base


def load_c2_sentence_weights(model, optimizer, scaler, args, device):
    del optimizer, scaler
    if args.resume is None:
        raise ValueError("C3 requires --resume pointing to the best pattern-only C2 Sentence MLM checkpoint.")
    checkpoint_path = base.resolve_path(args.resume)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint["model"]
    target = model.state_dict()
    shared = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    missing = sorted(set(target) - set(shared))
    unexpected = sorted(set(source) - set(shared))
    illegal_missing = [key for key in missing if not key.startswith("backbone.spa_embed.")]
    if illegal_missing:
        raise RuntimeError(f"C3 initialization missed non-spatial parameters: {illegal_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"C3 initialization found incompatible source parameters: {unexpected[:20]}")
    result = model.load_state_dict(shared, strict=False)
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for key in missing:
            if key not in named_parameters:
                raise RuntimeError(f"Fresh C3 spatial state is not a trainable parameter: {key}")
            named_parameters[key].zero_()
    print(
        f"Initialized C3 from {checkpoint_path}; source_epoch={checkpoint.get('epoch')} "
        f"loaded_keys={len(shared)} zero_initialized_spatial_keys={missing} "
        f"load_missing={list(result.missing_keys)}",
        flush=True,
    )
    # C3 is a new curriculum stage: restart epoch/LR schedule and optimizer.
    return 0, None


base.load_checkpoint = load_c2_sentence_weights


if __name__ == "__main__":
    base.main()
