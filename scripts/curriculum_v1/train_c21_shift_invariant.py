#!/usr/bin/env python3
"""C2.1: continue pattern-only C2 with shift-invariant morphology losses."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT / "scripts" / "curriculum_v1"))

import train_c2_from_c1 as c2


base = c2.base
INIT_CHECKPOINT = Path(os.environ["C21_INIT_CHECKPOINT"])
SHIFT_LATENT_WEIGHT = float(os.environ.get("C21_SHIFT_LATENT_WEIGHT", "0.05"))
SHIFT_SID_WEIGHT = float(os.environ.get("C21_SHIFT_SID_WEIGHT", "0.05"))
SHIFT_TEMPERATURE = float(os.environ.get("C21_SHIFT_TEMPERATURE", "0.1"))
_original_write_config = c2._original_write_config
_original_evaluate = base.evaluate


def shift_radius(epoch: int | None) -> int:
    epoch = int(epoch or 1)
    if epoch <= 10:
        return 1
    if epoch <= 25:
        return 2
    return 4


def non_circular_shift(beats: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Shift each [96] beat independently, filling uncovered samples with zero."""
    shifted = torch.zeros_like(beats)
    for delta in torch.unique(deltas).tolist():
        mask = deltas == int(delta)
        if delta > 0:
            shifted[mask, delta:] = beats[mask, :-delta]
        elif delta < 0:
            amount = -int(delta)
            shifted[mask, :-amount] = beats[mask, amount:]
        else:
            shifted[mask] = beats[mask]
    return shifted


class ShiftInvariantMorphologyVQ(base.MorphologyV2PhysAwareVQ):
    def _sample_deltas(self, count: int, radius: int, device: torch.device) -> torch.Tensor:
        choices = torch.cat(
            [torch.arange(-radius, 0, device=device), torch.arange(1, radius + 1, device=device)]
        )
        if self.training:
            indices = torch.randint(0, len(choices), (count,), device=device)
        else:
            indices = torch.arange(count, device=device) % len(choices)
        return choices[indices]

    def forward(self, *args, **kwargs):
        loss_weights = kwargs.get("loss_weights") or {}
        x = args[0] if args else kwargs["x"]
        chans = kwargs.get("in_chan_matrix")
        times = kwargs.get("in_time_matrix")
        split = "train" if self.training else "val"

        captured_z = []
        captured_quant = []
        encoder_handle = self.encoder.register_forward_hook(lambda _m, _i, out: captured_z.append(out))
        quant_handle = self.quantize.register_forward_hook(lambda _m, _i, out: captured_quant.append(out))
        try:
            base_loss, log = super().forward(*args, **kwargs)
        finally:
            encoder_handle.remove()
            quant_handle.remove()

        valid = self.valid_mask(x, chans, times)
        if not valid.any() or not captured_z or not captured_quant:
            zero = base_loss * 0.0
            log[f"{split}/shift_latent_loss"] = zero.detach()
            log[f"{split}/shift_sid_loss"] = zero.detach()
            log[f"{split}/shift_sid_agreement"] = zero.detach()
            log[f"{split}/shift_radius"] = torch.as_tensor(
                float(loss_weights.get("shift_radius", 1)), device=x.device
            )
            return base_loss, log

        beats = x[valid]
        z_target = captured_z[0].detach()
        sid_target = captured_quant[0][2].detach().reshape(-1)
        radius = int(loss_weights.get("shift_radius", 1))
        deltas = self._sample_deltas(beats.shape[0], radius, beats.device)
        shifted_beats = non_circular_shift(beats, deltas)
        z_shift = self.encoder(shifted_beats)

        z_shift_float = z_shift.float()
        shift_latent_loss = 1.0 - F.cosine_similarity(
            z_shift_float, z_target.float(), dim=-1
        ).mean()
        codebook = F.normalize(self.quantize.embedding.weight.detach().float(), dim=-1)
        logits = F.normalize(z_shift_float, dim=-1) @ codebook.T
        logits = logits / max(float(loss_weights.get("shift_temperature", 0.1)), 1e-6)
        shift_sid_loss = F.cross_entropy(logits, sid_target)
        sid_shift = logits.argmax(dim=-1)
        agreement = (sid_shift == sid_target).float().mean()

        total = (
            base_loss
            + float(loss_weights.get("lambda_shift_latent", 0.05)) * shift_latent_loss
            + float(loss_weights.get("lambda_shift_sid", 0.05)) * shift_sid_loss
        )
        log[f"{split}/shift_latent_loss"] = shift_latent_loss.detach()
        log[f"{split}/shift_sid_loss"] = shift_sid_loss.detach()
        log[f"{split}/shift_sid_agreement"] = agreement.detach()
        log[f"{split}/shift_radius"] = torch.as_tensor(float(radius), device=x.device)
        log[f"{split}/total_loss"] = total.detach()
        return total, log


def build_model(args, continuous_stats: dict):
    model = ShiftInvariantMorphologyVQ(
        codebook_size=args.codebook_n_emd,
        codebook_dim=args.codebook_emd_dim,
        token_len=args.token_len,
        ema_decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
        phys_head_source=args.phys_head_source,
        continuous_mean=continuous_stats["mean"],
        continuous_std=continuous_stats["std"],
    )
    checkpoint_path = base.resolve_path(INIT_CHECKPOINT)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(checkpoint["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"C2.1 initialization mismatch: {result}")
    print(f"Initialized all C2.1 parameters from pattern-only C2: {checkpoint_path}", flush=True)
    return model


def loss_weights(args, epoch: int | None = None) -> dict:
    return {
        "lambda_waveform": 1.0,
        "lambda_corr": 0.0,
        "lambda_deriv": 0.0,
        "lambda_vq": 1.0,
        "lambda_phys": 0.2,
        "discrete_alpha": 1.0,
        "continuous_alpha": 0.0,
        "continuous_clip_z": 0.0,
        "continuous_loss_reduction": "element",
        "lambda_feature_contrast": 0.0,
        "lambda_usage_diversity": 0.0,
        "feature_contrast_samples": args.feature_contrast_samples,
        "feature_contrast_k": args.feature_contrast_k,
        "feature_contrast_temperature": args.feature_contrast_temperature,
        "usage_diversity_samples": args.usage_diversity_samples,
        "usage_diversity_temperature": args.usage_diversity_temperature,
        "lambda_shift_latent": SHIFT_LATENT_WEIGHT,
        "lambda_shift_sid": SHIFT_SID_WEIGHT,
        "shift_temperature": SHIFT_TEMPERATURE,
        "shift_radius": shift_radius(epoch),
        "stage": f"c21_shift_radius_{shift_radius(epoch)}",
    }


def write_config(args, data_dir, cache_dir, train_len, val_len, continuous_stats, base_lr):
    _original_write_config(args, data_dir, cache_dir, train_len, val_len, continuous_stats, base_lr)
    path = args.output_dir / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(
        {
            "experiment": "curriculum_v1_c21_pattern_shift_invariant",
            "stage": "C2.1",
            "init_pattern_only_c2_checkpoint": str(base.resolve_path(INIT_CHECKPOINT)),
            "split_index_dir": str(base.resolve_path(c2.INDEX_DIR)),
            "loss": {
                "waveform": 1.0,
                "vq": 1.0,
                "discrete_pattern": 0.2,
                "corr": 0.0,
                "derivative": 0.0,
                "continuous_pattern": 0.0,
                "shift_latent": SHIFT_LATENT_WEIGHT,
                "shift_sid": SHIFT_SID_WEIGHT,
                "shift_temperature": SHIFT_TEMPERATURE,
            },
            "shift_implementation": "non-circular zero-filled per-heartbeat shift",
            "shift_sampling": "uniform non-zero integer delta in [-radius,+radius]",
            "shift_curriculum": {"epochs_1_10": 1, "epochs_11_25": 2, "epochs_26_plus": 4},
            "shift_target": "original z_e and SID with stop-gradient",
        }
    )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(model, loader, device, args, weights):
    """Exclude the easier ±1/±2 curriculum stages from best-checkpoint selection."""
    stats = _original_evaluate(model, loader, device, args, weights)
    stats["objective_loss"] = stats["loss"]
    if int(weights.get("shift_radius", 1)) < 4:
        stats["loss"] = 1.0e9
        stats["best_checkpoint_eligible"] = 0
    else:
        stats["best_checkpoint_eligible"] = 1
    return stats


base.build_model = build_model
base.loss_weights = loss_weights
base.write_config = write_config
base.evaluate = evaluate


if __name__ == "__main__":
    base.main()
