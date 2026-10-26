#!/usr/bin/env python3
"""Fit frozen linear probes from encoder/quantized latents to Stable4 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_morphology_only_comparison import load_model


STABLE_FEATURES = ["r_amplitude", "s_amplitude", "r_s_ratio", "t_amplitude"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Frozen z_e versus z_q Stable4 linear probes")
    parser.add_argument("--run", action="append", required=True, help="NAME,CHECKPOINT")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--phys_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_records", type=int, default=128)
    parser.add_argument("--max_train_tokens", type=int, default=250000)
    parser.add_argument("--max_val_tokens", type=int, default=250000)
    parser.add_argument("--ridge_alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    return parser.parse_args()


def parse_run(spec: str) -> tuple[str, Path]:
    parts = [part.strip() for part in spec.split(",", 1)]
    if len(parts) != 2:
        raise ValueError("--run must be NAME,CHECKPOINT")
    return parts[0], Path(parts[1])


class RidgeAccumulator:
    def __init__(self, dim: int, n_features: int):
        self.xtx = np.zeros((n_features, dim + 1, dim + 1), dtype=np.float64)
        self.xty = np.zeros((n_features, dim + 1), dtype=np.float64)
        self.count = np.zeros(n_features, dtype=np.int64)

    def update(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        design = np.concatenate([x.astype(np.float64), np.ones((len(x), 1))], axis=1)
        for feature_idx in range(y.shape[1]):
            keep = mask[:, feature_idx] & np.isfinite(y[:, feature_idx])
            if not keep.any():
                continue
            xx = design[keep]
            yy = y[keep, feature_idx].astype(np.float64)
            self.xtx[feature_idx] += xx.T @ xx
            self.xty[feature_idx] += xx.T @ yy
            self.count[feature_idx] += int(keep.sum())

    def solve(self, alpha: float) -> np.ndarray:
        weights = np.zeros_like(self.xty)
        for feature_idx in range(len(weights)):
            penalty = np.eye(self.xtx.shape[-1], dtype=np.float64) * alpha
            penalty[-1, -1] = 0.0
            weights[feature_idx] = np.linalg.solve(
                self.xtx[feature_idx] + penalty,
                self.xty[feature_idx],
            )
        return weights


class ProbeMetrics:
    def __init__(self, n_features: int):
        self.count = np.zeros(n_features, dtype=np.int64)
        self.sum_y = np.zeros(n_features, dtype=np.float64)
        self.sum_y_sq = np.zeros(n_features, dtype=np.float64)
        self.sse = np.zeros(n_features, dtype=np.float64)
        self.sae = np.zeros(n_features, dtype=np.float64)

    def update(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> None:
        design = np.concatenate([x.astype(np.float64), np.ones((len(x), 1))], axis=1)
        pred = design @ weights.T
        for feature_idx in range(y.shape[1]):
            keep = mask[:, feature_idx] & np.isfinite(y[:, feature_idx])
            if not keep.any():
                continue
            yy = y[keep, feature_idx].astype(np.float64)
            error = pred[keep, feature_idx] - yy
            self.count[feature_idx] += int(keep.sum())
            self.sum_y[feature_idx] += yy.sum()
            self.sum_y_sq[feature_idx] += np.square(yy).sum()
            self.sse[feature_idx] += np.square(error).sum()
            self.sae[feature_idx] += np.abs(error).sum()

    def summary(self, feature_names: list[str], raw_std: np.ndarray) -> dict:
        result = {}
        for idx, feature in enumerate(feature_names):
            count = int(self.count[idx])
            sst = self.sum_y_sq[idx] - self.sum_y[idx] ** 2 / max(count, 1)
            result[feature] = {
                "count": count,
                "r2": float(1.0 - self.sse[idx] / sst) if sst > 0 else None,
                "normalized_mae": float(self.sae[idx] / count) if count else None,
                "raw_mae": float(self.sae[idx] / count * raw_std[idx]) if count else None,
            }
        valid_r2 = [item["r2"] for item in result.values() if item["r2"] is not None]
        return {"features": result, "macro_r2": float(np.mean(valid_r2))}


def load_split(data_dir: Path, phys_dir: Path, split: str, feature_indices: list[int]):
    return (
        np.load(data_dir / f"{split}_data.npy", mmap_mode="r"),
        np.load(data_dir / f"{split}_data_in_chans.npy", mmap_mode="r"),
        np.load(data_dir / f"{split}_data_in_times.npy", mmap_mode="r"),
        np.load(phys_dir / f"{split}_phys_continuous.npy", mmap_mode="r"),
        np.load(phys_dir / f"{split}_phys_continuous_mask.npy", mmap_mode="r"),
        feature_indices,
    )


@torch.no_grad()
def iterate_latents(model, arrays, max_tokens: int, batch_records: int, seed: int, device):
    data, chans, times, targets, masks, feature_indices = arrays
    rng = np.random.default_rng(seed)
    record_order = rng.permutation(len(data))
    emitted = 0
    for start in range(0, len(record_order), batch_records):
        indices = record_order[start : start + batch_records]
        x_np = np.asarray(data[indices])
        chans_np = np.asarray(chans[indices])
        times_np = np.asarray(times[indices])
        target_np = np.asarray(targets[indices])[..., feature_indices]
        mask_np = np.asarray(masks[indices])[..., feature_indices]
        x = torch.from_numpy(x_np.copy()).float().to(device)
        in_chans = torch.from_numpy(chans_np.copy()).to(device)
        in_times = torch.from_numpy(times_np.copy()).to(device)
        valid, _beats, z_e, z_q, _ids, _loss = model.encode_valid(x, in_chans, in_times)
        valid_np = valid.cpu().numpy()
        y = target_np[valid_np]
        y_mask = mask_np[valid_np]
        ze_np = z_e.float().cpu().numpy()
        zq_np = z_q.float().cpu().numpy()
        remaining = max_tokens - emitted
        if len(y) > remaining:
            picked = rng.choice(len(y), size=remaining, replace=False)
            y, y_mask, ze_np, zq_np = y[picked], y_mask[picked], ze_np[picked], zq_np[picked]
        yield ze_np, zq_np, y, y_mask
        emitted += len(y)
        if emitted >= max_tokens:
            break


def evaluate_model(name: str, checkpoint: Path, args, train_arrays, val_arrays, mean, std) -> dict:
    run = {"name": name, "model_type": "morphology_v2", "checkpoint": checkpoint}
    model_args = SimpleNamespace(codebook_size=args.codebook_size, codebook_dim=args.codebook_dim)
    model, resolved = load_model(run, model_args, torch.device(args.device))
    accumulators = {
        "z_e": RidgeAccumulator(args.codebook_dim, len(STABLE_FEATURES)),
        "z_q": RidgeAccumulator(args.codebook_dim, len(STABLE_FEATURES)),
    }
    for z_e, z_q, y_raw, mask in iterate_latents(
        model, train_arrays, args.max_train_tokens, args.batch_records, args.seed, torch.device(args.device)
    ):
        y = np.clip((y_raw - mean) / std, -5.0, 5.0)
        accumulators["z_e"].update(z_e, y, mask)
        accumulators["z_q"].update(z_q, y, mask)
    weights = {key: value.solve(args.ridge_alpha) for key, value in accumulators.items()}
    metrics = {"z_e": ProbeMetrics(len(STABLE_FEATURES)), "z_q": ProbeMetrics(len(STABLE_FEATURES))}
    for z_e, z_q, y_raw, mask in iterate_latents(
        model, val_arrays, args.max_val_tokens, args.batch_records, args.seed + 1, torch.device(args.device)
    ):
        y = np.clip((y_raw - mean) / std, -5.0, 5.0)
        metrics["z_e"].update(z_e, y, mask, weights["z_e"])
        metrics["z_q"].update(z_q, y, mask, weights["z_q"])
    result = {
        "run": name,
        "checkpoint": str(resolved),
        "train_tokens": int(args.max_train_tokens),
        "val_tokens": int(args.max_val_tokens),
        "z_e": metrics["z_e"].summary(STABLE_FEATURES, std),
        "z_q": metrics["z_q"].summary(STABLE_FEATURES, std),
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.phys_dir = args.phys_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((args.phys_dir / "physiology_feature_schema.json").read_text(encoding="utf-8"))
    all_features = schema["continuous_features"]
    feature_indices = [all_features.index(name) for name in STABLE_FEATURES]
    train_stats = next(item for item in schema["splits"] if item["split"] == "train")["continuous_stats"]
    mean = np.asarray([train_stats[name]["mean"] for name in STABLE_FEATURES], dtype=np.float64)
    std = np.asarray([max(train_stats[name]["std"], 1e-6) for name in STABLE_FEATURES], dtype=np.float64)
    train_arrays = load_split(args.data_dir, args.phys_dir, "train", feature_indices)
    val_arrays = load_split(args.data_dir, args.phys_dir, "val", feature_indices)
    results = [
        evaluate_model(name, checkpoint, args, train_arrays, val_arrays, mean, std)
        for name, checkpoint in (parse_run(spec) for spec in args.run)
    ]
    output = {
        "probe": "ridge_linear_with_intercept",
        "stable_features": STABLE_FEATURES,
        "ridge_alpha": args.ridge_alpha,
        "results": results,
    }
    path = args.output_dir / "ze_zq_stable4_probe.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
