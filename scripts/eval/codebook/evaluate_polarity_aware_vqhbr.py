#!/usr/bin/env python3
"""Evaluate polarity-aware VQ-HBR codebook quality with real NN assignments."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from timm.models import create_model


ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models import modeling_vqhbr as modeling_vqhbr  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Polarity-aware VQ-HBR evaluator.")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_pairs", default="", help="Comma list like 823:3,517:6.")
    parser.add_argument("--reservoir_per_sid", type=int, default=256)
    parser.add_argument("--min_count_for_distance", type=int, default=50)
    parser.add_argument("--intra_pairs_per_sid", type=int, default=256)
    parser.add_argument("--inter_pairs", type=int, default=200000)
    parser.add_argument("--max_shift", type=int, default=6)
    parser.add_argument("--silhouette_samples", type=int, default=5000)
    return parser.parse_args()


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


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
    data_path = data_path.expanduser().resolve()
    data = np.load(data_path / f"{split}_data.npy", mmap_mode="r")
    chans = np.load(data_path / f"{split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_path / f"{split}_data_in_times.npy", mmap_mode="r")
    return data, chans, times


def parse_pairs(raw: str) -> list[tuple[int, int]]:
    if not raw.strip():
        return []
    pairs = []
    for item in raw.split(","):
        left, right = item.split(":")
        pairs.append((int(left), int(right)))
    return pairs


def zscore_one(wave: np.ndarray) -> np.ndarray:
    wave = wave.astype(np.float32, copy=False)
    return ((wave - wave.mean()) / max(float(wave.std()), 1e-6)).astype(np.float32, copy=True)


def polarity_of(wave: np.ndarray) -> int:
    idx = int(np.argmax(np.abs(wave)))
    return 1 if float(wave[idx]) >= 0 else -1


def update_reservoir(reservoirs, seen_counts, sid: int, wave: np.ndarray, capacity: int) -> None:
    seen_counts[sid] += 1
    seen = int(seen_counts[sid])
    bucket = reservoirs[sid]
    zwave = zscore_one(wave)
    if len(bucket) < capacity:
        bucket.append(zwave)
        return
    replace_idx = random.randrange(seen)
    if replace_idx < capacity:
        bucket[replace_idx] = zwave


def overlap_for_shift(a: np.ndarray, b: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    if shift == 0:
        return a, b
    if shift > 0:
        return a[shift:], b[:-shift]
    shift_abs = -shift
    return a[:-shift_abs], b[shift_abs:]


def corr_distance_pre_zscored(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def shift_invariant_corr_distance(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[float, int]:
    best_distance = math.inf
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        aa, bb = overlap_for_shift(a, b, shift)
        if len(aa) < 8:
            continue
        distance = corr_distance_pre_zscored(aa, bb)
        if distance < best_distance:
            best_distance = distance
            best_shift = shift
    return float(best_distance), int(best_shift)


def entropy_perplexity(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(np.exp(-(probs * np.log(probs)).sum()))


def summarize_polarity_tables(counts, lead_pol_counts):
    rows = []
    for sid, count in enumerate(counts):
        if count <= 0:
            continue
        lead_items = lead_pol_counts.get(sid, {})
        neg_total = sum(v[0] for v in lead_items.values())
        pos_total = sum(v[1] for v in lead_items.values())
        total = neg_total + pos_total
        per_lead_majorities = []
        for neg_n, pos_n in lead_items.values():
            n = neg_n + pos_n
            if n > 0:
                per_lead_majorities.append(max(neg_n, pos_n) / n)
        rows.append(
            {
                "sid": sid,
                "count": int(count),
                "neg_n": int(neg_total),
                "pos_n": int(pos_total),
                "global_polarity_majority": float(max(neg_total, pos_total) / total) if total else 0.0,
                "per_lead_polarity_majority": float(np.mean(per_lead_majorities)) if per_lead_majorities else 0.0,
                "active_leads": len(lead_items),
            }
        )
    return pd.DataFrame(rows)


def compute_intra_inter(counts, reservoirs, args, sibling_lookup):
    rng = np.random.default_rng(args.seed)
    eligible = [
        int(sid)
        for sid in np.where(counts >= args.min_count_for_distance)[0]
        if len(reservoirs[int(sid)]) >= 2
    ]
    per_sid_rows = []
    all_intra = []
    for sid in eligible:
        waves = np.asarray(reservoirs[sid], dtype=np.float32)
        pair_count = min(args.intra_pairs_per_sid, len(waves) * (len(waves) - 1) // 2)
        dists = []
        for _ in range(pair_count):
            i, j = rng.choice(len(waves), size=2, replace=False)
            d, _ = shift_invariant_corr_distance(waves[int(i)], waves[int(j)], args.max_shift)
            dists.append(d)
        if dists:
            all_intra.extend(dists)
            per_sid_rows.append(
                {
                    "sid": sid,
                    "count": int(counts[sid]),
                    "intra_shift_corr_mean": float(np.mean(dists)),
                    "intra_shift_corr_p90": float(np.percentile(dists, 90)),
                }
            )

    active = [int(sid) for sid in np.where(counts > 0)[0] if len(reservoirs[int(sid)]) > 0]
    weights = counts[active].astype(np.float64)
    weights = weights / weights.sum()
    inter_all, inter_sibling, inter_nonsibling = [], [], []
    for _ in range(args.inter_pairs):
        a_idx, b_idx = rng.choice(len(active), size=2, replace=False, p=weights)
        sid_a = active[int(a_idx)]
        sid_b = active[int(b_idx)]
        wa = reservoirs[sid_a][int(rng.integers(0, len(reservoirs[sid_a])))]
        wb = reservoirs[sid_b][int(rng.integers(0, len(reservoirs[sid_b])))]
        d, _ = shift_invariant_corr_distance(wa, wb, args.max_shift)
        inter_all.append(d)
        if sibling_lookup.get(sid_a) == sid_b:
            inter_sibling.append(d)
        else:
            inter_nonsibling.append(d)

    intra_mean = float(np.mean(all_intra)) if all_intra else math.nan
    inter_mean = float(np.mean(inter_all)) if inter_all else math.nan
    return {
        "eligible_sids": len(per_sid_rows),
        "macro_intra_shift_corr": float(np.mean([r["intra_shift_corr_mean"] for r in per_sid_rows])) if per_sid_rows else math.nan,
        "sample_weighted_intra_shift_corr": intra_mean,
        "intra_shift_corr_p90": float(np.percentile(all_intra, 90)) if all_intra else math.nan,
        "overall_inter_shift_corr": inter_mean,
        "inter_over_intra": float(inter_mean / intra_mean) if intra_mean and not math.isnan(intra_mean) else math.nan,
        "paired_sibling_inter_shift_corr": float(np.mean(inter_sibling)) if inter_sibling else math.nan,
        "non_sibling_inter_shift_corr": float(np.mean(inter_nonsibling)) if inter_nonsibling else math.nan,
        "intra_pairs": len(all_intra),
        "inter_pairs": len(inter_all),
        "sibling_inter_pairs": len(inter_sibling),
    }, pd.DataFrame(per_sid_rows)


def compute_silhouette(reservoirs, counts, max_samples, seed):
    rng = np.random.default_rng(seed)
    samples = []
    labels = []
    active = [int(sid) for sid in np.where(counts > 0)[0] if len(reservoirs[int(sid)]) > 1]
    for sid in active:
        for wave in reservoirs[sid]:
            samples.append(wave)
            labels.append(sid)
    if len(samples) < 10 or len(set(labels)) < 2:
        return math.nan
    if len(samples) > max_samples:
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[int(i)] for i in idx]
        labels = [labels[int(i)] for i in idx]
    return float(silhouette_score(np.asarray(samples, dtype=np.float32), np.asarray(labels), metric="cosine"))


@torch.no_grad()
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    pairs = parse_pairs(args.target_pairs)
    sibling_lookup = {}
    for pos_code, neg_code in pairs:
        sibling_lookup[pos_code] = neg_code
        sibling_lookup[neg_code] = pos_code

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, resolved = load_tokenizer(args.checkpoint_path, args.codebook_size, args.codebook_dim, device)
    data, chans, times = load_arrays(args.data_path, args.split)
    print(f"Loaded checkpoint: {resolved}", flush=True)
    print(f"Loaded split: {args.split}, samples={len(data)}, shape={data.shape}", flush=True)

    counts = np.zeros(args.codebook_size, dtype=np.int64)
    reservoirs = defaultdict(list)
    seen_counts = np.zeros(args.codebook_size, dtype=np.int64)
    lead_pol_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    margin_sum = 0.0
    margin_n = 0
    pair_assign_total = 0
    pair_assign_correct = 0

    sse = 0.0
    sae = 0.0
    target_energy = 0.0
    signal_power_sum = 0.0
    noise_power_sum = 0.0
    valid_points = 0

    for start in range(0, len(data), args.batch_size):
        end = min(start + args.batch_size, len(data))
        batch_data_np = np.asarray(data[start:end])
        batch_chans_np = np.asarray(chans[start:end])
        batch_times_np = np.asarray(times[start:end])
        x = torch.from_numpy(np.array(batch_data_np, dtype=np.float32, copy=True)).to(device)
        in_chans = torch.from_numpy(np.array(batch_chans_np, dtype=np.int64, copy=True)).to(device)
        in_times = torch.from_numpy(np.array(batch_times_np, dtype=np.int64, copy=True)).to(device)
        valid = (in_chans > 0) & (in_times > 0) & ~(x == 0).all(dim=-1)

        z = model.encode_to_quantizer_features(x, in_chan_matrix=in_chans, in_time_matrix=in_times)
        z_flat = z.reshape(-1, args.codebook_dim)
        valid_flat = valid.reshape(-1)
        z_valid = z_flat[valid_flat]
        sim = z_valid @ model.quantize.embedding.weight.t()
        top2 = torch.topk(sim, k=2, dim=1).values
        margin_sum += float((top2[:, 0] - top2[:, 1]).sum().item())
        margin_n += int(z_valid.shape[0])
        code_valid = torch.argmax(sim, dim=1)

        quantize, _, _ = model.encode(x, in_chan_matrix=in_chans, in_time_matrix=in_times, valid_token_mask=valid)
        rec = model.decode(quantize, in_chan_matrix=in_chans, in_time_matrix=in_times)
        diff = (rec - x)[valid]
        target = x[valid]
        sse += float((diff * diff).sum().item())
        sae += float(torch.abs(diff).sum().item())
        target_energy += float((target * target).sum().item())
        signal_power_sum += float((target * target).mean(dim=-1).sum().item())
        noise_power_sum += float((diff * diff).mean(dim=-1).sum().item())
        valid_points += int(diff.numel())

        code_idx = 0
        positions = np.argwhere(((batch_chans_np > 0) & (batch_times_np > 0) & ~(np.isclose(batch_data_np, 0.0).all(axis=-1))))
        code_valid_np = code_valid.detach().cpu().numpy()
        pair_set = set(sibling_lookup.keys())
        for local_idx, token_idx in positions:
            sid = int(code_valid_np[code_idx])
            code_idx += 1
            wave = batch_data_np[local_idx, token_idx]
            lead = int(batch_chans_np[local_idx, token_idx])
            pol = polarity_of(wave)
            counts[sid] += 1
            update_reservoir(reservoirs, seen_counts, sid, wave, args.reservoir_per_sid)
            lead_pol_counts[sid][lead][1 if pol > 0 else 0] += 1
            if sid in pair_set:
                pair_assign_total += 1
                sibling = sibling_lookup[sid]
                is_correct = (pol > 0 and sid < sibling) or (pol < 0 and sid > sibling)
                # For our mapping the original positive code is listed before its negative paired code.
                if pairs:
                    positive_codes = {p[0] for p in pairs}
                    negative_codes = {p[1] for p in pairs}
                    is_correct = (pol > 0 and sid in positive_codes) or (pol < 0 and sid in negative_codes)
                pair_assign_correct += int(is_correct)

        if start == 0 or end == len(data) or (start // args.batch_size) % 20 == 0:
            print(f"Evaluated {end}/{len(data)}; valid tokens={int(counts.sum())}", flush=True)

    usage = {
        "total_tokens": int(counts.sum()),
        "active_codes": int((counts > 0).sum()),
        "dead_codes": int((counts == 0).sum()),
        "perplexity": entropy_perplexity(counts),
        "top1pct_usage": float(np.sort(counts)[::-1][: max(1, math.ceil(args.codebook_size * 0.01))].sum() / counts.sum()),
    }
    reconstruction = {
        "mse": float(sse / valid_points),
        "rmse": float(math.sqrt(sse / valid_points)),
        "mae": float(sae / valid_points),
        "prd": float(100.0 * math.sqrt(sse / max(target_energy, 1e-12))),
        "snr": float(10.0 * math.log10(max(signal_power_sum, 1e-12) / max(noise_power_sum, 1e-12))),
    }

    polarity_df = summarize_polarity_tables(counts, lead_pol_counts)
    polarity = {
        "macro_global_polarity_majority": float(polarity_df["global_polarity_majority"].mean()),
        "weighted_global_polarity_majority": float(np.average(polarity_df["global_polarity_majority"], weights=polarity_df["count"])),
        "macro_per_lead_polarity_majority": float(polarity_df["per_lead_polarity_majority"].mean()),
        "weighted_per_lead_polarity_majority": float(np.average(polarity_df["per_lead_polarity_majority"], weights=polarity_df["count"])),
        "paired_assignment_accuracy": float(pair_assign_correct / pair_assign_total) if pair_assign_total else math.nan,
        "paired_polarity_collision_rate": float(1.0 - pair_assign_correct / pair_assign_total) if pair_assign_total else math.nan,
        "paired_assigned_tokens": int(pair_assign_total),
    }

    distance, per_sid_distance = compute_intra_inter(counts, reservoirs, args, sibling_lookup)
    distance["silhouette_cosine"] = compute_silhouette(reservoirs, counts, args.silhouette_samples, args.seed)
    distance["nearest_code_margin"] = float(margin_sum / margin_n) if margin_n else math.nan

    weight = model.quantize.embedding.weight.detach().cpu().numpy()
    target_rows = []
    for pos_code, neg_code in pairs:
        pos_row = polarity_df[polarity_df["sid"] == pos_code]
        neg_row = polarity_df[polarity_df["sid"] == neg_code]
        pos_intra = per_sid_distance[per_sid_distance["sid"] == pos_code]
        neg_intra = per_sid_distance[per_sid_distance["sid"] == neg_code]
        target_rows.append(
            {
                "positive_code": pos_code,
                "negative_code": neg_code,
                "positive_count": int(counts[pos_code]),
                "negative_count": int(counts[neg_code]),
                "positive_intra_shift_corr": float(pos_intra.iloc[0]["intra_shift_corr_mean"]) if len(pos_intra) else math.nan,
                "negative_intra_shift_corr": float(neg_intra.iloc[0]["intra_shift_corr_mean"]) if len(neg_intra) else math.nan,
                "positive_polarity_majority": float(pos_row.iloc[0]["global_polarity_majority"]) if len(pos_row) else math.nan,
                "negative_polarity_majority": float(neg_row.iloc[0]["global_polarity_majority"]) if len(neg_row) else math.nan,
                "paired_code_cosine_distance": float(1.0 - np.dot(weight[pos_code], weight[neg_code]) / (np.linalg.norm(weight[pos_code]) * np.linalg.norm(weight[neg_code]) + 1e-12)),
            }
        )

    polarity_df.to_csv(args.output_dir / "per_sid_polarity.csv", index=False)
    per_sid_distance.to_csv(args.output_dir / "per_sid_distance.csv", index=False)
    pd.DataFrame(target_rows).to_csv(args.output_dir / "target_pairs.csv", index=False)
    summary = {
        "checkpoint": str(resolved),
        "data_path": str(args.data_path),
        "split": args.split,
        "target_pairs": pairs,
        "usage": usage,
        "reconstruction": reconstruction,
        "polarity": polarity,
        "distance": distance,
        "target_pairs_table": target_rows,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
