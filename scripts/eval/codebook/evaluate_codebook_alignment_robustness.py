#!/usr/bin/env python3
"""Evaluate whether small heartbeat crop shifts fragment a VQ vocabulary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval" / "codebook"))

from evaluate_morphology_only_comparison import (  # noqa: E402
    infer_batch,
    load_arrays,
    load_model,
    parse_run_spec,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Codebook alignment robustness evaluation")
    parser.add_argument("--run", action="append", required=True, help="NAME,TYPE,CHECKPOINT")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--pool_size", type=int, default=60000)
    parser.add_argument("--anchors", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--feature_points", type=int, default=24)
    parser.add_argument("--shift_windows", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    parser.add_argument("--min_std", type=float, default=1e-4)
    parser.add_argument("--min_ptp", type=float, default=1e-3)
    parser.add_argument("--examples_per_model", type=int, default=20)
    parser.add_argument("--random_pairs", type=int, default=200000)
    return parser.parse_args()


def zscore_rows(waves: np.ndarray) -> np.ndarray:
    waves = waves.astype(np.float32, copy=False)
    mean = waves.mean(axis=1, keepdims=True)
    std = waves.std(axis=1, keepdims=True)
    return (waves - mean) / np.maximum(std, 1e-6)


def shifted_with_edge(waves: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return waves
    out = np.empty_like(waves)
    if shift > 0:
        out[:, :shift] = waves[:, :1]
        out[:, shift:] = waves[:, :-shift]
    else:
        amount = -shift
        out[:, :-amount] = waves[:, amount:]
        out[:, -amount:] = waves[:, -1:]
    return out


def downsample_features(waves: np.ndarray, points: int) -> np.ndarray:
    if waves.shape[1] % points != 0:
        indices = np.linspace(0, waves.shape[1] - 1, points).round().astype(int)
        feature = waves[:, indices]
    else:
        feature = waves.reshape(len(waves), points, waves.shape[1] // points).mean(axis=2)
    feature = zscore_rows(feature)
    norms = np.linalg.norm(feature, axis=1, keepdims=True)
    return feature / np.maximum(norms, 1e-6)


def collect_pool(data, chans, times, args):
    valid_positions = []
    counters = defaultdict(int)
    for start in range(0, len(data), args.batch_size):
        end = min(start + args.batch_size, len(data))
        wave = np.asarray(data[start:end], dtype=np.float32)
        channel = np.asarray(chans[start:end])
        time = np.asarray(times[start:end])
        slot = (channel > 0) & (time > 0)
        finite = np.isfinite(wave).all(axis=-1)
        std = np.nanstd(wave, axis=-1)
        ptp = np.nanmax(wave, axis=-1) - np.nanmin(wave, axis=-1)
        counters["total_slots"] += int(slot.size)
        counters["padding_or_invalid_position"] += int((~slot).sum())
        counters["nonfinite"] += int((slot & ~finite).sum())
        counters["near_constant"] += int((slot & finite & (std < args.min_std)).sum())
        counters["low_amplitude"] += int((slot & finite & (std >= args.min_std) & (ptp < args.min_ptp)).sum())
        valid = slot & finite & (std >= args.min_std) & (ptp >= args.min_ptp)
        local_record, token = np.nonzero(valid)
        if len(local_record):
            valid_positions.append(
                np.column_stack((local_record + start, token)).astype(np.int32, copy=False)
            )
    positions = np.concatenate(valid_positions, axis=0)
    counters["valid_heartbeats"] = int(len(positions))
    rng = np.random.default_rng(args.seed)
    pool_size = min(args.pool_size, len(positions))
    selected = np.sort(rng.choice(len(positions), size=pool_size, replace=False))
    positions = positions[selected]
    waves = np.asarray(data[positions[:, 0], positions[:, 1]], dtype=np.float32)
    return positions, waves, dict(counters)


def retrieve_pairs(waves_z: np.ndarray, args):
    rng = np.random.default_rng(args.seed + 1)
    anchor_count = min(args.anchors, len(waves_z))
    anchors = np.sort(rng.choice(len(waves_z), size=anchor_count, replace=False))
    base_features = downsample_features(waves_z, args.feature_points)
    tree = cKDTree(base_features)
    pair_chunks = []
    max_shift = max(args.shift_windows)
    for shift in range(-max_shift, max_shift + 1):
        shifted = shifted_with_edge(waves_z[anchors], shift)
        query = downsample_features(shifted, args.feature_points)
        _dist, neighbors = tree.query(query, k=min(args.top_k + 1, len(waves_z)), workers=-1)
        neighbors = np.atleast_2d(neighbors)
        aa = np.repeat(anchors, neighbors.shape[1])
        bb = neighbors.reshape(-1)
        keep = aa != bb
        lo = np.minimum(aa[keep], bb[keep])
        hi = np.maximum(aa[keep], bb[keep])
        pair_chunks.append(np.column_stack((lo, hi)).astype(np.int32, copy=False))
    pairs = np.unique(np.concatenate(pair_chunks, axis=0), axis=0)
    return anchors.astype(np.int32), pairs


def row_corr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 1e-8
    out = np.full(len(a), np.nan, dtype=np.float32)
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return np.clip(out, -1.0, 1.0), valid


def exact_pair_correlations(waves: np.ndarray, pairs: np.ndarray, max_shift: int, chunk_size=50000):
    all_corr = np.full((len(pairs), 2 * max_shift + 1), np.nan, dtype=np.float32)
    valid_pair = np.ones(len(pairs), dtype=bool)
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        a = waves[pairs[start:end, 0]]
        b = waves[pairs[start:end, 1]]
        chunk_valid = np.ones(end - start, dtype=bool)
        for column, shift in enumerate(range(-max_shift, max_shift + 1)):
            if shift == 0:
                aa, bb = a, b
            elif shift > 0:
                aa, bb = a[:, shift:], b[:, :-shift]
            else:
                amount = -shift
                aa, bb = a[:, :-amount], b[:, amount:]
            corr, valid = row_corr(aa, bb)
            all_corr[start:end, column] = corr
            chunk_valid &= valid
        valid_pair[start:end] = chunk_valid
    return all_corr, valid_pair


def window_result(corr_by_shift: np.ndarray, max_shift: int, window: int):
    center = max_shift
    sub = corr_by_shift[:, center - window : center + window + 1]
    arg = np.nanargmax(sub, axis=1)
    best_corr = sub[np.arange(len(sub)), arg]
    best_shift = arg.astype(np.int16) - window
    return best_corr.astype(np.float32), best_shift


def infer_selected_sids(run, args, device, data, chans, times, positions):
    model, checkpoint = load_model(run, args, device)
    lookup = np.full((len(data), data.shape[1]), -1, dtype=np.int32)
    lookup[positions[:, 0], positions[:, 1]] = np.arange(len(positions), dtype=np.int32)
    selected_sid = np.full(len(positions), -1, dtype=np.int32)
    for start in range(0, len(data), args.batch_size):
        end = min(start + args.batch_size, len(data))
        _rec, ids = infer_batch(
            model,
            run["model_type"],
            np.asarray(data[start:end]),
            np.asarray(chans[start:end]),
            np.asarray(times[start:end]),
            device,
        )
        local_lookup = lookup[start:end]
        rows, tokens = np.nonzero(local_lookup >= 0)
        if len(rows):
            selected_sid[local_lookup[rows, tokens]] = ids[rows, tokens]
    code_vectors = model.quantize.embedding.weight.detach().float().cpu().numpy()
    if code_vectors.shape[0] != args.codebook_size and code_vectors.shape[1] == args.codebook_size:
        code_vectors = code_vectors.T
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if (selected_sid < 0).any():
        raise RuntimeError(f"Missing SID assignments for {(selected_sid < 0).sum()} selected heartbeats")
    return selected_sid, code_vectors, checkpoint


def describe(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": math.nan, "median": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    if frame.empty:
        return "(no rows)"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                text = "nan" if not np.isfinite(value) else f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def code_cosine(code_vectors: np.ndarray, sid_a: np.ndarray, sid_b: np.ndarray) -> np.ndarray:
    vectors = code_vectors / np.maximum(np.linalg.norm(code_vectors, axis=1, keepdims=True), 1e-8)
    return np.sum(vectors[sid_a] * vectors[sid_b], axis=1)


def aligned_for_plot(wave: np.ndarray, shift: int) -> np.ndarray:
    out = np.full_like(wave, np.nan)
    if shift == 0:
        return wave.copy()
    if shift > 0:
        out[shift:] = wave[:-shift]
    else:
        amount = -shift
        out[:-amount] = wave[amount:]
    return out


def save_examples(model_name, output_dir, waves, pairs, sid, raw_corr, shift_corr, best_shift, limit):
    eligible = (shift_corr >= 0.95) & (sid[pairs[:, 0]] != sid[pairs[:, 1]]) & (np.abs(best_shift) >= 2)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return 0
    score = shift_corr[indices] + (shift_corr[indices] - raw_corr[indices])
    indices = indices[np.argsort(score)[::-1][:limit]]
    model_dir = output_dir / "alignment_examples" / model_name.replace("/", "_").replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    for rank, pair_idx in enumerate(indices, 1):
        i, j = pairs[pair_idx]
        shift = int(best_shift[pair_idx])
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(waves[i], label="heartbeat A")
        axes[0].plot(waves[j], label="heartbeat B")
        axes[0].legend()
        axes[0].set_title("Original")
        axes[1].plot(waves[i], label="heartbeat A")
        axes[1].plot(aligned_for_plot(waves[j], shift), label="aligned heartbeat B")
        axes[1].legend()
        axes[1].set_title("After best-shift alignment")
        fig.suptitle(
            f"SID {sid[i]} vs {sid[j]} | raw={raw_corr[pair_idx]:.4f} | "
            f"shift={shift_corr[pair_idx]:.4f} | best_shift={shift:+d}"
        )
        fig.tight_layout()
        fig.savefig(model_dir / f"pair_{rank:02d}.png", dpi=150)
        plt.close(fig)
    return len(indices)


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [parse_run_spec(spec) for spec in args.run]
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    data_path, data, chans, times = load_arrays(args.data_path, args.split)

    positions, waves, filter_counts = collect_pool(data, chans, times, args)
    waves_z = zscore_rows(waves)
    anchors, pairs = retrieve_pairs(waves_z, args)
    max_shift = max(args.shift_windows)
    corr_by_shift, pair_valid = exact_pair_correlations(waves, pairs, max_shift)
    invalid_pair_count = int((~pair_valid).sum())
    pairs = pairs[pair_valid]
    corr_by_shift = corr_by_shift[pair_valid]
    raw_corr = corr_by_shift[:, max_shift]
    window_results = {window: window_result(corr_by_shift, max_shift, window) for window in args.shift_windows}

    rng = np.random.default_rng(args.seed + 2)
    random_pairs = rng.integers(0, len(waves), size=(args.random_pairs * 2, 2), dtype=np.int32)
    random_pairs = random_pairs[random_pairs[:, 0] != random_pairs[:, 1]][: args.random_pairs]

    summary_rows = []
    shift_rows = []
    disagreement_rows = []
    group_rows = []
    code_rows = []
    model_rows = []

    for run in runs:
        print(f"Evaluating SID assignments: {run['name']}", flush=True)
        sid, code_vectors, checkpoint = infer_selected_sids(run, args, device, data, chans, times, positions)
        sid_a = sid[pairs[:, 0]]
        sid_b = sid[pairs[:, 1]]
        different = sid_a != sid_b
        random_different = sid[random_pairs[:, 0]] != sid[random_pairs[:, 1]]
        model_rows.append(
            {
                "experiment_name": run["name"],
                "model_type": run["model_type"],
                "checkpoint": str(checkpoint),
                "codebook_size": int(code_vectors.shape[0]),
                "latent_dim": int(code_vectors.shape[1]),
                "evaluation_dataset": str(data_path),
                "split": args.split,
                "evaluation_heartbeats": int(filter_counts["valid_heartbeats"]),
                "sampled_pool_heartbeats": int(len(waves)),
            }
        )
        for window in args.shift_windows:
            shift_corr, best_shift = window_results[window]
            for threshold in args.thresholds:
                high = shift_corr >= threshold
                high_count = int(high.sum())
                diff_count = int((high & different).sum())
                summary_rows.append(
                    {
                        "model": run["name"],
                        "shift_window": window,
                        "threshold": threshold,
                        "high_corr_pairs": high_count,
                        "same_sid_pairs": high_count - diff_count,
                        "different_sid_pairs": diff_count,
                        "disagreement_rate": diff_count / high_count if high_count else math.nan,
                    }
                )
            high95 = shift_corr >= 0.95
            high95_diff = high95 & different
            signed_counts = pd.Series(best_shift[high95_diff]).value_counts().to_dict()
            abs_counts = pd.Series(np.abs(best_shift[high95_diff])).value_counts().to_dict()
            total_diff = int(high95_diff.sum())
            for shift in range(-window, window + 1):
                count = int(signed_counts.get(shift, 0))
                shift_rows.append({"model": run["name"], "shift_window": window, "distribution": "signed", "best_shift": shift, "count": count, "ratio": count / total_diff if total_diff else math.nan})
            for shift in range(0, window + 1):
                count = int(abs_counts.get(shift, 0))
                shift_rows.append({"model": run["name"], "shift_window": window, "distribution": "absolute", "best_shift": shift, "count": count, "ratio": count / total_diff if total_diff else math.nan})
                bucket = high95 & (np.abs(best_shift) == shift)
                bucket_count = int(bucket.sum())
                bucket_diff = int((bucket & different).sum())
                disagreement_rows.append({"model": run["name"], "shift_window": window, "abs_shift": shift, "pairs": bucket_count, "different_sid": bucket_diff, "disagreement_rate": bucket_diff / bucket_count if bucket_count else math.nan})
            for group_name, group_mask in (("same_sid", high95 & ~different), ("different_sid", high95 & different)):
                corr_gain = shift_corr[group_mask] - raw_corr[group_mask]
                abs_shift = np.abs(best_shift[group_mask]).astype(np.float32)
                group_rows.append(
                    {
                        "model": run["name"],
                        "shift_window": window,
                        "group": group_name,
                        "pairs": int(group_mask.sum()),
                        "raw_corr_mean": float(np.mean(raw_corr[group_mask])) if group_mask.any() else math.nan,
                        "shift_corr_mean": float(np.mean(shift_corr[group_mask])) if group_mask.any() else math.nan,
                        "corr_gain": float(np.mean(corr_gain)) if len(corr_gain) else math.nan,
                        "abs_shift_mean": float(np.mean(abs_shift)) if len(abs_shift) else math.nan,
                        "abs_shift_p50": float(np.percentile(abs_shift, 50)) if len(abs_shift) else math.nan,
                        "abs_shift_p90": float(np.percentile(abs_shift, 90)) if len(abs_shift) else math.nan,
                    }
                )
            high_cos = code_cosine(code_vectors, sid_a[high95_diff], sid_b[high95_diff])
            random_cos = code_cosine(
                code_vectors,
                sid[random_pairs[random_different, 0]],
                sid[random_pairs[random_different, 1]],
            )
            for group_name, values in (("high_corr_different_sid", high_cos), ("random_different_sid", random_cos)):
                stats = describe(values)
                code_rows.append({"model": run["name"], "shift_window": window, "group": group_name, "pairs": int(len(values)), "code_cosine_mean": stats["mean"], "median": stats["median"], "p10": stats["p10"], "p90": stats["p90"]})
        shift4_corr, shift4_best = window_results[4]
        examples = save_examples(run["name"], output_dir, waves, pairs, sid, raw_corr, shift4_corr, shift4_best, args.examples_per_model)
        model_rows[-1]["saved_examples"] = examples

    summary = pd.DataFrame(summary_rows)
    best_shift_df = pd.DataFrame(shift_rows)
    disagreement_df = pd.DataFrame(disagreement_rows)
    groups_df = pd.DataFrame(group_rows)
    code_df = pd.DataFrame(code_rows)
    models_df = pd.DataFrame(model_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    best_shift_df.to_csv(output_dir / "best_shift_distribution.csv", index=False)
    disagreement_df.to_csv(output_dir / "disagreement_by_shift.csv", index=False)
    groups_df.to_csv(output_dir / "same_vs_different_analysis.csv", index=False)
    code_df.to_csv(output_dir / "code_neighbor_analysis.csv", index=False)
    models_df.to_csv(output_dir / "model_metadata.csv", index=False)
    np.save(output_dir / "sampled_positions.npy", positions)
    np.save(output_dir / "candidate_pairs.npy", pairs)

    core = summary[(summary.threshold == 0.95)].copy()
    best_by_window = core.loc[core.groupby("shift_window").disagreement_rate.idxmin(), ["shift_window", "model", "disagreement_rate"]]
    shift4 = core[core.shift_window == 4].sort_values("disagreement_rate")
    nonzero = best_shift_df[(best_shift_df.shift_window == 4) & (best_shift_df.distribution == "absolute")]
    nonzero_ratio = {
        model: float(frame.loc[frame.best_shift > 0, "ratio"].sum())
        for model, frame in nonzero.groupby("model")
    }
    trends = {}
    for model, frame in disagreement_df[disagreement_df.shift_window == 4].groupby("model"):
        usable = frame.dropna(subset=["disagreement_rate"])
        trends[model] = float(usable[["abs_shift", "disagreement_rate"]].corr().iloc[0, 1]) if len(usable) >= 2 else math.nan
    code4 = code_df[code_df.shift_window == 4].pivot(index="model", columns="group", values="code_cosine_mean")
    code_delta = (code4.get("high_corr_different_sid") - code4.get("random_different_sid")).to_dict()

    readme = [
        "# Codebook Alignment Robustness\n",
        "## 实验设置\n",
        f"- 数据：`{data_path}`，split=`{args.split}`。",
        f"- 有效 heartbeat：{filter_counts['valid_heartbeats']:,}；固定抽样池：{len(waves):,}；anchor：{len(anchors):,}；候选 pair：{len(pairs):,}。",
        f"- 候选检索：波形 z-score 后降采样到 {args.feature_points} 维，以 ±{max_shift} 个采样点的查询版本在同一波形池中做 KD-tree top-{args.top_k} 检索；不使用 SID 或模型 latent。",
        "- 注意：这是 morphology-neighbor enriched candidate set，用于稳定发现跨 SID 的高相似 pair；disagreement 是检索到的高相关 pair 的条件比例，不代表所有 heartbeat 两两组合中的总体发生率。",
        f"- 精确比较：overlap-only Pearson correlation，不使用 circular roll；shift windows={args.shift_windows}；thresholds={args.thresholds}。",
        f"- 固定随机种子：{args.seed}。所有模型使用相同 heartbeat、anchor 和 candidate pairs。",
        f"- 无效过滤：{json.dumps(filter_counts, ensure_ascii=False)}；无效 pair：{invalid_pair_count}。\n",
        "## 模型\n",
        dataframe_to_markdown(models_df),
        "\n## 核心结果（threshold=0.95）\n",
        dataframe_to_markdown(core),
        "\n## 自动回答\n",
        f"1. High-corr SID Disagreement 最低：{shift4.iloc[0]['model']}（±4 时 {shift4.iloc[0]['disagreement_rate']:.4f}）。",
        "2. ±2/±4 各窗口最鲁棒模型：" + "; ".join(f"±{int(r.shift_window)} {r.model} ({r.disagreement_rate:.4f})" for r in best_by_window.itertuples()) + "。",
        "3. 高相关但不同 SID pair 的 non-zero best shift 比例（±4）：" + "; ".join(f"{k}={v:.3f}" for k, v in nonzero_ratio.items()) + "。",
        "4. |best_shift| 与 disagreement 的相关方向（±4）：" + "; ".join(f"{k}={v:.3f}" for k, v in trends.items()) + "。正值表示随偏移增大而上升。",
        "5. 高相关不同 SID code cosine 相对随机不同 SID 的均值增量（±4）：" + "; ".join(f"{k}={v:.3f}" for k, v in code_delta.items()) + "。正值表示这些 code 更像相邻冗余词。",
    ]
    median_disagreement = float(shift4.disagreement_rate.median())
    median_nonzero = float(np.median(list(nonzero_ratio.values())))
    zero_shift = disagreement_df[(disagreement_df.shift_window == 4) & (disagreement_df.abs_shift == 0)]
    median_zero_shift_disagreement = float(zero_shift.disagreement_rate.median())
    if median_disagreement < 0.2:
        case = "Case A：整体 high-corr disagreement 较低，现有 encoder 已具备较好的平移鲁棒性，暂不需要专门增加 shift consistency loss。"
    elif median_nonzero >= 0.6:
        case = (
            "Case B（但不是单一原因）：high-corr disagreement 较高，且多数不同 SID pair 需要 non-zero shift 才达到高相关，"
            "说明存在 segmentation-induced fragmentation，值得下一步测试 random shift augmentation + latent consistency loss。"
            f"不过 ±4 条件下 best_shift=0 的 disagreement 中位数仍为 {median_zero_shift_disagreement:.3f}，"
            "因此大量 fragmentation 同时来自 latent/codebook redundancy、VQ assignment boundary 或模型监督方式；shift consistency 预计只能解决其中一部分。"
        )
    else:
        case = "Case C：high-corr disagreement 较高，但 best shift 多数为 0；问题更可能来自 latent/codebook fragmentation、监督信号或 VQ 边界，而非切分偏移。"
    readme.extend(["\n## 结论\n", case, "\n完整原始数值见同目录 CSV；代表性波形见 `alignment_examples/`。\n"])
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
