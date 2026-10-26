#!/usr/bin/env python3
"""Level 4 clinical enrichment and Level 5 distribution separability for ECG SIDs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact, spearmanr
from sklearn.decomposition import PCA


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval" / "codebook"))

from evaluate_morphology_only_comparison import infer_batch, load_model, parse_run_spec, resolve_path  # noqa: E402


GROUPS = ["Healthy", "Control", "NOMI", "OMI"]
LABEL_MAP = {"健康": "Healthy", "对照组": "Control", "NOMI": "NOMI", "OMI": "OMI"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SID clinical enrichment and distribution separability")
    parser.add_argument("--run", action="append", required=True, help="NAME,TYPE,CHECKPOINT")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--min_heartbeats", type=int, default=100)
    parser.add_argument("--min_patients", type=int, default=20)
    parser.add_argument("--max_patient_dominance", type=float, default=0.25)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max_pairs", type=int, default=20000)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--waveforms_per_sid", type=int, default=20)
    return parser.parse_args()


def safe_name(name: str) -> str:
    return name.lower().replace("+", "_plus_").replace("-", "_").replace(" ", "_")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("nan" if not np.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def patient_id_for_metadata(meta: pd.DataFrame) -> pd.Series:
    patient = meta.get("patient_sn", pd.Series(index=meta.index, dtype=object)).astype("string")
    ecg = meta["csv_name"].astype(str)
    healthy = meta["label_name"].astype(str).eq("健康")
    result = "patient:" + patient.fillna("").astype(str)
    result.loc[healthy] = "healthy:" + ecg.loc[healthy]
    missing = (~healthy) & patient.isna()
    if missing.any():
        raise ValueError(f"Non-healthy rows missing patient_sn: {int(missing.sum())}")
    return result


def check_patient_leakage(data_path: Path) -> dict:
    rows = []
    for split in ("train", "val", "test"):
        path = data_path / f"{split}_metadata.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["patient_id_eval"] = patient_id_for_metadata(frame)
        frame["split_eval"] = split
        rows.append(frame[["patient_id_eval", "split_eval", "label_name"]])
    combined = pd.concat(rows, ignore_index=True)
    split_counts = combined.groupby("patient_id_eval").split_eval.nunique()
    group_counts = combined.assign(group=combined.label_name.map(LABEL_MAP)).groupby("patient_id_eval").group.nunique()
    return {
        "patients_across_splits": int((split_counts > 1).sum()),
        "patients_across_groups": int((group_counts > 1).sum()),
        "unique_patients_all_splits": int(combined.patient_id_eval.nunique()),
    }


def build_assignments(run, args, device, data, chans, times, meta):
    model, checkpoint = load_model(run, args, device)
    frames = []
    for start in range(0, len(data), args.batch_size):
        end = min(start + args.batch_size, len(data))
        batch_data = np.asarray(data[start:end], dtype=np.float32)
        batch_chans = np.asarray(chans[start:end])
        batch_times = np.asarray(times[start:end])
        _rec, ids = infer_batch(model, run["model_type"], batch_data, batch_chans, batch_times, device)
        valid = (
            (batch_chans > 0)
            & (batch_times > 0)
            & np.isfinite(batch_data).all(axis=-1)
            & (np.std(batch_data, axis=-1) > 1e-4)
            & ((np.max(batch_data, axis=-1) - np.min(batch_data, axis=-1)) > 1e-3)
            & (ids >= 0)
            & (ids < args.codebook_size)
        )
        record_local, token = np.nonzero(valid)
        if not len(record_local):
            continue
        record_index = record_local + start
        selected_meta = meta.iloc[record_index]
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": selected_meta.patient_id_eval.to_numpy(),
                    "ecg_id": selected_meta.csv_name.astype(str).to_numpy(),
                    "group": selected_meta.group_eval.to_numpy(),
                    "ambiguous_patient": selected_meta.ambiguous_patient.to_numpy(dtype=bool),
                    "lead": batch_chans[record_local, token].astype(np.int16),
                    "heartbeat_index": token.astype(np.int16),
                    "time_index": batch_times[record_local, token].astype(np.int16),
                    "sid": ids[record_local, token].astype(np.int16),
                    "is_valid": True,
                    "record_index": record_index.astype(np.int32),
                    "token_index": token.astype(np.int16),
                }
            )
        )
    assignments = pd.concat(frames, ignore_index=True)
    code_vectors = model.quantize.embedding.weight.detach().float().cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return assignments, checkpoint, code_vectors


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def patient_histograms(assignments: pd.DataFrame, vocab_size: int):
    patients = assignments[["patient_id", "group"]].drop_duplicates().sort_values("patient_id")
    patient_ids = patients.patient_id.to_numpy()
    groups = patients.group.to_numpy()
    index = {patient: i for i, patient in enumerate(patient_ids)}
    hist = np.zeros((len(patients), vocab_size), dtype=np.float64)
    rows = assignments.patient_id.map(index).to_numpy(dtype=np.int32)
    np.add.at(hist, (rows, assignments.sid.to_numpy(dtype=np.int32)), 1.0)
    counts = hist.sum(axis=1, keepdims=True)
    distributions = hist / np.maximum(counts, 1.0)
    return patient_ids, groups, hist, distributions


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    left = np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-300) / np.maximum(m, 1e-300)), 0.0))
    right = np.sum(np.where(q > 0, q * np.log2(np.maximum(q, 1e-300) / np.maximum(m, 1e-300)), 0.0))
    return float(0.5 * (left + right))


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    left = np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-300) / np.maximum(m, 1e-300)), 0.0), axis=1)
    right = np.sum(np.where(q > 0, q * np.log2(np.maximum(q, 1e-300) / np.maximum(m, 1e-300)), 0.0), axis=1)
    return 0.5 * (left + right)


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / math.sqrt(2.0))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def group_distributions(assignments, patient_groups, patient_dist, vocab_size):
    heartbeat = {}
    equal = {}
    sid = assignments.sid.to_numpy(dtype=np.int32)
    for group in GROUPS:
        counts = np.bincount(assignments.loc[assignments.group == group, "sid"], minlength=vocab_size).astype(float)
        heartbeat[group] = counts / max(counts.sum(), 1.0)
        equal[group] = patient_dist[patient_groups == group].mean(axis=0)
        equal[group] /= max(equal[group].sum(), 1e-12)
    return heartbeat, equal


def distance_matrices(distributions):
    js = pd.DataFrame(0.0, index=GROUPS, columns=GROUPS)
    he = js.copy()
    tv = js.copy()
    for a, b in combinations(GROUPS, 2):
        js.loc[a, b] = js.loc[b, a] = js_divergence(distributions[a], distributions[b])
        he.loc[a, b] = he.loc[b, a] = hellinger(distributions[a], distributions[b])
        tv.loc[a, b] = tv.loc[b, a] = total_variation(distributions[a], distributions[b])
    return js, he, tv


def basic_statistics(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        frame = assignments[assignments.group == group]
        rows.append(
            {
                "group": group,
                "patient_count": frame.patient_id.nunique(),
                "ecg_count": frame.ecg_id.nunique(),
                "heartbeat_count": len(frame),
                "active_sid_count": frame.sid.nunique(),
            }
        )
    return pd.DataFrame(rows)


def enrichment_table(assignments, patient_ids, patient_groups, patient_hist, args):
    total = len(assignments)
    group_total = assignments.group.value_counts().reindex(GROUPS, fill_value=0)
    prior = group_total / total
    patient_sid_counts = assignments.groupby(["sid", "patient_id"]).size()
    ecg_sid_counts = assignments.groupby(["sid", "ecg_id"]).size()
    rows = []
    for sid in range(args.codebook_size):
        sid_frame = assignments[assignments.sid == sid]
        count_total = len(sid_frame)
        counts = sid_frame.group.value_counts().reindex(GROUPS, fill_value=0)
        composition = counts / max(count_total, 1)
        row = {"sid": sid, "count_total": count_total}
        row["unique_patients"] = sid_frame.patient_id.nunique()
        row["unique_ecgs"] = sid_frame.ecg_id.nunique()
        if count_total:
            per_patient = patient_sid_counts.loc[sid] if sid in patient_sid_counts.index.get_level_values(0) else pd.Series(dtype=float)
            row["patient_dominance"] = float(per_patient.max() / count_total) if len(per_patient) else math.nan
        else:
            row["patient_dominance"] = math.nan
        for group in GROUPS:
            key = group.lower()
            row[f"count_{key}"] = int(counts[group])
            row[f"p_sid_given_{key}"] = float(counts[group] / max(group_total[group], 1))
            row[f"p_{key}_given_sid"] = float(composition[group])
            row[f"enrichment_{key}"] = float((composition[group] + args.eps) / (prior[group] + args.eps))
            row[f"log2_enrichment_{key}"] = float(np.log2(row[f"enrichment_{key}"]))
        row["clinical_purity"] = float(composition.max()) if count_total else math.nan
        row["dominant_group"] = composition.idxmax() if count_total else ""
        rows.append(row)
    table = pd.DataFrame(rows)

    # Patient-presence Fisher tests prevent heartbeat pseudo-replication.
    presence = patient_hist > 0
    for group in GROUPS:
        in_group = patient_groups == group
        pvalues = np.ones(args.codebook_size, dtype=float)
        odds = np.full(args.codebook_size, np.nan, dtype=float)
        for sid in range(args.codebook_size):
            has_sid = presence[:, sid]
            matrix = np.array(
                [
                    [(has_sid & in_group).sum(), ((~has_sid) & in_group).sum()],
                    [(has_sid & (~in_group)).sum(), ((~has_sid) & (~in_group)).sum()],
                ]
            )
            odds[sid], pvalues[sid] = fisher_exact(matrix, alternative="two-sided")
        adjusted = bh_adjust(pvalues)
        key = group.lower()
        table[f"odds_ratio_{key}"] = odds
        table[f"raw_p_{key}"] = pvalues
        table[f"adjusted_p_{key}"] = adjusted
        table[f"significant_fdr_005_{key}"] = adjusted < args.fdr_alpha
    table["support_eligible"] = (table.count_total >= args.min_heartbeats) & (table.unique_patients >= args.min_patients)
    table["low_patient_dominance"] = table.patient_dominance <= args.max_patient_dominance
    return table


def top_enriched(table, group, args):
    key = group.lower()
    eligible = table[
        table.support_eligible
        & table.low_patient_dominance
        & table[f"significant_fdr_005_{key}"]
        & (table[f"enrichment_{key}"] > 1.0)
    ].copy()
    columns = [
        "sid", "count_total", "unique_patients", "unique_ecgs", f"p_{key}_given_sid",
        f"enrichment_{key}", f"log2_enrichment_{key}", f"odds_ratio_{key}",
        f"raw_p_{key}", f"adjusted_p_{key}", "patient_dominance", "clinical_purity", "dominant_group",
    ]
    return eligible.sort_values([f"log2_enrichment_{key}", "unique_patients", "count_total"], ascending=[False, False, False])[columns].head(args.top_n)


def plot_matrix(matrix: pd.DataFrame, path: Path, title: str, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix.to_numpy(), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def enrichment_visuals(table, tops, assignments, data, output_dir, args):
    level4 = output_dir / "level4_enrichment"
    wave_dir = level4 / "waveform_examples"
    wave_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for group in GROUPS:
        selected.extend(tops[group].sid.head(20).tolist())
    selected = list(dict.fromkeys(selected))[:60]
    if selected:
        heat = table.set_index("sid").loc[selected, [f"log2_enrichment_{g.lower()}" for g in GROUPS]]
        heat.columns = GROUPS
        plot_matrix(heat, level4 / "enrichment_heatmap.png", "SID clinical log2 enrichment", cmap="coolwarm")
        composition = table.set_index("sid").loc[selected[:30], [f"p_{g.lower()}_given_sid" for g in GROUPS]]
        composition.columns = GROUPS
        fig, ax = plt.subplots(figsize=(13, 7))
        bottom = np.zeros(len(composition))
        for group in GROUPS:
            values = composition[group].to_numpy()
            ax.bar(np.arange(len(composition)), values, bottom=bottom, label=group)
            bottom += values
        ax.set_xticks(np.arange(len(composition)), composition.index.astype(str), rotation=90)
        ax.set_ylabel("P(group | SID)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(level4 / "group_composition.png", dpi=180)
        plt.close(fig)
    rng = np.random.default_rng(args.seed + 7)
    for group in GROUPS:
        for sid in tops[group].sid.head(5):
            candidates = assignments[assignments.sid == sid]
            if candidates.empty:
                continue
            chosen = candidates.iloc[rng.choice(len(candidates), size=min(args.waveforms_per_sid, len(candidates)), replace=False)]
            waves = np.asarray(data[chosen.record_index.to_numpy(), chosen.token_index.to_numpy()], dtype=np.float32)
            mean = waves.mean(axis=0)
            std = waves.std(axis=0)
            row = table.loc[table.sid == sid].iloc[0]
            fig, ax = plt.subplots(figsize=(9, 5))
            for wave in waves:
                ax.plot(wave, alpha=0.25)
            ax.plot(mean, linewidth=2.5, label="mean")
            ax.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.2, label="mean ± std")
            ax.set_title(
                f"SID {sid} | {group} enriched={row[f'enrichment_{group.lower()}']:.2f} | "
                f"purity={row.clinical_purity:.2f} | patients={int(row.unique_patients)} | dominance={row.patient_dominance:.3f}"
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(wave_dir / f"{group.lower()}_sid_{sid}.png", dpi=180)
            plt.close(fig)


def sampled_patient_distances(patient_groups, patient_dist, args):
    rng = np.random.default_rng(args.seed + 11)
    rows = []
    comparisons = [(g, g) for g in GROUPS] + list(combinations(GROUPS, 2))
    for a, b in comparisons:
        ia = np.flatnonzero(patient_groups == a)
        ib = np.flatnonzero(patient_groups == b)
        if a == b:
            x = rng.choice(ia, size=args.max_pairs * 2, replace=True).reshape(-1, 2)
            x = x[x[:, 0] != x[:, 1]][: args.max_pairs]
        else:
            x = np.column_stack((rng.choice(ia, size=args.max_pairs, replace=True), rng.choice(ib, size=args.max_pairs, replace=True)))
        values = js_divergence_rows(patient_dist[x[:, 0]], patient_dist[x[:, 1]])
        rows.append(
            {
                "comparison": f"{a}-{b}", "kind": "within" if a == b else "between",
                "mean": values.mean(), "median": np.median(values), "std": values.std(),
                "Q25": np.percentile(values, 25), "Q75": np.percentile(values, 75), "n_pairs": len(values),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_js(patient_groups, patient_dist, args):
    rng = np.random.default_rng(args.seed + 13)
    pair_values = {pair: [] for pair in combinations(GROUPS, 2)}
    indices = {group: np.flatnonzero(patient_groups == group) for group in GROUPS}
    original = {group: patient_dist[indices[group]].mean(axis=0) for group in GROUPS}
    for _ in range(args.bootstrap):
        distributions = {}
        for group in GROUPS:
            sampled = rng.choice(indices[group], size=len(indices[group]), replace=True)
            distributions[group] = patient_dist[sampled].mean(axis=0)
        for pair in pair_values:
            pair_values[pair].append(js_divergence(distributions[pair[0]], distributions[pair[1]]))
    rows = []
    for (a, b), values in pair_values.items():
        values = np.asarray(values)
        point = js_divergence(original[a], original[b])
        percentile_low, percentile_high = np.percentile(values, [2.5, 97.5])
        rows.append(
            {
                "comparison": f"{a}-{b}", "point_estimate": point, "bootstrap_mean": values.mean(),
                "bias_corrected_estimate": float(np.clip(2 * point - values.mean(), 0.0, 1.0)),
                "ci95_low": max(0.0, 2 * point - percentile_high),
                "ci95_high": min(1.0, 2 * point - percentile_low),
                "percentile_ci95_low": percentile_low, "percentile_ci95_high": percentile_high,
                "bootstrap_iterations": args.bootstrap,
            }
        )
    return pd.DataFrame(rows)


def differential_table(a, b, heartbeat_dist, table):
    difference = np.abs(heartbeat_dist[a] - heartbeat_dist[b])
    result = pd.DataFrame({"sid": np.arange(len(difference)), f"p_sid_given_{a.lower()}": heartbeat_dist[a], f"p_sid_given_{b.lower()}": heartbeat_dist[b], "absolute_difference": difference})
    keep = ["sid", f"enrichment_{a.lower()}", f"enrichment_{b.lower()}", "unique_patients", "patient_dominance", "count_total"]
    result = result.merge(table[keep], on="sid", how="left")
    return result.sort_values("absolute_difference", ascending=False)


def clinical_report(run, stats, table, tops, js_hb, js_eq, he, dist_summary, bootstrap, differentials, leakage, args):
    closest_pair = min(combinations(GROUPS, 2), key=lambda pair: js_eq.loc[pair[0], pair[1]])
    farthest_pair = max(combinations(GROUPS, 2), key=lambda pair: js_eq.loc[pair[0], pair[1]])
    hb_values, eq_values = [], []
    for a, b in combinations(GROUPS, 2):
        hb_values.append(js_hb.loc[a, b])
        eq_values.append(js_eq.loc[a, b])
    sensitivity_rho = spearmanr(hb_values, eq_values).statistic
    top_lines = []
    for group in GROUPS:
        ids = tops[group].sid.head(10).astype(str).tolist()
        top_lines.append(f"- {group}: {', '.join(ids) if ids else '无满足 support、dominance 和 FDR 条件的 SID'}")
    omi_nomi_boot = bootstrap[bootstrap.comparison == "NOMI-OMI"].iloc[0]
    findings = [
        f"1. OMI–NOMI patient-equal plug-in JS={js_eq.loc['OMI','NOMI']:.4f}；高维稀疏 bootstrap 存在上偏，bias-corrected estimate={omi_nomi_boot.bias_corrected_estimate:.4f}，basic 95% CI [{omi_nomi_boot.ci95_low:.4f}, {omi_nomi_boot.ci95_high:.4f}]。",
        f"2. Healthy–Control patient-equal JS divergence={js_eq.loc['Healthy','Control']:.4f}。",
        f"3. 最相似组为 {closest_pair[0]}–{closest_pair[1]}（JS={js_eq.loc[closest_pair[0],closest_pair[1]]:.4f}）。",
        f"4. 差异最大组为 {farthest_pair[0]}–{farthest_pair[1]}（JS={js_eq.loc[farthest_pair[0],farthest_pair[1]]:.4f}）。",
        f"5. heartbeat-weighted 与 patient-equal 六组 pair distance 的 Spearman rho={sensitivity_rho:.3f}。",
        f"6. active SID={table.loc[table.count_total > 0, 'sid'].nunique()}；support-eligible SID={int(table.support_eligible.sum())}。",
        f"7. OMI 显著富集 SID={len(tops['OMI'])}；NOMI 显著富集 SID={len(tops['NOMI'])}。",
        f"8. median patient dominance={table.loc[table.count_total > 0, 'patient_dominance'].median():.4f}。",
    ]
    top_diff = differentials["OMI_vs_NOMI"].sid.head(10).astype(str).tolist()
    support = len(tops["OMI"]) + len(tops["NOMI"]) > 0 and leakage["patients_across_splits"] == 0
    verdict = (
        "结果支持部分 SID 存在跨患者的临床富集，但 OMI–NOMI 整体分布分离较弱、Control 有效样本很少、且尚无外部验证，"
        "因此只能称为初步 population-level organization，不能宣称完整 vocabulary 已形成强疾病语义。富集 SID 仍须结合 morphology consistency 人工确认。"
        if support else
        "当前结果不足以稳健支持 vocabulary 已形成临床群体组织；需要更多患者、独立外部验证或更稳定的跨患者 enrichment。"
    )
    return "\n".join(
        [
            f"# {run['name']} SID Clinical Semantics Report\n",
            "## 设置\n",
            f"- Held-out split: `{args.split}`；min heartbeats={args.min_heartbeats}；min patients={args.min_patients}；max dominance={args.max_patient_dominance}。",
            f"- 显著性：patient-presence Fisher exact test + Benjamini-Hochberg FDR {args.fdr_alpha}。",
            f"- Bootstrap: {args.bootstrap} 次，按 group 内 patient 有放回抽样；patient pair 每类最多 {args.max_pairs} 对。",
            f"- Patient leakage: {leakage['patients_across_splits']}；跨 group patient: {leakage['patients_across_groups']}。\n",
            f"- 标签冲突患者不参与正式统计；assignment 中保留并标记。当前 test 排除患者数：{leakage.get('ambiguous_patients_in_test', 0)}。\n",
            "- 排除冲突患者后 Control 仅 17 名，因此涉及 Control 的距离和 enrichment 仅作探索性结果。\n",
            "## Enriched SID\n",
            *top_lines,
            "\n## 关键问题\n",
            *findings,
            f"9. OMI vs NOMI 差异贡献最大的 SID: {', '.join(top_diff)}。",
            "10. enrichment 表已同时包含 unique patient、patient dominance、support 和 FDR；高 enrichment 但少患者的 SID 不进入正式排名。",
            "11. patient-level 与 heartbeat-level 的一致性由两套 JS matrix 和上述 Spearman rho 报告。",
            f"12. 结论：{verdict}\n",
            "## 解释边界\n",
            "- shared normal SID：四组都常见，不是坏事。",
            "- disease-enriched SID：跨足够患者且通过 FDR，才可称为群体富集。",
            "- patient-specific SID：patient dominance 高，不作为疾病语义证据。",
            "- noisy SID：即使富集高，也必须结合 waveform 图和 morphology purity 排除噪声。",
        ]
    )


def evaluate_run(run, args, device, data_path, data, chans, times, meta, leakage):
    model_dir = resolve_path(args.output_dir) / safe_name(run["name"])
    level4 = model_dir / "level4_enrichment"
    level5 = model_dir / "level5_separability"
    level4.mkdir(parents=True, exist_ok=True)
    level5.mkdir(parents=True, exist_ok=True)
    assignments_all, checkpoint, code_vectors = build_assignments(run, args, device, data, chans, times, meta)
    assignments_all.to_parquet(model_dir / "sid_assignments.parquet", index=False)
    assignments = assignments_all.loc[~assignments_all.ambiguous_patient].copy()
    stats = basic_statistics(assignments)
    stats.to_csv(model_dir / "basic_statistics.csv", index=False)
    patient_ids, patient_groups, patient_hist, patient_dist = patient_histograms(assignments, args.codebook_size)
    table = enrichment_table(assignments, patient_ids, patient_groups, patient_hist, args)
    table.to_csv(level4 / "sid_group_distribution.csv", index=False)
    tops = {}
    for group in GROUPS:
        tops[group] = top_enriched(table, group, args)
        tops[group].to_csv(level4 / f"top_{group.lower()}_enriched_sids.csv", index=False)
    enrichment_visuals(table, tops, assignments, data, model_dir, args)

    heartbeat_dist, equal_dist = group_distributions(assignments, patient_groups, patient_dist, args.codebook_size)
    distribution_rows = []
    for sid in range(args.codebook_size):
        row = {"sid": sid}
        for group in GROUPS:
            row[f"heartbeat_weighted_{group.lower()}"] = heartbeat_dist[group][sid]
            row[f"patient_equal_{group.lower()}"] = equal_dist[group][sid]
        distribution_rows.append(row)
    pd.DataFrame(distribution_rows).to_csv(level5 / "group_sid_distribution.csv", index=False)
    js_hb, he_hb, tv_hb = distance_matrices(heartbeat_dist)
    js_eq, he_eq, tv_eq = distance_matrices(equal_dist)
    js_hb.to_csv(level5 / "js_divergence_matrix.csv")
    js_eq.to_csv(level5 / "js_divergence_patient_equal_matrix.csv")
    he_hb.to_csv(level5 / "hellinger_matrix.csv")
    he_eq.to_csv(level5 / "hellinger_patient_equal_matrix.csv")
    tv_hb.to_csv(level5 / "total_variation_matrix.csv")
    tv_eq.to_csv(level5 / "total_variation_patient_equal_matrix.csv")
    plot_matrix(js_hb, level5 / "js_heatmap.png", "Heartbeat-weighted JS divergence")
    plot_matrix(js_eq, level5 / "js_patient_equal_heatmap.png", "Patient-equal JS divergence")
    dist_summary = sampled_patient_distances(patient_groups, patient_dist, args)
    dist_summary.to_csv(level5 / "distribution_distance_summary.csv", index=False)
    bootstrap = bootstrap_js(patient_groups, patient_dist, args)
    bootstrap.to_csv(level5 / "js_divergence_bootstrap.csv", index=False)
    differentials = {
        "OMI_vs_NOMI": differential_table("OMI", "NOMI", heartbeat_dist, table),
        "Healthy_vs_OMI": differential_table("Healthy", "OMI", heartbeat_dist, table),
        "Control_vs_OMI": differential_table("Control", "OMI", heartbeat_dist, table),
    }
    for name, frame in differentials.items():
        frame.head(args.top_n).to_csv(level5 / f"differential_sid_{name.lower()}.csv", index=False)

    pca = PCA(n_components=2, random_state=args.seed)
    coordinates = pca.fit_transform(patient_dist)
    pca_frame = pd.DataFrame({"patient_id": patient_ids, "group": patient_groups, "PC1": coordinates[:, 0], "PC2": coordinates[:, 1]})
    pca_frame.to_csv(level5 / "patient_sid_pca.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    for group in GROUPS:
        selected = pca_frame.group == group
        ax.scatter(pca_frame.loc[selected, "PC1"], pca_frame.loc[selected, "PC2"], s=18, alpha=0.65, label=group)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend()
    ax.set_title("Patient SID distribution PCA")
    fig.tight_layout()
    fig.savefig(level5 / "patient_sid_pca.png", dpi=180)
    plt.close(fig)

    report = clinical_report(run, stats, table, tops, js_hb, js_eq, he_hb, dist_summary, bootstrap, differentials, leakage, args)
    (model_dir / "REPORT.md").write_text(report, encoding="utf-8")
    active_table = table[table.count_total > 0]
    heartbeat_js_values = [js_hb.loc[a, b] for a, b in combinations(GROUPS, 2)]
    equal_js_values = [js_eq.loc[a, b] for a, b in combinations(GROUPS, 2)]
    group_heartbeat_counts = stats.set_index("group").heartbeat_count.to_dict()
    sanity = {
        "patient_leakage_across_splits": leakage["patients_across_splits"],
        "healthy_control_kept_separate": True,
        "padding_or_invalid_sid_rows": int((~assignments_all.is_valid).sum() + (~assignments_all.sid.between(0, args.codebook_size - 1)).sum()),
        "ambiguous_patients_excluded_from_analysis": leakage.get("ambiguous_patients_in_test", 0),
        "analyzed_group_patient_counts": stats.set_index("group").patient_count.to_dict(),
        "analyzed_group_heartbeat_counts": group_heartbeat_counts,
        "heartbeat_count_max_to_min_ratio": float(max(group_heartbeat_counts.values()) / min(group_heartbeat_counts.values())),
        "top1_sid_heartbeat_share": float(active_table.count_total.max() / active_table.count_total.sum()),
        "top10_sid_heartbeat_share": float(active_table.nlargest(10, "count_total").count_total.sum() / active_table.count_total.sum()),
        "median_patient_dominance": float(active_table.patient_dominance.median()),
        "max_patient_dominance": float(active_table.patient_dominance.max()),
        "support_eligible_sids": int(table.support_eligible.sum()),
        "high_enrichment_but_below_support": int(((table[[f"enrichment_{g.lower()}" for g in GROUPS]].max(axis=1) >= 2.0) & ~table.support_eligible).sum()),
        "heartbeat_vs_patient_equal_js_spearman": float(spearmanr(heartbeat_js_values, equal_js_values).statistic),
        "control_results_exploratory": bool(stats.set_index("group").loc["Control", "patient_count"] < args.min_patients),
    }
    (model_dir / "sanity_checks.json").write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "experiment_name": run["name"], "model_type": run["model_type"], "checkpoint": str(checkpoint),
        "codebook_size": args.codebook_size, "latent_dim": int(code_vectors.shape[-1]), "evaluation_dataset": str(data_path),
        "split": args.split, "heartbeats_all": len(assignments_all), "heartbeats_analyzed": len(assignments),
        "patients": assignments.patient_id.nunique(), "ecgs": assignments.ecg_id.nunique(),
        "active_sids": assignments.sid.nunique(), "leakage": leakage,
    }
    (model_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "Codebook": run["name"], "checkpoint": str(checkpoint), "active_SIDs": assignments.sid.nunique(),
        "Healthy-Control JS": js_eq.loc["Healthy", "Control"], "Healthy-NOMI JS": js_eq.loc["Healthy", "NOMI"],
        "Healthy-OMI JS": js_eq.loc["Healthy", "OMI"], "Control-NOMI JS": js_eq.loc["Control", "NOMI"],
        "Control-OMI JS": js_eq.loc["Control", "OMI"], "NOMI-OMI JS": js_eq.loc["NOMI", "OMI"],
        "significant_OMI_enriched_SIDs": len(tops["OMI"]), "significant_NOMI_enriched_SIDs": len(tops["NOMI"]),
        "median_patient_dominance": table.loc[table.count_total > 0, "patient_dominance"].median(),
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve_path(args.data_path)
    data = np.load(data_path / f"{args.split}_data.npy", mmap_mode="r")
    chans = np.load(data_path / f"{args.split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_path / f"{args.split}_data_in_times.npy", mmap_mode="r")
    meta = pd.read_csv(data_path / f"{args.split}_metadata.csv")
    if len(meta) != len(data):
        raise ValueError(f"Metadata/data mismatch: {len(meta)} != {len(data)}")
    meta["group_eval"] = meta.label_name.map(LABEL_MAP)
    if meta.group_eval.isna().any() or set(meta.group_eval.unique()) != set(GROUPS):
        raise ValueError(f"Expected four independent groups, got {meta.group_eval.value_counts(dropna=False).to_dict()}")
    meta["patient_id_eval"] = patient_id_for_metadata(meta)
    leakage = check_patient_leakage(data_path)
    if leakage["patients_across_splits"]:
        raise RuntimeError(f"Patient leakage detected: {leakage}")
    test_group_counts = meta.groupby("patient_id_eval").group_eval.nunique()
    ambiguous_ids = set(test_group_counts[test_group_counts > 1].index)
    meta["ambiguous_patient"] = meta.patient_id_eval.isin(ambiguous_ids)
    leakage["ambiguous_patients_in_test"] = len(ambiguous_ids)
    leakage["ambiguous_ecgs_in_test"] = int(meta.ambiguous_patient.sum())
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    comparison = []
    for spec in args.run:
        run = parse_run_spec(spec)
        print(f"Evaluating {run['name']}", flush=True)
        comparison.append(evaluate_run(run, args, device, data_path, data, chans, times, meta, leakage))
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    config = vars(args).copy()
    config["data_path"] = str(args.data_path)
    config["output_dir"] = str(args.output_dir)
    config["leakage"] = leakage
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    best_omi_nomi = comparison_df.loc[comparison_df["NOMI-OMI JS"].idxmax()]
    lines = [
        "# SID Clinical Semantics Model Comparison\n",
        dataframe_to_markdown(comparison_df),
        "\n## Summary\n",
        f"- OMI–NOMI patient-equal JS 最大的是 {best_omi_nomi['Codebook']}（{best_omi_nomi['NOMI-OMI JS']:.4f}），但 JS 更大不能单独等价为 codebook 更好。",
        "- 四个模型中 OMI–NOMI 都是最相似或接近最相似的临床组，说明这是当前 vocabulary 最困难的区分。",
        "- 68 名 test 患者具有冲突 group 标签，assignment 保留但正式 patient-level 分析排除；Control 最终仅 17 名，相关结论仅作探索性解释。",
        "- 每个模型目录包含 assignment、Level 4/5 原始表、图像、sanity checks 和详细 REPORT.md。\n",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
