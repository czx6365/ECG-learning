#!/usr/bin/env python3
"""E1 三种子汇总与患者实体配对 bootstrap。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
BASE = ROOT / "results/ecg_representation_comparison"
MAIN = BASE / "experiment1_dual_encoder"
CAP = BASE / "experiment1_capacity_control"
METHODS = ["E1-0_CLS", "E1-1_z_e", "E1-2_CLS_z_e", "E1-3_CLS_z_q", "E1-0_CLS_matched"]
NAMES = ["OMI", "NOMI", "Control", "Healthy"]


def folder(task, method, seed):
    if method.endswith("_matched"):
        return CAP / task / "E1-0_CLS" / f"seed{seed}"
    return MAIN / task / method / f"seed{seed}"


def metrics_array(task, method, seed):
    record = json.loads((folder(task, method, seed) / "summary.json").read_text())
    return record


def prediction(task, method, seed):
    return pd.read_csv(folder(task, method, seed) / "test_predictions.csv")


def score(task, frame):
    if task == "stenosis":
        return float(r2_score(frame["target_overall_max_stenosis"], frame["prediction"]))
    logits = frame[[f"logit_{name}" for name in ["OMI", "NOMI", "Clinical Control", "Healthy"]]].to_numpy()
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    truth = frame["label_id"].to_numpy()
    return float(np.mean([roc_auc_score(truth == k, prob[:, k]) for k in range(4)]))


def paired_bootstrap(task, candidate, reference, repeats=500):
    frames = {(m, seed): prediction(task, m, seed) for m in (candidate, reference) for seed in range(3)}
    anchor = frames[(reference, 0)]
    for frame in frames.values():
        assert len(frame) == len(anchor)
        assert frame["entity_id"].equals(anchor["entity_id"])
        assert frame["label_id"].equals(anchor["label_id"])
    ids = anchor["entity_id"].astype(str).to_numpy()
    entities = np.unique(ids)
    locations = {name: np.flatnonzero(ids == name) for name in entities}
    rng = np.random.default_rng(20261016)
    diffs = []
    for _ in range(repeats):
        sampled = rng.choice(entities, len(entities), replace=True)
        selected = np.concatenate([locations[name] for name in sampled])
        values = {}
        try:
            for method in (candidate, reference):
                values[method] = np.mean([score(task, frames[(method, seed)].iloc[selected]) for seed in range(3)])
        except ValueError:
            continue
        diffs.append(float(values[candidate] - values[reference]))
    point = np.mean([score(task, frames[(candidate, seed)]) - score(task, frames[(reference, seed)]) for seed in range(3)])
    return {"point_delta": float(point), "ci95": np.quantile(diffs, [0.025, 0.975]).tolist(),
            "valid_resamples": len(diffs), "entity_count": len(entities)}


def main():
    if not (MAIN / "COMPLETE").exists() or not (CAP / "COMPLETE").exists():
        raise RuntimeError("E1 主实验或参数匹配对照尚未完成")
    result = {"methods": {}, "paired_entity_bootstrap": {}}
    for task in ("classification", "stenosis"):
        result["methods"][task] = {}
        for method in METHODS:
            rows = [metrics_array(task, method, seed) for seed in range(3)]
            keys = (["macro_auc_ovr", "macro_f1", "balanced_accuracy", "accuracy", "omi_nomi_auc"]
                    if task == "classification" else ["r2", "rmse", "mae", "pearson", "spearman"])
            result["methods"][task][method] = {
                "trainable_parameters": rows[0]["trainable_parameters"],
                "best_epochs": [r["best_epoch"] for r in rows],
                "test_n": rows[0]["test"]["n"],
                "metrics": {k: {"mean": float(np.mean([r["test"][k] for r in rows])),
                                "std_ddof1": float(np.std([r["test"][k] for r in rows], ddof=1)),
                                "seeds": [r["test"][k] for r in rows]}
                            for k in keys},
                "val_primary_mean": float(np.mean([r["val"]["macro_auc_ovr" if task == "classification" else "r2"] for r in rows]))}
        for method in ("E1-2_CLS_z_e", "E1-3_CLS_z_q"):
            for baseline in ("E1-0_CLS", "E1-0_CLS_matched"):
                key = f"{method}_minus_{baseline}"
                result["paired_entity_bootstrap"][f"{task}:{key}"] = paired_bootstrap(task, method, baseline)
    out = BASE / "experiment1_summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E1：双 Encoder 冻结特征融合正式结果", "",
             "固定 E3 tokenizer(epoch30)、Sentence MLM(epoch60)、原患者级 6640/1344/1366 划分。分类使用全体样本；狭窄回归仅使用原 split 中有 `overall_max_stenosis` 的患者（3320/672/683），健康人没有该标签。三个 seed 均按验证集选模，测试集只用于报告。", "",
             "训练头统一为输入标准化（仅训练集拟合）→LayerNorm→Linear→GELU→Dropout(0.1)→Linear；hidden256、30epoch、AdamW lr1e-3、5轮warmup。分类 CE 按 val Macro OVR AUC 选模；狭窄 SmoothL1 β=2（原始 0–100 单位）按 val R² 选模。", ""]
    for task in ("classification", "stenosis"):
        lines += [f"## {'四分类' if task == 'classification' else '患者冠脉最大狭窄回归'}", ""]
        primary = "macro_auc_ovr" if task == "classification" else "r2"
        second = "macro_f1" if task == "classification" else "rmse"
        lines += [f"| 方法 | 参数量 | val {primary} | test {primary} (均值±SD) | test {second} |", "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            r = result["methods"][task][method]
            p, s = r["metrics"][primary], r["metrics"][second]
            lines.append(f"| {method} | {r['trainable_parameters']:,} | {r['val_primary_mean']:.6f} | {p['mean']:.6f} ± {p['std_ddof1']:.6f} | {s['mean']:.6f} |")
        lines += ["", "配对实体 bootstrap（500 次，三 seed 指标差的均值；95% 百分位区间）：", ""]
        for method in ("E1-2_CLS_z_e", "E1-3_CLS_z_q"):
            for baseline in ("E1-0_CLS", "E1-0_CLS_matched"):
                r = result["paired_entity_bootstrap"][f"{task}:{method}_minus_{baseline}"]
                lines.append(f"- {method} − {baseline}: {r['point_delta']:+.6f}，CI [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]。")
        lines.append("")
    lines += ["## 解释边界", "",
              "CLS+z_e 的分类提升需与参数匹配 CLS 对照一起看；参数匹配并非完全控制所有优化与归纳偏置差异。狭窄回归所有模型的 test R² 仍为负值，不能称已获得可靠预测。E1 使用 E3 既有 checkpoint 和缓存，不等于 E2 的四架构公平预训练。历史项目已多次查看同一 test，bootstrap 仅描述该固定队列的不确定性，不是外部验证。", ""]
    (BASE / "E1双Encoder实验报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
