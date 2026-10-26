#!/usr/bin/env python3
"""Create one machine-readable and one concise Chinese C5b-lite report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINES = {
    "C5a_e30": {"four_r2": 0.24659894240493696, "binary_r2": 0.4415223667932045},
    "C5a_e50": {"four_r2": 0.24676890925413172, "binary_r2": 0.44036822056159364},
    "C2.2b_ensemble": {"four_r2": 0.24806343624440427, "binary_r2": 0.4406415147419437},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.checkpoint_dir / "test_regression_summary.json").read_text())
    tokenizer = json.loads((args.run_dir / "tokenizer_comparison.json").read_text())
    summary = {
        "experiment": "C5b-lite continuous residual bridge + staged unfreezing + slow EMA codebook",
        "test": result,
        "baselines": BASELINES,
        "delta_vs_baselines": {
            name: {
                "four_r2": result["r2"] - values["four_r2"],
                "binary_r2": result["binary"]["r2"] - values["binary_r2"],
            }
            for name, values in BASELINES.items()
        },
        "tokenizer": tokenizer,
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    p = result["per_output_r2"]
    b = result["binary"]["per_output_r2"]
    c21 = tokenizer["C2.1_pattern_shift"]
    c5b = tokenizer["C5b_lite_joint"]
    lines = [
        "# C5b-lite 完整结果",
        "",
        f"- 最佳轮次：{result['best_epoch']}；验证 Overall R²：{result['best_val_r2']:.9f}",
        f"- 测试 Overall R²：{result['r2']:.9f}",
        f"- OMI/NOMI/Control/Healthy：{p['OMI']:.9f} / {p['NOMI']:.9f} / {p['Clinical Control']:.9f} / {p['Healthy']:.9f}",
        f"- Disease/Healthy/二输出Overall：{b['Disease']:.9f} / {b['Healthy']:.9f} / {result['binary']['r2']:.9f}",
        f"- MSE/MAE：{result['mse']:.9f} / {result['mae']:.9f}",
        "",
        "## 与主要基线比较",
        "",
    ]
    for name, delta in summary["delta_vs_baselines"].items():
        lines.append(f"- {name}：四输出 Δ={delta['four_r2']:+.9f}；二输出 Δ={delta['binary_r2']:+.9f}")
    lines += [
        "",
        "## Tokenizer",
        "",
        f"- Active SID：C2.1 {c21['active_sid']} → C5b {c5b['active_sid']}",
        f"- Perplexity：C2.1 {c21['perplexity']:.6f} → C5b {c5b['perplexity']:.6f}",
        f"- ±2/±4/±6 disagreement：C2.1 {c21['sid_disagreement']}；C5b {c5b['sid_disagreement']}",
        f"- Morphology purity：C2.1 {c21['morphology_purity']}；C5b {c5b['morphology_purity']}",
    ]
    (args.run_dir / "RESULTS_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
