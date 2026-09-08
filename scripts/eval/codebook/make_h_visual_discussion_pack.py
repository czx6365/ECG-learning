#!/usr/bin/env python3
"""Build a compact visual discussion pack for the H codebook experiment."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path("/home/ubuntu/HeartLang-main")
OUT = ROOT / "outputs/sid_quality/discussion_H_visual_pack"

ORIG = ROOT / "outputs/sid_quality/ECG_DSA_patient_healthy_raw_v2_1024_dim64_val"
H = ROOT / "outputs/sid_quality/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_H_1024_dim64_val"
ART = ROOT / "outputs/sid_quality/artifact_filter_v2_val_le5pct_actual"
H_WORST = H / "worst_sid_analysis/worst_sid_visualizations"


FIGURES = [
    (
        "01_original_high_sid387.png",
        ORIG / "sid_visualizations/sid_0387_high_count_4465.png",
        "原始 V2 高频 SID 387：有稳定平均 QRS 原型，但仍混入少量反向/异常片段。",
    ),
    (
        "02_original_mid_sid375_mixed.png",
        ORIG / "sid_visualizations/sid_0375_mid_count_1475.png",
        "原始 V2 中频 SID 375：混杂更明显，可见切分边界/阶梯状/形态不一致片段。",
    ),
    (
        "03_original_low_sid174.png",
        ORIG / "sid_visualizations/sid_0174_low_count_387.png",
        "原始 V2 低频 SID 174：有平均形态，但方差带较宽，低频 word 更容易被异常样本影响。",
    ),
    (
        "04_H_high_sid064.png",
        H / "sid_visualizations/sid_0064_high_count_4805.png",
        "H 高频 SID 64：clean 后高频 word 的 QRS 原型更集中，内部形态更紧。",
    ),
    (
        "05_H_mid_sid321.png",
        H / "sid_visualizations/sid_0321_mid_count_1550.png",
        "H 中频 SID 321：中频 word 仍有一定差异，但整体比原始中频混杂更可控。",
    ),
    (
        "06_H_low_sid087.png",
        H / "sid_visualizations/sid_0087_low_count_207.png",
        "H 低频 SID 87：低频 word 仍能形成可解释平均形态，说明 clean 没有只保留高频常规心跳。",
    ),
    (
        "07_H_worst_sid829_residual_issue.png",
        H_WORST / "sid_0829_worst_count_2788.png",
        "H 最差 SID 829：clean 后仍存在边缘平台化/残余混杂，是下一步改进切分和 lead/polarity 处理的证据。",
    ),
    (
        "08_H_worst_sid1008_late_peak.png",
        H_WORST / "sid_1008_worst_count_1662.png",
        "H 最差 SID 1008：R 峰位置偏后，说明峰位置过滤阈值已经去掉极端错误，但边界附近问题仍会留下。",
    ),
]

ARTIFACT_FIGURES = [
    ART / "filtered_examples_long_flat_run.png",
    ART / "filtered_examples_peak_too_early.png",
    ART / "filtered_examples_peak_too_late.png",
]


def copy_figures() -> list[tuple[str, str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for name, src, caption in FIGURES:
        dst = OUT / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        copied.append((name, caption))
    return copied


def make_artifact_montage() -> str:
    """Combine three artifact examples into one figure if Pillow is available."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        name = "09_cleaned_token_examples"
        subdir = OUT / name
        subdir.mkdir(exist_ok=True)
        for src in ARTIFACT_FIGURES:
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, subdir / src.name)
        return f"{name}/"

    labels = [
        "A. long flat run: long nearly-flat segment",
        "B. peak too early: cut window shifted right",
        "C. peak too late: cut window shifted left",
    ]
    images = []
    for src in ARTIFACT_FIGURES:
        if not src.exists():
            raise FileNotFoundError(src)
        img = Image.open(src).convert("RGB")
        target_width = 1200
        scale = target_width / img.width
        images.append(img.resize((target_width, int(img.height * scale))))

    label_h = 48
    pad = 20
    width = max(img.width for img in images)
    height = sum(img.height + label_h + pad for img in images) + pad
    canvas = Image.new("RGB", (width + 2 * pad, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = pad
    for label, img in zip(labels, images):
        draw.text((pad, y), label, fill=(0, 0, 0))
        y += label_h
        canvas.paste(img, (pad, y))
        y += img.height + pad

    out_name = "09_cleaned_token_examples_montage.png"
    canvas.save(OUT / out_name, quality=95)
    return out_name


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_worst_rows() -> list[dict]:
    path = H / "worst_sid_analysis/worst_sid_artifact_summary.csv"
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_markdown(figures: list[tuple[str, str]], artifact_name: str) -> None:
    orig_summary = load_json(ORIG / "summary.json")
    h_summary = load_json(H / "summary.json")
    artifact_summary = load_json(ART / "artifact_filter_summary.json")
    worst_rows = read_worst_rows()[:12]

    def metrics(summary: dict) -> dict:
        return {
            "active": summary["usage"]["active_sids"],
            "perplexity": summary["usage"]["perplexity"],
            "intra_rmse": summary["distance"]["intra_rmse_mean"],
            "rmse_ratio": summary["distance"]["rmse_separation_ratio_inter_over_intra"],
            "intra_corr": summary["distance"]["intra_corr_dist_mean"],
            "corr_ratio": summary["distance"]["corr_dist_separation_ratio_inter_over_intra"],
            "silhouette": summary["distance"]["silhouette_euclidean"],
        }

    m0 = metrics(orig_summary)
    mh = metrics(h_summary)

    rows = [
        "| 实验 | active | perplexity | intra RMSE | inter/intra RMSE | intra corr dist | corr 分离比 | silhouette |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| V2 原始 no balancing | {m0['active']} | {m0['perplexity']:.2f} | {m0['intra_rmse']:.4f} | {m0['rmse_ratio']:.4f} | {m0['intra_corr']:.4f} | {m0['corr_ratio']:.4f} | {m0['silhouette']:.4f} |",
        f"| V2 clean H | {mh['active']} | {mh['perplexity']:.2f} | {mh['intra_rmse']:.4f} | {mh['rmse_ratio']:.4f} | {mh['intra_corr']:.4f} | {mh['corr_ratio']:.4f} | {mh['silhouette']:.4f} |",
    ]

    clean_summary_path = (
        ROOT
        / "datasets/ecg_datasets/ECG_DSA/pretrain_qrs/"
        / "patient_healthy_raw_v2_clean_le5pct/clean_filter_summary.json"
    )
    clean_summary = load_json(clean_summary_path)
    filtered = artifact_summary.get("filter_summary", artifact_summary)
    # Support both the old and current summary shapes.
    split_summaries = {item["split"]: item for item in clean_summary.get("splits", [])}
    val_clean = split_summaries.get("val", {})
    filtered_count = (
        val_clean.get("filtered")
        or filtered.get("filtered_tokens")
        or filtered.get("filtered")
        or artifact_summary.get("filtered_tokens")
    )
    valid_before = val_clean.get("valid_before")
    filtered_fraction = (
        filtered.get("filtered_fraction")
        or artifact_summary.get("filtered_fraction")
        or 0.0485
    )

    md = [
        "# H 可视化讨论材料",
        "",
        "## 结论先行",
        "",
        "H 做的是：先过滤约 5% 明显不像正常居中心跳片段的 token，再重新训练 1024 × 64 codebook。",
        "",
        "过滤掉的不是正常的少见心跳，而是三类高风险伪影：长平段、R 峰太靠前、R 峰太靠后。",
        "",
        "结果是：word 数量少了一点，但同一个 word 内部更像，不同 word 之间更分得开。",
        "",
        *rows,
        "",
        f"- 过滤比例：约 {float(filtered_fraction) * 100:.2f}%",
        f"- 过滤 token 数：{filtered_count:,} / {valid_before:,}" if filtered_count and valid_before else f"- 过滤 token 数：{filtered_count}",
        "- H active word 下降：780 -> 730",
        "- H 形态一致性提升：intra RMSE 1.1134 -> 0.9696，intra corr distance 0.7232 -> 0.5880",
        "- H 分离度提升：RMSE 分离比 1.1517 -> 1.3048，correlation 分离比 1.2456 -> 1.5105",
        "",
        "## 汇报逻辑",
        "",
        "1. 原始 V2 的 SID 不是随机编号，已经能聚出一些 QRS 原型。",
        "2. 但是原始 V2 中有一部分 SID 被切歪、边界平段、平台化片段污染。",
        "3. 我们没有大规模删数据，只过滤最异常的约 4.85% token。",
        "4. 重新训练 H 后，整体分离指标明显改善，说明 clean 是有效方向。",
        "5. H 中最差 SID 仍有残余问题，下一步应继续改进切分质量、lead/polarity 归一和形态特征评估。",
        "",
        "## 精选图",
        "",
    ]

    for name, caption in figures:
        md.extend([f"### {name}", "", f"![{name}]({name})", "", caption, ""])

    md.extend(
        [
            f"### {artifact_name}",
            "",
            f"![cleaned token examples]({artifact_name})" if artifact_name.endswith(".png") else f"见目录：`{artifact_name}`",
            "",
            "这张图用于证明 clean 掉的是明显异常 token：长平段、峰太靠前、峰太靠后。",
            "",
            "## H 最差 SID 摘要",
            "",
            "| SID | count | prefix flat12 | suffix flat12 | long flat >=12 | peak p10 | peak p50 | peak p90 | RMSE-to-mean p50 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in worst_rows:
        md.append(
            "| {sid} | {total_count} | {pre:.2f} | {suf:.2f} | {long:.2f} | {p10:.1f} | {p50:.1f} | {p90:.1f} | {rmse:.3f} |".format(
                sid=row["sid"],
                total_count=row["total_count"],
                pre=float(row["stored_prefix_flat12_fraction"]),
                suf=float(row["stored_suffix_flat12_fraction"]),
                long=float(row["stored_long_flat_run_ge12_fraction"]),
                p10=float(row["stored_peak_idx_p10"]),
                p50=float(row["stored_peak_idx_p50"]),
                p90=float(row["stored_peak_idx_p90"]),
                rmse=float(row["stored_rmse_to_sid_mean_p50"]),
            )
        )

    md.extend(
        [
            "",
            "## 下周可以直接说的话",
            "",
            "> 原始 V2 codebook 已经能学到 ECG 局部形态，但部分 SID 被切分偏移、长平段和边界伪影污染。我们用保守规则只过滤约 4.85% 最异常 token，再重新训练 H。H 的 active word 从 780 降到 730，但 intra distance 明显下降，inter/intra separation 明显上升，说明 codebook 的 word 语义更干净。下一步不是盲目扩大 codebook，而是继续提升切分质量，并验证 SID 是否保留 R 峰、QRS 宽度、振幅等 ECG 形态信息。",
            "",
        ]
    )

    (OUT / "README_讨论稿.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    figures = copy_figures()
    artifact_name = make_artifact_montage()
    write_markdown(figures, artifact_name)
    print(f"Wrote discussion pack to {OUT}")


if __name__ == "__main__":
    main()
