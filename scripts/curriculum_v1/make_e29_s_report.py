"""Report the fixed E29-S comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_e29_s import OUT,e

def load(n):return json.loads((OUT/n).read_text())

def table_pair(lines,title,a,b,ci):
    lines += ["",title,"","|指标|E25|Shared|差值|95% 实体区间|","|---|---:|---:|---:|---|"]
    for k in ("overall","clinical3","OMI","NOMI","Control","Healthy","patient_raw3_r2","patient_conditional3_r2","omi_nomi_auc","macro_auc","macro_f1","brier","log_loss","omi_sensitivity_at_90pct_specificity","nomi_sensitivity_at_90pct_omi_sensitivity"):
        interval=ci.get(k,{}).get("ci95")
        lines.append(f"|{k}|{a[k]:.6f}|{b[k]:.6f}|{b[k]-a[k]:+.6f}|{interval or '—'}|")

def main():
    assert (OUT/"status.txt").read_text().strip()=="complete"
    h,c,u=map(load,("historical_ensembles.json","cv_summary.json","uniform_ablation.json"))
    overlap=load("pretraining_overlap.json")
    verified=load("pretraining_membership_verified.json")
    lines=["# E29-S Shared Query 泛化复核","",
      "E12-A、tokenizer、E11 和两个读取方案均固定。历史 Validation 由 6 个现有模型等权集成。内部 CV 使用原 Train 的 3 个外层患者/健康实体分组，每个外层开发集再按组划出内层选轮集。每折各从头训练两个读取头，每方法三个下游 seed，每次 30 轮。按内层 Overall R² 选 checkpoint，外层只用于 OOF 评价。",
      "单模型概率取 softmax。Clinical-3 是全体 ECG 上前三个 one-hot 输出 R² 的均值。患者子集的 raw3 和 conditional3 仅评估前三列，后者在患者子集上用前三类概率之和重新归一化。OMI/NOMI AUC 使用 pOMI/(pOMI+pNOMI)，只在真实 OMI/NOMI 样本计算。固定点以 ROC 中满足约束的最大敏感度报告。每条 ECG 等权，实体 bootstrap 每次成簇配对抽取，2000 次。","",
      f"预训练 Train 中按 ECG 文件名匹配到 {overlap['matched_ecg_rows']} 条下游 Train ECG，对应 {sum(v['train']['downstream_entities_with_matched_ecg'] for v in verified.values())} 个下游实体。预训练清单不含 patient_sn，实体与文件名复核见 pretraining_membership_verified.json。E12-A 使用 Masked SID 目标，CV 外层记录可能已作为无标签输入参与预训练。结果属于固定预训练下的内部监督复核。","",
      "|历史组合|E25 单 seed Overall mean±SD|Shared 单 seed Overall mean±SD|E25 集成增益|Shared 集成增益|","|---|---:|---:|---:|---:|"]
    for n,v in h.items():
        a,b=v["e25"],v["shared"]
        lines.append(f"|{n}|{a['mean']['overall']:.6f} ± {a['sd']['overall']:.6f}|{b['mean']['overall']:.6f} ± {b['sd']['overall']:.6f}|{a['per_output_ensemble_gain']['overall']:+.6f}|{b['per_output_ensemble_gain']['overall']:+.6f}|")
    for n,v in h.items():
        table_pair(lines,f"## 历史 Validation 组合 {n}",v["e25"]["ensemble"],v["shared"]["ensemble"],v["bootstrap"])
    a,b=c["learned_mix"],c["shared"]
    lines+=["","## Train-only CV","","|方案|OOF 单 seed Overall mean±SD|Clinical-3 mean±SD|条件 OMI/NOMI AUC mean±SD|OOF 集成增益|","|---|---:|---:|---:|---:|"]
    for m,v in (("E25",a),("Shared",b)):
        lines.append("|"+m+"|"+"|".join(f"{v['mean'][k]:.6f} ± {v['sd'][k]:.6f}" for k in ("overall","clinical3","omi_nomi_auc"))+f"|{v['per_output_ensemble_gain']['overall']:+.6f}|")
    lines+=["","|外层折|Overall R² 差|Clinical-3 差|条件 OMI/NOMI AUC 差|","|---|---:|---:|---:|"]
    for f,v in c["fold_ensemble"].items():
        lines.append("|"+f+"|"+"|".join(f"{v['shared'][k]-v['learned_mix'][k]:+.6f}" for k in ("overall","clinical3","omi_nomi_auc"))+"|")
    table_pair(lines,"## 固定 OOF 集成",a["ensemble"],b["ensemble"],c["ensemble_bootstrap"])
    lines+=["","## 注意力行为",""]
    df=pd.read_csv(OUT/"attention_diagnostics.csv")
    valid=df.valid_tokens>1
    lines.append(f"有效多 token 记录的平均归一化 entropy 为 {df.loc[valid,'normalized_entropy'].mean():.6f}，平均最大 token 权重为 {df.maximum_mass.mean():.6f}。")
    exposure=pd.read_csv(OUT/"attention_exposure_adjusted.csv")
    cols=[k for k in exposure if k.endswith("_excess")]
    if cols:
        overall=exposure[cols].mean()
        lines.append("按 seed 和类别等权汇总的相对均匀权重超额质量 " + ", ".join(f"{k} {v:+.4f}" for k,v in overall.items()))
    for key in ("valid_tokens","padding_fraction","trimmed_rr_hr","possible_asymmetric_alignment_fraction"):
        if key in df:
            lines.append(f"{key} 与归一化 entropy 的 Spearman ρ 为 {df[[key,'normalized_entropy']].corr(method='spearman').iloc[0,1]:.4f}。")
    y=b["ensemble"]
    lines += [f"历史六 seed Shared 固定权重的 Overall R² 为 {u['learned']['overall']:.6f}，替换为有效 token 均匀权重后为 {u['uniform']['overall']:.6f}。学习权重减均匀权重的配对实体区间为 {u['learned_minus_uniform']['overall']['ci95']}。",
        "注意力按有效 token、实测导联和时间块汇总。心率与对齐变量基于 E27 检测峰，后者是 RR 不对称几何代理。相关性是描述性模型行为，不代表病灶定位。","",
        "历史和 OOF 的逐类别集成增益、seed 残差相关矩阵、概率分散和 Brier 抵消均见 JSON。OOF bootstrap 固定已有预测，未重复训练，区间不覆盖训练划分和独立预训练的方差。历史 Validation 曾参与多轮开发。"]
    (OUT/"E29_S_REPORT_CN.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
