"""Build E29-R report from verified Validation outputs."""
import json
from pathlib import Path
import sys
import pandas as pd

r=Path(sys.argv[1])
d=pd.read_csv(r/"per_seed_metrics.csv")
ens=json.loads((r/"ensemble_metrics.json").read_text())
boot=json.loads((r/"paired_bootstrap.json").read_text())
div=json.loads((r/"ensemble_diversity.json").read_text())
err=pd.read_csv(r/"error_transition_analysis.csv")
transition=pd.crosstab(err.truth,err.transition)
transition.to_csv(r/"error_transition_counts.csv")
lines=["# E29-R 验证集复核","","历史概率文件不带行 ID。九份概率均通过逐行 checkpoint 重放和 softmax 重算校验。按固定元数据顺序校验标签，每条 ECG 等权，bootstrap 按实体整簇配对抽样。","","|方法|历史 seed 均值 Overall R²|历史集成 Overall R²|集成增益|历史 seed 均值 Clinical-3|历史集成 Clinical-3|条件 OMI/NOMI 集成 AUC|","|---|---:|---:|---:|---:|---:|---:|"]
for m,v in div.items():
    a=ens[m]
    avg=d[(d.method==m)&(d.seed<3)].clinical3.mean()
    lines.append(f"|{m}|{v['seed_mean_r2']:.6f}|{v['ensemble_r2']:.6f}|{v['ensemble_gain']:+.6f}|{avg:.6f}|{a['clinical3']:.6f}|{a['omi_nomi_auc']:.6f}|")
lines+=["","## 新旧 seed 的配对结果","","|seed|Overall R² 差|Clinical-3 R² 差|条件 OMI/NOMI AUC 差|","|---:|---:|---:|---:|"]
for seed,g in d[d.method.isin(["e25","shared"])].groupby("seed"):
    a=g.set_index("method")
    v=a.loc["shared",["overall","clinical3","omi_nomi_auc"]]-a.loc["e25",["overall","clinical3","omi_nomi_auc"]]
    lines.append(f"|{seed}|{v.iloc[0]:+.6f}|{v.iloc[1]:+.6f}|{v.iloc[2]:+.6f}|")
lines+=["","|方案|seed 组|Overall R² mean ± SD|Clinical-3 mean ± SD|条件 AUC mean ± SD|","|---|---|---:|---:|---:|"]
for method in ("e25","shared"):
    for label,mask in (("历史",d.seed<3),("新增",d.seed>=3)):
        a=d[(d.method==method)&mask]
        if len(a):
            vals=[f"{a[k].mean():.6f} ± {a[k].std():.6f}" for k in ("overall","clinical3","omi_nomi_auc")]
            lines.append("|"+"|".join([method,label,*vals])+"|")
lines+=["","原有 OMI vs NOMI 主 AUC 使用 logit 差，单模型时与条件概率排序等价。概率差 AUC 为另一个辅助指标。集成的条件概率由三 seed 平均四类概率重新归一化。Clinical-3 在全部验证 ECG 上计算前三个 one-hot 输出 R² 的均值。仅患者子集的四输出 R² 保留常量 Healthy 列及 sklearn 默认 force_finite。","","新增 seed 保留原模型及优化设置。初始化辅助分支后恢复 CPU RNG，两个头在开始训练时共享相同的 dropout 随机状态和批次顺序。历史 Shared 的新增参数初始化消耗了 CPU RNG，历史 run 不具备这项控制。提取使用 no_grad 与 eval，冻结张量保持普通张量属性，下游可反传至可训练权重。","","## 原三个 seed 的集成配对差","","|指标|Shared 减 E25|95% 实体区间|","|---|---:|---|"]
for k,v in boot["shared/ensemble"].items():
    lines.append(f"|{k}|{v['delta']:+.6f}|[{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]|")
lines+=["","## 误差转移","","|指标|E25|Shared|","|---|---:|---:|"]
for label,truth,pred in [("OMI→NOMI",0,1),("NOMI→OMI",1,0)]:
    lines.append(f"|{label}|{((err.truth==truth)&(err.e25==pred)).sum()}|{((err.truth==truth)&(err.shared==pred)).sum()}|")
lines += [f"|患者→Healthy|{((err.truth!=3)&(err.e25==3)).sum()}|{((err.truth!=3)&(err.shared==3)).sum()}|",
          f"|高置信错误（最大概率≥0.9）|{err.e25_high_conf_wrong.sum()}|{err.shared_high_conf_wrong.sum()}|",
          "", "误差案例使用验证行匿名编号。四类错误转移统计见 error_transition_counts.csv。", "",
          "当前 Validation 参与过历史模型选择。bootstrap 区间只描述固定集成在该划分上的实体重采样不确定性。新增 seed 仍来自同一个预训练 checkpoint。"]
a=d[d.seed<3].groupby("method").mean(numeric_only=True)
diff=a.loc["shared"]-a.loc["e25"]
base=div["e25"]
shared=div["shared"]
lines+=["","## 增量归因","",
    f"历史单模型 Overall 差 {diff.overall:+.6f}，Healthy R² 差的四分之一为 {diff.Healthy/4:+.6f}。"
    f"集成差 {shared['ensemble_r2']-base['ensemble_r2']:+.6f}，其中额外误差抵消贡献 {shared['ensemble_gain']-base['ensemble_gain']:+.6f}。",
    f"E25 的两两残差相关范围为 {min(base['residual_correlations']):.4f}–{max(base['residual_correlations']):.4f}，"
    f"Shared 为 {min(shared['residual_correlations']):.4f}–{max(shared['residual_correlations']):.4f}。"
    f"平均概率的 Brier 误差抵消分别为 {base['brier_cancellation']:.6f} 和 {shared['brier_cancellation']:.6f}。",
    "新增 24737 个参数。是否足以支持采用，取决于独立确认中的患者三类及直接 OMI/NOMI 区分表现。"]
(r/"E29_R_REPORT_CN.md").write_text("\n".join(lines)+"\n")
