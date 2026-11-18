"""Report fixed all-layer ablations from saved OOF probabilities."""
import json
import numpy as np
import pandas as pd
from run_e29_l import OUT,e,s

def main():
    assert (OUT/"status.txt").read_text().strip()=="complete"
    d=json.loads((OUT/"summary.json").read_text())
    frame=pd.read_csv(OUT/"fold_seed_metrics.csv")
    weights=pd.read_csv(OUT/"layer_weights.csv")
    cols=[f"layer{i}" for i in range(1,13)]
    c=weights[weights.method=="C"]
    lines_seed=["# Fold与seed汇总","","|折|seed|方法|Overall R²|Clinical-3|条件AUC|","|---|---|---|---:|---:|---:|"]
    for row in frame.itertuples():
        lines_seed.append(f"|{row.fold}|{row.seed}|{row.method}|{row.overall:.6f}|{row.clinical3:.6f}|{row.omi_nomi_auc:.6f}|")
    (OUT/"per_seed_metrics_CN.md").write_text("\n".join(lines_seed))
    b=s.b
    b.dump(OUT/"weight_statistics.json",dict(mean=c[cols].mean().tolist(),sd=c[cols].std().tolist(),
        top_layer_counts=c[cols].idxmax(1).value_counts().to_dict(),entropy=(-c[cols]*np.log(c[cols].clip(1e-12))).sum(1).tolist()))
    labels=np.load(e.DOWNSTREAM/"train_labels.npy")
    probs={m:np.load(OUT/f"oof_{m}.npy") for m in ("A","B","C")}
    held=pd.read_csv(s.OUT/"fold_assignment.csv")
    rows=[]
    fold_intervals={}
    for f in range(3):
        idx=held[(held.fold==f)&(held.role=="outer")].row.to_numpy()
        for m in probs:
            for seed in range(3):
                np.testing.assert_allclose(probs[m][seed,idx].sum(1),1,atol=1e-6)
                metric=s.metrics(labels[idx],probs[m][seed,idx])
                stored=frame[(frame.fold==f)&(frame.method==m)&(frame.seed==seed)].iloc[0]
                assert abs(metric["overall"]-stored.overall)<1e-10
            rows.append(dict(fold=f,method=m,**s.metrics(labels[idx],probs[m][:,idx].mean(0))))
        fold_intervals[str(f)]={f"C_minus_{m}":s.contrast(labels[idx],probs["C"][:,idx].mean(0),probs[m][:,idx].mean(0),pd.read_csv(e.DOWNSTREAM/"train_metadata.csv").iloc[idx].entity_id.to_numpy()) for m in ("A","B")}
    b.dump(OUT/"fold_paired_bootstrap.json",fold_intervals)
    pd.DataFrame(rows).to_csv(OUT/"fold_ensemble_metrics.csv",index=False)
    old=np.load(s.OUT/"oof_shared.npy")
    np.testing.assert_allclose(probs["A"],old,atol=3e-6,rtol=1e-5)
    error=[]
    for m in ("B","C"):
        a=probs["A"].mean(0).argmax(1)
        z=probs[m].mean(0).argmax(1)
        y=labels.argmax(1)
        for i in np.flatnonzero(a!=z):
            error.append(dict(ecg_id=f"T{i:05d}",truth=int(y[i]),method=m,A=int(a[i]),new=int(z[i]),
                corrected=bool(a[i]!=y[i] and z[i]==y[i]),regressed=bool(a[i]==y[i] and z[i]!=y[i])))
    pd.DataFrame(error).to_csv(OUT/"prediction_change_cases.csv",index=False)
    lines=["# E29-L 全层融合实验","","保持 E12-A、tokenizer、Shared第9层query和E11路径固定。A为6/9/12层可学习凸组合，B为12层均匀，C为12层softmax标量组合。所有层在完整block后复用原冻结最终LayerNorm，排除CLS并采用相同有效token mask。保持原尺度，无额外拟合标准化。","",
        "E29-S的三折fit/select/outer不变，每折三seed，每次30轮。仅inner select按原Overall R²选轮。原Train内CV的外层样本已经作为无标签ECG参与E12-A预训练。","","|方案|参数量|单seed OOF平均 R²|单seed Clinical-3均值|三seed OOF集成 R²|集成Clinical-3|集成条件OMI/NOMI AUC|集成Macro AUC|集成Macro F1|","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    counts=json.loads((OUT/"parameter_count.json").read_text())
    for m in ("A","B","C"):
        v=d[m]
        lines.append(f"|{m}|{counts[m]}|{v['mean']['overall']:.6f}|{v['mean']['clinical3']:.6f}|"+"|".join(f"{v['ensemble'][k]:.6f}" for k in ("overall","clinical3","omi_nomi_auc","macro_auc","macro_f1"))+"|")
    lines+=["","## 配对实体区间","","固定三seed平均概率，2000次实体成簇配对bootstrap。点估计按ECG记录计算，患者/健康实体作为bootstrap重采样簇；未把重复ECG视作独立患者。逐折区间另见fold_paired_bootstrap.json。","",
        "|比较|指标|差值|95% CI|","|---|---|---:|---|"]
    for contrast,v in d["bootstrap"].items():
        for metric,z in v.items():
            lines.append(f"|{contrast}|{metric}|{z['delta']:+.6f}|{z['ci95']}|")
    lines+=["","## 各折固定集成","","|折|比较|Overall差|Clinical-3差|条件AUC差|","|---|---|---:|---:|---:|"]
    for f in range(3):
        a={m:next(x for x in rows if x["fold"]==f and x["method"]==m) for m in ("A","B","C")}
        for other in ("A","B"):
            lines.append(f"|{f}|C−{other}|"+"|".join(f"{a['C'][k]-a[other][k]:+.6f}" for k in ("overall","clinical3","omi_nomi_auc"))+"|")
    lines+=["","## 十二层权重","","|层|均值|SD|","|---|---:|---:|"]
    for col in cols:lines.append(f"|{col}|{c[col].mean():.6f}|{c[col].std():.6f}|")
    lines += ["","逐折逐seed权重及top层计数见layer_weights.csv、weight_statistics.json。权重是当前结构和输入尺度下的优化参数，不能解释为医学特征重要性。缓存保存十二层Mean+Max充分统计量，以及Shared所需第9层token和mask；未保存其余十一层完整token，以节约存储。","",
        "缓存尺度见cache_audit.json；A检查及OOF重算见baseline_parity.json。错误与纠正案例见prediction_change_cases.csv，均为匿名Train行号。",
        "",f"总运行秒数 {json.loads((OUT/'execution_audit.json').read_text())['seconds']:.1f}。未访问Test。三个下游seed共享一个预训练checkpoint，bootstrap固定预测，不覆盖独立预训练、重新训练或重新划分的方差。"]
    (OUT/"E29_L_REPORT_CN.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
