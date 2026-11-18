import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from run_e32 import OUT,e,SOURCE

def main():
    d=json.loads((OUT/"clinical_subgroup_metrics.json").read_text())
    boot=json.loads((OUT/"paired_entity_bootstrap.json").read_text())
    manifest=json.loads((OUT/"fold_manifest.json").read_text())
    folded=pd.read_csv(SOURCE/"fold_assignment.csv")
    metadata=pd.read_csv(e.DOWNSTREAM/"train_metadata.csv")
    labels=np.load(e.DOWNSTREAM/"train_labels.npy")
    pair=labels[:,:2].sum(1)==1
    base=np.load(OUT/"A0_probability.npy")
    y=labels[:,0].astype(int)
    errors=pd.read_csv(OUT/"error_transition_analysis.csv")
    transition=errors.groupby("method").apply(lambda f:pd.Series({
        "probe_only_correct":int((~f.baseline_correct&f.probe_correct).sum()),
        "baseline_only_correct":int((f.baseline_correct&~f.probe_correct).sum()),
        "both_wrong":int((~f.baseline_correct&~f.probe_correct).sum()),
        "both_correct":int((f.baseline_correct&f.probe_correct).sum())}),include_groups=False)
    transition.to_csv(OUT/"error_transition_counts.csv")
    for item in manifest:
        f=item["fold"]
        for role in ("fit","select","outer"):
            ids=item[role]
            assert len(set(ids))==len(ids)
        for first,second in (("fit","select"),("fit","outer"),("select","outer")):
            assert not set(metadata.iloc[item[first]].entity_id)&set(metadata.iloc[item[second]].entity_id)
        paths=[]
        for seed in range(3):
            paths.extend([SOURCE/f"fold{f}/downstream/shared/seed{seed}/checkpoint-best.pth",
                          SOURCE/f"fold{f}/downstream/learned_mix/seed{seed}/checkpoint-best.pth"])
        assert [e.sha256(p) for p in paths]==item["checkpoints"]
    source=e.DEFAULT_RUN/"mlm/scratch/checkpoint-best.pth"
    old=json.loads((e.ROOT/"results/e25_intermediate_layers/source_audit.json").read_text())["sha256"]
    assert e.sha256(source)==old[str(source)]
    cfg=json.loads((OUT/"model_config.json").read_text())
    cfg["regularization_selection"]="fixed C1, no hyperparameter search"
    cfg["model_parameters"]={"A1_layer9":1537,"A2_multilayer":1537,"A3_shared":97}
    (OUT/"model_config.json").write_text(json.dumps(cfg,indent=2))
    env=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,torch=torch.__version__)
    (OUT/"environment.json").write_text(json.dumps(env,indent=2))
    lines=["# E32-A 冻结表示 OMI/NOMI 线性 Probe","","## 结果与阶段范围","",
      "当前 E29 原条件概率的 OOF AUC 高于三个固定 Logistic Probe。A3 是三个 Probe 中最高。三个 Probe 相对 A0 的实体配对区间均在零以下。错误互补作为探索性记录，尚未通过 cross-fitted 融合验证。本轮未进入 E32-B，不能据此断言线性残差不存在增量。",
      "", "|模型|OOF AUC|PR-AUC|二分类 Brier|Log loss|内层选定阈值的 OMI sensitivity|实际 NOMI specificity|AUC 差95% CI|",
      "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name,a in d.items():
        lines.append("|"+name+"|"+"|".join(f"{a[k]:.6f}" for k in ("auc","pr_auc","brier","log_loss","inner_threshold_omi_sensitivity","inner_threshold_nomi_specificity"))+f"|{boot[name]['ci95']}|")
    lines+=["","## 实际实现","",
      "- 复用 E29-S 三个 outer folds 和每折 fit/select/outer 分配。Probe 只在该折 fit 中真实 OMI/NOMI 样本拟合，StandardScaler 同样限于 fit。固定 C=1 的 L2 Logistic Regression。select 用于确定满足 90% specificity 的工作点。outer 样本仅用于预测和评价。",
      "- A1 使用第9层有效 token Mean+Max，1536维。A2 使用该折 E25 三个 checkpoint 的 softmax 融合权重均值，1536维。A3 拼接该折三个 Shared checkpoint 各自的32维 attention 聚合结果，96维。A2 权重与 A3 投影均来自对应外层训练侧，未使用全量Train监督头。",
      "- A0 工作点来自该折 Shared 的内层 select 原始预测。0.5阈值的二分类指标与内层设定工作点分别保存。工作点在外层可能偏离90% specificity，这是实际折外结果。",
      "- L2强度为预先固定的 C1。这一结果只覆盖当前线性配置。A1/A2 的概率校准较差，不据此认定冻结表示缺乏所有可用判别信息。没有据外层结果继续修改正则化。",
      "- 首次尝试的 A2 曾读取 Shared 的融合权重，已经更正为 E25 对应折 checkpoint。正式结果重跑后覆盖根目录产物，首次尝试保存在 attempt1。首次尝试的公共loader读取了历史Validation缓存，未用于拟合或选参。正式重跑仅加载Train。两次都未读取Test。",
      "", "## 互补分析", "", "\n".join(["|method|"+"|".join(transition.columns)+"|","|---|"+"|".join(["---"]*len(transition.columns))+"|"]+["|"+str(i)+"|"+"|".join(map(str,row))+"|" for i,row in zip(transition.index,transition.values)]), "",
      "纠正和回退使用条件概率0.5阈值。互补计数没有证明泛化增量。与基线的相关性及完整ROC在 clinical_subgroup_metrics.json，病例仅输出匿名行号。"]
    lines+=["","## 每折基线与 Probe AUC","","|外层折|A0|A1|A2|A3|","|---|---:|---:|---:|---:|"]
    for f in range(3):
        idx=folded[(folded.fold==f)&(folded.role=="outer")].row.to_numpy()
        idx=idx[pair[idx]]
        values=[roc_auc_score(y[idx],np.load(OUT/f"{name}_probability.npy")[idx]) for name in d]
        lines.append("|"+str(f)+"|"+"|".join(f"{v:.6f}" for v in values)+"|")
    lines+=["","## 验收与结论","",
      "- E12-A SHA 与历史审计值一致，六个每折 E25/Shared checkpoint SHA 均与提取前一致。旧 E29 OOF 指标重算通过，fit/select/outer实体无交叉，全部概率有限。",
      "- Probe 是二分类分数，完整四分类 R²不直接定义。将其注入四类概率需要指定并验证 E32-B 的概率重分配规则，本轮未将 Probe 分数当作四分类输出。",
      "- B未执行，lambda=0、Control/Healthy保持及校准对照不属于已验证结果。残差有效性、是否影响Control/Healthy、是否应替换E29均未获得新证据。保留E29 Shared。",
      "- Bootstrap 2000次，按患者/健康实体成簇，四种方法使用同一索引，只评估真实OMI/NOMI子集。区间不覆盖模型重训、outer划分和预训练随机性。原预训练包含这些ECG的无标签输入。"]
    (OUT/"E32_A_REPORT_CN.md").write_text("\n".join(lines)+"\n")
    pd.DataFrame([dict(status="not_run",reason="no standalone probe surpassed A0; complementarity remains exploratory; no cross-fitted fusion evidence")]).to_csv(OUT/"residual_model_results.csv",index=False)
    (OUT/"E32_B_REPORT_CN.md").write_text("# E32-B 阶段状态\n\n未执行。三个 Probe 的折外 AUC 均低于基线，互补计数仍是探索性线索。当前没有通过内层 cross-fitting 验证的融合增量。不能声称线性残差无效，也不能把同批 OOF 拟合后的分数作为融合证据。\n")
    (OUT/"verification.json").write_text(json.dumps(dict(E12_hash_unchanged=True,fold_checkpoint_hashes_unchanged=True,entity_isolation=True,baseline_reproduced=True,test_access_in_executed_code=False,B_executed=False),indent=2))

if __name__=="__main__":
    main()
