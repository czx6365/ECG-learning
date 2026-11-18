"""Compare completed validation experiments without opening Test artifacts."""
import json
from pathlib import Path

ROOT = Path("/home/ubuntu/HeartLang-main/results")
groups = {"E29": "e29_class_conditioned_pooling", "E30": "e30_physiology_increment"}
out = ROOT / "e29_e30_comparison"
lines = [
    "# E29 与 E30 的验证集比较", "",
    "E29 检查冻结第 9 层的类别条件池化。E30 检查连续 100 Hz 波形的固定窗口代理和 RR 特征。E30 的幅值处于现有标准化波形尺度，窗口特征不等同于临床 ST/T 测量。",
    "", "保持 E12-A 冻结，沿用 E25 的 E11 头、30 轮训练、三个下游 seed 和 Validation 选模。",
    "Overall R² 为各类别 one-hot 输出与 softmax 概率的均匀平均。临床三类均值为前三个类别各自 R² 的均值。表中为单 seed 指标的均值及样本标准差，区间来自固定三 seed 平均概率的配对实体 bootstrap。",
    "当前 Validation 参与过历史开发和选模，独立确认仍待完成。", "",
    "|实验|方法|Overall R²|Clinical-3 R²|OMI|NOMI|Control|Healthy|OMI vs NOMI AUC|Macro AUC|Macro F1|配对 R² 差|集成差 95% CI|",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
]
stable = {}
for exp,name in groups.items():
    folder = ROOT/name
    assert (folder/"status.txt").read_text().strip()=="complete"
    results=json.loads((folder/"validation_summary.json").read_text())
    boots=json.loads((folder/"entity_bootstrap.json").read_text())
    stable[exp]=False
    for method,item in results.items():
        a=item["aggregate"]
        b=boots[method]
        keys=["r2","patient_mean_r2","OMI","NOMI","Clinical Control","Healthy","omi_vs_nomi_auc","macro_roc_auc","macro_f1"]
        vals=[f"{a[k]['mean']:.6f} ± {a[k]['sd']:.6f}" for k in keys]
        lines.append("|"+"|".join([exp,method,*vals,str([round(v,6) for v in b["paired_seed_delta"]]),str([round(v,6) for v in b["ci95"]])])+"|")
        if method==("class" if exp=="E29" else "increment"):
            stable[exp]=min(b["paired_seed_delta"])>0 and b["ci95"][0]>0 and a["patient_mean_r2"]["mean"]>results["learned_mix"]["aggregate"]["patient_mean_r2"]["mean"]
lines.extend(["","预先使用三个配对 seed 均提高、集成差区间下界大于零且临床三类均值提高作为稳定增益的筛选条件。",""])
if all(stable.values()):
    lines.append("两方向均通过筛选，先独立确认后再研究组合。")
elif stable["E29"]:
    lines.append("E29 通过筛选，下一步优先研究冻结表示的 token 聚合。")
elif stable["E30"]:
    lines.append("E30 通过筛选，下一步可独立确认波形增量并据此规划双分辨率分支。")
else:
    lines.append("两个预先指定主方案均未通过稳定增益筛选。E29 共享 query 的三个 seed 整体 R² 均提升，可作为优先锁定复核的简单读取方案。类别 query 是否优于共享 query，需要结合二者直接对照区间判断。")
lines += ["","E29 类别 query 与容量对照均为 280845 参数，共享 query 为 280749 参数，E25 为 256012 参数。E29 的首次提取使用 inference_mode，梯度检查失败后改为 no_grad，并从已完成基线继续。新增分支初始化会改变 dropout 的随机数流，各组仍使用同一基础头初始化和批次顺序。"]
for experiment, filename in groups.items():
    path=ROOT/filename/"clinical_entity_contrasts.json"
    if path.exists():
        lines += ["", f"### {experiment} 临床分项的实体配对区间", "", "|对照|指标|集成差值|95% CI|","|---|---|---:|---|"]
        for name, metrics in json.loads(path.read_text()).items():
            for metric,v in metrics.items():
                lines.append(f"|{name}|{metric}|{v['delta']:.6f}|{v['ci95']}|")
lines += ["","原始 ZIP 路径可用性审计显示，患者 Train 3320/3320、Validation 672/672，健康 Train 0/3320、Validation 0/672。本轮使用统一 100 Hz 连续波形，以避免采样率与来源可用性和患者身份重合。J 点、QT/QTc、T 末端及物理单位的验证仍未完成。"]
lines += ["","E30 固定模型敏感性与覆盖分组见 increment_diagnostics.json。固定模型替换特征组评估的是依赖性，单独特征组的训练增量仍待实验。"]
(out/"README_CN.md").write_text("\n".join(lines))
