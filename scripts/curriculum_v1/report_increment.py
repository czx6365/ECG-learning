import csv
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import run_e25 as base

def report(run, rows, data):
    y = data["val"]["y"].cpu().numpy()
    meta = pd.read_csv(base.e.DOWNSTREAM/"val_metadata.csv")
    groups = list(meta.groupby("entity_id").indices.values())
    rng = np.random.default_rng(25006)
    samples = [np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]) for _ in range(2000)]
    keys = ("r2","patient_mean_r2","omi_vs_nomi_auc","macro_roc_auc","macro_f1")
    baseline = [r for r in rows if r["method"]=="learned_mix"]
    def predict(method):
        return np.mean([np.load(run/"downstream"/method/f"seed{s}"/"val_probabilities.npy") for s in range(3)],0)
    reference = predict("learned_mix")
    result, boots = {}, {}
    for method in dict.fromkeys(r["method"] for r in rows):
        rr = [r for r in rows if r["method"]==method]
        metrics = [{**r["val"],**r["val"]["per_output_r2"]} for r in rr]
        agg = {k:{"mean":float(np.mean([m[k] for m in metrics])),"sd":float(np.std([m[k] for m in metrics],ddof=1))} for k in (*keys,*rr[0]["val"]["per_output_r2"])}
        result[method] = dict(aggregate=agg,per_seed=rr)
        pred = predict(method)
        for r in rr:
            p = np.load(run/"downstream"/method/f"seed{r['seed']}"/"val_probabilities.npy")
            assert abs(r2_score(y,p)-r["val"]["r2"])<1e-7
        boots[method] = dict(paired_seed_delta=[r["val"]["r2"]-b["val"]["r2"] for r,b in zip(rr,baseline)],
                             ensemble_delta=float(r2_score(y,pred)-r2_score(y,reference)),
                             ci95=np.quantile([r2_score(y[i],pred[i])-r2_score(y[i],reference[i]) for i in samples],[.025,.975]).tolist())
    base.dump(run/"validation_summary.json",result)
    base.dump(run/"entity_bootstrap.json",boots)
    columns=list(next(iter(result.values()))["aggregate"])
    with (run/"ablation_results.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["method",*columns])
        for m,v in result.items():w.writerow([m,*[v["aggregate"][k]["mean"] for k in columns]])
    lines=["# Validation 增量实验", "", "三下游 seed 配对。实体 bootstrap 使用三 seed 平均概率。", "", "|方法|Overall R²|患者三类 R²|Macro AUC|差值 95% CI|","|---|---:|---:|---:|---|"]
    for m,v in result.items():
        a=v["aggregate"]
        lines.append(f"|{m}|{a['r2']['mean']:.6f}|{a['patient_mean_r2']['mean']:.6f}|{a['macro_roc_auc']['mean']:.6f}|{boots[m]['ci95']}|")
    (run/"README_CN.md").write_text("\n".join(lines))
