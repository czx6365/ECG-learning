"""Run after locked Test evaluation. Emit paper tables and grouped bootstrap."""
import json
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_e30_sid as run

out, e = run.OUT, run.e
assert (out/'DOWNSTREAM_COMPLETE_preseed0').exists()
labels = np.load(e.DOWNSTREAM/'test_labels.npy').astype(float)
meta = pd.read_csv(e.DOWNSTREAM/'test_metadata.csv')
group_id = meta.entity_id.astype('string')
assert not group_id.isna().any() and len(meta)==len(labels)
groups = [np.asarray(g.index) for _,g in meta.groupby(group_id,sort=False)]
rows, probs = [], {}
classes = ['OMI','NOMI','Clinical Control','Healthy']
for obj in 'ABC':
    for probe,method in [('Full','e12a_region'),('Transformer-only','e12a_mlm_only')]:
        for seed in run.SEEDS:
            d = out/f'downstream/{obj}_preseed0/downstream/{method}/seed{seed}'
            for split in ('val','test'):
                m = json.loads((d/('val_summary.json' if split=='val' else 'test_metrics.json')).read_text())
                if split=='val':
                    m=m['val']
                rows.append(dict(method=obj,probe=probe,seed=seed,split=split,
                    **{k:m[k] for k in ('r2','rmse','mae','macro_roc_auc','macro_f1','omi_vs_nomi_auc')},
                    clinical3=m['patient_three_output_r2_on_all_ecg'],**m['per_output_r2']))
            q=np.load(d/'test_probabilities.npy').astype(float)
            assert q.shape==labels.shape and np.isfinite(q).all()
            probs.setdefault((probe,obj),[]).append(q)
df=pd.DataFrame(rows)
df.to_csv(out/'downstream_metrics.csv',index=False)
rng=np.random.default_rng(30002000)
draws=[np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]) for _ in range(2000)]
comparisons=[]
def r2_values(y,q):
    sst=((y-y.mean(axis=0))**2).sum(axis=0)
    return 1-((y-q)**2).sum(axis=-2)/sst
for probe in df.probe.unique():
    for lhs,rhs in [('B','A'),('C','A'),('C','B')]:
        q,p=[np.stack(probs[probe,k]) for k in (lhs,rhs)]
        for scope in ['seed_mean','ensemble']:
            l,r=(q,p) if scope=='seed_mean' else (q.mean(0)[None],p.mean(0)[None])
            delta=lambda i:(r2_values(labels[i],l[:,i])-r2_values(labels[i],r[:,i])).mean(0)
            d=delta(np.arange(len(labels)))
            boot=np.asarray([delta(i) for i in draws])
            for name,estimate,samples in [('Overall',d.mean(),boot.mean(1)),('Clinical-3',d[:3].mean(),boot[:,:3].mean(1)),
                    *((k,d[j],boot[:,j]) for j,k in enumerate(classes))]:
                lo,hi=np.nanquantile(samples,[.025,.975])
                comparisons.append(dict(probe=probe,scope=scope,comparison=f'{lhs}-{rhs}',metric=name,delta=estimate,lower=lo,upper=hi))
            if probe=='Full' and scope=='seed_mean':
                plt.hist(boot.mean(1),bins=35,alpha=.45,label=f'{lhs}-{rhs}')
fig=out/'figures'
fig.mkdir(exist_ok=True)
plt.axvline(0,color='black',lw=.8)
plt.xlabel('Paired Overall R² difference')
plt.legend()
plt.savefig(fig/'05_paired_delta.png',dpi=220,bbox_inches='tight')
plt.close()
pd.DataFrame(comparisons).to_csv(out/'bootstrap_statistics.csv',index=False)
for keys,field,title in [
    (['split'],'r2','01_val_test'), (['class'],'r2','02_per_output'), (['probe'],'r2','03_probe')
]:
    if keys==['class']:
        sub=df.query("probe=='Full' and split=='test'").melt(id_vars=['method'],value_vars=classes,var_name='class',value_name='r2')
    else:
        sub=df.query("probe=='Full'") if keys==['split'] else df.query("split=='test'")
    pivot=sub.groupby([*keys,'method'])[field].mean().unstack()
    pivot.plot.bar(rot=0,yerr=sub.groupby([*keys,'method'])[field].std().unstack(),capsize=3)
    plt.axhline(0,color='black',lw=.8)
    plt.ylabel('R² (mean ± downstream seed SD)')
    plt.savefig(fig/f'{title}.png',dpi=220,bbox_inches='tight')
    plt.close()

audit=json.loads((out/'audit.json').read_text())
source=e.DEFAULT_RUN
a_hist=pd.read_csv(source/'mlm/scratch/training_history.csv').dropna(subset=['val_loss'])
plt.plot(a_hist.epoch,a_hist.val_loss,label='A historical CE (random masks)')
cost=[dict(method='A',steps=340*100,seconds=audit['summary']['elapsed_seconds'],
    parameters=audit['trainable_pretrain_parameters'],val_loss=audit['summary']['best_val_loss'])]
for obj in 'BC':
    d=out/f'checkpoints/{obj}_seed0'
    history=json.loads((d/'history.json').read_text())
    val=[r for r in history if 'val' in r]
    plt.plot([r['epoch'] for r in val],[r['val']['loss'] for r in val],label=f'{obj} fixed-mask objective')
    config=json.loads((d/'config.json').read_text())
    summary=json.loads((d/'summary.json').read_text())
    cost.append(dict(method=obj,steps=summary['scheduled_steps'],seconds=summary['seconds'],
        parameters=config['parameters'],val_loss=min(r['val']['loss'] for r in val)))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Objective-specific loss')
plt.savefig(fig/'04_pretraining.png',dpi=220,bbox_inches='tight')
plt.close()
pd.DataFrame(cost).to_csv(out/'training_cost.csv',index=False)
def ms(series):
    return f'{series.mean():.6f} ± {series.std():.6f}'
primary=df.query("probe=='Full'")
test=primary.query("split=='test'")
lines=['# E30 SID 监督价值验证','','使用原固定 Validation/Test 和冻结 Transformer。表中均值与标准差基于三个下游 seed，预训练仅一个 seed。','',
    '|方法|输入|目标|Val Overall R²|Test Overall R²|ΔTest R²|Test Clinical-3 R²|','|---|---|---|---:|---:|---:|---:|']
for obj,objective in zip('ABC',['SID CE','标准化 z_e MSE','SID CE + λ MSE']):
    t=test[test.method==obj]
    v=primary.query("split=='val' and method==@obj")
    delta=t.r2.mean()-test[test.method=='A'].r2.mean()
    lines.append(f'|{obj}|z_e|{objective}|{ms(v.r2)}|{ms(t.r2)}|{delta:+.6f}|{ms(t.clinical3)}|')
lines+=['','|方法|OMI R²|NOMI R²|Control R²|Healthy R²|Overall R²|','|---|---:|---:|---:|---:|---:|']
for obj,g in test.groupby('method'):
    lines.append('|'+obj+'|'+'|'.join(ms(g[k]) for k in classes+['r2'])+'|')
lines+=['','|方法|Full Test R²|Transformer-only Test R²|','|---|---:|---:|']
for obj in 'ABC':
    d=df.query("split=='test' and method==@obj")
    lines.append(f"|{obj}|{ms(d[d.probe=='Full'].r2)}|{ms(d[d.probe=='Transformer-only'].r2)}|")
lines+=['','|比较|配对 ΔTest R²|95% CI|','|---|---:|---:|']
primary_ci=[r for r in comparisons if r['probe']=='Full' and r['scope']=='seed_mean' and r['metric']=='Overall']
for r in primary_ci:
    lines.append(f"|{r['comparison']}|{r['delta']:+.6f}|[{r['lower']:+.6f}, {r['upper']:+.6f}]|")
lines+=['','|方法|计划更新步数|GPU 作业小时|参数量|最优 Val objective|','|---|---:|---:|---:|---:|']
for r in cost:
    lines.append(f"|{r['method']}|{r['steps']}|{r['seconds']/3600:.2f}|{r['parameters']}|{r['val_loss']:.6f}|")
winner=test.groupby('method').r2.mean().idxmax()
robust=[r for r in primary_ci if r['comparison']==f'{winner}-A' and r['lower']>0]
decision={'B':'优先复核连续重建','C':'优先复核混合目标'}.get(winner,'暂保留 SID CE') if robust or winner=='A' else '证据尚不充分，暂保留 SID CE'
lines+=['',f'固定 Test 的点估计最高为 {winner}。研究决策为 **{decision}**。','',
    'Clinical-3 是全体 Test ECG 上的三列平均 R²。实体 bootstrap 使用同一组患者或 Healthy 实体配对重采样，2000 次。区间固定已训练模型，无法覆盖独立预训练的方差。历史 A 的训练 mask RNG 和选模验证 mask 与本次 B/C 存在差异。训练损失属于各自目标。','',
    'Full 与 Transformer-only 的差异可以检验共享连续特征是否缩小表示差距。类别差异和条件 OMI/NOMI AUC 见逐 seed 表。观察到的目标差异不能识别丢失的具体形态信息，也无法鉴别 Hybrid 的表征改进与正则化机制。三组共享 VQ 训练过的 E3。']
(out/'E30_FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
