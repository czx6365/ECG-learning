"""Summarize the frozen E32-B outer-fold predictions."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, r2_score
import run_e29_s as s

e=s.e
root=e.ROOT/'results/e32_omi_nomi_residual'
src=s.OUT
meta=pd.read_csv(e.DOWNSTREAM/'train_metadata.csv')
y=np.load(e.DOWNSTREAM/'train_labels.npy'); target=y.argmax(1)
pred={n:np.load(root/f'{n}_oof.npy') for n in ('B0','B1','B2')}
folds=pd.read_csv(src/'fold_assignment.csv')
def dump(n,v): (root/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=float))
def score(rows,q):
    yy=y[rows];pp=q[rows];t=yy.argmax(1);mask=t<2
    cond=pp[mask,0]/pp[mask,:2].sum(1)
    return {'auc':float(roc_auc_score(t[mask]==0,cond)),
        'overall':float(r2_score(yy,pp)),
        'clinical3':float(r2_score(yy[:,:3],pp[:,:3])),
        'omi':float(r2_score(yy[:,0],pp[:,0])),
        'nomi':float(r2_score(yy[:,1],pp[:,1])),
        'control':float(r2_score(yy[:,2],pp[:,2])),
        'healthy':float(r2_score(yy[:,3],pp[:,3])),
        'binary_brier':float(np.square(cond-(t[mask]==0)).mean())}
allrows=np.arange(len(y));scores={n:score(allrows,q) for n,q in pred.items()}
foldrows=[]
for f in range(3):
    ix=folds[(folds.fold==f)&(folds.role=='outer')].row.to_numpy()
    row={'fold':f}
    for n,q in pred.items():
        for k,v in score(ix,q).items():row[f'{n}_{k}']=v
    foldrows.append(row)
pd.DataFrame(foldrows).to_csv(root/'per_fold_metrics.csv',index=False)
groups=list(meta.reset_index().groupby('entity_id').indices.values())
rng=np.random.default_rng(32072)
keys=('auc','overall','clinical3','omi','nomi','control','healthy','binary_brier')
pairs=(('B1','B0'),('B2','B0'),('B2','B1'))
draws={f'{a}-{b}':{k:[] for k in keys} for a,b in pairs}
for _ in range(2000):
    ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
    v={n:score(ix,q) for n,q in pred.items()}
    for a,b in pairs:
        for k in keys:draws[f'{a}-{b}'][k].append(v[a][k]-v[b][k])
boot={name:{k:{'delta':scores[name.split('-')[0]][k]-scores[name.split('-')[1]][k],
    'ci95':np.quantile(vals,[.025,.975]).tolist()} for k,vals in metrics.items()} for name,metrics in draws.items()}
dump('paired_entity_bootstrap.json',boot)
pair=target<2
q={n:p[:,0]/p[:,:2].sum(1) for n,p in pred.items()}
err=[]
for i in np.flatnonzero(pair):
    err.append({'row':int(i),'entity_id':str(meta.iloc[i].entity_id),'fold':int(folds[(folds.role=='outer')&(folds.row==i)].fold.iloc[0]),
      'truth':int(target[i]),'B0_correct':bool((q['B0'][i]>=.5)==(target[i]==0)),
      'B2_correct':bool((q['B2'][i]>=.5)==(target[i]==0))})
pd.DataFrame(err).to_csv(root/'error_transition_analysis.csv',index=False)
diagnostics={}
for f in range(3):
    z=np.load(root/f'fold{f}_held.npz');d=z['delta'];r=np.log(z['prob'][:,0]/z['prob'][:,1])
    diagnostics[str(f)]={'mean':float(d.mean()),'sd':float(d.std()),'quantiles':np.quantile(d,[.01,.1,.5,.9,.99]).tolist(),
      'abs_quantiles':np.quantile(abs(d),[.5,.9,.99]).tolist(),'r_delta_corr':float(np.corrcoef(r,d)[0,1])}
dump('residual_diagnostics.json',diagnostics)
invariance={}
for n,p in pred.items():
    invariance[n]={'max_mass_diff':float(np.abs(p[:,:2].sum(1)-pred['B0'][:,:2].sum(1)).max()),
      'max_control_diff':float(np.abs(p[:,2]-pred['B0'][:,2]).max()),'max_healthy_diff':float(np.abs(p[:,3]-pred['B0'][:,3]).max()),
      'max_sum_error':float(np.abs(p.sum(1)-1).max()),'finite':bool(np.isfinite(p).all())}
dump('probability_invariance_tests.json',invariance)
dump('calibration_metrics.json',{n:{'binary_brier':v['binary_brier']} for n,v in scores.items()})
dump('oof_metrics.json',scores)
lines=['# E32-B 受约束线性残差融合报告','',
 '基于原 E29-S 三折实体分组；每个外层开发集内两折交叉拟合三 seed Shared 模型，内层模型另划分选轮集。只在交叉拟合预测上训练 B1/B2。E12-A 预训练保持固定，未读取 Test。','',
 '|模型|OMI/NOMI AUC|Overall R²|Clinical-3 R²|二分类 Brier|','|---|---:|---:|---:|---:|']
for n,v in scores.items():lines.append(f"|{n}|{v['auc']:.6f}|{v['overall']:.6f}|{v['clinical3']:.6f}|{v['binary_brier']:.6f}|")
lines+=['','## 配对实体 bootstrap（2000次）','', '|对照|AUC差|95%区间|Overall R²差|Clinical-3 R²差|','|---|---:|---|---:|---:|']
for n,v in boot.items():lines.append(f"|{n}|{v['auc']['delta']:+.6f}|{v['auc']['ci95']}|{v['overall']['delta']:+.6f}|{v['clinical3']['delta']:+.6f}|")
lines+=['','## 各外层折 AUC','', '|折|B0|B1|B2|B2-B0|','|---|---:|---:|---:|---:|']
for v in foldrows:lines.append(f"|{v['fold']}|{v['B0_auc']:.6f}|{v['B1_auc']:.6f}|{v['B2_auc']:.6f}|{v['B2_auc']-v['B0_auc']:+.6f}|")
lines+=['','B1 在每一折内部为严格单调映射；不同折分别校准后，拼接 OOF 的全局排序可能变化。Bootstrap 仅覆盖固定预测的实体重采样。']
(root/'E32_B_REPORT_CN.md').write_text('\n'.join(lines))
print('\n'.join(lines))
