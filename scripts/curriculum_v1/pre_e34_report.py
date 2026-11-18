"""Evidence and limitations for the pre-E34 study."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,r2_score,average_precision_score,log_loss

root=Path('/home/ubuntu/HeartLang-main');out=root/'results/pre_e34_evaluation'
data=root/'datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1'
meta=pd.read_csv(data/'train_metadata.csv');y4=np.load(data/'train_labels.npy');y=y4.argmax(1)
folds=pd.read_csv(root/'results/e29_shared_generalization/fold_assignment.csv')
base=np.load(root/'results/e29_shared_generalization/oof_shared.npy').mean(0)
ref=json.loads((root/'results/e29_shared_generalization/cv_summary.json').read_text())['shared']['ensemble']
raw=np.load(out/'04_e33c_raw_information/C1_oof_ensemble.npy')
d2=np.load(out/'04_e33c_raw_information/D2_oof.npy')
b1=np.load(root/'results/e32_omi_nomi_residual/B1_oof.npy')
b2=np.load(root/'results/e32_omi_nomi_residual/B2_oof.npy')
pair=y<2
def metrics(ix,p):
 yy=y4[ix];pp=p[ix];t=y[ix];pos=t<2;q=pp[pos,0]/pp[pos,:2].sum(1)
 return {'auc':float(roc_auc_score(t[pos]==0,q)),'overall':float(r2_score(yy,pp)),
 'clinical3':float(r2_score(yy[:,:3],pp[:,:3])), 'binary_brier':float(np.square(q-(t[pos]==0)).mean())}
allix=np.arange(len(y));pred={'E29':base,'E32_B1':b1,'E32_B2':b2,'Raw_C1':raw,'Raw_D2':d2}
res={k:metrics(allix,v) for k,v in pred.items()}
for key,field in [('overall','overall'),('clinical3','clinical3'),('omi_nomi_auc','auc')]:assert abs(ref[key]-res['E29'][field])<1e-9
res['macro_auc']=float(roc_auc_score(y4,base,average='macro'))
(out/'00_baseline_audit/baseline_metrics.json').write_text(json.dumps(res,indent=2))
foldrows=[]
for f in range(3):
 ix=folds[(folds.fold==f)&(folds.role=='outer')].row.to_numpy();row={'fold':f}
 for name,p in pred.items():row[name]=metrics(ix,p)
 foldrows.append(row)
(out/'04_e33c_raw_information/fold_metrics.json').write_text(json.dumps(foldrows,indent=2))
groups=list(meta.reset_index().groupby('entity_id').indices.values());rng=np.random.default_rng(34201)
contrast={name:{metric:[] for metric in ('auc','overall','clinical3')} for name in ('Raw_C1','Raw_D2')}
for _ in range(2000):
 ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]);bv=metrics(ix,base)
 for name in contrast:
  v=metrics(ix,pred[name])
  for metric in contrast[name]:contrast[name][metric].append(v[metric]-bv[metric])
boots={name:{m:{'delta':res[name][m]-res['E29'][m],'ci95':np.quantile(vals,[.025,.975]).tolist()} for m,vals in d.items()} for name,d in contrast.items()}
(out/'04_e33c_raw_information/paired_entity_bootstrap.json').write_text(json.dumps(boots,indent=2))
geom=pd.read_csv(out/'02_e33a_tokenizer_coverage/geometry_train.csv')
complete={'single_complete':float((geom.r400_single>=.999).mean()),'sequence_complete':float((geom.r400_sequence>=.999).mean()),
 'single_mean':float(geom.r400_single.mean()),'sequence_mean':float(geom.r400_sequence.mean()),
 'n_unresolved':int(geom.r400_single.isna().sum()),'n_exact_replay':int(geom.replay_exact.sum()),
 'by_class':geom.groupby('label')[['r400_single','r400_sequence','valid_fraction']].mean().to_dict()}
(out/'02_e33a_tokenizer_coverage/coverage_summary.json').write_text(json.dumps(complete,indent=2))
assoc=geom.copy();assoc['q']=base[:,0]/base[:,:2].sum(1);assoc=assoc[assoc.label<2].copy()
assoc['loss']=np.where(assoc.label==0,-np.log(assoc.q.clip(1e-8,1-1e-8)),-np.log((1-assoc.q).clip(1e-8,1-1e-8)))
assoc['error']=((assoc.q>=.5)!=(assoc.label==0)).astype(float)
def group(z):
 if not np.isfinite(z):return 'Unknown'
 if z>=.999:return 'Complete'
 if z>=.5:return 'Partial'
 return 'Severe Loss'
assoc['coverage_group']=assoc.r400_single.map(group)
gstats=[]
for name,g in assoc.groupby('coverage_group'):
 stats={'group':name,'n':len(g),'entities':g.entity_id.nunique(),'omi':int((g.label==0).sum()),'nomi':int((g.label==1).sum()),
 'error_rate':float(g.error.mean()),'mean_loss':float(g.loss.mean())}
 if stats['omi']>=10 and stats['nomi']>=10:stats['auc']=float(roc_auc_score(g.label==0,g.q))
 gstats.append(stats)
pd.DataFrame(gstats).to_csv(out/'03_e33b_error_association/coverage_groups.csv',index=False)
# 预设协变量的实体聚类稳健线性回归，仅作观察关联。
reg=assoc.dropna(subset=['r400_single','valid_fraction','rr_ms','n_tokens','loss']).copy()
reg['fold']=reg.row.map(folds[folds.role=='outer'].set_index('row').fold)
X=pd.DataFrame({'intercept':1.,'r400_single':reg.r400_single,'valid_fraction':reg.valid_fraction,
 'rr_ms':reg.rr_ms,'n_tokens':reg.n_tokens,'omi':(reg.label==0).astype(float),
 'fold1':(reg.fold==1).astype(float),'fold2':(reg.fold==2).astype(float)})
for col in ('r400_single','valid_fraction','rr_ms','n_tokens'):
 X[col]=(X[col]-X[col].mean())/X[col].std()
xx=X.to_numpy();target=reg.loss.to_numpy();inv=np.linalg.pinv(xx.T@xx);coef=inv@xx.T@target;err=target-xx@coef
meat=np.zeros((len(coef),len(coef)))
for _,ids in reg.groupby('entity_id').indices.items():
 v=xx[ids].T@err[ids];meat+=np.outer(v,v)
se=np.sqrt(np.diag(inv@meat@inv));adjust={'n':len(reg),'entities':reg.entity_id.nunique(),
 'outcome':'conditional log loss','coef_per_1sd_coverage':float(coef[1]),'cluster_se':float(se[1]),
 'covariates':list(X.columns),'interpretation':'observational geometry proxy, not clinical ST/T boundary or causation'}
(out/'03_e33b_error_association/adjusted_association.json').write_text(json.dumps(adjust,indent=2))
qbase=base[pair,0]/base[pair,:2].sum(1);qraw=raw[pair,0]/raw[pair,:2].sum(1)
compl={'probability_correlation':float(np.corrcoef(qbase,qraw)[0,1]),
 'raw_only_correct':int((((qraw>=.5)==(y[pair]==0))&((qbase>=.5)!=(y[pair]==0))).sum()),
 'e29_only_correct':int((((qbase>=.5)==(y[pair]==0))&((qraw>=.5)!=(y[pair]==0))).sum())}
(out/'04_e33c_raw_information/complementarity.json').write_text(json.dumps(compl,indent=2))
decision={'decision':'CONDITIONAL GO','basis':'E32-B negative; geometric post-R gap largely compensated by adjacent tokens; raw C1 weaker than E29; legal D2 increment must be judged with fold and entity CI; clinical ST/T boundaries and high-rate all-class cohort unavailable',
 'e29_auc':res['E29']['auc'],'raw_c1_auc':res['Raw_C1']['auc'],'raw_d2_auc':res['Raw_D2']['auc'],
 'raw_d2_auc_ci':boots['Raw_D2']['auc']['ci95'],'e34_started':False}
(out/'05_integrated_decision/PRE_E34_DECISION.json').write_text(json.dumps(decision,indent=2))
print(json.dumps({'metrics':res,'folds':foldrows,'bootstrap':boots,'coverage':complete,'association':adjust,'complementarity':compl,'decision':decision},indent=2))
