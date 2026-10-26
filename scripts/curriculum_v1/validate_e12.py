import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_e12_control_diagnostic as c
import run_e4_aggregation as e4
p=c.RUN
r=json.loads((p/'pairwise_probes.json').read_text()); supplemental=json.loads((p/'heartbeat_summary_pairwise_probes.json').read_text())
labels=np.load(e4.FEATURES/'test_labels.npy').argmax(1)
assert len(r)==24 and len(supplemental)==9
for row in r:
 name,pos,neg=next(t for t in c.PAIRS if t[0]==row['pair']);mask=(labels==pos)|(labels==neg);y=(labels[mask]==pos).astype(int)
 score=np.load(p/'probes'/row['representation']/name/'test_scores.npy')
 for k,v in c.binary_metrics(y,score).items():assert np.isclose(v,row['test'][k],atol=1e-12),k
 chosen=max(row['candidates'],key=lambda a:a['val']['roc_auc']);assert row['selected_C']==chosen['C'] and row['val']==chosen['val']
for row in supplemental:
 chosen=max(row['candidates'],key=lambda a:a['val']['roc_auc']);assert row['selected_C']==chosen['C'] and row['val']==chosen['val']
 name,pos,neg=next(t for t in c.PAIRS if t[0]==row['pair']);assert row['test']['n_records']==int(((labels==pos)|(labels==neg)).sum())
groups=json.loads((p/'Control_error_group_diagnostics.json').read_text());cases=pd.read_csv(p/'Control_error_cases.csv')
for seed in (0,1,2):
 pred=np.load(c.E7/f'CE_seed{seed}/test_raw_outputs.npy').argmax(1)
 for k in range(4):assert groups[str(seed)][e4.LABELS[k]]['n_records']==int(((labels==2)&(pred==k)).sum())
 assert len(cases[cases.seed==seed])==101
for f in ('core_validation.json','heartbeat_validation.json'):
 data=json.loads((p/f).read_text());assert all(v is True for v in data.values() if isinstance(v,bool))
assert (p/'COMPLETE').exists() and (p/'exit_code.txt').read_text().strip()=='0' and (p/'status.txt').read_text().strip()=='complete'
# Descriptive plots, training-fit PCA coordinates only, no additional fitting/selection.
points=pd.read_csv(p/'PCA_visualization_points.csv');fig,ax=plt.subplots(figsize=(7,5))
for label,color in zip(e4.LABELS[:3],['#e76f51','#457b9d','#2a9d8f']):
 d=points[points.label==label];ax.scatter(d.pc1,d.pc2,s=18,alpha=.6,label=f'{label} (n={len(d)})',color=color)
ax.set(xlabel='PC1 (train-fit)',ylabel='PC2 (train-fit)',title='E12 test patient representations: HeartLang Mean+Max')
ax.legend();fig.tight_layout();fig.savefig(p/'PCA_patient_distribution.png',dpi=180);plt.close(fig)
reps=c.REPS;mat=np.array([[next(t['test']['roc_auc'] for t in r if t['representation']==rep and t['pair']==pair[0]) for pair in c.PAIRS] for rep in reps])
fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(mat,vmin=.5,vmax=.85,cmap='YlGnBu');ax.set_xticks(range(3),[x[0].replace('_',' ') for x in c.PAIRS]);ax.set_yticks(range(len(reps)),reps)
for i in range(len(reps)):
 for j in range(3):ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='black')
ax.set_title('Validation-selected linear probes: test ROC-AUC');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(p/'pairwise_auc.png',dpi=180);plt.close(fig)
(p/'report_validation.json').write_text(json.dumps({'complete_exit0':True,'24_primary_saved_scores_metrics_recomputed':True,'33_val_C_selection_verified':True,'patient_test_pair_counts_verified':True,'E7_error_counts_and_303_cases_verified':True,'frozen_and_cache_validation_passed':True,'supplemental_scope':'9 summary probes have metrics and validation candidates preserved; fitted coefficients/test scores were not saved by extraction helper, so these supplemental metrics are not independently reconstructed here','plots':'training-fit PCA projection and stored AUC only; no extra test inference/training'},indent=2))
print('E12 validation passed')
