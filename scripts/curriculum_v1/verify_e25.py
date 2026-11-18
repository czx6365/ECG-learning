"""Recompute E25 metrics, verify frozen split and compare historical anchor."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import run_e25 as x
e=x.e;r=e.ROOT/'results/e25_intermediate_layers'
assert (r/'COMPLETE').exists()
contract=json.loads((e.ROOT/'configs/curriculum_v1/data_contract.json').read_text());manifest=e.DOWNSTREAM/'split_manifest.csv'
assert e.sha256(manifest)==contract['split_manifest_sha256']
ids={};labels={}
for s in ('train','val'):
 f=pd.read_csv(e.DOWNSTREAM/f'{s}_metadata.csv');labels[s]=np.load(e.DOWNSTREAM/f'{s}_labels.npy');healthy=f['cohort'].astype(str).eq('healthy')
 ids[s]=np.where(healthy,'healthy:'+f['csv_name'].astype(str).str.rsplit('_',n=1).str[-1],'patient:'+f['patient_sn'].astype(str))
assert not set(ids['train'])&set(ids['val'])
y=labels['val'];base=json.loads((r/'validation_summary.json').read_text());audit={'split_manifest_sha256':e.sha256(manifest),'train_val_identity_overlap':0,'validation_records':len(y),'validation_entities':len(set(ids['val'])),'test_loaded':False,'historical_layer12':[],'verified_runs':0}
for m in x.METHODS:
 for s in (0,1,2):
  p=r/'downstream'/m/f'seed{s}';row=json.loads((p/'summary.json').read_text());raw=np.load(p/'val_logits.npy');metrics,prob=e.probability_metrics(raw,y);assert np.allclose(prob,np.load(p/'val_probabilities.npy'),atol=1e-7)
  assert abs(metrics['r2']-row['val']['r2'])<1e-10
  history=json.loads((p/'history.json').read_text());assert max(history,key=lambda z:z['val']['r2'])['epoch']==row['best_epoch'];audit['verified_runs']+=1
  if m=='layer12':
   old=json.loads((e.DEFAULT_RUN/'downstream/e12a_region'/f'seed{s}'/'val_summary.json').read_text());audit['historical_layer12'].append({'seed':s,'new':metrics['r2'],'historical':old['val']['r2'],'delta':metrics['r2']-old['val']['r2']})
# Entity-cluster paired bootstrap of the fixed three-seed ensemble.
unique=np.unique(ids['val']);groups=[np.flatnonzero(ids['val']==a) for a in unique];rng=np.random.default_rng(25006);samples=[np.concatenate([groups[j] for j in row]) for row in rng.integers(0,len(groups),size=(2000,len(groups)))];ref=np.mean([np.load(r/'downstream/layer12'/f'seed{s}/val_probabilities.npy') for s in (0,1,2)],0);comparisons={}
for m in x.METHODS:
 if m=='layer12':continue
 p=np.mean([np.load(r/'downstream'/m/f'seed{s}/val_probabilities.npy') for s in (0,1,2)],0)
 deltas=[r2_score(y[ii],p[ii])-r2_score(y[ii],ref[ii]) for ii in samples]
 comparisons[m]={'ensemble_delta':float(r2_score(y,p)-r2_score(y,ref)),'entity_cluster_bootstrap_ci95':np.quantile(deltas,[.025,.975]).tolist(),'paired_downstream_seed_deltas':[base[m]['per_seed'][s]['val']['r2']-base['layer12']['per_seed'][s]['val']['r2'] for s in range(3)]}
x.dump(r/'verification.json',audit);x.dump(r/'entity_bootstrap.json',comparisons)
print(json.dumps({'audit':audit,'comparisons':comparisons},ensure_ascii=False,indent=2))
