import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,confusion_matrix,accuracy_score
import run_e13_control_audit as e
p=e.RUN;rows=json.loads((p/'comparison.json').read_text())['results'];validation=json.loads((p/'report_validation.json').read_text());classes=e.NAMES
for row in rows:
 out=p/f"{row['method']}_seed{row['seed']}"
 for s in ('val','test'):
  ix=np.load(p/f'{s}_parent_patient_indices.npy');y=np.load(e.e4.FEATURES/f'{s}_labels.npy')[ix,:3];raw=np.load(out/f'{s}_raw_outputs.npy');pred=raw.argmax(1);target=y.argmax(1);z=np.exp(raw-raw.max(1,keepdims=True));scores=z/z.sum(1,keepdims=True);pr,rc,f1,sup=precision_recall_fscore_support(target,pred,labels=[0,1,2],zero_division=0);m=row[s];assert confusion_matrix(target,pred,labels=[0,1,2]).tolist()==m['confusion_matrix'];assert np.isclose(accuracy_score(target,pred),m['accuracy']);assert np.isclose(rc.mean(),m['balanced_accuracy']) and np.isclose(f1.mean(),m['macro_f1'])
  for k,name in enumerate(classes):
   for key,v in [('precision',pr[k]),('recall',rc[k]),('f1',f1[k]),('auc',roc_auc_score(y[:,k],scores[:,k]))]:assert np.isclose(v,m['per_class'][name][key],atol=1e-12)
  for key,a,b in [('control_vs_omi_auc',2,0),('control_vs_nomi_auc',2,1),('omi_vs_nomi_auc',0,1)]:
   mask=(target==a)|(target==b);assert np.isclose(roc_auc_score(target[mask]==a,raw[mask,a]-raw[mask,b]),m[key],atol=1e-12)
# Check identical padded-wave hashes across split, distinguishing empty records.
seen={};duplicates={'within_split_nonempty_excess':0,'cross_split_nonempty_excess':0,'empty_records':0};groups={}
for s in e.SPLITS:
 lab=np.load(e.e4.FEATURES/f'{s}_labels.npy').argmax(1);wave=np.load(e.e4.DATA/f'{s}_data.npy',mmap_mode='r');valid=np.load(e.e4.FEATURES/f'{s}_valid_heartbeat_mask.npy')
 for row in np.flatnonzero(lab<3):
  if not valid[row].any():duplicates['empty_records']+=1;continue
  key=hashlib.sha256(np.asarray(wave[row]).tobytes()).hexdigest()
  if key in seen:duplicates['within_split_nonempty_excess' if seen[key]==s else 'cross_split_nonempty_excess']+=1
  else:seen[key]=s
validation.update({'independent_sklearn_val_test_metrics_and_pair_AUC_verified':True,'nonempty_waveform_duplicate_audit':duplicates,'source_fingerprints_match_now':all(e.e4.sha(Path(k))==v for k,v in json.loads((p/'source_fingerprints.json').read_text()).items()),'extended_audit_source_sha256':e.e4.sha(Path(__file__).with_name('e13_extended_audit.py'))})
assert validation['source_fingerprints_match_now'];(p/'report_validation.json').write_text(json.dumps(validation,indent=2))
print(validation)
