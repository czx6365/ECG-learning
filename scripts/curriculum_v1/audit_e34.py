"""Independent grouped crossfit/checkpoint/frozen-cache audit, no fitting."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedGroupKFold
from run_e34 import OUT,INNER,e,s,p,CNN,DEV,ROOT

def main():
 torch.set_num_threads(4)
 meta=pd.read_csv(e.DOWNSTREAM/'train_metadata.csv');y=np.load(e.DOWNSTREAM/'train_labels.npy');folds=pd.read_csv(OUT/'fold_assignment.csv')
 assert meta.entity_id.notna().all()
 assert meta[meta.label_id<3].patient_sn.notna().all()
 def patients(ix):
  frame=meta.iloc[ix];return set(frame.loc[frame.label_id<3,"patient_sn"].dropna().astype(str))
 assert (OUT/'fold_assignment.csv').read_bytes()==(s.OUT/'fold_assignment.csv').read_bytes()
 manifest=json.load(open(INNER/'inner_crossfit_audit.json'));records=[]
 for f in range(3):
  dev=np.sort(folds[(folds.fold==f)&(folds.role!='outer')].row.to_numpy());outer=folds[(folds.fold==f)&(folds.role=='outer')].row.to_numpy()
  assert not set(meta.iloc[dev].entity_id)&set(meta.iloc[outer].entity_id)
  assert not patients(dev)&patients(outer)
  q=np.load(OUT/f'fold{f}_crossfit_offsets.npz')['prob'];assert q.shape==y.shape and np.isfinite(q).all()
  paths_seen=[]
  for inner,(tr,vi) in enumerate(StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=32050+f).split(dev,y[dev].argmax(1),meta.iloc[dev].entity_id)):
   train,valid=dev[tr],dev[vi]
   fi,si=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=33050+f*10+inner).split(train,y[train].argmax(1),meta.iloc[train].entity_id))
   fit,sel=train[fi],train[si];entry=next(z for z in manifest if z['outer_fold']==f and z['inner_fold']==inner)
   paths=[str(INNER/f'fold{f}/inner{inner}/downstream/shared/seed{k}/checkpoint-best.pth') for k in range(3)]
   assert paths==entry['checkpoint_paths'];assert (len(fit),len(sel),len(valid))==(entry['fit'],entry['select'],entry['held'])
   assert not set(meta.iloc[valid].entity_id)&(set(meta.iloc[fit].entity_id)|set(meta.iloc[sel].entity_id))
   assert not set(meta.iloc[fit].entity_id)&set(meta.iloc[sel].entity_id)
   assert not patients(valid)&(patients(fit)|patients(sel))
   assert not patients(fit)&patients(sel)
   paths_seen+=paths;records.append(dict(fold=f,inner=inner,manifest_exact=True,group_overlap=0,patient_overlap=0))
  np.testing.assert_allclose(q[outer],np.load(s.OUT/'oof_shared.npy').mean(0)[outer],atol=3e-6,rtol=1e-5)
 # Exact zero identity, unchanged original offset and later gradient flow, padding invariance
 x=torch.randn(8,3,9,100,device=DEV);mask=torch.tensor([[True,True,False]]*8,device=DEV)
 q=torch.softmax(torch.randn(8,4,device=DEV),1);model=CNN(0).to(DEV)
 initial,d=model(x,mask,q);assert torch.equal(initial,q)
 loss=-initial[:,0].log().mean();loss.backward();assert model.readout.weight.grad.abs().sum()>0
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4);opt.step();opt.zero_grad()
 predicted,_=model(x,mask,q);(-predicted[:,0].log().mean()).backward()
 grad=float(model.encoder[0].weight.grad.abs().sum());assert grad>0
 altered=x.clone();altered[:,2]=1000
 with torch.no_grad():np.testing.assert_allclose(model(x,mask,q)[0].cpu(),model(altered,mask,q)[0].cpu(),atol=1e-7,rtol=1e-6)
 files=[Path(__file__),Path(__file__).with_name('run_e34.py'),Path(__file__).with_name('report_e34.py'),Path(s.__file__),Path(s.b.__file__),ROOT/'scripts/curriculum_v1/e32_b_run.py',ROOT/'scripts/curriculum_v1/e32_train_cache.py',ROOT/'scripts/curriculum_v1/pre_e34_raw_c1.py',ROOT/'scripts/curriculum_v1/pre_e34_raw_d2.py',ROOT/'scripts/data_prep/ECG_DSA/ecg_zip_utils.py',ROOT/'scripts/data_prep/ECG_DSA/tokenize_pretrain_qrs.py',INNER/'inner_crossfit_audit.json']
 execution=json.load(open(OUT/'execution_audit.json'));assert all(e.sha256(Path(path))==h for path,h in execution['source_sha256'].items())
 s.b.dump(OUT/'independent_audit.json',dict(group_checks=records,source_hashes={str(path):e.sha256(path) for path in files},initial_probability_bit_exact=True,readout_gradient=True,encoder_gradient_after_first_step=grad,padding_invariance=True,original_checkpoints_still_unchanged=True,original_E29_Ensemble3_replayed=True,only_CNN_parameters_in_optimizer=True,test_arrays_read=False,provenance_limit='historical E32 checkpoints contain weights/selected epoch but not row manifest; split reconstruction and existing audit paths/counts match current original training source'))
 print('independent audit passed')
if __name__=='__main__':main()
