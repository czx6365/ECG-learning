"""E34 fixed R-relative local CNN residual pilot; frozen E29 and strict crossfit offsets."""
import json,time,copy,sys,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold
import run_e29_s as s
from scripts.data_prep.ECG_DSA.tokenize_pretrain_qrs import QRSTokenizerLite
b,e,p=s.b,s.e,s.p
ROOT=e.ROOT;OUT=ROOT/'results/e34_raw_ecg_pilot';DEV=torch.device('cuda')
RAW=ROOT/'results/pre_e34_evaluation/04_e33c_raw_information/train_100hz.npy'
INNER=ROOT/'results/e32_omi_nomi_residual'
WINDOWS={'B':(-8,12),'C':(8,40),'D':(-30,70)}
METHODS=('B','C','D','E')
def dump(name,x):b.dump(OUT/name,x)
def status(x):b.status(OUT,x)
class CNN(nn.Module):
 def __init__(self,seed):
  super().__init__();torch.manual_seed(seed)
  self.encoder=nn.Sequential(nn.Conv1d(9,16,5,padding=2),nn.GELU(),nn.Conv1d(16,24,5,padding=2),nn.GELU(),nn.Conv1d(24,24,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool1d(1))
  self.readout=nn.Linear(24,4);nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
 def delta(self,x,mask):
  n,k,c,l=x.shape
  z=self.encoder(x.reshape(n*k,c,l)).reshape(n,k,24)
  z=(z*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
  return 2*torch.tanh(self.readout(z)/2)
 def forward(self,x,mask,q):
  d=self.delta(x,mask)
  # algebraically q*exp(delta)/sum(q*exp(delta)); exact identity at delta=0
  a=torch.expm1(d);u=(q*a).sum(1,keepdim=True)
  return q+q*(a-u)/(1+u),d
@torch.no_grad()
def predict(model,x,mask,q,ids):
 model.eval();out=[]
 for ix in np.array_split(np.asarray(ids),max(1,int(np.ceil(len(ids)/128)))):
  out.append(model(x[ix],mask[ix],q[ix])[0].cpu().numpy())
 return np.concatenate(out)
def raw_cache(meta,y):
 if (OUT/'raw_cache_complete').exists():return
 wave=np.load(RAW,mmap_mode='r');assert wave.shape==(len(meta),9,1000)
 raw=np.empty_like(wave);sources={};alignment=[]
 for j,row in enumerate(meta.itertuples()):
  src=Path(str(row.source_dataset).replace('/pretrain_qrs/','/pretrain_waveforms/').replace('/qrs/','/waveforms/'))
  src=src if src.is_absolute() else ROOT/src;ss=str(row.source_split);assert ss in ('train','val')
  key=(str(src),ss)
  if key not in sources:
   sm=json.load(open(src/'prepare_summary.json'))
   sources[key]=(np.load(src/f'{ss}_data.npy',mmap_mode='r'),pd.read_csv(src/f'{ss}_metadata.csv'),sm)
  array,frame,sm=sources[key];idx=int(row.source_index)
  if str(frame.iloc[idx].csv_name)!=str(row.csv_name):
   found=np.flatnonzero(frame.csv_name.astype(str).values==str(row.csv_name));assert len(found)==1;idx=int(found[0])
  np.testing.assert_array_equal(wave[j],array[idx]);assert np.isfinite(wave[j]).all()
  raw[j]=wave[j]*sm['global_std']+sm['global_mean']
  if key not in [v['key'] for v in alignment]:alignment.append(dict(key=key,source_mean=sm['global_mean'],source_std=sm['global_std'],lead_names=sm['lead_names']))
 np.save(OUT/'train_unscaled_wave.npy',raw)
 windows={m:np.zeros((len(meta),16,9,hi-lo),np.float32) for m,(lo,hi) in WINDOWS.items()}
 mask=np.zeros((len(meta),16),bool);quality=[];tok=QRSTokenizerLite(fs=100)
 for i,w in enumerate(raw):
  peaks=tok.qrs_detection(w)
  usable=[int(r) for r,nxt in zip(peaks[:-1],peaks[1:]) if r>=30 and r+70<=1000 and nxt-r>=48]
  if len(usable)>16:usable=np.asarray(usable)[np.linspace(0,len(usable)-1,16).astype(int)].tolist()
  mask[i,:len(usable)]=True
  for m,(lo,hi) in WINDOWS.items():
   for k,r in enumerate(usable):windows[m][i,k]=w[:,r+lo:r+hi]
  quality.append(dict(row=i,label=int(y[i].argmax()),detected=len(peaks),usable=len(usable),rr_mean=float(np.diff(peaks).mean()/100) if len(peaks)>1 else None,
   extended_next_r_overlap=sum(any((peaks>r)&(peaks<r+70)) for r in usable)))
  if i%1000==0:print('raw cache',i,flush=True)
 for m,x in windows.items():np.save(OUT/f'windows_{m}.npy',x)
 np.save(OUT/'beat_mask.npy',mask);pd.DataFrame(quality).to_csv(OUT/'r_peak_quality.csv',index=False)
 # inspect only selected Train patient CSVs from the available original archive
 archive=ROOT/'datasets/患者/data_0521.zip';time_audit=[]
 with zipfile.ZipFile(archive) as z:
  lookup={Path(n).name:n for n in z.namelist() if n.endswith('.csv')}
  for label in range(3):
   row=meta[meta.label_id==label].iloc[0];frame=pd.read_csv(z.open(lookup[row.csv_name]));t=frame['t'].to_numpy();dt=np.median(np.diff(t))
   time_audit.append(dict(label=label,rows=len(t),dt=float(dt),duration=float(t[-1]-t[0]+dt),fs=float(1/dt),columns=list(frame.columns)))
 dump('data_audit.json',dict(records=len(meta),class_counts=np.bincount(y.argmax(1)).tolist(),all_rows_source_alignment_exact=True,
  source_transform=alignment,fs_cached=100,source_patient_time_audit=time_audit,fs_evidence='selected Train source CSV time columns; healthy local audit separate; cache resample_poly to1000',
  physical_units='CSV has no unit metadata; unscaled source numerical units, NOT verified mV',preprocessing='invert source global mean/std; each outer-dev fits one shared per-lead mean/std; no beat normalization or baseline subtraction',
  windows_samples=WINDOWS,windows_seconds={m:[v/100 for v in z] for m,z in WINDOWS.items()},clinical_boundaries=False,
  peak_detector='WFDB XQRS on lead I at100Hz, unchanged E30 detector',beat_selection='same centers all branches; complete extended window; next R at least480ms away to avoid next QRS in late proxy; at most16 uniformly selected',
  missing_by_class={str(k):dict(n=int((y.argmax(1)==k).sum()),no_usable=int((~mask.any(1)&(y.argmax(1)==k)).sum())) for k in range(4)},
  source_cache_sha256=e.sha256(RAW),metadata_sha256=e.sha256(e.DOWNSTREAM/'train_metadata.csv'),test_arrays_read=False))
 (OUT/'raw_cache_complete').touch()
def frozen_data(y):
 cache=ROOT/'results/e29_all_layer_fusion/cache'
 audit=json.load(open(cache.parent/'cache_audit.json'));assert audit['metadata_sha256']==e.sha256(e.DOWNSTREAM/'train_metadata.csv')
 pools=np.load(cache/'pooled.npy',mmap_mode='r');tokens=np.load(cache/'layer9.npy',mmap_mode='r')
 z=torch.empty((len(y),4608+256*769),device=DEV)
 z[:,:4608]=torch.tensor(pools[:,[5,8,11]].reshape(len(y),-1).copy(),device=DEV)
 z[:,4608:]=torch.tensor(np.array(tokens).reshape(len(y),-1),device=DEV)
 np.testing.assert_allclose(z[:,:4608].cpu().numpy().reshape(-1,3,1536),np.load(p.source/'features/train.npy'),atol=2e-5,rtol=1e-5)
 load=lambda name:torch.tensor(np.load(e.E10_RUN/f'features/train_{name}.npy'),device=DEV)
 regions,mask=e.e11.masked_region_means(load('lead_ze_mean_max'),load('lead_present').bool())
 return dict(z=z,g=load('global_ze_mean_max'),regions=regions,mask=mask,y=torch.tensor(y,device=DEV))
@torch.no_grad()
def base_predict(paths,data,ids):
 result=[];sub=s.subset(data,ids)
 for path in paths:
  m=s.PairedHead('shared',0).to(DEV);m.load_state_dict(torch.load(path,map_location=DEV,weights_only=False)['model']);m.eval().requires_grad_(False)
  assert not any(p.requires_grad for p in m.parameters())
  result.append(b.evaluate(m,sub)[1]);del m
 return np.mean(result,0)
def offsets(fold,folds,meta,y,data,original):
 file=OUT/f'fold{fold}_crossfit_offsets.npz'
 if file.exists():return np.load(file)['prob']
 dev=np.sort(folds[(folds.fold==fold)&(folds.role!='outer')].row.to_numpy());outer=folds[(folds.fold==fold)&(folds.role=='outer')].row.to_numpy()
 assert not set(meta.iloc[dev].entity_id)&set(meta.iloc[outer].entity_id)
 q=np.full((len(y),4),np.nan,np.float32);audit=[];assign=[]
 split=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=32050+fold)
 for inner,(tr,vi) in enumerate(split.split(dev,y[dev].argmax(1),meta.iloc[dev].entity_id)):
  train,valid=dev[tr],dev[vi]
  split2=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=33050+fold*10+inner)
  fi,si=next(split2.split(train,y[train].argmax(1),meta.iloc[train].entity_id));fit,sel=train[fi],train[si]
  assert not set(meta.iloc[valid].entity_id)&(set(meta.iloc[fit].entity_id)|set(meta.iloc[sel].entity_id))
  paths=[INNER/f'fold{fold}/inner{inner}/downstream/shared/seed{k}/checkpoint-best.pth' for k in range(3)]
  # Match existing source manifest against reconstructed training partition sizes
  old=json.load(open(INNER/'inner_crossfit_audit.json'))
  entry=next(a for a in old if a['outer_fold']==fold and a['inner_fold']==inner)
  assert (entry['fit'],entry['select'],entry['held'])==(len(fit),len(sel),len(valid))
  q[valid]=base_predict(paths,data,valid)
  audit.append(dict(fold=fold,inner=inner,fit=len(fit),select=len(sel),held=len(valid),entity_overlap=0,checkpoint_sha256={str(path):e.sha256(path) for path in paths}))
  for role,ix in [('base_fit',fit),('base_select',sel),('fusion_crossfit',valid)]:assign.extend(dict(fold=fold,inner=inner,role=role,row=int(j)) for j in ix)
 # Original model's select/outer predictions are separate from crossfit training probabilities.
 paths=[s.OUT/f'fold{fold}/downstream/shared/seed{k}/checkpoint-best.pth' for k in range(3)]
 q[outer]=base_predict(paths,data,outer)
 np.testing.assert_allclose(q[outer],original[outer],atol=3e-6,rtol=1e-5)
 assert np.isfinite(q).all();np.savez(file,prob=q,dev=dev,outer=outer)
 dump(f'fold{fold}_crossfit_audit.json',audit);pd.DataFrame(assign).to_csv(OUT/f'fold{fold}_nested_assignment.csv',index=False)
 return q

def main():
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);start=time.time()
 meta=pd.read_csv(e.DOWNSTREAM/'train_metadata.csv');y=np.load(e.DOWNSTREAM/'train_labels.npy');assert np.array_equal(meta.label_id,y.argmax(1))
 folds=pd.read_csv(s.OUT/'fold_assignment.csv');folds.to_csv(OUT/'fold_assignment.csv',index=False)
 original=np.load(s.OUT/'oof_shared.npy').mean(0);ref=json.load(open(s.OUT/'cv_summary.json'))['shared']['ensemble']
 assert abs(s.metrics(y,original)['overall']-ref['overall'])<1e-9
 dump('config.json',dict(methods={'A':'original frozen E29 Ensemble3','B':'R[-80,+120)ms QRS proxy','C':'R[+80,+400)ms ST-T proxy','D':'R[-300,+700)ms extended beat','E':'same CNN capacity using E29 probabilities repeated in time, NO raw'},epochs=30,checkpoint='fixed final epoch30, no tuning or epoch selection',seeds=[0,1,2],optimizer='AdamW lr3e-4 wd.01',loss='NLL(corrected probability)+0.02 mean(delta^2)',residual='bounded four-class log-prob correction, exact zero identity',fusion_fit='outer development (original fit+select); twofold grouped OOF frozen E32 E29 models; same outer folds; no new E29 training',scaler='outer development per-lead; no outer fitting',test_access=False,windows=WINDOWS))
 dump('environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),trainable_parameters_per_raw_seed=sum(p.numel() for p in CNN(0).parameters())))
 paths=[e.TOKENIZER,e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth',*s.OUT.glob('fold*/downstream/shared/seed*/checkpoint-best.pth'),*INNER.glob('fold*/inner*/downstream/shared/seed*/checkpoint-best.pth')]
 hashes={str(path):e.sha256(path) for path in paths}
 status('data_audit');raw_cache(meta,y)
 data=frozen_data(y);wave=np.load(OUT/'train_unscaled_wave.npy',mmap_mode='r');mask=torch.tensor(np.load(OUT/'beat_mask.npy'),device=DEV)
 output={m:np.full((3,len(y),4),np.nan,np.float32) for m in METHODS};logs=[];smoke=[]
 for fold in range(3):
  status(f'fold{fold}_crossfit_frozen_E29');q=offsets(fold,folds,meta,y,data,original)
  dev=np.sort(folds[(folds.fold==fold)&(folds.role!='outer')].row.to_numpy());outer=folds[(folds.fold==fold)&(folds.role=='outer')].row.to_numpy()
  mean=np.asarray(wave[dev]).mean((0,2));std=np.asarray(wave[dev]).std((0,2)).clip(1e-3);np.savez(OUT/f'fold{fold}_scaler.npz',mean=mean,std=std)
  qt=torch.tensor(q,device=DEV);target=torch.tensor(y.argmax(1),device=DEV)
  for method in METHODS:
   if method=='E':
    values=q[:,np.arange(9)%4];x=torch.tensor(np.broadcast_to(values[:,None,:,None],(len(y),1,9,100)).copy(),device=DEV);bm=torch.ones((len(y),1),dtype=torch.bool,device=DEV)
   else:
    values=np.load(OUT/f'windows_{method}.npy');values=(values-mean[None,None,:,None])/std[None,None,:,None];values[~mask.cpu().numpy()]=0
    x=torch.tensor(values,device=DEV);bm=mask
   for seed in range(3):
    status(f'fold{fold}_{method}_seed{seed}');folder=OUT/f'fold{fold}/{method}/seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    model=CNN(seed).to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);gen=torch.Generator().manual_seed(seed)
    with torch.no_grad():
     init,_=model(x[:8],bm[:8],qt[:8]);assert torch.equal(init,qt[:8]);smoke.append(dict(fold=fold,method=method,seed=seed,initial_prob_exact=True))
    if (folder/'COMPLETE').exists():
     model.load_state_dict(torch.load(folder/'checkpoint.pth',map_location=DEV,weights_only=False)['model'])
     log=json.load(open(folder/'training.json'))
    else:
     log=[];timer=time.time()
     for epoch in range(1,31):
      model.train();order=dev[torch.randperm(len(dev),generator=gen).numpy()];losses=[]
      for ix in np.array_split(order,int(np.ceil(len(order)/64))):
       opt.zero_grad();prob,delta=model(x[ix],bm[ix],qt[ix]);loss=F.nll_loss(prob.clamp_min(1e-8).log(),target[ix])+.02*delta.square().mean();assert torch.isfinite(loss)
       loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step();losses.append(float(loss.detach()))
      log.append(dict(epoch=epoch,loss=float(np.mean(losses))))
     torch.save(dict(model=model.state_dict(),epoch=30,fold=fold,seed=seed,method=method,fit_rows=dev),folder/'checkpoint.pth');b.dump(folder/'training.json',log);(folder/'COMPLETE').touch()
    pred=predict(model,x,bm,qt,outer);output[method][seed,outer]=pred
    metric=s.metrics(y[outer],pred);logs.append(dict(fold=fold,method=method,seed=seed,**metric));np.save(folder/'outer_prob.npy',pred)
    np.save(OUT/f'oof_{method}_seeds.npy',output[method]);pd.DataFrame(logs).to_csv(OUT/'fold_seed_metrics.csv',index=False)
   del x
  dump('smoke.json',smoke)
  torch.cuda.empty_cache()
 np.save(OUT/'oof_A.npy',original)
 summary={'A':dict(ensemble=s.metrics(y,original))}
 for method,z in output.items():
  assert np.isfinite(z).all() and (z>=0).all();np.testing.assert_allclose(z.sum(-1),1,atol=2e-6)
  summary[method]=s.summarize(y,z)
 dump('metrics.json',summary)
 assert all(e.sha256(Path(path))==h for path,h in hashes.items())
 dump('execution_audit.json',dict(seconds=time.time()-start,original_hashes_unchanged=True,source_sha256=hashes,test_arrays_read=False,frozen_modules='all original E29/E25/E11, E12-A, MLM, tokenizer, codebook; only new CNN parameters optimized'))
 status('training_complete')

if __name__=='__main__':main()
