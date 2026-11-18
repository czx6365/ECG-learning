"""Nested cross-fitted raw prediction residual against fixed E29."""
import json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,r2_score
import e32_b_run as b32

ROOT=b32.e.ROOT;OUT=ROOT/'results/pre_e34_evaluation/04_e33c_raw_information';DEV=torch.device('cuda')
meta=pd.read_csv(b32.e.DOWNSTREAM/'train_metadata.csv');y4=np.load(b32.e.DOWNSTREAM/'train_labels.npy');y=y4.argmax(1)
folds=pd.read_csv(b32.SRC/'fold_assignment.csv')
wave=np.load(OUT/'train_100hz.npy',mmap_mode='r')
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(9,32,9,stride=2,padding=4),nn.GELU(),nn.Conv1d(32,64,7,stride=2,padding=3),nn.GELU(),nn.Conv1d(64,64,5,stride=2,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1));self.head=nn.Linear(64,4)
 def forward(self,x):return self.head(self.net(x).squeeze(-1))
def rawfit(fit,sel,held,fold,inner):
 mean=wave[fit].mean(axis=(0,2),keepdims=True);std=wave[fit].std(axis=(0,2),keepdims=True).clip(1e-3)
 x=torch.tensor((np.asarray(wave)-mean)/std,device=DEV);yt=torch.tensor(y,device=DEV)
 pp=[]
 for seed in range(3):
  torch.manual_seed(seed);m=Net().to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01)
  gen=torch.Generator().manual_seed(seed);best=-1e9
  for epoch in range(1,31):
   m.train();order=torch.tensor(fit[torch.randperm(len(fit),generator=gen)],device=DEV)
   for ids in order.split(64):
    opt.zero_grad();loss=F.cross_entropy(m(x[ids]),yt[ids]);loss.backward();opt.step()
   m.eval()
   with torch.no_grad():q=torch.cat([m(x[ids]).softmax(1) for ids in torch.tensor(sel,device=DEV).split(128)]).cpu().numpy()
   score=float(r2_score(y4[sel],q))
   if score>best:
    best=score;torch.save({'model':m.state_dict(),'epoch':epoch},OUT/f'd2_fold{fold}_inner{inner}_seed{seed}.pth')
  m.load_state_dict(torch.load(OUT/f'd2_fold{fold}_inner{inner}_seed{seed}.pth',map_location=DEV,weights_only=False)['model']);m.eval()
  with torch.no_grad():pp.append(torch.cat([m(x[ids]).softmax(1) for ids in torch.tensor(held,device=DEV).split(128)]).cpu().numpy())
  print('D2 raw inner',fold,inner,seed,flush=True)
 return np.mean(pp,axis=0)
load=lambda name:torch.tensor(np.load(b32.e.E10_RUN/f'features/train_{name}.npy'),device=DEV)
regions,mask=b32.e.e11.masked_region_means(load('lead_ze_mean_max'),load('lead_present').bool())
data={'train':{'z':torch.tensor(np.load(b32.p.source/'features/train.npy'),device=DEV),'g':load('global_ze_mean_max'),'regions':regions,'mask':mask,'y':torch.tensor(y4,device=DEV)}}
from e32_train_cache import extract
extract(data);data=data['train'];torch.set_num_threads(4)
e29_oof=np.load(b32.SRC/'oof_shared.npy').mean(0);raw_oof=np.load(OUT/'C1_oof_ensemble.npy')
d2=np.full_like(e29_oof,np.nan)
for fold in range(3):
 dev=np.sort(folds[(folds.fold==fold)&(folds.role!='outer')].row.to_numpy());held=folds[(folds.fold==fold)&(folds.role=='outer')].row.to_numpy()
 splitter=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=32050+fold)
 raw=np.full((len(dev),4),np.nan);base=np.full((len(dev),4),np.nan)
 for inner,(tr,vi) in enumerate(splitter.split(dev,y[dev],meta.iloc[dev].entity_id)):
  train=dev[tr];valid=dev[vi]
  selection=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=33050+fold*10+inner)
  fi,si=next(selection.split(train,y[train],meta.iloc[train].entity_id));fit,sel=train[fi],train[si]
  raw[vi]=rawfit(fit,sel,valid,fold,inner)
  models=[]
  for seed in range(3):
   path=b32.OUT/f'fold{fold}/inner{inner}/downstream/shared/seed{seed}/checkpoint-best.pth'
   m=b32.s.PairedHead('shared',seed).to(DEV);m.load_state_dict(torch.load(path,map_location=DEV,weights_only=False)['model']);models.append(m.eval())
  base[vi]=b32.predict(models,b32.restrict(data,valid))[0]
 assert np.isfinite(raw).all() and np.isfinite(base).all()
 pair=y[dev]<2;re=np.log(base[pair,0]/base[pair,1]);rr=np.log(raw[pair,0]/raw[pair,1]);t=(y[dev][pair]==0).astype(float)
 def loss(w):
  delta=w[0]*rr+w[1];u=re+delta
  return np.logaddexp(0,u).mean()-(t*u).mean()+.02*np.square(delta).mean()
 fit=minimize(loss,[0.,0.],method='BFGS');w=fit.x
 r0=np.log(e29_oof[held,0]/e29_oof[held,1]);rraw=np.log(raw_oof[held,0]/raw_oof[held,1])
 d2[held]=b32.apply(e29_oof[held],r0+w[0]*rraw+w[1]);np.save(OUT/'D2_oof.npy',d2)
 (OUT/f'd2_fold{fold}_weights.json').write_text(json.dumps({'weights':w.tolist(),'fit_n':int(pair.sum()),'inner_crossfit':2},indent=2))
 print('D2 fold',fold,w,flush=True)
assert np.isfinite(d2).all()
pair=y<2
q=d2[pair,0]/d2[pair,:2].sum(1)
result={'auc':float(roc_auc_score(y[pair]==0,q)),'overall':float(r2_score(y4,d2)),'clinical3':float(r2_score(y4[:,:3],d2[:,:3]))}
(OUT/'D2_metrics.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
