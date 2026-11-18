"""Predefined raw 100 Hz full-window 1D-CNN diagnostic, E33-C C1."""
from pathlib import Path
import sys
import json,time
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,r2_score
sys.path.insert(0,'/home/ubuntu/HeartLang-main')

ROOT=Path('/home/ubuntu/HeartLang-main');DATA=ROOT/'datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1'
OUT=ROOT/'results/pre_e34_evaluation/04_e33c_raw_information';OUT.mkdir(parents=True,exist_ok=True)
meta=pd.read_csv(DATA/'train_metadata.csv');y=np.load(DATA/'train_labels.npy');labels=y.argmax(1)
folds=pd.read_csv(ROOT/'results/e29_shared_generalization/fold_assignment.csv');dev=torch.device('cuda')
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(9,32,9,stride=2,padding=4),nn.GELU(),nn.Conv1d(32,64,7,stride=2,padding=3),nn.GELU(),nn.Conv1d(64,64,5,stride=2,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1));self.head=nn.Linear(64,4)
    def forward(self,x):return self.head(self.net(x).squeeze(-1))
def raw():
    path=OUT/'train_100hz.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    arr=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(meta),9,1000));srcs={}
    for j,row in enumerate(meta.itertuples()):
        src=Path(str(row.source_dataset));src=src if src.is_absolute() else ROOT/src
        path0=Path(str(src).replace('/pretrain_qrs/','/pretrain_waveforms/').replace('/qrs/','/waveforms/'))
        key=(str(path0),str(row.source_split))
        if key not in srcs:srcs[key]=(np.load(path0/f'{key[1]}_data.npy',mmap_mode='r'),pd.read_csv(path0/f'{key[1]}_metadata.csv'))
        wave,frame=srcs[key];idx=int(row.source_index)
        if str(frame.iloc[idx].csv_name)!=str(row.csv_name):
            found=np.flatnonzero(frame.csv_name.astype(str).values==str(row.csv_name));assert len(found)==1;idx=int(found[0])
        arr[j]=wave[idx]
    arr.flush();return np.load(path,mmap_mode='r')
wave=raw();torch.set_num_threads(4)
oof=np.full((3,len(y),4),np.nan,np.float32);results=[]
for fold in range(3):
    roles={role:z.row.to_numpy() for role,z in folds[folds.fold==fold].groupby('role')}
    fit,select,held=[roles[z] for z in ('fit','select','outer')]
    mean=wave[fit].mean(axis=(0,2),keepdims=True);std=wave[fit].std(axis=(0,2),keepdims=True).clip(1e-3)
    np.savez(OUT/f'fold{fold}_scaler.npz',mean=mean,std=std)
    x=torch.tensor((np.asarray(wave)-mean)/std,device=dev);yt=torch.tensor(labels,device=dev)
    for seed in range(3):
        torch.manual_seed(seed);model=Net().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        best=-1e9;gen=torch.Generator().manual_seed(seed);start=time.time()
        for epoch in range(1,31):
            model.train();order=torch.tensor(fit[torch.randperm(len(fit),generator=gen)],device=dev)
            for ids in order.split(64):
                opt.zero_grad();loss=F.cross_entropy(model(x[ids]),yt[ids]);loss.backward();opt.step()
            model.eval()
            with torch.no_grad():q=torch.cat([model(x[ids]).softmax(1) for ids in torch.tensor(select,device=dev).split(128)]).cpu().numpy()
            metric=float(r2_score(y[select],q))
            if metric>best:
                best=metric;best_epoch=epoch;torch.save({'model':model.state_dict(),'epoch':epoch,'metric':metric},OUT/f'fold{fold}_seed{seed}.pth')
        model.load_state_dict(torch.load(OUT/f'fold{fold}_seed{seed}.pth',map_location=dev,weights_only=False)['model']);model.eval()
        with torch.no_grad():oof[seed,held]=torch.cat([model(x[ids]).softmax(1) for ids in torch.tensor(held,device=dev).split(128)]).cpu().numpy()
        results.append({'fold':fold,'seed':seed,'best_epoch':best_epoch,'selection_r2':best,'seconds':time.time()-start})
        np.save(OUT/'C1_oof_seeds.npy',oof);pd.DataFrame(results).to_csv(OUT/'training_log.csv',index=False)
        print('trained',fold,seed,best_epoch,flush=True)
assert np.isfinite(oof).all()
np.save(OUT/'C1_oof_ensemble.npy',oof.mean(0))
q=oof.mean(0);pair=labels<2;cond=q[pair,0]/q[pair,:2].sum(1)
metrics={'auc':float(roc_auc_score(labels[pair]==0,cond)),'overall':float(r2_score(y,q)),
 'clinical3':float(r2_score(y[:,:3],q[:,:3])),'per_class_r2':r2_score(y,q,multioutput='raw_values').tolist(),
 'per_fold_auc':{str(f):float(roc_auc_score(labels[ix]==0,q[ix,0]/q[ix,:2].sum(1))) for f in range(3) for ix in [folds[(folds.fold==f)&(folds.role=='outer')].row.to_numpy()] for ix in [ix[labels[ix]<2]]}}
(OUT/'C1_metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics,flush=True)
