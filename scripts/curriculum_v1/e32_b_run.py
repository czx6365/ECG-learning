"""E32-B: strict nested cross-fit residual study."""
import json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import run_e29_s as s
import run_e32 as a

b,e,p=s.b,s.e,s.p
SRC=s.OUT
OUT=e.ROOT/'results/e32_omi_nomi_residual'
DEV=torch.device('cuda')

def save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def restrict(data,rows): return s.subset(data,rows)

@torch.no_grad()
def features(model,data):
    model.eval(); out=[]
    for i in range(0,len(data['y']),64):
        x=data['z'][i:i+64,4608:].reshape(-1,256,769)
        v=x[...,768].bool(); h=model.projection(x[...,:768])
        att=(h@model.query.T).masked_fill(~v[...,None],-1e9).softmax(1)*v[...,None]
        out.append((att.transpose(1,2)@h).squeeze(1).cpu().numpy())
    return np.concatenate(out)

def predict(models,data):
    prob=[]; z=[]
    for m in models:
        prob.append(b.evaluate(m,data)[1]); z.append(features(m,data))
    return np.mean(prob,axis=0),np.concatenate(z,axis=1)

def load_outer(fold):
    models=[]; hashes=[]
    for seed in range(3):
        path=SRC/f'fold{fold}/downstream/shared/seed{seed}/checkpoint-best.pth'
        hashes.append(e.sha256(path)); m=s.PairedHead('shared',seed).to(DEV)
        m.load_state_dict(torch.load(path,map_location=DEV,weights_only=False)['model']);models.append(m.eval())
    return models,hashes

def residual(prob,z,y,beta=.02):
    r=np.log(prob[:,0]/prob[:,1]); sc=StandardScaler().fit(z)
    x=sc.transform(z); pair=y<2; target=(y[pair]==0).astype(float)
    def loss(w):
        d=x[pair]@w[:-1]+w[-1]; u=r[pair]+d; q=expit(u)
        val=np.logaddexp(0,u).mean()-(target*u).mean()+beta*np.square(d).mean()
        grad=x[pair].T@(q-target+2*beta*d)/len(d)
        gb=(q-target+2*beta*d).mean()
        return val,np.r_[grad,gb]
    fit=minimize(loss,np.zeros(z.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':500})
    assert fit.success,fit.message
    return sc,fit.x

def calibration(prob,y):
    r=np.log(prob[:,0]/prob[:,1]); pair=y<2; target=(y[pair]==0).astype(float)
    def loss(v):
        u=np.exp(v[0])*r[pair]+v[1]
        return np.logaddexp(0,u).mean()-(target*u).mean()
    fit=minimize(loss,[0.,0.],method='BFGS');return fit.x

def apply(prob,score):
    out=prob.copy(); mass=prob[:,:2].sum(1); q=expit(score)
    out[:,0]=mass*q;out[:,1]=mass*(1-q)
    np.testing.assert_allclose(out[:,:2].sum(1),mass,atol=2e-7)
    np.testing.assert_array_equal(out[:,2:],prob[:,2:])
    np.testing.assert_allclose(out.sum(1),1,atol=2e-6)
    return out

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(e.DOWNSTREAM/'train_metadata.csv');y4=np.load(e.DOWNSTREAM/'train_labels.npy');y=y4.argmax(1)
    folds=pd.read_csv(SRC/'fold_assignment.csv'); base=np.load(SRC/'oof_shared.npy').mean(0)
    ref=json.loads((SRC/'cv_summary.json').read_text())['shared']['ensemble']; calc=s.metrics(y4,base)
    for key in ('overall','clinical3','omi_nomi_auc'): assert abs(calc[key]-ref[key])<1e-9
    save('baseline_parity.json',{k:calc[k] for k in ('overall','clinical3','omi_nomi_auc')})
    load=lambda name:torch.tensor(np.load(e.E10_RUN/f'features/train_{name}.npy'),device=DEV)
    regions,mask=e.e11.masked_region_means(load('lead_ze_mean_max'),load('lead_present').bool())
    data={'train':{'z':torch.tensor(np.load(p.source/'features/train.npy'),device=DEV),'g':load('global_ze_mean_max'),'regions':regions,'mask':mask,'y':torch.tensor(y4,device=DEV)}}
    from e32_train_cache import extract
    extract(data);data=data['train'];b.Head=s.PairedHead
    oof={'B0':base,'B1':np.full_like(base,np.nan),'B2':np.full_like(base,np.nan)}
    audits=[];manifest=[];configs=[]
    for fold in range(3):
        held=folds[(folds.fold==fold)&(folds.role=='outer')].row.to_numpy()
        dev=np.sort(folds[(folds.fold==fold)&(folds.role!='outer')].row.to_numpy())
        assert not set(meta.iloc[held].entity_id)&set(meta.iloc[dev].entity_id)
        split=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=32050+fold)
        pp=np.full((len(dev),4),np.nan);zz=np.full((len(dev),96),np.nan)
        for inner,(train_ix,valid_ix) in enumerate(split.split(dev,y[dev],meta.iloc[dev].entity_id)):
            train=dev[train_ix];valid=dev[valid_ix]
            selectsplit=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=33050+fold*10+inner)
            fit_ix,sel_ix=next(selectsplit.split(train,y[train],meta.iloc[train].entity_id))
            fit,sel=train[fit_ix],train[sel_ix]
            assert not set(meta.iloc[valid].entity_id)&set(meta.iloc[train].entity_id)
            folder=OUT/f'fold{fold}/inner{inner}';models=[]
            for seed in range(3):
                b.train(folder,{'train':restrict(data,fit),'val':restrict(data,sel)},DEV,'shared',seed)
                path=folder/f'downstream/shared/seed{seed}/checkpoint-best.pth'
                m=s.PairedHead('shared',seed).to(DEV);m.load_state_dict(torch.load(path,map_location=DEV,weights_only=False)['model']);models.append(m.eval())
            pp[valid_ix],zz[valid_ix]=predict(models,restrict(data,valid))
            audits.append(dict(outer_fold=fold,inner_fold=inner,fit=len(fit),select=len(sel),held=len(valid),fit_entities=meta.iloc[fit].entity_id.nunique(),held_entities=meta.iloc[valid].entity_id.nunique(),checkpoint_paths=[str(folder/f'downstream/shared/seed{k}/checkpoint-best.pth') for k in range(3)]))
            save('inner_crossfit_audit.json',audits)
        assert np.isfinite(pp).all() and np.isfinite(zz).all()
        t=calibration(pp,y[dev]);sc,w=residual(pp,zz,y[dev]);models,hashes=load_outer(fold)
        outer_p,outer_z=predict(models,restrict(data,held))
        np.testing.assert_allclose(outer_p,base[held],rtol=2e-5,atol=3e-6)
        r=np.log(outer_p[:,0]/outer_p[:,1]);oof['B1'][held]=apply(outer_p,np.exp(t[0])*r+t[1])
        delta=sc.transform(outer_z)@w[:-1]+w[-1];oof['B2'][held]=apply(outer_p,r+delta)
        np.savez(OUT/f'fold{fold}_model.npz',t=t,scaler_mean=sc.mean_,scaler_scale=sc.scale_,weights=w)
        np.savez(OUT/f'fold{fold}_held.npz',rows=held,prob=outer_p,z=outer_z,delta=delta)
        manifest.append(dict(fold=fold,development=len(dev),held=len(held),outer_checkpoint_sha256=hashes))
        configs.append(dict(fold=fold,temperature=float(np.exp(t[0])),bias=float(t[1]),delta_abs_p90=float(np.quantile(np.abs(delta),.9))))
        save('fold_manifest.json',manifest);save('model_config.json',configs)
    for name,v in oof.items():
        assert np.isfinite(v).all();np.save(OUT/f'{name}_oof.npy',v)
    save('oof_metrics.json',{name:s.metrics(y4,v) for name,v in oof.items()})
    save('prediction_manifest.json',dict(label_map=['OMI','NOMI','Clinical Control','Healthy'],folds='fold_manifest.json',models=list(oof),test_loaded=False))
    (OUT/'status.txt').write_text('training_complete')

if __name__=='__main__':main()
