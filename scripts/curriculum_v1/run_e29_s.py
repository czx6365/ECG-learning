"""Fixed Shared Query comparison using nested entity-grouped Train CV."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from sklearn.model_selection import StratifiedGroupKFold

import run_e29_r as r

b, e, p = r.b, r.e, r.p
OUT = e.ROOT / "results/e29_shared_generalization"
DEVICE = torch.device("cuda")
METHODS = ("learned_mix", "shared")

class PairedHead(p.Head):
    def __init__(self, method, seed):
        state = None
        if method == "shared":
            super().__init__("learned_mix",seed)
            state = torch.get_rng_state()
            self.variant = method
            self.projection = torch.nn.Linear(768,32,bias=False)
            self.query = torch.nn.Parameter(torch.empty(1,32))
            torch.nn.init.normal_(self.query,std=.1)
            self.readout = torch.nn.Parameter(torch.empty(4,32))
            torch.nn.init.normal_(self.readout,std=.1)
            self.gamma = torch.nn.Parameter(torch.zeros(()))
            torch.set_rng_state(state)
        else:
            super().__init__(method,seed)

def metrics(y,q):
    d=r.metrics(y,q)
    pt=y[:,3]==0
    target=y.argmax(1)
    d.pop("patient_four_output_r2",None)
    d["patient_raw3_r2"]=float(r2_score(y[pt,:3],q[pt,:3]))
    d["patient_conditional3_r2"]=float(r2_score(y[pt,:3],q[pt,:3]/q[pt,:3].sum(1,keepdims=True)))
    idx=target<2
    cond=q[idx,0]/q[idx,:2].sum(1)
    fpr,tpr,_=roc_curve(target[idx]==0,cond)
    d["omi_sensitivity_at_90pct_specificity"]=float(tpr[fpr<=.1].max())
    d["nomi_sensitivity_at_90pct_omi_sensitivity"]=float((1-fpr[tpr>=.9]).max())
    pred=q.argmax(1)
    d.update(omi_to_nomi=int(((target==0)&(pred==1)).sum()),nomi_to_omi=int(((target==1)&(pred==0)).sum()),
             control_errors=int(((target==2)&(pred!=2)).sum()),patient_to_healthy=int((pt&(pred==3)).sum()),
             confident_errors=int(((pred!=target)&(q.max(1)>=.9)).sum()))
    return d

KEYS=("overall","clinical3","OMI","NOMI","Control","Healthy","omi_nomi_auc","macro_auc","patient_raw3_r2","patient_conditional3_r2")
def core(y,q):
    rs=r2_score(y,q,multioutput="raw_values")
    pair=y[:,:2].sum(1)==1
    pt=y[:,3]==0
    return np.r_[rs.mean(),rs[:3].mean(),rs,roc_auc_score(y[pair,0],q[pair,0]/q[pair,:2].sum(1)),
        roc_auc_score(y,q,average="macro"),r2_score(y[pt,:3],q[pt,:3]),r2_score(y[pt,:3],q[pt,:3]/q[pt,:3].sum(1,keepdims=True))]

def contrast(y,a,c,groups):
    rng=np.random.default_rng(29051)
    indices=list(pd.Series(groups).groupby(np.asarray(groups)).indices.values())
    draws=[np.concatenate([indices[j] for j in rng.integers(len(indices),size=len(indices))]) for _ in range(2000)]
    delta=core(y,a)-core(y,c)
    ci=np.quantile([core(y[i],a[i])-core(y[i],c[i]) for i in draws],[.025,.975],axis=0).T
    return {k:dict(delta=float(v),ci95=z.tolist()) for k,v,z in zip(KEYS,delta,ci)}

def summarize(y,a):
    mean=a.mean(0)
    individual=[metrics(y,q) for q in a]
    ens=metrics(y,mean)
    residual=a-y
    def corr(x):
        return np.corrcoef(x).tolist()
    return dict(per_seed=individual,mean={k:float(np.mean([v[k] for v in individual])) for k in ens},
        sd={k:float(np.std([v[k] for v in individual],ddof=1)) for k in ens},ensemble=ens,
        per_output_ensemble_gain={k:ens[k]-float(np.mean([v[k] for v in individual])) for k in (*r.names,"overall","clinical3")},
        probability_correlations=corr(a.reshape(len(a),-1)),
        residual_correlations={k:corr(residual[:,:,j]) for j,k in enumerate(r.names)},
        mean_probability_variance=np.var(a,axis=0).mean(0).tolist(),
        brier_cancellation=float(np.square(residual).sum(2).mean()-ens["brier"]))

def historical(data):
    meta,y,old,_=r.audit_predictions(data)
    q={"learned_mix":list(old["e25"]),"shared":list(old["shared"])}
    for method in METHODS:
        for seed in (3,4,5):
            folder=r.out/"downstream"/method/f"seed{seed}"
            pred=np.load(folder/"val_probabilities.npy")
            model=p.Head(method,seed).to(DEVICE)
            model.load_state_dict(torch.load(folder/"checkpoint-best.pth",map_location=DEVICE,weights_only=False)["model"])
            np.testing.assert_allclose(b.evaluate(model,data["val"])[1],pred,rtol=1e-5,atol=3e-6)
            np.testing.assert_allclose(pred,r.softmax(np.load(folder/"val_logits.npy"),axis=1),atol=1e-7)
            assert np.isfinite(pred).all() and (pred>=0).all()
            np.testing.assert_allclose(pred.sum(1),1,atol=2e-6)
            q[method].append(pred)
    for m in METHODS: np.save(OUT/f"val_{m}_six_seeds.npy",q[m])
    result={}
    for name,sl in (("012",slice(0,3)),("345",slice(3,6)),("all",slice(None))):
        a,c=[np.asarray(q[m][sl]) for m in METHODS]
        result[name]={"e25":summarize(y,a),"shared":summarize(y,c),
                      "bootstrap":contrast(y,c.mean(0),a.mean(0),meta.entity_id.to_numpy())}
    b.dump(OUT/"historical_ensembles.json",result)
    return meta,y

def subset(data,indices):
    ids=torch.as_tensor(indices,device=DEVICE)
    return {k:v[ids] for k,v in data.items()}

def cv(data,train):
    y=data["y"].cpu().numpy()
    group=train.entity_id.to_numpy()
    label=y.argmax(1)
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=29051)
    folds=list(splitter.split(y,label,group))
    split_rows=[]
    for f,(dev,held) in enumerate(folds):
        inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=29060+f)
        fit,select=next(inner.split(y[dev],label[dev],group[dev]))
        fit,select=dev[fit],dev[select]
        assert not(set(group[fit])&set(group[select]) or set(group[held])&set(group[dev]))
        for role,ids in (("fit",fit),("select",select),("outer",held)):
            split_rows.extend(dict(row=int(i),fold=f,role=role,label=int(label[i])) for i in ids)
    split=pd.DataFrame(split_rows)
    split.to_csv(OUT/"fold_assignment.csv",index=False)
    pre=pd.read_csv(e.PRETRAIN_INDEX/"train_metadata.csv",low_memory=False)
    patient=set(pre.patient_sn.dropna().astype(str)) if "patient_sn" in pre else set()
    stems=set(pre.csv_name.astype(str).map(lambda s:Path(s).stem))
    b.dump(OUT/"pretraining_overlap.json",{
        "pretraining_train_records":len(pre),
        "matched_ecg_rows":int(train.csv_name.astype(str).map(lambda s:Path(s).stem).isin(stems).sum()),
        "matched_patient_ids":len(patient&set(train.patient_sn.dropna().astype(str))),
        "per_fold":{str(f):{"ecg":int(train.iloc[held].csv_name.astype(str).map(lambda s:Path(s).stem).isin(stems).sum()),"patient_ids":len(patient&set(train.iloc[held].patient_sn.dropna().astype(str)))} for f,(_,held) in enumerate(folds)},
        "objective":"E12-A Masked SID, no downstream class target; overlap limits independence"})
    preds={m:np.full((3,len(y),4),np.nan,np.float32) for m in METHODS}
    b.Head=PairedHead
    rows=[]
    for f,(_,held) in enumerate(folds):
        fit=split[(split.fold==f)&(split.role=="fit")].row.to_numpy()
        select=split[(split.fold==f)&(split.role=="select")].row.to_numpy()
        d={"train":subset(data,fit),"val":subset(data,select)}
        outer=subset(data,held)
        for m in METHODS:
            for seed in (0,1,2):
                b.status(OUT,f"fold{f}_{m}_{seed}")
                result=b.train(OUT/f"fold{f}",d,DEVICE,m,seed)
                model=PairedHead(m,seed).to(DEVICE)
                ck=OUT/f"fold{f}/downstream/{m}/seed{seed}/checkpoint-best.pth"
                model.load_state_dict(torch.load(ck,map_location=DEVICE,weights_only=False)["model"])
                _,q,_=b.evaluate(model,outer)
                preds[m][seed,held]=q
                rows.append(dict(fold=f,method=m,seed=seed,best_epoch=result["best_epoch"],**metrics(y[held],q)))
        del d,outer
    pd.DataFrame(rows).to_csv(OUT/"fold_seed_metrics.csv",index=False)
    for m in METHODS:
        assert np.isfinite(preds[m]).all()
        np.save(OUT/f"oof_{m}.npy",preds[m])
    result={m:summarize(y,a) for m,a in preds.items()}
    result["ensemble_bootstrap"]=contrast(y,preds["shared"].mean(0),preds["learned_mix"].mean(0),group)
    result["seed_bootstrap"]={s:contrast(y,preds["shared"][s],preds["learned_mix"][s],group) for s in range(3)}
    result["fold_ensemble"]={f:{m:metrics(y[idx],preds[m][:,idx].mean(0)) for m in METHODS} for f,(_,idx) in enumerate(folds)}
    b.dump(OUT/"cv_summary.json",result)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    r.out=OUT
    meta={s:pd.read_csv(e.DOWNSTREAM/f"{s}_metadata.csv") for s in ("train","val")}
    assert not set(meta["train"].entity_id)&set(meta["val"].entity_id)
    for s,frame in meta.items():
        assert np.array_equal(frame.label_id.to_numpy(),np.load(e.DOWNSTREAM/f"{s}_labels.npy").argmax(1))
    b.dump(OUT/"config.json",dict(outer_folds=3,inner_folds=5,inner_selection_fold=0,seeds=[0,1,2],epochs=30,
           test_loaded=False,selection="inner Overall R2",torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    b.status(OUT,"extract_frozen")
    data=b.data(p.source,DEVICE)
    p.extract(DEVICE,data)
    # E29-R new seeds live in the original audit directory.
    r.out=e.ROOT/"results/e29_shared_confirmation_audit"
    audit=r.audit_predictions
    def redirected(d):
        r.out=OUT
        try: return audit(d)
        finally: r.out=e.ROOT/"results/e29_shared_confirmation_audit"
    r.audit_predictions=redirected
    b.status(OUT,"historical_ensemble")
    historical(data)
    b.status(OUT,"attention")
    import e29_s_attention
    e29_s_attention.run(data["val"],OUT)
    del data["val"]
    cv(data["train"],meta["train"])
    b.status(OUT,"complete")

if __name__=="__main__":
    main()
