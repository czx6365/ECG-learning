"""E32-A using frozen fold-specific representations and inner-only thresholds."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, log_loss, balanced_accuracy_score
import run_e29_s as s

b,e,p=s.b,s.e,s.p
OUT=e.ROOT/"results/e32_omi_nomi_probe"
SOURCE=s.OUT
DEVICE=torch.device("cuda")

def binary(y,q,t):
    fpr,tpr,threshold=roc_curve(y,q)
    return dict(auc=float(roc_auc_score(y,q)),pr_auc=float(average_precision_score(y,q)),
        balanced_accuracy=float(balanced_accuracy_score(y,q>=t)),
        omi_sensitivity=float((q[y==1]>=t).mean()),nomi_specificity=float((q[y==0]<t).mean()),
        brier=float(np.square(q-y).mean()),log_loss=float(log_loss(y,np.c_[1-q,q])),
        threshold=float(t),roc={"fpr":fpr.tolist(),"tpr":tpr.tolist(),"threshold":threshold.tolist()})

def choose(y,q):
    fpr,tpr,threshold=roc_curve(y,q)
    idx=np.flatnonzero(fpr<=.1)
    return float(threshold[idx[np.argmax(tpr[idx])]])

@torch.no_grad()
def representations(data,fold):
    pooled=data["z"][:,:4608].reshape(-1,3,1536)
    tokens=data["z"][:,4608:].reshape(-1,256,769)
    weights=[]
    values=[]
    hashes=[]
    for seed in range(3):
        path=SOURCE/f"fold{fold}/downstream/shared/seed{seed}/checkpoint-best.pth"
        head=p.Head("shared",seed).to(DEVICE).eval()
        head.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=False)["model"])
        hashes.append(e.sha256(path))
        mix_path=SOURCE/f"fold{fold}/downstream/learned_mix/seed{seed}/checkpoint-best.pth"
        mix=p.Head("learned_mix",seed).to(DEVICE).eval()
        mix.load_state_dict(torch.load(mix_path,map_location=DEVICE,weights_only=False)["model"])
        hashes.append(e.sha256(mix_path))
        weights.append(mix.adapt.softmax(0))
        chunks=[]
        for i in range(0,len(tokens),64):
            h=head.projection(tokens[i:i+64,:,:768])
            valid=tokens[i:i+64,:,768].bool()
            a=(h@head.query.T).masked_fill(~valid[...,None],-1e9).softmax(1)*valid[...,None]
            chunks.append((a.transpose(1,2)@h).squeeze(1).cpu().numpy())
        values.append(np.concatenate(chunks))
    features={"A1_layer9":pooled[:,1].cpu().numpy(),
        "A2_multilayer":(pooled*torch.stack(weights).mean(0)[None,:,None]).sum(1).cpu().numpy(),
        "A3_shared":np.concatenate(values,1)}
    return features,hashes

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    meta=pd.read_csv(e.DOWNSTREAM/"train_metadata.csv")
    y4=np.load(e.DOWNSTREAM/"train_labels.npy")
    pair=y4[:,:2].sum(1)==1
    label=y4[:,0].astype(int)
    folds=pd.read_csv(SOURCE/"fold_assignment.csv")
    assert sorted(folds[folds.role=="outer"].row)==list(range(len(meta)))
    b.dump(OUT/"model_config.json",dict(C=1.0,penalty="L2",solver="lbfgs",max_iter=3000,threshold="inner select specificity>=0.9",
        folds_sha256=e.sha256(SOURCE/"fold_assignment.csv"),A3="concatenate three 32-dimensional seed-specific shared summaries",
        A2="mean of three fold-specific convex-mixing weights; same frozen hidden coordinate basis",
        test_loaded=False,scope="A-stage probes, no residual fitting yet"))
    b.dump(OUT/"environment.json",dict(torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    load=lambda name:torch.tensor(np.load(e.E10_RUN/f"features/train_{name}.npy"),device=DEVICE)
    regions,mask=e.e11.masked_region_means(load("lead_ze_mean_max"),load("lead_present").bool())
    data={"train":{"z":torch.tensor(np.load(p.source/"features/train.npy"),device=DEVICE),
        "g":load("global_ze_mean_max"),"regions":regions,"mask":mask,"y":torch.tensor(y4,device=DEVICE)}}
    # Extract only Train, using the same tested layered routine.
    from run_e29_s import p as module
    original_extract=module.extract
    # Its function iterates train/val; a dedicated copy of the same extraction is supplied.
    from e32_train_cache import extract
    extract(data)
    data=data["train"]
    baseline=np.load(SOURCE/"oof_shared.npy").mean(0)
    old=s.metrics(y4,baseline)
    reference=json.loads((SOURCE/"cv_summary.json").read_text())["shared"]["ensemble"]
    assert abs(old["overall"]-reference["overall"])<1e-10
    b.dump(OUT/"baseline_reproduction.json",old)
    scores={m:np.full(len(label),np.nan) for m in ("A0","A1_layer9","A2_multilayer","A3_shared")}
    thresholds={m:np.full(len(label),np.nan) for m in scores}
    manifest=[]
    rows=[]
    for fold in range(3):
        b.status(OUT,f"fold{fold}_features")
        roles={role:f.row.to_numpy() for role,f in folds[folds.fold==fold].groupby("role")}
        assert all(not set(meta.iloc[roles[x]].entity_id)&set(meta.iloc[roles[z]].entity_id) for x,z in (("fit","select"),("fit","outer"),("select","outer")))
        feature,hashes=representations(data,fold)
        fit=roles["fit"][pair[roles["fit"]]]
        select=roles["select"][pair[roles["select"]]]
        held=roles["outer"]
        # Baseline operating threshold from this fold's held-out inner-selection predictions.
        qsel=np.mean([np.load(SOURCE/f"fold{fold}/downstream/shared/seed{i}/val_probabilities.npy") for i in range(3)],0)
        assert len(qsel)==len(roles["select"])
        q=qsel[: ,0]/qsel[:,:2].sum(1)
        selmask=pair[roles["select"]]
        t=choose(label[select],q[selmask])
        scores["A0"][held]=baseline[held,0]/baseline[held,:2].sum(1)
        thresholds["A0"][held]=t
        for name,x in feature.items():
            scaler=StandardScaler().fit(x[fit])
            model=LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs").fit(scaler.transform(x[fit]),label[fit])
            assert model.n_iter_.max()<3000
            q=model.predict_proba(scaler.transform(x[select]))[:,1]
            t=choose(label[select],q)
            q=model.predict_proba(scaler.transform(x[held]))[:,1]
            scores[name][held]=q
            thresholds[name][held]=t
            folder=OUT/f"fold{fold}"
            folder.mkdir(exist_ok=True)
            np.savez(folder/f"{name}.npz",mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=model.intercept_)
            valid=pair[held]
            rows.append(dict(fold=fold,method=name,**{k:v for k,v in binary(label[held][valid],q[valid],t).items() if k!="roc"}))
        manifest.append(dict(fold=fold,fit=fit.tolist(),select=select.tolist(),outer=held.tolist(),checkpoints=hashes))
    b.dump(OUT/"fold_manifest.json",manifest)
    pd.DataFrame(rows).to_csv(OUT/"representation_probe_results.csv",index=False)
    summary={}
    for name,q in scores.items():
        assert np.isfinite(q).all()
        pred=(q>=thresholds[name])
        m=binary(label[pair],q[pair],.5)
        m["inner_threshold_omi_sensitivity"]=float(pred[pair][label[pair]==1].mean())
        m["inner_threshold_nomi_specificity"]=float((~pred[pair][label[pair]==0]).mean())
        m["threshold_note"]="threshold varies by outer fold, all set on inner select"
        m["correlation_with_A0"]=float(np.corrcoef(q[pair],scores["A0"][pair])[0,1])
        np.save(OUT/f"{name}_probability.npy",q)
        np.save(OUT/f"{name}_threshold.npy",thresholds[name])
        summary[name]=m
    groups=list(meta[pair].reset_index().groupby("entity_id").indices.values())
    rng=np.random.default_rng(32001)
    draws=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) for _ in range(2000)]
    boots={}
    y=label[pair]
    base=scores["A0"][pair]
    for name,q in scores.items():
        q=q[pair]
        delta=roc_auc_score(y,q)-roc_auc_score(y,base)
        boots[name]=dict(auc_delta=float(delta),ci95=np.quantile([roc_auc_score(y[i],q[i])-roc_auc_score(y[i],base[i]) for i in draws],[.025,.975]).tolist())
    b.dump(OUT/"paired_entity_bootstrap.json",boots)
    b.dump(OUT/"clinical_subgroup_metrics.json",summary)
    records=[]
    for name,q in scores.items():
        for i in range(len(q)):
            records.append(dict(ecg_id=f"T{i:05d}",entity_id=str(meta.iloc[i].entity_id),row=i,outer_fold=int(folds[(folds.role=="outer")&(folds.row==i)].fold.iloc[0]),
                seed="three-frozen-seed-features" if name!="A1_layer9" else "fixed",model=name,label=int(y4[i].argmax()),probability=float(q[i]),threshold=float(thresholds[name][i])))
    pd.DataFrame(records).to_csv(OUT/"predictions.csv",index=False)
    errors=[]
    for name,q in scores.items():
        for i in np.flatnonzero(pair):
            errors.append(dict(ecg_id=f"T{i:05d}",method=name,truth=int(label[i]),baseline_correct=bool((scores["A0"][i]>=.5)==label[i]),probe_correct=bool((q[i]>=.5)==label[i])))
    pd.DataFrame(errors).to_csv(OUT/"error_transition_analysis.csv",index=False)
    b.dump(OUT/"prediction_manifest.json",dict(file="predictions.csv",label_map={0:"OMI",1:"NOMI",2:"Clinical Control",3:"Healthy"},
        score="P(OMI | OMI or NOMI)",meta_sha256=e.sha256(e.DOWNSTREAM/"train_metadata.csv")))
    b.dump(OUT/"inner_crossfit_audit.json",dict(A="No stacking head trained. Features from corresponding outer-fold models; probe fit and operating thresholds use disjoint entities.",
        B="not executed in A stage; inner cross-fitting required before any residual training",
        overlap="E12-A pretraining includes these ECGs as unlabeled inputs"))
    b.status(OUT,"A_complete")

if __name__=="__main__":
    main()
