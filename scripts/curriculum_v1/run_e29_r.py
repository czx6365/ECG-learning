import hashlib
import json
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score, r2_score
from scipy.special import softmax

import run_e29_pooling as p

b, e = p.base, p.e
root = e.ROOT / "results"
out = root / "e29_shared_confirmation_audit"
sources = {"e25":(p.source,"learned_mix"), "shared":(p.run,"shared"), "class":(p.run,"class")}
names = ["OMI","NOMI","Control","Healthy"]

def metrics(y, q):
    r = r2_score(y,q,multioutput="raw_values")
    target, pred = y.argmax(1), q.argmax(1)
    pair = target < 2
    conditional = q[pair,0]/q[pair,:2].sum(1)
    positive = target[pair]==0
    confidence = q.max(1)
    bins = np.minimum((confidence*15).astype(int),14)
    ece = sum(m.mean()*abs((pred[m]==target[m]).mean()-confidence[m].mean())
              for i in range(15) if (m:=bins==i).any())
    from sklearn.metrics import f1_score
    return dict(zip(names,r.tolist())) | dict(overall=float(r.mean()),clinical3=float(r[:3].mean()),
        patient_four_output_r2=float(r2_score(y[target!=3],q[target!=3])),
        patient_accuracy=float((pred[target!=3]==target[target!=3]).mean()),
        omi_nomi_auc=float(roc_auc_score(positive,conditional)),
        omi_nomi_margin_auc=float(roc_auc_score(positive,q[pair,0]-q[pair,1])),
        omi_pair_sensitivity=float((conditional[positive]>=.5).mean()),
        omi_pair_specificity=float((conditional[~positive]<.5).mean()),
        omi_four_class_sensitivity=float((pred[target==0]==0).mean()),
        omi_four_class_specificity=float((pred[target!=0]!=0).mean()),
        healthy_auc=float(roc_auc_score(y[:,3],q[:,3])),
        healthy_sensitivity=float((pred[target==3]==3).mean()),
        healthy_specificity=float((pred[target!=3]!=3).mean()),
        macro_auc=float(roc_auc_score(y,q,average="macro")),
        macro_f1=float(f1_score(target,pred,average="macro")),
        brier=float(np.square(q-y).sum(1).mean()),log_loss=float(log_loss(y,q)),ece15=float(ece))

def audit_predictions(data):
    meta=pd.read_csv(e.DOWNSTREAM/"val_metadata.csv")
    train=pd.read_csv(e.DOWNSTREAM/"train_metadata.csv")
    assert not set(meta.entity_id)&set(train.entity_id)
    y=data["val"]["y"].cpu().numpy()
    assert len(meta)==len(y) and np.array_equal(meta.label_id.to_numpy(),y.argmax(1))
    arrays, rows, alignment = {}, [], {}
    for name,(folder,method) in sources.items():
        arrays[name]=[]
        for seed in range(3):
            path=folder/"downstream"/method/f"seed{seed}"
            q=np.load(path/"val_probabilities.npy")
            raw=np.load(path/"val_logits.npy")
            assert q.shape==y.shape and np.isfinite(q).all()
            np.testing.assert_allclose(q,softmax(raw,axis=1),atol=1e-7)
            model=p.Head(method,seed).cuda()
            checkpoint=torch.load(path/"checkpoint-best.pth",map_location="cuda",weights_only=False)
            model.load_state_dict(checkpoint["model"])
            _, replay, _=b.evaluate(model,data["val"])
            np.testing.assert_allclose(replay,q,atol=3e-6,rtol=1e-5)
            alignment[f"{name}/{seed}"]=dict(max_replay_error=float(np.max(abs(q-replay))),
                probabilities_sha256=e.sha256(path/"val_probabilities.npy"),checkpoint_sha256=e.sha256(path/"checkpoint-best.pth"))
            arrays[name].append(q)
            rows.append(dict(method=name,seed=seed,**metrics(y,q)))
    b.dump(out/"prediction_alignment_audit.json",dict(records=len(y),entities=meta.entity_id.nunique(),
       checkpoints=alignment,ordered_metadata_sha256=e.sha256(e.DOWNSTREAM/"val_metadata.csv"),
       labels_sha256=e.sha256(e.DOWNSTREAM/"val_labels.npy"),aggregation="equal-weight ECG rows, entity clusters for bootstrap",test_loaded=False,
       ids="Historical NPY files have no embedded IDs. Row alignment verified by replay against ordered inputs and metadata."))
    return meta,y,arrays,rows

def report(meta,y,arrays,rows):
    ens={m:np.mean(a,axis=0) for m,a in arrays.items()}
    scores={m:metrics(y,q) for m,q in ens.items()}
    b.dump(out/"ensemble_metrics.json",scores)
    b.dump(out/"clinical_subset_metrics.json",dict(ensemble=scores,per_seed=rows,
        definitions="Conditional pair probability pOMI/(pOMI+pNOMI), threshold 0.5; patient subset R² retains all four outputs including constant Healthy; historical margin AUC uses pOMI-pNOMI."))
    diversity={}
    for m,a in arrays.items():
        residual=np.asarray(a)-y
        per_class=np.var(a,axis=0).mean(0)
        mean_r2=float(np.mean([metrics(y,q)["overall"] for q in a]))
        diversity[m]=dict(seed_mean_r2=mean_r2,ensemble_r2=scores[m]["overall"],ensemble_gain=scores[m]["overall"]-mean_r2,
            residual_correlations=[float(np.corrcoef(residual[i].ravel(),residual[j].ravel())[0,1]) for i,j in combinations(range(len(a)),2)],
            probability_variance=per_class.tolist(),brier_cancellation=float(np.square(residual).sum(2).mean()-scores[m]["brier"]))
    b.dump(out/"ensemble_diversity.json",diversity)
    groups=list(meta.groupby("entity_id").indices.values())
    rng=np.random.default_rng(25006)
    draws=[np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]) for _ in range(2000)]
    keys=names+["clinical3","overall","omi_nomi_auc","macro_auc"]
    def core(y,q):
        r=r2_score(y,q,multioutput="raw_values")
        pair=y[:,:2].sum(1)==1
        return np.r_[r,r[:3].mean(),r.mean(),roc_auc_score(y[pair,0],q[pair,0]/q[pair,:2].sum(1)),roc_auc_score(y,q,average="macro")]
    boot={}
    for method in ("shared","class"):
        cases=[("ensemble",ens[method],ens["e25"])]+[(str(s),arrays[method][s],arrays["e25"][s]) for s in range(3)]
        for seed,a,c in cases:
            values=core(y,a)-core(y,c)
            ci=np.quantile([core(y[i],a[i])-core(y[i],c[i]) for i in draws],[.025,.975],axis=0).T
            boot[f"{method}/{seed}"]={k:dict(delta=float(v),ci95=z.tolist()) for k,v,z in zip(keys,values,ci)}
    b.dump(out/"paired_bootstrap.json",boot)
    pd.DataFrame(rows).to_csv(out/"per_seed_metrics.csv",index=False)
    pd.DataFrame([dict(comparison=k,metric=m,delta=v["delta"],low=v["ci95"][0],high=v["ci95"][1])
        for k,a in boot.items() for m,v in a.items()]).to_csv(out/"per_class_deltas.csv",index=False)
    t=y.argmax(1)
    x,z=ens["e25"].argmax(1),ens["shared"].argmax(1)
    pd.DataFrame(dict(anonymous_ecg=[f"V{i:05d}" for i in range(len(y))],truth=t,e25=x,shared=z,
        e25_high_conf_wrong=(x!=t)&(ens["e25"].max(1)>=.9),shared_high_conf_wrong=(z!=t)&(ens["shared"].max(1)>=.9),
        transition=np.select([(x==t)&(z==t),(x==t)&(z!=t),(x!=t)&(z==t)],["both_correct","regressed","corrected"],default="both_wrong")
    )).to_csv(out/"error_transition_analysis.csv",index=False)

def main():
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    b.status(out,"extract_frozen")
    data=b.data(p.source,torch.device("cuda"))
    p.extract(torch.device("cuda"),data)
    b.status(out,"audit_historical")
    meta,y,arrays,rows=audit_predictions(data)
    report(meta,y,arrays,rows)
    b.status(out,"paired_seeds")
    class PairedHead(p.Head):
        def __init__(self,method,seed):
            super().__init__("learned_mix",seed)
            rng=torch.get_rng_state()
            if method=="shared":
                self.variant=method
                self.projection=torch.nn.Linear(768,32,bias=False)
                self.query=torch.nn.Parameter(torch.empty(1,32))
                torch.nn.init.normal_(self.query,std=.1)
                self.readout=torch.nn.Parameter(torch.empty(4,32))
                torch.nn.init.normal_(self.readout,std=.1)
                self.gamma=torch.nn.Parameter(torch.zeros(()))
                torch.set_rng_state(rng)
    b.Head=PairedHead
    new=[]
    for seed in (3,4,5):
        for method in ("learned_mix","shared"):
            b.status(out,f"{method}_{seed}")
            r=b.train(out,data,torch.device("cuda"),method,seed)
            new.append(dict(method="e25" if method=="learned_mix" else method,seed=seed,
                **metrics(y,np.load(out/"downstream"/method/f"seed{seed}"/"val_probabilities.npy"))))
    pd.DataFrame(rows+new).to_csv(out/"per_seed_metrics.csv",index=False)
    b.status(out,"complete")

if __name__=="__main__":
    main()
