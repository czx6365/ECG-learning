"""Frozen attention diagnostics and uniform-weight inference ablation."""
import numpy as np
import pandas as pd
import torch

from run_e29_r import b,e,p
from run_e29_s import metrics,contrast

@torch.no_grad()
def run(data,out):
    y=data["y"].cpu().numpy()
    meta=pd.read_csv(e.DOWNSTREAM/"val_metadata.csv")
    ch=np.load(e.DOWNSTREAM/"val_data_in_chans.npy")
    coarse=np.load(e.DOWNSTREAM/"val_data_in_times.npy")
    true_path=e.ROOT/"results/e27_mask_diagnostics/cache/true_r_times.npy"
    times=np.load(true_path) if true_path.exists() else None
    rows,normal,uniform=[],[],[]
    for seed in range(6):
        root=p.run if seed<3 else e.ROOT/"results/e29_shared_confirmation_audit"
        model=p.Head("shared",seed).cuda().eval()
        ck=root/"downstream/shared"/f"seed{seed}"/"checkpoint-best.pth"
        model.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"])
        normal.append(b.evaluate(model,data)[1])
        for i in range(0,len(y),64):
            tokens=data["z"][i:i+64,4608:].reshape(-1,256,769)
            v=tokens[...,768].bool()
            s=(model.projection(tokens[...,:768]) @ model.query.T).squeeze(-1)
            weight=(s.masked_fill(~v,-1e9).softmax(1)*v).cpu().numpy()
            valid=v.cpu().numpy()
            for j,a in enumerate(weight):
                n=i+j
                count=int(valid[j].sum())
                entropy=-float(np.sum(a*np.log(np.maximum(a,1e-30))))
                rec=dict(seed=seed,row=n,valid_tokens=count,padding_fraction=1-count/256,
                         entropy=entropy,normalized_entropy=entropy/np.log(count) if count>1 else np.nan,
                         maximum_mass=float(a.max()),label=int(y[n].argmax()))
                if times is not None:
                    rr=np.diff(np.unique(times[n,valid[j]]))
                    rec["trimmed_rr_hr"]=6000/rr.mean() if len(rr) else np.nan
                    rec["possible_asymmetric_alignment_fraction"]=np.mean((rr[:-1]//2>48)&
                        (rr[:-1]//2+rr[1:]//2<=96)) if len(rr)>1 else np.nan
                rec.update({f"lead_{c}_mass":float(a[ch[n]==c].sum()) for c in (1,2,3,7,8,9,10,11,12)})
                rec.update({f"time_bin_{t}_mass":float(a[coarse[n]==t].sum()) for t in range(1,11)})
                rows.append(rec)
        model.query.zero_()
        uniform.append(b.evaluate(model,data)[1])
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"attention_diagnostics.csv",index=False)
    frame.groupby("label").mean(numeric_only=True).to_csv(out/"attention_by_class.csv")
    frame.select_dtypes("number").corr(method="spearman").to_csv(out/"attention_correlations.csv")
    a,c=np.mean(normal,0),np.mean(uniform,0)
    np.save(out/"val_shared_uniform_six_seeds.npy",uniform)
    b.dump(out/"uniform_ablation.json",dict(learned=metrics(y,a),uniform=metrics(y,c),
        learned_minus_uniform=contrast(y,a,c,meta.entity_id.to_numpy()),
        definition="zero query produces uniform effective weights over valid tokens; all other weights fixed",
        alignment="RR proxy from recovered detected peaks when present; not clinical R-detection error"))
