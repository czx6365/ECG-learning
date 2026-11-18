import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
import run_e25 as base

def run(folder, contrasts):
    y = np.load(base.e.DOWNSTREAM/"val_labels.npy")
    groups = list(pd.read_csv(base.e.DOWNSTREAM/"val_metadata.csv").groupby("entity_id").indices.values())
    rng = np.random.default_rng(25006)
    samples = [np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) for _ in range(2000)]
    def probabilities(m):
        return np.mean([np.load(folder/"downstream"/m/f"seed{s}"/"val_probabilities.npy") for s in range(3)],0)
    def score(p, idx):
        targets, predictions = y[idx], p[idx]
        pair = targets[:,:2].sum(1)==1
        auc = roc_auc_score(targets[pair,0],predictions[pair,0]-predictions[pair,1])
        r = r2_score(targets,predictions,multioutput="raw_values")
        return np.r_[r,r[:3].mean(),r.mean(),auc]
    output = {}
    for m,ref in contrasts:
        a,b = probabilities(m),probabilities(ref)
        point = score(a,np.arange(len(y)))-score(b,np.arange(len(y)))
        interval = np.quantile([score(a,i)-score(b,i) for i in samples],[.025,.975],axis=0).T
        output[m+"_minus_"+ref]={k:{"delta":float(v),"ci95":c.tolist()} for k,v,c in zip(
          ("OMI","NOMI","Clinical Control","Healthy","clinical3","overall","omi_nomi_probability_difference_auc"),point,interval)}
    base.dump(folder/"clinical_entity_contrasts.json",output)

if __name__ == "__main__":
    root=base.e.ROOT/"results"
    run(root/"e29_class_conditioned_pooling",[("shared","learned_mix"),("class","learned_mix"),("class","shared")])
    p=root/"e30_physiology_increment"
    if (p/"status.txt").exists() and (p/"status.txt").read_text().strip()=="complete":
        run(p,[("increment","learned_mix"),("increment","capacity")])
