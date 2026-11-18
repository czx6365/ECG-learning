"""Fixed all-layer fusion comparison under saved patient-group CV."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
import run_e29_s as s

b,e,p=s.b,s.e,s.p
OUT=e.ROOT/"results/e29_all_layer_fusion"
DEVICE=torch.device("cuda")

class Head(s.PairedHead):
    def __init__(self,method,seed):
        super().__init__("shared",seed)
        self.fusion=method
        if method=="C":
            self.adapt=nn.Parameter(torch.zeros(12))
        elif method=="B":
            del self.adapt

    def forward(self,z,g,regions,mask,training_outputs=False):
        pools=z[:,:18432].reshape(-1,12,1536)
        if self.fusion=="A":
            feature=(pools[:,[5,8,11]]*self.adapt.softmax(0)[None,:,None]).sum(1)
        elif self.fusion=="B":
            feature=pools.mean(1)
        else:
            feature=(pools*self.adapt.softmax(0)[None,:,None]).sum(1)
        original=self.base.training_outputs(feature,g,regions,mask) if training_outputs else self.base(feature,g,regions,mask)
        tokens=z[:,18432:].reshape(-1,256,769)
        valid=tokens[...,768].bool()
        h=self.projection(tokens[...,:768])
        score=(h@self.query.T).masked_fill(~valid[...,None],-1e9)
        weights=score.softmax(1)*valid[...,None]
        values=(weights.transpose(1,2)@h).squeeze(1)
        delta=self.gamma*(values[:,None,:]*self.readout[None,:,:]).sum(-1)
        return (original[0]+delta,original[1]) if training_outputs else original+delta

@torch.no_grad()
def cache(data):
    folder=OUT/"cache"
    folder.mkdir(exist_ok=True)
    if (folder/"COMPLETE").exists():
        return np.load(folder/"pooled.npy",mmap_mode="r"),np.load(folder/"layer9.npy",mmap_mode="r")
    tok=e.e4.load_model(e.TOKENIZER,DEVICE).eval().requires_grad_(False)
    model=e.load_continuous_checkpoint(e.DEFAULT_RUN/"mlm/scratch/checkpoint-best.pth",DEVICE).eval().requires_grad_(False)
    before=(e.state_sha(tok.state_dict()),e.state_sha(model.state_dict()))
    paths=[e.DOWNSTREAM/f"train_{k}.npy" for k in ("data","data_in_chans","data_in_times")]
    w,c,t=[np.load(path,mmap_mode="r") for path in paths]
    pooled=np.lib.format.open_memmap(folder/"pooled.npy",mode="w+",dtype=np.float32,shape=(len(w),12,1536))
    layer9=np.lib.format.open_memmap(folder/"layer9.npy",mode="w+",dtype=np.float32,shape=(len(w),256,769))
    norm=model.backbone
    for i in range(0,len(w),32):
        x,ch,ti=[torch.tensor(np.array(a[i:i+32]),device=DEVICE) for a in (w,c,t)]
        valid=tok.valid_mask(x,ch,ti)
        ze=tok.encode_to_quantizer_features(x,ch.long(),ti.long()).float()
        h=norm.input_proj(ze)*valid[...,None]
        h=torch.cat((norm.cls_token[None,None].expand(len(x),1,-1),h),1)
        h=norm.dropout(h+norm.pos_embed)
        fv=torch.cat((torch.ones_like(valid[:,:1]),valid),1)
        for layer,block in enumerate(norm.transformer.layers,1):
            h=e.masked_transformer_forward(SimpleNamespace(layers=[block]),h,fv)
            state=norm.norm_layer(h)[:,1:]*valid[...,None]
            q=b.pool(state,valid)
            pooled[i:i+len(x),layer-1]=q.cpu().numpy()
            if layer in (6,9,12):
                torch.testing.assert_close(q,data["z"][i:i+len(x),(6,9,12).index(layer)],atol=2e-5,rtol=1e-5)
            if layer==9:
                layer9[i:i+len(x)]=torch.cat((state,valid[...,None]),-1).cpu().numpy()
    pooled.flush()
    layer9.flush()
    assert before==(e.state_sha(tok.state_dict()),e.state_sha(model.state_dict()))
    b.dump(OUT/"cache_audit.json",dict(mapping="index0 = full Transformer block1 output after same frozen final LayerNorm; CLS excluded",pooled_sha256=e.sha256(folder/"pooled.npy"),layer9_sha256=e.sha256(folder/"layer9.npy"),
        metadata_sha256=e.sha256(e.DOWNSTREAM/"train_metadata.csv"),frozen_states_unchanged=True,
        layer_mean=np.asarray(pooled).mean((0,2)).tolist(),layer_std=np.asarray(pooled).std((0,2)).tolist(),
        precision="float32, no autocast",normalization="same frozen final LN as E25; no new fitted scaling"))
    (folder/"COMPLETE").touch()
    return pooled,layer9

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    started=time.time()
    meta=pd.read_csv(e.DOWNSTREAM/"train_metadata.csv")
    y=np.load(e.DOWNSTREAM/"train_labels.npy")
    assert np.array_equal(y.argmax(1),meta.label_id)
    b.dump(OUT/"config.json",dict(methods=["A","B","C"],outer_folds=3,seeds=[0,1,2],epochs=30,
        lr=5e-4,normalization="frozen final LayerNorm on each block, unchanged E25 scale",
        fold_sha256=e.sha256(s.OUT/"fold_assignment.csv"),test_loaded=False))
    b.dump(OUT/"environment.json",dict(torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    load=lambda name:torch.tensor(np.load(e.E10_RUN/f"features/train_{name}.npy"),device=DEVICE)
    regions,mask=e.e11.masked_region_means(load("lead_ze_mean_max"),load("lead_present").bool())
    d=dict(z=torch.tensor(np.load(p.source/"features/train.npy"),device=DEVICE),
        g=load("global_ze_mean_max"),regions=regions,mask=mask,y=torch.tensor(y,device=DEVICE))
    b.status(OUT,"cache")
    pooled,tokens=cache(d)
    packed=torch.empty((len(y),18432+256*769),device=DEVICE)
    packed[:,:18432]=torch.tensor(np.asarray(pooled).reshape(len(y),-1),device=DEVICE)
    packed[:,18432:]=torch.tensor(np.asarray(tokens).reshape(len(y),-1),device=DEVICE)
    d["z"]=packed
    del pooled,tokens
    b.dump(OUT/"parameter_count.json",{m:sum(a.numel() for a in Head(m,0).parameters()) for m in ("A","B","C")})
    folds=pd.read_csv(s.OUT/"fold_assignment.csv")
    predictions={m:np.full((3,len(y),4),np.nan,np.float32) for m in ("A","B","C")}
    rows,weights,parity=[],[],[]
    sources=[e.TOKENIZER,e.DEFAULT_RUN/"mlm/scratch/checkpoint-best.pth",*s.OUT.glob("fold*/downstream/shared/seed*/checkpoint-best.pth")]
    hashes={str(a):e.sha256(a) for a in sources}
    b.Head=Head
    for fold in range(3):
        roles={role:f.row.to_numpy() for role,f in folds[folds.fold==fold].groupby("role")}
        assert all(not set(meta.iloc[roles[a]].entity_id)&set(meta.iloc[roles[c]].entity_id) for a,c in (("fit","select"),("fit","outer"),("select","outer")))
        data={"train":s.subset(d,roles["fit"]),"val":s.subset(d,roles["select"])}
        outer=s.subset(d,roles["outer"])
        for method in ("A","B","C"):
            for seed in range(3):
                b.status(OUT,f"fold{fold}_{method}_{seed}")
                result=b.train(OUT/f"fold{fold}",data,DEVICE,method,seed)
                ck=OUT/f"fold{fold}/downstream/{method}/seed{seed}/checkpoint-best.pth"
                head=Head(method,seed).to(DEVICE)
                head.load_state_dict(torch.load(ck,map_location=DEVICE,weights_only=False)["model"])
                _,q,_=b.evaluate(head,outer)
                predictions[method][seed,roles["outer"]]=q
                rows.append(dict(fold=fold,method=method,seed=seed,best_epoch=result["best_epoch"],**s.metrics(y[roles["outer"]],q)))
                if method=="A":
                    original=s.OUT/f"fold{fold}/downstream/shared/seed{seed}/checkpoint-best.pth"
                    old=torch.load(original,map_location=DEVICE,weights_only=False)
                    if result["best_epoch"]!=old["best_epoch"]:
                        raise AssertionError("baseline selected epoch mismatch")
                    assert all(torch.equal(v,old["model"][k].to(v.device)) for k,v in head.state_dict().items())
                    parity.append(dict(fold=fold,seed=seed,checkpoint_tensors_exact=True))
                w=np.zeros(12)
                if method=="A":w[[5,8,11]]=head.adapt.softmax(0).detach().cpu().numpy()
                elif method=="B":w[:]=1/12
                else:w=head.adapt.softmax(0).detach().cpu().numpy()
                weights.append(dict(fold=fold,seed=seed,method=method,**{f"layer{i+1}":float(a) for i,a in enumerate(w)}))
        del data,outer
    pd.DataFrame(rows).to_csv(OUT/"fold_seed_metrics.csv",index=False)
    pd.DataFrame(weights).to_csv(OUT/"layer_weights.csv",index=False)
    b.dump(OUT/"baseline_parity.json",parity)
    summary={m:s.summarize(y,q) for m,q in predictions.items()}
    for m,q in predictions.items():
        assert np.isfinite(q).all()
        np.save(OUT/f"oof_{m}.npy",q)
    summary["bootstrap"]={f"C_minus_{m}":s.contrast(y,predictions["C"].mean(0),predictions[m].mean(0),meta.entity_id.to_numpy()) for m in ("A","B")}
    b.dump(OUT/"summary.json",summary)
    assert all(e.sha256(Path(path))==h for path,h in hashes.items())
    b.dump(OUT/"execution_audit.json",dict(seconds=time.time()-started,source_sha256=hashes,unchanged=True))
    b.status(OUT,"complete")

if __name__=="__main__":
    main()
