"""Raw continuous 100 Hz waveform increment, with conservative window proxies."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import run_e25 as base
from report_increment import report
from scripts.data_prep.ECG_DSA.tokenize_pretrain_qrs import QRSTokenizerLite

e = base.e
source = e.ROOT / "results/e25_intermediate_layers"
run = e.ROOT / "results/e30_physiology_increment"
OriginalHead = base.Head
DIM = 9*6+3

class Head(OriginalHead):
    def __init__(self, method, seed):
        if method == "only":
            nn.Module.__init__(self)
        else:
            super().__init__("learned_mix",seed)
        self.variant = method
        if method != "learned_mix":
            self.branch = nn.Linear(DIM,4)
            if method != "only":
                self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self,z,g,regions,mask,training_outputs=False):
        if self.variant == "only":
            delta = self.branch(z[:,4608:])
            return (delta,delta[:,:3]) if training_outputs else delta
        original = super().forward(z[:,:4608].reshape(-1,3,1536),g,regions,mask,training_outputs)
        if self.variant == "learned_mix":
            return original
        f = z[:,4608:]
        if self.variant == "capacity":
            f = z[:,:DIM]
        delta = self.branch(f)
        delta = delta*self.gamma
        return (original[0]+delta,original[1]) if training_outputs else original+delta

def waveform_features(wave, peaks):
    beats = []
    for left,r,right in zip(peaks,peaks[1:],peaks[2:]):
        if r-20 < left+12 or r+40 >= right-8 or r-20 < 0 or r+40 > wave.shape[1]:
            continue
        baseline = np.median(wave[:,r-20:r-12],axis=1)
        early = wave[:,r+8:r+16]-baseline[:,None]
        late = wave[:,r+20:r+40]-baseline[:,None]
        qrs = wave[:,r-5:r+7]-baseline[:,None]
        peak = late[np.arange(9),np.argmax(np.abs(late),axis=1)]
        beats.append(np.stack([early.mean(1),(early[:,-3:].mean(1)-early[:,:3].mean(1))/.05,peak,late.sum(1)*.01,qrs.max(1),qrs.min(1)],1))
    morphology = np.median(beats,axis=0).ravel() if beats else np.full(54,np.nan)
    rr = np.diff(peaks)/100
    rhythm = [60/rr.mean(),rr.mean(),rr.std()] if len(rr)>1 else [np.nan]*3
    return np.r_[morphology,rhythm]

def extract(split):
    frame = pd.read_csv(e.DOWNSTREAM/f"{split}_metadata.csv")
    sources = {}
    result = []
    coverage = []
    tok = QRSTokenizerLite(fs=100)
    for row in frame.itertuples():
        src = Path(str(row.source_dataset))
        src = src if src.is_absolute() else e.ROOT/src
        raw = Path(str(src).replace("/pretrain_qrs/","/pretrain_waveforms/").replace("/qrs/","/waveforms/"))
        ss = str(row.source_split)
        assert ss in ("train","val")
        key = raw,ss
        if key not in sources:
            sources[key] = (np.load(raw/f"{ss}_data.npy",mmap_mode="r"),pd.read_csv(raw/f"{ss}_metadata.csv"))
        waves,meta = sources[key]
        i = int(row.source_index)
        if str(meta.iloc[i].csv_name) != str(row.csv_name):
            found = np.flatnonzero(meta.csv_name.astype(str).values==str(row.csv_name))
            assert len(found)==1
            i = int(found[0])
        w = np.asarray(waves[i])
        assert w.shape == (9,1000)
        peaks = tok.qrs_detection(w)
        result.append(waveform_features(w,peaks))
        coverage.append(float(np.mean(np.diff(peaks)[1:] >= 82)) if len(peaks)>2 else np.nan)
    return np.asarray(result), np.asarray(coverage)

def main():
    run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device("cuda")
    d=base.data(source,device)
    frames=[pd.read_csv(e.DOWNSTREAM/f"{s}_metadata.csv") for s in ("train","val")]
    assert not set(frames[0].entity_id)&set(frames[1].entity_id)
    base.status(run,"raw_feature_audit")
    extracted={s:extract(s) for s in ("train","val")}
    raw={s:v[0] for s,v in extracted.items()}
    np.save(run/"val_post_r400_coverage.npy",extracted["val"][1])
    for s,x in raw.items():
        np.save(run/f"{s}_raw_features.npy",x)
    base.dump(run/"feature_audit.json",{s:dict(valid_fraction=np.isfinite(x).mean(0).tolist(),median=np.nanmedian(x,axis=0).tolist()) for s,x in raw.items()})
    assert np.isfinite(raw["train"]).any(axis=0).all(), "A predefined feature has no usable Train observations"
    med=np.nanmedian(raw["train"],axis=0)
    clean={s:np.where(np.isfinite(x),x,med) for s,x in raw.items()}
    mean=clean["train"].mean(0)
    scale=clean["train"].std(0).clip(1e-6)
    np.savez(run/"train_scaler.npz",median=med,mean=mean,scale=scale)
    for s in d:
        d[s]["z"]=torch.cat([d[s]["z"].flatten(1),torch.tensor((clean[s]-mean)/scale,device=device,dtype=torch.float32)],1)
    # Check exact zero-residual parity and gate gradient before fitting.
    a=d["train"]
    args=[a[k][:8] for k in ("z","g","regions","mask")]
    ref=Head("learned_mix",0).to(device).eval()
    for m in ("increment","capacity"):
        h=Head(m,0).to(device).eval()
        torch.testing.assert_close(h(*args),ref(*args),rtol=0,atol=0)
        h(*args).sum().backward()
        assert h.gamma.grad.abs()>0
    base.Head=Head
    methods=("learned_mix","only","increment","capacity")
    base.dump(run/"config.json",dict(epochs=30,seeds=[0,1,2],fs=100,leads=[1,2,3,7,8,9,10,11,12],test_loaded=False,features="R-relative early/late proxies, not delineated ST/T; standardized waveform units"))
    base.dump(run/"environment.json",dict(torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    base.dump(run/"model_parameter_count.json",{m:sum(p.numel() for p in Head(m,0).parameters()) for m in methods})
    rows=[]
    for m in methods:
        for seed in range(3):
            base.status(run,f"{m}_{seed}")
            rows.append(base.train(run,d,device,m,seed))
        if m=="learned_mix":
            old=json.loads((source/"validation_summary.json").read_text())["learned_mix"]["per_seed"]
            delta=[a["val"]["r2"]-b["val"]["r2"] for a,b in zip(rows,old)]
            base.dump(run/"baseline_parity.json",delta)
            assert max(map(abs,delta))<1e-6
    base.dump(run/"seed_results.json",rows)
    report(run,rows,d)
    from sklearn.metrics import r2_score
    val=d["val"]["y"].cpu().numpy()
    ref=np.mean([np.load(run/"downstream/learned_mix"/f"seed{s}"/"val_probabilities.npy") for s in range(3)],0)
    full=np.mean([np.load(run/"downstream/increment"/f"seed{s}"/"val_probabilities.npy") for s in range(3)],0)
    diagnostics={}
    coverage=extracted["val"][1]
    for group,mask in {"poor_coverage":coverage<.5,"better_coverage":coverage>=.5}.items():
        diagnostics[group]={"records":int(mask.sum()),"r2_delta":float(r2_score(val[mask],full[mask])-r2_score(val[mask],ref[mask])) if mask.sum()>1 else None}
    wrong=ref.argmax(1)!=val.argmax(1)
    correct=full.argmax(1)==val.argmax(1)
    np.save(run/"val_corrected_row_indices.npy",np.flatnonzero(wrong&correct))
    diagnostics["corrected_records"]=int((wrong&correct).sum())
    diagnostics["regressed_records"]=int((~wrong&~correct).sum())
    for group,mask in {"patient":val[:,3]==0,"healthy":val[:,3]==1}.items():
        diagnostics[group]={"records":int(mask.sum()),"accuracy_delta":float((full[mask].argmax(1)==val[mask].argmax(1)).mean()-(ref[mask].argmax(1)==val[mask].argmax(1)).mean())}
    # Fixed model sensitivity, replacing groups by the Train mean.
    for group,cols in {"early_late": [i*6+j for i in range(9) for j in range(4)],
                       "rhythm":list(range(54,57))}.items():
        ablated={**d["val"],"z":d["val"]["z"].clone()}
        ablated["z"][:,[4608+j for j in cols]]=0
        scores=[]
        for seed in range(3):
            m=Head("increment",seed).to(device)
            ck=torch.load(run/"downstream/increment"/f"seed{seed}"/"checkpoint-best.pth",map_location=device,weights_only=False)
            m.load_state_dict(ck["model"])
            scores.append(base.evaluate(m,ablated)[1])
        diagnostics[group+"_fixed_model_ablation"]={"r2_delta_after_removal":float(r2_score(val,np.mean(scores,0))-r2_score(val,full))}
    base.dump(run/"increment_diagnostics.json",diagnostics)
    base.status(run,"complete")

if __name__=="__main__":
    main()
