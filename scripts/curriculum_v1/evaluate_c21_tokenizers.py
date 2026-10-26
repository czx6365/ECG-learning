#!/usr/bin/env python3
"""Compare exact shift disagreement, cached morphology purity and SID usage."""
import argparse, json, math, pathlib, sys
from pathlib import Path
import numpy as np, torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'heartlang').exists())
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'/'experiments'))
import train_morphology_v2_physaware_vq as morphology

FEATURES=(('r_polarity',2),('t_polarity',3),('st_state',3),('qrs_type',2))

def shift_zero(x, delta):
    y=np.zeros_like(x)
    if delta>0: y[...,delta:]=x[...,:-delta]
    elif delta<0: y[...,:delta]=x[...,-delta:]
    else: y[:]=x
    return y

@torch.no_grad()
def encode(model,x,c,t,device):
    xt=torch.from_numpy(np.array(x,dtype=np.float32,copy=True)).to(device)
    ct=torch.from_numpy(np.array(c,dtype=np.int64,copy=True)).to(device)
    tt=torch.from_numpy(np.array(t,dtype=np.int64,copy=True)).to(device)
    with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
        return model.get_codebook_indices(xt,in_chan_matrix=ct,in_time_matrix=tt).cpu().numpy()

def evaluate(name,ckpt,data,chans,times,labels,device,batch):
    model=morphology.MorphologyV2PhysAwareVQ(
        codebook_size=1024,codebook_dim=64,token_len=96,
        ema_decay=0.99,quantize_kmeans_init=False,phys_head_source='z_q')
    state=torch.load(ckpt,map_location='cpu',weights_only=False)['model']
    target=model.state_dict()
    shared={k:v for k,v in state.items() if k in target and target[k].shape==v.shape
            and k.startswith(('encoder.','quantize.'))}
    required=[k for k in target if k.startswith(('encoder.','quantize.')) and k not in shared]
    if required: raise RuntimeError(f'{name}: missing tokenizer keys: {required[:10]}')
    model.load_state_dict(shared,strict=False)
    model=model.to(device).eval()
    usage=np.zeros(1024,np.int64)
    joint={(f,c):np.zeros((1024,c),np.int64) for f,c in FEATURES}
    disagree={str(k):[0,0] for k in (2,4,6)}
    for start in range(0,len(data),batch):
        end=min(start+batch,len(data)); x=np.asarray(data[start:end]); c=np.asarray(chans[start:end]); t=np.asarray(times[start:end])
        valid=(c>0)&(t>0)&~np.isclose(x,0).all(axis=-1)
        sid=encode(model,x,c,t,device); s=sid[valid]
        usage += np.bincount(s,minlength=1024)
        lab=np.asarray(labels[start:end])
        for fi,(fname,nclass) in enumerate(FEATURES):
            y=lab[...,fi][valid]; ok=(y>=0)&(y<nclass)
            joint[(fname,nclass)] += np.bincount(s[ok]*nclass+y[ok],minlength=1024*nclass).reshape(1024,nclass)
        for mag in (2,4,6):
            for delta in (-mag,mag):
                ss=encode(model,shift_zero(x,delta),c,t,device)[valid]
                disagree[str(mag)][0] += int((ss!=s).sum()); disagree[str(mag)][1] += int(len(s))
        if start==0 or end==len(data) or (start//batch)%20==0: print(name,end,'/',len(data),flush=True)
    total=int(usage.sum()); p=usage[usage>0]/max(total,1)
    purity={}
    for fname,nclass in FEATURES:
        table=joint[(fname,nclass)]; denom=int(table.sum())
        purity[fname]=float(table.max(axis=1).sum()/denom) if denom else 0.0
    return {'checkpoint':str(ckpt),'total_tokens':total,'active_sid':int((usage>0).sum()),
            'perplexity':float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.0,
            'sid_disagreement':{k:v[0]/v[1] for k,v in disagree.items()},
            'morphology_purity':{**purity,'macro':float(np.mean(list(purity.values())))}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--batch_size',type=int,default=256)
    a=ap.parse_args(); root=ROOT
    d=root/'datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct'
    cache=root/'results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct'
    data=np.load(d/'val_data.npy',mmap_mode='r'); chans=np.load(d/'val_data_in_chans.npy',mmap_mode='r'); times=np.load(d/'val_data_in_times.npy',mmap_mode='r'); labels=np.load(cache/'val_phys_discrete.npy',mmap_mode='r')
    checkpoints={
      'C1':root/'checkpoints/curriculum_v1/c1_word_recon_vq_1024_dim64_bs256_e50/checkpoint-best.pth',
      'C2_old_corr_deriv':root/'checkpoints/curriculum_v1/c2_discrete_patterns_from_c1_e50/checkpoint-best.pth',
      'C2_pattern_only':root/'checkpoints/curriculum_v1/c2_pattern_only_from_c1_e50/checkpoint-best.pth',
      'C2.1_pattern_shift':root/'checkpoints/curriculum_v1/c21_pattern_shift_from_c2_e50/checkpoint-best.pth'}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out={}
    for name,ckpt in checkpoints.items(): out[name]=evaluate(name,ckpt,data,chans,times,labels,device,a.batch_size)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
