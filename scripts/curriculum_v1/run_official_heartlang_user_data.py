#!/usr/bin/env python3
"""Run the official HeartLang waveform architecture on the locked user dataset.

The ST_ECGFormer forward path is used unchanged: official TokenEmbedding
Conv1d -> mask replacement -> spatial/temporal embeddings -> Transformer.
Only valid-token mask sampling is adapted so padding cannot become a target.
All downstream data, E11 fusion, seeds, losses, and metrics are reused.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

from heartlang.models.modeling_pretrain import ST_ECGFormer
from heartlang.utils.utils import set_seed
import run_e9_ze_fusion_r2 as e9
import run_e10_lead_aware_ze_fusion as e10
import run_e11_region_residual_fusion as e11
import run_e12_continuous_input as e12
import run_e13_continuous_discrete_ablation as e13

EXP = "heartlang_official_user_data_20261019"
DEFAULT_RUN = ROOT / "results" / EXP
DOWNSTREAM = e9.DATA
E10_RUN = ROOT / "results/curriculum_v1/e10_lead_aware_ze_fusion_20261018"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def status(run: Path, text: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "status.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def official_model(device: torch.device) -> ST_ECGFormer:
    set_seed(0)
    return ST_ECGFormer(seq_len=256, time_window=96, depth=12, embed_dim=768,
                        heads=8, mlp_dim=1024, dim_head=64, vocab_size=1024,
                        dropout=.1, emb_dropout=.1).to(device)


class OfficialDataset(Dataset):
    def __init__(self, cache: Path, split: str):
        self.wave = np.load(cache / f"{split}_wave.npy", mmap_mode="r")
        self.chan = np.load(cache / f"{split}_chan.npy", mmap_mode="r")
        self.time = np.load(cache / f"{split}_time.npy", mmap_mode="r")
        self.sid = np.load(e12.PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        if not (self.wave.shape[:2] == self.chan.shape == self.time.shape == self.sid.shape):
            raise RuntimeError(f"{split}: cache alignment mismatch")
    def __len__(self): return len(self.sid)
    def __getitem__(self, i):
        return (np.array(self.wave[i], copy=True), np.array(self.chan[i], copy=True),
                np.array(self.time[i], copy=True), np.array(self.sid[i], copy=True))


def cache_inputs(run: Path) -> None:
    out = run / "pretrain_input_cache"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    waves = {s: np.load(e12.PRETRAIN_SOURCE / f"{s}_data.npy", mmap_mode="r") for s in ("train", "val")}
    chans = {s: np.load(e12.PRETRAIN_SOURCE / f"{s}_data_in_chans.npy", mmap_mode="r") for s in ("train", "val")}
    times = {s: np.load(e12.PRETRAIN_SOURCE / f"{s}_data_in_times.npy", mmap_mode="r") for s in ("train", "val")}
    audit = {}
    for split in ("train", "val"):
        code = np.load(e12.PRETRAIN_INDEX / f"{split}_source_code.npy")
        index = np.load(e12.PRETRAIN_INDEX / f"{split}_source_index.npy")
        specs = (("wave", np.float32, (len(index),256,96), waves),
                 ("chan", np.int16, (len(index),256), chans),
                 ("time", np.int16, (len(index),256), times))
        outputs = {name: np.lib.format.open_memmap(out/f"{split}_{name}.npy", mode="w+", dtype=dtype, shape=shape)
                   for name,dtype,shape,_ in specs}
        for st in range(0,len(index),512):
            en=min(st+512,len(index)); c=code[st:en]; ii=index[st:en]
            for name,dtype,_shape,source in specs:
                outputs[name][st:en]=e12.gather_source(source,c,ii).astype(dtype)
            if st==0 or en==len(index) or (st//512)%20==0: print(f"cache {split} {en}/{len(index)}",flush=True)
        for mm in outputs.values(): mm.flush()
        audit[split]={"records":len(index),"wave_shape":[len(index),256,96],
                      "valid_tokens":int(((np.load(e12.PRETRAIN_SID/f"{split}_sid.npy",mmap_mode="r")>=0)&
                                          (np.load(e12.PRETRAIN_SID/f"{split}_sid.npy",mmap_mode="r")<1024)).sum())}
    dump(out/"audit.json",audit);(out/"COMPLETE").touch()


@torch.inference_mode()
def implementation_audit(run: Path, device: torch.device) -> None:
    out=run/"audit";out.mkdir(parents=True,exist_ok=True)
    model=official_model(device).eval();x=torch.randn(1,256,96,device=device);mask=torch.zeros(1,256,dtype=torch.bool,device=device);mask[:,5]=True
    ch=torch.ones(1,256,dtype=torch.long,device=device);ti=torch.ones_like(ch)
    x2=x.clone();x2[:,5]+=100
    a=model.forward_feature(x,mask,ch,ti);b=model.forward_feature(x2,mask,ch,ti)
    # Official Conv precedes mask replacement; neighboring token leakage is expected and recorded.
    neighbor_delta=float((a[:,1:]-b[:,1:]).abs().amax())
    dump(out/"official_forward_audit.json",{
        "class":"heartlang.models.modeling_pretrain.ST_ECGFormer",
        "official_forward_unmodified":True,"token_embedding":"official Conv1d(96,768,kernel=3,circular)",
        "spatial_embedding":True,"temporal_embedding":True,"explicit_key_padding_mask":False,
        "mask_sampling_adaptation":"50% of valid tokens only; padding excluded from targets",
        "masked_content_mutation_max_all_token_delta":neighbor_delta,
        "interpretation":"nonzero is expected because official Conv1d is applied before mask replacement"})
    e12.overlap_audit(run);(out/"COMPLETE").touch()


def lr_at(step,total,warmup): return e12.lr_at(step,total,warmup,5e-4,1e-5,1e-6)


def one_epoch(model,loader,device,opt,scaler,ep,epochs,train):
    model.train(train);sums={"loss":0.,"masked_sid_accuracy":0.,"batches":0}
    for step,(wave,chan,times,sid) in enumerate(loader):
        wave=wave.to(device,non_blocking=True).float();chan=chan.to(device,non_blocking=True).long();times=times.to(device,non_blocking=True).long();sid=sid.to(device,non_blocking=True).long()
        valid=(sid>=0)&(sid<1024);mask=e12.random_mask(valid,.5);labels=sid[mask]
        if train:
            gs=(ep-1)*len(loader)+step;lr=lr_at(gs,epochs*len(loader),5*len(loader))
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
            logits=model(wave,mask_bool_matrix=mask,in_chan_matrix=chan,in_time_matrix=times)
            loss=F.cross_entropy(logits.float(),labels)
        if not torch.isfinite(loss):raise FloatingPointError(f"nonfinite epoch={ep} step={step}")
        if train:
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),3.0);scaler.step(opt);scaler.update()
        sums["loss"]+=float(loss);sums["masked_sid_accuracy"]+=float((logits.argmax(-1)==labels).float().mean());sums["batches"]+=1
        if step==0 or (step+1)%50==0 or step+1==len(loader):print(f"{'train' if train else 'val'} epoch={ep} {step+1}/{len(loader)} loss={float(loss):.6f}",flush=True)
    n=max(1,sums["batches"]);return {"loss":sums["loss"]/n,"masked_sid_accuracy":sums["masked_sid_accuracy"]/n,"batches":sums["batches"]}


def pretrain(run: Path,device:torch.device,epochs:int):
    out=run/"pretrain"; 
    if (out/"COMPLETE").exists():return
    out.mkdir(parents=True,exist_ok=False);model=official_model(device)
    train=OfficialDataset(run/"pretrain_input_cache","train");val=OfficialDataset(run/"pretrain_input_cache","val")
    gen=torch.Generator().manual_seed(0)
    tl=DataLoader(train,256,shuffle=True,generator=gen,num_workers=8,pin_memory=True,drop_last=True)
    vl=DataLoader(val,384,shuffle=False,num_workers=8,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05);scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=float("inf");best_ep=None;history=[];start=time.time()
    for ep in range(1,epochs+1):
        tr=one_epoch(model,tl,device,opt,scaler,ep,epochs,True);row={"epoch":ep,"train":tr}
        if ep==1 or ep%10==0 or ep==epochs:
            with torch.no_grad():va=one_epoch(model,vl,device,None,None,ep,epochs,False)
            row["val"]=va
            if va["loss"]<best:
                best=va["loss"];best_ep=ep;torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"epoch":ep,"best_val_loss":best},out/"checkpoint-best.pth")
        history.append(row);dump(out/"training_history.json",history);print(json.dumps(row),flush=True)
    dump(out/"summary.json",{"epochs":epochs,"pretrain_seed":0,"best_epoch":best_ep,"best_val_loss":best,"gpu_wall_seconds":time.time()-start,
                              "parameters":sum(p.numel() for p in model.parameters())});(out/"COMPLETE").touch()


def load_checkpoint(run:Path,device):
    m=official_model(device);ck=torch.load(run/"pretrain/checkpoint-best.pth",map_location=device,weights_only=False);m.load_state_dict(ck["model"],strict=True);return m.eval()


@torch.inference_mode()
def features(run:Path,device,batch=24):
    out=run/"features/e13_official";
    if (out/"COMPLETE").exists():return
    out.mkdir(parents=True,exist_ok=False);model=load_checkpoint(run,device).requires_grad_(False);audit={}
    for split in ("train","val","test"):
        waves=np.load(DOWNSTREAM/f"{split}_data.npy",mmap_mode="r");chans=np.load(DOWNSTREAM/f"{split}_data_in_chans.npy",mmap_mode="r");times=np.load(DOWNSTREAM/f"{split}_data_in_times.npy",mmap_mode="r");sid=np.load(e9.SID/f"{split}_sid.npy",mmap_mode="r")
        result=np.empty((len(waves),1536),np.float32)
        for st in range(0,len(waves),batch):
            en=min(st+batch,len(waves));x=torch.from_numpy(np.asarray(waves[st:en],dtype=np.float32).copy()).to(device);c=torch.from_numpy(np.asarray(chans[st:en]).copy()).long().to(device);t=torch.from_numpy(np.asarray(times[st:en]).copy()).long().to(device);s=torch.from_numpy(np.asarray(sid[st:en]).copy()).long().to(device);valid=(s>=0)&(s<1024)
            token=model(x,in_chan_matrix=c,in_time_matrix=t,return_qrs_tokens=True).float();count=valid.sum(1).clamp_min(1).unsqueeze(-1);mean=(token*valid.unsqueeze(-1)).sum(1)/count;mx=token.masked_fill(~valid.unsqueeze(-1),-torch.inf).amax(1);mx[~valid.any(1)]=0;result[st:en]=torch.cat([mean,mx],1).cpu().numpy()
        np.save(out/f"{split}_heartlang_mean_max.npy",result);audit[split]={"shape":list(result.shape),"finite":bool(np.isfinite(result).all()),"mean":float(result.mean()),"std":float(result.std())}
    dump(out/"audit.json",audit);(out/"COMPLETE").touch()


def downstream(run:Path,device): e13.train_heads(run,"e13_official",device)


def test(run:Path,device):
    out=run/"final_test_summary.json"
    if out.exists():return
    data=e13.load_arrays(run,"e13_official",device);rows=[]
    for seed in (0,1,2):
        ck=torch.load(run/f"downstream/e13_official/seed{seed}/checkpoint-best.pth",map_location=device,weights_only=False);m=e11.E11RegionResidual(seed).to(device);m.load_state_dict(ck["model"],strict=True);met,prob,logits=e13.eval_head(m,data["test"]);od=run/f"downstream/e13_official/seed{seed}";np.save(od/"test_logits.npy",logits);np.save(od/"test_probabilities.npy",prob);rows.append({"seed":seed,"test":met})
    keys=("r2","mse","mae","macro_roc_auc","macro_f1","accuracy","brier_score","ece15_top_label")
    agg={k:{"mean":float(np.mean([r["test"][k] for r in rows])),"sd":float(np.std([r["test"][k] for r in rows],ddof=1))} for k in keys};agg["per_output_r2"]={n:{"mean":float(np.mean([r["test"]["per_output_r2"][n] for r in rows])),"sd":float(np.std([r["test"]["per_output_r2"][n] for r in rows],ddof=1))} for n in e9.LABEL_NAMES};dump(out,{"method":"official HeartLang on user data","rows":rows,"aggregate":agg})


def report(run:Path):
    pre=json.loads((run/"pretrain/summary.json").read_text());val=json.loads((run/"downstream/e13_official/aggregate_val.json").read_text())["aggregate"];te=json.loads((run/"final_test_summary.json").read_text())["aggregate"]
    lines=["# 官方 HeartLang 方法在当前数据上的结果","","## 实现","","- 使用项目官方 `ST_ECGFormer` 原始前向：96点心拍 → Conv1d → Mask → Spatial/Temporal Embedding → Transformer → SID预测。","- 官方 forward 原样保留：无显式 Key-Padding Mask，Conv1d发生在Mask替换前。","- 当前数据适配仅限于从有效心拍中抽取50% Mask，padding不作为SID监督目标。","- 固定患者划分、E3 Codebook SID、100轮预训练、E11区域残差下游、seed0/1/2和Val R²选模。","",f"预训练最佳：epoch {pre['best_epoch']}，Val SID CE {pre['best_val_loss']:.6f}。","","## Val（三种下游seed）","",f"Overall R² {val['r2']['mean']:.6f} ± {val['r2']['sd']:.6f}",f"Macro AUC {val['macro_roc_auc']['mean']:.6f}，Macro F1 {val['macro_f1']['mean']:.6f}","四类R²："+", ".join(f"{n}={val['per_output_r2'][n]['mean']:.6f}" for n in e9.LABEL_NAMES),"","## Test（三种下游seed）","",f"Overall R² {te['r2']['mean']:.6f} ± {te['r2']['sd']:.6f}",f"Macro AUC {te['macro_roc_auc']['mean']:.6f}，Macro F1 {te['macro_f1']['mean']:.6f}","四类R²："+", ".join(f"{n}={te['per_output_r2'][n]['mean']:.6f}" for n in e9.LABEL_NAMES),""]
    (run/"README.md").write_text("\n".join(lines),encoding="utf-8");(run/"COMPLETE").touch();status(run,"complete")


def protocol(run:Path,epochs:int):
    if (run/"protocol.json").exists():return
    dump(run/"protocol.json",{"experiment":EXP,"architecture":"official ST_ECGFormer unchanged forward","input":"raw heartbeat [256,96]","token_embedding":"official Conv1d kernel3 circular before mask","spatial_embedding":True,"temporal_embedding":True,"explicit_key_padding_mask":False,"adaptation":"mask 50% valid tokens only","sid_target":"frozen E3 Codebook IDs","pretrain_epochs":epochs,"batch":256,"seed":0,"downstream_seeds":[0,1,2],"downstream":"official token Mean+Max + unchanged E11 region residual","selection":"best pretrain Val CE and best downstream Val Overall R2"})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["audit","cache","pretrain","features","downstream","test","report"]);ap.add_argument("--output_dir",type=Path,default=DEFAULT_RUN);ap.add_argument("--epochs",type=int,default=100);args=ap.parse_args();run=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir;run.mkdir(parents=True,exist_ok=True);protocol(run,args.epochs);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage=="audit":implementation_audit(run,device)
    elif args.stage=="cache":cache_inputs(run)
    elif args.stage=="pretrain":pretrain(run,device,args.epochs)
    elif args.stage=="features":features(run,device)
    elif args.stage=="downstream":downstream(run,device)
    elif args.stage=="test":test(run,device)
    elif args.stage=="report":report(run)

if __name__=="__main__":main()
