#!/usr/bin/env python3
"""E14-B: continuous z_e + real lead/time embeddings, validation only."""
from __future__ import annotations

import argparse, copy, csv, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

from heartlang.models.modeling_pretrain import SpatialEmbedding, TemporalEmbedding
from heartlang.utils.utils import set_seed
import run_e12_continuous_input as e12
import run_e11_region_residual_fusion as e11

RUN_DEFAULT = ROOT / "results/e14_lead_time_embedding"
E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
E13_RUN = ROOT / "results/e13_continuous_discrete_ablation"
LEAD_MAP = {1:"I",2:"II",3:"III",4:"aVR",5:"aVL",6:"aVF",7:"V1",8:"V2",9:"V3",10:"V4",11:"V5",12:"V6"}


class LeadTimeBackbone(e12.ContinuousBackbone):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.spa_embed = SpatialEmbedding(num_embeddings=16, embed_dim=self.embed_dim)
        self.tem_embed = TemporalEmbedding(num_embeddings=16, embed_dim=self.embed_dim)

    def forward_feature(self, ze, valid, lead, time_id, mask_bool_matrix=None):
        if ze.ndim != 3 or ze.shape[1:] != (self.seq_len, self.code_dim):
            raise ValueError(tuple(ze.shape))
        valid = valid.bool()
        if lead.shape != valid.shape or time_id.shape != valid.shape:
            raise ValueError("lead/time shape mismatch")
        if int(lead.min()) < 0 or int(lead.max()) > 15 or int(time_id.min()) < 0 or int(time_id.max()) > 15:
            raise ValueError("lead/time ID outside embedding vocabulary")
        x = self.input_proj(ze.float()) * valid.unsqueeze(-1).to(ze.dtype)
        mask = torch.zeros_like(valid) if mask_bool_matrix is None else mask_bool_matrix.bool() & valid
        # Full z_e replacement occurs before adding legitimate lead/time context.
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
        x = self.spa_embed(x, lead.long())
        x = self.tem_embed(x, time_id.long())
        x = x * valid.unsqueeze(-1).to(x.dtype)
        cls = self.cls_token.view(1,1,-1).expand(len(x),1,-1)
        x = torch.cat([cls,x],1)
        full_valid = torch.cat([torch.ones(len(x),1,dtype=torch.bool,device=x.device),valid],1)
        x = self.dropout(x + self.pos_embed)
        x = e12.masked_transformer_forward(self.transformer, x, full_valid)
        return self.norm_layer(x) * full_valid.unsqueeze(-1).to(x.dtype)

    def token_features(self, ze, valid, lead, time_id):
        return self.forward_feature(ze, valid, lead, time_id)[:,1:]


class LeadTimeSIDMLM(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        self.backbone = LeadTimeBackbone(**cfg)
        self.lm_head = nn.Linear(int(cfg["embed_dim"]), int(cfg["vocab_size"]))

    def forward(self, ze, valid, lead, time_id, mask):
        tokens = self.backbone.forward_feature(ze, valid, lead, time_id, mask)[:,1:]
        return self.lm_head(tokens[mask.bool() & valid.bool()])

    def extract_tokens(self, ze, valid, lead, time_id):
        return self.backbone.token_features(ze, valid, lead, time_id)


def initialize(device):
    set_seed(120012)
    model = LeadTimeSIDMLM(**e12.model_config())
    source = e12.fresh_sid_model(0).state_dict()
    state = model.state_dict(); copied=[]
    for k,v in source.items():
        if k == "backbone.sid_embed.weight": continue
        if k in state and state[k].shape == v.shape:
            state[k] = v.detach().clone(); copied.append(k)
    model.load_state_dict(state, strict=True)
    ref,_ = e12.initialize_continuous("scratch", torch.device("cpu"))
    shared_equal = all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in ref.state_dict().items())
    audit={"shared_initialization_exactly_matches_e12a":shared_equal,"copied_shared_keys":len(copied),
           "new_parameters":["backbone.spa_embed.embed.weight","backbone.tem_embed.embed.weight"],
           "lead_embedding_shape":list(model.backbone.spa_embed.embed.weight.shape),
           "time_embedding_shape":list(model.backbone.tem_embed.embed.weight.shape)}
    if not shared_equal: raise RuntimeError("E14 shared initialization differs from E12-A")
    return model.to(device),audit


def source_meta(split):
    src=e12.PRETRAIN_SOURCE; idx=e12.PRETRAIN_INDEX
    codes=np.load(idx/f"{split}_source_code.npy"); indices=np.load(idx/f"{split}_source_index.npy")
    chans={s:np.load(src/f"{s}_data_in_chans.npy",mmap_mode="r") for s in ("train","val")}
    times={s:np.load(src/f"{s}_data_in_times.npy",mmap_mode="r") for s in ("train","val")}
    return e12.gather_source(chans,codes,indices).astype(np.int16),e12.gather_source(times,codes,indices).astype(np.int16)


def prepare_meta(run):
    out=run/"pretrain_meta"
    if (out/"COMPLETE").exists(): return
    out.mkdir(parents=True,exist_ok=False); summary={}
    for split in ("train","val"):
        lead,t=source_meta(split); sid=np.load(e12.PRETRAIN_SID/f"{split}_sid.npy",mmap_mode="r")
        valid=(sid>=0)&(sid<1024)
        if lead.shape!=sid.shape or t.shape!=sid.shape: raise RuntimeError("metadata alignment")
        if np.any(valid & ((lead<=0)|(t<=0))): raise RuntimeError("valid token lacks lead/time")
        if lead.min()<0 or lead.max()>12 or t.min()<0 or t.max()>15: raise RuntimeError("ID range")
        np.save(out/f"{split}_lead.npy",lead);np.save(out/f"{split}_time.npy",t)
        ids,cnt=np.unique(lead[valid],return_counts=True); tids,tcnt=np.unique(t[valid],return_counts=True)
        summary[split]={"records":len(sid),"valid_tokens":int(valid.sum()),
          "lead_counts":{str(int(k)):int(v) for k,v in zip(ids,cnt)},
          "time_counts":{str(int(k)):int(v) for k,v in zip(tids,tcnt)}}
    observed=sorted({int(k) for s in summary.values() for k in s["lead_counts"]})
    e12.dump(out/"audit.json",{"lead_codebook":LEAD_MAP,"observed_lead_ids":observed,
      "observed_leads":[LEAD_MAP[i] for i in observed],"missing_standard_lead_ids":[i for i in range(1,13) if i not in observed],
      "note":"Source records contain nine measured leads I/II/III/V1-V6; aVR/aVL/aVF are not fabricated.",
      "time_semantics":"real R-peak sample index // 100 + 1 from tokenize_pretrain_qrs.py; not sequence ordinal",
      "splits":summary});(out/"COMPLETE").touch()


class DatasetLT(Dataset):
    def __init__(self,run,split):
        self.ze=np.load(E12_RUN/f"pretrain_ze_cache/{split}_ze.npy",mmap_mode="r")
        self.sid=np.load(e12.PRETRAIN_SID/f"{split}_sid.npy",mmap_mode="r")
        self.lead=np.load(run/f"pretrain_meta/{split}_lead.npy",mmap_mode="r")
        self.time=np.load(run/f"pretrain_meta/{split}_time.npy",mmap_mode="r")
    def __len__(self): return len(self.sid)
    def __getitem__(self,i):
        return tuple(np.array(x[i],copy=True) for x in (self.ze,self.sid,self.lead,self.time))


def epoch(model,loader,device,opt,scaler,ep,epochs,train):
    model.train(train); sums={"loss":0.,"mlm_acc":0.,"masked_tokens":0.,"batches":0}
    for step,(ze,sid,lead,ti) in enumerate(loader):
        ze=ze.float().to(device);sid=sid.long().to(device);lead=lead.long().to(device);ti=ti.long().to(device)
        valid=(sid>=0)&(sid<1024);mask=e12.random_mask(valid,.5);labels=sid[mask]
        if train:
            gs=(ep-1)*len(loader)+step;lr=e12.lr_at(gs,epochs*len(loader),5*len(loader),5e-4,1e-5,1e-6)
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
            logits=model(ze,valid,lead,ti,mask);loss=F.cross_entropy(logits.float(),labels)
        if train:
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),3.);scaler.step(opt);scaler.update()
        sums["loss"]+=float(loss);sums["mlm_acc"]+=float((logits.argmax(-1)==labels).float().mean());sums["masked_tokens"]+=labels.numel();sums["batches"]+=1
        if step==0 or (step+1)%50==0 or step+1==len(loader): print(f"{'train' if train else 'val'} epoch={ep} {step+1}/{len(loader)} loss={float(loss):.6f}",flush=True)
    n=max(1,sums["batches"]);return {k:(v/n if k!="batches" else v) for k,v in sums.items()}


def pretrain(run,device,epochs):
    out=run/"mlm/lead_time"
    if (out/"COMPLETE").exists():return
    out.mkdir(parents=True,exist_ok=False);model,audit=initialize(device);e12.dump(out/"initialization_audit.json",audit)
    gen=torch.Generator().manual_seed(0)
    tr=DataLoader(DatasetLT(run,"train"),batch_size=256,shuffle=True,generator=gen,num_workers=8,pin_memory=True,drop_last=True)
    va=DataLoader(DatasetLT(run,"val"),batch_size=384,shuffle=False,num_workers=8,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05);scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=float("inf");best_ep=None;hist=[];start=time.time()
    for ep in range(1,epochs+1):
        row={"epoch":ep,"train":epoch(model,tr,device,opt,scaler,ep,epochs,True)}
        if ep==1 or ep%10==0 or ep==epochs:
            with torch.no_grad():row["val"]=epoch(model,va,device,None,None,ep,epochs,False)
            if row["val"]["loss"]<best:
                best=row["val"]["loss"];best_ep=ep;torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":e12.model_config(),"epoch":ep,"best_val_loss":best},out/"checkpoint-best.pth")
        hist.append(row);e12.dump(out/"training_history.json",hist);print(json.dumps(row),flush=True)
    e12.dump(out/"summary.json",{"best_epoch":best_ep,"best_val_loss":best,"epochs":epochs,"pretrain_seed":0,"elapsed_seconds":time.time()-start,"trainable_parameters":sum(p.numel() for p in model.parameters())});(out/"COMPLETE").touch()


def load_model(run,device):
    ck=torch.load(run/"mlm/lead_time/checkpoint-best.pth",map_location="cpu",weights_only=False);m=LeadTimeSIDMLM(**ck["config"]);m.load_state_dict(ck["model"],strict=True);return m.to(device).eval()


def smoke(run,device):
    out=run/"smoke";out.mkdir(parents=True,exist_ok=False);m,_=initialize(device);m.eval()
    ze=torch.randn(2,256,64,device=device);sid=torch.full((2,256),1024,device=device);sid[:,:25]=torch.randint(0,1024,(2,25),device=device);valid=sid<1024
    lead=torch.zeros(2,256,dtype=torch.long,device=device);lead[:,:25]=torch.tensor([1,2,3,7,8]*5,device=device)
    ti=torch.zeros_like(lead);ti[:,:25]=torch.arange(25,device=device).remainder(10)+1;mask=torch.zeros_like(valid);mask[:,[2,7,13]]=True
    with torch.no_grad():
        a=m(ze,valid,lead,ti,mask);z2=ze.clone();z2[mask]=torch.randn_like(z2[mask])*1e6;b=m(z2,valid,lead,ti,mask)
    delta=float((a-b).abs().max());
    if delta!=0:raise RuntimeError(f"masked z_e leakage {delta}")
    loss=F.cross_entropy(m(ze,valid,lead,ti,mask),sid[mask]);loss.backward()
    e12.dump(out/"audit.json",{"passed":True,"masked_target_mutation_max_delta":delta,"padding_key_mask":True,"lead_time_added_after_mask":True,"finite_backward":bool(torch.isfinite(loss))});(out/"COMPLETE").touch()


@torch.inference_mode()
def fixed_mlm(run,device):
    out=run/"fixed_mask_mlm_comparison.json"
    if out.exists():return
    base=e12.load_continuous_checkpoint(E12_RUN/"mlm/scratch/checkpoint-best.pth",device);new=load_model(run,device)
    sid_np=np.load(e12.PRETRAIN_SID/"val_sid.npy",mmap_mode="r");ze_np=np.load(E12_RUN/"pretrain_ze_cache/val_ze.npy",mmap_mode="r")
    lead_np=np.load(run/"pretrain_meta/val_lead.npy",mmap_mode="r");ti_np=np.load(run/"pretrain_meta/val_time.npy",mmap_mode="r")
    sums={k:{"loss":0.,"correct":0,"tokens":0} for k in ("e14_a","e14_b")}
    for st in range(0,len(sid_np),128):
        en=min(st+128,len(sid_np));s0=np.asarray(sid_np[st:en]);mn=e12.fixed_masks(s0,st)
        s=torch.from_numpy(s0.copy()).long().to(device);mask=torch.from_numpy(mn).to(device);valid=(s>=0)&(s<1024);labels=s[mask]
        ze=torch.from_numpy(np.asarray(ze_np[st:en]).copy()).float().to(device);lead=torch.from_numpy(np.asarray(lead_np[st:en]).copy()).long().to(device);ti=torch.from_numpy(np.asarray(ti_np[st:en]).copy()).long().to(device)
        for k,logits in {"e14_a":base(ze,valid,mask),"e14_b":new(ze,valid,lead,ti,mask)}.items():
            sums[k]["loss"]+=float(F.cross_entropy(logits.float(),labels,reduction="sum"));sums[k]["correct"]+=int((logits.argmax(-1)==labels).sum());sums[k]["tokens"]+=labels.numel()
    e12.dump(out,{k:{"val_ce":v["loss"]/v["tokens"],"masked_sid_accuracy":v["correct"]/v["tokens"],"masked_tokens":v["tokens"]} for k,v in sums.items()}|{"mask_policy":"identical fixed 50% valid-position masks"})


@torch.inference_mode()
def features(run,device):
    out=run/"features/lead_time"
    if (out/"COMPLETE").exists():return
    out.mkdir(parents=True,exist_ok=False);tok=e12.e4.load_model(e12.TOKENIZER,device).eval().requires_grad_(False);m=load_model(run,device).requires_grad_(False);audit={}
    for split in ("train","val","test"):
        wave=np.load(e12.DOWNSTREAM/f"{split}_data.npy",mmap_mode="r");ch=np.load(e12.DOWNSTREAM/f"{split}_data_in_chans.npy",mmap_mode="r");ti=np.load(e12.DOWNSTREAM/f"{split}_data_in_times.npy",mmap_mode="r");res=np.empty((len(wave),1536),np.float32)
        for st in range(0,len(wave),32):
            en=min(st+32,len(wave));x=torch.from_numpy(np.asarray(wave[st:en],dtype=np.float32).copy()).to(device);c=torch.from_numpy(np.asarray(ch[st:en]).copy()).long().to(device);t=torch.from_numpy(np.asarray(ti[st:en]).copy()).long().to(device)
            valid=tok.valid_mask(x,c,t);ze=tok.encode_to_quantizer_features(x,c,t).float();z=m.extract_tokens(ze,valid,c,t).float();count=valid.sum(1).clamp_min(1).unsqueeze(-1);mean=(z*valid.unsqueeze(-1)).sum(1)/count;mx=z.masked_fill(~valid.unsqueeze(-1),-torch.inf).amax(1);mx[~valid.any(1)]=0;res[st:en]=torch.cat([mean,mx],1).cpu().numpy()
        np.save(out/f"{split}_heartlang_mean_max.npy",res);audit[split]={"shape":list(res.shape),"finite":bool(np.isfinite(res).all())}
    e12.dump(out/"audit.json",audit);(out/"COMPLETE").touch()


def train_one(run,seed,device):
    out=run/f"downstream/e14b/seed{seed}"
    if (out/"VAL_COMPLETE").exists():return json.loads((out/"val_summary.json").read_text())
    out.mkdir(parents=True,exist_ok=False);data=e12.load_downstream_arrays(run,"lead_time",device);model=e11.E11RegionResidual(seed).to(device);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05);target=data["train"]["labels"].argmax(1);steps=len(target)//64;gen=torch.Generator().manual_seed(seed);best=-float("inf");best_ep=None;state=None;history=[]
    reg,regmask=e11.masked_region_means(data["train"]["lead"],data["train"]["lead_mask"])
    for ep in range(1,31):
        model.train();order=torch.randperm(len(target),generator=gen).to(device);lm=[];la=[]
        for step in range(steps):
            idx=order[step*64:(step+1)*64];lr=e12.lr_at((ep-1)*steps+step,30*steps,5*steps,5e-4,1e-5,1e-6)
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True);main,aux=model.training_outputs(data["train"]["mlm"][idx],data["train"]["global"][idx],reg[idx],regmask[idx]);l1=F.cross_entropy(main,target[idx]);l2=e12.e9.patient_auxiliary_loss(aux,target[idx]);loss=l1+.2*l2;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();lm.append(float(l1));la.append(float(l2))
        val,_,_=e12.eval_downstream(model,"e14b",data["val"]);history.append({"epoch":ep,"main_ce":float(np.mean(lm)),"patient_ce":float(np.mean(la)),"val":val})
        if val["r2"]>best:best=val["r2"];best_ep=ep;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);val,prob,logits=e12.eval_downstream(model,"e14b",data["val"]);torch.save({"model":{k:v.cpu() for k,v in state.items()},"seed":seed,"best_epoch":best_ep},out/"checkpoint-best.pth");np.save(out/"val_logits.npy",logits);np.save(out/"val_probabilities.npy",prob);e12.dump(out/"history.json",history);res={"seed":seed,"pretrain_seed":0,"best_epoch":best_ep,"val":val};e12.dump(out/"val_summary.json",res);(out/"VAL_COMPLETE").touch();return res


def downstream(run,device):
    out=run/"validation_summary.json"
    if out.exists():return
    rows=[train_one(run,s,device) for s in (0,1,2)];e13=json.loads((E13_RUN/"validation_selection.json").read_text());base=e13["val"]["e13_b"]
    def agg(key):
        vals=[r["val"][key] for r in rows];return {"mean":float(np.mean(vals)),"sd":float(np.std(vals,ddof=1))}
    new={k:agg(k) for k in ("r2","mse","macro_roc_auc","macro_f1","accuracy")};new["per_output_r2"]={n:{"mean":float(np.mean([r["val"]["per_output_r2"][n] for r in rows])),"sd":float(np.std([r["val"]["per_output_r2"][n] for r in rows],ddof=1))} for n in e12.LABEL_NAMES}
    e12.dump(out,{"selection_split":"validation","test_run":False,"e14_a":base,"e14_b":new,"delta":{"r2":new["r2"]["mean"]-base["r2"]["mean"],**{n:new["per_output_r2"][n]["mean"]-base["per_output_r2"][n]["mean"] for n in e12.LABEL_NAMES}},"per_seed":rows})


def audit(run):
    e12.overlap_audit(run)
    ck=torch.load(e12.TOKENIZER,map_location="cpu",weights_only=False);st=ck.get("model",ck);shape=list(st["quantize.embedding.weight"].shape)
    if shape!=[1024,64]:raise RuntimeError(shape)
    e12.dump(run/"audit.json",{"codebook_shape":shape,"e12_checkpoint_read_only":True,"e3_tokenizer_read_only":True,"test_policy":"no new test evaluation"})


def report(run):
    s=json.loads((run/"validation_summary.json").read_text());m=json.loads((run/"fixed_mask_mlm_comparison.json").read_text());a=s["e14_a"];b=s["e14_b"]
    rows=[]
    for metric,label in [("r2","Overall R²")]+[(n,n+" R²") for n in e12.LABEL_NAMES]+[("macro_roc_auc","Macro AUC"),("macro_f1","Macro F1")]:
        av=a["r2"]["mean"] if metric=="r2" else a["per_output_r2"][metric]["mean"] if metric in e12.LABEL_NAMES else a[metric]["mean"]
        bv=b["r2"]["mean"] if metric=="r2" else b["per_output_r2"][metric]["mean"] if metric in e12.LABEL_NAMES else b[metric]["mean"]
        rows.append((label,av,bv,bv-av))
    with (run/"metrics_summary.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["metric","E14-A","E14-B","delta"]);w.writerows(rows)
    improved=b["r2"]["mean"]>a["r2"]["mean"]
    lines=["# E14：Lead + Time Embedding 对照实验","","## 审计","","- E14-A只读复用E12-A/E13-B；E14-B完整训练100 epoch。","- 当前源数据实测9条导联：I、II、III、V1–V6；官方12导联ID中aVR/aVL/aVF未出现在数据里，未伪造。","- Time ID是R峰真实采样位置按100点分箱，不是序列顺序。","- Mask后完整替换z_e，再加入Lead/Time Embedding；显式padding key mask保持启用。","- 未运行新的Test评估。","","## 固定Mask MLM","","|模型|Val CE|Masked SID Accuracy|","|---|---:|---:|",f"|E14-A|{m['e14_a']['val_ce']:.6f}|{m['e14_a']['masked_sid_accuracy']:.6f}|",f"|E14-B|{m['e14_b']['val_ce']:.6f}|{m['e14_b']['masked_sid_accuracy']:.6f}|","","## Validation（三下游seed均值）","","|指标|E14-A|E14-B|差值|","|---|---:|---:|---:|"]
    lines += [f"|{lab}|{av:.6f}|{bv:.6f}|{d:+.6f}|" for lab,av,bv,d in rows]
    lines += ["",f"## 结论","",f"同时加入Lead+Time Embedding{'提高' if improved else '没有提高'} Val Overall R²（{a['r2']['mean']:.6f} → {b['r2']['mean']:.6f}）。",f"OMI差值{s['delta']['OMI']:+.6f}，NOMI差值{s['delta']['NOMI']:+.6f}。", "因此保留E14-B。" if improved else "因此继续保留E14-A/E12-A为最佳模型。"]
    (run/"README.md").write_text("\n".join(lines),encoding="utf-8");(run/"COMPLETE").touch();(run/"status.txt").write_text("complete")


def protocol(run):
    if (run/"protocol.json").exists():return
    e12.dump(run/"protocol.json",{"experiment":"E14","baseline":"read-only E12-A/E13-B","variant":"continuous z_e + lead + real-time embeddings","pretrain_records":[87066,8941],"epochs":100,"batch":256,"pretrain_seed":0,"downstream_seeds":[0,1,2],"mask_ratio":.5,"codebook":[1024,64],"explicit_padding_mask":True,"test_policy":"validation only; no new test"})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["audit","smoke","meta","pretrain","fixed_mlm","features","downstream","report"]);ap.add_argument("--output_dir",type=Path,default=RUN_DEFAULT);ap.add_argument("--epochs",type=int,default=100);args=ap.parse_args();run=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir;run.mkdir(parents=True,exist_ok=True);protocol(run);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage=="audit":audit(run)
    elif args.stage=="smoke":smoke(run,device)
    elif args.stage=="meta":prepare_meta(run)
    elif args.stage=="pretrain":pretrain(run,device,args.epochs)
    elif args.stage=="fixed_mlm":fixed_mlm(run,device)
    elif args.stage=="features":features(run,device)
    elif args.stage=="downstream":downstream(run,device)
    elif args.stage=="report":report(run)

if __name__=="__main__":main()
