#!/usr/bin/env python3
"""E15-Q: quality-aware token weighting on the unchanged E12-A SID MLM.

All quality and SID-reliability decisions are computed once from the frozen E3
tokenizer and the unsupervised training split.  No current-model prediction,
clinical label, or downstream validation result enters the weights.
"""
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
import run_e12_continuous_input as e12
import run_e11_region_residual_fusion as e11
import run_e14_lead_time_embedding as e14

RUN_DEFAULT = ROOT / "results/e15_quality_aware_mlm"
E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
E13_RUN = ROOT / "results/e13_continuous_discrete_ablation"
SEQ = 256
QUALITY_COLUMNS = ["flat_run", "extreme_fraction", "artifact_z", "isolated_jump_ratio", "ptp", "std"]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def source_arrays(split: str):
    codes = np.load(e12.PRETRAIN_INDEX / f"{split}_source_code.npy")
    indices = np.load(e12.PRETRAIN_INDEX / f"{split}_source_index.npy")
    waves = {s: np.load(e12.PRETRAIN_SOURCE / f"{s}_data.npy", mmap_mode="r") for s in ("train", "val")}
    leads = {s: np.load(e12.PRETRAIN_SOURCE / f"{s}_data_in_chans.npy", mmap_mode="r") for s in ("train", "val")}
    return codes, indices, waves, leads


def non_circular_shift(x: torch.Tensor, delta: int) -> torch.Tensor:
    y = torch.zeros_like(x)
    if delta > 0: y[:, delta:] = x[:, :-delta]
    elif delta < 0: y[:, :delta] = x[:, -delta:]
    else: y.copy_(x)
    return y


def waveform_metrics(wave: np.ndarray) -> np.ndarray:
    """Technical-failure proxies; no amplitude-only decision is made."""
    d = np.diff(wave, axis=-1)
    ad = np.abs(d)
    flat = ad < 1e-4
    run = np.zeros(flat.shape[:-1], np.int16); longest = np.zeros_like(run)
    for j in range(flat.shape[-1]):
        run = np.where(flat[..., j], run + 1, 0); longest = np.maximum(longest, run)
    ptp = np.ptp(wave, axis=-1); std = wave.std(axis=-1)
    tol = np.maximum(1e-5, ptp * 5e-4)[..., None]
    hi = wave.max(axis=-1, keepdims=True); lo = wave.min(axis=-1, keepdims=True)
    extreme = ((hi - wave) <= tol) | ((wave - lo) <= tol)
    extreme_fraction = extreme.mean(axis=-1)
    # Exclude the central QRS zone. Require both a large robust jump and an
    # isolated first-vs-second jump before calling a technical spike.
    outside = np.concatenate([ad[..., :31], ad[..., 64:]], axis=-1)
    part = np.partition(outside, -2, axis=-1)
    largest, second = part[..., -1], part[..., -2]
    med = np.median(ad, axis=-1); mad = np.median(np.abs(ad - med[..., None]), axis=-1)
    artifact_z = largest / np.maximum(med + 3.0 * mad, 1e-5)
    isolated = largest / np.maximum(second, 1e-5)
    return np.stack([longest, extreme_fraction, artifact_z, isolated, ptp, std], -1).astype(np.float32)


@torch.inference_mode()
def encode_shift_disagreement(tok, beats: np.ndarray, original_sid: np.ndarray, device: torch.device) -> np.ndarray:
    out = np.zeros(len(beats), np.uint8)
    codebook = F.normalize(tok.quantize.embedding.weight.detach().float().to(device), dim=-1)
    for st in range(0, len(beats), 4096):
        en = min(st + 4096, len(beats)); x = torch.from_numpy(beats[st:en].copy()).float().to(device)
        target = torch.from_numpy(original_sid[st:en].astype(np.int64)).to(device)
        count = torch.zeros(len(x), dtype=torch.uint8, device=device)
        for delta in (-2, -1, 1, 2):
            z = F.normalize(tok.encoder(non_circular_shift(x, delta)).float(), dim=-1)
            count += (z @ codebook.T).argmax(-1).ne(target).to(torch.uint8)
        out[st:en] = count.cpu().numpy()
    return out


def quantile_higher(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x.astype(np.float32), q, method="higher"))


def calibrate(meta: Path) -> dict:
    lead = np.load(meta / "train_lead.npy", mmap_mode="r")
    sid = np.load(e12.PRETRAIN_SID / "train_sid.npy", mmap_mode="r")
    margin = np.load(meta / "train_margin.npy", mmap_mode="r")
    metrics = np.load(meta / "train_quality_metrics.npy", mmap_mode="r")
    valid = (sid >= 0) & (sid < 1024)
    thresholds = {}
    for lid in sorted(int(x) for x in np.unique(lead[valid]) if x > 0):
        m = valid & (lead == lid); q = np.asarray(metrics[m], np.float32); mar = np.asarray(margin[m], np.float32)
        thresholds[str(lid)] = {
            # The source dataset already applied the project's conservative
            # >=50-sample flat-run filter.  Never make E15-Q more aggressive.
            "flat_run": max(50.0, quantile_higher(q[:, 0], .999)),
            "extreme_fraction": max(.12, quantile_higher(q[:, 1], .999)),
            "artifact_z": max(25.0, quantile_higher(q[:, 2], .999)),
            "isolated_jump_ratio": max(3.0, quantile_higher(q[:, 3], .99)),
            "sid_margin": quantile_higher(mar, .005), "tokens": int(m.sum())}
    return thresholds


def assign_risk(meta: Path, split: str, thresholds: dict) -> dict:
    lead = np.load(meta / f"{split}_lead.npy", mmap_mode="r")
    sid = np.load(e12.PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
    margin = np.load(meta / f"{split}_margin.npy", mmap_mode="r")
    shift = np.load(meta / f"{split}_shift_disagree.npy", mmap_mode="r")
    q = np.load(meta / f"{split}_quality_metrics.npy", mmap_mode="r")
    valid = (sid >= 0) & (sid < 1024)
    wave_bad = np.zeros(sid.shape, bool); sid_bad = np.zeros(sid.shape, bool); severity = np.zeros(sid.shape, np.float32)
    for key, th in thresholds.items():
        lid = int(key); m = valid & (lead == lid)
        flat = np.asarray(q[..., 0]) >= th["flat_run"]
        clip = np.asarray(q[..., 1]) >= th["extreme_fraction"]
        spike = (np.asarray(q[..., 2]) >= th["artifact_z"]) & (np.asarray(q[..., 3]) >= th["isolated_jump_ratio"])
        wave_bad[m] = (flat | clip | spike)[m]
        sid_bad[m] = ((np.asarray(margin) <= th["sid_margin"]) & (np.asarray(shift) >= 3))[m]
        severity[m] = np.maximum.reduce([
            np.asarray(q[..., 0], np.float32) / max(th["flat_run"], 1e-6),
            np.asarray(q[..., 1], np.float32) / max(th["extreme_fraction"], 1e-6),
            np.asarray(q[..., 2], np.float32) / max(th["artifact_z"], 1e-6),
            np.maximum(0, th["sid_margin"] - np.asarray(margin, np.float32)) / max(abs(th["sid_margin"]), 1e-4)
        ])[m]
    risk = (wave_bad | sid_bad) & valid
    # A hard cap is a safety bound, not a target. Keep only the clearest risks.
    cap = int(math.floor(.10 * valid.sum()))
    if risk.sum() > cap:
        idx = np.flatnonzero(risk); keep = idx[np.argpartition(severity.ravel()[idx], -cap)[-cap:]]
        capped = np.zeros(risk.size, bool); capped[keep] = True; capped = capped.reshape(risk.shape)
        wave_bad &= capped; sid_bad &= capped; risk = capped
    weights = np.ones(sid.shape, np.float16)
    weights[sid_bad & ~wave_bad] = .7; weights[wave_bad] = .5; weights[~valid] = 0
    reason = np.zeros(sid.shape, np.uint8); reason[sid_bad] |= 1; reason[wave_bad] |= 2
    np.save(meta / f"{split}_target_weight.npy", weights); np.save(meta / f"{split}_risk_reason.npy", reason)
    lead_stats = {str(int(x)): {"tokens": int((valid & (lead == x)).sum()),
                                "downweighted": int((risk & (lead == x)).sum()),
                                "fraction": float((risk & (lead == x)).sum() / max(1, (valid & (lead == x)).sum()))}
                  for x in np.unique(lead[valid]) if x > 0}
    sid_total = np.bincount(np.asarray(sid[valid], np.int64), minlength=1024)
    sid_down = np.bincount(np.asarray(sid[risk], np.int64), minlength=1024)
    eligible = np.flatnonzero(sid_total >= 100)
    sid_rate = np.divide(sid_down, sid_total, out=np.zeros(1024, float), where=sid_total > 0)
    top = eligible[np.argsort(sid_rate[eligible])[-30:][::-1]] if len(eligible) else []
    return {"valid_tokens": int(valid.sum()), "downweighted": int(risk.sum()), "downweight_fraction": float(risk.sum()/valid.sum()),
            "waveform_bad": int((wave_bad & valid).sum()), "sid_unreliable": int((sid_bad & valid).sum()),
            "overlap": int((wave_bad & sid_bad & valid).sum()), "weight_0_5": int((weights == .5).sum()),
            "weight_0_7": int((weights == .7).sum()), "per_lead": lead_stats,
            "highest_downweight_rate_sids_min100": {str(int(x)): {"tokens": int(sid_total[x]), "downweighted": int(sid_down[x]), "fraction": float(sid_rate[x])} for x in top}}


def audit_samples(run: Path, thresholds: dict, summary: dict) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = run / "quality_meta"; rng = np.random.default_rng(150015)
    sid = np.load(e12.PRETRAIN_SID / "train_sid.npy", mmap_mode="r")
    reason = np.load(meta / "train_risk_reason.npy", mmap_mode="r")
    lead = np.load(meta / "train_lead.npy", mmap_mode="r")
    margin = np.load(meta / "train_margin.npy", mmap_mode="r")
    shift = np.load(meta / "train_shift_disagree.npy", mmap_mode="r")
    metrics = np.load(meta / "train_quality_metrics.npy", mmap_mode="r")
    codes, indices, waves, _ = source_arrays("train")
    valid = (sid >= 0) & (sid < 1024); risk_idx = np.flatnonzero((reason > 0) & valid); normal_idx = np.flatnonzero((reason == 0) & valid)
    nr = min(24, len(risk_idx)); nn = min(24, len(normal_idx)); chosen = np.r_[rng.choice(risk_idx, nr, replace=False), rng.choice(normal_idx, nn, replace=False)]
    rec, pos = np.divmod(chosen, SEQ); wave = np.empty((len(chosen), 96), np.float32)
    for j, (r, p) in enumerate(zip(rec, pos)):
        src = "train" if codes[r] == 0 else "val"; wave[j] = waves[src][indices[r], p]
    rows=[]
    fig, axes = plt.subplots(8, 6, figsize=(18, 17), sharex=True)
    for j, ax in enumerate(axes.flat):
        if j >= len(chosen): ax.axis("off"); continue
        k=chosen[j]; r,p=divmod(int(k),SEQ); ax.plot(wave[j],lw=.8); rs=int(reason.ravel()[k])
        ax.set_title(f"{'down' if rs else 'normal'} r{rs} L{int(lead[r,p])} SID{int(sid[r,p])}",fontsize=8); ax.axvspan(32,64,color="orange",alpha=.08)
        rows.append({"flat_index":int(k),"record":r,"token":p,"group":"downweighted" if rs else "normal","reason":rs,
                     "lead":int(lead[r,p]),"sid":int(sid[r,p]),"margin":float(margin[r,p]),"shift_disagree":int(shift[r,p]),
                     **{QUALITY_COLUMNS[c]:float(metrics[r,p,c]) for c in range(len(QUALITY_COLUMNS))}})
    fig.tight_layout(); fig.savefig(run/"quality_audit_samples.png",dpi=160); plt.close(fig)
    with (run/"quality_audit_samples.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Explicit checks against amplitude-only and rarity-based weighting.
    train_weight=np.load(meta/"train_target_weight.npy",mmap_mode="r"); ptp=np.asarray(metrics[...,4],np.float32)
    vptp=ptp[valid]; low_cut=float(np.quantile(vptp,.10)); low=valid&(ptp<=low_cut)
    counts=np.bincount(np.asarray(sid[valid],np.int64),minlength=1024);active=np.flatnonzero(counts);rare_cut=int(np.quantile(counts[active],.25,method="higher"));rare=np.zeros(1024,bool);rare[active[counts[active]<=rare_cut]]=True;rm=valid&rare[np.clip(sid,0,1023)]
    checks={"low_amplitude_decile_downweight_fraction":float((train_weight[low]<1).mean()),"overall_downweight_fraction":summary["train"]["downweight_fraction"],
            "rare_sid_downweight_fraction":float((train_weight[rm]<1).mean()),"rare_sid_frequency_cutoff":rare_cut,
            "amplitude_used_as_rule":False,"sid_frequency_used_as_rule":False,"current_prediction_used_as_rule":False,
            "visual_audit_figure":str(run/"quality_audit_samples.png")}
    dump(run/"quality_audit_checks.json",checks)


@torch.inference_mode()
def prepare_quality_audit(run: Path, device: torch.device) -> None:
    meta=run/"quality_meta"
    if (meta/"COMPLETE").exists(): return
    meta.mkdir(parents=True,exist_ok=False)
    tok=e12.e4.load_model(e12.TOKENIZER,device).eval().requires_grad_(False)
    cb=F.normalize(tok.quantize.embedding.weight.detach().float().to(device),dim=-1)
    for split in ("train","val"):
        ze=np.load(E12_RUN/f"pretrain_ze_cache/{split}_ze.npy",mmap_mode="r");sid=np.load(e12.PRETRAIN_SID/f"{split}_sid.npy",mmap_mode="r")
        codes,indices,waves,leads=source_arrays(split)
        lead_out=np.lib.format.open_memmap(meta/f"{split}_lead.npy",mode="w+",dtype=np.int16,shape=sid.shape)
        margin_out=np.lib.format.open_memmap(meta/f"{split}_margin.npy",mode="w+",dtype=np.float16,shape=sid.shape)
        shift_out=np.lib.format.open_memmap(meta/f"{split}_shift_disagree.npy",mode="w+",dtype=np.uint8,shape=sid.shape)
        # artifact_z can legitimately exceed float16 range when a nearly flat
        # baseline contains one acquisition jump.  Preserve it in float32 so
        # train-derived thresholds remain numerically meaningful.
        q_out=np.lib.format.open_memmap(meta/f"{split}_quality_metrics.npy",mode="w+",dtype=np.float32,shape=sid.shape+(len(QUALITY_COLUMNS),))
        for st in range(0,len(sid),128):
            en=min(st+128,len(sid));c=codes[st:en];ii=indices[st:en];s=np.asarray(sid[st:en]);valid=(s>=0)&(s<1024)
            wave=e12.gather_source(waves,c,ii).astype(np.float32);lead=e12.gather_source(leads,c,ii).astype(np.int16)
            lead_out[st:en]=lead;q=waveform_metrics(wave);q[~valid]=0;q_out[st:en]=q
            mar=np.zeros(s.shape,np.float32);dis=np.zeros(s.shape,np.uint8)
            if valid.any():
                z=torch.from_numpy(np.asarray(ze[st:en])[valid].copy()).float().to(device);sim=F.normalize(z,dim=-1)@cb.T
                top=sim.topk(2,dim=1).values;mar[valid]=(top[:,0]-top[:,1]).cpu().numpy()
                dis[valid]=encode_shift_disagreement(tok,wave[valid],s[valid],device)
            margin_out[st:en]=mar.astype(np.float16);shift_out[st:en]=dis
            if st==0 or en==len(sid) or (st//128)%50==0: print(f"quality_meta {split} {en}/{len(sid)}",flush=True)
        for a in (lead_out,margin_out,shift_out,q_out):a.flush()
    thresholds=calibrate(meta);dump(run/"quality_thresholds.json",thresholds)
    summary={s:assign_risk(meta,s,thresholds) for s in ("train","val")}
    audit_samples(run,thresholds,summary)
    checks=json.loads((run/"quality_audit_checks.json").read_text())
    passed=summary["train"]["downweight_fraction"]<=.10 and checks["low_amplitude_decile_downweight_fraction"]<.02
    dump(run/"quality_audit.json",{"passed":passed,"threshold_source":"unsupervised train split, per measured lead","rules":{"waveform":"very long flat run OR persistent extrema/clipping OR severe isolated off-QRS jump","sid":"bottom 0.5% normalized top1-top2 margin per lead AND >=3/4 SID changes under shifts {-2,-1,+1,+2}","cap":"at most 10%, never filled"},"splits":summary,"checks":checks,"clinical_labels_used":False})
    if not passed: raise RuntimeError("Quality audit failed; inspect audit artifacts before training")
    (meta/"COMPLETE").touch()


class QualityDataset(Dataset):
    def __init__(self,run:Path,split:str):
        self.ze=np.load(E12_RUN/f"pretrain_ze_cache/{split}_ze.npy",mmap_mode="r");self.sid=np.load(e12.PRETRAIN_SID/f"{split}_sid.npy",mmap_mode="r");self.w=np.load(run/f"quality_meta/{split}_target_weight.npy",mmap_mode="r");self.reason=np.load(run/f"quality_meta/{split}_risk_reason.npy",mmap_mode="r")
    def __len__(self):return len(self.sid)
    def __getitem__(self,i):return np.array(self.ze[i],copy=True),np.array(self.sid[i],copy=True),np.array(self.w[i],copy=True),np.array(self.reason[i],copy=True)


def epoch_alpha(epoch:int)->float:
    if epoch<=10:return 0.
    if epoch<=20:return (epoch-10)/10.
    return 1.


def run_epoch(model,loader,device,opt,scaler,epoch,epochs,train):
    model.train(train);keys=("loss","equal_ce","accuracy","down_ce","normal_ce","effective_weight","down_fraction","tokens","batches");z={k:0. for k in keys}
    for step,(ze,sid,target_w,reason) in enumerate(loader):
        ze=ze.float().to(device,non_blocking=True);sid=sid.long().to(device,non_blocking=True);target_w=target_w.float().to(device,non_blocking=True);reason=reason.to(device,non_blocking=True)
        valid=(sid>=0)&(sid<1024);mask=e12.random_mask(valid,.5);labels=sid[mask];tw=target_w[mask];down=reason[mask]>0
        alpha=epoch_alpha(epoch) if train else 0.;w=1.+alpha*(tw-1.)
        if train:
            gs=(epoch-1)*len(loader)+step;lr=e12.lr_at(gs,epochs*len(loader),5*len(loader),5e-4,1e-5,1e-6)
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
            logits=model(ze,valid,mask);ce=F.cross_entropy(logits.float(),labels,reduction="none");loss=(w*ce).sum()/w.sum().clamp_min(1)
        if train:
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),3.);scaler.step(opt);scaler.update()
        z["loss"]+=float(loss);z["equal_ce"]+=float(ce.mean());z["accuracy"]+=float((logits.argmax(-1)==labels).float().mean());z["down_ce"]+=float(ce[down].mean()) if down.any() else 0.;z["normal_ce"]+=float(ce[~down].mean()) if (~down).any() else 0.;z["effective_weight"]+=float(w.mean());z["down_fraction"]+=float(down.float().mean());z["tokens"]+=labels.numel();z["batches"]+=1
        if step==0 or (step+1)%50==0 or step+1==len(loader):print(f"{'train' if train else 'val'} ep={epoch} {step+1}/{len(loader)} loss={float(loss):.6f} alpha={alpha:.2f}",flush=True)
    n=max(1,z["batches"]);return {k:(v/n if k not in ("tokens","batches") else v) for k,v in z.items()}


def pretrain(run:Path,device:torch.device,epochs:int):
    out=run/"mlm/quality"
    if (out/"COMPLETE").exists():return
    if not json.loads((run/"quality_audit.json").read_text())["passed"]:raise RuntimeError("audit not passed")
    out.mkdir(parents=True,exist_ok=False);model,init=e12.initialize_continuous("scratch",device);dump(out/"initialization_audit.json",init)
    trds=QualityDataset(run,"train");vads=QualityDataset(run,"val");gen=torch.Generator().manual_seed(0)
    tr=DataLoader(trds,batch_size=256,shuffle=True,generator=gen,num_workers=8,pin_memory=True,drop_last=True);va=DataLoader(vads,batch_size=384,shuffle=False,num_workers=8,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05);scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda");best=float("inf");bestep=None;hist=[];start=time.time()
    for ep in range(1,epochs+1):
        ts=run_epoch(model,tr,device,opt,scaler,ep,epochs,True);row={"epoch":ep,"transition_alpha":epoch_alpha(ep),**{f"train_{k}":v for k,v in ts.items()}}
        if ep==1 or ep%10==0 or ep==epochs:
            with torch.no_grad():vs=run_epoch(model,va,device,None,None,ep,epochs,False)
            row.update({f"val_{k}":v for k,v in vs.items()})
            if vs["equal_ce"]<best:
                best=vs["equal_ce"];bestep=ep;torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":e12.model_config(),"epoch":ep,"best_val_loss":best},out/"checkpoint-best.pth")
        hist.append(row);dump(out/"training_history.json",hist);print(json.dumps(row),flush=True)
        if ep in (10,20,50,epochs):torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":e12.model_config(),"epoch":ep},out/f"checkpoint-{ep}.pth")
    dump(out/"summary.json",{"best_epoch":bestep,"best_val_equal_ce":best,"elapsed_seconds":time.time()-start,"epochs":epochs,"pretrain_seed":0,"trainable_parameters":sum(p.numel() for p in model.parameters()),"loss":"weighted sum CE / sum weight","transition":"ep1-10 all1; ep11-20 linear; ep21-100 target"});(out/"COMPLETE").touch()


@torch.inference_mode()
def fixed_mlm(run:Path,device:torch.device):
    out=run/"fixed_mask_mlm_comparison.json"
    if out.exists():return
    models={"e12_a":e12.load_continuous_checkpoint(E12_RUN/"mlm/scratch/checkpoint-best.pth",device),"e15_q":e12.load_continuous_checkpoint(run/"mlm/quality/checkpoint-best.pth",device)}
    sidnp=np.load(e12.PRETRAIN_SID/"val_sid.npy",mmap_mode="r");zenp=np.load(E12_RUN/"pretrain_ze_cache/val_ze.npy",mmap_mode="r");sums={k:{"loss":0.,"correct":0,"tokens":0} for k in models}
    for st in range(0,len(sidnp),128):
        en=min(st+128,len(sidnp));s0=np.asarray(sidnp[st:en]);mn=e12.fixed_masks(s0,st);sid=torch.from_numpy(s0.copy()).long().to(device);mask=torch.from_numpy(mn).to(device);valid=(sid>=0)&(sid<1024);lab=sid[mask];ze=torch.from_numpy(np.asarray(zenp[st:en]).copy()).float().to(device)
        for k,m in models.items():
            logits=m(ze,valid,mask);sums[k]["loss"]+=float(F.cross_entropy(logits.float(),lab,reduction="sum"));sums[k]["correct"]+=int((logits.argmax(-1)==lab).sum());sums[k]["tokens"]+=lab.numel()
    dump(out,{k:{"val_ce":v["loss"]/v["tokens"],"masked_sid_accuracy":v["correct"]/v["tokens"],"masked_tokens":v["tokens"]} for k,v in sums.items()}|{"mask_policy":"identical fixed 50% valid-token mask; all positions equally evaluated"})


def features(run:Path,device:torch.device):e12.extract_downstream(run,"quality",device,32)


def train_one(run:Path,seed:int,device:torch.device)->dict:
    out=run/f"downstream/e15q/seed{seed}"
    if (out/"VAL_COMPLETE").exists():return json.loads((out/"val_summary.json").read_text())
    out.mkdir(parents=True,exist_ok=False);data=e12.load_downstream_arrays(run,"quality",device);model=e11.E11RegionResidual(seed).to(device);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05);target=data["train"]["labels"].argmax(1);steps=len(target)//64;gen=torch.Generator().manual_seed(seed);best=-float("inf");bestep=None;state=None;hist=[];reg,regmask=e11.masked_region_means(data["train"]["lead"],data["train"]["lead_mask"])
    for ep in range(1,31):
        model.train();order=torch.randperm(len(target),generator=gen).to(device);lm=[];la=[]
        for step in range(steps):
            idx=order[step*64:(step+1)*64];lr=e12.lr_at((ep-1)*steps+step,30*steps,5*steps,5e-4,1e-5,1e-6)
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True);main,aux=model.training_outputs(data["train"]["mlm"][idx],data["train"]["global"][idx],reg[idx],regmask[idx]);l1=F.cross_entropy(main,target[idx]);l2=e12.e9.patient_auxiliary_loss(aux,target[idx]);loss=l1+.2*l2;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();lm.append(float(l1));la.append(float(l2))
        val,_,_=e12.eval_downstream(model,"e15q",data["val"]);hist.append({"epoch":ep,"main_ce":float(np.mean(lm)),"patient_ce":float(np.mean(la)),"val":val})
        if val["r2"]>best:best=val["r2"];bestep=ep;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);val,prob,logits=e12.eval_downstream(model,"e15q",data["val"]);torch.save({"model":{k:v.cpu() for k,v in state.items()},"seed":seed,"best_epoch":bestep},out/"checkpoint-best.pth");np.save(out/"val_logits.npy",logits);np.save(out/"val_probabilities.npy",prob);dump(out/"history.json",hist);res={"seed":seed,"pretrain_seed":0,"best_epoch":bestep,"val":val};dump(out/"val_summary.json",res);(out/"VAL_COMPLETE").touch();return res


def downstream(run:Path,device:torch.device):
    out=run/"validation_summary.json"
    if out.exists():return
    rows=[train_one(run,s,device) for s in (0,1,2)];base=json.loads((E13_RUN/"validation_selection.json").read_text())["val"]["e13_b"]
    def agg(k):
        x=[r["val"][k] for r in rows];return {"mean":float(np.mean(x)),"sd":float(np.std(x,ddof=1))}
    new={k:agg(k) for k in ("r2","mse","mae","macro_roc_auc","macro_f1","accuracy")};new["per_output_r2"]={n:{"mean":float(np.mean([r["val"]["per_output_r2"][n] for r in rows])),"sd":float(np.std([r["val"]["per_output_r2"][n] for r in rows],ddof=1))} for n in e12.LABEL_NAMES};dump(out,{"selection_split":"validation","test_run":False,"e12_a":base,"e15_q":new,"delta":{"r2":new["r2"]["mean"]-base["r2"]["mean"],**{n:new["per_output_r2"][n]["mean"]-base["per_output_r2"][n]["mean"] for n in e12.LABEL_NAMES}},"per_seed":rows})


def report(run:Path):
    s=json.loads((run/"validation_summary.json").read_text());m=json.loads((run/"fixed_mask_mlm_comparison.json").read_text());a=s["e12_a"];b=s["e15_q"];qa=json.loads((run/"quality_audit.json").read_text());metrics=[("Overall R²",a["r2"]["mean"],b["r2"]["mean"])]+[(n+" R²",a["per_output_r2"][n]["mean"],b["per_output_r2"][n]["mean"]) for n in e12.LABEL_NAMES]+[("Macro AUC",a["macro_roc_auc"]["mean"],b["macro_roc_auc"]["mean"]),("Macro F1",a["macro_f1"]["mean"],b["macro_f1"]["mean"])]
    with (run/"validation_comparison.csv").open("w",newline="") as f:w=csv.writer(f);w.writerow(["metric","E12-A","E15-Q","delta"]);w.writerows((n,x,y,y-x) for n,x,y in metrics)
    lines=["# E15-Q：质量感知 SID 监督实验","","## 质量审计","",f"- 训练有效心拍降权比例：{qa['splits']['train']['downweight_fraction']:.6f}。",f"- 明确波形技术异常：{qa['splits']['train']['waveform_bad']}；SID不可靠：{qa['splits']['train']['sid_unreliable']}；重叠：{qa['splits']['train']['overlap']}。","- 波形规则只使用长平直、持续极值/截幅、严重孤立且位于QRS窗口外的跳点；低幅、SID稀有、当前模型难度均不作为规则。","- SID仅在原型间隔处于各导联训练分布最低0.5%，且±1/±2轻移中至少3次改变SID时降权。","- 抽样图：`quality_audit_samples.png`；样本清单：`quality_audit_samples.csv`。","","## 固定Mask MLM","","|模型|Val CE|Masked SID Accuracy|","|---|---:|---:|",f"|E12-A|{m['e12_a']['val_ce']:.6f}|{m['e12_a']['masked_sid_accuracy']:.6f}|",f"|E15-Q|{m['e15_q']['val_ce']:.6f}|{m['e15_q']['masked_sid_accuracy']:.6f}|","","## Validation（三个下游seed，均值）","","|指标|E12-A|E15-Q|差值|","|---|---:|---:|---:|"]+[f"|{n}|{x:.6f}|{y:.6f}|{y-x:+.6f}|" for n,x,y in metrics]+["","未运行Test；历史checkpoint只读。","",f"结论：Overall R²变化 {b['r2']['mean']-a['r2']['mean']:+.6f}，OMI {s['delta']['OMI']:+.6f}，NOMI {s['delta']['NOMI']:+.6f}。"]
    (run/"README.md").write_text("\n".join(lines),encoding="utf-8");(run/"COMPLETE").touch();(run/"status.txt").write_text("complete")


def smoke(run:Path,device:torch.device):
    out=run/"smoke";out.mkdir(parents=True,exist_ok=False);model,_=e12.initialize_continuous("scratch",device);model.eval();ze=torch.randn(3,256,64,device=device);sid=torch.full((3,256),1024,dtype=torch.long,device=device);sid[:,:20]=torch.randint(0,1024,(3,20),device=device);valid=sid<1024;mask=torch.zeros_like(valid);mask[:,[2,5,9,13,17]]=True
    with torch.no_grad():a=model(ze,valid,mask);z2=ze.clone();z2[mask]=torch.randn_like(z2[mask])*1e6;b=model(z2,valid,mask)
    tw=torch.tensor([.5,.7,1.,1.,1.]*3,device=device);ce=F.cross_entropy(model(ze,valid,mask),sid[mask],reduction="none");vals={e:(1+epoch_alpha(e)*(tw-1)).tolist() for e in (10,11,20,21)};loss=(tw*ce).sum()/tw.sum();loss.backward();passed=float((a-b).abs().max())==0 and vals[10]==[1.]*15 and abs(vals[20][0]-.5)<1e-6 and torch.isfinite(loss)
    dump(out/"audit.json",{"passed":bool(passed),"masked_z_e_mutation_delta":float((a-b).abs().max()),"weights_detached":not tw.requires_grad,"transition":vals,"finite_backward":bool(torch.isfinite(loss)),"padding_excluded_by_mask":True});
    if not passed:raise RuntimeError("smoke failed")
    (out/"COMPLETE").touch()


def protocol(run:Path):
    if (run/"protocol.json").exists():return
    dump(run/"protocol.json",{"experiment":"E15-Q","baseline":"read-only E12-A/E13-B","architecture":"unchanged E12-A","only_change":"per-token MLM loss weights","pretrain_records":[87066,8941],"epochs":100,"batch":256,"pretrain_seed":0,"downstream_seeds":[0,1,2],"mask_ratio":.5,"weights":{"normal":1.,"sid_unreliable":.7,"waveform_bad":.5,"both":.5},"schedule":"ep1-10 all1; ep11-20 linear transition; ep21-100 target","validation":"fixed mask, all positions equal","test_policy":"no test","random_control":"not run unless main experiment completes and resources remain"})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["audit","smoke","pretrain","fixed_mlm","features","downstream","report"]);ap.add_argument("--output_dir",type=Path,default=RUN_DEFAULT);ap.add_argument("--epochs",type=int,default=100);args=ap.parse_args();run=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir;run.mkdir(parents=True,exist_ok=True);protocol(run);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage=="audit":e12.overlap_audit(run);prepare_quality_audit(run,device)
    elif args.stage=="smoke":smoke(run,device)
    elif args.stage=="pretrain":pretrain(run,device,args.epochs)
    elif args.stage=="fixed_mlm":fixed_mlm(run,device)
    elif args.stage=="features":features(run,device)
    elif args.stage=="downstream":downstream(run,device)
    elif args.stage=="report":report(run)


if __name__=="__main__":main()
