"""E30 objective comparison. Run beside the original E12 runner or add it to PYTHONPATH."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import run_e12_continuous_input as e

SOURCE = e.DEFAULT_RUN
OUT = e.ROOT / 'results/E30_SID_utility'
CACHE = SOURCE / 'pretrain_ze_cache'
SEEDS = (0, 1, 2)
DEVICE = torch.device('cuda')


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def model_for(objective, seed):
    model, audit = e.initialize_continuous('scratch', DEVICE)
    if seed != 0:
        torch.manual_seed(seed)
        fresh = e.ContinuousSIDMLM(**e.model_config()).to(DEVICE)
        model.load_state_dict(fresh.state_dict())
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(30000 + seed)
        model.ze_head = nn.Linear(768, 64).to(DEVICE)
    if objective == 'B':
        del model.lm_head
    return model, audit


def loss_terms(model, objective, ze, sid, valid, mask, mean, scale):
    mask = mask & valid
    hidden = model.backbone.forward_feature(ze, valid, mask)[:, 1:][mask]
    ce = acc = mse = cosine = pearson = None
    if objective != 'B':
        logits = model.lm_head(hidden).float()
        ce = F.cross_entropy(logits, sid[mask])
        acc = (logits.argmax(-1) == sid[mask]).float().mean()
    if objective != 'A':
        pred = model.ze_head(hidden).float()
        target = (ze.detach()[mask] - mean) / scale
        mse = F.mse_loss(pred, target)
        with torch.no_grad():
            cosine = F.cosine_similarity(pred, target).mean()
            p = pred.flatten() - pred.mean()
            t = target.flatten() - target.mean()
            pearson = F.cosine_similarity(p, t, dim=0)
    return ce, acc, mse, cosine, pearson


def statistics():
    path = OUT / 'target_stats.npz'
    if not path.exists():
        ds = e.ZEDataset(CACHE, 'train')
        total = np.zeros(64)
        squared = np.zeros(64)
        count = 0
        for start in range(0, len(ds), 256):
            sid = ds.sid[start:start+256]
            z = ds.ze[start:start+256][(sid >= 0) & (sid < 1024)].astype(np.float64)
            total += z.sum(0)
            squared += (z*z).sum(0)
            count += len(z)
        mean = total / count
        np.savez(path, mean=mean, scale=np.sqrt(np.maximum(squared/count-mean**2, 1e-12)), count=count)
    stats = np.load(path)
    return tuple(torch.as_tensor(stats[k], device=DEVICE, dtype=torch.float32) for k in ('mean','scale'))


def smoke():
    a, _ = model_for('A', 0)
    b, _ = model_for('B', 0)
    c, _ = model_for('C', 0)
    h = [e.state_sha(model.backbone.state_dict()) for model in (a,b,c)]
    assert len(set(h)) == 1
    for model in (a,b,c):
        model.eval()
    sid = torch.full((2,256),1024,device=DEVICE)
    sid[:,:8] = torch.arange(8,device=DEVICE)
    valid = sid < 1024
    ze = torch.randn((2,256,64),device=DEVICE)
    masks = [e.random_mask(valid,.5,torch.Generator(device=DEVICE).manual_seed(7)) for _ in (a,b,c)]
    assert all(torch.equal(masks[0], m) for m in masks)
    mask = masks[0]
    changed = ze.clone()
    changed[mask | ~valid] = 100
    with torch.no_grad():
        for model in (a,b,c):
            torch.testing.assert_close(model.backbone.forward_feature(ze,valid,mask),
                model.backbone.forward_feature(changed,valid,mask), rtol=0,atol=0)
    m = torch.zeros(64,device=DEVICE)
    s = torch.ones_like(m)
    ce,_,mse,_,_ = loss_terms(c,'C',ze.requires_grad_(),sid,valid,mask | ~valid,m,s)
    (ce+mse).backward()
    assert ze.grad[mask | ~valid].count_nonzero()==0
    assert c.ze_head.weight.grad.norm()>0 and c.lm_head.weight.grad.norm()>0
    e.dump(OUT/'smoke.json',dict(initial_backbone_sha256=h[0],mask_alignment=True,
        masked_input_invariance=True,padding_excluded=True,both_heads_receive_gradient=True))


def calibration(model, ds, stats):
    model.eval()
    norms = []
    params = tuple(model.backbone.parameters())
    for offset in range(0, 128, 32):
        ze = torch.tensor(np.array(ds.ze[offset:offset+32]),device=DEVICE).float()
        sid = torch.tensor(np.array(ds.sid[offset:offset+32]),device=DEVICE).long()
        valid = (sid >= 0) & (sid < 1024)
        mask = e.random_mask(valid,.5,torch.Generator(device=DEVICE).manual_seed(31000+offset))
        ce,_,mse,_,_ = loss_terms(model,'C',ze,sid,valid,mask,*stats)
        values = []
        for loss, retain in [(ce,True),(mse,False)]:
            grads = torch.autograd.grad(loss,params,retain_graph=retain)
            values.append(math.sqrt(sum(float(g.float().square().sum()) for g in grads)))
        norms.append(values)
    avg = np.mean(norms,axis=0)
    weight = float(avg[0]/avg[1])
    assert math.isfinite(weight) and weight > 0
    return weight,norms


def pretrain(objective, seed):
    folder = OUT / 'checkpoints' / f'{objective}_seed{seed}'
    if (folder/'COMPLETE').exists():
        return
    folder.mkdir(parents=True,exist_ok=True)
    model, init = model_for(objective, seed)
    # Reuse the E12 initialization RNG for epoch 1 with seed 0.
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = (e.ZEDataset(CACHE, split) for split in ('train','val'))
    train = DataLoader(train_ds,batch_size=256,shuffle=True,generator=gen,num_workers=8,pin_memory=True,drop_last=True)
    val = DataLoader(val_ds,batch_size=384,num_workers=8,pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda')
    stats = statistics()
    fixed = e.fixed_masks(val_ds.sid,0)
    history = []
    elapsed = 0.
    best = float('inf')
    resume = folder/'checkpoint-last.pth'
    if resume.exists():
        state = torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        gen.set_state(state['loader_rng'])
        history,best,weight,elapsed = (state[k] for k in ('history','best','weight','elapsed'))
    else:
        weight,norms = calibration(model,train_ds,stats) if objective == 'C' else (1.,[])
        e.dump(folder/'config.json',dict(objective=objective,seed=seed,epochs=100,batch=256,
            val_batch=384,optimizer='AdamW',weight_decay=.05,clip=3,amp='fp16',
            peak_lr=.0005,min_lr=.00001,warmup_lr=.000001,warmup_epochs=5,
            steps_per_epoch=len(train),lambda_ze=weight,calibration_grad_norms=norms,
            backbone_init_sha256=e.state_sha(model.backbone.state_dict()),initialization=init,
            mask_ratio=.5,validation_mask_seed_base=20261018,val_mask_sha256=hashlib.sha256(fixed.tobytes()).hexdigest(),
            selection='minimum fixed-mask validation objective on epochs 1,10,...,100',
            parameters=sum(p.numel() for p in model.parameters())))
    start = time.monotonic()
    for epoch in range(len(history)+1,101):
        row = dict(epoch=epoch)
        for training, loader in [(True,train),(False,val)]:
            if not training and epoch != 1 and epoch%10:
                continue
            model.train(training)
            totals = np.zeros(6)
            batches = 0
            # Validation must not advance the training RNG stream.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()],enabled=not training):
                for step,(ze,sid) in enumerate(loader):
                    ze = ze.to(DEVICE,non_blocking=True).float()
                    sid = sid.to(DEVICE,non_blocking=True).long()
                    valid = (sid>=0)&(sid<1024)
                    # Use an explicit step generator so validation and GradScaler cannot perturb masks.
                    if training:
                        mask_gen = torch.Generator(device=DEVICE).manual_seed(seed*1000000+(epoch-1)*len(train)+step)
                        mask = e.random_mask(valid,.5,mask_gen)
                    else:
                        mask = torch.from_numpy(fixed[step*384:step*384+len(sid)]).to(DEVICE)
                    if not bool(mask.any()):
                        continue
                    if training:
                        lr = e.lr_at((epoch-1)*len(train)+step,100*len(train),5*len(train),5e-4,1e-5,1e-6)
                        opt.param_groups[0]['lr'] = lr
                        opt.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(training), torch.amp.autocast('cuda'):
                        ce,acc,mse,cos,pearson = loss_terms(model,objective,ze,sid,valid,mask,*stats)
                        loss = mse if objective=='B' else ce+weight*mse
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'{objective} epoch {epoch} batch {step}: nonfinite loss')
                    if training:
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(),3.)
                        scaler.step(opt)
                        scaler.update()
                    totals += [float(x.detach()) if x is not None else 0. for x in (loss,ce,acc,mse,cos,pearson)]
                    batches += 1
                    if step%50==0:
                        print(objective,epoch,'train' if training else 'val',step,len(loader),float(loss.detach()),flush=True)
            keys = ('loss','sid_ce','sid_accuracy','ze_mse','ze_cosine','ze_pearson_flat')
            row['train' if training else 'val'] = dict(zip(keys,(totals/batches).tolist()))
            current = row['train' if training else 'val']
            current['ze_rmse'] = math.sqrt(current['ze_mse'])
            if objective == 'B':
                current['sid_ce'] = current['sid_accuracy'] = None
        history.append(row)
        state = dict(model=model.state_dict(),epoch=epoch, config=e.model_config())
        if 'val' in row and row['val']['loss'] < best:
            best = row['val']['loss']
            save(folder/'checkpoint-best.pth',state)
        if epoch==100:
            save(folder/'checkpoint-final.pth',state)
        save(resume,state | dict(optimizer=opt.state_dict(),scaler=scaler.state_dict(),
            history=history,best=best,weight=weight,elapsed=elapsed+time.monotonic()-start,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),loader_rng=gen.get_state()))
        e.dump(folder/'history.json',history)
        print(json.dumps(row),flush=True)
    e.dump(folder/'summary.json',dict(seconds=elapsed+time.monotonic()-start,
        best_epoch=min((r for r in history if 'val' in r),key=lambda r:r['val']['loss'])['epoch'],
        scheduled_steps=100*len(train),checkpoint_sha256=e.sha256(folder/'checkpoint-best.pth')))
    (folder/'COMPLETE').touch()


@torch.no_grad()
def extract(objective, seed, split, target):
    path=target/f'{split}_heartlang_mean_max.npy'
    if path.exists():
        return
    model,_=model_for(objective,seed)
    ck = SOURCE/'mlm/scratch/checkpoint-best.pth' if objective=='A' else OUT/f'checkpoints/{objective}_seed{seed}/checkpoint-best.pth'
    model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model'],strict=objective!='A')
    model.eval().requires_grad_(False)
    tokenizer=e.e4.load_model(e.TOKENIZER,DEVICE).eval().requires_grad_(False)
    w,c,t=[np.load(e.DOWNSTREAM/f'{split}_{k}.npy',mmap_mode='r') for k in ('data','data_in_chans','data_in_times')]
    rows=[]
    for start in range(0,len(w),32):
        x,ch,tm=[torch.from_numpy(np.array(a[start:start+32])).to(DEVICE) for a in (w,c,t)]
        valid=tokenizer.valid_mask(x,ch,tm)
        ze=tokenizer.encode_to_quantizer_features(x,ch.long(),tm.long()).float()
        h=model.extract_tokens(ze,valid)
        mx=h.masked_fill(~valid[...,None],-torch.inf).amax(1)
        mx[~valid.any(1)]=0
        rows.append(torch.cat(((h*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1),mx),1).cpu().numpy())
    target.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npy')
    np.save(temp,np.concatenate(rows))
    os.replace(temp,path)


def arrays(target,split):
    get=lambda p:torch.from_numpy(np.load(p)).to(DEVICE)
    return dict(labels=get(e.DOWNSTREAM/f'{split}_labels.npy').float(),
        mlm=get(target/f'{split}_heartlang_mean_max.npy').float(),
        lead=get(e.E10_RUN/f'features/{split}_lead_ze_mean_max.npy').float(),
        lead_mask=get(e.E10_RUN/f'features/{split}_lead_present.npy').bool(),
        global_=get(e.E10_RUN/f'features/{split}_global_ze_mean_max.npy').float())


def eval_data(target,split):
    result=arrays(target,split)
    result['global']=result.pop('global_')
    return result


def downstream(seed):
    targets={}
    for obj in ('A','B','C'):
        folder=OUT/f'downstream/{obj}_preseed{seed}'
        target=folder/'features/scratch'
        targets[obj]=(folder,target)
        original_loader=e.load_downstream_arrays
        def training_only(run_dir,mode,device):
            return {split:eval_data(target,split) for split in ('train','val')}
        e.load_downstream_arrays=training_only
        try:
            for split in ('train','val'):
                extract(obj,seed,split,target)
            for method in ('e12a_region','e12a_mlm_only'):
                for ds_seed in SEEDS:
                    d=folder/f'downstream/{method}/seed{ds_seed}'
                    if d.exists() and not (d/'VAL_COMPLETE').exists():
                        d.rename(d.with_name(f'{d.name}.interrupted.{time.time_ns()}'))
                    e.train_downstream_one(folder,method,ds_seed,SimpleNamespace(),DEVICE)
        finally:
            e.load_downstream_arrays=original_loader
    # Do not read downstream Test features or labels until all 18 selections are frozen.
    selected={str(p):e.sha256(p) for folder,_ in targets.values()
        for p in folder.glob('downstream/*/seed*/checkpoint-best.pth')}
    assert len(selected)==18
    lock=OUT/f'selection_lock_preseed{seed}.json'
    if lock.exists():
        assert json.loads(lock.read_text())==selected
    else:
        e.dump(lock,selected)
    for obj,(folder,target) in targets.items():
        extract(obj,seed,'test',target)
        test=eval_data(target,'test')
        for method in ('e12a_region','e12a_mlm_only'):
            for ds_seed in SEEDS:
                d=folder/f'downstream/{method}/seed{ds_seed}'
                model=e.build_downstream(method,ds_seed,DEVICE)
                model.load_state_dict(torch.load(d/'checkpoint-best.pth',map_location='cpu',weights_only=False)['model'])
                result,prob,_=e.eval_downstream(model,method,test)
                np.save(d/'test_probabilities.npy',prob)
                e.dump(d/'test_metrics.json',result)
    (OUT/f'DOWNSTREAM_COMPLETE_preseed{seed}').touch()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['smoke','B','C','downstream','all'])
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.seed!=0 and args.stage in ('downstream','all'):
        parser.error('historical A is available for pretraining seed 0 only')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.stage in ('smoke','all'):
        smoke()
    try:
        for obj in ('B','C'):
            if args.stage in (obj,'all'):
                pretrain(obj,args.seed)
        if args.stage in ('downstream','all'):
            downstream(args.seed)
    except torch.cuda.OutOfMemoryError:
        print('CUDA OOM, no reduction in batch size applied',flush=True)
        raise


if __name__=='__main__':
    main()
