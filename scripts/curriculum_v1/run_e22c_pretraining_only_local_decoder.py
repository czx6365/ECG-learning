#!/usr/bin/env python3
"""E22-C: pretraining-only three-beat local SID decoder.

The E12-A encoder path remains unchanged. Each original E3 continuous z_e is one
token with its original SID target. A lightweight decoder uses the exact previous
and next heartbeat from the same measured lead only for an auxiliary masked-SID
loss. The auxiliary weight is zero for epochs 1-10, ramps to 0.1 in epochs 11-20,
and remains 0.1. The decoder is discarded for downstream feature extraction.
No lead/time embedding, text, clinical label, or Test evaluation is used.
"""
from __future__ import annotations
import argparse, copy, csv, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'heartlang').exists())
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/experiments'),str(ROOT/'scripts/curriculum_v1')]
import run_e12_continuous_input as e12
import run_e22_three_beat_grouped_sid as olde22

DEFAULT_RUN=ROOT/'results/e22c_pretraining_only_local_decoder'
E12_RUN=e12.DEFAULT_RUN
E22B_RUN=ROOT/'results/e22b_three_beat_local_context'
LOCAL_HEADS=4; LOCAL_DIM_HEAD=32; LOCAL_AUX_MAX=0.1

def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def set_status(run,value):run.mkdir(parents=True,exist_ok=True);(run/'status.txt').write_text(value+'\n');print(value,flush=True)

def make_neighbors(chans:np.ndarray,times:np.ndarray,valid:np.ndarray)->np.ndarray:
    """Return [B,256,3] indices for previous/self/next in same-lead time order."""
    out=np.full((len(chans),chans.shape[1],3),-1,dtype=np.int16)
    for b in range(len(chans)):
        ids=np.flatnonzero(valid[b])
        for lead in np.unique(chans[b,ids]):
            q=ids[chans[b,ids]==lead]
            q=q[np.argsort(times[b,q],kind='stable')]
            out[b,q,1]=q
            if len(q)>1:
                out[b,q[1:],0]=q[:-1];out[b,q[:-1],2]=q[1:]
    return out

def build_context_cache(run:Path)->None:
    out=run/'context_cache'
    if (out/'COMPLETE').exists():return
    out.mkdir(parents=True,exist_ok=False)
    chan_src={s:np.load(e12.PRETRAIN_SOURCE/f'{s}_data_in_chans.npy',mmap_mode='r') for s in ('train','val')}
    time_src={s:np.load(e12.PRETRAIN_SOURCE/f'{s}_data_in_times.npy',mmap_mode='r') for s in ('train','val')}
    audit={'definition':'previous/self/next valid heartbeat by real R-peak time within same measured lead','cross_lead_neighbors':0,'splits':{}}
    for split in ('train','val'):
        codes=np.load(e12.PRETRAIN_INDEX/f'{split}_source_code.npy');indices=np.load(e12.PRETRAIN_INDEX/f'{split}_source_index.npy')
        sid=np.load(e12.PRETRAIN_SID/f'{split}_sid.npy',mmap_mode='r');dest=np.lib.format.open_memmap(out/f'{split}_neighbors.npy',mode='w+',dtype=np.int16,shape=(len(sid),256,3))
        total=full=prev=nextn=pairs=noncontig=bad_time=cross=0
        for start in range(0,len(sid),1024):
            end=min(start+1024,len(sid));c=codes[start:end];ii=indices[start:end]
            ch=e12.gather_source(chan_src,c,ii).astype(np.int64);ti=e12.gather_source(time_src,c,ii).astype(np.int64)
            valid=(np.asarray(sid[start:end])>=0)&(np.asarray(sid[start:end])<1024);nei=make_neighbors(ch,ti,valid);dest[start:end]=nei
            total+=int(valid.sum());prev+=int((nei[:,:,0]>=0).sum());nextn+=int((nei[:,:,2]>=0).sum());full+=int(((nei[:,:,0]>=0)&(nei[:,:,2]>=0)).sum())
            for b in range(len(ch)):
                for lead in np.unique(ch[b,valid[b]]):
                    q=np.flatnonzero(valid[b]& (ch[b]==lead));q=q[np.argsort(ti[b,q],kind='stable')]
                    if len(q)>1:
                        pairs+=len(q)-1;noncontig+=int(np.sum(np.diff(q)!=1));bad_time+=int(np.sum(np.diff(ti[b,q])<0))
                for slot in (0,2):
                    q=np.flatnonzero(nei[b,:,slot]>=0)
                    if len(q):cross+=int(np.sum(ch[b,q]!=ch[b,nei[b,q,slot]]))
            print(f'context {split} {end}/{len(sid)}',flush=True)
        dest.flush();del dest
        audit['cross_lead_neighbors']+=cross
        audit['splits'][split]={'records':len(sid),'valid_tokens':total,'tokens_with_previous':prev,'tokens_with_next':nextn,'tokens_with_full_triplet':full,'full_triplet_fraction':full/max(1,total),'same_lead_consecutive_pairs':pairs,'noncontiguous_array_pairs':noncontig,'noncontiguous_array_pair_fraction':noncontig/max(1,pairs),'negative_time_steps':bad_time,'cross_lead_neighbors':cross,'neighbors_sha256':e12.sha256(out/f'{split}_neighbors.npy')}
    if audit['cross_lead_neighbors'] or any(x['negative_time_steps'] for x in audit['splits'].values()):raise RuntimeError(f'context audit failed: {audit}')
    dump(out/'audit.json',audit);(out/'COMPLETE').touch()

class PretrainingOnlyLocalDecoder(nn.Module):
    """Attend to exact previous/next same-lead hidden states; never rewrites encoder tokens."""
    def __init__(self,embed_dim=768,heads=LOCAL_HEADS,dim_head=LOCAL_DIM_HEAD,dropout=.1):
        super().__init__();self.heads=heads;self.dim_head=dim_head;inner=heads*dim_head
        self.norm=nn.LayerNorm(embed_dim);self.to_q=nn.Linear(embed_dim,inner,bias=False);self.to_k=nn.Linear(embed_dim,inner,bias=False);self.to_v=nn.Linear(embed_dim,inner,bias=False);self.to_out=nn.Sequential(nn.Linear(inner,embed_dim),nn.Dropout(dropout));self.scale=dim_head**-0.5;self.gate=nn.Parameter(torch.tensor(0.0))
    def forward(self,h,valid,neighbors):
        # Exclude the self slot: only previous and next heartbeats are auxiliary context.
        side=neighbors[:,:,[0,2]];b,l,_=side.shape;y=self.norm(h);q=self.to_q(y).view(b,l,self.heads,self.dim_head);ka=self.to_k(y).view(b,l,self.heads,self.dim_head);va=self.to_v(y).view(b,l,self.heads,self.dim_head)
        idx=side.clamp_min(0).long();bi=torch.arange(b,device=h.device)[:,None,None];k=ka[bi,idx];v=va[bi,idx];nvalid=(side>=0)&valid[bi,idx];scores=torch.einsum('blhd,blkhd->blkh',q,k)*self.scale;scores=scores.masked_fill(~nvalid.unsqueeze(-1),torch.finfo(scores.dtype).min)
        weights=torch.softmax(scores,dim=2);weights=torch.where(nvalid.unsqueeze(-1),weights,torch.zeros_like(weights));den=weights.sum(2,keepdim=True).clamp_min(1e-6);weights=weights/den
        out=torch.einsum('blkh,blkhd->blhd',weights,v).reshape(b,l,-1);has_neighbor=nvalid.any(-1)&valid;decoded=h+self.gate*self.to_out(out);return decoded*valid.unsqueeze(-1).to(decoded.dtype),has_neighbor

class LocalDecoderSIDMLM(nn.Module):
    def __init__(self,**cfg):
        super().__init__();self.backbone=e12.ContinuousBackbone(**cfg);self.lm_head=nn.Linear(int(cfg['embed_dim']),int(cfg['vocab_size']))
        with torch.random.fork_rng(devices=[]):torch.manual_seed(220024);self.local_decoder=PretrainingOnlyLocalDecoder(int(cfg['embed_dim']))
    def encode(self,ze,valid,mask=None):return self.backbone.forward_feature(ze,valid,mask)[:,1:]
    def forward(self,ze,valid,neighbors,mask,with_local=True):
        h=self.encode(ze,valid,mask);selected=mask.bool()&valid.bool();main=self.lm_head(h[selected])
        if not with_local:return main,None,None
        local_h,has_neighbor=self.local_decoder(h,valid,neighbors);local_selected=selected&has_neighbor;local=self.lm_head(local_h[local_selected]);return main,local,local_selected
    def extract_tokens(self,ze,valid):return self.backbone.token_features(ze,valid)

def initialize_model(device):
    e12.set_seed(120012);model=LocalDecoderSIDMLM(**e12.model_config());source=e12.fresh_sid_model(0).state_dict();target=model.state_dict();copied=[]
    for k,v in source.items():
        if k!='backbone.sid_embed.weight' and k in target and target[k].shape==v.shape:target[k]=v.detach().clone();copied.append(k)
    model.load_state_dict(target,strict=True);baseline,_=e12.initialize_continuous('scratch',torch.device('cpu'));bs=baseline.state_dict();common=max(float((model.state_dict()[k].cpu()-v).abs().max()) for k,v in bs.items())
    audit={'source':'same matched random E12-A initialization','copied_keys':len(copied),'common_e12_state_max_abs_diff':common,'local_decoder':'pretraining-only 4 heads x 32 dim; exact previous/next same-lead hidden states','local_gate_initial':float(model.local_decoder.gate),'local_parameters':sum(p.numel() for p in model.local_decoder.parameters()),'total_parameters':sum(p.numel() for p in model.parameters()),'downstream_decoder_discarded':True}
    if common!=0:raise RuntimeError(f'E12 common init drift: {common}')
    return model.to(device),audit

class ContextDataset(Dataset):
    def __init__(self,run,split):
        self.ze=np.load(E12_RUN/f'pretrain_ze_cache/{split}_ze.npy',mmap_mode='r');self.sid=np.load(e12.PRETRAIN_SID/f'{split}_sid.npy',mmap_mode='r');self.nei=np.load(run/f'context_cache/{split}_neighbors.npy',mmap_mode='r')
        if self.ze.shape[:2]!=self.sid.shape or self.nei.shape!=(len(self.sid),256,3):raise RuntimeError('context cache shape mismatch')
    def __len__(self):return len(self.sid)
    def __getitem__(self,i):return np.array(self.ze[i],copy=True),np.array(self.sid[i],copy=True),np.array(self.nei[i],copy=True)

def aux_weight(epoch:int)->float:
    if epoch<=10:return 0.0
    if epoch<=20:return LOCAL_AUX_MAX*(epoch-10)/10.0
    return LOCAL_AUX_MAX

def run_epoch(model,loader,device,optimizer,scaler,epoch,epochs,train):
    model.train(train);sums={'main_loss':0.,'main_correct':0.,'main_tokens':0.,'local_loss':0.,'local_correct':0.,'local_tokens':0.,'total_loss':0.,'batches':0};lam=aux_weight(epoch)
    for step,(ze,sid,nei) in enumerate(loader):
        ze=ze.float().to(device,non_blocking=True);sid=sid.long().to(device,non_blocking=True);nei=nei.to(device,non_blocking=True);valid=(sid>=0)&(sid<1024);mask=e12.random_mask(valid,.5);labels=sid[mask]
        if train:
            total=epochs*len(loader);warm=5*len(loader);lr=e12.lr_at((epoch-1)*len(loader)+step,total,warm,5e-4,1e-5,1e-6)
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train),torch.amp.autocast('cuda',enabled=device.type=='cuda'):
            main,local,local_selected=model(ze,valid,nei,mask,with_local=True);main_loss=F.cross_entropy(main.float(),labels);local_labels=sid[local_selected];local_loss=F.cross_entropy(local.float(),local_labels) if local_labels.numel() else main_loss.new_zeros(());loss=main_loss+lam*local_loss
        if train:
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),3.);scaler.step(optimizer);scaler.update()
        sums['main_loss']+=float(main_loss)*labels.numel();sums['main_correct']+=float((main.argmax(-1)==labels).sum());sums['main_tokens']+=labels.numel();sums['local_loss']+=float(local_loss)*local_labels.numel();sums['local_correct']+=float((local.argmax(-1)==local_labels).sum()) if local_labels.numel() else 0.;sums['local_tokens']+=local_labels.numel();sums['total_loss']+=float(loss)*labels.numel();sums['batches']+=1
    return {'main_loss':sums['main_loss']/sums['main_tokens'],'main_mlm_acc':sums['main_correct']/sums['main_tokens'],'local_loss':sums['local_loss']/max(1,sums['local_tokens']),'local_mlm_acc':sums['local_correct']/max(1,sums['local_tokens']),'total_loss':sums['total_loss']/sums['main_tokens'],'aux_weight':lam,'masked_tokens':sums['main_tokens']/sums['batches'],'local_tokens':sums['local_tokens']/sums['batches'],'batches':sums['batches']}

def checkpoint(model,opt,scaler,epoch,best,best_epoch,generator):
    return {'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best_main_val_loss':best,'best_epoch':best_epoch,'torch_rng_state':torch.get_rng_state(),'cuda_rng_state_all':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'loader_generator_state':generator.get_state(),'config':e12.model_config()|{'local_heads':LOCAL_HEADS,'local_dim_head':LOCAL_DIM_HEAD,'local_aux_max':LOCAL_AUX_MAX}}

def pretrain(run,device,epochs=100):
    out=run/'mlm'
    if (out/'COMPLETE').exists():return
    out.mkdir(parents=True,exist_ok=True);model,init=initialize_model(device);dump(out/'initialization_audit.json',init);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    tr=ContextDataset(run,'train');va=ContextDataset(run,'val');vloader=DataLoader(va,batch_size=384,shuffle=False,num_workers=8,pin_memory=True);gen=torch.Generator().manual_seed(0);start=1;best=float('inf');best_epoch=None;last=out/'checkpoint-last.pth'
    if last.exists():
        c=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);scaler.load_state_dict(c['scaler']);start=c['epoch']+1;best=c['best_main_val_loss'];best_epoch=c.get('best_epoch');torch.set_rng_state(c['torch_rng_state']);
        if torch.cuda.is_available() and c.get('cuda_rng_state_all') is not None:torch.cuda.set_rng_state_all(c['cuda_rng_state_all'])
        gen.set_state(c['loader_generator_state'])
    history=out/'training_history.jsonl';started=time.time()
    for epoch in range(start,epochs+1):
        loader=DataLoader(tr,batch_size=256,shuffle=True,generator=gen,num_workers=8,pin_memory=True,drop_last=True);ts=run_epoch(model,loader,device,opt,scaler,epoch,epochs,True);row={'epoch':epoch,**{f'train_{k}':v for k,v in ts.items()},'local_gate':float(model.local_decoder.gate.detach())}
        if epoch==1 or epoch%10==0 or epoch==epochs:
            vs=run_epoch(model,vloader,device,None,None,epoch,epochs,False);row.update({f'val_{k}':v for k,v in vs.items()})
            # Select only by the unchanged E12 main MLM head, never by auxiliary decoder loss.
            if vs['main_loss']<best:
                best=vs['main_loss'];best_epoch=epoch;torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'best_main_val_loss':best,'config':e12.model_config()},out/'checkpoint-best.pth')
        torch.save(checkpoint(model,opt,scaler,epoch,best,best_epoch,gen),last)
        with history.open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    dump(out/'summary.json',{'epochs':epochs,'best_epoch':best_epoch,'best_main_val_loss':best,'pretrain_seed':0,'train_records':len(tr),'val_records':len(va),'elapsed_seconds_this_process':time.time()-started,'final_local_gate':float(model.local_decoder.gate.detach()),'selection':'main E12-compatible MLM Val CE only','decoder_discarded_downstream':True});(out/'COMPLETE').touch()

def load_model(path,device):
    c=torch.load(path,map_location='cpu',weights_only=False);m=LocalDecoderSIDMLM(**e12.model_config());m.load_state_dict(c['model'],strict=True);return m.to(device).eval()

@torch.inference_mode()
def fixed_mlm(run,device):
    path=run/'fixed_mask_mlm_comparison.json'
    if path.exists():return
    ds=ContextDataset(run,'val');m=load_model(run/'mlm/checkpoint-best.pth',device);main_loss=main_correct=main_tokens=local_loss=local_correct=local_tokens=0
    for s in range(0,len(ds),128):
        e=min(s+128,len(ds));ze=torch.from_numpy(np.asarray(ds.ze[s:e]).copy()).float().to(device);sid_np=np.asarray(ds.sid[s:e]);sid=torch.from_numpy(sid_np.copy()).long().to(device);nei=torch.from_numpy(np.asarray(ds.nei[s:e]).copy()).to(device);mask=torch.from_numpy(e12.fixed_masks(sid_np,s)).to(device);valid=(sid>=0)&(sid<1024);labels=sid[mask];main,local,local_selected=m(ze,valid,nei,mask,with_local=True);local_labels=sid[local_selected]
        main_loss+=float(F.cross_entropy(main.float(),labels,reduction='sum'));main_correct+=int((main.argmax(-1)==labels).sum());main_tokens+=labels.numel();local_loss+=float(F.cross_entropy(local.float(),local_labels,reduction='sum'));local_correct+=int((local.argmax(-1)==local_labels).sum());local_tokens+=local_labels.numel()
    base=json.loads((E12_RUN/'fixed_mask_mlm_comparison.json').read_text())['e12a_scratch'];dump(path,{'same_label_space':True,'mask_policy':'identical E12 stateless fixed 50% valid-token mask','selection_metric':'main MLM CE only','e12a':base,'e22c_main_encoder':{'loss':main_loss/main_tokens,'masked_sid_accuracy':main_correct/main_tokens,'masked_tokens':main_tokens},'e22c_pretraining_only_local_decoder_diagnostic':{'loss':local_loss/local_tokens,'masked_sid_accuracy':local_correct/local_tokens,'masked_tokens_with_neighbor':local_tokens},'checkpoint':str(run/'mlm/checkpoint-best.pth')})

@torch.inference_mode()
def extract_features(run,device):
    out=run/'features'
    if (out/'COMPLETE').exists():return
    out.mkdir(parents=True,exist_ok=True);tok=e12.e4.load_model(e12.TOKENIZER,device).eval().requires_grad_(False);model=load_model(run/'mlm/checkpoint-best.pth',device).requires_grad_(False);audit={'pooling':'individual-token mean+max from unchanged E12 encoder path; local decoder discarded','text_used':False,'test_loaded':False,'local_decoder_used':False,'splits':{}}
    for split in ('train','val'):
        waves=np.load(e12.DOWNSTREAM/f'{split}_data.npy',mmap_mode='r');chans=np.load(e12.DOWNSTREAM/f'{split}_data_in_chans.npy',mmap_mode='r');times=np.load(e12.DOWNSTREAM/f'{split}_data_in_times.npy',mmap_mode='r');result=np.empty((len(waves),1536),np.float32);vt=0
        for s in range(0,len(waves),32):
            e=min(s+32,len(waves));wn=np.asarray(waves[s:e],dtype=np.float32);cn=np.asarray(chans[s:e]);tn=np.asarray(times[s:e]);x=torch.from_numpy(wn.copy()).to(device);c=torch.from_numpy(cn.copy()).long().to(device);t=torch.from_numpy(tn.copy()).long().to(device);valid=tok.valid_mask(x,c,t);ze=tok.encode_to_quantizer_features(x,c,t).float();nei=torch.from_numpy(make_neighbors(cn,tn,valid.cpu().numpy())).to(device);tokens=model.extract_tokens(ze,valid).float();count=valid.sum(1).clamp_min(1).unsqueeze(-1);mean=(tokens*valid.unsqueeze(-1)).sum(1)/count;maximum=tokens.masked_fill(~valid.unsqueeze(-1),-torch.inf).amax(1);maximum[~valid.any(1)]=0;result[s:e]=torch.cat([mean,maximum],1).cpu().numpy();vt+=int(valid.sum())
        np.save(out/f'{split}_heartlang_mean_max.npy',result);audit['splits'][split]={'records':len(result),'valid_tokens':vt,'shape':list(result.shape),'sha256':e12.sha256(out/f'{split}_heartlang_mean_max.npy'),'finite':bool(np.isfinite(result).all())}
    dump(out/'audit.json',audit);(out/'COMPLETE').touch()

def load_downstream(run,device):
    data={}
    for split in ('train','val'):
        labels=np.load(e12.DOWNSTREAM/f'{split}_labels.npy').astype(np.float32);mlm=np.load(run/f'features/{split}_heartlang_mean_max.npy');lead=np.load(e12.E10_RUN/f'features/{split}_lead_ze_mean_max.npy');lead_mask=np.load(e12.E10_RUN/f'features/{split}_lead_present.npy');global_ze=np.load(e12.E10_RUN/f'features/{split}_global_ze_mean_max.npy')
        data[split]={'labels':torch.from_numpy(labels).to(device),'mlm':torch.from_numpy(mlm).float().to(device),'lead':torch.from_numpy(lead).float().to(device),'lead_mask':torch.from_numpy(lead_mask).bool().to(device),'global':torch.from_numpy(global_ze).float().to(device)}
    return data

def train_downstream_seed(run,seed,device):
    out=run/'downstream'/f'seed{seed}'
    if (out/'VAL_COMPLETE').exists():return json.loads((out/'val_summary.json').read_text())
    out.mkdir(parents=True,exist_ok=False);data=load_downstream(run,device);model=e12.e11.E11RegionResidual(seed).to(device);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05);target=data['train']['labels'].argmax(1);steps=len(target)//64;gen=torch.Generator().manual_seed(seed);regions,rm=e12.e11.masked_region_means(data['train']['lead'],data['train']['lead_mask']);history=[];best=-float('inf');best_epoch=None;best_state=None
    for epoch in range(1,31):
        model.train();order=torch.randperm(len(target),generator=gen).to(device);losses=[]
        for step in range(steps):
            idx=order[step*64:(step+1)*64];lr=e12.lr_at((epoch-1)*steps+step,30*steps,5*steps,5e-4,1e-5,1e-6)
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True);main,aux=model.training_outputs(data['train']['mlm'][idx],data['train']['global'][idx],regions[idx],rm[idx]);lm=F.cross_entropy(main,target[idx]);la=e12.e9.patient_auxiliary_loss(aux,target[idx]);loss=lm+.2*la;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();losses.append(float(loss))
        val,_,_=e12.eval_downstream(model,'e12a_region',data['val']);history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val':val})
        if val['r2']>best:best=val['r2'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);val,prob,logits=e12.eval_downstream(model,'e12a_region',data['val']);torch.save({'model':{k:v.cpu() for k,v in best_state.items()},'seed':seed,'best_epoch':best_epoch,'selection':'Val Overall R2','test_run':False},out/'checkpoint-best.pth');np.save(out/'val_logits.npy',logits);np.save(out/'val_probabilities.npy',prob);dump(out/'history.json',history);result={'method':'e22c_pretraining_only_local_decoder','seed':seed,'best_epoch':best_epoch,'val':val,'test_run':False,'trainable_parameters':sum(p.numel() for p in model.parameters())};dump(out/'val_summary.json',result);(out/'VAL_COMPLETE').touch();return result

def downstream(run,device):
    path=run/'validation_summary.json'
    if path.exists():return
    rows=[train_downstream_seed(run,s,device) for s in (0,1,2)];base_rows=json.loads((E12_RUN/'downstream/validation_selection.json').read_text())['methods']['e12a_region']['per_seed'];b=json.loads((E22B_RUN/'validation_summary.json').read_text())['e22b_local_context']
    dump(path,{'validation_only':True,'test_run':False,'e12a_historical':{'aggregate':olde22.aggregate(base_rows),'per_seed':base_rows},'e22b_writeback_historical':b,'e22c_pretraining_only_decoder':{'aggregate':olde22.aggregate(rows),'per_seed':rows}})

def smoke(run,device):
    path=run/'smoke.json'
    if path.exists():return
    build_context_cache(run);ds=ContextDataset(run,'train');ze=torch.from_numpy(np.asarray(ds.ze[:4]).copy()).float().to(device);sid=torch.from_numpy(np.asarray(ds.sid[:4]).copy()).long().to(device);nei=torch.from_numpy(np.asarray(ds.nei[:4]).copy()).to(device);valid=(sid>=0)&(sid<1024);mask=e12.random_mask(valid,.5,torch.Generator(device=device).manual_seed(22));model,init=initialize_model(device);baseline,_=e12.initialize_continuous('scratch',device);model.eval();baseline.eval()
    with torch.no_grad():
        main,local,local_selected=model(ze,valid,nei,mask,with_local=True);base=baseline(ze,valid,mask);changed=ze.clone();changed[mask]=torch.randn_like(changed[mask])*1e5;main_changed,local_changed,_=model(changed,valid,nei,mask,with_local=True);features=model.extract_tokens(ze,valid);base_features=baseline.backbone.token_features(ze,valid)
    init_delta=float((main-base).abs().max());main_leak=float((main-main_changed).abs().max());local_leak=float((local-local_changed).abs().max());feature_init_delta=float((features-base_features).abs().max());model.train();labels=sid[local_selected];loss=F.cross_entropy(model(ze,valid,nei,mask,with_local=True)[1].float(),labels);loss.backward();gate_grad=float(model.local_decoder.gate.grad.abs());backbone_grad=float(model.backbone.input_proj.weight.grad.norm());opt=torch.optim.SGD(model.parameters(),lr=1e-3);opt.step();opt.zero_grad();local2=model(ze,valid,nei,mask,with_local=True)[1];F.cross_entropy(local2.float(),labels).backward();local_q_grad=float(model.local_decoder.to_q.weight.grad.norm())
    result={'passed':all([init['common_e12_state_max_abs_diff']==0,init_delta==0,feature_init_delta==0,main_leak==0,local_leak==0,gate_grad>0,backbone_grad>0,local_q_grad>0]),'common_e12_state_max_abs_diff':init['common_e12_state_max_abs_diff'],'initial_main_logits_vs_e12_max_abs_diff':init_delta,'initial_downstream_features_vs_e12_max_abs_diff':feature_init_delta,'masked_ze_main_mutation_delta':main_leak,'masked_ze_local_mutation_delta':local_leak,'local_gate_grad':gate_grad,'local_q_grad_after_gate_step':local_q_grad,'backbone_grad_from_local_auxiliary':backbone_grad,'neighbors_cross_lead':json.loads((run/'context_cache/audit.json').read_text())['cross_lead_neighbors'],'same_label_space_as_e12':True,'individual_tokens_preserved':True,'decoder_discarded_downstream':True,'main_mlm_full_weight':True,'aux_schedule':'epochs1-10=0; epochs11-20 linear to0.1; epochs21-100=0.1','no_lead_or_time_embedding':True,'text_used':False,'test_loaded':False,'initialization':init}
    if not result['passed']:raise RuntimeError(f'smoke failed {result}')
    dump(path,result)

def report(run):
    s=json.loads((run/'validation_summary.json').read_text());f=json.loads((run/'fixed_mask_mlm_comparison.json').read_text());ctx=json.loads((run/'context_cache/audit.json').read_text());mlm=json.loads((run/'mlm/summary.json').read_text());base=s['e12a_historical']['aggregate'];b=s['e22b_writeback_historical']['aggregate'];new=s['e22c_pretraining_only_decoder']['aggregate'];rows=[]
    for label,key in [('E12-A','e12a_historical'),('E22-B-Writeback','e22b_writeback_historical'),('E22-C-PretrainDecoder','e22c_pretraining_only_decoder')]:
        for x in s[key]['per_seed']:
            v=x['val'];rows.append({'method':label,'seed':x.get('seed',x.get('downstream_seed')),'overall_r2':v['r2'],'patient_mean_r2':np.mean([v['per_output_r2'][n] for n in e12.LABEL_NAMES[:3]]),**{f"{n.lower().replace(' ','_')}_r2":v['per_output_r2'][n] for n in e12.LABEL_NAMES},'macro_auc':v['macro_roc_auc'],'macro_f1':v['macro_f1']})
    pd.DataFrame(rows).to_csv(run/'metrics_per_seed.csv',index=False)
    def row(label,x):return f"| {label} | {x['r2']['mean']:.6f} ± {x['r2']['sd']:.6f} | {x['patient_mean_r2']['mean']:.6f} | "+' | '.join(f"{x['per_output_r2'][n]['mean']:.6f}" for n in e12.LABEL_NAMES)+f" | {x['macro_roc_auc']['mean']:.6f} | {x['macro_f1']['mean']:.6f} |"
    a=ctx['splits']['train'];lines=['# E22-C：仅预训练使用的三心拍局部SID解码器','','## 方法','','- E12-A encoder路径完全保留；每个96点心拍仍是独立z_e token和原始SID目标。','- 主MLM CE始终完整权重1.0；额外局部解码器只在Mask位置读取同导联前后邻拍隐藏状态。','- 局部辅助权重Epoch1-10为0，Epoch11-20线性升至0.1，Epoch21-100固定0.1。','- Checkpoint只按主MLM Val CE选择；局部解码器在下游完全丢弃。','- 无Lead/Time Embedding、文字或临床标签；下游seed0/1/2，仅Validation，未运行Test。','','## 调研依据','','- MAE使用预训练轻量解码器并在下游丢弃，避免重建结构直接污染下游encoder表示。','- HuBERT/wav2vec 2.0仅在Mask位置预测离散目标，保留连续encoder作为下游表示。','','## 邻域审计','',f"- Train有效心拍：{a['valid_tokens']}；完整前后邻域：{a['full_triplet_fraction']:.2%}；跨导联邻居：{a['cross_lead_neighbors']}。",'', '## 固定Mask MLM（同一SID标签空间）','', '| 方法 | Val CE | Accuracy |','|---|---:|---:|',f"| E12-A主头 | {f['e12a']['loss']:.6f} | {f['e12a']['masked_sid_accuracy']:.6f} |",f"| E22-C主头 | {f['e22c_main_encoder']['loss']:.6f} | {f['e22c_main_encoder']['masked_sid_accuracy']:.6f} |",f"| E22-C局部解码器（诊断） | {f['e22c_pretraining_only_local_decoder_diagnostic']['loss']:.6f} | {f['e22c_pretraining_only_local_decoder_diagnostic']['masked_sid_accuracy']:.6f} |",'', '## 下游Validation（三个seed）','', '| 方法 | Overall R² | 患者三类均值 | OMI | NOMI | Control | Healthy | Macro AUC | Macro F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|',row('E12-A',base),row('E22-B写回局部上下文',b),row('E22-C预训练局部解码器',new),'',f"E22-C相对E12-A Overall R²：**{new['r2']['mean']-base['r2']['mean']:+.6f}**；相对E22-B：**{new['r2']['mean']-b['r2']['mean']:+.6f}**。",f"最佳MLM epoch：{mlm['best_epoch']}；最终局部解码门值：{mlm['final_local_gate']:.6f}。",'']
    (run/'README.md').write_text('\n'.join(lines),encoding='utf-8');set_status(run,'complete');(run/'COMPLETE').touch()

def protocol(run,epochs):
    if (run/'protocol.json').exists():return
    dump(run/'protocol.json',{'experiment':'E22-C unchanged E12 encoder plus pretraining-only exact same-lead previous/next SID decoder','baseline':'E12-A historical read-only','research_basis':['MAE asymmetric lightweight decoder discarded downstream','HuBERT/wav2vec2 masked-only discrete target prediction'],'tokenizer':'frozen E3 MorphologyEncoder and original 1024x64 Codebook','token':'one 96-point heartbeat -> one z_e[64] and original SID','encoder':'unchanged E12-A continuous-input 12-layer Transformer from identical initialization','local_decoder':'pretraining-only 4 heads x 32 dim; reads exact previous/next same-lead hidden states; never writes encoder tokens','loss':'main MLM CE + scheduled auxiliary local decoder CE','aux_schedule':'epochs1-10=0; epochs11-20 linear to0.1; epochs21-100=0.1','selection':'main E12-compatible MLM Val CE only','mlm_epochs':epochs,'batch':256,'seed':0,'mask_ratio':.5,'disabled':['text','lead_embedding','time_embedding','clinical_labels'],'downstream':'discard local decoder; frozen E12 encoder Mean+Max1536 plus unchanged E11 region residual; seeds0/1/2','evaluation':'Validation only; Test not loaded or run'})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','smoke','pretrain','fixed_mlm','features','downstream','report']);p.add_argument('--output_dir',type=Path,default=DEFAULT_RUN);p.add_argument('--epochs',type=int,default=100);args=p.parse_args();run=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir;run.mkdir(parents=True,exist_ok=True);protocol(run,args.epochs);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.stage=='audit':set_status(run,'audit_exact_same_lead_triplet_neighbors');build_context_cache(run)
    elif args.stage=='smoke':set_status(run,'smoke_pretraining_only_decoder_no_leakage');smoke(run,device)
    elif args.stage=='pretrain':set_status(run,'pretrain_e22c_pretraining_only_local_decoder');pretrain(run,device,args.epochs)
    elif args.stage=='fixed_mlm':set_status(run,'fixed_mask_mlm');fixed_mlm(run,device)
    elif args.stage=='features':set_status(run,'extract_train_val_features');extract_features(run,device)
    elif args.stage=='downstream':set_status(run,'downstream_seeds_0_1_2_validation_only');downstream(run,device)
    else:report(run)
if __name__=='__main__':main()
