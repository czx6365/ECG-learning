#!/usr/bin/env python3
"""E22-B: retain individual heartbeat tokens and add exact same-lead 3-beat local attention.

Each original E3 continuous z_e remains one token and keeps its original frozen
E3 SID target.  After masked z_e replacement, every token attends only to its
previous/self/next heartbeat in real R-peak time order within the same measured
lead.  A zero-initialized residual gate makes the model start exactly as E12-A.
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

DEFAULT_RUN=ROOT/'results/e22b_three_beat_local_context'
E12_RUN=e12.DEFAULT_RUN
OLD_E22_RUN=ROOT/'results/e22_three_beat_grouped_sid'
LOCAL_HEADS=4; LOCAL_DIM_HEAD=32

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

class LocalTripletAttention(nn.Module):
    def __init__(self,embed_dim=768,heads=LOCAL_HEADS,dim_head=LOCAL_DIM_HEAD,dropout=.1):
        super().__init__();self.heads=heads;self.dim_head=dim_head;inner=heads*dim_head
        self.norm=nn.LayerNorm(embed_dim);self.to_q=nn.Linear(embed_dim,inner,bias=False);self.to_k=nn.Linear(embed_dim,inner,bias=False);self.to_v=nn.Linear(embed_dim,inner,bias=False);self.to_out=nn.Sequential(nn.Linear(inner,embed_dim),nn.Dropout(dropout));self.scale=dim_head**-0.5
    def forward(self,x,valid,neighbors):
        b,l,_=x.shape;y=self.norm(x);q=self.to_q(y).view(b,l,self.heads,self.dim_head);ka=self.to_k(y).view(b,l,self.heads,self.dim_head);va=self.to_v(y).view(b,l,self.heads,self.dim_head)
        idx=neighbors.clamp_min(0).long();bi=torch.arange(b,device=x.device)[:,None,None];k=ka[bi,idx];v=va[bi,idx]
        nvalid=(neighbors>=0)&valid[bi,idx];scores=torch.einsum('blhd,blkhd->blkh',q,k)*self.scale;scores=scores.masked_fill(~nvalid.unsqueeze(-1),torch.finfo(scores.dtype).min)
        weights=torch.softmax(scores,dim=2);out=torch.einsum('blkh,blkhd->blhd',weights,v).reshape(b,l,-1);return self.to_out(out)*valid.unsqueeze(-1).to(out.dtype)

class LocalContextBackbone(e12.ContinuousBackbone):
    def __init__(self,**cfg):
        super().__init__(**cfg)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(220023);self.local_attention=LocalTripletAttention(self.embed_dim)
        self.local_gate=nn.Parameter(torch.tensor(0.0))
    def forward_feature(self,ze,valid,neighbors,mask_bool_matrix=None):
        if ze.ndim!=3 or ze.shape[1:]!=(self.seq_len,self.code_dim):raise ValueError(f'bad ze {tuple(ze.shape)}')
        valid=valid.bool();x=self.input_proj(ze.float());x=x*valid.unsqueeze(-1).to(x.dtype)
        mask=torch.zeros_like(valid) if mask_bool_matrix is None else mask_bool_matrix.bool()&valid
        x=torch.where(mask.unsqueeze(-1),self.mask_token.expand_as(x),x)
        # Local context is computed only after complete masked-value replacement.
        x=(x+self.local_gate*self.local_attention(x,valid,neighbors))*valid.unsqueeze(-1).to(x.dtype)
        cls=self.cls_token.view(1,1,-1).expand(len(x),1,-1);x=torch.cat([cls,x],1);full_valid=torch.cat([torch.ones(len(x),1,dtype=torch.bool,device=x.device),valid],1)
        x=self.dropout(x+self.pos_embed);x=e12.masked_transformer_forward(self.transformer,x,full_valid);return self.norm_layer(x)*full_valid.unsqueeze(-1).to(x.dtype)
    def token_features(self,ze,valid,neighbors):return self.forward_feature(ze,valid,neighbors)[:,1:]

class LocalContextSIDMLM(nn.Module):
    def __init__(self,**cfg):super().__init__();self.backbone=LocalContextBackbone(**cfg);self.lm_head=nn.Linear(int(cfg['embed_dim']),int(cfg['vocab_size']))
    def forward(self,ze,valid,neighbors,mask):return self.lm_head(self.backbone.forward_feature(ze,valid,neighbors,mask)[:,1:][mask.bool()&valid.bool()])
    def extract_tokens(self,ze,valid,neighbors):return self.backbone.token_features(ze,valid,neighbors)

def initialize_model(device):
    e12.set_seed(120012);model=LocalContextSIDMLM(**e12.model_config());source=e12.fresh_sid_model(0).state_dict();target=model.state_dict();copied=[]
    for k,v in source.items():
        if k!='backbone.sid_embed.weight' and k in target and target[k].shape==v.shape:target[k]=v.detach().clone();copied.append(k)
    model.load_state_dict(target,strict=True);baseline,_=e12.initialize_continuous('scratch',torch.device('cpu'));bs=baseline.state_dict();common=max(float((model.state_dict()[k].cpu()-v).abs().max()) for k,v in bs.items())
    audit={'source':'same matched random E12-A initialization','copied_keys':len(copied),'common_e12_state_max_abs_diff':common,'local_attention':'4 heads x 32 dim; previous/self/next exact same-lead time order','local_gate_initial':float(model.backbone.local_gate),'local_parameters':sum(p.numel() for n,p in model.named_parameters() if 'local_' in n or 'local_attention' in n),'total_parameters':sum(p.numel() for p in model.parameters())}
    if common!=0:raise RuntimeError(f'E12 common init drift: {common}')
    return model.to(device),audit

class ContextDataset(Dataset):
    def __init__(self,run,split):
        self.ze=np.load(E12_RUN/f'pretrain_ze_cache/{split}_ze.npy',mmap_mode='r');self.sid=np.load(e12.PRETRAIN_SID/f'{split}_sid.npy',mmap_mode='r');self.nei=np.load(run/f'context_cache/{split}_neighbors.npy',mmap_mode='r')
        if self.ze.shape[:2]!=self.sid.shape or self.nei.shape!=(len(self.sid),256,3):raise RuntimeError('context cache shape mismatch')
    def __len__(self):return len(self.sid)
    def __getitem__(self,i):return np.array(self.ze[i],copy=True),np.array(self.sid[i],copy=True),np.array(self.nei[i],copy=True)

def run_epoch(model,loader,device,optimizer,scaler,epoch,epochs,train):
    model.train(train);sums={'loss':0.,'correct':0.,'tokens':0.,'batches':0}
    for step,(ze,sid,nei) in enumerate(loader):
        ze=ze.float().to(device,non_blocking=True);sid=sid.long().to(device,non_blocking=True);nei=nei.to(device,non_blocking=True)
        valid=(sid>=0)&(sid<1024);mask=e12.random_mask(valid,.5);labels=sid[mask]
        if train:
            total=epochs*len(loader);warm=5*len(loader);lr=e12.lr_at((epoch-1)*len(loader)+step,total,warm,5e-4,1e-5,1e-6)
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train),torch.amp.autocast('cuda',enabled=device.type=='cuda'):
            logits=model(ze,valid,nei,mask);loss=F.cross_entropy(logits.float(),labels)
        if train:
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),3.);scaler.step(optimizer);scaler.update()
        sums['loss']+=float(loss)*labels.numel();sums['correct']+=float((logits.argmax(-1)==labels).sum());sums['tokens']+=labels.numel();sums['batches']+=1
    return {'loss':sums['loss']/sums['tokens'],'mlm_acc':sums['correct']/sums['tokens'],'masked_tokens':sums['tokens']/sums['batches'],'batches':sums['batches']}

def checkpoint(model,opt,scaler,epoch,best,best_epoch,generator):
    return {'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best_val_loss':best,'best_epoch':best_epoch,'torch_rng_state':torch.get_rng_state(),'cuda_rng_state_all':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'loader_generator_state':generator.get_state(),'config':e12.model_config()|{'local_heads':LOCAL_HEADS,'local_dim_head':LOCAL_DIM_HEAD}}

def pretrain(run,device,epochs=100):
    out=run/'mlm'
    if (out/'COMPLETE').exists():return
    out.mkdir(parents=True,exist_ok=True);model,init=initialize_model(device);dump(out/'initialization_audit.json',init);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    tr=ContextDataset(run,'train');va=ContextDataset(run,'val');vloader=DataLoader(va,batch_size=384,shuffle=False,num_workers=8,pin_memory=True);gen=torch.Generator().manual_seed(0);start=1;best=float('inf');best_epoch=None;last=out/'checkpoint-last.pth'
    if last.exists():
        c=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);scaler.load_state_dict(c['scaler']);start=c['epoch']+1;best=c['best_val_loss'];best_epoch=c.get('best_epoch');torch.set_rng_state(c['torch_rng_state']);
        if torch.cuda.is_available() and c.get('cuda_rng_state_all') is not None:torch.cuda.set_rng_state_all(c['cuda_rng_state_all'])
        gen.set_state(c['loader_generator_state'])
    history=out/'training_history.jsonl';started=time.time()
    for epoch in range(start,epochs+1):
        loader=DataLoader(tr,batch_size=256,shuffle=True,generator=gen,num_workers=8,pin_memory=True,drop_last=True);ts=run_epoch(model,loader,device,opt,scaler,epoch,epochs,True);row={'epoch':epoch,**{f'train_{k}':v for k,v in ts.items()},'local_gate':float(model.backbone.local_gate.detach())}
        if epoch==1 or epoch%10==0 or epoch==epochs:
            vs=run_epoch(model,vloader,device,None,None,epoch,epochs,False);row.update({f'val_{k}':v for k,v in vs.items()})
            if vs['loss']<best:
                best=vs['loss'];best_epoch=epoch;torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'best_val_loss':best,'config':e12.model_config()},out/'checkpoint-best.pth')
        torch.save(checkpoint(model,opt,scaler,epoch,best,best_epoch,gen),last)
        with history.open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    dump(out/'summary.json',{'epochs':epochs,'best_epoch':best_epoch,'best_val_loss':best,'pretrain_seed':0,'train_records':len(tr),'val_records':len(va),'elapsed_seconds_this_process':time.time()-started,'final_local_gate':float(model.backbone.local_gate.detach())});(out/'COMPLETE').touch()

def load_model(path,device):
    c=torch.load(path,map_location='cpu',weights_only=False);m=LocalContextSIDMLM(**e12.model_config());m.load_state_dict(c['model'],strict=True);return m.to(device).eval()

@torch.inference_mode()
def fixed_mlm(run,device):
    path=run/'fixed_mask_mlm_comparison.json'
    if path.exists():return
    ds=ContextDataset(run,'val');m=load_model(run/'mlm/checkpoint-best.pth',device);loss=correct=tokens=0
    for s in range(0,len(ds),128):
        e=min(s+128,len(ds));ze=torch.from_numpy(np.asarray(ds.ze[s:e]).copy()).float().to(device);sid_np=np.asarray(ds.sid[s:e]);sid=torch.from_numpy(sid_np.copy()).long().to(device);nei=torch.from_numpy(np.asarray(ds.nei[s:e]).copy()).to(device);mask=torch.from_numpy(e12.fixed_masks(sid_np,s)).to(device);valid=(sid>=0)&(sid<1024);labels=sid[mask];logits=m(ze,valid,nei,mask);loss+=float(F.cross_entropy(logits.float(),labels,reduction='sum'));correct+=int((logits.argmax(-1)==labels).sum());tokens+=labels.numel()
    base=json.loads((E12_RUN/'fixed_mask_mlm_comparison.json').read_text())['e12a_scratch'];dump(path,{'same_label_space':True,'mask_policy':'identical E12 stateless fixed 50% valid-token mask','e12a':base,'e22b_local_context':{'loss':loss/tokens,'masked_sid_accuracy':correct/tokens,'masked_tokens':tokens},'checkpoint':str(run/'mlm/checkpoint-best.pth')})

@torch.inference_mode()
def extract_features(run,device):
    out=run/'features'
    if (out/'COMPLETE').exists():return
    out.mkdir(parents=True,exist_ok=True);tok=e12.e4.load_model(e12.TOKENIZER,device).eval().requires_grad_(False);model=load_model(run/'mlm/checkpoint-best.pth',device).requires_grad_(False);audit={'pooling':'individual-token mean+max after local+global Transformer','text_used':False,'test_loaded':False,'splits':{}}
    for split in ('train','val'):
        waves=np.load(e12.DOWNSTREAM/f'{split}_data.npy',mmap_mode='r');chans=np.load(e12.DOWNSTREAM/f'{split}_data_in_chans.npy',mmap_mode='r');times=np.load(e12.DOWNSTREAM/f'{split}_data_in_times.npy',mmap_mode='r');result=np.empty((len(waves),1536),np.float32);vt=0
        for s in range(0,len(waves),32):
            e=min(s+32,len(waves));wn=np.asarray(waves[s:e],dtype=np.float32);cn=np.asarray(chans[s:e]);tn=np.asarray(times[s:e]);x=torch.from_numpy(wn.copy()).to(device);c=torch.from_numpy(cn.copy()).long().to(device);t=torch.from_numpy(tn.copy()).long().to(device);valid=tok.valid_mask(x,c,t);ze=tok.encode_to_quantizer_features(x,c,t).float();nei=torch.from_numpy(make_neighbors(cn,tn,valid.cpu().numpy())).to(device);tokens=model.extract_tokens(ze,valid,nei).float();count=valid.sum(1).clamp_min(1).unsqueeze(-1);mean=(tokens*valid.unsqueeze(-1)).sum(1)/count;maximum=tokens.masked_fill(~valid.unsqueeze(-1),-torch.inf).amax(1);maximum[~valid.any(1)]=0;result[s:e]=torch.cat([mean,maximum],1).cpu().numpy();vt+=int(valid.sum())
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
    model.load_state_dict(best_state);val,prob,logits=e12.eval_downstream(model,'e12a_region',data['val']);torch.save({'model':{k:v.cpu() for k,v in best_state.items()},'seed':seed,'best_epoch':best_epoch,'selection':'Val Overall R2','test_run':False},out/'checkpoint-best.pth');np.save(out/'val_logits.npy',logits);np.save(out/'val_probabilities.npy',prob);dump(out/'history.json',history);result={'method':'e22b_three_beat_local_context','seed':seed,'best_epoch':best_epoch,'val':val,'test_run':False,'trainable_parameters':sum(p.numel() for p in model.parameters())};dump(out/'val_summary.json',result);(out/'VAL_COMPLETE').touch();return result

def downstream(run,device):
    path=run/'validation_summary.json'
    if path.exists():return
    rows=[train_downstream_seed(run,s,device) for s in (0,1,2)];base_rows=json.loads((E12_RUN/'downstream/validation_selection.json').read_text())['methods']['e12a_region']['per_seed'];old=json.loads((OLD_E22_RUN/'validation_summary.json').read_text())['e22_three_beat']
    dump(path,{'validation_only':True,'test_run':False,'e12a_historical':{'aggregate':olde22.aggregate(base_rows),'per_seed':base_rows},'e22_compressed_historical':old,'e22b_local_context':{'aggregate':olde22.aggregate(rows),'per_seed':rows}})

def smoke(run,device):
    path=run/'smoke.json'
    if path.exists():return
    build_context_cache(run);ds=ContextDataset(run,'train');ze=torch.from_numpy(np.asarray(ds.ze[:4]).copy()).float().to(device);sid=torch.from_numpy(np.asarray(ds.sid[:4]).copy()).long().to(device);nei=torch.from_numpy(np.asarray(ds.nei[:4]).copy()).to(device);valid=(sid>=0)&(sid<1024);mask=e12.random_mask(valid,.5,torch.Generator(device=device).manual_seed(22));model,init=initialize_model(device);baseline,_=e12.initialize_continuous('scratch',device);model.eval();baseline.eval()
    with torch.no_grad():
        a=model(ze,valid,nei,mask);b=baseline(ze,valid,mask);changed=ze.clone();changed[mask]=torch.randn_like(changed[mask])*1e5;c=model(changed,valid,nei,mask)
    init_delta=float((a-b).abs().max());leak=float((a-c).abs().max());model.train();opt=torch.optim.SGD(model.parameters(),lr=1e-3);labels=sid[mask];loss=F.cross_entropy(model(ze,valid,nei,mask).float(),labels);loss.backward();gate_grad=float(model.backbone.local_gate.grad.abs());opt.step();opt.zero_grad();loss2=F.cross_entropy(model(ze,valid,nei,mask).float(),labels);loss2.backward();local_grad=float(model.backbone.local_attention.to_q.weight.grad.norm())
    result={'passed':all([init['common_e12_state_max_abs_diff']==0,init_delta==0,leak==0,math.isfinite(gate_grad),gate_grad>0,math.isfinite(local_grad),local_grad>0]),'common_e12_state_max_abs_diff':init['common_e12_state_max_abs_diff'],'zero_gate_initial_logits_vs_e12_max_abs_diff':init_delta,'masked_ze_mutation_delta':leak,'first_step_gate_grad':gate_grad,'second_step_local_q_grad':local_grad,'neighbors_cross_lead':json.loads((run/'context_cache/audit.json').read_text())['cross_lead_neighbors'],'same_label_space_as_e12':True,'individual_tokens_preserved':True,'no_lead_or_time_embedding':True,'text_used':False,'test_loaded':False,'initialization':init}
    if not result['passed']:raise RuntimeError(f'smoke failed {result}')
    dump(path,result)

def report(run):
    s=json.loads((run/'validation_summary.json').read_text());f=json.loads((run/'fixed_mask_mlm_comparison.json').read_text());ctx=json.loads((run/'context_cache/audit.json').read_text());mlm=json.loads((run/'mlm/summary.json').read_text());base=s['e12a_historical']['aggregate'];old=s['e22_compressed_historical']['aggregate'];new=s['e22b_local_context']['aggregate'];rows=[]
    for label,key in [('E12-A','e12a_historical'),('E22-Compressed','e22_compressed_historical'),('E22-B-Local','e22b_local_context')]:
        for x in s[key]['per_seed']:
            v=x['val'];rows.append({'method':label,'seed':x.get('seed',x.get('downstream_seed')),'overall_r2':v['r2'],'patient_mean_r2':np.mean([v['per_output_r2'][n] for n in e12.LABEL_NAMES[:3]]),**{f"{n.lower().replace(' ','_')}_r2":v['per_output_r2'][n] for n in e12.LABEL_NAMES},'macro_auc':v['macro_roc_auc'],'macro_f1':v['macro_f1']})
    pd.DataFrame(rows).to_csv(run/'metrics_per_seed.csv',index=False)
    def row(label,x):return f"| {label} | {x['r2']['mean']:.6f} ± {x['r2']['sd']:.6f} | {x['patient_mean_r2']['mean']:.6f} | "+' | '.join(f"{x['per_output_r2'][n]['mean']:.6f}" for n in e12.LABEL_NAMES)+f" | {x['macro_roc_auc']['mean']:.6f} | {x['macro_f1']['mean']:.6f} |"
    a=ctx['splits']['train'];lines=['# E22-B：保留单心拍Token的三心拍局部上下文','','## 方法','','- 冻结原始E3 Encoder和1024×64 Codebook；每个96点心拍仍是独立连续z_e token及原始SID目标。','- 被Mask心拍先完整替换为[MASK]，随后每个token只对同一实测导联、真实时间排序的前一拍/本拍/后一拍做轻量局部注意力。','- 零初始化残差门使训练初始输出与E12-A完全一致；之后仍通过原始12层全局Transformer。','- 不使用Lead/Time Embedding、文字或临床标签；100 epoch，seed0；下游seed0/1/2，仅Validation，未运行Test。','','## 邻域审计','',f"- Train有效心拍：{a['valid_tokens']}；具有完整前后邻域：{a['full_triplet_fraction']:.2%}。",f"- 数组中非紧邻但由真实时间恢复的同导联相邻对：{a['noncontiguous_array_pairs']}（{a['noncontiguous_array_pair_fraction']:.2%}）。",f"- 跨导联邻居：{a['cross_lead_neighbors']}；时间逆序：{a['negative_time_steps']}。",'', '## 固定Mask MLM（与E12-A为同一SID标签空间）','', '| 方法 | Val CE | Accuracy |','|---|---:|---:|',f"| E12-A | {f['e12a']['loss']:.6f} | {f['e12a']['masked_sid_accuracy']:.6f} |",f"| E22-B Local | {f['e22b_local_context']['loss']:.6f} | {f['e22b_local_context']['masked_sid_accuracy']:.6f} |",'', '## 下游Validation（三个seed）','', '| 方法 | Overall R² | 患者三类均值 | OMI | NOMI | Control | Healthy | Macro AUC | Macro F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|',row('E12-A',base),row('E22压缩三拍',old),row('E22-B局部三拍',new),'',f"E22-B相对E12-A Overall R²：**{new['r2']['mean']-base['r2']['mean']:+.6f}**；相对旧E22：**{new['r2']['mean']-old['r2']['mean']:+.6f}**。",f"最佳MLM epoch：{mlm['best_epoch']}；最终局部门值：{mlm['final_local_gate']:.6f}。",'']
    (run/'README.md').write_text('\n'.join(lines),encoding='utf-8');set_status(run,'complete');(run/'COMPLETE').touch()

def protocol(run,epochs):
    if (run/'protocol.json').exists():return
    dump(run/'protocol.json',{'experiment':'E22-B individual heartbeat tokens plus exact same-lead sliding 3-beat local attention','baseline':'E12-A historical read-only','tokenizer':'frozen E3 MorphologyEncoder and original 1024x64 Codebook','token':'one 96-point heartbeat -> one z_e[64] and original SID','local_context':'previous/self/next by real R-peak time within measured lead','local_attention':'4 heads x 32 dim, zero-initialized residual gate, applied after mask replacement','global_transformer':'unchanged E12-A 12-layer Transformer from identical initialization','mlm_epochs':epochs,'batch':256,'seed':0,'mask_ratio':.5,'disabled':['text','lead_embedding','time_embedding','clinical_labels'],'downstream':'frozen Transformer Mean+Max1536 plus unchanged E11 region residual; seeds0/1/2','evaluation':'Validation only; Test not loaded or run'})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','smoke','pretrain','fixed_mlm','features','downstream','report']);p.add_argument('--output_dir',type=Path,default=DEFAULT_RUN);p.add_argument('--epochs',type=int,default=100);args=p.parse_args();run=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir;run.mkdir(parents=True,exist_ok=True);protocol(run,args.epochs);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.stage=='audit':set_status(run,'audit_exact_same_lead_triplet_neighbors');build_context_cache(run)
    elif args.stage=='smoke':set_status(run,'smoke_local_context_no_leakage');smoke(run,device)
    elif args.stage=='pretrain':set_status(run,'pretrain_e22b_local_context');pretrain(run,device,args.epochs)
    elif args.stage=='fixed_mlm':set_status(run,'fixed_mask_mlm');fixed_mlm(run,device)
    elif args.stage=='features':set_status(run,'extract_train_val_features');extract_features(run,device)
    elif args.stage=='downstream':set_status(run,'downstream_seeds_0_1_2_validation_only');downstream(run,device)
    else:report(run)
if __name__=='__main__':main()
