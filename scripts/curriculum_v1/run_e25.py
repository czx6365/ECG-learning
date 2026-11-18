"""Frozen E12-A intermediate-layer readout, Train/Val only."""
import argparse, copy, csv, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'heartlang').exists())
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/experiments'),str(ROOT/'scripts/curriculum_v1')]
import run_e12_continuous_input as e
LAYERS=(6,9,12)
METHODS=('layer6','layer9','layer12','equal_mix','learned_mix','layer12_matched')
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str))
def status(r,s):
 (r/'status.txt').write_text(s);print(s,flush=True)
def pool(t,v):
 mean=(t*v[...,None]).sum(1)/v.sum(1).clamp_min(1)[:,None]
 maximum=t.masked_fill(~v[...,None],-torch.inf).amax(1);maximum[~v.any(1)]=0
 return torch.cat((mean,maximum),-1)
@torch.no_grad()
def read_layers(b,ze,v):
 x=b.input_proj(ze.float())*v[...,None];x=torch.cat((b.cls_token[None,None].expand(len(x),1,-1),x),1)
 x=b.dropout(x+b.pos_embed);valid=torch.cat((torch.ones(len(x),1,dtype=torch.bool,device=x.device),v),1);out=[]
 for i,layer in enumerate(b.transformer.layers,1):
  x=e.masked_transformer_forward(SimpleNamespace(layers=[layer]),x,valid)
  if i in LAYERS:
   t=(b.norm_layer(x)*valid[...,None])[:,1:];out.append(pool(t,v))
 return torch.stack(out,1)
def cache(r,device):
 out=r/'features';out.mkdir(exist_ok=True)
 if (out/'COMPLETE').exists():return
 tok=e.e4.load_model(e.TOKENIZER,device).eval().requires_grad_(False)
 model=e.load_continuous_checkpoint(e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth',device).eval().requires_grad_(False)
 audit={'layers':LAYERS,'normalization':'same frozen final LayerNorm applied separately to each selected block output','pooling':'each layer valid-token Mean+Max then convex feature mixture; not token mixture','splits':{},'test_loaded':False}
 before=e.state_sha(model.state_dict())
 for split in ('train','val'):
  w=np.load(e.DOWNSTREAM/f'{split}_data.npy',mmap_mode='r');c=np.load(e.DOWNSTREAM/f'{split}_data_in_chans.npy',mmap_mode='r');t=np.load(e.DOWNSTREAM/f'{split}_data_in_times.npy',mmap_mode='r')
  dest=np.lib.format.open_memmap(out/f'{split}.npy',mode='w+',dtype=np.float32,shape=(len(w),3,1536));maxdiff=0.
  ref=np.load(e.DEFAULT_RUN/f'features/scratch/{split}_heartlang_mean_max.npy',mmap_mode='r')
  for start in range(0,len(w),32):
   x=torch.tensor(np.asarray(w[start:start+32]).copy(),device=device,dtype=torch.float32);ch=torch.tensor(np.asarray(c[start:start+32]).copy(),device=device).long();ti=torch.tensor(np.asarray(t[start:start+32]).copy(),device=device).long();v=tok.valid_mask(x,ch,ti);ze=tok.encode_to_quantizer_features(x,ch,ti).float()
   z=read_layers(model.backbone,ze,v)
   if start==0:
    direct=pool(model.extract_tokens(ze,v).float(),v);assert torch.allclose(z[:,2],direct,atol=1e-6,rtol=1e-6)
   a=z.cpu().numpy();assert np.isfinite(a).all();dest[start:start+len(a)]=a;maxdiff=max(maxdiff,float(np.max(np.abs(a[:,2]-ref[start:start+len(a)]))))
   if start%1024==0:print('cache',split,start,len(w),flush=True)
  dest.flush();del dest
  assert maxdiff<2e-5,('historical layer12 mismatch',maxdiff)
  audit['splits'][split]={'records':len(w),'historical_layer12_max_abs_diff':maxdiff,'sha256':e.sha256(out/f'{split}.npy')}
 assert e.state_sha(model.state_dict())==before
 audit['frozen_model_unchanged']=True
 dump(out/'audit.json',audit);(out/'COMPLETE').touch()
class Head(nn.Module):
 def __init__(self,method,seed):
  super().__init__();self.method=method;self.base=e.e11.E11RegionResidual(seed)
  if method=='learned_mix':self.adapt=nn.Parameter(torch.zeros(3))
  if method=='layer12_matched':self.adapt=nn.Parameter(torch.zeros(3))
 def features(self,z):
  if self.method.startswith('layer'):
   f=z[:,{ 'layer6':0,'layer9':1,'layer12':2,'layer12_matched':2}[self.method]]
   if self.method=='layer12_matched':f=f*(1+.1*torch.tanh(self.adapt).repeat_interleave(512))
   return f
  weight=torch.softmax(self.adapt,0) if self.method=='learned_mix' else z.new_full((3,),1/3)
  return (z*weight[None,:,None]).sum(1)
 def forward(self,z,g,regions,mask,training_outputs=False):
  f=self.features(z)
  return self.base.training_outputs(f,g,regions,mask) if training_outputs else self.base(f,g,regions,mask)
def data(r,device):
 d={}
 for s in ('train','val'):
  lead=torch.tensor(np.load(e.E10_RUN/f'features/{s}_lead_ze_mean_max.npy'),device=device);mask=torch.tensor(np.load(e.E10_RUN/f'features/{s}_lead_present.npy'),device=device).bool();regions,rm=e.e11.masked_region_means(lead,mask)
  d[s]={'z':torch.tensor(np.load(r/f'features/{s}.npy'),device=device),'g':torch.tensor(np.load(e.E10_RUN/f'features/{s}_global_ze_mean_max.npy'),device=device),'regions':regions,'mask':rm,'y':torch.tensor(np.load(e.DOWNSTREAM/f'{s}_labels.npy'),device=device)}
 assert [len(d[s]['y']) for s in ('train','val')]==[6640,1344]
 return d
@torch.no_grad()
def evaluate(m,d):
 m.eval();raw=[]
 for i in range(0,len(d['y']),96):raw.append(m(*[d[k][i:i+96] for k in ('z','g','regions','mask')]).cpu().numpy())
 raw=np.concatenate(raw);metrics,prob=e.probability_metrics(raw,d['y'].cpu().numpy());metrics['patient_mean_r2']=float(np.mean(list(metrics['per_output_r2'].values())[:3]));return metrics,prob,raw
def smoke(r,d,device):
 a=d['train'];args=[a[k][:64] for k in ('z','g','regions','mask')];counts={};grads={}
 for method in METHODS:
  e.e4.set_seed(0);m=Head(method,0).to(device);counts[method]=sum(p.numel() for p in m.parameters());main,aux=m(*args,training_outputs=True);y=a['y'][:64].argmax(1);loss=F.cross_entropy(main,y)+.2*e.e9.patient_auxiliary_loss(aux,y);loss.backward()
  if hasattr(m,'adapt'):grads[method]=m.adapt.grad.detach().cpu().tolist();assert m.adapt.grad.abs().sum()>0
  assert torch.isfinite(loss)
 assert counts['learned_mix']==counts['layer12_matched']==counts['layer12']+3
 m=Head('layer12',0).to(device).eval();ref=e.e11.E11RegionResidual(0).to(device).eval()
 assert torch.equal(m(*args),ref(args[0][:,2],*args[1:]))
 dump(r/'smoke.json',{'passed':True,'trainable_parameters':counts,'adapter_gradients':grads,'layer12_initial_logits_exact':True,'test_loaded':False})
def train(r,d,device,method,seed):
 out=r/'downstream'/method/f'seed{seed}'
 if (out/'VAL_COMPLETE').exists():return json.loads((out/'summary.json').read_text())
 out.mkdir(parents=True,exist_ok=True);e.e4.set_seed(seed);m=Head(method,seed).to(device);opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=.05);a=d['train'];y=a['y'].argmax(1);steps=len(y)//64;gen=torch.Generator().manual_seed(seed);best=-float('inf');hist=[]
 for epoch in range(1,31):
  m.train();order=torch.randperm(len(y),generator=gen).to(device)
  for step in range(steps):
   ix=order[step*64:(step+1)*64];lr=e.lr_at((epoch-1)*steps+step,30*steps,5*steps,5e-4,1e-5,1e-6)
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True);main,aux=m(*[a[k][ix] for k in ('z','g','regions','mask')],training_outputs=True);loss=F.cross_entropy(main,y[ix])+.2*e.e9.patient_auxiliary_loss(aux,y[ix]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3.);opt.step()
  val,_,_=evaluate(m,d['val']);hist.append({'epoch':epoch,'val':val})
  if val['r2']>best:best=val['r2'];state=copy.deepcopy(m.state_dict());best_epoch=epoch
 m.load_state_dict(state);val,prob,raw=evaluate(m,d['val']);np.save(out/'val_probabilities.npy',prob);np.save(out/'val_logits.npy',raw);torch.save({'model':state,'best_epoch':best_epoch},out/'checkpoint-best.pth');dump(out/'history.json',hist)
 row={'method':method,'seed':seed,'best_epoch':best_epoch,'val':val,'parameters':sum(p.numel() for p in m.parameters())}
 if hasattr(m,'adapt'):row['adapter']=m.adapt.detach().cpu().tolist()
 if method=='learned_mix':row['layer_weights']=m.adapt.softmax(0).detach().cpu().tolist()
 dump(out/'summary.json',row);(out/'VAL_COMPLETE').touch();print(json.dumps(row),flush=True);return row
def report(r,rows,d):
 result={};lines=['# E25：E12-A冻结中间层读取实验','','所有方案保持E11头、区域分支、初始化、30 epoch及seed0/1/2一致。只读取Train/Val；无Test。','第6/9/12层均使用同一个冻结最终LayerNorm，逐层Mean+Max后混合。','learned_mix为3参数softmax凸组合，初始等权；layer12_matched为3个有界特征分组增益，初始恒等。它是额外参数对照，函数族不同。','','|方法|Overall R² mean±SD|OMI|NOMI|Control|Healthy|Macro AUC|Macro F1|','|---|---:|---:|---:|---:|---:|---:|---:|']
 keys=('r2','patient_mean_r2','macro_roc_auc','macro_f1')
 for method in METHODS:
  rr=[x for x in rows if x['method']==method];agg={k:{'mean':float(np.mean([x['val'][k] for x in rr])),'sd':float(np.std([x['val'][k] for x in rr],ddof=1))} for k in keys};agg['per_output_r2']={k:float(np.mean([x['val']['per_output_r2'][k] for x in rr])) for k in rr[0]['val']['per_output_r2']};result[method]={'aggregate':agg,'per_seed':rr}
  p=list(agg['per_output_r2'].values());lines.append(f"|{method}|{agg['r2']['mean']:.6f} ± {agg['r2']['sd']:.6f}|"+'|'.join(f'{v:.6f}' for v in p)+f"|{agg['macro_roc_auc']['mean']:.6f}|{agg['macro_f1']['mean']:.6f}|")
 base=result['layer12'];rng=np.random.default_rng(25006);boots={};labels=d['val']['y'].cpu().numpy();idx=rng.integers(0,len(labels),size=(2000,len(labels)))
 from sklearn.metrics import r2_score
 for method in METHODS:
  if method=='layer12':continue
  pred=np.mean([np.load(r/'downstream'/method/f'seed{s}/val_probabilities.npy') for s in (0,1,2)],0);ref=np.mean([np.load(r/'downstream/layer12'/f'seed{s}/val_probabilities.npy') for s in (0,1,2)],0)
  delta=[float(r2_score(labels[ii],pred[ii])-r2_score(labels[ii],ref[ii])) for ii in idx];boots[method]={'paired_seed_r2_deltas':[result[method]['per_seed'][s]['val']['r2']-base['per_seed'][s]['val']['r2'] for s in range(3)],'paired_record_bootstrap_ensemble_delta_ci95':np.quantile(delta,[.025,.975]).tolist()}
 lines+=['','## 配对比较','',json.dumps(boots,ensure_ascii=False,indent=2),'','Bootstrap在固定三个seed的平均概率上按验证记录配对重采样；不覆盖预训练随机性、多重筛选或重复患者相关性。','所有预训练表示来自同一个E12-A checkpoint；本实验不证明独立预训练seed上的收益。']
 dump(r/'validation_summary.json',result);dump(r/'paired_comparison.json',boots);(r/'README.md').write_text('\n'.join(lines))
 with (r/'metrics_per_seed.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['method','seed','best_epoch',*keys]);w.writerows([[x['method'],x['seed'],x['best_epoch'],*[x['val'][k] for k in keys]] for x in rows])
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'results/e25_intermediate_layers');a=p.parse_args();r=a.run;r.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);device=torch.device('cuda')
 sources=[e.TOKENIZER,e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth',Path(e.__file__),Path(e.e11.__file__)];before={str(x):e.sha256(x) for x in sources}
 dump(r/'protocol.json',{'layers':LAYERS,'methods':METHODS,'seeds':[0,1,2],'pretrain_seed':0,'epochs':30,'batch':64,'loss':'CE4+0.2*patient_CE3','optimizer':'AdamW lr5e-4 wd0.05 cosine warmup5 min1e-5 clip3','selection':'Val Overall R2 per seed checkpoint; mean Val Overall R2 method selection','frozen_backbone':True,'test_loaded':False,'source_sha256':before,'comparison_to_E5_E6':'E5/E6 changed pooling and head budget on E3; E25 changes E12-A block readout with E11 fixed'})
 status(r,'cache');cache(r,device);d=data(r,device);status(r,'smoke');smoke(r,d,device);rows=[]
 for method in METHODS:
  for seed in (0,1,2):status(r,f'{method}_seed{seed}');rows.append(train(r,d,device,method,seed))
 status(r,'report');report(r,rows,d);assert all(e.sha256(Path(k))==v for k,v in before.items());dump(r/'source_audit.json',{'source_hashes_unchanged':True,'sha256':before});status(r,'complete');(r/'COMPLETE').touch()
if __name__=='__main__':main()
