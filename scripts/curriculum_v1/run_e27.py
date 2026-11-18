"""Matched-budget shared-target masking diagnosis, no training and no Test."""
import sys,json,copy,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'heartlang').exists());sys.path[:0]=[str(ROOT),str(ROOT/'scripts/curriculum_v1'),str(ROOT/'scripts/experiments')]
import run_e12_continuous_input as e
import run_e25 as e25
from scripts.data_prep.ECG_DSA.tokenize_pretrain_qrs import QRSTokenizerLite
R=ROOT/'results/e27_mask_diagnostics';E25=ROOT/'results/e25_intermediate_layers'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str))
def status(s):R.mkdir(parents=True,exist_ok=True);(R/'status.txt').write_text(s);print(s,flush=True)
@torch.inference_mode()
def layered(model,ze,v,mask=None,return_targets=False):
 b=model.backbone;h=b.input_proj(ze.float())*v[...,None]
 if mask is not None:h=torch.where(mask[...,None],b.mask_token.expand_as(h),h)
 h=torch.cat((b.cls_token[None,None].expand(len(h),1,-1),h),1);h=b.dropout(h+b.pos_embed);fv=torch.cat((torch.ones(len(h),1,dtype=torch.bool,device=h.device),v),1);pooled=[];tokens=[]
 for i,layer in enumerate(b.transformer.layers,1):
  h=e.masked_transformer_forward(SimpleNamespace(layers=[layer]),h,fv)
  if i in (6,9,12):
   z=b.norm_layer(h)[:,1:]*v[...,None];pooled.append(e25.pool(z,v));tokens.append(z)
 return torch.stack(pooled,1),tokens

def cache(device):
 out=R/'cache';out.mkdir(exist_ok=True)
 if (out/'COMPLETE').exists():return
 tok=e.e4.load_model(e.TOKENIZER,device).eval().requires_grad_(False);before=e.state_sha(tok.state_dict());w=np.load(e.DOWNSTREAM/'val_data.npy',mmap_mode='r');c=np.load(e.DOWNSTREAM/'val_data_in_chans.npy');t=np.load(e.DOWNSTREAM/'val_data_in_times.npy');frames=pd.read_csv(e.DOWNSTREAM/'val_metadata.csv');dest=np.lib.format.open_memmap(out/'ze.npy',mode='w+',dtype=np.float32,shape=(len(w),256,64))
 for start in range(0,len(w),32):
  x=torch.tensor(np.array(w[start:start+32]),device=device).float();ch=torch.tensor(c[start:start+32],device=device).long();ti=torch.tensor(t[start:start+32],device=device).long();dest[start:start+len(x)]=tok.encode_to_quantizer_features(x,ch,ti).cpu().numpy()
 dest.flush();assert before==e.state_sha(tok.state_dict());del tok
 times=np.full((len(w),256),-1,np.int16);sources={};qt=QRSTokenizerLite();exact=0
 for i,row in frames.iterrows():
  src=Path(str(row.source_dataset));src=src if src.is_absolute() else ROOT/src;rawdir=Path(str(src).replace('/pretrain_qrs/','/pretrain_waveforms/').replace('/qrs/','/waveforms/'));ss=str(row.source_split);assert ss in ('train','val');key=(str(rawdir),ss)
  if key not in sources:sources[key]=(np.load(rawdir/f'{ss}_data.npy',mmap_mode='r'),pd.read_csv(rawdir/f'{ss}_metadata.csv'))
  arr,meta=sources[key];si=int(row.source_index)
  if str(meta.iloc[si].csv_name)!=str(row.csv_name):matches=np.flatnonzero(meta.csv_name.astype(str).values==str(row.csv_name));assert len(matches)==1;si=int(matches[0])
  wave=np.asarray(arr[si]);peaks=qt.qrs_detection(wave);mid=qt.trim_edge_qrs(peaks);segments=qt.extract_qrs_segments(wave,peaks);replay,ch,ti=qt.qrs_to_sequence(segments,mid);assert np.array_equal(replay,w[i]) and np.array_equal(ch,c[i]) and np.array_equal(ti,t[i]);exact+=1
  actual=np.tile(mid,9)[:256];times[i,:len(actual)]=actual
  if i%256==0:print('true_time_replay',i,len(w),flush=True)
 v=(c>0)&(t>0)&(np.abs(w).max(-1)>1e-8);assert np.array_equal(times>=0,v);np.save(out/'true_r_times.npy',times);np.save(out/'lead.npy',c);np.save(out/'valid.npy',v)
 sid=np.load(e.e9.SID/'val_sid.npy');assert sid.shape==v.shape and np.array_equal((sid>=0)&(sid<1024),v);np.save(out/'sid.npy',sid)
 dump(out/'audit.json',{'raw_replay_exact':exact,'records':len(w),'true_r_times':'recovered from original waveform XQRS, not coarse time IDs','tokenizer_full_state_unchanged':True,'test_loaded':False});(out/'COMPLETE').touch()
def make_masks(valid,lead,times,ratio,repeat):
 masks={p:np.zeros_like(valid) for p in ('random','time_span','lead_priority')};anchors=np.full(len(valid),-1,np.int16)
 for i,v in enumerate(valid):
  ids=np.flatnonzero(v)
  if len(ids)<2:continue
  rng=np.random.default_rng(270000+repeat*10000+i);anchor=int(rng.choice(ids));anchors[i]=anchor;k=max(1,min(len(ids)-1,int(round(len(ids)*ratio))));others=ids[ids!=anchor];tie={j:x for j,x in zip(ids,rng.random(len(ids)))}
  random=np.r_[anchor,rng.permutation(others)[:k-1]];span=sorted(ids,key=lambda j:(abs(int(times[i,j])-int(times[i,anchor])),j!=anchor,tie[j]))[:k]
  otherleads=list(np.unique(lead[i,ids]));otherleads.remove(lead[i,anchor]);rng.shuffle(otherleads);order=[lead[i,anchor],*otherleads];rank={c:k for k,c in enumerate(order)};grouped=sorted(ids,key=lambda j:(rank[lead[i,j]],j!=anchor,tie[j]))[:k]
  for p,q in zip(masks,(random,span,grouped)):masks[p][i,q]=True;assert masks[p][i,anchor] and masks[p][i].sum()==k
 assert np.array_equal(masks['random'].sum(1),masks['time_span'].sum(1)) and np.array_equal(masks['random'].sum(1),masks['lead_priority'].sum(1));return masks,anchors

def side_features(ze,lead,visible):
 globalze=e25.pool(ze,visible);perlead=[];lm=[]
 for c in (1,2,3,7,8,9,10,11,12):
  v=visible&(lead==c);perlead.append(e25.pool(ze,v));lm.append(v.any(1))
 pl=torch.stack(perlead,1);mask=torch.stack(lm,1);regions,rm=e.e11.masked_region_means(pl,mask);return globalze,regions,rm
@torch.inference_mode()
def diagnose(device):
 out=R/'diagnostics';out.mkdir(exist_ok=True);data={k:np.load(R/'cache'/f'{k}.npy',mmap_mode='r') for k in ('ze','lead','valid','true_r_times','sid')};labels=np.load(e.DOWNSTREAM/'val_labels.npy');model=e.load_continuous_checkpoint(e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth',device).eval().requires_grad_(False);before=e.state_sha(model.state_dict());heads={}
 for method in ('layer12','learned_mix'):
  for seed in (0,1,2):
   m=e25.Head(method,seed).to(device).eval();payload=torch.load(E25/'downstream'/method/f'seed{seed}/checkpoint-best.pth',map_location=device,weights_only=False);m.load_state_dict(payload['model']);m.requires_grad_(False);heads[method,seed]=m
 # Readout path and side-feature parity at zero mask.
 z=torch.tensor(np.array(data['ze'][:32]),device=device);v=torch.tensor(np.array(data['valid'][:32]),device=device);lead=torch.tensor(np.array(data['lead'][:32]),device=device);zero=torch.zeros_like(v);p,h=layered(model,z,v,zero);assert torch.allclose(h[-1],model.extract_tokens(z,v),atol=1e-6,rtol=1e-6)
 mask=torch.tensor(make_masks(data['valid'][:32],data['lead'][:32],data['true_r_times'][:32],.5,0)[0]['random'],device=device);assert mask.any();changed=z.clone();changed[mask]=torch.randn_like(changed[mask])*1e4;a,_=layered(model,z,v,mask);b,_=layered(model,changed,v,mask);assert torch.equal(a,b);sa=side_features(z,lead,v&~mask);sb=side_features(changed,lead,v&~mask);assert all(torch.equal(x,y) for x,y in zip(sa,sb));dump(R/'smoke.json',{'passed':True,'masked_input_value_invariance':True,'side_branch_masked_value_invariance':True,'layer12_forward_parity':True,'test_loaded':False})
 rows=[];cases=[('clean',0.,0)]+[(p,r,j) for r in (.25,.5,.75) for j in range(3) for p in ('random','time_span','lead_priority')]
 for policy,ratio,rep in cases:
  name=f'{policy}_r{ratio}_rep{rep}';dest=out/name;dest.mkdir(exist_ok=True)
  if (dest/'COMPLETE').exists():rows.extend(json.loads((dest/'metrics.json').read_text()));continue
  status(name)
  if policy=='clean':mask=np.zeros_like(data['valid']);anchor=np.full(len(mask),-1)
  else:allm,anchor=make_masks(data['valid'],data['lead'],data['true_r_times'],ratio,rep);mask=allm[policy]
  np.save(dest/'mask.npy',mask);np.save(dest/'anchor.npy',anchor);raws={k:[] for k in heads};anchor_ce=np.full(len(labels),np.nan);anchor_ok=np.full(len(labels),np.nan);allce=0.;allcorrect=allcount=0
  for start in range(0,len(labels),32):
   end=min(start+32,len(labels));z=torch.tensor(np.array(data['ze'][start:end]),device=device);v=torch.tensor(np.array(data['valid'][start:end]),device=device).bool();lead=torch.tensor(np.array(data['lead'][start:end]),device=device);ms=torch.tensor(mask[start:end],device=device);sid=torch.tensor(np.array(data['sid'][start:end]),device=device).long();pool,h=layered(model,z,v,ms);g,reg,rm=side_features(z,lead,v&~ms)
   for k,m in heads.items():raws[k].append(m(pool,g,reg,rm).cpu().numpy())
   if ratio:
    aa=torch.tensor(anchor[start:end],device=device).long();ok=aa>=0;bi=torch.arange(len(z),device=device)[ok];logit=model.lm_head(h[-1][bi,aa[ok]]).float();target=sid[bi,aa[ok]];anchor_ce[start:end][ok.cpu().numpy()]=F.cross_entropy(logit,target,reduction='none').cpu().numpy();anchor_ok[start:end][ok.cpu().numpy()]=(logit.argmax(-1)==target).float().cpu().numpy();log=model.lm_head(h[-1][ms]).float();tar=sid[ms];allce+=float(F.cross_entropy(log,tar,reduction='sum'));allcorrect+=int((log.argmax(-1)==tar).sum());allcount+=len(tar)
  np.save(dest/'anchor_ce.npy',anchor_ce);local=[]
  for (method,seed),chunks in raws.items():
   logits=np.concatenate(chunks);metrics,prob=e.probability_metrics(logits,labels);metrics['patient_mean_r2']=float(np.mean(list(metrics['per_output_r2'].values())[:3]));np.save(dest/f'{method}_seed{seed}_prob.npy',prob);local.append({'policy':policy,'ratio':ratio,'repeat':rep,'readout':method,'seed':seed,'val':metrics,'anchor_sid_ce':float(np.nanmean(anchor_ce)) if ratio else None,'anchor_sid_acc':float(np.nanmean(anchor_ok)) if ratio else None,'all_masked_sid_ce':allce/max(1,allcount) if ratio else None,'all_masked_sid_acc':allcorrect/max(1,allcount) if ratio else None,'mask_fraction':float(mask.sum()/data['valid'].sum())})
  dump(dest/'metrics.json',local);(dest/'COMPLETE').touch();rows.extend(local)
 assert before==e.state_sha(model.state_dict());dump(R/'validation_summary.json',rows)
 grouped={}
 for p in ('clean','random','time_span','lead_priority'):
  for ratio in ((0.,) if p=='clean' else (.25,.5,.75)):
   for method in ('layer12','learned_mix'):
    a=[x for x in rows if x['policy']==p and x['ratio']==ratio and x['readout']==method];grouped[f'{method}/{p}/{ratio}']={k:float(np.mean([x['val'][k] for x in a])) for k in ('r2','patient_mean_r2','macro_roc_auc','macro_f1')}
 anchor_results={};frame=pd.read_csv(e.DOWNSTREAM/'val_metadata.csv');ids=frame.entity_id.astype(str).to_numpy();unique=np.unique(ids);groups=[np.flatnonzero(ids==v) for v in unique];rng=np.random.default_rng(270007);boot=[np.concatenate([groups[j] for j in r]) for r in rng.integers(0,len(groups),size=(1000,len(groups)))]
 for ratio in (.25,.5,.75):
  ce={p:np.nanmean(np.stack([np.load(out/f'{p}_r{ratio}_rep{j}/anchor_ce.npy') for j in range(3)]),axis=0) for p in ('random','time_span','lead_priority')}
  anchor_results[str(ratio)]={'mean_ce':{p:float(np.nanmean(v)) for p,v in ce.items()}}
  for p in ('time_span','lead_priority'):
   delta=ce[p]-ce['random'];anchor_results[str(ratio)][p+'_minus_random']={'mean':float(np.nanmean(delta)),'entity_bootstrap_ci95':np.quantile([np.nanmean(delta[ii]) for ii in boot],[.025,.975]).tolist()}
 dump(R/'aggregate.json',grouped);dump(R/'shared_anchor_comparison.json',anchor_results);dump(R/'verification.json',{'frozen_model_unchanged':True,'equal_mask_counts_per_record':True,'shared_anchor_target_per_record_and_repeat':True,'side_branches_recomputed_from_visible_tokens':True,'test_loaded':False})

def main():
 R.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);device=torch.device('cuda');sources=[e.TOKENIZER,e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth',Path(__file__)];before={str(p):e.sha256(p) for p in sources};dump(R/'protocol.json',{'ratios':[.25,.5,.75],'mask_repeats':[0,1,2],'readouts':['E25 layer12','E25 learned_mix'],'seeds':[0,1,2],'teacher_involved':False,'policy_detail':'time_span chooses nearest true detected R times to common anchor; lead_priority masks anchor lead then randomized other leads; exact budget can partially mask boundary beat/lead','primary':'Same anchor SID target under matched mask counts; aggregate masked CE has target-composition confounding','inference':'Transformer valid tokens retained as MASK; raw z_e global and region branches recomputed using visible tokens only','test_loaded':False,'source_sha256':before});status('cache');cache(device);diagnose(device);assert all(e.sha256(Path(p))==v for p,v in before.items());status('complete');(R/'COMPLETE').touch()
if __name__=='__main__':main()
