"""E26 non-destructive information-retention audit. Train/Val only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,copy,csv,hashlib,json,sys,time,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error,roc_auc_score,f1_score,balanced_accuracy_score
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'heartlang').exists());sys.path[:0]=[str(ROOT),str(ROOT/'scripts/curriculum_v1'),str(ROOT/'scripts/experiments')]
import run_e12_continuous_input as e
from scripts.data_prep.ECG_DSA.tokenize_pretrain_qrs import QRSTokenizerLite
from scripts.data_prep.ECG_DSA.ecg_zip_utils import CSV_LEADS,resample_waveforms
LAYERS=(6,9,12);PROXIES=('baseline_early','r_window_peak_to_peak','r_window_signed_peak','post_r_80_160_mean','post_r_200_400_mean','post_r_200_400_abs_peak','wave_rms','wave_mean')
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str))
def status(r,s):(r/'status.txt').write_text(s);print(s,flush=True)
def sha(p):return e.sha256(p)
def metadata_audit(r):
 contract=json.loads((ROOT/'configs/curriculum_v1/data_contract.json').read_text());assert sha(e.DOWNSTREAM/'split_manifest.csv')==contract['split_manifest_sha256'];frames={s:pd.read_csv(e.DOWNSTREAM/f'{s}_metadata.csv') for s in ('train','val')};ids={s:set(f['entity_id'].astype(str)) for s,f in frames.items()};assert not ids['train']&ids['val'];assert [len(frames[s]) for s in frames]==[6640,1344]
 dump(r/'split_audit.json',{'manifest_sha256':contract['split_manifest_sha256'],'counts':{s:len(f) for s,f in frames.items()},'identity_overlap':0,'test_loaded':False});return frames

def proxy(x):
 base=x[:,8:20].mean(1);q=x[:,40:58];late=x[:,68:88]-base[:,None];dom=q[np.arange(len(q)),np.argmax(np.abs(q-base[:,None]),1)]-base
 return np.stack((base,np.ptp(q,axis=1),dom,x[:,56:64].mean(1)-base,late.mean(1),late[np.arange(len(late)),np.argmax(np.abs(late),1)],np.sqrt(np.mean(x*x,1)),x.mean(1)),1).astype(np.float32)

def source_audit(r,frames):
 out=r/'raw_audit';out.mkdir(exist_ok=True)
 if (out/'COMPLETE').exists():return
 tok=QRSTokenizerLite(fs=100,max_len=256,token_len=96);rows=[];examples=[];sources={};zip_checks=[];zips={};raw_arrays={};rng=np.random.default_rng(26006)
 for split,f in frames.items():
  waves=np.load(e.DOWNSTREAM/f'{split}_data.npy',mmap_mode='r');ch=np.load(e.DOWNSTREAM/f'{split}_data_in_chans.npy',mmap_mode='r');ti=np.load(e.DOWNSTREAM/f'{split}_data_in_times.npy',mmap_mode='r');y=np.load(e.DOWNSTREAM/f'{split}_labels.npy').argmax(1)
  chosen=np.concatenate([rng.choice(np.flatnonzero(y==k),min(24,int((y==k).sum())),replace=False) for k in range(4)])
  for idx in chosen:
   row=f.iloc[idx];src=Path(str(row.source_dataset));src=src if src.is_absolute() else ROOT/src
   rawdir=Path(str(src).replace('/pretrain_qrs/','/pretrain_waveforms/').replace('/qrs/','/waveforms/'));ss=str(row.source_split)
   assert ss in ('train','val'),('unexpected source split',ss)
   key=(str(rawdir),ss)
   if key not in raw_arrays:
    if not (rawdir/f'{ss}_data.npy').exists():rows.append({'split':split,'row':int(idx),'available':False,'reason':'source waveform missing','source':str(rawdir)});continue
    raw_arrays[key]=(np.load(rawdir/f'{ss}_data.npy',mmap_mode='r'),pd.read_csv(rawdir/f'{ss}_metadata.csv'))
    summaries={}
    for p in rawdir.glob('*.json'):
     try:
      dd=json.loads(p.read_text());summaries[p.name]={k:v for k,v in dd.items() if any(z in k.lower() for z in ('mean','std','length','sampl','fs','source_zip'))}
     except Exception:pass
    sources[str(rawdir)]={'summary':summaries,'raw_shape':list(raw_arrays[key][0].shape),'source_metadata_columns':list(raw_arrays[key][1].columns)}
   arr,meta=raw_arrays[key];si=int(row.source_index)
   if 'csv_name' in meta and str(meta.iloc[si].csv_name)!=str(row.csv_name):
    matches=np.flatnonzero(meta.csv_name.astype(str).values==str(row.csv_name));assert len(matches)==1;si=int(matches[0])
   x=np.asarray(arr[si],dtype=np.float32)
   if x.shape==(1000,9):x=x.T
   assert x.shape==(9,1000),x.shape
   peaks=tok.qrs_detection(x);mid=tok.trim_edge_qrs(peaks);segments=tok.extract_qrs_segments(x,peaks)
   if len(mid)==0:rows.append({'split':split,'row':int(idx),'available':True,'no_peaks':True});continue
   replay,rc,rt=tok.qrs_to_sequence(segments,mid);assert replay.shape==(256,96)
   err=float(np.max(np.abs(replay-waves[idx])));valid=(ch[idx]>0)&(ti[idx]>0)
   beat=[]
   for j in range(1,len(peaks)-1):
    center=int(peaks[j]);left=int((peaks[j]+peaks[j-1])//2);right=int((peaks[j]+peaks[j+1])//2);actual=right-left
    if actual>96:left=max(center-48,0);right=min(left+96,1000)
    shift=max(48-(center-left),0);right=min(right,left+96-shift)
    beat.append({'pre_ms':(center-left)*10,'post_ms':(right-center-1)*10,'source_points':right-left,'zero_pad_points':96-(right-left),'cropped':actual>96,'has_160ms':right-center>16,'has_300ms':right-center>30,'has_400ms':right-center>40})
   rows.append({'split':split,'row':int(idx),'label':int(y[idx]),'available':True,'max_abs_replay_error':err,'channel_exact':bool(np.array_equal(rc,ch[idx])),'time_exact':bool(np.array_equal(rt,ti[idx])),'n_r_peaks':len(peaks),'beats_used':len(mid),'untruncated_tokens':9*len(mid),'sentence_tokens_dropped':max(0,9*len(mid)-256),'valid_tokens':int(valid.sum()),'coverage':beat})
   if len(examples)<16 and int(idx)%4==int(y[idx]):examples.append((split,int(idx),int(y[idx]),x,peaks,replay))
   # Up to four original high-rate CSVs per class per split, if ZIP remains accessible.
   count=sum(z['split']==split and z['label']==int(y[idx]) for z in zip_checks)
   mr=meta.iloc[si]
   if count<4 and 'source_zip_path' in meta and 'member_name' in meta:
    zp=Path(str(mr.source_zip_path));member=str(mr.member_name)
    if zp.exists():
     try:
      if str(zp) not in zips:zips[str(zp)]=zipfile.ZipFile(zp)
      with zips[str(zp)].open(member) as handle:df=pd.read_csv(handle,usecols=CSV_LEADS)
      high=df[CSV_LEADS].to_numpy(dtype=np.float32);low=resample_waveforms(high,1000);g=np.gcd(len(high),1000);back=resample_poly(low,len(high)//g,1000//g,axis=0,padtype='line')[:len(high)];scale=float(np.std(high));ratio=float(np.sqrt(np.mean((back-high)**2))/max(scale,1e-12));gain=float(np.linalg.lstsq(np.column_stack((low.T.ravel(),np.ones(low.size))),x.ravel(),rcond=None)[0][0]);offset=float(np.mean(x-gain*low.T));fiterr=float(np.sqrt(np.mean((x-gain*low.T-offset)**2)))
      zip_checks.append({'split':split,'row':int(idx),'label':int(y[idx]),'input_samples':len(high),'target_samples':1000,'roundtrip_rmse_over_raw_std':ratio,'raw_to_stored_affine_gain':gain,'raw_to_stored_affine_offset':offset,'affine_fit_rmse':fiterr,'units':'CSV native units; physical calibration not assumed'})
     except Exception as exc:zip_checks.append({'split':split,'row':int(idx),'label':int(y[idx]),'error':str(exc)})
  print('source replay',split,flush=True)
 for z in zips.values():z.close()
 good=[z for z in rows if 'coverage'in z];beats=[b for z in good for b in z['coverage']];summary={'records_requested':192,'records_replayed':len(good),'replay_exact_records':sum(z['max_abs_replay_error']==0 and z['channel_exact'] and z['time_exact'] for z in good),'max_replay_abs_error':max([z['max_abs_replay_error'] for z in good],default=None),'sampling_rate_used_hz':100,'nominal_96_sample_duration_ms':960,'sample_time_span_ms':950,'r_index':48,'max_pre_ms':480,'max_post_ms':470,'beat_count':len(beats),'coverage_note':'Geometric coverage after detected R, not delineated ST/T coverage or clinical ground truth','beat_means':{k:float(np.mean([b[k] for b in beats])) for k in beats[0]} if beats else {},'post_ms_quantiles':np.quantile([b['post_ms'] for b in beats],[0,.1,.5,.9,1]).tolist() if beats else [],'sentence_tokens_dropped':sum(z['sentence_tokens_dropped'] for z in good),'sources':sources,'zip_checks':zip_checks,'highrate_csv_available_records':sum('input_samples'in z for z in zip_checks),'test_loaded':False}
 dump(out/'summary.json',summary);dump(out/'record_replay.json',rows)
 if examples:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axs=plt.subplots(4,2,figsize=(14,12));
  for ax,item in zip(axs.flat,examples):
   s,i,y,x,p,q=item;k=min(2,len(p)-2);center=int(p[k]);l=max(0,center-65);rr=min(1000,center+70);ax.plot((np.arange(l,rr)-center)*10,x[0,l:rr],label='stored 100 Hz waveform');ax.plot((np.arange(96)-48)*10,q[k-1],label='actual 96-point token');ax.axvline(0,color='gray');ax.set_title(f'{s} row {i}, class {y}');ax.set_xlabel('ms relative to detected R');ax.legend(fontsize=7)
  fig.tight_layout();fig.savefig(out/'window_replay.png',dpi=140);plt.close(fig)
 (out/'COMPLETE').touch()

def pure_quantize(tok,z):
 z=torch.nn.functional.normalize(z,p=2,dim=-1);w=tok.quantize.embedding.weight
 distances=z.pow(2).sum(1,keepdim=True)+w.pow(2).sum(1)-2*torch.einsum('bd,nd->bn',z,w)
 ids=distances.argmin(1);q=w[ids];return z+(q-z),ids

@torch.inference_mode()
def extract(r,device):
 out=r/'token_cache';out.mkdir(exist_ok=True)
 if (out/'COMPLETE').exists():return
 tok=e.e4.load_model(e.TOKENIZER,device).eval().requires_grad_(False);model=e.load_continuous_checkpoint(e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth',device).eval().requires_grad_(False);before=(e.state_sha(tok.state_dict()),e.state_sha(model.state_dict()));audit={}
 check=torch.randn(32,96,device=device);zz=tok.encoder(check);q,si=pure_quantize(tok,zz);reference=copy.deepcopy(tok.quantize).eval();qr,_,sir=reference(zz);assert torch.equal(si,sir) and torch.allclose(q,qr,atol=1e-7);del reference
 for split in ('train','val'):
  w=np.load(e.DOWNSTREAM/f'{split}_data.npy',mmap_mode='r');c=np.load(e.DOWNSTREAM/f'{split}_data_in_chans.npy',mmap_mode='r');t=np.load(e.DOWNSTREAM/f'{split}_data_in_times.npy',mmap_mode='r');ys=np.load(e.DOWNSTREAM/f'{split}_labels.npy').argmax(1);chunks={k:[] for k in ('wave','ze','zq','sid','lead','record','label','h6','h9','h12')};norms=[];perturb={k:[] for k in ('gain_cos','offset_cos','shift1_cos','shift2_cos','gain_sid_change','offset_sid_change','shift1_sid_change','shift2_sid_change')}
  for start in range(0,len(w),32):
   x=torch.tensor(np.asarray(w[start:start+32]).copy(),device=device,dtype=torch.float32);ch=torch.tensor(np.asarray(c[start:start+32]).copy(),device=device).long();ti=torch.tensor(np.asarray(t[start:start+32]).copy(),device=device).long();v=tok.valid_mask(x,ch,ti);beats=x[v];z=tok.encoder(beats);zq,sid=pure_quantize(tok,z);ze=torch.zeros(*v.shape,64,device=device);ze[v]=z;zqfull=torch.zeros_like(ze);zqfull[v]=zq;sidfull=torch.zeros_like(ch);sidfull[v]=sid
   br=[];sl=[]
   for j in range(len(x)):
    rng=np.random.default_rng(260006+(start+j)+(0 if split=='train' else 100000))
    for lead in torch.unique(ch[j,v[j]]).cpu().tolist():
     ids=torch.where(v[j]&(ch[j]==lead))[0].cpu().numpy();br.append(j);sl.append(int(rng.choice(ids)))
   bi=torch.tensor(br,device=device);slot=torch.tensor(sl,device=device);selected=x[bi,slot];selectedze=ze[bi,slot];selectedsid=sidfull[bi,slot];norms.extend(selectedze.norm(dim=-1).cpu().tolist());chunks['wave'].append(selected.cpu().numpy());chunks['ze'].append(selectedze.cpu().numpy());chunks['zq'].append(zqfull[bi,slot].cpu().numpy());chunks['sid'].append(selectedsid.cpu().numpy());chunks['lead'].append(ch[bi,slot].cpu().numpy());chunks['record'].append(np.asarray(br)+start);chunks['label'].append(ys[np.asarray(br)+start])
   b=model.backbone;h=b.input_proj(ze)*v[...,None];h=torch.cat((b.cls_token[None,None].expand(len(x),1,-1),h),1);h=b.dropout(h+b.pos_embed);fv=torch.cat((torch.ones(len(x),1,dtype=torch.bool,device=device),v),1)
   for depth,layer in enumerate(b.transformer.layers,1):
    h=e.masked_transformer_forward(SimpleNamespace(layers=[layer]),h,fv)
    if depth in LAYERS:chunks[f'h{depth}'].append(b.norm_layer(h)[bi,slot+1].cpu().numpy())
   # Technical sensitivity, no labels; input transformations are standardized units.
   for name,changed in [('gain',selected*1.1),('offset',selected+.1),('shift1',torch.cat((selected[:,:1],selected[:,:-1]),1)),('shift2',torch.cat((selected[:,:1].expand(-1,2),selected[:,:-2]),1))]:
    zz=tok.encoder(changed);_,ss=pure_quantize(tok,zz);perturb[name+'_cos'].extend(torch.nn.functional.cosine_similarity(selectedze,zz).cpu().tolist());perturb[name+'_sid_change'].extend((selectedsid!=ss).float().cpu().tolist())
   if start%1024==0:print('extract',split,start,len(w),flush=True)
  dest=out/split;dest.mkdir(exist_ok=True)
  for name,values in chunks.items():a=np.concatenate(values);assert np.isfinite(a).all();np.save(dest/f'{name}.npy',a)
  np.save(dest/'targets.npy',proxy(np.load(dest/'wave.npy')));audit[split]={'records':len(w),'sampled_tokens':len(np.load(dest/'sid.npy')),'sampling':'one deterministic random valid beat per measured lead per ECG; at most9','ze_norm_range':[min(norms),max(norms)],'sensitivity':{k:float(np.mean(v)) for k,v in perturb.items()}}
 assert before==(e.state_sha(tok.state_dict()),e.state_sha(model.state_dict()));dump(out/'audit.json',{'frozen_states_unchanged':True,'splits':audit,'test_loaded':False});(out/'COMPLETE').touch()

def regress(xtr,xv,ytr,yv):
 xs=StandardScaler().fit(xtr);ys=StandardScaler().fit(ytr);model=Ridge(alpha=10.,solver='cholesky').fit(xs.transform(xtr),ys.transform(ytr));pred=ys.inverse_transform(model.predict(xs.transform(xv)));return pred,{'per_target_r2':dict(zip(PROXIES,map(float,r2_score(yv,pred,multioutput='raw_values')))),'per_target_mae':dict(zip(PROXIES,map(float,np.mean(abs(yv-pred),0))))}
def group_residual(train,val,gt,gv):
 # Train-only SID/lead means; unseen pairs fall back to train SID mean, unseen SID global mean.
 nt=1024*13;counts=np.bincount(gt,minlength=nt);sums=np.zeros((nt,train.shape[1]));np.add.at(sums,gt,train);means=sums/np.maximum(counts[:,None],1);sc=np.bincount(gt//13,minlength=1024);sm=np.zeros((1024,train.shape[1]));np.add.at(sm,gt//13,train);sm=sm/np.maximum(sc[:,None],1);sm[sc==0]=train.mean(0);means[counts==0]=sm[np.flatnonzero(counts==0)//13];return train-means[gt],val-means[gv],means[gt],means[gv],counts

def probes(r,frames):
 out=r/'probes';out.mkdir(exist_ok=True)
 if (out/'COMPLETE').exists():return
 load=lambda s,k:np.load(r/'token_cache'/s/f'{k}.npy',mmap_mode='r');yt=np.array(load('train','targets'));yv=np.array(load('val','targets'));gt=np.array(load('train','sid'))*13+np.array(load('train','lead'));gv=np.array(load('val','sid'))*13+np.array(load('val','lead'));rt=np.array(load('train','record'));rv=np.array(load('val','record'));results={};predictions={}
 for name in ('wave','ze','zq','h6','h9','h12'):
  print('proxy_probe',name,flush=True);pred,res=regress(np.array(load('train',name)),np.array(load('val',name)),yt,yv);results[name]=res;predictions[name]=pred;np.save(out/f'{name}_proxy_predictions.npy',pred)
 yrt,yrv,ymt,ymv,counts=group_residual(yt,yv,gt,gv);results['sid_lead_train_mean']={'per_target_r2':dict(zip(PROXIES,map(float,r2_score(yv,ymv,multioutput='raw_values'))))}
 ze_t,ze_v,_,_,_=group_residual(np.array(load('train','ze')),np.array(load('val','ze')),gt,gv);p,res=regress(ze_t,ze_v,yrt,yrv);total=ymv+p;results['sid_lead_plus_within_ze']={'per_target_r2':dict(zip(PROXIES,map(float,r2_score(yv,total,multioutput='raw_values')))),'within_group_residual_r2':res['per_target_r2'],'val_seen_sid_lead_fraction':float(np.mean(counts[gv]>0))};np.save(out/'sid_lead_proxy_predictions.npy',ymv);np.save(out/'sid_lead_plus_within_ze_proxy_predictions.npy',total)
 dump(out/'physiology_proxy_probes.json',{'protocol':'Ridge alpha10 fixed, train-only X/Y StandardScaler; unmasked contextual tokens; no validation hyperparameter selection','targets':'operational waveform proxies in standardized signal units, not annotated ST/T landmarks or mV','results':results})
 # Record-level patient subtype probes, with identical train-only preprocessing and fixed Ridge.
 def aggregate(feat,rec,n):
  result=np.zeros((n,feat.shape[1]*2),np.float32)
  for i in range(n):
   q=feat[rec==i]
   if len(q):result[i]=np.r_[q.mean(0),q.max(0)]
  return result
 full_y={s:np.load(e.DOWNSTREAM/f'{s}_labels.npy').argmax(1) for s in ('train','val')};patient={s:full_y[s]<3 for s in full_y};ytr=full_y['train'][patient['train']];yval=full_y['val'][patient['val']];ntr=len(full_y['train']);nv=len(full_y['val'])
 def hist(s,rec,n):
  h=np.zeros((n,1024),np.float32);np.add.at(h,(rec,np.array(load(s,'sid'))),1);h/=h.sum(1,keepdims=True).clip(1);lead=np.zeros((n,13),np.float32);np.add.at(lead,(rec,np.array(load(s,'lead'))),1);lead/=lead.sum(1,keepdims=True).clip(1);return np.c_[h,lead]
 ht=hist('train',rt,ntr);hv=hist('val',rv,nv);classres={};pairpred={}
 def shuffled(a,g,seed):
  out=a.copy();rng=np.random.default_rng(seed)
  for key in np.unique(g):
   ids=np.flatnonzero(g==key);out[ids]=a[rng.permutation(ids)]
  return out
 specs={'sid_lead_hist':(ht,hv),'sid_lead_hist_plus_within_ze':(np.c_[ht,aggregate(ze_t,rt,ntr)],np.c_[hv,aggregate(ze_v,rv,nv)]),'sid_lead_hist_plus_shuffled_within_ze':(np.c_[ht,aggregate(shuffled(ze_t,gt,260016),rt,ntr)],np.c_[hv,aggregate(shuffled(ze_v,gv,260017),rv,nv)])}
 for name in ('wave','ze','zq','h6','h9','h12'):specs[name]=(aggregate(np.array(load('train',name)),rt,ntr),aggregate(np.array(load('val',name)),rv,nv))
 for name,(tx,vx) in specs.items():
  print('subtype_probe',name,flush=True);sc=StandardScaler().fit(tx[patient['train']]);m=Ridge(alpha=10.,solver='cholesky').fit(sc.transform(tx[patient['train']]),np.eye(3)[ytr]);raw=m.predict(sc.transform(vx[patient['val']]));ex=np.exp(raw-raw.max(1,keepdims=True));prob=ex/ex.sum(1,keepdims=True);pairpred[name]=prob;pred=prob.argmax(1);classres[name]={'macro_auc':float(roc_auc_score(np.eye(3)[yval],prob,average='macro',multi_class='ovr')),'macro_f1':float(f1_score(yval,pred,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(yval,pred)),'onehot_r2':float(r2_score(np.eye(3)[yval],prob)),'input_dimension':tx.shape[1]};np.save(out/f'{name}_subtype_probabilities.npy',prob)
 # Entity-cluster paired bootstrap for pre-specified conditional comparison.
 ids=frames['val'].loc[patient['val'],'entity_id'].astype(str).to_numpy();unique=np.unique(ids);groups=[np.flatnonzero(ids==v) for v in unique];rng=np.random.default_rng(260006);deltas=[];base=pairpred['sid_lead_hist'];extra=pairpred['sid_lead_hist_plus_within_ze']
 for _ in range(1000):
  ii=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]);
  if len(np.unique(yval[ii]))==3:deltas.append(roc_auc_score(np.eye(3)[yval[ii]],extra[ii],average='macro',multi_class='ovr')-roc_auc_score(np.eye(3)[yval[ii]],base[ii],average='macro',multi_class='ovr'))
 dump(out/'patient_subtype_probes.json',{'protocol':'ECG-level Train-patient-only Ridge alpha10, Mean+Max of same sampled tokens; Val patients only; no clinical auxiliary training; histogram + within-SID/lead residual adds dimensions and is not capacity matched','train_patients':len(ytr),'val_patients':len(yval),'val_patient_entities':len(unique),'results':classres,'within_ze_minus_sid_hist_macro_auc':classres['sid_lead_hist_plus_within_ze']['macro_auc']-classres['sid_lead_hist']['macro_auc'],'entity_bootstrap_auc_delta_ci95':np.quantile(deltas,[.025,.975]).tolist(),'same_sid_caveat':'Train-only group residualization tests extra signal conditional on SID+lead in this probe, not causal evidence or disease specificity; contextual layers may use entire ECG.'});(out/'COMPLETE').touch()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'results/e26_information_retention');a=p.parse_args();r=a.run;r.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 sources=[Path(__file__),Path(e.__file__),e.TOKENIZER,e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth',ROOT/'scripts/data_prep/ECG_DSA/tokenize_pretrain_qrs.py',ROOT/'scripts/data_prep/ECG_DSA/ecg_zip_utils.py'];before={str(p):sha(p) for p in sources};dump(r/'protocol.json',{'scope':'source replay + token proxy probes + patient-subtype conditional residual probes','layers':LAYERS,'sample':'one random valid beat per measured lead/record, deterministic seed260006','raw_replay':'24 records/class/split, seed26006; max4 original CSV/class/split if accessible','ridge_alpha':10,'hyperparameters_selected_on_val':False,'test_loaded':False,'source_sha256':before});frames=metadata_audit(r);status(r,'raw_replay');source_audit(r,frames);status(r,'feature_extract');extract(r,torch.device('cuda'));status(r,'probes');probes(r,frames);assert all(sha(Path(k))==v for k,v in before.items());dump(r/'source_audit.json',{'sources_unchanged':True,'sha256':before});status(r,'complete');(r/'COMPLETE').touch()
if __name__=='__main__':main()
