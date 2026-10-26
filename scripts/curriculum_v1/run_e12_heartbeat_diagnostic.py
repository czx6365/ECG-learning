#!/usr/bin/env python3
"""E12 existing physiology rules, frozen HeartLang heartbeat summaries, saved Control errors."""
import argparse,ast,csv,hashlib,inspect,json,pickle,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_e4_aggregation as e4
import run_e12_control_diagnostic as core
from extract_information_flow_representations import load_heartlang
import train_morphology_v2_physaware_vq as base
ROOT=e4.ROOT;RUN=core.RUN;E7=core.E7

def summarize_vector(feature,labels):
 report={}
 for split in labels:
  report[split]={e4.LABELS[k]:{name:core.describe(feature[split][labels[split]==k,j]) for j,name in enumerate(feature['names'])} for k in range(3)}
 return report

def effect(a,b):
 a=np.asarray(a);a=a[np.isfinite(a)];b=np.asarray(b);b=b[np.isfinite(b)]
 if len(a)<2 or len(b)<2:return None
 v=((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2)
 return float((a.mean()-b.mean())/np.sqrt(v)) if v>1e-12 else 0.

def simple_feature_probes(features,labels):
 results=[]
 for rep,x in features.items():
  for name,pos,neg in core.PAIRS:
   masks={s:(labels[s]==pos)|(labels[s]==neg) for s in labels};ys={s:(labels[s][masks[s]]==pos).astype(int) for s in labels}
   imp=SimpleImputer(strategy='median').fit(x['train'][masks['train']]);sc=StandardScaler().fit(imp.transform(x['train'][masks['train']]))
   xx={s:sc.transform(imp.transform(x[s][masks[s]])) for s in labels};candidates=[];models=[]
   for C in (.01,.1,1.):
    with warnings.catch_warnings(record=True) as w:
     warnings.simplefilter('always',ConvergenceWarning)
     model=LogisticRegression(C=C,solver='liblinear',max_iter=2000,tol=1e-4,random_state=0).fit(xx['train'],ys['train']);assert not any(issubclass(t.category,ConvergenceWarning) for t in w)
    candidates.append({'C':C,'val':core.binary_metrics(ys['val'],model.decision_function(xx['val']))});models.append(model)
   selected=max(range(3),key=lambda i:candidates[i]['val']['roc_auc']);model=models[selected]
   score=model.decision_function(xx['test']);results.append({'representation':rep,'pair':name,'selected_C':candidates[selected]['C'],'val':candidates[selected]['val'],'test':core.binary_metrics(ys['test'],score),'candidates':candidates})
 core.save('heartbeat_summary_pairwise_probes.json',results)


def main():
 assert (RUN/'core_validation.json').exists(),'Run the core diagnostics first'
 core.status('heartbeat_morphology_extract')
 files=[e4.TOKENIZER,e4.MLM,Path(__file__),Path(inspect.getfile(base))]+[e4.DATA/f'{s}_data.npy' for s in ('train','val','test')]
 before={str(p):e4.sha(p) for p in files}
 payload=torch.load(e4.TOKENIZER,map_location='cpu',weights_only=False);saved=payload.get('args',{});saved=vars(saved) if isinstance(saved,argparse.Namespace) else saved
 tree=ast.parse(inspect.getsource(base.parse_args));defaults={}
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant):
   for kw in node.keywords:
    if kw.arg=='default':
     try:defaults[node.args[0].value.lstrip('-').replace('-','_')]=ast.literal_eval(kw.value)
     except (ValueError,TypeError):pass
 args_names={n.attr for n in ast.walk(ast.parse(inspect.getsource(base.extract_physiology_vectorized))) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='args'}
 params={k:saved[k] if k in saved else defaults[k] for k in sorted(args_names)};args=argparse.Namespace(**params)
 base.set_physiology_schema(base.LEGACY_DISCRETE_FEATURES,base.LEGACY_CONTINUOUS_FEATURES)
 core.save('morphology_rule_config.json',{'actual_params':params,'source':'existing extractor, E3 checkpoint args override source parser defaults','continuous_features':base.CONTINUOUS_FEATURES,'discrete_features':base.DISCRETE_FEATURES,'units':'wave amplitudes in stored waveform normalization units, not claimed clinical mV','missing':'phys extractor continuous validity masks respected','not_clinical_ground_truth':True})
 labels={s:np.load(e4.FEATURES/f'{s}_labels.npy').argmax(1) for s in ('train','val','test')}
 device=torch.device('cuda');torch.set_num_threads(4);model,_=load_heartlang(e4.MLM,device);model.requires_grad_(False)
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 cont=list(base.CONTINUOUS_FEATURES);disc=list(base.DISCRETE_FEATURES);names=[k+'_mean' for k in cont]+[k+'_std' for k in cont]+[k+'_'+v+'_fraction' for k,values in base.DISCRETE_FEATURES.items() for v in values]
 morph={'names':names};token={'names':['token_norm_mean','token_norm_std','relative_token_dispersion','mean_pairwise_cosine','max_distance_from_mean']};records={'names':['valid_token_count','distinct_lead_count','distinct_time_id_count','time_id_span','sid_active_per_record','sid_entropy','mean_ze_zq_squared_distance']}
 fractions={};sid_report={};lead_rows=[]
 for split in labels:
  waves=np.load(e4.DATA/f'{split}_data.npy',mmap_mode='r');sid=np.load(e4.SID/f'{split}_sid.npy',mmap_mode='r');ch=np.load(e4.SID/f'{split}_data_in_chans.npy',mmap_mode='r');ti=np.load(e4.SID/f'{split}_data_in_times.npy',mmap_mode='r')
  valid=np.load(e4.FEATURES/f'{split}_valid_heartbeat_mask.npy');assert np.array_equal(valid,(sid>=0)&(sid<1024))
  mm=np.load(e4.FEATURES/f'{split}_heartlang_mean_max.npy');sf=np.load(e4.FEATURES/f'{split}_sid_frequency.npy');ze=np.load(e4.FEATURES/f'{split}_z_e_tokens.npy',mmap_mode='r');zq=np.load(e4.FEATURES/f'{split}_z_q_tokens.npy',mmap_mode='r')
  n=len(labels[split]);morph[split]=np.full((n,len(names)),np.nan,np.float32);token[split]=np.full((n,5),np.nan,np.float32);records[split]=np.full((n,7),np.nan,np.float32)
  patients=np.flatnonzero(labels[split]<3);disc_per_lead=[]
  for start in range(0,len(patients),16):
   rows=patients[start:start+16];vv=valid[rows];batch=np.asarray(waves[rows]);beats=batch[vv]
   categorical,continuous,cmask=base.extract_physiology_vectorized(beats,args)
   ids=torch.tensor(np.array(sid[rows]),device=device,dtype=torch.long);channels=torch.tensor(np.array(ch[rows]),device=device,dtype=torch.long);times=torch.tensor(np.array(ti[rows]),device=device,dtype=torch.long)
   with torch.inference_mode():alltok,mask=model.forward_feature(ids,in_chan_matrix=channels,in_time_matrix=times)
   assert np.array_equal(mask.cpu().numpy(),vv)
   tok=alltok[:,1:].cpu().numpy();mean=(tok*vv[...,None]).sum(1)/np.maximum(vv.sum(1),1)[:,None];maximum=np.max(np.where(vv[...,None],tok,-np.inf),axis=1);maximum[~vv.any(1)]=0
   assert np.allclose(np.concatenate([mean,maximum],1),mm[rows],atol=1e-5),'Cached HeartLang tokens mismatch'
   offset=0
   for local,row in enumerate(rows):
    take=vv[local];count=int(take.sum());cc=categorical[offset:offset+count];co=continuous[offset:offset+count];maskc=cmask[offset:offset+count];offset+=count
    tokens=tok[local][take]
    if count:
     meanvalues=[float(co[maskc[:,j],j].mean()) if maskc[:,j].any() else np.nan for j in range(4)];stdvalues=[float(co[maskc[:,j],j].std()) if maskc[:,j].any() else np.nan for j in range(4)]
     bins=[float((cc[:,j]==k).mean()) for j,d in enumerate(base.DISCRETE_FEATURES.values()) for k in range(len(d))];morph[split][row]=meanvalues+stdvalues+bins
     norm=np.linalg.norm(tokens,axis=1);mu=tokens.mean(0);units=tokens/np.maximum(norm[:,None],1e-12)
     token[split][row]=[norm.mean(),norm.std(),1-float(np.dot(mu,mu))/max(float((norm**2).mean()),1e-12),float((count*np.square(units.mean(0)).sum()-1)/max(count-1,1)) if count>1 else 1.,np.linalg.norm(tokens-mu,axis=1).max()]
     freq=sf[row];prob=freq[freq>0];tm=ti[row][take];records[split][row]=[count,len(np.unique(ch[row][take])),len(np.unique(tm)),float(tm.max()-tm.min()),len(prob),float(-(prob*np.log(prob)).sum()),float(np.square(ze[row][take]-zq[row][take]).mean())]
     # Per-lead summaries: separate existing lead IDs, avoid inventing clinical lead labels.
     for lead in np.unique(ch[row][take]):
      pick=ch[row][take]==lead
      lead_rows.append({'split':split,'parent_row':int(row),'class':e4.LABELS[int(labels[split][row])],'lead_id':int(lead),'n_tokens':int(pick.sum()),'wide_fraction':float((cc[pick,3]==1).mean()),'st_non_normal_fraction':float((cc[pick,2]!=1).mean()),'t_inverted_fraction':float((cc[pick,1]==1).mean())})
    else:records[split][row,:]=[0,0,0,0,0,0,np.nan]
   print(f'heartbeat {split} {min(start+16,len(patients))}/{len(patients)}',flush=True)
  for kind,feature in [('morphology',morph),('heartlang_token_statistics',token),('record_statistics',records)]:np.save(RUN/f'{split}_{kind}.npy',feature[split])
  fractions[split]={e4.LABELS[k]:sf[(labels[split]==k)&valid.any(1)].mean(0) for k in range(3)}
  sid_report[split]={}
  for name,pos,neg in core.PAIRS:
   a=fractions[split][e4.LABELS[pos]];b=fractions[split][e4.LABELS[neg]];sid_report[split][name]={'JS_divergence_base2':float(jensenshannon(a,b,base=2)**2),'records_positive':int((labels[split]==pos).sum()),'records_negative':int((labels[split]==neg).sum())}
 pd.DataFrame(lead_rows).to_csv(RUN/'per_lead_morphology_records.csv',index=False)
 stats={'morphology':summarize_vector(morph,labels),'heartlang_tokens':summarize_vector(token,labels),'record_quality':summarize_vector(records,labels),'SID_pair_divergence':sid_report}
 effect_rows=[]
 for kind,feature in [('morphology',morph),('heartlang_tokens',token),('record_quality',records)]:
  for name,pos,neg in core.PAIRS[:2]:
   for j,k in enumerate(feature['names']):
    d={s:effect(feature[s][labels[s]==pos,j],feature[s][labels[s]==neg,j]) for s in labels};stable=all(v is not None for v in d.values()) and len(set(np.sign(list(d.values()))))==1 and min(abs(v) for v in d.values())>=.2
    effect_rows.append({'family':kind,'pair':name,'feature':k,'d_positive_minus_negative':d,'same_sign_all3_and_minabs_point2':stable})
 core.save('heartbeat_descriptive.json',stats);core.save('heartbeat_effect_sizes.json',effect_rows)
 sid_features=[]
 for name,pos,neg in core.PAIRS[:2]:
  diff=fractions['train'][e4.LABELS[pos]]-fractions['train'][e4.LABELS[neg]];top=np.argsort(np.abs(diff))[-20:][::-1]
  for code in top:
   deltas={s:float(fractions[s][e4.LABELS[pos]][code]-fractions[s][e4.LABELS[neg]][code]) for s in labels};sid_features.append({'pair':name,'sid':int(code),'selected_train_abs_diff_only':True,'fraction_difference':deltas,'same_sign_all_splits':len(set(np.sign(list(deltas.values()))))==1})
 core.save('SID_train_selected_difference_replication.json',sid_features)
 case_rows=[];groups={}
 xt=np.load(e4.FEATURES/'test_heartlang_mean_max.npy');control=labels['test']==2;orig=np.flatnonzero(control)
 pca=np.load(RUN/'test_patient_PCA20.npy');pidx=np.load(RUN/'test_patient_parent_indices.npy');coords={int(row):pca[i] for i,row in enumerate(pidx)}
 for seed in (0,1,2):
  pred=np.load(E7/f'CE_seed{seed}/test_raw_outputs.npy').argmax(1);report={}
  for k in range(4):
   sel=control&(pred==k);report[e4.LABELS[k]]={'n_records':int(sel.sum()),'features':{},'mean_SID_frequency':np.load(e4.FEATURES/'test_sid_frequency.npy')[sel].mean(0).tolist() if sel.any() else []}
   for family,f in [('morphology',morph),('heartlang_tokens',token),('record_quality',records)]:report[e4.LABELS[k]]['features'][family]={name:core.describe(f['test'][sel,j]) for j,name in enumerate(f['names'])}
   report[e4.LABELS[k]]['representation_PCA']={f'pc{j+1}':core.describe([coords[int(row)][j] for row in np.flatnonzero(sel)]) for j in range(5)}
  groups[str(seed)]=report
  for row in orig:
   case={'seed':seed,'parent_test_row':int(row),'true_label':'Clinical Control','E7_prediction':e4.LABELS[int(pred[row])],'correct':bool(pred[row]==2),'PC1':float(coords[int(row)][0]),'PC2':float(coords[int(row)][1])}
   for family,f in [('morphology',morph),('tokens',token),('quality',records)]:case.update({family+'_'+name:float(f['test'][row,j]) for j,name in enumerate(f['names'])})
   case_rows.append(case)
 core.save('Control_error_group_diagnostics.json',groups);pd.DataFrame(case_rows).to_csv(RUN/'Control_error_cases.csv',index=False)
 # Additional simple probes quantify existing numeric summaries, not a new neural model.
 with threadpool_limits(limits=4):simple_feature_probes({'morphology_summary':{s:morph[s] for s in labels},'sid_frequency':{s:np.load(e4.FEATURES/f'{s}_sid_frequency.npy') for s in labels},'heartlang_token_statistics':{s:token[s] for s in labels}},labels)
 assert all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in initial.items())
 assert all(e4.sha(p)==before[str(p)] for p in files)
 core.save('heartbeat_validation.json',{'frozen_heartlang_all_state_tensors_unchanged':True,'source_and_wave_hashes_unchanged':True,'mean_max_matches_existing_cache':True,'patient_records':{s:int((v<3).sum()) for s,v in labels.items()},'morphology_validity_masks_respected':True,'classification_labels_not_used_by_frozen_inference':True})
 core.save('heartbeat_source_fingerprints.json',before);core.status('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
