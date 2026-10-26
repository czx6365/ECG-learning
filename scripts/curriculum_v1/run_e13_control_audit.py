#!/usr/bin/env python3
import argparse,copy,json,shutil,hashlib,re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
import run_e4_aggregation as e4
import run_e10_patient_diagnostic as e10
ROOT=e4.ROOT;RUN=ROOT/'results/curriculum_v1/e13_control_audit_readout_20261012';OLD=e10.RUN
SPLITS=('train','val','test');NAMES=e4.LABELS[:3]
def save(f,x):e4.dump(RUN/f,x)
def status(x):(RUN/'status.txt').write_text(x);print(x,flush=True)
def describe(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)]
 return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p25':float(np.percentile(a,25)) if len(a) else None,'p75':float(np.percentile(a,75)) if len(a) else None,'min':float(a.min()) if len(a) else None,'max':float(a.max()) if len(a) else None}
def dist(series):
 x=series.dropna().astype(str);v=x.value_counts();return {'missing':int(series.isna().sum()),'nonmissing':len(x),'unique':len(v),'top20_counts':v.head(20).to_dict(),'top1_fraction_nonmissing':float(v.iloc[0]/len(x)) if len(x) else None}
def audit(labels):
 status('metadata_audit');meta={s:pd.read_csv(e4.DATA/f'{s}_metadata.csv') for s in SPLITS};out={};identities={};recordkeys={};fields={}
 for s in SPLITS:
  assert len(meta[s])==len(labels[s]) and np.array_equal(meta[s].label_id.to_numpy(),labels[s]);d=meta[s];valid=np.load(e4.FEATURES/f'{s}_valid_heartbeat_mask.npy');ch=np.load(e4.SID/f'{s}_data_in_chans.npy');times=np.load(e4.SID/f'{s}_data_in_times.npy');wave=np.load(e4.DATA/f'{s}_data.npy',mmap_mode='r')
  counts=valid.sum(1);leadcounts=np.array([len(np.unique(ch[i][valid[i]])) for i in range(len(d))]);leadsets=pd.Series([','.join(map(str,np.unique(ch[i][valid[i]]))) for i in range(len(d))]);recordkeys[s]=set(d.loc[labels[s]<3,'csv_name'].dropna().astype(str));identities[s]=set(d.loc[labels[s]<3,'entity_id'].dropna().astype(str));out[s]={}
  fields[s]={col:{NAMES[k]:{'available':int(d.loc[labels[s]==k,col].notna().sum()),'records':int((labels[s]==k).sum())} for k in range(3)} for col in d.columns}
  for k,name in enumerate(NAMES):
   ix=np.flatnonzero(labels[s]==k);group=d.iloc[ix];per=group.groupby('entity_id').size();fn=group.csv_name.dropna().astype(str);nonfinite=0;nanrecords=0;wavehash={}
   for row in ix:
    a=np.asarray(wave[row]);n=int((~np.isfinite(a)).sum());nonfinite+=n;nanrecords+=n>0;key=hashlib.sha256(a.tobytes()).hexdigest();wavehash[key]=wavehash.get(key,0)+1
   sources={col:dist(group[col]) for col in ('source_dataset','source_cohort','source_zip','source_split','cohort','original_label_name','selected_by') if col in group}
   datecol='心电图时间（日期）';dates=group[datecol].astype('string');years=dates.str.extract(r'((?:19|20)\d{2})',expand=False)
   entitylabels=group[['entity_id','patient_sn']].dropna().drop_duplicates().groupby('entity_id').size()
   out[s][name]={'n_records':len(ix),'n_entities':int(group.entity_id.nunique()),'missing_entity_id':int(group.entity_id.isna().sum()),'n_patient_sn':int(group.patient_sn.nunique()),'records_per_entity':describe(per),'records_per_entity_histogram':per.value_counts().sort_index().to_dict(),'entities_multiple_records':int((per>1).sum()),'top_entity_record_share':float(per.max()/len(ix)), 'duplicate_csv_name_excess':int(len(fn)-fn.nunique()),'duplicate_waveform_excess_within_class_split':int(sum(v-1 for v in wavehash.values())),'entities_multiple_patient_sn':int((entitylabels>1).sum()),'valid_tokens':describe(counts[ix]),'empty_token_records':int((counts[ix]==0).sum()),'distinct_lead_ids':describe(leadcounts[ix]),'lead_id_sets':dist(leadsets.iloc[ix]),'expected_clinical_lead_completeness':'cannot determine from numeric input IDs alone; stored lead coverage is reported','stored_tensor_shape_per_record':list(wave.shape[1:]),'nonfinite_wave_values':nonfinite,'records_nonfinite_wave':int(nanrecords),'raw_ECG_length_seconds':'unavailable in inspected metadata/cache; fixed padded heartbeat tensor is not original ECG duration','preprocessing_success_rate':'unknown: only retained dataset present, no source attempt/failure denominator','source_fields':sources,'ECG_date':{'raw_nonmissing':int(group[datecol].notna().sum()),'year_extractable':int(years.notna().sum()),'year_distribution':dist(years),'full_date_unique':int(group[datecol].nunique()),'time_of_day_nonmissing':int(group['心电图时间（时刻）'].notna().sum())}}
 assert not identities['train']&identities['val'] and not identities['train']&identities['test'] and not identities['val']&identities['test']
 allpatient=pd.concat([meta[s].loc[labels[s]<3].assign(audit_split=s) for s in SPLITS]);conflict=allpatient.groupby('entity_id').label_id.nunique();sourcecols=[x for x in allpatient.columns if re.search('source|设备|批次|时间|日期|device|batch',x,re.I)]
 globalinfo={'patient_entity_disjoint':True,'cross_split_csv_name_intersection':{a+'_'+b:len(recordkeys[a]&recordkeys[b]) for a,b in [('train','val'),('train','test'),('val','test')]},'entities_multiple_existing_class_labels':int((conflict>1).sum()),'available_source_time_related_columns':sourcecols,'device_or_acquisition_batch_fields':[x for x in allpatient.columns if re.search('设备|批次|device|batch',x,re.I)],'all_existing_columns':list(meta['train'].columns),'clinical_fields':'existing clinical fields inventoried for missingness only; no clinical subtype or new labels inferred'}
 # Patient composition across full split, not independently pooled records.
 pooled={}
 for k,name in enumerate(NAMES):
  d=allpatient[allpatient.label_id==k];counts=d.groupby('entity_id').size();pooled[name]={'records':len(d),'entities':int(d.entity_id.nunique()),'records_per_entity':describe(counts),'multiple_record_entities':int((counts>1).sum()),'maximum_records_one_entity':int(counts.max()),'source_fields':{x:dist(d[x]) for x in ('source_dataset','source_cohort','source_zip','source_split')},'ECG_year':dist(d['心电图时间（日期）'].astype('string').str.extract(r'((?:19|20)\d{2})',expand=False))}
 save('metadata_audit.json',{'splits':out,'pooled':pooled,'global':globalinfo});save('metadata_field_availability.json',fields)

def metrics(raw,y):
 m,s=e10.metrics(raw,y,'B_patient_only');target=y.argmax(1)
 for name,a,b in [('control_vs_omi_auc',2,0),('control_vs_nomi_auc',2,1),('omi_vs_nomi_auc',0,1)]:
  mask=(target==a)|(target==b);m[name]=float(roc_auc_score(target[mask]==a,raw[mask,a]-raw[mask,b]))
 return m,s
@torch.no_grad()
def evaluate(model,x,y):
 model.eval();raw=np.concatenate([model(x[i:i+96]).cpu().numpy() for i in range(0,len(y),96)]);m,s=metrics(raw,y.cpu().numpy());return m,s,raw

def train_mean(seed,arrays,labels,args,device):
 out=RUN/f'Mean_seed{seed}';out.mkdir();e4.set_seed(seed)
 # Match E7/E10 initialization construction: initialize a 4-output head, keep first 3 rows.
 original=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,4));model=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,3))
 state=original.state_dict();state['3.weight']=state['3.weight'][:3].clone();state['3.bias']=state['3.bias'][:3].clone();model.load_state_dict(state);model.to(device);torch.save(state,out/'initial_state.pth')
 # E10 has only one sequential initialization after set_seed; restore RNG to that construction for dropout.
 e4.set_seed(seed);unused=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,4));del unused
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay);steps=len(labels['train'])//64;target=labels['train'].argmax(1);generator=torch.Generator().manual_seed(seed);best=-float('inf');best_epoch=None;beststate=None;history=[]
 for epoch in range(1,31):
  model.train();order=torch.randperm(len(target),generator=generator).to(device)
  for step in range(steps):
   ix=order[step*64:(step+1)*64];lr=e4.lr_for_step((epoch-1)*steps+step,30*steps,5*steps,args)
   for group in opt.param_groups:group['lr']=lr
   opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(arrays['train'][ix]),target[ix]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  val,_,_=evaluate(model,arrays['val'],labels['val']);history.append({'epoch':epoch,'val':val})
  if val['macro_roc_auc']>best:best=val['macro_roc_auc'];best_epoch=epoch;beststate=copy.deepcopy(model.state_dict())
 model.load_state_dict(beststate);val,_,vr=evaluate(model,arrays['val'],labels['val']);test,score,raw=evaluate(model,arrays['test'],labels['test']);torch.save({'model':{k:v.cpu() for k,v in beststate.items()},'best_epoch':best_epoch},out/'checkpoint-best.pth')
 np.save(out/'val_raw_outputs.npy',vr);np.save(out/'test_raw_outputs.npy',raw);np.save(out/'test_scores.npy',score);e4.dump(out/'history.json',history)
 result={'method':'Mean','seed':seed,'input_dim':768,'hidden':256,'trainable_parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'steps_per_epoch':steps,'total_updates':steps*30,'val':val,'test':test};save(f'Mean_seed{seed}/summary.json',result);print(json.dumps(result),flush=True);return result

def main():
 RUN.mkdir(parents=True,exist_ok=False);count=lambda d,h:(d+4)*h+3;budget=count(768,256);hidden=min(range(1,1025),key=lambda h:abs(count(1536,h)-budget));assert hidden==128 and count(1536,hidden)==197123 and budget==197635
 protocol={'seeds':[0,1,2],'methods':{'Mean':{'input':768,'hidden':256,'parameters':budget},'MeanMax':{'input':1536,'hidden':hidden,'parameters':count(1536,hidden)}},'parameter_formula':'(input+4)*hidden+3; Linear biases included, no LayerNorm','matching':'integer hidden1..1024 nearest Mean768→256→3 budget, no projection; difference512=.2591%','epochs':30,'batch':64,'optimizer':'AdamW','lr':.0005,'min_lr':.00001,'warmup_lr':.000001,'warmup_epochs':5,'weight_decay':.05,'clip':3,'dropout':.1,'loss':'plain CE3','selection':'patient-val Macro OVR AUC; earliest tie; no test tuning','features_standardized':False,'reused':'E10 MeanMax three completed seeds; saved logits extended with pairwise metrics, no repeat training/test inference','initialization':'Mean original PyTorch 4-output initialization then first3 outputs; E10 MeanMax uses same-seed E7 initial shared layer + first3 output rows; architecture-dependent initial weights/RNG, not tensor-identical','pair_auc':'positive first named class; logit difference for every pair; not OVR AUC','tokenizer':str(e4.TOKENIZER),'heartlang':str(e4.MLM),'features':str(e4.FEATURES),'audit_limit':'existing fields only; original acquisition duration/device/failure denominator may be unavailable'};save('protocol.json',protocol)
 files=[e4.TOKENIZER,e4.MLM,Path(__file__),ROOT/'scripts/curriculum_v1/run_e10_patient_diagnostic.py']+[e4.FEATURES/f'{s}_{n}.npy' for s in SPLITS for n in ('heartlang_mean','heartlang_mean_max','labels','valid_heartbeat_mask')]+[e4.DATA/f'{s}_metadata.csv' for s in SPLITS]+[OLD/f'B_patient_only_seed{k}'/f'{s}_raw_outputs.npy' for k in range(3) for s in ('val','test')]
 before={str(p):e4.sha(p) for p in files};save('source_fingerprints.json',before);shutil.copy2(OLD/'frozen_split_audit.json',RUN/'frozen_split_audit.json')
 y={s:np.load(e4.FEATURES/f'{s}_labels.npy') for s in SPLITS};audit({s:v.argmax(1) for s,v in y.items()});device=torch.device('cuda');torch.set_num_threads(4);labels={};arrays={};indices={}
 for s in SPLITS:
  ix=np.flatnonzero(y[s][:,3]==0);indices[s]=ix;assert np.array_equal(ix,np.load(OLD/f'{s}_parent_patient_indices.npy'));np.save(RUN/f'{s}_parent_patient_indices.npy',ix);labels[s]=torch.tensor(y[s][ix,:3],device=device,dtype=torch.float32);arrays[s]=torch.tensor(np.load(e4.FEATURES/f'{s}_heartlang_mean.npy')[ix],device=device,dtype=torch.float32)
 args=argparse.Namespace(lr=.0005,min_lr=.00001,warmup_lr=.000001,weight_decay=.05);results=[]
 for seed in range(3):
  out=RUN/f'MeanMax_seed{seed}';shutil.copytree(OLD/f'B_patient_only_seed{seed}',out);old=json.loads((out/'summary.json').read_text());row={**old,'method':'MeanMax','input_dim':1536,'hidden':128,'reused_from':str(OLD/f'B_patient_only_seed{seed}')}
  for s in ('val','test'):row[s]=metrics(np.load(out/f'{s}_raw_outputs.npy'),y[s][indices[s],:3])[0]
  save(f'MeanMax_seed{seed}/summary.json',row);results.append(row)
 for seed in range(3):status(f'training_Mean_seed{seed}');results.append(train_mean(seed,arrays,labels,args,device));save('partial_results.json',results)
 for row in results:
  out=RUN/f"{row['method']}_seed{row['seed']}";history=json.loads((out/'history.json').read_text());assert max(history,key=lambda x:x['val']['macro_roc_auc'])['epoch']==row['best_epoch'];assert row['test']==metrics(np.load(out/'test_raw_outputs.npy'),y['test'][indices['test'],:3])[0];assert row['test']['n_records']==683 and row['trainable_parameters']==protocol['methods'][row['method']]['parameters']
 assert all(e4.sha(p)==before[str(p)] for p in files);save('comparison.json',{'protocol':protocol,'results':results});save('report_validation.json',{'frozen_sources_hashes_unchanged':True,'original_patient_rows_match_E10':True,'six_saved_test_outputs_metrics_recomputed':True,'six_val_checkpoints_verified':True,'parameters_matched_without_projection':True,'metadata_labels_match_feature_labels':True,'patient_entity_disjoint':True,'no_class_weights_auxiliary_hierarchy':True});status('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
