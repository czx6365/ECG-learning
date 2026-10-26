#!/usr/bin/env python3
"""E14 read-only source association and saved-prediction diagnostic, no model training."""
import csv,hashlib,json,pickle,re,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import run_e4_aggregation as e4
ROOT=e4.ROOT;RUN=ROOT/'results/curriculum_v1/e14_original_association_audit_20261012';E7=ROOT/'results/curriculum_v1/e7_objective_20261011';E12=ROOT/'results/curriculum_v1/e12_control_diagnostic_20261012';LABELDIR=ROOT/'datasets/ecg_datasets/ECG_DSA/labels';RAW=ROOT/'datasets/标签_Label.xlsx';ZIP=ROOT/'datasets/患者/data_0521.zip';PAT=ROOT/'datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label3';LM={'OMI':0,'NOMI':1,'对照组':2}
def save(f,x):e4.dump(RUN/f,x)
def status(s):(RUN/'status.txt').write_text(s);print(s,flush=True)
def parts(x):return [s.strip() for s in re.split('[;；]',str(x)) if s.strip() and s.strip()!='nan']
def canonical(x):return str(x).replace('.xml','.csv')
def describe(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)];return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'q25':float(np.percentile(a,25)) if len(a) else None,'q75':float(np.percentile(a,75)) if len(a) else None}
def dateparse(x):
 for fmt in ('%m-%d-%Y','%Y-%m-%d','%m/%d/%Y','%Y/%m/%d','%Y-%m-%d %H:%M:%S'):
  try:return pd.to_datetime(x,format=fmt).strftime('%Y-%m-%d')
  except (ValueError,TypeError):pass
 return None

def source_audit():
 status('raw_label_association');raw=pd.read_excel(RAW);expanded=pd.read_csv(LABELDIR/'expanded_label3_data_0521.csv');dedup=pd.read_csv(LABELDIR/'expanded_label3_data_0521_dedup.csv');conflicts=pd.read_csv(LABELDIR/'conflicting_csv_labels_data_0521.csv');members={};edges=[];temporal=[];problems=[]
 with zipfile.ZipFile(ZIP) as z:
  for info in z.infolist():
   if info.filename.lower().endswith('.csv') and not info.filename.startswith('__MACOSX/'):
    members.setdefault(Path(info.filename).name,[]).append(info.filename)
  for idx,row in raw.iterrows():
   filenames=parts(row['file_name']);dates=parts(row['心电图时间（日期）']);times=parts(row['心电图时间（时刻）']);label=str(row['标签']);sn=str(row['patient_sn'])
   if len(filenames)!=len(dates) or len(filenames)!=len(times):problems.append({'original_row_index':int(idx),'issue':'file_date_time_list_length_mismatch','files':len(filenames),'dates':len(dates),'times':len(times)})
   for i,name in enumerate(filenames):
    csvname=canonical(name);aligned=len(filenames)==len(dates)==len(times);dt=dateparse(dates[i]) if aligned else None;tm=times[i] if aligned else None
    edges.append({'original_row_index':int(idx),'excel_row':int(idx+2),'clinical_record_serial':str(row['序号']),'patient_sn':sn,'source_label':label,'label_id':LM.get(label),'source_file_name_list':str(row['file_name']),'file_position':i,'file_name':name,'csv_name_expected':csvname,'zip_matches':len(members.get(csvname,[])),'member_name':members.get(csvname,[''])[0],'ECG_date_positionally_aligned':dt,'ECG_time_positionally_aligned':tm,'date_time_list_alignment':aligned,'clinical_record_time':str(row['冠脉造影时间'])})
  edge=pd.DataFrame(edges);edge.to_csv(RUN/'原表到ECG关联明细.csv',index=False)
  tuples=lambda d:Counter((int(v.original_row_index),str(v.file_name),int(v.label_id),str(v.patient_sn)) for v in d.itertuples())
  assert tuples(edge)==tuples(expanded),'Source expansion differs from original rows and lists'
  for row in expanded.itertuples():
   orig=raw.iloc[int(row.original_row_index)];assert str(row.source_file_name_list)==str(orig['file_name']);assert str(row.label_name)==str(orig['标签']);assert int(row.label_id)==LM[str(orig['标签'])]
  byfile=edge.groupby('csv_name_expected');conflicting=byfile.label_id.nunique();conflict_names=set(conflicting[conflicting>1].index);repeated=byfile.size();crosssn=byfile.patient_sn.nunique();before=edge[~edge.csv_name_expected.isin(conflict_names)];expected=before.drop_duplicates('csv_name_expected',keep='first')
  assert set(conflicts.csv_name_expected)==conflict_names
  assert tuples(expected)==tuples(dedup),'Existing dedup differs from conflict exclusion and first same-label association'
  conflict_summary=[]
  for fn,group in byfile:
   if len(group)>1:
    conflict_summary.append({'csv_name':fn,'raw_label_links':len(group),'labels':sorted(group.source_label.unique()),'patient_sn_count':int(group.patient_sn.nunique()),'original_rows':list(map(int,group.original_row_index)),'same_label':group.label_id.nunique()==1,'mapped_ECG_datetimes':sorted(set(str(a)+' '+str(b) for a,b in zip(group.ECG_date_positionally_aligned,group.ECG_time_positionally_aligned)))})
  save('raw_repeated_file_links.json',conflict_summary)
  status('raw_ECG_content_verification');contentgroups=defaultdict(list);headercounts=Counter();linecounts=Counter();hashes={};tendpoints=Counter();inventory=[]
  for i,fn in enumerate(sorted(set(edge.csv_name_expected))):
   matches=members.get(fn,[])
   if len(matches)!=1:continue
   blob=z.read(matches[0]);sha=hashlib.sha256(blob).hexdigest();hashes[fn]=sha;contentgroups[sha].append(fn);lines=blob.splitlines();header=lines[0].decode('utf-8-sig');headercounts[header]+=1;linecounts[len(lines)-1]+=1
   head=next(csv.reader([header]));first=next(csv.reader([lines[1].decode('utf-8-sig')])) if len(lines)>1 else [];last=next(csv.reader([lines[-1].decode('utf-8-sig')])) if len(lines)>1 else [];tp=(first[head.index('t')],last[head.index('t')]) if 't' in head and first else None
   tendpoints[str(tp)]+=1;inventory.append({'csv_name':fn,'member_name':matches[0],'bytes':len(blob),'data_rows':len(lines)-1,'sha256':sha,'columns':header,'t_first_last':str(tp)})
   if i%1000==0:print('ECG verified',i,flush=True)
  pd.DataFrame(inventory).to_csv(RUN/'真实ECG文件核验.csv',index=False)
 current=[];mismatch=[];snmap={};selectedrows=set();allbase=[];array_mismatch=[]
 for split in ('train','val','test'):
  meta=pd.read_csv(e4.DATA/f'{split}_metadata.csv');base=pd.read_csv(PAT/f'{split}_metadata.csv');allbase.append(base.assign(audit_split=split));features=np.load(e4.FEATURES/f'{split}_labels.npy').argmax(1);wave=np.load(e4.DATA/f'{split}_data.npy',mmap_mode='r');bw=np.load(PAT/f'{split}_data.npy',mmap_mode='r');ch=np.load(e4.DATA/f'{split}_data_in_chans.npy');bc=np.load(PAT/f'{split}_data_in_chans.npy');ti=np.load(e4.DATA/f'{split}_data_in_times.npy');bt=np.load(PAT/f'{split}_data_in_times.npy');first=base.drop_duplicates('original_row_index',keep='first');selected_expected=set(first.index)
  chosen=meta[meta.label_id<3];assert set(chosen.source_index.astype(int))==selected_expected
  for parent,row in chosen.iterrows():
   ori=int(row.original_row_index);b=base.iloc[int(row.source_index)];orig=raw.iloc[ori];fn=str(row.csv_name);label=int(row.label_id);sn=str(row.patient_sn);selectedrows.add(ori)
   checks={'patient_sn_equal_raw':sn==str(orig['patient_sn']),'entity_derivation':'patient:'+sn==str(row.entity_id),'label_equal_original':label==LM[str(orig['标签'])],'filename_equal_parent':fn==str(b.csv_name),'csv_equals_xml_extension':fn==canonical(row.file_name),'file_in_original_list':str(row.file_name) in parts(orig['file_name']),'member_exact':str(row.member_name) in members.get(fn,[]),'features_label_equal':label==int(features[parent]),'source_file_list_equal':str(row.source_file_name_list)==str(orig['file_name'])}
   if not all(checks.values()):mismatch.append({'split':split,'parent_row':int(parent),'checks':checks})
   if not np.array_equal(wave[parent],bw[int(row.source_index)]) or not np.array_equal(ch[parent],bc[int(row.source_index)]) or not np.array_equal(ti[parent],bt[int(row.source_index)]):array_mismatch.append({'split':split,'parent_row':int(parent)})
   links=edge[(edge.original_row_index==ori)&(edge.csv_name_expected==fn)];assert len(links)==1;linked=links.iloc[0].to_dict();linked.update({'split':split,'parent_row':int(parent),'entity_id':str(row.entity_id),'current_label':e4.LABELS[label],'raw_csv_sha256':hashes.get(fn)});current.append(linked)
 cur=pd.DataFrame(current);cur.to_csv(RUN/'当前记录完整来源追溯.csv',index=False);baseall=pd.concat(allbase,ignore_index=True);assert tuples(baseall)==tuples(dedup)
 sets=cur.groupby('entity_id').label_id.apply(lambda s:tuple(sorted(set(map(int,s)))));multi=set(sets[sets.apply(len)>1].index);onlycontrol=set(sets[sets== (2,)].index);controlmulti=set(sets[sets.apply(lambda x:2 in x and len(x)>1)].index)
 timeline=cur[cur.entity_id.isin(multi)].copy();timeline['ECG_sort_date']=pd.to_datetime(timeline.ECG_date_positionally_aligned,errors='coerce');timeline=timeline.sort_values(['entity_id','ECG_sort_date','ECG_time_positionally_aligned','original_row_index'],na_position='last');timeline.drop(columns=['ECG_sort_date']).to_csv(RUN/'多类别entity时间与文件逐条追溯.csv',index=False)
 timeconf=[]
 for fn,d in edge.groupby('csv_name_expected'):
  aligned=d[d.date_time_list_alignment & d.ECG_date_positionally_aligned.notna()];vals=set(zip(aligned.ECG_date_positionally_aligned,aligned.ECG_time_positionally_aligned))
  if len(vals)>1:timeconf.append({'csv_name':fn,'distinct_metadata_datetime_count':len(vals),'raw_rows':list(map(int,d.original_row_index)),'labels':sorted(d.source_label.unique()),'present_in_current':fn in set(cur.csv_name_expected),'metadata_datetimes':sorted(str(a)+' '+str(b) for a,b in vals)})
 save('same_file_metadata_time_conflicts.json',timeconf)
 repeatsame=0;sametime_diff_labels=0
 for ent,d in timeline.groupby('entity_id'):
  repeatsame+=int(len(d)-d.csv_name_expected.nunique())
  for dt,group in d.groupby(['ECG_date_positionally_aligned','ECG_time_positionally_aligned'],dropna=False):
   if pd.notna(dt[0]) and group.label_id.nunique()>1:sametime_diff_labels+=1
 excluded=sorted(set(range(len(raw)))-selectedrows);unused=set(members)-set(edge.csv_name_expected)
 summary={'original_workbook_rows':len(raw),'original_workbook_columns':len(raw.columns),'sheet':'Sheet1','raw_labels':raw['标签'].value_counts().to_dict(),'raw_patient_sn_unique':int(raw.patient_sn.nunique()),'raw_file_links':len(edge),'raw_unique_referenced_csv':int(edge.csv_name_expected.nunique()),'zip_csv_members':sum(map(len,members.values())),'zip_duplicate_basenames':int(sum(len(v)>1 for v in members.values())),'missing_or_ambiguous_original_links':int((edge.zip_matches!=1).sum()),'zip_unreferenced_csv':len(unused),'original_expansion_exactly_reproduced':True,'expanded_rows':len(expanded),'same_label_duplicate_links_removed':int(len(before)-len(expected)),'conflicting_csv_files':len(conflict_names),'historically_excluded_conflicting_links':int(edge.csv_name_expected.isin(conflict_names).sum()),'conflicting_file_labels':Counter('/'.join(x['labels']) for x in conflict_summary if not x['same_label']),'files_multiple_patient_sn':int((crosssn>1).sum()),'dedup_exactly_reproduced':True,'dedup_rows':len(dedup),'current_rows':len(cur),'original_rows_absent_current':excluded,'absent_rows_labels':[str(raw.iloc[i]['标签']) for i in excluded],'current_association_mismatch_count':len(mismatch),'current_tensor_parent_mismatch_count':len(array_mismatch),'current_unique_csv':int(cur.csv_name_expected.nunique()),'current_unique_original_row_index':int(cur.original_row_index.nunique()),'current_entity_to_multiple_patient_sn':int((cur.groupby('entity_id').patient_sn.nunique()>1).sum()),'current_patient_sn_to_multiple_entity':int((cur.groupby('patient_sn').entity_id.nunique()>1).sum()),'raw_list_alignment_mismatch_records':len(problems),'same_file_metadata_time_conflict_count':len(timeconf),'current_same_file_time_conflicts':sum(x['present_in_current'] for x in timeconf),'current_multilabel_same_file_reuse_excess':repeatsame,'current_multilabel_same_ECG_datetime_different_labels_groups':sametime_diff_labels,'raw_ECG_content_hash_groups_different_filenames':[v for v in contentgroups.values() if len(v)>1],'raw_csv_columns':dict(headercounts),'raw_csv_data_row_distribution':dict(linecounts),'raw_csv_relative_t_endpoints':dict(tendpoints),'current_missing_positionally_mapped_dates':int(cur.ECG_date_positionally_aligned.isna().sum()),'original_date_semantics':'Only positions where filename/date/time list lengths agree are mapped; no clinical chronology inference. CSV t lacks explicit absolute acquisition timestamp/patient ID columns. Filename prefix is not assumed acquisition date.','initial_expansion_source_code':'Standalone initial expansion script not found in existing project; original→expanded→dedup relation independently reproduced from saved artifacts, not a claimed recovered execution log','clinical_Control_label_generation':'Original Excel 标签=对照组 propagated to expanded file label_name and numeric2; currentbuilder maps 对照组→2 and onehot4. Clinical criteria for original 标签 not recoverable from these construction scripts.'}
 save('association_summary.json',summary);save('association_issues.json',{'current_mismatches':mismatch,'tensor_parent_mismatches':array_mismatch,'raw_file_date_time_list_length_mismatches':problems});return cur,onlycontrol,controlmulti

def subgroup_diagnostic(cur,onlycontrol,controlmulti):
 status('saved_E7_group_diagnostic');labels=np.load(e4.FEATURES/'test_labels.npy').argmax(1);metadata=pd.read_csv(e4.DATA/'test_metadata.csv');masks={'Control_only':(labels==2)&metadata.entity_id.isin(onlycontrol).to_numpy(),'Control_multilabel':(labels==2)&metadata.entity_id.isin(controlmulti).to_numpy()};assert sum(int(m.sum()) for m in masks.values())==101
 metrics=[]
 for seed in range(3):
  raw=np.load(E7/f'CE_seed{seed}/test_raw_outputs.npy');z=np.exp(raw-raw.max(1,keepdims=True));score=z/z.sum(1,keepdims=True);pred=raw.argmax(1)
  for group,pos in masks.items():
   row={'seed':seed,'group':group,'positive_records':int(pos.sum()),'positive_entities':int(metadata.loc[pos,'entity_id'].nunique()),'control_recall':float((pred[pos]==2).mean()),'predicted_counts':dict(zip(e4.LABELS,[int((pred[pos]==k).sum()) for k in range(4)])),'predicted_fraction':dict(zip(e4.LABELS,[float((pred[pos]==k).mean()) for k in range(4)]))}
   for pool,neg in [('all_nonControl',labels!=2),('patient_nonControl',(labels<3)&(labels!=2))]:
    sel=pos|neg;truth=pos[sel];pp=pred[sel]==2;row[pool]={'negative_records':int(neg.sum()),'positive_prevalence':float(truth.mean()),'control_ovr_auc':float(roc_auc_score(truth,score[sel,2])),'control_f1':float(f1_score(truth,pp,zero_division=0)),'control_precision':float(precision_score(truth,pp,zero_division=0)),'negative_predicted_Control':int(((pred==2)&neg).sum()),'definition':'Group positive Control plus identical nonControl negative pool; other Control group excluded, not relabeled negative. F1/precision depend on group positive count.'}
   metrics.append(row)
 save('Control_group_E7_metrics.json',metrics)
 with (E12/'geometry_transform.pkl').open('rb') as f:transform=pickle.load(f)
 scaler=transform['scaler'];ytr=np.load(e4.FEATURES/'train_labels.npy').argmax(1);xt=np.load(e4.FEATURES/'train_heartlang_mean_max.npy')[ytr<3];yt=ytr[ytr<3];rng=np.random.default_rng(0);ref=np.concatenate([rng.choice(np.flatnonzero(yt==k),512,replace=False) for k in range(3)]);nn=NearestNeighbors(n_neighbors=5,metric='cosine').fit(normalize(scaler.transform(xt)[ref]));xe=np.load(e4.FEATURES/'test_heartlang_mean_max.npy');_,idx=nn.kneighbors(normalize(scaler.transform(xe[labels==2])));neighbor=yt[ref][idx];controlrows=np.flatnonzero(labels==2);descriptions={};case=[]
 morph=np.load(E12/'test_morphology.npy');records=np.load(E12/'test_record_statistics.npy');tokens=np.load(E12/'test_heartlang_token_statistics.npy');md=json.loads((E12/'morphology_rule_config.json').read_text());mnames=[x+'_mean' for x in md['continuous_features']]+[x+'_std' for x in md['continuous_features']]+[k+'_'+v+'_fraction' for k,vals in md['discrete_features'].items() for v in vals];rnames=['valid_token_count','distinct_lead_count','distinct_time_id_count','time_id_span','sid_active_per_record','sid_entropy','mean_ze_zq_squared_distance'];tnames=['token_norm_mean','token_norm_std','relative_token_dispersion','mean_pairwise_cosine','max_distance_from_mean']
 for group,mask in masks.items():
  select=mask[controlrows];descriptions[group]={'records':int(mask.sum()),'entities':int(metadata.loc[mask,'entity_id'].nunique()),'neighbor_composition':dict(zip(e4.LABELS[:3],[(neighbor[select]==k).mean().item() for k in range(3)])),'morphology':dict(zip(mnames,[describe(morph[mask,j]) for j in range(len(mnames))])),'record_statistics':dict(zip(rnames,[describe(records[mask,j]) for j in range(len(rnames))])),'token_statistics':dict(zip(tnames,[describe(tokens[mask,j]) for j in range(len(tnames))]))}
  for i,row in enumerate(controlrows):
   if mask[row]:case.append({'parent_test_row':int(row),'group':group,'entity_id':str(metadata.iloc[row].entity_id),'original_row_index':int(metadata.iloc[row].original_row_index),'neighbor_Control_fraction':float((neighbor[i]==2).mean()),**{name:float(records[row,j]) for j,name in enumerate(rnames)},**{name:float(morph[row,j]) for j,name in enumerate(mnames)}})
 pd.DataFrame(case).to_csv(RUN/'Control测试组逐例诊断.csv',index=False);save('Control_group_features.json',descriptions)
 protocol={'group_definition':'All current train/val/test patient records determine entity label sets; A only2, B contains2 and0/1. No regrouping of dataset/split.','E7_outputs':'saved original CE logits seed0/1/2, original softmax4 Control OVR; no new inference/model selection','metric_caveat':'True-Control-only subgroups have no negatives, so standalone AUC undefined. Report common all-nonControl1265 and patient-nonControl582 pools separately; F1/precision denominator/prevalence differs between groups, shared FP pool not causal assignment to entity group. Recall/final label fractions computed true group only.','neighbor':'Reuse E12 train-fitted scaler, balanced diagnostic reference512 per patient class, cosineK5; no Healthy reference; only query changed','morphology':'Reuse E12 numeric proxies and validity-aware summaries; no new clinical label','no_model_training_or_data_mutation':True};save('protocol.json',protocol)

def main():
 RUN.mkdir(parents=True,exist_ok=False);files=[RAW,ZIP,LABELDIR/'expanded_label3_data_0521.csv',LABELDIR/'expanded_label3_data_0521_dedup.csv',LABELDIR/'conflicting_csv_labels_data_0521.csv',LABELDIR/'label_mapping_data_0521_label3.json',Path(__file__),ROOT/'scripts/build_ecg_dsa_label4_patient_healthy_curriculum_split.py',ROOT/'scripts/data_prep/ECG_DSA/make_ecg_dsa_dataset.py',e4.TOKENIZER,e4.MLM]+[e4.DATA/f'{s}_metadata.csv' for s in ('train','val','test')]+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ('labels','heartlang_mean_max')]+[E7/f'CE_seed{k}/test_raw_outputs.npy' for k in range(3)];before={str(p):e4.sha(p) for p in files};save('source_fingerprints.json',before)
 cur,only,multi=source_audit();subgroup_diagnostic(cur,only,multi);assert all(e4.sha(p)==before[str(p)] for p in files);save('validation.json',{'original_expansion_and_historical_dedup_reproduced':True,'all_current_parent_array_rows_checked':True,'originals_metadata_model_outputs_hashes_unchanged':True,'no_model_training':True,'no_label_or_record_or_split_mutation':True,'group_positive_test_records_cover_101Control':True});status('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
