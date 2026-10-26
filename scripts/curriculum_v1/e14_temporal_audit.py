import json
from collections import Counter
import pandas as pd
import run_e14_association_audit as e
raw=pd.read_excel(e.RAW);cur=pd.read_csv(e.RUN/'当前记录完整来源追溯.csv');edges=pd.read_csv(e.RUN/'原表到ECG关联明细.csv');out={};duplicate=[]
for key,d in raw.groupby(['patient_sn','冠脉造影时间'],dropna=False):
 if len(d)>1:duplicate.append({'patient_sn':str(key[0]),'clinical_time':str(key[1]),'rows':list(map(int,d.index)),'row_count':len(d),'existing_labels':sorted(d['标签'].astype(str).unique()),'file_lists_same':d.file_name.nunique()==1})
out['raw_same_patient_clinical_time_multiple_rows']=duplicate;out['raw_serial_unique']=int(raw['序号'].nunique());out['raw_exact_full_row_duplicate_excess']=int(raw.duplicated().sum());out['raw_patient_missing']=int(raw.patient_sn.isna().sum());out['raw_clinical_time_missing']=int(raw['冠脉造影时间'].isna().sum());out['same_patient_clinical_time_groups']=len(duplicate);out['same_patient_clinical_time_multiple_labels']=sum(len(x['existing_labels'])>1 for x in duplicate)
cur['dt']=pd.to_datetime(cur.ECG_date_positionally_aligned+' '+cur.ECG_time_positionally_aligned,format='mixed',errors='coerce');cur['clin']=pd.to_datetime(cur.clinical_record_time,format='mixed',errors='coerce');out['selected_ECG_datetime_parse_failed']=int(cur.dt.isna().sum());out['selected_clinical_time_parse_failed']=int(cur.clin.isna().sum());patterns=Counter();ents=[]
for ent,d in cur.groupby('entity_id'):
 labels=set(d.label_id)
 if 2 not in labels or len(labels)<2:continue
 ordered=d.sort_values(['dt','original_row_index']);pattern='/'.join(ordered.current_label);patterns[pattern]+=1
 ents.append({'entity_id':ent,'records':len(d),'chronological_label_sequence':pattern,'distinct_file_count':int(d.csv_name_expected.nunique()),'distinct_original_rows':int(d.original_row_index.nunique()),'distinct_ECG_datetime_count':int(d.dt.nunique()),'span_days':float((d.dt.max()-d.dt.min()).total_seconds()/86400) if d.dt.notna().all() else None})
out['Control_multilabel_chronological_sequence_counts']=dict(patterns);out['Control_multilabel_entity_details']=ents;out['interpretation']='Temporal ordering and exact key repeats only. 冠脉造影时间 is an existing field, not a proven unique clinical encounter identifier. No inferred clinical trajectory or relabeling.';e.save('temporal_audit.json',out)
print({k:v for k,v in out.items() if k not in ('raw_same_patient_clinical_time_multiple_rows','Control_multilabel_entity_details')})
