"""Resolve structured ECG lists using source_file_name_list and audited MM-DD-YYYY format."""
import json,re
from pathlib import Path
import numpy as np,pandas as pd
import run_e29_s as s
ROOT=s.e.ROOT;OUT=ROOT/'results/e35_data_and_error_audit'
def main():
 m=pd.read_csv(s.e.DOWNSTREAM/'train_metadata.csv');rows=[];anchors=[]
 split=lambda x:[z.strip() for z in str(x).split(';')] if pd.notna(x) else []
 all_dates=m.loc[m.label_id<3,'心电图时间（日期）'].dropna().str.split(';').explode().str.split('-',expand=True)
 assert (pd.to_numeric(all_dates[0],errors='coerce')>12).sum()==0
 second_gt12=int((pd.to_numeric(all_dates[1],errors='coerce')>12).sum());assert second_gt12>0
 for i,r in m.iterrows():
  dates=split(r.get('心电图时间（日期）'));times=split(r.get('心电图时间（时刻）'));files=split(r.get('source_file_name_list')) or split(r.get('file_name'))
  hits=[k for k,z in enumerate(files) if Path(z).stem==Path(str(r.csv_name)).stem]
  method='unresolved';position=None;dt=pd.NaT
  if r.label_id<3:
   if len(hits)==1 and len(files)==len(dates)==len(times):position=hits[0];method='source_filename_list_position'
   elif len(files)<=1 and len(dates)==len(times)==1:position=0;method='single_structured'
   if position is not None:dt=pd.to_datetime(dates[position]+' '+times[position],format='%m-%d-%Y %H:%M:%S',errors='coerce')
  at=pd.to_datetime(r.get('冠脉造影时间'),errors='coerce');clock=bool(pd.notna(at) and (at.hour or at.minute or at.second));diff=(dt-at).total_seconds()/86400 if pd.notna(dt) and pd.notna(at) else np.nan
  relation='unknown'
  if np.isfinite(diff):
   if clock:relation='before_angio' if diff<0 else 'after_angio'
   elif dt.normalize()<at.normalize():relation='before_angio_day'
   elif dt.normalize()>at.normalize():relation='after_angio_day'
   else:relation='same_day_time_unknown'
  match=re.match(r'^(\d{17})_',str(r.csv_name));fn=pd.to_datetime(match[1],format='%Y%m%d%H%M%S%f',errors='coerce') if match else pd.NaT
  fname_diff=(dt-fn).total_seconds() if pd.notna(dt) and pd.notna(fn) else np.nan
  rows.append(dict(row=i,ecg_time_resolved=bool(pd.notna(dt)),ecg_time_method=method,ecg_list_length=len(dates),filename_position=position,angio_available=bool(pd.notna(at)),angio_clock_informative=clock,ecg_minus_angio_days=diff,temporal_relation=relation,pci_status_present=pd.notna(r.get('是否接受了PCI治疗')),pci_time_available=False,patient_label_count=int(m[m.entity_id==r.entity_id].label_id.nunique()),ecg_filename_time_difference_seconds=fname_diff))
  anchors.append(dict(row=i,patient=r.entity_id,ecg=dt,angio=at,label=r.label_id))
 t=pd.DataFrame(rows);t.to_csv(OUT/'05_label_temporal_audit/temporal_cases.csv',index=False)
 old=pd.read_csv(OUT/'04_omi_nomi_errors/cases.csv');old=old.drop(columns=[c for c in t.columns if c!='row' and c in old]);old=old.merge(t,on='row');old.to_csv(OUT/'04_omi_nomi_errors/cases.csv',index=False)
 a=pd.DataFrame(anchors);patient=[]
 for pid,g in a[a.label<3].groupby('patient'):
  ecg=g.ecg.dropna();ag=g.angio.dropna();patient.append(dict(patient=s.e.sha256 if False else old.loc[g.row.iloc[0],'entity'],ecgs=len(g),labels=g.label.nunique(),ecg_span_days=(ecg.max()-ecg.min()).total_seconds()/86400 if len(ecg) else None,angio_dates=ag.nunique(),different_label_and_different_angio=g.label.nunique()>1 and ag.nunique()>1))
 pd.DataFrame(patient).to_csv(OUT/'05_label_temporal_audit/patient_episode_proxies.csv',index=False)
 stats=dict(date_format='MM-DD-YYYY, explicitly parsed; first field >12 count=0; second field >12 count='+str(second_gt12),resolved_by_class=t.groupby(m.label_id).ecg_time_resolved.agg(['count','sum']).to_dict(),relations=t.groupby(m.label_id).temporal_relation.value_counts().to_dict(),clock_informative=int(t.angio_clock_informative.sum()),filename_time_validation=dict(n=int(t.ecg_filename_time_difference_seconds.notna().sum()),within1s=int((t.ecg_filename_time_difference_seconds.abs()<=1).sum()),median_abs_seconds=float(t.ecg_filename_time_difference_seconds.abs().median())),pci_time_available=False,final_diagnosis_timestamp_available=False,clinical_label_definition='source labels copied and remapped; no verified operational OMI/NOMI adjudication rule found in current builder; TIMI alone not assumed definition',caveat='Angiography midnight is treated as date-only. Relative to angiography is NOT pre/post PCI. Same patient/angio date is only episode proxy.')
 # tuple dictionary keys must be encoded explicitly
 stats['relations']={str(k):v for k,v in stats['relations'].items()}
 (OUT/'05_label_temporal_audit/temporal_summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False))
 print('temporal audit fixed',t.ecg_time_resolved.sum(),flush=True)
if __name__=='__main__':main()
