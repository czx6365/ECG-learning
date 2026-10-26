import io,json,hashlib,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import run_e14_association_audit as e
from scripts.data_prep.ECG_DSA.ecg_zip_utils import load_waveform,CSV_LEADS
p=e.RUN;edges=pd.read_csv(p/'原表到ECG关联明细.csv');cur=pd.read_csv(p/'当前记录完整来源追溯.csv');inventory=pd.read_csv(p/'真实ECG文件核验.csv');groups=[];validmaps={}
for s in ('train','val','test'):
 meta=pd.read_csv(e.e4.DATA/f'{s}_metadata.csv');mask=np.load(e.e4.FEATURES/f'{s}_valid_heartbeat_mask.npy');validmaps.update({(s,int(i)):bool(mask[i].any()) for i in range(len(meta))})
with zipfile.ZipFile(e.ZIP) as z:
 for sha,d in inventory.groupby('sha256'):
  if len(d)<2:continue
  refs=edges[edges.csv_name_expected.isin(d.csv_name)];selected=cur[cur.csv_name_expected.isin(d.csv_name)];blob=z.read(d.member_name.iloc[0]);wave=pd.read_csv(io.BytesIO(blob),usecols=CSV_LEADS).to_numpy(dtype=np.float32)
  groups.append({'sha256':sha,'files':d.csv_name.tolist(),'file_count':len(d),'patient_sn_count':int(refs.patient_sn.nunique()),'original_record_count':int(refs.original_row_index.nunique()),'raw_labels':sorted(refs.source_label.unique()),'raw_rows':sorted(set(map(int,refs.original_row_index))),'contains_Control':bool((refs.label_id==2).any()),'current_records':len(selected),'current_label_ids':sorted(set(map(int,selected.label_id))),'current_splits':sorted(set(selected.split)),'current_rows':[{'split':v.split,'parent_row':int(v.parent_row),'label':int(v.label_id),'nonempty_token':validmaps[(v.split,int(v.parent_row))]} for v in selected.itertuples()],'raw_wave_all_zero':bool(np.isfinite(wave).all() and np.all(wave==0)),'raw_wave_variance':float(np.nanvar(wave))})
 e.save('raw_content_duplicate_groups.json',groups)
 summary={'groups':len(groups),'files_in_groups':sum(x['file_count'] for x in groups),'duplicate_file_excess':sum(x['file_count']-1 for x in groups),'max_group_size':max(x['file_count'] for x in groups),'groups_multiple_patient_sn':sum(x['patient_sn_count']>1 for x in groups),'groups_multiple_source_labels':sum(len(x['raw_labels'])>1 for x in groups),'groups_multiple_original_rows':sum(x['original_record_count']>1 for x in groups),'groups_containing_Control':sum(x['contains_Control'] for x in groups),'groups_all_zero_wave':sum(x['raw_wave_all_zero'] for x in groups),'groups_with_atleast_two_current_records':sum(x['current_records']>=2 for x in groups),'groups_current_cross_split':sum(len(x['current_splits'])>1 for x in groups),'groups_current_cross_label':sum(len(x['current_label_ids'])>1 for x in groups),'groups_current_two_nonempty_records':sum(sum(q['nonempty_token'] for q in x['current_rows'])>=2 for x in groups)};e.save('raw_content_duplicate_summary.json',summary);print(summary,flush=True)
 # Verify actual raw CSV → existing standardized waveform rows. No output dataset is regenerated.
 source=e.ROOT/'datasets/ecg_datasets/ECG_DSA/waveforms/data_0521_label3';config=json.loads((source/'prepare_summary.json').read_text());mean=config['global_mean'];std=config['global_std'];files=[source/'prepare_summary.json']+[source/f'{s}_{n}' for s in ('train','val','test') for n in ('metadata.csv','data.npy')];before={str(f):e.e4.sha(f) for f in files};checks=[]
 for s in ('train','val','test'):
  m=pd.read_csv(source/f'{s}_metadata.csv');arr=np.load(source/f'{s}_data.npy',mmap_mode='r');byname={str(v.csv_name):i for i,v in enumerate(m.itertuples())};selected=cur[cur.split==s];maxerr=0.;mismatch=[]
  for i,row in enumerate(selected.itertuples()):
   ix=byname[row.csv_name_expected];assert str(m.iloc[ix].member_name)==row.member_name
   waveform=load_waveform(z,row.member_name,arr.shape[2]);expected=(waveform-mean)/std;err=float(np.max(np.abs(expected-arr[ix])));maxerr=max(maxerr,err)
   if not np.allclose(expected,arr[ix],atol=1e-6,rtol=1e-6):mismatch.append({'csv_name':row.csv_name_expected,'max_abs_error':err})
   if i%500==0:print('rawwave',s,i,flush=True)
  checks.append({'split':s,'selected_rows_checked':len(selected),'max_abs_error':maxerr,'mismatch_count':len(mismatch),'mismatches':mismatch})
 e.save('raw_wave_to_saved_wave_validation.json',{'method':'Existing ecg_zip_utils.load_waveform (9 leads, resample_poly to stored1000), existing recorded normalization mean/std; compared raw file reconstruction to existing waveform parent rows; no dataset writes','global_mean':mean,'global_std':std,'checks':checks,'parent_wave_sources_hashes_unchanged':all(e.e4.sha(Path(k))==v for k,v in before.items()),'source_fingerprints':before,'source_script_sha256':e.e4.sha(Path(__file__))})
 print('wave trace complete',flush=True)
