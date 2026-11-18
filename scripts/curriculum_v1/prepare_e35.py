"""Fresh Train-only baseline, provenance, geometry, temporal metadata audit. No fitting."""
import json,hashlib,re,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import welch
from scipy.special import xlogy
import run_e29_s as s
from scripts.data_prep.ECG_DSA.tokenize_pretrain_qrs import QRSTokenizerLite
b,e=s.b,s.e;ROOT=e.ROOT;OUT=ROOT/'results/e35_data_and_error_audit'
DIRS=['00_baseline','01_raw_availability','02_500hz_100hz_fidelity','03_source_confounding','04_omi_nomi_errors','05_label_temporal_audit','06_joint_analysis','figures','code','private']
def sha(x):return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float32).tobytes()).hexdigest()
def entity(x):return hashlib.sha256(('e35-anon-v1|'+str(x)).encode()).hexdigest()[:16]
def main():
 start=time.time()
 for d in DIRS:(OUT/d).mkdir(parents=True,exist_ok=True)
 m=pd.read_csv(e.DOWNSTREAM/'train_metadata.csv');y=np.load(e.DOWNSTREAM/'train_labels.npy');seeds=np.load(s.OUT/'oof_shared.npy');q=seeds.mean(0);label=y.argmax(1)
 assert np.array_equal(label,m.label_id);assert seeds.shape==(3,6640,4) and np.isfinite(seeds).all();np.testing.assert_allclose(seeds.sum(2),1,atol=2e-6)
 folds=pd.read_csv(s.OUT/'fold_assignment.csv');outer=folds[folds.role=='outer'].set_index('row').fold
 assert len(outer)==len(m) and outer.index.is_unique
 for f in range(3):
  dev=folds[(folds.fold==f)&(folds.role!='outer')].row;held=folds[(folds.fold==f)&(folds.role=='outer')].row
  assert not set(m.iloc[dev].entity_id)&set(m.iloc[held].entity_id)
 metrics=s.metrics(y,q)
 for key,val in [('overall',.378010),('clinical3',.294061),('omi_nomi_auc',.792372)]:assert abs(metrics[key]-val)<.00000051
 b.dump(OUT/'00_baseline/metrics.json',metrics)
 np.save(OUT/'00_baseline/oof_seeds.npy',seeds);np.save(OUT/'00_baseline/labels.npy',y)
 (OUT/'00_baseline/fold_assignment.csv').write_bytes((s.OUT/'fold_assignment.csv').read_bytes())
 pred=q.argmax(1);pcond=q[:,0]/q[:,:2].sum(1);seedcond=seeds[:,:,0]/seeds[:,:,:2].sum(2)
 table=pd.DataFrame(dict(row=np.arange(len(m)),anonymous_id=[f'T{i:05d}' for i in range(len(m))],entity=m.entity_id.map(entity),label=label,fold=np.arange(len(m))))
 table['fold']=table.row.map(outer);table['patient']=m.patient_sn.map(lambda x:entity(x) if pd.notna(x) else '')
 for j,n in enumerate(['OMI','NOMI','Control','Healthy']):table['p_'+n]=q[:,j]
 table['prediction']=pred;table['p_cond_omi']=pcond;table['conditional_error']=((pcond>=.5)!=(label==0)).astype(int);table['four_error']=(pred!=label).astype(int)
 table['confidence']=q.max(1);sort=np.sort(q,axis=1);table['top_margin']=sort[:,-1]-sort[:,-2]
 table['true_margin']=q[np.arange(len(m)),label]-np.max(np.where(y==1,-np.inf,q),axis=1)
 table['four_logloss']=-np.log(q[np.arange(len(m)),label].clip(1e-8));table['four_brier']=np.square(q-y).sum(1)
 table['conditional_logloss']=np.where(label==0,-np.log(pcond.clip(1e-8,1-1e-8)),-np.log((1-pcond).clip(1e-8,1-1e-8)))
 table['conditional_brier']=np.square(pcond-(label==0));table['seed_cond_sd']=seedcond.std(0,ddof=1)
 table['seed_unanimous_wrong']=np.all(seeds.argmax(2)==pred[None],0)&(pred!=label)
 table['high_conf_error']=(pred!=label)&(q.max(1)>=.9);table['low_margin']=table.top_margin<.1
 table['source_archive']=m.source_zip.astype(str);table['source_pipeline']=m.source_dataset.astype(str);table['multi_ecg']=m.entity_id.map(m.entity_id.value_counts()).to_numpy()>1
 table['label_conflict']=m.entity_id.map(m.groupby('entity_id').label_id.nunique()).to_numpy()>1
 # Exact token support replay from currently available source cache, not old summary.
 wave=np.load(ROOT/'results/pre_e34_evaluation/04_e33c_raw_information/train_100hz.npy',mmap_mode='r')
 stored=np.load(e.DOWNSTREAM/'train_data.npy',mmap_mode='r');chans=np.load(e.DOWNSTREAM/'train_data_in_chans.npy',mmap_mode='r')
 tim=np.load(e.DOWNSTREAM/'train_data_in_times.npy',mmap_mode='r');tok=QRSTokenizerLite(fs=100);sources={};private=[];qualities=[];temporal=[]
 for i,row in enumerate(m.itertuples()):
  src=Path(str(row.source_dataset).replace('/pretrain_qrs/','/pretrain_waveforms/').replace('/qrs/','/waveforms/'));src=src if src.is_absolute() else ROOT/src;key=(str(src),row.source_split)
  if key not in sources:sources[key]=(np.load(src/f'{row.source_split}_data.npy',mmap_mode='r'),pd.read_csv(src/f'{row.source_split}_metadata.csv'),json.load(open(src/'prepare_summary.json')))
  arr,mm,params=sources[key];idx=int(row.source_index)
  if str(mm.iloc[idx].csv_name)!=str(row.csv_name):
   hits=np.flatnonzero(mm.csv_name.astype(str).to_numpy()==str(row.csv_name));assert len(hits)==1;idx=int(hits[0])
  assert np.array_equal(wave[i],arr[idx]);w=np.asarray(wave[i]);peaks=tok.qrs_detection(w);n=max(0,len(peaks)-2)
  intervals=[];exact=True
  for lead in range(9):
   for k in range(n):
    slot=lead*n+k
    if slot>=256:continue
    j=k+1;r=int(peaks[j]);lo=(r+int(peaks[j-1]))//2;hi=(r+int(peaks[j+1]))//2
    if hi-lo>96:lo=max(r-48,0);hi=min(lo+96,1000)
    pad=max(48-(r-lo),0);end=pad+hi-lo
    if end>96:hi-=end-96;end=96
    z=np.zeros(96,np.float32);z[pad:end]=w[lead,lo:hi]
    exact&=np.array_equal(z,stored[i,slot]);intervals.append((lo,hi,r,(hi-lo)/96))
  def coverage(a,b):return np.mean([max(0,min(hi,r+b)-max(lo,r+a))/(b-a) for lo,hi,r,_ in intervals]) if intervals else np.nan
  f,ps=welch(w,fs=100,nperseg=500,axis=-1);power=ps.sum(1);hf=ps[:,(f>=20)&(f<=45)].sum(1)/power.clip(1e-20);wander=ps[:,f<.5].sum(1)/power.clip(1e-20)
  valid=(chans[i]>0)&(tim[i]>0)&np.any(stored[i]!=0,axis=1)
  qualities.append(dict(row=i,hr=60/(np.median(np.diff(peaks))/100) if len(peaks)>1 else np.nan,n_peaks=len(peaks),n_tokens=int(valid.sum()),padding=1-valid.mean(),token_valid_fraction=np.mean([x[3] for x in intervals]) if intervals else np.nan,qrs_coverage=coverage(-8,12),post_coverage=coverage(8,40),token_replay_exact=bool(exact),hf_noise_proxy=float(np.median(hf)),baseline_wander_proxy=float(np.median(wander)),flat_fraction=float(np.mean(np.diff(w,axis=1)==0)),source_std=params['global_std'],source_mean=params['global_mean'],unscaled_rms=float(np.sqrt(np.mean((w*params['global_std']+params['global_mean'])**2)))))
  private.append(dict(row=i,csv_name=row.csv_name,member_name=row.member_name,source_zip=row.source_zip,source_index=idx,source_split=row.source_split,label=int(label[i]),fold=int(outer[i]),entity=entity(row.entity_id),global_mean=params['global_mean'],global_std=params['global_std'],cache_sha=sha(w),cache_lead_sha=[sha(z) for z in w],r_peaks=peaks.tolist()))
  # The clinical sheet has lists; resolve ECG-specific date/time by its matched filename position.
  def parts(x):return [z.strip() for z in str(x).split(';')] if pd.notna(x) else []
  dates=parts(getattr(row,'_0',None)) if False else parts(m.iloc[i].get('心电图时间（日期）'))
  times=parts(m.iloc[i].get('心电图时间（时刻）'));files=parts(m.iloc[i].get('file_name'))
  hits=[k for k,z in enumerate(files) if Path(z).stem==Path(str(row.csv_name)).stem]
  dt=pd.NaT;method='missing';position=None
  if label[i]<3:
   if len(hits)==1 and len(files)==len(dates)==len(times):position=hits[0];method='filename_position'
   elif len(dates)==len(times)==1 and len(files)<=1:position=0;method='single_structured'
   if position is not None:dt=pd.to_datetime(dates[position]+' '+times[position],dayfirst=True,errors='coerce')
  ang=m.iloc[i].get('冠脉造影时间');at=pd.to_datetime(ang,errors='coerce');has_clock=bool(re.search(r'\d{1,2}:\d{2}',str(ang)))
  # Midnight event values may be administrative date-only despite a timestamp format.
  informative_clock=bool(pd.notna(at) and (at.hour!=0 or at.minute!=0 or at.second!=0))
  diff=(dt-at).total_seconds()/86400 if pd.notna(dt) and pd.notna(at) else np.nan
  relation='unknown'
  if np.isfinite(diff):
   if informative_clock:relation='before_angio' if diff<0 else 'after_angio'
   elif dt.normalize()<at.normalize():relation='before_angio_day'
   elif dt.normalize()>at.normalize():relation='after_angio_day'
   else:relation='same_day_time_unknown'
  temporal.append(dict(row=i,ecg_time_resolved=bool(pd.notna(dt)),ecg_time_method=method,ecg_list_length=len(dates),filename_position=position,angio_available=bool(pd.notna(at)),angio_clock_informative=informative_clock,ecg_minus_angio_days=diff,temporal_relation=relation,pci_status_present=pd.notna(m.iloc[i].get('是否接受了PCI治疗')),pci_time_available=False,patient_label_count=int(m[m.entity_id==row.entity_id].label_id.nunique())))
  if i%1000==0:print('prepared',i,flush=True)
 quality=pd.DataFrame(qualities);temporal=pd.DataFrame(temporal);table=table.merge(quality,on='row').merge(temporal,on='row')
 table.to_csv(OUT/'04_omi_nomi_errors/cases.csv',index=False)
 temporal.to_csv(OUT/'05_label_temporal_audit/temporal_cases.csv',index=False)
 (OUT/'private/raw_matching.json').write_text(json.dumps(private,ensure_ascii=False))
 b.dump(OUT/'05_label_temporal_audit/metadata_audit.json',dict(available_columns=list(m.columns),device_fields=[c for c in m if re.search('device|设备|厂商|machine',c,re.I)],clinical_labels='copied original label_id; operational OMI/NOMI adjudication definition not established by this script',multi_ecg_entities=int((m.groupby('entity_id').size()>1).sum()),conflicting_entities=int((m.groupby('entity_id').label_id.nunique()>1).sum()),time_resolved=temporal.groupby(m.label_id).ecg_time_resolved.agg(['count','sum']).to_dict(),temporal_counts=temporal.temporal_relation.value_counts().to_dict(),pci_time_present=False,episode='same patient or angio date is only an episode proxy, not verified episode identity'))
 b.dump(OUT/'00_baseline/audit.json',dict(records=len(m),metadata_sha256=e.sha256(e.DOWNSTREAM/'train_metadata.csv'),prob_sha256=e.sha256(s.OUT/'oof_shared.npy'),fold_sha256=e.sha256(s.OUT/'fold_assignment.csv'),token_exact_replay=int(table.token_replay_exact.sum()),source_alignment_exact=True,test_read=False,seconds=time.time()-start))
 (OUT/'status.txt').write_text('prepared');print('prepared complete',flush=True)
if __name__=='__main__':main()
