"""E33 geometric token support audit; no clinical delineation claims."""
from pathlib import Path
import sys
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/home/ubuntu/HeartLang-main')
from scripts.data_prep.ECG_DSA.tokenize_pretrain_qrs import QRSTokenizerLite

ROOT=Path('/home/ubuntu/HeartLang-main')
DATA=ROOT/'datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1'
OUT=ROOT/'results/pre_e34_evaluation'
for name in ('00_baseline_audit','01_e32b_residual','02_e33a_tokenizer_coverage','03_e33b_error_association','04_e33c_raw_information','05_integrated_decision'):(OUT/name).mkdir(parents=True,exist_ok=True)
meta=pd.read_csv(DATA/'train_metadata.csv');labels=np.load(DATA/'train_labels.npy').argmax(1)
stored=np.load(DATA/'train_data.npy',mmap_mode='r');chans=np.load(DATA/'train_data_in_chans.npy',mmap_mode='r')
source={};tok=QRSTokenizerLite(fs=100)
rows=[];bad=[];maps=[]
for j,row in enumerate(meta.itertuples()):
    src=Path(str(row.source_dataset));src=src if src.is_absolute() else ROOT/src
    raw=Path(str(src).replace('/pretrain_qrs/','/pretrain_waveforms/').replace('/qrs/','/waveforms/'))
    split=str(row.source_split);key=(str(raw),split)
    if key not in source:
        source[key]=(np.load(raw/f'{split}_data.npy',mmap_mode='r'),pd.read_csv(raw/f'{split}_metadata.csv'))
    waves,frame=source[key];idx=int(row.source_index)
    if str(frame.iloc[idx].csv_name)!=str(row.csv_name):
        found=np.flatnonzero(frame.csv_name.astype(str).values==str(row.csv_name))
        if len(found)!=1:bad.append({'row':j,'reason':'source mapping'});continue
        idx=int(found[0])
    wave=np.asarray(waves[idx]);peaks=tok.qrs_detection(wave);n=max(len(peaks)-2,0)
    if n==0:bad.append({'row':j,'reason':'no internal peak'})
    intervals=[];post=[];valid=[];match=True
    for lead in range(9):
        for k in range(n):
            slot=lead*n+k
            if slot>=256:continue
            i=k+1;r=int(peaks[i]);lo=(r+int(peaks[i-1]))//2;hi=(r+int(peaks[i+1]))//2
            if hi-lo>96:lo=max(r-48,0);hi=min(lo+96,1000)
            pad=max(48-(r-lo),0);end=pad+hi-lo
            if end>96:hi-=end-96;end=96
            segment=np.zeros(96,np.float32);segment[pad:end]=wave[lead,lo:hi]
            if not np.array_equal(segment,stored[j,slot]):match=False
            intervals.append((lead,lo,hi,r,k,slot,pad,end));post.append((hi-r)/100);valid.append((hi-lo)/96)
            if j<192:maps.append({'row':j,'lead':lead,'beat':k,'token_slot':slot,'raw_start':lo,'raw_end_exclusive':hi,'r_sample':r,'token_valid_start':pad,'token_valid_end_exclusive':end})
    if not match:bad.append({'row':j,'reason':'token replay mismatch'})
    # R 后窗口只是时间几何代理，不代表 ST/T 生理边界。
    def coverage(a,b,sequence=False):
        if not intervals:return np.nan
        ok=[]
        for lead,lo,hi,r,k,slot,_,_ in intervals:
            if sequence:
                support=[(x,y) for l,x,y,_,_,_,_,_ in intervals if l==lead]
                mask=np.zeros(b-a,bool)
                for x,y in support:
                    left=max(x-r-a,0);right=max(0,min(y-r-a,b-a))
                    if right>left:mask[left:right]=True
                ok.append(mask.mean())
            else:ok.append(max(0,min(hi,r+b)-max(lo,r+a))/(b-a))
        return float(np.mean(ok))
    rows.append({'row':j,'ecg_id':f'T{j:05d}','entity_id':str(row.entity_id),'label':int(labels[j]),'source':str(raw),'source_split':split,
      'n_peaks':int(len(peaks)),'n_tokens':len(intervals),'valid_fraction':float(np.mean(valid)) if valid else np.nan,
      'post_ms_median':float(np.median(post)*1000) if post else np.nan,
      'r400_single':coverage(0,40),'r400_sequence':coverage(0,40,True),'qrs_proxy':coverage(-5,7),
      'early_post_proxy':coverage(8,16),'late_post_proxy':coverage(20,40),'rr_ms':float(np.median(np.diff(peaks))*10) if len(peaks)>1 else np.nan,
      'replay_exact':match})
    if j%1000==0:print('geometry',j,len(meta),flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'02_e33a_tokenizer_coverage/geometry_train.csv',index=False)
pd.DataFrame(maps).to_csv(OUT/'02_e33a_tokenizer_coverage/sample_point_mapping.csv',index=False)
pd.DataFrame(bad).to_csv(OUT/'02_e33a_tokenizer_coverage/failures.csv',index=False)
summary={'n':len(df),'replay_exact':int(df.replay_exact.sum()),'failed':len(bad),'single_r400_mean':float(df.r400_single.mean()),
 'sequence_r400_mean':float(df.r400_sequence.mean()),'valid_fraction':float(df.valid_fraction.mean()),
 'source_counts':{f'{a}|{b}':int(v) for (a,b),v in df.groupby(['source','label']).size().items()}}
(OUT/'02_e33a_tokenizer_coverage/summary.json').write_text(json.dumps(summary,indent=2,default=str))
prob=np.load(ROOT/'results/e29_shared_generalization/oof_shared.npy').mean(0)
fold=pd.read_csv(ROOT/'results/e29_shared_generalization/fold_assignment.csv')
fmap=fold[fold.role=='outer'].set_index('row').fold
df['fold']=df.row.map(fmap);df['q']=prob[df.row,0]/prob[df.row,:2].sum(1)
df['error']=((df.q>=.5)!=(df.label==0)).astype(int)
df['binary_logloss']=np.where(df.label==0,-np.log(df.q.clip(1e-7,1-1e-7)),-np.log((1-df.q).clip(1e-7,1-1e-7)))
pair=df[df.label<2].copy();pair.to_csv(OUT/'03_e33b_error_association/analysis_table.csv',index=False)
covars=['r400_single','r400_sequence','valid_fraction','qrs_proxy','rr_ms','n_tokens']
assoc={}
for col in covars:
    valid=pair[[col,'binary_logloss','error']].dropna()
    assoc[col]={'n':len(valid),'spearman_logloss':float(spearmanr(valid[col],valid.binary_logloss).statistic),
                'spearman_error':float(spearmanr(valid[col],valid.error).statistic)}
(OUT/'03_e33b_error_association/associations.json').write_text(json.dumps(assoc,indent=2))
print('geometry and association complete',summary,flush=True)
