"""Full selected Train patient CSV replay against existing100Hz cache; no label-driven method choice."""
import json,zipfile,io,math,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import resample_poly
import run_e29_s as s
OUT=s.e.ROOT/'results/e35_data_and_error_audit'
def main():
 refs=json.load(open(OUT/'private/raw_matching.json'));cache=np.load(s.e.ROOT/'results/pre_e34_evaluation/04_e33c_raw_information/train_100hz.npy',mmap_mode='r');rows=[]
 with zipfile.ZipFile(s.e.ROOT/'datasets/患者/data_0521.zip') as z:
  lookup={Path(n).name:n for n in z.namelist() if n.endswith('.csv') and '__MACOSX/' not in n}
  for r in refs:
   if r['label']==3:continue
   content=z.read(lookup[r['csv_name']]);f=pd.read_csv(io.BytesIO(content));x=f[['I','II','III','V1','V2','V3','V4','V5','V6']].to_numpy(np.float32);g=math.gcd(len(x),1000)
   down=resample_poly(x,1000//g,len(x)//g,axis=0,padtype='line').astype(np.float32).T;expected=(down-r['global_mean'])/r['global_std'];actual=cache[r['row']];diff=expected.astype(float)-actual
   for j,lead in enumerate(['I','II','III','V1','V2','V3','V4','V5','V6']):
    a=expected[j].astype(float);b=actual[j].astype(float);cor=np.corrcoef(a,b)[0,1] if np.std(a)>0 and np.std(b)>0 else np.nan
    cs=[]
    for lag in range(-2,3):
     aa=a[20+lag:980+lag];bb=b[20:980];cs.append(float(np.corrcoef(aa,bb)[0,1]))
    rows.append(dict(row=r['row'],lead=lead,cache_rmse=float(np.sqrt(np.mean(diff[j]**2))),cache_mae=float(np.mean(abs(diff[j]))),cache_max_abs=float(np.max(abs(diff[j]))),cache_correlation=float(cor),cache_allclose=bool(np.allclose(a,b,atol=1e-6,rtol=1e-5)),best_integer_lag=range(-2,3)[int(np.nanargmax(cs))] if np.isfinite(cs).any() else None,scale_ratio=float(np.std(a)/max(np.std(b),1e-20)),source_file_sha256=hashlib.sha256(content).hexdigest()))
   if r['row']%1000<2:print('compare',r['row'],flush=True)
 pd.DataFrame(rows).to_csv(OUT/'02_500hz_100hz_fidelity/patient_cache_replay.csv',index=False)
 print('patient comparison complete',len(rows),flush=True)
if __name__=='__main__':main()
