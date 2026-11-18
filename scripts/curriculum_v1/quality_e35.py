"""All6640x9 current-cache quality metrics, including records without native files."""
import json
import numpy as np,pandas as pd
from scipy.signal import welch
import run_e29_s as s
OUT=s.e.ROOT/'results/e35_data_and_error_audit'
def main():
 refs=json.load(open(OUT/'private/raw_matching.json'));w=np.load(s.e.ROOT/'results/pre_e34_evaluation/04_e33c_raw_information/train_100hz.npy',mmap_mode='r');rows=[]
 for r in refs:
  x=np.asarray(w[r['row']]);f,ps=welch(x,fs=100,nperseg=500,axis=-1);total=ps.sum(1).clip(1e-20)
  for j,lead in enumerate(['I','II','III','V1','V2','V3','V4','V5','V6']):
   rows.append(dict(row=r['row'],lead=lead,cache100_mean=float(x[j].mean()),cache100_std=float(x[j].std()),cache100_min=float(x[j].min()),cache100_max=float(x[j].max()),cache100_hf20to45_proxy=float(ps[j,(f>=20)&(f<=45)].sum()/total[j]),cache100_wander_below0_5_proxy=float(ps[j,f<.5].sum()/total[j]),cache100_flat_fraction=float(np.mean(np.diff(x[j])==0))))
 pd.DataFrame(rows).to_csv(OUT/'02_500hz_100hz_fidelity/all_cache_per_lead_quality.csv.gz',index=False,compression='gzip')
if __name__=='__main__':main()
