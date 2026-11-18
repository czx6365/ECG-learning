import sys,json,pathlib,collections
import numpy as np,pandas as pd
ROOT=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'heartlang').exists());sys.path[:0]=[str(ROOT/'scripts/curriculum_v1'),str(ROOT/'scripts/experiments'),str(ROOT)]
import run_e26 as x
r=ROOT/'results/e26_information_retention';rows=json.loads((r/'raw_audit/record_replay.json').read_text());beats=[b for row in rows if 'coverage'in row for b in row['coverage']];positions=[max(48,int(b['pre_ms']/10)) for b in beats];counts=collections.Counter(positions)
tok=x.QRSTokenizerLite(fs=100,max_len=256,token_len=96)
# Explicit algorithmic boundary regression, using known impulse locations.
synth=np.zeros((9,400),np.float32);synth[:,242]=1;seg=tok.extract_qrs_segments(synth,np.array([100,242,280]));assert np.argmax(seg[0][0])==71
normal=np.zeros((9,400),np.float32);normal[:,180]=1;seg=tok.extract_qrs_segments(normal,np.array([100,180,260]));assert np.argmax(seg[0][0])==48
out={'sampled_ecgs':sum('coverage'in row for row in rows),'beats':len(positions),'r_index_not48':sum(p!=48 for p in positions),'r_index_not48_fraction':float(np.mean(np.array(positions)!=48)),'r_index_max':max(positions),'max_offset_ms':(max(positions)-48)*10,'r_index_distribution':dict(sorted(counts.items())),'synthetic_asymmetric_rr_impulse_index':71,'synthetic_symmetric_rr_impulse_index':48,'reason':'If midpoint interval <=96 but left RR half exceeds48, segment_start clamps at0 rather than shifting/cropping left boundary. Actual detected R then lies at center-start >48.','proxy_caveat':'All fixed-window probes are relative to nominal token index48, not per-token recovered R; their numeric results are not actual R-aligned ST/T measurements. Raw coverage proportions use true detected R and remain valid.','historical_data_changed':False,'test_loaded':False};x.dump(r/'verification/r_alignment_audit.json',out)
# Draw one normal and one most-asymmetric example per class, with true detected-R alignment.
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axs=plt.subplots(4,2,figsize=(13,11));frames={s:pd.read_csv(x.e.DOWNSTREAM/f'{s}_metadata.csv') for s in ('train','val')}
for label in range(4):
 rr=[row for row in rows if 'coverage'in row and row['label']==label];normal=next((row for row in rr if all(b['pre_ms']<=480 for b in row['coverage'])),rr[0]);asym=max(rr,key=lambda row:max(b['pre_ms'] for b in row['coverage']))
 for col,row in enumerate((normal,asym)):
  meta=frames[row['split']].iloc[row['row']];src=pathlib.Path(str(meta.source_dataset));src=src if src.is_absolute() else ROOT/src;rawdir=pathlib.Path(str(src).replace('/pretrain_qrs/','/pretrain_waveforms/').replace('/qrs/','/waveforms/'));ss=str(meta.source_split);arr=np.load(rawdir/f'{ss}_data.npy',mmap_mode='r');fm=pd.read_csv(rawdir/f'{ss}_metadata.csv');si=int(meta.source_index)
  if str(fm.iloc[si].csv_name)!=str(meta.csv_name):si=int(np.flatnonzero(fm.csv_name.astype(str).values==str(meta.csv_name))[0])
  signal=np.asarray(arr[si]);p=tok.qrs_detection(signal);j=1+int(np.argmax([b['pre_ms'] for b in row['coverage']])) if col else min(3,len(p)-2);center=int(p[j]);start=(int(p[j])+int(p[j-1]))//2;end=(int(p[j])+int(p[j+1]))//2
  if end-start>96:start=max(center-48,0);end=min(start+96,1000)
  shift=max(48-(center-start),0);actual_r=center-start+shift;segments=tok.extract_qrs_segments(signal,p);token=np.asarray(segments[0][j-1]);ax=axs[label,col];left=max(0,center-65);right=min(1000,center+75);ax.plot((np.arange(left,right)-center)*10,signal[0,left:right],label='stored 100 Hz waveform',lw=1.1);ax.plot((np.arange(96)-actual_r)*10,token,label='96-point token, true R aligned',lw=1.1);ax.axvline(0,color='gray',lw=.6);ax.set_title(f'class {label}, {row["split"]} row {row["row"]}; detected R at token {actual_r}');ax.set_xlabel('ms relative to detected R');ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(r/'raw_audit/window_replay_true_r.png',dpi=140);plt.close(fig)
from PIL import Image
im=Image.open(r/'raw_audit/window_replay_true_r.png').convert('RGB');im.thumbnail((1050,1000));im.save(r/'raw_audit/window_replay_true_r_preview.jpg',quality=65)
print(json.dumps(out),flush=True)
