#!/usr/bin/env python3
"""E9 inverse-sqrt train-frequency CE; only objective weights change."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
import run_e4_aggregation as e4
import run_e7_objective as e7
from run_e7_objective import evaluate
ROOT=e4.ROOT;RUN=ROOT/'results/curriculum_v1/e9_class_balanced_ce_20261011'
SOURCE=ROOT/'results/curriculum_v1/e7_objective_20261011'
def train(objective,seed,arrays,labels,args,device):
 out=RUN/f'{objective}_seed{seed}';out.mkdir(exist_ok=False)
 e4.set_seed(seed)
 model=nn.Sequential(nn.Linear(1536,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,4)).to(device)
 assert sum(p.numel() for p in model.parameters())==197252
 initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 counts=labels['train'].sum(0);assert (counts>0).all();w=counts.rsqrt();w=w/w.mean()
 target=labels['train'].argmax(1);steps=len(target)//64;gen=torch.Generator().manual_seed(seed)
 history=[];best=-float('inf');state=None;epochbest=None
 for epoch in range(1,31):
  model.train();order=torch.randperm(len(target),generator=gen).to(device)
  for step in range(steps):
   idx=order[step*64:(step+1)*64];lr=e4.lr_for_step((epoch-1)*steps+step,30*steps,5*steps,args)
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True);raw=model(arrays['train'][idx])
   loss=nn.functional.cross_entropy(raw,target[idx],weight=w)
   assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  val,_,_=evaluate(model,arrays['val'],labels['val'],'CE');history.append({'epoch':epoch,'val':val})
  if val['macro_roc_auc']>best:best=val['macro_roc_auc'];state=copy.deepcopy(model.state_dict());epochbest=epoch
 model.load_state_dict(state)
 val,_,valraw=evaluate(model,arrays['val'],labels['val'],'CE')
 test,scores,raw=evaluate(model,arrays['test'],labels['test'],'CE')
 torch.save({'model':{k:v.cpu() for k,v in state.items()},'best_epoch':epochbest,'objective':objective},out/'checkpoint-best.pth')
 torch.save(initial,out/'initial_state.pth')
 np.save(out/'test_raw_outputs.npy',raw);np.save(out/'test_scores.npy',scores);np.save(out/'val_raw_outputs.npy',valraw)
 e4.dump(out/'history.json',history)
 r={'objective':objective,'seed':seed,'trainable_parameters':197252,'best_epoch':epochbest,'val':val,'test':test,'loss_weights':w.cpu().tolist()}
 e4.dump(out/'summary.json',r);print(json.dumps({'objective':objective,'seed':seed,'best_epoch':epochbest,'val_macro_auc':best,'test_macro_auc':test['macro_roc_auc'],'test_macro_f1':test['macro_f1']}),flush=True)
 return r

def main():
 RUN.mkdir(parents=True,exist_ok=False)
 # Freeze the scheme before accessing validation or test labels.
 ytrain=np.load(e4.FEATURES/'train_labels.npy').astype(np.float32)
 assert np.isin(ytrain,[0,1]).all() and np.all(ytrain.sum(1)==1)
 counts=ytrain.sum(0);weights=1/np.sqrt(counts);weights=weights/weights.mean()
 config={'methods':['CE','class_balanced_CE'],'seeds':[0,1,2],
 'weight_rule_predefined':'n_class^(-1/2), arithmetic mean normalized to1, train labels only, no clipping/manual Control adjustment',
 'train_class_counts':dict(zip(e4.LABELS,map(int,counts))),'class_weights':dict(zip(e4.LABELS,map(float,weights))),
 'loss':'torch cross_entropy(logits,target,weight=w,reduction=mean); sum weighted NLL / sum target weights within batch',
 'architecture':'1536→128→GELU→Dropout0.1→4; bias; no LayerNorm;197252 trainable parameters',
 'epochs':30,'batch':64,'optimizer':'AdamW','lr':.0005,'min_lr':.00001,'warmup_lr':.000001,'warmup_epochs':5,'weight_decay':.05,'clip':3,
 'selection':'E7 main-head val Macro OVR AUC uniform mean; ties earliest epoch',
 'sampling':'E7 unchanged random shuffle/drop-last, no resampling; same seeded independent shuffle generator',
 'baseline_reuse':'E7 CE seeds0/1/2 completed results, no repeated formal test',
 'inference':'4class softmax and argmax, no label routing or threshold adjustment',
 'frozen_tokenizer':str(e4.TOKENIZER),'frozen_heartlang':str(e4.MLM),'features':str(e4.FEATURES),'source_protocol':json.loads((SOURCE/'protocol.json').read_text())}
 e4.dump(RUN/'protocol.json',config)
 files=[e4.TOKENIZER,e4.MLM,Path(__file__),ROOT/'scripts/curriculum_v1/run_e7_objective.py']+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ('heartlang_mean_max','labels')]
 before={str(p):e4.sha(p) for p in files};e4.dump(RUN/'source_fingerprints.json',before)
 shutil.copy2(SOURCE/'frozen_split_audit.json',RUN/'frozen_split_audit.json')
 # Synthetic mathematical precheck; not data/test evaluation.
 toy=torch.tensor([[1.,0.,-1.,2.],[0.,1.,2.,-1.]])
 targets=torch.tensor([0,2]);wt=torch.tensor(weights)
 loss=nn.functional.cross_entropy(toy,targets,weight=wt)
 manual=-(toy.log_softmax(1)[torch.arange(2),targets]*wt[targets]).sum()/wt[targets].sum()
 assert torch.allclose(loss,manual)
 e4.dump(RUN/'preflight.json',{'weighted_ce_manual_reduction_matches':True,'weight_source_train_only':True,'all_weights_finite_positive':bool(np.isfinite(weights).all() and (weights>0).all()),'fixture_not_result':True})
 device=torch.device('cuda');torch.set_num_threads(4)
 arrays,_=e4.load_arrays('heartlang_mean_max',device)
 labels={s:torch.tensor(np.load(e4.FEATURES/f'{s}_labels.npy'),device=device,dtype=torch.float32) for s in arrays}
 for s,y in labels.items():assert torch.equal(y.sum(1),torch.ones(len(y),device=device)) and arrays[s].shape==(len(y),1536)
 args=argparse.Namespace(lr=.0005,min_lr=.00001,warmup_lr=.000001,weight_decay=.05)
 results=[]
 for seed in (0,1,2):
  src=SOURCE/f'CE_seed{seed}';shutil.copytree(src,RUN/f'CE_seed{seed}')
  r=json.loads((src/'summary.json').read_text());r['reused_from']=str(src);results.append(r)
 for seed in (0,1,2):
  (RUN/'status.txt').write_text(f'training_class_balanced_CE_seed{seed}')
  results.append(train('class_balanced_CE',seed,arrays,labels,args,device));e4.dump(RUN/'partial_results.json',results)
 checks=[]
 for r in results:
  out=RUN/f"{r['objective']}_seed{r['seed']}";raw=np.load(out/'test_raw_outputs.npy');metrics,_=e7.classification(raw,labels['test'].cpu().numpy(),'CE');assert metrics==r['test']
  h=json.loads((out/'history.json').read_text());assert max(h,key=lambda x:x['val']['macro_roc_auc'])['epoch']==r['best_epoch'] and metrics['n_records']==1366
  checks.append({'objective':r['objective'],'seed':r['seed'],'all_saved_metrics_recomputed':True,'val_selection_verified':True})
 for seed in (0,1,2):
  a=torch.load(RUN/f'CE_seed{seed}/initial_state.pth',weights_only=True);b=torch.load(RUN/f'class_balanced_CE_seed{seed}/initial_state.pth',weights_only=True);assert all(torch.equal(a[k],b[k]) for k in a)
 assert all(e4.sha(p)==before[str(p)] for p in files)
 e4.dump(RUN/'comparison.json',{'config':config,'results':results})
 e4.dump(RUN/'report_validation.json',{'checks':checks,'identical_paired_initial_states':True,'frozen_sources_and_cache_hashes_unchanged':True,'weights_computed_before_val_test_access':True,'split_counts':{s:len(y) for s,y in labels.items()},'no_resampling':True})
 (RUN/'status.txt').write_text('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
