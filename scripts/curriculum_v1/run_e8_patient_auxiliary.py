#!/usr/bin/env python3
"""E8: main-only inference with masked patient auxiliary supervision, fixed lambda=.2."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
import run_e4_aggregation as e4
import run_e7_objective as e7
ROOT=e4.ROOT;RUN=ROOT/'results/curriculum_v1/e8_patient_auxiliary_20261011'
SOURCE=ROOT/'results/curriculum_v1/e7_objective_20261011'
LAMBDA=.2

class PatientAuxiliary(nn.Module):
 def __init__(self,baseline,seed):
  super().__init__()
  self.shared=nn.Sequential(baseline[0],baseline[1],baseline[2])
  self.main=baseline[3]
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(83000+seed)
   self.aux=nn.Linear(128,3)
 def forward(self,x):
  # No labels or auxiliary output used by inference.
  return self.main(self.shared(x))
 def training_outputs(self,x):
  h=self.shared(x)
  return self.main(h),self.aux(h)

def auxiliary_loss(logits,target):
 patient=target!=3
 return nn.functional.cross_entropy(logits[patient],target[patient]) if patient.any() else logits.sum()*0

def preflight():
 logits=torch.randn(4,3,requires_grad=True);target=torch.tensor([0,3,1,2]);loss=auxiliary_loss(logits,target);loss.backward()
 assert torch.equal(logits.grad[1],torch.zeros(3)) and (logits.grad[[0,2,3]].abs().sum(1)>0).all()
 logits=torch.randn(3,3,requires_grad=True);loss=auxiliary_loss(logits,torch.tensor([3,3,3]));loss.backward();assert loss.item()==0 and torch.equal(logits.grad,torch.zeros_like(logits))
 return {'healthy_rows_zero_auxiliary_gradient':True,'empty_patient_batch_zero_loss_and_gradient':True,'fixture_not_experiment_result':True}

def train(seed,arrays,labels,args,device):
 out=RUN/f'CE_patient_aux_seed{seed}';out.mkdir(exist_ok=False)
 e4.set_seed(seed)
 base=nn.Sequential(nn.Linear(1536,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,4))
 initial_e7=torch.load(SOURCE/f'CE_seed{seed}/initial_state.pth',weights_only=True)
 assert all(torch.equal(base.state_dict()[k],initial_e7[k]) for k in initial_e7)
 model=PatientAuxiliary(base,seed).to(device)
 assert sum(p.numel() for p in model.parameters())==197639
 initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
 torch.save(initial,out/'initial_state.pth')
 model.eval()
 with torch.no_grad():assert torch.equal(model(arrays['train'][:8]),base.to(device)(arrays['train'][:8]))
 # Baseline shares the same modules; no optimizer updates have occurred.
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 target=labels['train'].argmax(1);steps=len(target)//64;gen=torch.Generator().manual_seed(seed)
 history=[];best=-float('inf');state=None;epochbest=None
 for epoch in range(1,31):
  model.train();order=torch.randperm(len(target),generator=gen).to(device);main_losses=[];aux_losses=[];patient_counts=[]
  for step in range(steps):
   idx=order[step*64:(step+1)*64];lr=e4.lr_for_step((epoch-1)*steps+step,30*steps,5*steps,args)
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True);main,aux=model.training_outputs(arrays['train'][idx])
   loss_main=nn.functional.cross_entropy(main,target[idx]);loss_aux=auxiliary_loss(aux,target[idx]);loss=loss_main+LAMBDA*loss_aux
   assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
   main_losses.append(loss_main.item());aux_losses.append(loss_aux.item());patient_counts.append(int((target[idx]!=3).sum()))
  val,_,_=e7.evaluate(model,arrays['val'],labels['val'],'CE')
  history.append({'epoch':epoch,'val':val,'train_main_ce':float(np.mean(main_losses)),'train_patient_ce':float(np.mean(aux_losses)),'patient_counts':patient_counts})
  if val['macro_roc_auc']>best:best=val['macro_roc_auc'];state=copy.deepcopy(model.state_dict());epochbest=epoch
 model.load_state_dict(state)
 val,_,valraw=e7.evaluate(model,arrays['val'],labels['val'],'CE')
 test,scores,raw=e7.evaluate(model,arrays['test'],labels['test'],'CE')
 torch.save({'model':{k:v.cpu() for k,v in state.items()},'best_epoch':epochbest,'lambda':LAMBDA},out/'checkpoint-best.pth')
 np.save(out/'test_raw_outputs.npy',raw);np.save(out/'test_scores.npy',scores);np.save(out/'val_raw_outputs.npy',valraw)
 e4.dump(out/'history.json',history)
 r={'objective':'CE_patient_aux','seed':seed,'lambda':LAMBDA,'trainable_parameters':197639,'main_inference_parameters':197252,'auxiliary_parameters':387,'best_epoch':epochbest,'val':val,'test':test}
 e4.dump(out/'summary.json',r)
 print(json.dumps({'objective':r['objective'],'seed':seed,'best_epoch':epochbest,'val_macro_auc':best,'test_macro_auc':test['macro_roc_auc'],'test_macro_f1':test['macro_f1'],'patient_macro_f1':test['patient_subset']['macro_f1'],'control_recall':test['per_class']['Clinical Control']['recall']}),flush=True)
 return r

def main():
 RUN.mkdir(parents=True,exist_ok=False)
 config={'methods':['CE','CE_patient_aux'],'seeds':[0,1,2],'lambda_predefined':LAMBDA,'lambda_candidates':[LAMBDA],
 'architecture':'shared Linear1536→128/GELU/Dropout0.1; main Linear128→4; auxiliary Linear128→3',
 'loss':'mean CE4 on all batch + .2 * mean CE3 on patient rows only; no patient batch => auxiliary zero',
 'epochs':30,'batch':64,'optimizer':'AdamW','lr':.0005,'min_lr':.00001,'warmup_lr':.000001,'warmup_epochs':5,'weight_decay':.05,'clip':3,
 'selection':'main-head val Macro OVR AUC mean, ties earliest epoch; lambda not selected using test',
 'inference':'main4 head only; forward takes features only; no real-label or auxiliary routing',
 'baseline_reuse':'E7 CE seed0/1/2 complete results/checkpoints, no duplicate test',
 'representation':'frozen E3 HeartLang final valid token MeanMax1536; original E4 caches and split',
 'initialization':'paired shared/main initial weights exactly E7; isolated aux RNG seed83000+seed preserves main RNG',
 'tokenizer':str(e4.TOKENIZER),'heartlang':str(e4.MLM),'source_protocol':json.loads((SOURCE/'protocol.json').read_text())}
 e4.dump(RUN/'protocol.json',config)
 files=[e4.TOKENIZER,e4.MLM,Path(__file__),ROOT/'scripts/curriculum_v1/run_e7_objective.py']+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ('heartlang_mean_max','labels')]
 before={str(p):e4.sha(p) for p in files};e4.dump(RUN/'source_fingerprints.json',before)
 shutil.copy2(SOURCE/'frozen_split_audit.json',RUN/'frozen_split_audit.json')
 e4.dump(RUN/'preflight.json',preflight())
 device=torch.device('cuda');torch.set_num_threads(4)
 arrays,_=e4.load_arrays('heartlang_mean_max',device)
 labels={s:torch.tensor(np.load(e4.FEATURES/f'{s}_labels.npy'),device=device,dtype=torch.float32) for s in arrays}
 for s,y in labels.items():assert torch.equal(y.sum(1),torch.ones(len(y),device=device)) and arrays[s].shape==(len(y),1536)
 results=[]
 for seed in (0,1,2):
  src=SOURCE/f'CE_seed{seed}';shutil.copytree(src,RUN/f'CE_seed{seed}')
  r=json.loads((src/'summary.json').read_text());r.update(reused_from=str(src),lambda_value=0,main_inference_parameters=197252,auxiliary_parameters=0);results.append(r)
 args=argparse.Namespace(lr=.0005,min_lr=.00001,warmup_lr=.000001,weight_decay=.05)
 for seed in (0,1,2):
  (RUN/'status.txt').write_text(f'training_CE_patient_aux_seed{seed}')
  results.append(train(seed,arrays,labels,args,device));e4.dump(RUN/'partial_results.json',results)
 checks=[]
 for r in results:
  out=RUN/f"{r['objective']}_seed{r['seed']}";raw=np.load(out/'test_raw_outputs.npy');m,_=e7.classification(raw,labels['test'].cpu().numpy(),'CE');assert m==r['test']
  h=json.loads((out/'history.json').read_text());assert max(h,key=lambda x:x['val']['macro_roc_auc'])['epoch']==r['best_epoch'] and m['n_records']==1366
  checks.append({'objective':r['objective'],'seed':r['seed'],'all_saved_metrics_recomputed':True,'val_selection_verified':True})
 for seed in (0,1,2):
  a=torch.load(RUN/f'CE_seed{seed}/initial_state.pth',weights_only=True);b=torch.load(RUN/f'CE_patient_aux_seed{seed}/initial_state.pth',weights_only=True)
  mapping={'0.weight':'shared.0.weight','0.bias':'shared.0.bias','3.weight':'main.weight','3.bias':'main.bias'}
  assert all(torch.equal(a[k],b[v]) for k,v in mapping.items())
 assert all(e4.sha(p)==before[str(p)] for p in files)
 e4.dump(RUN/'comparison.json',{'config':config,'results':results})
 e4.dump(RUN/'report_validation.json',{'checks':checks,'paired_shared_main_initial_weights_equal':True,'frozen_sources_and_cache_hashes_unchanged':True,'split_counts':{s:len(y) for s,y in labels.items()},'auxiliary_params':387,'main_only_inference':True})
 (RUN/'status.txt').write_text('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
