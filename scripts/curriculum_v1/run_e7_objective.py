#!/usr/bin/env python3
"""E7 frozen E3/E6 representation, two losses and shared val macro-AUC selection."""
import argparse,copy,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,confusion_matrix,accuracy_score,balanced_accuracy_score,f1_score
import run_e4_aggregation as e4
ROOT=e4.ROOT;RUN=ROOT/'results/curriculum_v1/e7_objective_20261011'
LABELS=e4.LABELS

def classification(raw,y,objective):
 target=y.argmax(1);pred=raw.argmax(1)
 if objective=='CE':
  z=raw-raw.max(1,keepdims=True);prob=np.exp(z);scores=prob/prob.sum(1,keepdims=True)
 else:scores=raw
 pr,re,f1,support=precision_recall_fscore_support(target,pred,labels=range(4),zero_division=0)
 cm=confusion_matrix(target,pred,labels=range(4))
 aucs=[float(roc_auc_score(y[:,k],scores[:,k])) for k in range(4)]
 pair=(target==0)|(target==1);patient=target!=3
 result=e4.metrics(scores,y)
 result.update(macro_roc_auc=float(np.mean(aucs)),macro_f1=float(f1_score(target,pred,labels=range(4),average='macro',zero_division=0)),
  balanced_accuracy=float(balanced_accuracy_score(target,pred)),accuracy=float(accuracy_score(target,pred)),
  per_class={k:{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'auc':aucs[i],'support':int(support[i])} for i,k in enumerate(LABELS)},
  confusion_matrix=cm.tolist(),omi_vs_nomi_auc=float(roc_auc_score(y[pair,0],raw[pair,0]-raw[pair,1])),
  omi_vs_nomi_output_score_difference_auc=float(roc_auc_score(y[pair,0],scores[pair,0]-scores[pair,1])),
  patient_subset={'n_records':int(patient.sum()),'accuracy':float(accuracy_score(target[patient],pred[patient])),
   'balanced_accuracy':float(np.mean([re[i] for i in range(3)])),
   'macro_f1':float(f1_score(target[patient],pred[patient],labels=range(3),average='macro',zero_division=0)),
   'macro_roc_auc':float(np.mean([roc_auc_score(y[patient,k],scores[patient,k]) for k in range(3)])),
   'predicted_healthy':int((pred[patient]==3).sum()),
   'confusion_matrix_rows_patient_columns_all4':confusion_matrix(target[patient],pred[patient],labels=range(4))[:3].tolist()})
 assert int(cm.sum())==len(y) and np.allclose(np.diag(cm)/cm.sum(1),re)
 return result,scores

@torch.no_grad()
def evaluate(model,x,y,objective):
 model.eval();raw=np.concatenate([model(x[i:i+96]).cpu().numpy() for i in range(0,len(y),96)])
 return (*classification(raw,y.cpu().numpy(),objective),raw)

def train(objective,seed,arrays,labels,args,device):
 out=RUN/f'{objective}_seed{seed}';out.mkdir(exist_ok=False)
 e4.set_seed(seed)
 model=nn.Sequential(nn.Linear(1536,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,4)).to(device)
 assert sum(p.numel() for p in model.parameters())==197252
 initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 v=labels['train'].var(0,unbiased=False).clamp_min(1e-8);w=1/v;w=w/w.mean()
 target=labels['train'].argmax(1);steps=len(target)//64;gen=torch.Generator().manual_seed(seed)
 history=[];best=-float('inf');state=None;epochbest=None
 for epoch in range(1,31):
  model.train();order=torch.randperm(len(target),generator=gen).to(device)
  for step in range(steps):
   idx=order[step*64:(step+1)*64];lr=e4.lr_for_step((epoch-1)*steps+step,30*steps,5*steps,args)
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True);raw=model(arrays['train'][idx])
   loss=(((raw-labels['train'][idx]).square()*w).mean() if objective=='weighted_MSE' else nn.functional.cross_entropy(raw,target[idx]))
   assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  val,_,_=evaluate(model,arrays['val'],labels['val'],objective);history.append({'epoch':epoch,'val':val})
  if val['macro_roc_auc']>best:best=val['macro_roc_auc'];state=copy.deepcopy(model.state_dict());epochbest=epoch
 model.load_state_dict(state)
 val,_,valraw=evaluate(model,arrays['val'],labels['val'],objective)
 test,scores,raw=evaluate(model,arrays['test'],labels['test'],objective)
 torch.save({'model':{k:v.cpu() for k,v in state.items()},'best_epoch':epochbest,'objective':objective},out/'checkpoint-best.pth')
 torch.save(initial,out/'initial_state.pth')
 np.save(out/'test_raw_outputs.npy',raw);np.save(out/'test_scores.npy',scores);np.save(out/'val_raw_outputs.npy',valraw)
 e4.dump(out/'history.json',history)
 r={'objective':objective,'seed':seed,'trainable_parameters':197252,'best_epoch':epochbest,'val':val,'test':test,'loss_weights':w.cpu().tolist() if objective=='weighted_MSE' else None}
 e4.dump(out/'summary.json',r);print(json.dumps({'objective':objective,'seed':seed,'best_epoch':epochbest,'val_macro_auc':best,'test_macro_auc':test['macro_roc_auc'],'test_macro_f1':test['macro_f1']}),flush=True)
 return r

def main():
 RUN.mkdir(parents=True,exist_ok=False)
 config={'objectives':['weighted_MSE','CE'],'seeds':[0,1,2],'selection':'val Macro ROC-AUC (unweighted mean of four OVR AUCs), ties earliest epoch, both losses',
 'architecture':'1536→128→GELU→Dropout0.1→4, bias, no LayerNorm','trainable_parameters':197252,
 'epochs':30,'batch':64,'optimizer':'AdamW','lr':.0005,'min_lr':.00001,'warmup_lr':.000001,'warmup_epochs':5,'weight_decay':.05,'clip':3,
 'CE':'plain 4-class CE logits, no weights or label smoothing','MSE':'E6 inverse-train-label-variance weighted one-hot regression',
 'classification_scores':'MSE raw outputs; CE softmax probabilities; argmax4 with no threshold tuning',
 'OMI_NOMI_primary':'MSE raw output difference; CE logit difference, equivalent ranking to conditional pair softmax',
 'patient_evaluation':'subset true patients, predictions remain 4-way; Healthy errors retained',
 'R2_MSE_continuity':'MSE raw outputs; CE softmax probabilities; four-output uniform-average R2, unweighted reporting MSE',
 'tokenizer':str(e4.TOKENIZER),'heartlang':str(e4.MLM),'features':str(e4.FEATURES),'no_pretrained_training':True,
 'difference_from_E6':'both heads retrained with identical val macro-AUC selection, E6 historical R2-selected heads not primary E7 baseline'}
 e4.dump(RUN/'protocol.json',config)
 files=[e4.TOKENIZER,e4.MLM,Path(__file__),ROOT/'scripts/curriculum_v1/run_e4_aggregation.py',ROOT/'scripts/curriculum_v1/run_e6_pooling_parameter_matched.py']+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ('heartlang_mean_max','labels')]
 before={str(p):e4.sha(p) for p in files};e4.dump(RUN/'source_fingerprints.json',before)
 shutil.copy2(ROOT/'results/curriculum_v1/e4_e3_aggregation_20261011/frozen_split_audit.json',RUN/'frozen_split_audit.json')
 device=torch.device('cuda');torch.set_num_threads(4)
 arrays,_=e4.load_arrays('heartlang_mean_max',device)
 labels={s:torch.tensor(np.load(e4.FEATURES/f'{s}_labels.npy'),device=device,dtype=torch.float32) for s in arrays}
 for s,y in labels.items():
  assert np.all(np.isin(y.cpu().numpy(),[0,1])) and torch.equal(y.sum(1),torch.ones(len(y),device=device))
  assert arrays[s].shape==(len(y),1536)
 args=argparse.Namespace(lr=.0005,min_lr=.00001,warmup_lr=.000001,weight_decay=.05)
 # Tiny deterministic metric fixture checks, unrelated to actual train/val/test.
 toy=np.eye(4,dtype=np.float32);m,_=classification(toy,toy,'weighted_MSE');assert m['accuracy']==m['macro_roc_auc']==1
 m,prob=classification(toy,toy,'CE');assert m['accuracy']==m['macro_roc_auc']==1 and np.allclose(prob.sum(1),1)
 results=[]
 for objective in config['objectives']:
  for seed in config['seeds']:
   (RUN/'status.txt').write_text(f'training_{objective}_seed{seed}')
   results.append(train(objective,seed,arrays,labels,args,device));e4.dump(RUN/'partial_results.json',results)
 checks=[]
 for r in results:
  out=RUN/f"{r['objective']}_seed{r['seed']}";raw=np.load(out/'test_raw_outputs.npy');m,score=classification(raw,labels['test'].cpu().numpy(),r['objective'])
  assert m==r['test'];hist=json.loads((out/'history.json').read_text());assert max(hist,key=lambda h:h['val']['macro_roc_auc'])['epoch']==r['best_epoch']
  assert m['n_records']==1366 and abs(np.mean(list(m['per_output_r2'].values()))-m['r2'])<1e-10
  checks.append({'objective':r['objective'],'seed':r['seed'],'all_saved_metrics_recomputed':True,'val_selection_verified':True})
 for seed in config['seeds']:
  a=torch.load(RUN/f'weighted_MSE_seed{seed}/initial_state.pth',weights_only=True);b=torch.load(RUN/f'CE_seed{seed}/initial_state.pth',weights_only=True)
  assert all(torch.equal(a[k],b[k]) for k in a)
 assert all(e4.sha(p)==before[str(p)] for p in files)
 e4.dump(RUN/'comparison.json',{'config':config,'results':results})
 e4.dump(RUN/'report_validation.json',{'checks':checks,'identical_paired_initial_states':True,'frozen_sources_and_cache_hashes_unchanged':True,'split_counts':{s:len(y) for s,y in labels.items()},'class_counts':{s:dict(zip(LABELS,map(int,y.sum(0).cpu().tolist()))) for s,y in labels.items()},'fixture_not_result':True})
 (RUN/'status.txt').write_text('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
