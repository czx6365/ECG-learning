#!/usr/bin/env python3
"""E10 oracle removal diagnostic + patient-only CE3, original parent split."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,confusion_matrix,accuracy_score,f1_score
import run_e4_aggregation as e4
ROOT=e4.ROOT;RUN=ROOT/'results/curriculum_v1/e10_patient_diagnostic_20261011'
SOURCE=ROOT/'results/curriculum_v1/e7_objective_20261011';LABELS=e4.LABELS[:3]

def softmax(x):
 z=x-x.max(1,keepdims=True);z=np.exp(z);return z/z.sum(1,keepdims=True)

def metrics(raw,y,mode):
 target=y.argmax(1)
 scores=softmax(raw) if mode=='A_four_way' else softmax(raw[:,:3])
 pred=raw.argmax(1) if mode=='A_four_way' else raw[:,:3].argmax(1)
 pr,re,f1,support=precision_recall_fscore_support(target,pred,labels=range(3),zero_division=0)
 auc=[float(roc_auc_score(y[:,i],scores[:,i])) for i in range(3)]
 pair=(target==0)|(target==1)
 cm=confusion_matrix(target,pred,labels=range(4))[:3,:] if mode=='A_four_way' else confusion_matrix(target,pred,labels=range(3))
 result={'macro_roc_auc':float(np.mean(auc)),'macro_f1':float(f1_score(target,pred,labels=range(3),average='macro',zero_division=0)),
 'balanced_accuracy':float(np.mean(re)),'accuracy':float(accuracy_score(target,pred)),
 'omi_vs_nomi_auc':float(roc_auc_score(y[pair,0],raw[pair,0]-raw[pair,1])),
 'per_class':{k:{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'auc':auc[i],'support':int(support[i])} for i,k in enumerate(LABELS)},
 'confusion_matrix':cm.tolist(),'n_records':len(y),'predicted_healthy':int((pred==3).sum())}
 if raw.shape[1]==4:
  old=softmax(raw);result['macro_roc_auc_original_four_class_scores']=float(np.mean([roc_auc_score(y[:,i],old[:,i]) for i in range(3)]))
 assert cm.sum()==len(y) and np.allclose(np.diag(cm[:,:3])/cm.sum(1),re)
 return result,scores

@torch.no_grad()
def evaluate(model,x,y):
 model.eval();raw=np.concatenate([model(x[i:i+96]).cpu().numpy() for i in range(0,len(y),96)])
 m,score=metrics(raw,y.cpu().numpy(),'B_patient_only');return m,score,raw

def train(seed,arrays,labels,args,device):
 out=RUN/f'B_patient_only_seed{seed}';out.mkdir(exist_ok=False);e4.set_seed(seed)
 model=nn.Sequential(nn.Linear(1536,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,3))
 old=torch.load(SOURCE/f'CE_seed{seed}/initial_state.pth',weights_only=True)
 initial={'0.weight':old['0.weight'],'0.bias':old['0.bias'],'3.weight':old['3.weight'][:3].clone(),'3.bias':old['3.bias'][:3].clone()}
 model.load_state_dict(initial,strict=True);model.to(device);assert sum(p.numel() for p in model.parameters())==197123
 torch.save(initial,out/'initial_state.pth')
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 target=labels['train'].argmax(1);steps=len(target)//64;generator=torch.Generator().manual_seed(seed)
 best=-float('inf');state=None;best_epoch=None;history=[]
 for epoch in range(1,31):
  model.train();order=torch.randperm(len(target),generator=generator).to(device)
  for step in range(steps):
   idx=order[step*64:(step+1)*64];lr=e4.lr_for_step((epoch-1)*steps+step,30*steps,5*steps,args)
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True);raw=model(arrays['train'][idx]);loss=nn.functional.cross_entropy(raw,target[idx]);assert torch.isfinite(loss)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  val,_,_=evaluate(model,arrays['val'],labels['val']);history.append({'epoch':epoch,'val':val})
  if val['macro_roc_auc']>best:best=val['macro_roc_auc'];state=copy.deepcopy(model.state_dict());best_epoch=epoch
 model.load_state_dict(state);val,_,valraw=evaluate(model,arrays['val'],labels['val']);test,score,raw=evaluate(model,arrays['test'],labels['test'])
 torch.save({'model':{k:v.cpu() for k,v in state.items()},'best_epoch':best_epoch},out/'checkpoint-best.pth')
 np.save(out/'test_raw_outputs.npy',raw);np.save(out/'test_scores.npy',score);np.save(out/'val_raw_outputs.npy',valraw);e4.dump(out/'history.json',history)
 result={'method':'B_patient_only','seed':seed,'trainable_parameters':197123,'best_epoch':best_epoch,'steps_per_epoch':steps,'total_updates':steps*30,'val':val,'test':test}
 e4.dump(out/'summary.json',result);print(json.dumps({'seed':seed,'best_epoch':best_epoch,'val_macro_auc':best,'test_macro_auc':test['macro_roc_auc'],'macro_f1':test['macro_f1'],'control_recall':test['per_class']['Clinical Control']['recall']}),flush=True);return result

def main():
 RUN.mkdir(parents=True,exist_ok=False)
 config={'methods':['A_four_way','A_drop_healthy','B_patient_only'],'seeds':[0,1,2],
 'A':'fixed E7 saved val/test logits, true patient subset; four-way argmax vs patient-logits argmax, no retraining/reselection',
 'A_score_caveat':'A_drop_healthy softmax3 changes OVR ranks via renormalization; original softmax4 macro AUC also reported; pair logit-difference AUC unchanged',
 'B':'patient-only plain CE3, no Healthy train/val/test, original row indices retained',
 'architecture':'1536→128→GELU→Dropout0.1→3, bias, no LayerNorm;197123 parameters',
 'epochs':30,'batch':64,'optimizer':'AdamW','lr':.0005,'min_lr':.00001,'warmup_lr':.000001,'warmup_epochs':5,'weight_decay':.05,'clip':3,
 'selection':'B patient-val Macro OVR AUC uniform mean, ties earliest; A keeps existing E7 checkpoint',
 'budget':'B51steps/epoch1530updates; E7103steps/epoch3090updates; identical epoch/batch not equal optimization steps, no patient repetition to match count',
 'initialization':'same-seed E7 initial shared layer + first3 output rows, no trained head initialization',
 'no_weights_auxiliary_or_hierarchical_model':True,'no_test_tuning':True,'diagnostic_not_deployment':True,
 'tokenizer':str(e4.TOKENIZER),'heartlang':str(e4.MLM),'features':str(e4.FEATURES)}
 e4.dump(RUN/'protocol.json',config)
 files=[e4.TOKENIZER,e4.MLM,Path(__file__),ROOT/'scripts/curriculum_v1/run_e7_objective.py']+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ('heartlang_mean_max','labels')]+[SOURCE/f'CE_seed{k}'/f'{s}_raw_outputs.npy' for k in (0,1,2) for s in ('val','test')]
 before={str(p):e4.sha(p) for p in files};e4.dump(RUN/'source_fingerprints.json',before);shutil.copy2(SOURCE/'frozen_split_audit.json',RUN/'frozen_split_audit.json')
 device=torch.device('cuda');torch.set_num_threads(4)
 arrays={};labels={};indices={};audit={}
 for split in ('train','val','test'):
  y=np.load(e4.FEATURES/f'{split}_labels.npy').astype(np.float32);assert np.isin(y,[0,1]).all() and np.all(y.sum(1)==1)
  index=np.flatnonzero(y[:,3]==0);indices[split]=index;np.save(RUN/f'{split}_parent_patient_indices.npy',index)
  x=np.load(e4.FEATURES/f'{split}_heartlang_mean_max.npy')[index];yy=y[index,:3];assert np.all(yy.sum(1)==1)
  arrays[split]=torch.tensor(x,device=device,dtype=torch.float32);labels[split]=torch.tensor(yy,device=device,dtype=torch.float32)
  audit[split]={'parent_records':len(y),'patient_records':len(index),'patient_class_counts':dict(zip(LABELS,map(int,yy.sum(0)))),'parent_order_preserved':True}
 assert [len(indices[s]) for s in ('train','val','test')]==[3320,672,683];e4.dump(RUN/'patient_subset_audit.json',audit)
 toy=np.eye(3,dtype=np.float32);m,_=metrics(toy,toy,'B_patient_only');assert m['accuracy']==m['macro_roc_auc']==1
 toy4=np.concatenate([toy,np.full((3,1),2,dtype=np.float32)],axis=1);a,_=metrics(toy4,toy,'A_four_way');b,_=metrics(toy4,toy,'A_drop_healthy');assert a['accuracy']==0 and b['accuracy']==1 and a['omi_vs_nomi_auc']==b['omi_vs_nomi_auc']
 e4.dump(RUN/'preflight.json',{'oracle_argmax_removes_only_healthy_competition':True,'three_class_metrics_fixture_passed':True,'fixture_not_result':True})
 results=[]
 for seed in (0,1,2):
  old=json.loads((SOURCE/f'CE_seed{seed}/summary.json').read_text())
  for method in ('A_four_way','A_drop_healthy'):
   row={'method':method,'seed':seed,'best_epoch':old['best_epoch'],'trainable_parameters':197252,'reused_from':str(SOURCE/f'CE_seed{seed}')}
   for split in ('val','test'):
    raw=np.load(SOURCE/f'CE_seed{seed}'/f'{split}_raw_outputs.npy')[indices[split]];row[split]=metrics(raw,labels[split].cpu().numpy(),method)[0]
   results.append(row)
  assert results[-1]['test']['omi_vs_nomi_auc']==results[-2]['test']['omi_vs_nomi_auc']
 args=argparse.Namespace(lr=.0005,min_lr=.00001,warmup_lr=.000001,weight_decay=.05)
 for seed in (0,1,2):
  (RUN/'status.txt').write_text(f'training_patient_only_seed{seed}');results.append(train(seed,arrays,labels,args,device));e4.dump(RUN/'partial_results.json',results)
 checks=[]
 for row in results:
  if row['method']=='B_patient_only':
   out=RUN/f"B_patient_only_seed{row['seed']}";raw=np.load(out/'test_raw_outputs.npy');h=json.loads((out/'history.json').read_text());assert max(h,key=lambda a:a['val']['macro_roc_auc'])['epoch']==row['best_epoch']
  else:raw=np.load(SOURCE/f"CE_seed{row['seed']}"/'test_raw_outputs.npy')[indices['test']]
  assert metrics(raw,labels['test'].cpu().numpy(),row['method'])[0]==row['test'] and row['test']['n_records']==683
  checks.append({'method':row['method'],'seed':row['seed'],'saved_metrics_recomputed':True})
 assert all(e4.sha(p)==before[str(p)] for p in files)
 e4.dump(RUN/'comparison.json',{'config':config,'results':results});e4.dump(RUN/'report_validation.json',{'checks':checks,'frozen_sources_and_cache_hashes_unchanged':True,'original_E7_logits_unchanged':True,'subset_only_no_resplit':True,'pair_auc_invariance_A_verified':True,'B_val_selection_verified':True})
 (RUN/'status.txt').write_text('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
