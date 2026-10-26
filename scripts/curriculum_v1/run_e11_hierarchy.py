#!/usr/bin/env python3
"""E11 frozen ECG features, predicted binary gate + reused E10 conditional head."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,confusion_matrix,accuracy_score,f1_score
import run_e4_aggregation as e4
import run_e10_patient_diagnostic as e10
ROOT=e4.ROOT;RUN=ROOT/'results/curriculum_v1/e11_predicted_hierarchy_20261012'
E7=ROOT/'results/curriculum_v1/e7_objective_20261011';E10=ROOT/'results/curriculum_v1/e10_patient_diagnostic_20261011'
LABELS=e4.LABELS

def gate_metrics(raw,y):
 target=y[:,3].astype(int);p=e10.softmax(raw);pred=p.argmax(1);pr,re,f,s=precision_recall_fscore_support(target,pred,labels=[0,1],zero_division=0)
 return {'roc_auc':float(roc_auc_score(1-target,p[:,0])),'patient_recall':float(re[0]),'healthy_recall':float(re[1]),'accuracy':float(accuracy_score(target,pred)),
 'confusion_matrix':confusion_matrix(target,pred,labels=[0,1]).tolist(),'class_order':['Patient','Healthy'],'n_records':len(y)}

@torch.no_grad()
def outputs(model,x):
 model.eval();return np.concatenate([model(x[i:i+96]).cpu().numpy() for i in range(0,len(x),96)])

def combine(gate_raw,patient_raw,mode):
 # Inference API accepts model outputs only, never ground-truth labels.
 g=e10.softmax(gate_raw);q=e10.softmax(patient_raw)
 pp=(g[:,0]>0.5).astype(np.float32) if mode=='Hard' else g[:,0]
 return np.concatenate([pp[:,None]*q,(1-pp)[:,None]],axis=1)

def final_metrics(score,y,pair_logits):
 target=y.argmax(1);pred=score.argmax(1);patient=target!=3;pair=(target==0)|(target==1)
 pr,re,f,s=precision_recall_fscore_support(target,pred,labels=range(4),zero_division=0)
 auc=[float(roc_auc_score(y[:,i],score[:,i])) for i in range(4)]
 cm=confusion_matrix(target,pred,labels=range(4));result=e4.metrics(score,y)
 result.update(macro_roc_auc=float(np.mean(auc)),macro_f1=float(f1_score(target,pred,labels=range(4),average='macro',zero_division=0)),
 balanced_accuracy=float(np.mean(re)),accuracy=float(accuracy_score(target,pred)),
 per_class={k:{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f[i]),'auc':auc[i],'support':int(s[i])} for i,k in enumerate(LABELS)},
 confusion_matrix=cm.tolist(),omi_vs_nomi_auc=float(roc_auc_score(y[pair,0],score[pair,0]-score[pair,1])),
 conditional_omi_vs_nomi_auc=float(roc_auc_score(y[pair,0],pair_logits[pair,0]-pair_logits[pair,1])),
 patient_subset={'n_records':int(patient.sum()),'accuracy':float(accuracy_score(target[patient],pred[patient])),
 'balanced_accuracy':float(np.mean(re[:3])),'macro_f1':float(f1_score(target[patient],pred[patient],labels=range(3),average='macro',zero_division=0)),
 'macro_roc_auc':float(np.mean([roc_auc_score(y[patient,i],score[patient,i]) for i in range(3)])),
 'predicted_healthy':int((pred[patient]==3).sum()),'confusion_matrix_rows_patient_columns_all4':cm[:3].tolist()})
 assert cm.sum()==len(y) and np.allclose(np.diag(cm)/cm.sum(1),re)
 return result

def train_gate(seed,arrays,labels,args,device):
 out=RUN/f'gate_seed{seed}';out.mkdir(exist_ok=False);e4.set_seed(seed)
 model=nn.Sequential(nn.Linear(1536,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,2)).to(device)
 assert sum(p.numel() for p in model.parameters())==196994
 torch.save({k:v.cpu().clone() for k,v in model.state_dict().items()},out/'initial_state.pth')
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 target=labels['train'][:,3].long();steps=len(target)//64;gen=torch.Generator().manual_seed(seed);history=[];best=-float('inf');state=None;epochbest=None
 for epoch in range(1,31):
  model.train();order=torch.randperm(len(target),generator=gen).to(device)
  for step in range(steps):
   idx=order[step*64:(step+1)*64];lr=e4.lr_for_step((epoch-1)*steps+step,30*steps,5*steps,args)
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(arrays['train'][idx]),target[idx]);assert torch.isfinite(loss)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  val=gate_metrics(outputs(model,arrays['val']),labels['val'].cpu().numpy());history.append({'epoch':epoch,'val':val})
  if val['roc_auc']>best:best=val['roc_auc'];state=copy.deepcopy(model.state_dict());epochbest=epoch
 model.load_state_dict(state);rawval=outputs(model,arrays['val']);rawtest=outputs(model,arrays['test'])
 torch.save({'model':{k:v.cpu() for k,v in state.items()},'best_epoch':epochbest},out/'checkpoint-best.pth')
 np.save(out/'val_raw_outputs.npy',rawval);np.save(out/'test_raw_outputs.npy',rawtest);e4.dump(out/'history.json',history)
 result={'seed':seed,'best_epoch':epochbest,'trainable_parameters':196994,'val':gate_metrics(rawval,labels['val'].cpu().numpy()),'test':gate_metrics(rawtest,labels['test'].cpu().numpy())}
 e4.dump(out/'summary.json',result);print(json.dumps(result),flush=True);return rawval,rawtest,result

def error_decomposition(gate_raw,patient_raw,score,y):
 true=y.argmax(1);ispatient=true!=3;g=e10.softmax(gate_raw);miss=ispatient&(g[:,0]<=.5)
 qpred=patient_raw.argmax(1);oraclewrong=ispatient&(qpred!=true);pred=score.argmax(1)
 return {'patient_count':int(ispatient.sum()),'gate_patient_false_negatives':int(miss.sum()),
 'gate_misses_patient_head_would_be_correct':int((miss&~oraclewrong).sum()),'gate_misses_patient_head_would_already_be_wrong':int((miss&oraclewrong).sum()),
 'patient_head_oracle_wrong':int(oraclewrong.sum()),'gate_passed_patient_head_wrong':int((ispatient&~miss&oraclewrong).sum()),
 'final_patient_wrong':int((ispatient&(pred!=true)).sum()),'final_patient_predicted_healthy':int((ispatient&(pred==3)).sum()),
 'additional_healthy_predictions_despite_gate_patient':int((ispatient&~miss&(pred==3)).sum()),
 'gate_false_negative_by_class':dict(zip(LABELS[:3],[int((miss&(true==i)).sum()) for i in range(3)]))}

def main():
 RUN.mkdir(parents=True,exist_ok=False)
 config={'methods':['CE','Hard','Soft'],'seeds':[0,1,2],'gate':'1536→128 GELU/dropout.1→2 plain CE; Patient0 Healthy1',
 'patient_head':'reuse E10 B1536→128→3 fixed checkpoint, seed-matched; no retraining',
 'threshold_predefined':.5,'threshold_tie':'Patient iff P_patient>.5, tie Healthy, no threshold fitting',
 'gate_selection':'val Healthy/Patient ROC-AUC, earliest ties','stage2_selection':'existing E10 patient-val Macro AUC',
 'hierarchy_selection':'no separate selection on test/end-to-end; stages chosen independently on respective val AUC',
 'soft_formula':'[Pgate(Patient)*P3(OMI), *P3(NOMI), *P3(Control), Pgate(Healthy)]',
 'hard_scores':'Patient gate=>P3 and Healthy0; Healthy gate=>[0,0,0,1]; AUC includes zeros/ties from hard routing',
 'pair_auc':'primary final output probability difference, identical definition for all3 methods; conditional logit-difference AUC additionally reported',
 'epochs':30,'batch':64,'optimizer':'AdamW','lr':.0005,'min_lr':.00001,'warmup_lr':.000001,'warmup_epochs':5,'weight_decay':.05,'clip':3,
 'parameters':'gate196994 + reused patient197123 =394117 vs CE197252; separate trunks, not parameter-matched',
 'frozen':{'tokenizer':str(e4.TOKENIZER),'heartlang':str(e4.MLM),'features':str(e4.FEATURES)},'no_label_inference_routing':True,'no_new_weight_loss_or_sampling':True}
 e4.dump(RUN/'protocol.json',config)
 files=[e4.TOKENIZER,e4.MLM,Path(__file__)]+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ('heartlang_mean_max','labels')]+[E10/f'B_patient_only_seed{i}/checkpoint-best.pth' for i in (0,1,2)]+[E7/f'CE_seed{i}/test_raw_outputs.npy' for i in (0,1,2)]
 before={str(p):e4.sha(p) for p in files};e4.dump(RUN/'source_fingerprints.json',before);shutil.copy2(E7/'frozen_split_audit.json',RUN/'frozen_split_audit.json')
 # Pure inference composition fixture, unrelated to real test.
 gs=np.array([[2.,0.],[0.,2.]],np.float32);qs=np.array([[2.,0.,0.],[0.,2.,0.]],np.float32)
 for mode in ('Hard','Soft'):assert np.allclose(combine(gs,qs,mode).sum(1),1)
 e4.dump(RUN/'preflight.json',{'probability_composition_normalized':True,'combination_takes_no_labels':True,'fixture_not_result':True})
 device=torch.device('cuda');torch.set_num_threads(4);arrays,_=e4.load_arrays('heartlang_mean_max',device)
 labels={s:torch.tensor(np.load(e4.FEATURES/f'{s}_labels.npy'),device=device,dtype=torch.float32) for s in arrays}
 args=argparse.Namespace(lr=.0005,min_lr=.00001,warmup_lr=.000001,weight_decay=.05);results=[];gates=[]
 old10=json.loads((E10/'comparison.json').read_text())['results']
 for seed in (0,1,2):
  (RUN/'status.txt').write_text(f'training_gate_seed{seed}')
  gv,gt,gate=train_gate(seed,arrays,labels,args,device);gates.append(gate)
  patient_model=nn.Sequential(nn.Linear(1536,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,3)).to(device)
  patient_model.load_state_dict(torch.load(E10/f'B_patient_only_seed{seed}/checkpoint-best.pth',map_location=device,weights_only=False)['model'],strict=True)
  # Frozen inference over all ECGs; no subset routing, targets are not model inputs.
  patient_model.requires_grad_(False);pv=outputs(patient_model,arrays['val']);pt=outputs(patient_model,arrays['test'])
  np.save(RUN/f'patient_head_all_val_seed{seed}.npy',pv);np.save(RUN/f'patient_head_all_test_seed{seed}.npy',pt)
  idx=np.flatnonzero(labels['test'][:,3].cpu().numpy()==0)
  assert np.allclose(pt[idx],np.load(E10/f'B_patient_only_seed{seed}/test_raw_outputs.npy'),atol=1e-5)
  for method in ('CE','Hard','Soft'):
   row={'method':method,'seed':seed,'trainable_parameters':197252 if method=='CE' else 394117,'gate_epoch':None if method=='CE' else gate['best_epoch']}
   for split,gr,pr in [('val',gv,pv),('test',gt,pt)]:
    if method=='CE':raw=np.load(E7/f'CE_seed{seed}/{split}_raw_outputs.npy');score=e10.softmax(raw);pair=raw
    else:score=combine(gr,pr,method);pair=pr
    row[split]=final_metrics(score,labels[split].cpu().numpy(),pair);np.save(RUN/f'{method}_{split}_scores_seed{seed}.npy',score)
   if method!='CE':row['test_error_decomposition']=error_decomposition(gt,pt,np.load(RUN/f'{method}_test_scores_seed{seed}.npy'),labels['test'].cpu().numpy())
   row['E10_oracle_reference']={x['method']:x['test'] for x in old10 if x['seed']==seed};results.append(row)
  e4.dump(RUN/'partial_results.json',results)
 for row in results:
  s=row['seed'];score=np.load(RUN/f"{row['method']}_test_scores_seed{s}.npy");pair=np.load(E7/f'CE_seed{s}/test_raw_outputs.npy') if row['method']=='CE' else np.load(RUN/f'patient_head_all_test_seed{s}.npy')
  assert final_metrics(score,labels['test'].cpu().numpy(),pair)==row['test'] and row['test']['n_records']==1366
 for g in gates:
  h=json.loads((RUN/f"gate_seed{g['seed']}/history.json").read_text());assert max(h,key=lambda x:x['val']['roc_auc'])['epoch']==g['best_epoch']
 assert all(e4.sha(p)==before[str(p)] for p in files)
 e4.dump(RUN/'comparison.json',{'config':config,'gate_results':gates,'results':results})
 e4.dump(RUN/'report_validation.json',{'all_9_saved_score_metrics_recomputed':True,'gate_val_best_epochs_verified':True,'frozen_sources_and_cache_hashes_unchanged':True,'reused_E10_patient_outputs_consistent':True,'no_ground_truth_input_to_combine':True,'split_counts':{s:len(y) for s,y in labels.items()}})
 (RUN/'status.txt').write_text('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
