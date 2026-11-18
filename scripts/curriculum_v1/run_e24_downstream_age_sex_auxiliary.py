#!/usr/bin/env python3
"""E24: patient-only age and sex auxiliary supervision on E12-A downstream.

Age and sex are training targets only, never model inputs. Healthy records have
neither target and contribute zero demographic auxiliary loss. Classification
checkpoint selection and inference remain ECG-only and use Validation Overall
R2; Test is never loaded or evaluated.
"""
from __future__ import annotations
import argparse, copy, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'heartlang').exists())
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/experiments'), str(ROOT/'scripts/curriculum_v1')]
import run_e12_continuous_input as e12
import run_e23_downstream_age_auxiliary as e23

E12_RUN = e23.E12_RUN
DEFAULT_RUN = ROOT/'results/e24_downstream_age_sex_auxiliary'
AGE_WEIGHT = 0.10
SEX_WEIGHT = 0.10

def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def set_status(run: Path, value: str) -> None:
    run.mkdir(parents=True, exist_ok=True); (run/'status.txt').write_text(value+'\n'); print(value, flush=True)

def build_targets(run: Path) -> None:
    e23.build_age_targets(run)
    out = run/'sex'
    if (out/'COMPLETE').exists(): return
    out.mkdir(parents=True, exist_ok=False)
    audit = {
        'experiment':'E24 downstream patient-only age and sex auxiliary',
        'sex_source':'frozen downstream metadata column gender',
        'encoding':{'女':0,'男':1}, 'sex_is_model_input':False,
        'healthy_policy':'missing age and sex; both auxiliary losses masked; no fabricated targets',
        'training_scope':'downstream Train patients only',
        'validation_policy':'classification checkpoint selected only by Val Overall R2; demographics are post-selection diagnostics',
        'test_policy':'Test metadata/features/labels are not loaded or evaluated', 'splits':{}
    }
    for split in ('train','val'):
        frame=pd.read_csv(e12.DOWNSTREAM/f'{split}_metadata.csv', low_memory=False)
        labels=np.load(e12.DOWNSTREAM/f'{split}_labels.npy').argmax(1)
        patient=labels!=3
        raw=frame['gender']
        recognized=raw.isin(['男','女']).to_numpy()
        if not recognized[patient].all() or recognized[~patient].any():
            raise RuntimeError(f'{split}: gender presence/encoding does not exactly match patient status')
        target=np.zeros(len(frame),dtype=np.int64); target[raw.eq('男').to_numpy()]=1
        per_patient=frame.loc[patient,['patient_sn','gender']].drop_duplicates()
        conflicts=per_patient.groupby('patient_sn')['gender'].nunique()
        if (conflicts>1).any(): raise RuntimeError(f'{split}: conflicting gender within patient')
        by_class={}
        for i,name in enumerate(e12.LABEL_NAMES):
            m=labels==i
            by_class[name]={'records':int(m.sum()),'male':int((m & (target==1) & recognized).sum()),
                            'female':int((m & (target==0) & recognized).sum()),
                            'missing':int((m & ~recognized).sum())}
        audit['splits'][split]={'records':len(frame),'patient_records':int(patient.sum()),
            'healthy_records':int((~patient).sum()),'patient_sex_present':int(recognized[patient].sum()),
            'healthy_sex_present':int(recognized[~patient].sum()),
            'unique_patients':int(frame.loc[patient,'patient_sn'].nunique()),'conflicting_patients':0,
            'male':int((patient & (target==1)).sum()),'female':int((patient & (target==0)).sum()),
            'by_class':by_class}
        np.save(out/f'{split}_sex_target.npy',target)
        np.save(out/f'{split}_sex_mask.npy',patient.astype(bool))
    n0=audit['splits']['train']['female']; n1=audit['splits']['train']['male']; n=n0+n1
    weights=[n/(2*n0), n/(2*n1)]
    audit['train_class_weights']={'female':weights[0],'male':weights[1],
        'formula':'N/(2*N_class); applied only to Train patient sex CE'}
    dump(out/'audit.json',audit); (out/'COMPLETE').touch()

class AgeSexAuxE11(e23.AgeAuxE11):
    def __init__(self, seed:int):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(240000+int(seed)); self.sex_head=nn.Linear(128,2)
    def training_outputs_with_demographics(self, mlm, global_ze, regions, region_mask):
        x,_,_=self.encode(mlm,global_ze,regions,region_mask); hidden=self.classifier.shared(x)
        return self.classifier.main(hidden),self.classifier.aux(hidden),self.age_head(hidden).squeeze(-1),self.sex_head(hidden)
    def predict_demographics(self, mlm, global_ze, regions, region_mask):
        x,_,_=self.encode(mlm,global_ze,regions,region_mask); hidden=self.classifier.shared(x)
        return self.age_head(hidden).squeeze(-1),self.sex_head(hidden)

def sex_loss(logits,target,mask,class_weights):
    return F.cross_entropy(logits[mask].float(),target[mask].long(),weight=class_weights) if mask.any() else logits.sum()*0.0

def smoke(run:Path,device:torch.device)->None:
    if (run/'smoke.json').exists(): return
    build_targets(run); data=e23.load_ecg_data(device)
    mask_np=np.load(run/'age/train_age_mask.npy'); age_np=np.load(run/'age/train_age_z.npy')
    sex_np=np.load(run/'sex/train_sex_target.npy'); sex_mask_np=np.load(run/'sex/train_sex_mask.npy')
    patient=np.flatnonzero(mask_np)[:32]; healthy=np.flatnonzero(~mask_np)[:32]; idx_np=np.r_[patient,healthy]
    idx=torch.from_numpy(idx_np).long().to(device); age=torch.from_numpy(age_np[idx_np]).float().to(device)
    age_mask=torch.from_numpy(mask_np[idx_np]).bool().to(device); sex=torch.from_numpy(sex_np[idx_np]).long().to(device)
    sex_mask=torch.from_numpy(sex_mask_np[idx_np]).bool().to(device)
    audit=json.loads((run/'sex/audit.json').read_text()); sw=torch.tensor([audit['train_class_weights']['female'],audit['train_class_weights']['male']],device=device)
    model=AgeSexAuxE11(0).to(device); age_only=e23.AgeAuxE11(0).to(device); baseline=e12.e11.E11RegionResidual(0).to(device)
    state=model.state_dict(); common=max(float((state[k]-v).abs().max()) for k,v in baseline.state_dict().items())
    age_init=max(float((state[k]-v).abs().max()) for k,v in age_only.state_dict().items())
    regions,region_mask=e12.e11.masked_region_means(data['train']['lead'][idx],data['train']['lead_mask'][idx])
    main,aux,pred_age,pred_sex=model.training_outputs_with_demographics(data['train']['mlm'][idx],data['train']['global'][idx],regions,region_mask)
    loss=e23.age_loss(pred_age,age,age_mask)+sex_loss(pred_sex,sex,sex_mask,sw); loss.backward()
    grads={'shared':float(model.classifier.shared[0].weight.grad.norm()),'age_head':float(model.age_head.weight.grad.norm()),'sex_head':float(model.sex_head.weight.grad.norm())}
    model.eval(); baseline.eval()
    with torch.no_grad():
        b=baseline(data['train']['mlm'][idx],data['train']['global'][idx],regions,region_mask)
        m=model(data['train']['mlm'][idx],data['train']['global'][idx],regions,region_mask)
    result={'common_e12_initialization_max_abs_diff':common,'age_head_initialization_vs_e23_max_abs_diff':age_init,
        'initial_inference_logits_max_abs_diff':float((b-m).abs().max()),'demographic_gradient_norms':grads,
        'patient_age_targets':int(age_mask.sum()),'healthy_age_targets':int(age_mask[len(patient):].sum()),
        'patient_sex_targets':int(sex_mask.sum()),'healthy_sex_targets':int(sex_mask[len(patient):].sum()),
        'loss_finite':bool(torch.isfinite(loss)),'demographics_are_inference_inputs':False,'test_loaded':False}
    if (not result['loss_finite'] or common!=0 or age_init!=0 or result['initial_inference_logits_max_abs_diff']!=0
        or result['healthy_age_targets']!=0 or result['healthy_sex_targets']!=0
        or not all(math.isfinite(x) and x>0 for x in grads.values())): raise RuntimeError(f'Smoke failed: {result}')
    dump(run/'smoke.json',result)

@torch.no_grad()
def evaluate_demographics(model,arrays,age_z,age_mask,sex_target,sex_mask,age_mean,age_sd):
    model.eval(); regions,rm=e12.e11.masked_region_means(arrays['lead'],arrays['lead_mask']); ap=[]; sl=[]
    for s in range(0,len(age_z),96):
        a,x=model.predict_demographics(arrays['mlm'][s:s+96],arrays['global'][s:s+96],regions[s:s+96],rm[s:s+96]); ap.append(a.cpu()); sl.append(x.cpu())
    ap=torch.cat(ap).numpy(); sl=torch.cat(sl).numpy(); am=age_mask.cpu().numpy().astype(bool); sm=sex_mask.cpu().numpy().astype(bool)
    at=age_z.cpu().numpy()[am]*age_sd+age_mean; ay=ap[am]*age_sd+age_mean; err=ay-at; denom=float(((at-at.mean())**2).sum())
    prob=torch.softmax(torch.from_numpy(sl[sm]),1).numpy()[:,1]; truth=sex_target.cpu().numpy()[sm]; pred=(prob>=.5).astype(int)
    return {'age':{'patients':int(am.sum()),'mae_years':float(np.abs(err).mean()),'rmse_years':float(np.sqrt((err**2).mean())),'r2':float(1-(err**2).sum()/denom),'selection_metric':False},
            'sex':{'patients':int(sm.sum()),'accuracy':float((pred==truth).mean()),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'roc_auc':float(roc_auc_score(truth,prob)),'male_prevalence':float(truth.mean()),'selection_metric':False}}

def aggregate(items):
    result=e23.aggregate(items)
    result['sex_diagnostic']={k:{'mean':float(np.mean([r['sex_diagnostic'][k] for r in items])),'sd':float(np.std([r['sex_diagnostic'][k] for r in items],ddof=1))} for k in ('accuracy','balanced_accuracy','roc_auc')}
    return result

def train_seed(run,seed,device):
    out=run/'downstream'/f'seed{seed}'
    if (out/'VAL_COMPLETE').exists(): return json.loads((out/'val_summary.json').read_text())
    out.mkdir(parents=True,exist_ok=False); data=e23.load_ecg_data(device)
    aa=json.loads((run/'age/audit.json').read_text()); sa=json.loads((run/'sex/audit.json').read_text()); mean=aa['normalization']['mean_years']; sd=aa['normalization']['sd_years']
    age={s:torch.from_numpy(np.load(run/f'age/{s}_age_z.npy')).float().to(device) for s in ('train','val')}; am={s:torch.from_numpy(np.load(run/f'age/{s}_age_mask.npy')).bool().to(device) for s in ('train','val')}
    sex={s:torch.from_numpy(np.load(run/f'sex/{s}_sex_target.npy')).long().to(device) for s in ('train','val')}; sm={s:torch.from_numpy(np.load(run/f'sex/{s}_sex_mask.npy')).bool().to(device) for s in ('train','val')}
    sw=torch.tensor([sa['train_class_weights']['female'],sa['train_class_weights']['male']],device=device)
    target=data['train']['labels'].argmax(1); model=AgeSexAuxE11(seed).to(device); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
    steps=len(target)//64; gen=torch.Generator().manual_seed(seed); regions,rm=e12.e11.masked_region_means(data['train']['lead'],data['train']['lead_mask'])
    history=[]; best=-float('inf'); best_epoch=None; best_state=None
    for epoch in range(1,31):
        model.train(); order=torch.randperm(len(target),generator=gen).to(device); sums=dict(main=0.,patient=0.,age=0.,sex=0.,batches=0)
        for step in range(steps):
            idx=order[step*64:(step+1)*64]; lr=e12.lr_at((epoch-1)*steps+step,30*steps,5*steps,5e-4,1e-5,1e-6)
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True); main,aux,pa,ps=model.training_outputs_with_demographics(data['train']['mlm'][idx],data['train']['global'][idx],regions[idx],rm[idx])
            lm=F.cross_entropy(main,target[idx]); lp=e12.e9.patient_auxiliary_loss(aux,target[idx]); la=e23.age_loss(pa,age['train'][idx],am['train'][idx]); ls=sex_loss(ps,sex['train'][idx],sm['train'][idx],sw)
            loss=lm+.2*lp+AGE_WEIGHT*la+SEX_WEIGHT*ls
            if not torch.isfinite(loss):raise RuntimeError(f'seed{seed}/epoch{epoch}: nonfinite')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step()
            for k,v in [('main',lm),('patient',lp),('age',la),('sex',ls)]:sums[k]+=float(v)
            sums['batches']+=1
        val,_,_=e12.eval_downstream(model,'e12a_region',data['val']); demo=evaluate_demographics(model,data['val'],age['val'],am['val'],sex['val'],sm['val'],mean,sd)
        row={'epoch':epoch,**{f'train_{k}':sums[k]/sums['batches'] for k in ('main','patient','age','sex')},'val':val,'val_demographic_diagnostic':demo};history.append(row);print(json.dumps({'seed':seed,**row},ensure_ascii=False),flush=True)
        if val['r2']>best:best=val['r2'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);val,prob,logits=e12.eval_downstream(model,'e12a_region',data['val']);demo=evaluate_demographics(model,data['val'],age['val'],am['val'],sex['val'],sm['val'],mean,sd)
    torch.save({'model':{k:v.cpu() for k,v in best_state.items()},'seed':seed,'best_epoch':best_epoch,'selection':'Val Overall R2','test_run':False,'age_weight':AGE_WEIGHT,'sex_weight':SEX_WEIGHT},out/'checkpoint-best.pth')
    np.save(out/'val_logits.npy',logits);np.save(out/'val_probabilities.npy',prob);dump(out/'history.json',history)
    result={'method':'e24_patient_age_sex_auxiliary','seed':seed,'best_epoch':best_epoch,'val':val,'age_diagnostic':demo['age'],'sex_diagnostic':demo['sex'],'test_run':False,'age_weight':AGE_WEIGHT,'sex_weight':SEX_WEIGHT,'inference_trainable_parameters':sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('age_head','sex_head'))),'training_only_demographic_parameters':sum(p.numel() for n,p in model.named_parameters() if n.startswith(('age_head','sex_head')))}
    dump(out/'val_summary.json',result);(out/'VAL_COMPLETE').touch();return result

def train_all(run,device):
    if (run/'validation_summary.json').exists():return
    rows=[train_seed(run,s,device) for s in (0,1,2)];baseline=json.loads((E12_RUN/'downstream/validation_selection.json').read_text())['methods']['e12a_region']
    age=json.loads((e23.DEFAULT_RUN/'validation_summary.json').read_text())['e23_age_auxiliary']
    dump(run/'validation_summary.json',{'validation_only':True,'test_run':False,'e12a_historical':{'aggregate':e23.aggregate(baseline['per_seed']),'per_seed':baseline['per_seed']},'e23_age_0p10':age,'e24_age_sex_0p10':{'aggregate':aggregate(rows),'per_seed':rows}})

def report(run):
    s=json.loads((run/'validation_summary.json').read_text()); b=s['e12a_historical']['aggregate'];a=s['e23_age_0p10']['aggregate'];m=s['e24_age_sex_0p10']['aggregate'];rows=[]
    for label,key in [('E12-A','e12a_historical'),('E23-Age','e23_age_0p10'),('E24-Age+Sex','e24_age_sex_0p10')]:
        for x in s[key]['per_seed']:
            v=x['val'];r={'method':label,'seed':x.get('seed',x.get('downstream_seed')),'overall_r2':v['r2'],**{f"{n.lower().replace(' ','_')}_r2":v['per_output_r2'][n] for n in e12.LABEL_NAMES},'macro_auc':v['macro_roc_auc'],'macro_f1':v['macro_f1']}
            if 'age_diagnostic'in x:r.update(age_mae=x['age_diagnostic']['mae_years'],age_r2=x['age_diagnostic']['r2'])
            if 'sex_diagnostic'in x:r.update(sex_auc=x['sex_diagnostic']['roc_auc'],sex_bal_acc=x['sex_diagnostic']['balanced_accuracy'])
            rows.append(r)
    pd.DataFrame(rows).to_csv(run/'metrics_per_seed.csv',index=False)
    def row(label,x):return f"| {label} | {x['r2']['mean']:.6f} ± {x['r2']['sd']:.6f} | {x['patient_mean_r2']['mean']:.6f} | "+' | '.join(f"{x['per_output_r2'][n]['mean']:.6f}" for n in e12.LABEL_NAMES)+f" | {x['macro_roc_auc']['mean']:.6f} | {x['macro_f1']['mean']:.6f} |"
    lines=['# E24：E12-A下游患者年龄+性别辅助监督','','- 年龄和性别均仅是Train患者的辅助目标，不是模型输入；Healthy两项损失均屏蔽。','- 年龄SmoothL1权重0.10；性别反频率加权CE权重0.10。','- checkpoint只按Val Overall R²选择；只做Validation，未运行Test。','','| 方法 | Overall R² | 患者三类均值 | OMI | NOMI | Control | Healthy | Macro AUC | Macro F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|',row('E12-A',b),row('E23 年龄0.10',a),row('E24 年龄+性别',m),'',f"E24相对E23年龄0.10 Overall R²差值：**{m['r2']['mean']-a['r2']['mean']:+.6f}**。",f"E24相对E12-A Overall R²差值：**{m['r2']['mean']-b['r2']['mean']:+.6f}**。",f"年龄诊断MAE：**{m['age_diagnostic']['mae_years']['mean']:.3f}岁**；性别诊断AUC：**{m['sex_diagnostic']['roc_auc']['mean']:.4f}**，balanced accuracy：**{m['sex_diagnostic']['balanced_accuracy']['mean']:.4f}**。",'']
    (run/'README.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','smoke','train','report']);p.add_argument('--output_dir',type=Path,default=DEFAULT_RUN);args=p.parse_args();run=args.output_dir.resolve();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.stage=='audit':set_status(run,'audit_patient_age_and_sex_targets');build_targets(run)
    elif args.stage=='smoke':set_status(run,'smoke_exact_e12_inference_and_demographic_gradients');smoke(run,device)
    elif args.stage=='train':set_status(run,'train_e24_age_sex_seeds_0_1_2');train_all(run,device)
    else:report(run);set_status(run,'complete');(run/'COMPLETE').touch()
if __name__=='__main__':main()
