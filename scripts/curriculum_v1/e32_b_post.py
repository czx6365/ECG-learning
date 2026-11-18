"""Complete threshold, metadata, and integrity audits for E32-B."""
import json, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, log_loss, f1_score
import e32_b_run as run

e,p,b,s=run.e,run.p,run.b,run.s
root=run.OUT;src=run.SRC;dev=run.DEV
meta=pd.read_csv(e.DOWNSTREAM/'train_metadata.csv')
y4=np.load(e.DOWNSTREAM/'train_labels.npy');y=y4.argmax(1)
folds=pd.read_csv(src/'fold_assignment.csv')
pred={name:np.load(root/f'{name}_oof.npy') for name in ('B0','B1','B2')}
load=lambda name:torch.tensor(np.load(e.E10_RUN/f'features/train_{name}.npy'),device=dev)
regions,mask=e.e11.masked_region_means(load('lead_ze_mean_max'),load('lead_present').bool())
data={'train':{'z':torch.tensor(np.load(p.source/'features/train.npy'),device=dev),'g':load('global_ze_mean_max'),'regions':regions,'mask':mask,'y':torch.tensor(y4,device=dev)}}
from e32_train_cache import extract
extract(data);data=data['train']
thresholds={n:np.zeros(len(y)) for n in pred}
audits=json.loads((root/'inner_crossfit_audit.json').read_text())
records=[]
for fold in range(3):
    held=folds[(folds.fold==fold)&(folds.role=='outer')].row.to_numpy()
    outerdev=np.sort(folds[(folds.fold==fold)&(folds.role!='outer')].row.to_numpy())
    split=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=32050+fold)
    pp=np.full((len(outerdev),4),np.nan);zz=np.full((len(outerdev),96),np.nan)
    for inner,(_,vi) in enumerate(split.split(outerdev,y[outerdev],meta.iloc[outerdev].entity_id)):
        models=[]
        for seed in range(3):
            ck=root/f'fold{fold}/inner{inner}/downstream/shared/seed{seed}/checkpoint-best.pth'
            m=s.PairedHead('shared',seed).to(dev)
            m.load_state_dict(torch.load(ck,map_location=dev,weights_only=False)['model']);models.append(m.eval())
        pp[vi],zz[vi]=run.predict(models,run.restrict(data,outerdev[vi]))
    model=np.load(root/f'fold{fold}_model.npz')
    r=np.log(pp[:,0]/pp[:,1]); x=(zz-model['scaler_mean'])/model['scaler_scale']
    rr={'B0':r,'B1':np.exp(model['t'][0])*r+model['t'][1],
        'B2':r+x@model['weights'][:-1]+model['weights'][-1]}
    pair=y[outerdev]<2
    for name,score in rr.items():
        fpr,tpr,th=roc_curve(y[outerdev][pair]==0,score[pair])
        ix=np.flatnonzero(fpr<=.1); t=th[ix[np.argmax(tpr[ix])]]
        thresholds[name][held]=float(t)
        q=pred[name][held,0]/pred[name][held,:2].sum(1)
        sens=float((np.log(pred[name][held,0]/pred[name][held,1])[y[held]==0]>=t).mean())
        spec=float((np.log(pred[name][held,0]/pred[name][held,1])[y[held]==1]<t).mean())
        records.append({'fold':fold,'model':name,'threshold':float(t),'omi_sensitivity':sens,'nomi_specificity':spec,
                        'inner_pair_n':int(pair.sum()),'outer_pair_n':int((y[held]<2).sum())})
    current=run.load_outer(fold)[1]
    saved=json.loads((root/'fold_manifest.json').read_text())[fold]['outer_checkpoint_sha256']
    assert current==saved
    audits[2*fold]['post_hoc_sha_match']=True
    audits[2*fold+1]['post_hoc_sha_match']=True
pd.DataFrame(records).to_csv(root/'threshold_metrics.csv',index=False)
run.save('inner_crossfit_audit.json',audits)
run.save('stacked_training_manifest.json',{'outer_folds':3,'crossfit_inner_folds':2,
    'model_selection':'inner training side separate five-fold group selection; first split used',
    'second_level_training':'all outer-development records have held-out predictions and representation',
    'threshold_source':'outer-development cross-fitted predictions only'})
run.save('parameter_counts.json',{'B0':0,'B1':2,'B2':97})
run.save('environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
    'test_loaded':False,'validation_loaded':False,'e12_sha256':e.sha256(e.DEFAULT_RUN/'mlm/scratch/checkpoint-best.pth')})
for name,arr in thresholds.items():np.save(root/f'{name}_threshold.npy',arr)
rows=[]
for name,prob in pred.items():
    for i in range(len(y)):
        rows.append({'ecg_id':str(meta.iloc[i].csv_name),'entity_id':str(meta.iloc[i].entity_id),
         'outer_fold':int(folds[(folds.role=='outer')&(folds.row==i)].fold.iloc[0]),'model':name,'label':int(y[i]),
         **{f'p_{j}':float(prob[i,j]) for j in range(4)},
         **{f'base_p_{j}':float(pred['B0'][i,j]) for j in range(4)}})
pd.DataFrame(rows).to_csv(root/'predictions.csv',index=False)
run.save('prediction_manifest.json',{'file':'predictions.csv','label_map':['OMI','NOMI','Clinical Control','Healthy'],
 'seed':'ensemble of 0,1,2','test_loaded':False})
metrics={}
for name,prob in pred.items():
    pair=y<2;z=(y[pair]==0).astype(float);q=prob[pair,0]/prob[pair,:2].sum(1)
    bins=np.minimum((q*10).astype(int),9);ece=sum((bins==k).mean()*abs(z[bins==k].mean()-q[bins==k].mean()) for k in range(10) if (bins==k).any())
    metrics[name]={'binary_brier':float(np.square(q-z).mean()),'binary_log_loss':float(log_loss(z,q)),
      'binary_ece10':float(ece),'pr_auc':float(average_precision_score(z,q)),'macro_f1':float(f1_score(y,prob.argmax(1),average='macro'))}
run.save('calibration_metrics.json',metrics)
(root/'README_CN.md').write_text('E32-B：固定 E29-S 外层分组；两折内层交叉拟合；B1 正温度+偏置，B2 96维 Shared 线性残差。见 E32_B_REPORT_CN.md。\n')
print('post audit complete')
