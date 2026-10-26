#!/usr/bin/env python3
"""E6 parameter-matched uncompressed pooling, preregistered seeds 0/1/2."""
import argparse, copy, csv, json, shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
import run_e4_aggregation as e4
from run_e4_aggregation import evaluate, dump, lr_for_step, set_seed
ROOT=e4.ROOT
RUN=ROOT/'results/curriculum_v1/e6_pooling_parameter_matched_20261011'
SOURCE=ROOT/'results/curriculum_v1/e4_e3_aggregation_20261011'
def train(method,kind,arrays,masks,labels,args,device,seed,hidden):
    set_seed(seed)
    attention=masks is not None
    dim=arrays['train'].shape[-1]
    if attention:
        model=AttentionPoolingHead(code_dim=dim,hidden_dim=64)
        model.head=head(dim,kind)
    else:model=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,4))
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    variance=labels['train'].var(0,unbiased=False).clamp_min(1e-8)
    weights=1/variance;weights=weights/weights.mean()
    if kind=='linear':weights=torch.ones_like(weights)
    steps=len(labels['train'])//64
    generator=torch.Generator().manual_seed(seed)
    history=[];best=-float('inf');best_state=None;best_epoch=None
    for epoch in range(1,31):
        model.train();order=torch.randperm(len(labels['train']),generator=generator).to(device)
        for step in range(steps):
            indices=order[step*64:(step+1)*64]
            lr=lr_for_step((epoch-1)*steps+step,30*steps,5*steps,args)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            pred=model(arrays['train'][indices],masks['train'][indices]) if attention else model(arrays['train'][indices])
            loss=((pred-labels['train'][indices]).square()*weights).mean()
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);optimizer.step()
        val,_=evaluate(model,arrays['val'],masks['val'] if attention else None,labels['val'])
        history.append({'epoch':epoch,'val':val})
        if val['r2']>best:
            best=val['r2'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val,_=evaluate(model,arrays['val'],masks['val'] if attention else None,labels['val'])
    test,pred=evaluate(model,arrays['test'],masks['test'] if attention else None,labels['test'])
    out=RUN/method/kind;out.mkdir(parents=True,exist_ok=False)
    torch.save({'model':{k:v.cpu() for k,v in best_state.items()},'best_epoch':best_epoch},out/'checkpoint-best.pth')
    np.save(out/'test_predictions.npy',pred)
    dump(out/'history.json',history)
    result={'method':method,'head':kind,'input_dim':dim,'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'best_epoch':best_epoch,'val':val,'test':test,'loss_weights':weights.cpu().tolist()}
    if attention:
        attention_summary={}
        model.eval()
        with torch.no_grad():
            # Concentration diagnostics only, not evidence of abnormal-beat detection.
            for split in ('val','test'):
                stats=[]
                for start in range(0,len(labels[split]),96):
                    _,w=model(arrays[split][start:start+96],masks[split][start:start+96],return_attention=True)
                    count=masks[split][start:start+96].sum(1);ok=count>0
                    if ok.any():
                        w=w[ok];n=count[ok]
                        stats.append(torch.stack([w.max(1).values,w.topk(5,dim=1).values.sum(1),
                            1/(w.square().sum(1)*n),-(w*w.clamp_min(1e-12).log()).sum(1)/n.clamp_min(2).log()],1).cpu())
                attention_summary[split]=dict(zip(['top1_mass','top5_mass','effective_fraction','normalized_entropy'],torch.cat(stats).mean(0).tolist()))
        result['attention_concentration']=attention_summary
    dump(out/'summary.json',result)
    print(json.dumps({'method':method,'head':kind,'best_epoch':best_epoch,'val_r2':best,'test_r2':test['r2']}),flush=True)
    return result


def main():
 RUN.mkdir(parents=True,exist_ok=False)
 count=lambda d,h:(d+5)*h+4
 budget=count(768,256);hidden=min(range(1,1025),key=lambda h:abs(count(1536,h)-budget))
 assert budget==197892 and hidden==128 and count(1536,hidden)==197252
 specs=[('CLS', 'heartlang',768,256),('Mean','heartlang_mean',768,256),('MeanMax_fair','heartlang_mean_max',1536,hidden),('MeanMax_reference','heartlang_mean_max',1536,256)]
 config={'methods':specs,'seeds':[0,1,2],'parameter_formula':'(input_dim+5)*hidden+4; Linear biases included, no LayerNorm',
  'auto_hidden_search':'integer1..1024 minimizes absolute distance from CLS parameter budget; chosen before training',
  'epochs':30,'batch':64,'optimizer':'AdamW','lr':.0005,'min_lr':.00001,'warmup_lr':.000001,'warmup_epochs':5,'weight_decay':.05,'clip':3,'dropout':.1,
  'loss':'E4 train-variance-normalized four-output MSE','standardization':False,'selection':'val four-output uniform-average R2; no test selection',
  'reused_seed0_anchors':'E4 completed CLS, Mean, raw MeanMax; no repeated test',
  'tokenizer':str(e4.TOKENIZER),'heartlang':str(e4.MLM),'source_protocol':json.loads((SOURCE/'protocol.json').read_text())}
 dump(RUN/'protocol.json',config)
 files=[e4.TOKENIZER,e4.MLM]+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ('heartlang','heartlang_mean','heartlang_mean_max','labels')]
 before={str(p):e4.sha(p) for p in files};dump(RUN/'source_fingerprints.json',before)
 shutil.copy2(SOURCE/'split_audit.json',RUN/'split_audit.json') if (SOURCE/'split_audit.json').exists() else None
 device=torch.device('cuda');torch.set_num_threads(4)
 labels={s:torch.tensor(np.load(e4.FEATURES/f'{s}_labels.npy'),device=device,dtype=torch.float32) for s in ('train','val','test')}
 args=argparse.Namespace(lr=.0005,min_lr=.00001,warmup_lr=.000001,weight_decay=.05)
 results=[]
 for method,cache,dim,h in specs:
  arrays,_=e4.load_arrays(cache,device)
  for seed in (0,1,2):
   name=f'{method}_seed{seed}';(RUN/'status.txt').write_text('training_'+name)
   if seed==0 and method!='MeanMax_fair':
    src=SOURCE/cache/'mlp2';dst=RUN/name/'mlp2';shutil.copytree(src,dst)
    r=json.loads((src/'summary.json').read_text());r['method']=name;r['reused_from']=str(src)
   else:r=train(name,'mlp2',arrays,None,labels,args,device,seed,h)
   r.update(pooling=method,seed=seed,hidden=h)
   assert r['trainable_parameters']==count(dim,h)
   pred=np.load(RUN/name/'mlp2/test_predictions.npy');y=labels['test'].cpu().numpy();patient=y[:,3]==0
   from sklearn.metrics import r2_score
   r['test']['patient_subset_per_output_r2']=dict(zip(e4.LABELS[:3],map(float,r2_score(y[patient,:3],pred[patient,:3],multioutput='raw_values'))))
   assert abs(e4.metrics(pred,y)['r2']-r['test']['r2'])<1e-10
   hist=json.loads((RUN/name/'mlp2/history.json').read_text());assert max(hist,key=lambda v:v['val']['r2'])['epoch']==r['best_epoch']
   dump(RUN/name/'mlp2/summary.json',r);results.append(r)
   dump(RUN/'partial_results.json',results)
 unchanged=all(e4.sha(p)==before[str(p)] for p in files);assert unchanged
 audit={'checkpoint_and_cache_hashes_unchanged':unchanged,'n_test':1366,'split_counts':{s:len(y) for s,y in labels.items()},'all_12_heads_metrics_recomputed':True,'all_val_best_epoch_verified':True,'fair_parameter_difference':count(1536,hidden)-budget,'fair_parameter_relative_difference':(count(1536,hidden)-budget)/budget}
 dump(RUN/'report_validation.json',audit);dump(RUN/'comparison.json',{'config':config,'results':results})
 (RUN/'status.txt').write_text('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
