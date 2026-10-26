#!/usr/bin/env python3
"""E4 fixed E3 representation aggregation; regression-only matched heads."""
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'heartlang').exists())
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/experiments'), str(ROOT/'scripts/curriculum_v1')]
from extract_information_flow_representations import extract_split, load_heartlang
from evaluate_vocabulary_pilot import load_model
from train_information_flow_linear_probes import regression_metrics, lr_for_step, set_seed
from train_attention_pooling_probes import AttentionPoolingHead

EXP = 'e4_e3_aggregation_20261011'
E3 = 'e3_healthy_patient_replay_20261011'
RUN = ROOT/'results/curriculum_v1'/EXP
FEATURES = ROOT/'outputs/curriculum_v1'/EXP
TOKENIZER = ROOT/f'checkpoints/curriculum_v1/{E3}_adapt_e30/checkpoint-best.pth'
MLM = ROOT/f'checkpoints/curriculum_v1/{E3}_sid_mlm_e100/checkpoint-best.pth'
SID = ROOT/f'datasets/ecg_datasets/ECG_DSA/qrs_sid/{E3}'
DATA = ROOT/'datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1'
LABELS = ['OMI','NOMI','Clinical Control','Healthy']
METHODS = ['z_e','z_e_mean_max','z_q','z_q_mean_max','sid_frequency',
           'z_e_attention','z_q_attention','heartlang','heartlang_mean','heartlang_mean_max']


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def status(value):
    (RUN/'status.txt').write_text(value)
    print(value, flush=True)


def pooled(tokens, valid):
    mean = (tokens * valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
    maximum = tokens.masked_fill(~valid[...,None], -torch.inf).amax(1)
    maximum[~valid.any(1)] = 0
    return mean, torch.cat([mean, maximum],1)


@torch.inference_mode()
def extract(device):
    FEATURES.mkdir(parents=True,exist_ok=False)
    tokenizer = load_model(TOKENIZER, device)
    heartlang, config = load_heartlang(MLM, device)
    for model in (tokenizer,heartlang):
        model.requires_grad_(False)
    before = {name: {k:v.detach().clone() for k,v in model.state_dict().items()}
              for name,model in [('tokenizer',tokenizer),('heartlang',heartlang)]}
    audit = {'heartlang_config':config,'splits':{},'test_features_only':True}
    for split in ('train','val','test'):
        features, labels = extract_split(tokenizer, heartlang, DATA, SID, split, 32, 64, device, FEATURES)
        sid = np.load(SID/f'{split}_sid.npy', mmap_mode='r')
        valid = np.load(FEATURES/f'{split}_valid_heartbeat_mask.npy')
        assert np.array_equal(valid, (sid>=0)&(sid<1024))
        frequency = np.zeros((len(labels),1024),np.float32)
        for i in range(len(labels)):
            s=sid[i][valid[i]]
            frequency[i]=np.bincount(s,minlength=1024)/max(len(s),1)
        assert np.allclose(frequency.sum(1),valid.any(1).astype(float))
        features['sid_frequency']=frequency
        means=[];mean_maxs=[]
        chans=np.load(SID/f'{split}_data_in_chans.npy',mmap_mode='r')
        times=np.load(SID/f'{split}_data_in_times.npy',mmap_mode='r')
        for start in range(0,len(sid),32):
            ids=torch.tensor(np.array(sid[start:start+32]),device=device,dtype=torch.long)
            c=torch.tensor(np.array(chans[start:start+32]),device=device,dtype=torch.long)
            t=torch.tensor(np.array(times[start:start+32]),device=device,dtype=torch.long)
            all_tokens, mask=heartlang.forward_feature(ids,in_chan_matrix=c,in_time_matrix=t)
            assert np.array_equal(mask.cpu().numpy(),valid[start:start+32])
            assert np.allclose(all_tokens[:,0].cpu().numpy(),features['heartlang'][start:start+32],atol=1e-5)
            m,mm=pooled(all_tokens[:,1:],mask)
            means.append(m.cpu().numpy());mean_maxs.append(mm.cpu().numpy())
        features['heartlang_mean']=np.concatenate(means)
        features['heartlang_mean_max']=np.concatenate(mean_maxs)
        for name,values in features.items():
            assert np.isfinite(values).all()
            np.save(FEATURES/f'{split}_{name}.npy',values)
        np.save(FEATURES/f'{split}_labels.npy',labels)
        audit['splits'][split]={'records':len(labels),'empty_ecg':int((~valid.any(1)).sum()),
                               'valid_heartbeats':int(valid.sum()),'feature_dimensions':{k:v.shape[1] for k,v in features.items()}}
    for name,model in [('tokenizer',tokenizer),('heartlang',heartlang)]:
        for key,value in before[name].items():
            if name=='tokenizer' and key=='quantize.cluster_size':continue
            assert torch.equal(value,model.state_dict()[key]), f'Frozen state changed: {name}/{key}'
    audit['frozen_learned_and_ema_state']='passed'
    audit['labels_channels_times_wave_vs_sid']='passed'
    audit['padding_masks_and_histogram']='passed'
    dump(RUN/'extraction_audit.json',audit)


def metrics(pred,y):
    result=regression_metrics(pred,y)
    patient=y[:,3]==0
    pair=(y[:,0]==1)|(y[:,1]==1)
    result.update({'patient_three_output_r2_on_all_ecg':float(np.mean([result['per_output_r2'][k] for k in LABELS[:3]])),
      'patient_subset_three_output_r2':float(r2_score(y[patient,:3],pred[patient,:3],multioutput='uniform_average')),
      'omi_vs_nomi_auc':float(roc_auc_score(y[pair,0],pred[pair,0]-pred[pair,1])),
      'n_records':len(y),'rmse':float(np.sqrt(result['mse']))})
    return result


def head(dim,kind):
    return nn.Linear(dim,4) if kind=='linear' else nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,4))


def load_arrays(method,device):
    attention=method.endswith('_attention')
    representation=method.replace('_attention','')
    arrays={}
    for split in ('train','val','test'):
        file=f'{split}_{representation}_tokens.npy' if attention else f'{split}_{method}.npy'
        arrays[split]=torch.tensor(np.load(FEATURES/file),device=device,dtype=torch.float32)
    masks={s:torch.tensor(np.load(FEATURES/f'{s}_valid_heartbeat_mask.npy'),device=device) for s in arrays} if attention else None
    return arrays,masks


@torch.no_grad()
def evaluate(model,x,mask,y):
    model.eval();pred=[]
    for start in range(0,len(y),96):
        output=model(x[start:start+96],mask[start:start+96]) if mask is not None else model(x[start:start+96])
        pred.append(output.cpu().numpy())
    pred=np.concatenate(pred)
    return metrics(pred,y.cpu().numpy()),pred


def train(method,kind,arrays,masks,labels,args,device):
    set_seed(0)
    attention=masks is not None
    dim=arrays['train'].shape[-1]
    if attention:
        model=AttentionPoolingHead(code_dim=dim,hidden_dim=64)
        model.head=head(dim,kind)
    else:model=head(dim,kind)
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    variance=labels['train'].var(0,unbiased=False).clamp_min(1e-8)
    weights=1/variance;weights=weights/weights.mean()
    if kind=='linear':weights=torch.ones_like(weights)
    steps=len(labels['train'])//64
    generator=torch.Generator().manual_seed(0)
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


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def report(results,config):
    dump(RUN/'comparison.json',{'config':config,'results':results})
    lines=['# E4：固定E3表示的ECG聚合方法完整中文对照报告','',
       '## 1. 结论与公平对照','',
       '本实验不训练tokenizer或HeartLang，固定E3最佳tokenizer epoch30与MLM epoch60。比较10种聚合，每种使用linear MSE和共享MLP方差归一化MSE两种头，共20个预先指定实验。主指标为四输出等权平均R²，epoch只由val选择。跨方法正式推荐也按val四输出R²选择，test成绩只用于最终对照。','']
    for kind in ('linear','mlp2'):
        rows=[r for r in results if r['head']==kind]
        chosen=max(rows,key=lambda r:r['val']['r2']);cls=next(r for r in rows if r['method']=='heartlang')
        observed=max(rows,key=lambda r:r['test']['r2'])
        lines += [f"- {kind}：val选择最佳方法 **{chosen['method']}**，test R²={chosen['test']['r2']:.6f}；相对同协议CLS {cls['test']['r2']:.6f}，差值 **{chosen['test']['r2']-cls['test']['r2']:+.6f}**。test表中观察到的最高方法为{observed['method']}（{observed['test']['r2']:.6f}），仅描述排名，不据此选择新配置。"]
    lines += ['', '## 2. 实际协议','',
      '固定下游split6640/1344/1366，验证身份互斥及既有split manifest。特征提取推理允许读取test波形，但不会拟合或更新任何模型。所有方法来自同一ECG、相同有效心拍mask；SID frequency为1024维L1归一化频率，忽略padding；无有效心拍的聚合向量为0。保留所有空心拍ECG，不丢弃或重分配记录。', '',
      'z_e/z_q Mean为64维，Mean+Max128维，Attention Pooling64维。HeartLang CLS与token Mean为768维，token Mean+Max1536维。HeartLang token Mean/Max使用冻结同一backbone最后层LayerNorm后的非CLS有效token。这两组与CLS是更直接的pooling对照；跨z_e/z_q/HeartLang比较还包含编码层差异。', '',
      '所有头：seed0、30epoch、batch64、每轮103步、train shuffle/drop-last、val/test完整评价；AdamW lr5e-4/min1e-5/warmup起始1e-6/5轮cosine，weight_decay0.05，clip3。linear为D→4普通MSE；MLP为D→256→GELU→Dropout0.1→4，使用训练标签方差逆数归一化权重，目标仍是原四标签one-hot回归。没有新增分类loss、监督或疾病码。所有方法不做feature standardization。', '',
      '复用extract_information_flow_representations的mask/Mean/Max与load_heartlang；复用已有AttentionPoolingHead的D→64→Tanh→1打分＋mask softmax；复用既有probe的学习率与回归指标。Attention比固定池化增加可训练参数，输入维数不同也导致head参数量不同，架构和预算相同不等于参数总数严格相等。缓存特征在float32下训练，随机顺序用独立seed发生器；新的匹配CLS成绩可能与E3旧入口因数值精度、随机消耗及学习率实现略有差别，不能将差异认定为pooling收益。', '',
      '## 3. 全部下游测试指标','',
      '患者前三输出平均R²分别报告：①全体test上OMI/NOMI/Control三维平均；②只在患者test子集上重新计算三输出R²。AUC只在OMI/NOMI样本，以预测OMI分数减NOMI分数排序。','']
    for kind in ('linear','mlp2'):
        rows=[]
        for r in results:
            if r['head']!=kind:continue
            t=r['test'];rows.append([r['method'],r['input_dim'],r['trainable_parameters'],r['best_epoch'],f"{r['val']['r2']:.6f}",f"{t['r2']:.6f}",
               *[f"{t['per_output_r2'][k]:.6f}" for k in LABELS],f"{t['patient_three_output_r2_on_all_ecg']:.6f}",f"{t['patient_subset_three_output_r2']:.6f}",f"{t['omi_vs_nomi_auc']:.6f}",f"{t['mse']:.6f}"])
        lines += [f'### {kind}','',table(['方法','输入维','参数','最佳轮','val平均R²','test平均R²',*LABELS,'前三输出均值','患者子集R²','OMI/NOMI AUC','MSE'],rows),'']
    lines += ['## 4. 五个研究问题：逐项对照','']
    for kind in ('linear','mlp2'):
        r={v['method']:v['test'] for v in results if v['head']==kind};cls=r['heartlang']
        lines += [f'### {kind}','', 'Mean/Mean+Max相对CLS的四输出ΔR²：'+', '.join(f"{m} {r[m]['r2']-cls['r2']:+.6f}" for m in METHODS[:4])+'。','',
          'VQ的同池化z_q−z_e差值：'+', '.join(f"{suffix or 'Mean'} ΔR² {r['z_q'+suffix]['r2']-r['z_e'+suffix]['r2']:+.6f}、ΔAUC {r['z_q'+suffix]['omi_vs_nomi_auc']-r['z_e'+suffix]['omi_vs_nomi_auc']:+.6f}" for suffix in ('','_mean_max','_attention'))+'。这些差值是特定probe的可解码信息变化，不是信息论上的不可逆损失证明。','',
          '同HeartLang层替换CLS：'+', '.join(f"{m} ΔR² {r[m]['r2']-cls['r2']:+.6f}、ΔAUC {r[m]['omi_vs_nomi_auc']-cls['omi_vs_nomi_auc']:+.6f}" for m in ('heartlang_mean','heartlang_mean_max'))+'。这是判断CLS聚合瓶颈的主要证据。','',
          'Attention相对同层Mean和CLS：'+', '.join(f"{m} 对Mean ΔR² {r[m]['r2']-r[m.replace('_attention','')]['r2']:+.6f}、对CLS ΔR² {r[m]['r2']-cls['r2']:+.6f}" for m in ('z_e_attention','z_q_attention'))+'。']
    mlp={r['method']:r for r in results if r['head']=='mlp2'}
    chosen=max(mlp.values(),key=lambda r:r['val']['r2'])
    cls=mlp['heartlang'];context=max((mlp[m] for m in ('heartlang_mean','heartlang_mean_max')),key=lambda r:r['val']['r2'])
    lines += ['', '## 5. 下一步优先级','',
       f"按主MLP验证协议推荐{chosen['method']}，相对CLS测试ΔR²={chosen['test']['r2']-cls['test']['r2']:+.6f}。同HeartLang层的验证较优替代{context['method']}，测试ΔR²={context['test']['r2']-cls['test']['r2']:+.6f}。"]
    if context['test']['r2']>cls['test']['r2']:
        lines += ['本次存在同backbone替换池化的提升信号，下一步优先重复验证pooling及患者级不确定性；不应立即改tokenizer/codebook。']
    else:
        lines += ['本次同backbone池化替换未提高测试主R²，不支持把CLS认定为主要瓶颈。下一步先补充多种子及训练收敛诊断；仅凭本次结果也不能推导必须继续改tokenizer/codebook。若静态表示患者AUC更高，可考虑针对患者亚型的表示融合，但属于尚未运行的新实验。']
    lines += ['', '## 6. Attention浓度与异常心拍解释边界','',
       '以下仅说明权重是否集中；没有异常心拍人工标注或定位真值，因此不能证明注意力找到或保留了少量异常心拍。没有引入新增监督。','']
    lines += [table(['方法','头','split','top1 mass','top5 mass','有效心拍比例','归一化熵'],[[r['method'],r['head'],s,*[f'{v:.6f}' for v in vals.values()]] for r in results if 'attention_concentration' in r for s,vals in r['attention_concentration'].items()]),'',
       '## 7. 历史E3 CLS与审计证据','',
       '历史E3 CLS：linear .224613、MLP .230181。主对照使用本轮重新训练的匹配CLS；历史数字只作入口复现参考，不能跨协议计算pooling增益。', '',
       '单seed，无置信区间或显著性检验；项目历史多次查看test，不能称全项目第一次盲测。各表示尺度/维数不同，统一有限轮数优化可能不足以达到各方法最佳性能，不能据此证明表示中完全没有信息。', '',
       f'实验目录：`{RUN}`；配置protocol.json，固定split审计frozen_split_audit.json，提取审计extraction_audit.json，checkpoint前后哈希checkpoint_integrity.json，原始comparison.json及全部指标明细.csv；每方法保存checkpoint-best、val history和最终test_predictions。', '']
    (RUN/'E4完整中文对比报告.md').write_text('\n'.join(lines))
    with (RUN/'全部指标明细.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','head','split','metric','value'])
        for r in results:
            for split in ('val','test'):
                for key,value in r[split].items():
                    if isinstance(value,dict):
                        for label,v in value.items():writer.writerow([r['method'],r['head'],split,key+'.'+label,v])
                    else:writer.writerow([r['method'],r['head'],split,key,value])


def smoke(device):
    tokens=torch.randn(3,8,64,device=device)
    valid=torch.tensor([[1,1,0,0,0,0,0,0],[1]*8,[0]*8],device=device,dtype=torch.bool)
    m,mm=pooled(tokens,valid)
    assert torch.allclose(m[0],tokens[0,:2].mean(0)) and torch.equal(m[2],torch.zeros_like(m[2]))
    assert torch.equal(mm[2],torch.zeros_like(mm[2]))
    att=AttentionPoolingHead().to(device).eval()
    a,w=att(tokens,valid,True)
    changed=tokens.clone();changed[~valid]=100000
    b,wb=att(changed,valid,True)
    assert torch.allclose(a,b) and torch.equal(w[~valid],torch.zeros_like(w[~valid]))
    assert torch.isfinite(a).all() and torch.allclose(w.sum(1),valid.any(1).float())
    perm=torch.randperm(8,device=device)
    assert torch.allclose(a,att(tokens[:,perm],valid[:,perm]),atol=1e-6)
    dump(RUN/'preflight_checks.json',{'masked_mean_max':'passed','empty_ecg_zero_pool':'passed',
         'attention_ignores_invalid':'passed','attention_permutation_invariant':'passed','no_real_test_in_smoke':True})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke_only',action='store_true');args=parser.parse_args()
    device=torch.device('cuda');RUN.mkdir(parents=True,exist_ok=False)
    (RUN/'pipeline.pid').write_text(str(__import__('os').getpid()))
    config={'methods_predefined':METHODS,'heads':['linear','mlp2'],'epochs':30,'seed':0,'batch_size':64,
       'lr':5e-4,'min_lr':1e-5,'warmup_lr':1e-6,'warmup_epochs':5,'weight_decay':.05,
       'tokenizer_checkpoint':str(TOKENIZER),'mlm_checkpoint':str(MLM),
       'selection':'validation four-output uniform R2 only','feature_standardization':False,
       'loss':'linear mse; mlp train-variance-normalized mse','no_tokenizer_or_backbone_updates':True}
    dump(RUN/'protocol.json',config)
    try:
        smoke(device)
        if args.smoke_only:return
        import subprocess
        with (RUN/'frozen_split_audit.json').open('w') as f:
            subprocess.run([sys.executable,str(ROOT/'scripts/curriculum_v1/verify_frozen_split.py')],stdout=f,check=True,cwd=ROOT)
        original={str(p):sha(p) for p in (TOKENIZER,MLM)}
        status('extracting_frozen_features');extract(device)
        labels={s:torch.tensor(np.load(FEATURES/f'{s}_labels.npy'),device=device,dtype=torch.float32) for s in ('train','val','test')}
        assert [len(labels[s]) for s in labels]==[6640,1344,1366]
        settings=argparse.Namespace(**{k:config[k] for k in ('lr','min_lr','warmup_lr','warmup_epochs','weight_decay')})
        results=[]
        for method in METHODS:
            arrays,masks=load_arrays(method,device)
            for kind in ('linear','mlp2'):
                status('training_'+method+'_'+kind)
                results.append(train(method,kind,arrays,masks,labels,settings,device))
                dump(RUN/'partial_results.json',results)
            del arrays,masks;torch.cuda.empty_cache()
        after={str(p):sha(p) for p in (TOKENIZER,MLM)};assert original==after
        dump(RUN/'checkpoint_integrity.json',{'before':original,'after':after,'unchanged':True})
        for r in results:
            assert r['test']['n_records']==1366
            assert abs(sum(r['test']['per_output_r2'].values())/4-r['test']['r2'])<1e-8
        status('writing_report');report(results,config)
        status('complete');(RUN/'COMPLETE').touch();(RUN/'exit_code.txt').write_text('0')
    except BaseException:
        status('failed');(RUN/'exit_code.txt').write_text('1');raise


if __name__=='__main__':main()
