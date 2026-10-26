#!/usr/bin/env python3
"""E5: preregistered zero-parameter compression; reuse completed E4 anchors."""
import argparse, csv, json, math, shutil
from pathlib import Path
import numpy as np
import torch
import run_e4_aggregation as e4
ROOT=e4.ROOT
SOURCE=ROOT/'results/curriculum_v1/e4_e3_aggregation_20261011'
RUN=ROOT/'results/curriculum_v1/e5_e3_pooling_fair_20261011'
METHODS=['CLS','Mean','Max','MeanMax_raw','MeanMax_sum768','MeanMax_orthogonal768']
def main():
    RUN.mkdir(parents=True,exist_ok=False);e4.RUN=RUN
    args=argparse.Namespace(lr=5e-4,min_lr=1e-5,warmup_lr=1e-6,weight_decay=.05)
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    config={'methods_predefined':METHODS,'head':'768→256→GELU→Dropout(0.1)→4; raw reference input1536',
      'epochs':30,'seed':0,'batch_size':64,'optimizer':'AdamW','lr':args.lr,'min_lr':args.min_lr,
      'warmup_lr':args.warmup_lr,'warmup_epochs':5,'weight_decay':args.weight_decay,'clip':3,
      'selection':'validation four-output uniform-average R2 only','projection_seed':20261011,
      'projections':'sum [I;I]/sqrt(2); Gaussian QR orthonormal columns, no labels/no fitting',
      'standardization':False,'reused_anchors':'completed E4 CLS, Mean, raw MeanMax; no repeated test',
      'frozen_tokenizer':str(e4.TOKENIZER),'frozen_heartlang':str(e4.MLM),'source_protocol':protocol}
    e4.dump(RUN/'protocol.json',config);e4.status('preflight')
    files=[e4.TOKENIZER,e4.MLM]+[e4.FEATURES/f'{s}_{n}.npy' for s in ('train','val','test') for n in ('heartlang','heartlang_mean','heartlang_mean_max','labels')]
    before={str(p):e4.sha(p) for p in files}
    e4.dump(RUN/'source_fingerprints.json',before)
    device=torch.device('cuda');torch.set_num_threads(4)
    labels={s:torch.tensor(np.load(e4.FEATURES/f'{s}_labels.npy'),device=device,dtype=torch.float32) for s in ('train','val','test')}
    mm={s:np.load(e4.FEATURES/f'{s}_heartlang_mean_max.npy') for s in labels}
    for s in labels:
        assert mm[s].shape==(len(labels[s]),1536)
        assert np.array_equal(mm[s][:,:768],np.load(e4.FEATURES/f'{s}_heartlang_mean.npy'))
    rng=np.random.default_rng(20261011)
    q=np.linalg.qr(rng.normal(size=(1536,768)),mode='reduced')[0].astype(np.float32)
    assert np.allclose(q.T@q,np.eye(768),atol=2e-6)
    np.save(RUN/'fixed_orthogonal_projection.npy',q)
    results=[]
    for method,old in [('CLS','heartlang'),('Mean','heartlang_mean'),('MeanMax_raw','heartlang_mean_max')]:
        src=SOURCE/old/'mlp2';dst=RUN/method/'mlp2';shutil.copytree(src,dst)
        r=json.loads((dst/'summary.json').read_text());r.update(method=method,reused_from=str(src),projection_trainable_parameters=0)
        results.append(r)
    for method in ('Max','MeanMax_sum768','MeanMax_orthogonal768'):
        e4.status('training_'+method)
        values={s: mm[s][:,768:] if method=='Max' else ((mm[s][:,:768]+mm[s][:,768:])/math.sqrt(2) if method=='MeanMax_sum768' else mm[s]@q) for s in labels}
        arrays={s:torch.tensor(v,device=device,dtype=torch.float32) for s,v in values.items()}
        r=e4.train(method,'mlp2',arrays,None,labels,args,device)
        r['projection_trainable_parameters']=0;results.append(r)
    results.sort(key=lambda r:METHODS.index(r['method']))
    audit={'checkpoint_and_cache_hashes_unchanged':all(e4.sha(p)==before[str(p)] for p in files),'split_counts':{s:len(y) for s,y in labels.items()},'checks':[]}
    for r in results:
        pred=np.load(RUN/r['method']/'mlp2/test_predictions.npy');computed=e4.metrics(pred,labels['test'].cpu().numpy())
        assert abs(computed['r2']-r['test']['r2'])<1e-10
        assert abs(np.mean(list(r['test']['per_output_r2'].values()))-r['test']['r2'])<1e-10
        assert r['test']['n_records']==1366
        assert r['trainable_parameters']==(394500 if r['method']=='MeanMax_raw' else 197892)
        hist=json.loads((RUN/r['method']/'mlp2/history.json').read_text())
        chosen=max(hist,key=lambda h:h['val']['r2']);assert chosen['epoch']==r['best_epoch']
        audit['checks'].append({'method':r['method'],'metrics_recomputed':True,'val_selection_verified':True})
    assert audit['checkpoint_and_cache_hashes_unchanged']
    e4.dump(RUN/'report_validation.json',audit)
    e4.dump(RUN/'comparison.json',{'config':config,'results':results})
    headers=['方法','输入维度','可训练参数','最佳epoch','val平均R2','test平均R2',*e4.LABELS,'前三输出平均R2','患者子集三输出R2','OMI_NOMI_AUC','MSE']
    rows=[]
    for r in results:
        t=r['test'];rows.append([r['method'],r['input_dim'],r['trainable_parameters'],r['best_epoch'],r['val']['r2'],t['r2'],*[t['per_output_r2'][k] for k in e4.LABELS],t['patient_three_output_r2_on_all_ecg'],t['patient_subset_three_output_r2'],t['omi_vs_nomi_auc'],t['mse']])
    with (RUN/'全部指标明细.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
    cls=results[0];fair=[r for r in results if r['method']!='MeanMax_raw'];best=max(fair,key=lambda r:r['val']['r2'])
    delta=best['test']['r2']-cls['test']['r2']
    lines=['# E5：HeartLang Pooling 参数公平验证','',f"按验证集选出的公平条件最优方法为 **{best['method']}**。test 四输出平均 R² **{best['test']['r2']:.6f}**，相对 CLS **{delta:+.6f}**；两者可训练参数均为 **197,892**。",'',
      '固定 E3 tokenizer 与 HeartLang，直接复用 E4 最后层有效非 CLS token 的 Mean/Mean+Max 缓存。Max 为 Mean+Max 后768维，未重新推理或训练预训练模型。原始 split 为6640/1344/1366，所有样本保留。CLS、Mean、原始 Mean+Max 直接复用 E4 完成的同协议 checkpoint 与正式评价，未重复评价 test。','',
      '新增 Max、两种固定投影在训练前指定：对应坐标相加除以√2；固定随机种子20261011生成1536×768正交列矩阵。两者输出768维且没有可训练参数、没有标签拟合；同一投影用于三份split。固定投影矩阵的存储不计为可训练参数。','',
      '全部采用同一 MLP：D→256→GELU→Dropout0.1→4；seed0、30epoch、batch64、AdamW lr5e-4、5epoch warmup、cosine最低1e-5、weight_decay0.05、clip3。沿用 E4 按训练标签方差逆数归一化的四输出MSE，不做特征标准化。相同768维方法使用相同初始化与shuffle协议。checkpoint仅按val四输出等权R²选择，新方案test各评价一次。','',
      '原始1536维head为394,500参数，比公平组多196,608（约99.35%）。未加入可训练密集投影：其额外参数为1,180,416，总计1,378,308，无法匹配CLS预算。','',
      e4.table(headers,[[f'{x:.6f}' if isinstance(x,float) else x for x in row] for row in rows]),'',
      '前三输出平均R²在全部test计算OMI/NOMI/Control均值；患者子集指标只在Healthy=0样本重新计算三输出R²。AUC仅对OMI/NOMI，以预测OMI减NOMI排序。','',
      '## 结论','']
    for r in fair[1:]:
        lines.append(f"- {r['method']} 相对CLS：test平均R² {r['test']['r2']-cls['test']['r2']:+.6f}，AUC {r['test']['omi_vs_nomi_auc']-cls['test']['omi_vs_nomi_auc']:+.6f}。")
    lines += ['', '固定投影对比隔离了下游可训练参数数量，但压缩本身可能损失信息；若某种投影失败，不能证明全部增益都来自参数。Mean、Max对照分别检验平均信息与极值信息，原始拼接同时保留两者。所有结论来自单seed，差值是描述性结果，不是统计显著性检验。','',
      '源码、缓存、checkpoint指纹与指标重算见 source_fingerprints.json、report_validation.json；主R²为四输出均值，六组n_test均1366，全部最佳epoch由val核验。']
    (RUN/'E5简洁中文报告.md').write_text('\n'.join(lines),encoding='utf-8')
    e4.status('complete');(RUN/'exit_code.txt').write_text('0');(RUN/'COMPLETE').touch()
if __name__=='__main__':main()
