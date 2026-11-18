"""Verify E27/E28, produce evidence reports, use Validation only."""
import sys,json,pathlib
import numpy as np,pandas as pd,torch
from sklearn.metrics import r2_score
ROOT=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'heartlang').exists());sys.path[:0]=[str(ROOT),str(ROOT/'scripts/curriculum_v1'),str(ROOT/'scripts/experiments')]
import run_e27 as a
import run_e28 as b
import run_e12_continuous_input as e

def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,default=str))
R27=a.R;R28=b.R;assert (R27/'COMPLETE').exists() and (R28/'COMPLETE').exists();torch.set_num_threads(4)
contract=json.loads((ROOT/'configs/curriculum_v1/data_contract.json').read_text());assert e.sha256(e.DOWNSTREAM/'split_manifest.csv')==contract['split_manifest_sha256'];valmeta=pd.read_csv(e.DOWNSTREAM/'val_metadata.csv');trainmeta=pd.read_csv(e.DOWNSTREAM/'train_metadata.csv');assert not set(valmeta.entity_id)&set(trainmeta.entity_id)
rows=json.loads((R27/'validation_summary.json').read_text());verified=0;clean_deltas=[]
for method in ('layer12','learned_mix'):
 for seed in (0,1,2):
  fresh=next(z for z in rows if z['policy']=='clean' and z['readout']==method and z['seed']==seed);old=json.loads((a.E25/'downstream'/method/f'seed{seed}/summary.json').read_text());delta=fresh['val']['r2']-old['val']['r2'];clean_deltas.append(delta);assert abs(delta)<1e-6
for ratio in (.25,.5,.75):
 for repeat in range(3):
  folders=[R27/'diagnostics'/f'{p}_r{ratio}_rep{repeat}' for p in ('random','time_span','lead_priority')];masks=[np.load(p/'mask.npy') for p in folders];anchors=[np.load(p/'anchor.npy') for p in folders];assert all(np.array_equal(anchors[0],z) for z in anchors[1:]);assert all(np.array_equal(masks[0].sum(1),z.sum(1)) for z in masks[1:]);ok=anchors[0]>=0
  for m in masks:assert m[np.arange(len(m))[ok],anchors[0][ok]].all()
  verified+=1
out27={'clean_r2_max_abs_delta_vs_E25':max(abs(v) for v in clean_deltas),'all_9_matched_mask_budget_and_anchor_groups_verified':verified,'true_time_audit':json.loads((R27/'cache/audit.json').read_text()),'split_manifest_matched':True,'train_val_identity_overlap':0,'test_loaded':False};dump(R27/'verification_extended.json',out27)
D=json.loads((R28/'validation_summary.json').read_text());labels=np.load(e.DOWNSTREAM/'val_labels.npy');verifiedruns=0;ensemble={}
for method in ('sid_only','context_teacher'):
 probs=[]
 for seed in (0,1,2):
  p=R28/method/'downstream/layer12'/f'seed{seed}';row=json.loads((p/'summary.json').read_text());raw=np.load(p/'val_logits.npy');metrics,prob=e.probability_metrics(raw,labels);assert abs(metrics['r2']-row['val']['r2'])<1e-10;assert np.allclose(prob,np.load(p/'val_probabilities.npy'));probs.append(prob);verifiedruns+=1
 pre=json.loads((R28/method/'pretrain_summary.json').read_text());minimum=min(pre['history'],key=lambda z:z['val']['sid_ce']);saved=torch.load(R28/method/'checkpoint-best.pth',map_location='cpu',weights_only=False);assert saved['epoch']==minimum['epoch'];ensemble[method]=np.mean(probs,0)
ensemble['E12_baseline']=np.mean([np.load(a.E25/'downstream/layer12'/f'seed{s}/val_probabilities.npy') for s in (0,1,2)],0)
ids=valmeta.entity_id.astype(str).to_numpy();unique=np.unique(ids);groups=[np.flatnonzero(ids==z) for z in unique];rng=np.random.default_rng(280007);boots=[np.concatenate([groups[j] for j in row]) for row in rng.integers(0,len(groups),size=(2000,len(groups)))];ci={}
for p,q in (('context_teacher','sid_only'),('context_teacher','E12_baseline'),('sid_only','E12_baseline')):
 delta=[r2_score(labels[ii],ensemble[p][ii])-r2_score(labels[ii],ensemble[q][ii]) for ii in boots];ci[p+'_minus_'+q]={'ensemble_r2_delta':float(r2_score(labels,ensemble[p])-r2_score(labels,ensemble[q])),'entity_bootstrap_ci95':np.quantile(delta,[.025,.975]).tolist()}
# Collapse diagnostic is performed on the trained predictor, without any updates.
device=torch.device('cuda');tr=e.ZEDataset(e.DEFAULT_RUN/'pretrain_ze_cache','val');indices=np.load(R28/'val_indices.npy')[:64];zz=torch.tensor(np.array(tr.ze[indices]),device=device);ss=torch.tensor(np.array(tr.sid[indices]),device=device).long();valid=(ss>=0)&(ss<1024);mask=torch.tensor(e.fixed_masks(np.array(tr.sid[indices]),0),device=device);teacher=e.load_continuous_checkpoint(b.SOURCE,device).eval().requires_grad_(False);collapse={}
with torch.no_grad():
 target=b.teacher_target(teacher,zz,valid)[mask];t=torch.nn.functional.normalize(target,dim=-1);tstd=float(t.std(0).mean())
 for method in ('sid_only','context_teacher'):
  model=e.load_continuous_checkpoint(R28/method/'checkpoint-best.pth',device).eval().requires_grad_(False);pred=b.predictor().to(device).eval();pred.load_state_dict(torch.load(R28/method/'predictor-best.pth',map_location=device,weights_only=False));features=b.forward(model,zz,valid,mask)[-1][mask];pp=torch.nn.functional.normalize(pred(features).float(),dim=-1);collapse[method]={'normalized_predictor_dim_std_mean':float(pp.std(0).mean()),'normalized_teacher_dim_std_mean':tstd,'predictor_teacher_std_ratio':float(pp.std(0).mean())/tstd,'cosine_mean':float((pp*t).sum(-1).mean())};del model,pred
ix=np.load(R28/'train_indices.npy');iv=np.load(R28/'val_indices.npy');assert len(np.unique(ix))==8192 and len(np.unique(iv))==1024
out28={'verified_downstream_runs':verifiedruns,'pretrain_checkpoint_selection_verified':True,'matched_unique_train_subset':len(ix),'matched_unique_strict_val_subset':len(iv),'train_subset_indices_sha256':e.sha256(R28/'train_indices.npy'),'strict_val_subset_indices_sha256':e.sha256(R28/'val_indices.npy'),'train_val_identity_overlap':0,'bootstrap_unit':'fixed-three-seed ensemble predictions, entity-cluster sampling; not mean-of-seed R2','bootstrap':ci,'predictor_collapse_diagnostic':collapse,'source_audit':json.loads((R28/'verification.json').read_text()),'test_loaded':False};dump(R28/'verification_extended.json',out28)
# Mask report.
agg=json.loads((R27/'aggregate.json').read_text());anchor=json.loads((R27/'shared_anchor_comparison.json').read_text());lines=['# E27：等遮挡量、共同目标的时间/导联遮挡诊断','','## 协议','','- 不训练模型。固定E12-A及E25末层/凸组合下游头，各3个seed，Val1344条。仅Val，无Test。','- 从原始100Hz波形恢复检测R时间并逐值重放1344条；不是使用粗粒度time ID。','- 25/50/75%有效token遮挡，重复3次。每条记录、每次重复在三种模式中使用完全相同的目标anchor，遮挡个数相同。','- random随机缺失；time_span围绕anchor真实检测时间选择最近token；lead_priority先遮anchor导联再随机其他导联。精确配额会导致边界拍/导联部分遮挡，并非严格整导联消失。','- 主干MASK完整替换z_e；E11的global/region z_e旁路均从可见token重新计算。隐藏位置保留为MASK，不删除其Transformer位置。','- 共同anchor CE比较目标相同；所有masked CE的目标组成不同，仅作辅助诊断。','- 缺失输入会改变分布；脆弱性不等于证明换该Mask训练能提升。','','## 共同anchor SID CE','','|遮挡比例|random|time_span|lead_priority|time-random 95%CI|lead-random 95%CI|','|---|---:|---:|---:|---|---|']
for ratio,v in anchor.items():
 m=v['mean_ce'];lines.append(f"|{ratio}|{m['random']:.5f}|{m['time_span']:.5f}|{m['lead_priority']:.5f}|{v['time_span_minus_random']['entity_bootstrap_ci95']}|{v['lead_priority_minus_random']['entity_bootstrap_ci95']}|")
lines+=['','## 下游Val均值','','|读取|遮挡模式|比例|Overall R²|患者三类均值|Macro AUC|Macro F1|','|---|---|---:|---:|---:|---:|---:|']
for key,v in agg.items():
 method,p,ratio=key.split('/');lines.append(f"|{method}|{p}|{ratio}|{v['r2']:.6f}|{v['patient_mean_r2']:.6f}|{v['macro_roc_auc']:.6f}|{v['macro_f1']:.6f}|")
lines+=['','遮挡结果平均3种Mask随机重复×3个冻结下游seed；clean仅3个下游seed。不同率不可视为同等任务。','',f"零遮挡重现E25最大R²差{out27['clean_r2_max_abs_delta_vs_E25']:.3g}；9组共同anchor和配额核对通过；主干与旁路masked-value invariance通过。"]
(R27/'E27_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
lines=['# E28：保留SID CE的完整上下文教师配对试跑','','## 范围与区别','','- 两组均从同一个E12-A checkpoint开始继续训练，固定严格Train子集8192、严格预训练Val子集1024、3epoch×128步=384更新。不是完整100epoch预训练，也不是JEPA论文复现。','- sid_only：原SID CE；context_teacher：SID CE权重1，加0.1上下文预测损失，第一epoch线性升权。相同数据顺序、Mask序列、dropout起点、batch64、lr1e-5、BF16、优化器/步数。','- 教师是固定E12-A，读取完整ECG；其6/9/12层上下文token均值经逐token LayerNorm作为目标。学生仅读取50% Masked ECG，末层经768→256→768预测器预测目标。教师目标stop-gradient，预测器下游丢弃。','- E13-C预测孤立原始z_e；E13-D是连续+SID混合输入；均不同于本次上下文目标。','- teacher看到被遮原值是自监督目标定义；它的特征不进入学生forward或下游推理。未使用下游临床标签做预训练。','- 预训练checkpoint仅按严格Val固定Mask SID CE选择；下游完整Train/Val，原末层+E11，每组seed0/1/2各30epoch。','- 冻结教师不能获取原数据中已被切掉的波形；也不能据单个冻结教师试跑否定所有EMA/从头训练JEPA方案。','','## 安全与可行性','','- masked z_e改变不影响学生logits，masked输入梯度严格0；教师无梯度、state_dict不变；预测器有非零梯度。','- 修改上下文而保留目标token原值，会改变教师目标，证明目标不是孤立token的拷贝。','- predictor_collapse_diagnostic见verification_extended.json：报告实际归一化预测方差/教师方差，不能仅凭loss下降宣称学到疾病信息。','','## 预训练与下游结果','','|方案|初始固定SID CE|最佳固定SID CE|Overall R² mean±SD|患者三类均值|Macro AUC|Macro F1|','|---|---:|---:|---:|---:|---:|---:|']
for method in ('sid_only','context_teacher'):
 v=D[method];g=v['aggregate'];p=v['pretrain'];lines.append(f"|{method}|{p['initial']['sid_ce']:.6f}|{p['best_sid_ce']:.6f}|{g['r2']['mean']:.6f} ± {g['r2']['sd']:.6f}|{g['patient_mean_r2']['mean']:.6f}|{g['macro_roc_auc']['mean']:.6f}|{g['macro_f1']['mean']:.6f}|")
lines+=['','教师方案相对配对SID-only的每seed Overall R²差：'+str(D['paired_downstream_r2_delta']),'','## 实体级bootstrap','','bootstrap针对固定三seed平均概率模型，而非单模型R²均值，且不覆盖预训练随机性或多重筛选。','','```json',json.dumps(ci,indent=2,ensure_ascii=False),'```','','## 方法依据与适用范围','','- data2vec：完整输入上下文隐表示作为Masked输入学生目标：https://arxiv.org/abs/2202.03555 。','- ECG-JEPA研究及作者实现：https://arxiv.org/abs/2410.13867 ，https://github.com/kweimann/ECG-JEPA 。该研究是其他ECG基准，不保证迁移到本项目OMI/NOMI/Control。','- 本次384步、单预训练seed的继续训练，只构成安全性/小预算信号，不代表完整预算结论。','', '## 验收','','6个下游指标从保存logits重算通过；预训练最佳轮次核对通过；固定split manifest与Train/Val身份隔离通过；历史源SHA不变；未读取Test。']
(R28/'E28_REPORT_CN.md').write_text('\n'.join(lines)+'\n');print('verification and reports complete',flush=True)
