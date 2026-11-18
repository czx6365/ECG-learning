"""Recompute fixed predictions and patient/entity-cluster paired intervals."""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,r2_score,f1_score
from run_e34 import OUT,e,s,b,WINDOWS
KEYS=['overall','clinical3','omi_nomi_auc','macro_auc','macro_f1','OMI','NOMI','Control','Healthy','patient_raw3_r2','patient_conditional3_r2']
class Scorer:
 def __init__(self,y,q):
  self.y=y;self.q=q;self.target=y.argmax(1);self.error=(y-q)**2;self.pred=q.argmax(1)
  self.pair=np.flatnonzero(self.target<2);self.patient=np.flatnonzero(self.target<3)
  self.patient_error=(y[self.patient,:3]-q[self.patient,:3])**2
  self.patient_cond_error=(y[self.patient,:3]-q[self.patient,:3]/q[self.patient,:3].sum(1,keepdims=True))**2
  self.auc=[]
  for ids,labels,score in [(self.pair,y[self.pair,0],q[self.pair,0]/q[self.pair,:2].sum(1))]+[(np.arange(len(y)),y[:,j],q[:,j]) for j in range(4)]:
   order=np.argsort(score,kind='stable');v=score[order];starts=np.r_[0,np.flatnonzero(np.diff(v))+1]
   self.auc.append((ids[order],labels[order],starts))
 def score(self,w):
  count=w.sum();positive=w@self.y;den=positive-positive**2/count
  rs=1-(w@self.error)/den;au=[]
  for ids,labels,starts in self.auc:
   pos=np.add.reduceat(w[ids]*labels,starts);neg=np.add.reduceat(w[ids]*(1-labels),starts)
   au.append(float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum())))
  confusion=np.bincount(self.target*4+self.pred,weights=w,minlength=16).reshape(4,4)
  f1=np.mean(2*np.diag(confusion)/(confusion.sum(0)+confusion.sum(1)))
  wp=w[self.patient];yp=wp@self.y[self.patient,:3];denp=yp-yp**2/wp.sum()
  pr=1-(wp@self.patient_error)/denp;pc=1-(wp@self.patient_cond_error)/denp
  return np.r_[rs.mean(),rs[:3].mean(),au[0],np.mean(au[1:]),f1,rs,pr.mean(),pc.mean()]
def paired(y,probs,groups,draws=2000):
 scorers={m:Scorer(y,q) for m,q in probs.items()};ones=np.ones(len(y))
 point={m:z.score(ones) for m,z in scorers.items()}
 # Check custom weighted implementation against standard metrics before resampling
 for m,z in scorers.items():
  ref=s.metrics(y,probs[m]);np.testing.assert_allclose(point[m],[ref[k] for k in KEYS],atol=2e-10,rtol=1e-9)
 _,inv=np.unique(groups,return_inverse=True);g=inv.max()+1;rng=np.random.default_rng(34001)
 samples={m:np.empty((draws,len(KEYS))) for m in scorers}
 for j in range(draws):
  mult=np.bincount(rng.integers(g,size=g),minlength=g);weight=mult[inv]
  for m,z in scorers.items():samples[m][j]=z.score(weight)
 contrasts={}
 for method in ('B','C','D','E'):
  for ref in (('A',) if method=='E' else ('A','E')):
   delta=point[method]-point[ref];ci=np.quantile(samples[method]-samples[ref],[.025,.975],axis=0)
   contrasts[f'{method}_minus_{ref}']={k:dict(delta=float(d),ci95=[float(lo),float(hi)]) for k,d,lo,hi in zip(KEYS,delta,*ci)}
 return contrasts

def main():
 assert (OUT/'status.txt').read_text().strip() in ('training_complete','complete')
 started=time.time();meta=pd.read_csv(e.DOWNSTREAM/'train_metadata.csv');y=np.load(e.DOWNSTREAM/'train_labels.npy');folds=pd.read_csv(OUT/'fold_assignment.csv')
 probabilities={'A':np.load(OUT/'oof_A.npy')}
 result=json.load(open(OUT/'metrics.json'))
 for m in ('B','C','D','E'):
  seeds=np.load(OUT/f'oof_{m}_seeds.npy');assert np.isfinite(seeds).all();np.testing.assert_allclose(seeds.sum(-1),1,atol=2e-6)
  probabilities[m]=seeds.mean(0)
  for seed in range(3):
   ref=s.metrics(y,seeds[seed]);np.testing.assert_allclose([ref[k] for k in KEYS],[result[m]['per_seed'][seed][k] for k in KEYS],atol=1e-12)
 np.testing.assert_allclose(probabilities['A'],np.load(s.OUT/'oof_shared.npy').mean(0),atol=0,rtol=0)
 recomputed={m:s.metrics(y,q) for m,q in probabilities.items()};b.dump(OUT/'ensemble_metrics.json',recomputed)
 bootfile=OUT/'paired_bootstrap.json'
 if bootfile.exists():boot=json.load(open(bootfile))
 else:
  boot=paired(y,probabilities,meta.entity_id.to_numpy());b.dump(bootfile,boot)
 fold_metrics=[];fold_boot={}
 for fold in range(3):
  ix=folds[(folds.fold==fold)&(folds.role=='outer')].row.to_numpy()
  for m,q in probabilities.items():fold_metrics.append(dict(fold=fold,method=m,**s.metrics(y[ix],q[ix])))
  file=OUT/f'fold{fold}_paired_bootstrap.json'
  if file.exists():fold_boot[str(fold)]=json.load(open(file))
  else:
   fold_boot[str(fold)]=paired(y[ix],{m:q[ix] for m,q in probabilities.items()},meta.iloc[ix].entity_id.to_numpy());b.dump(file,fold_boot[str(fold)])
  print('reported fold',fold,flush=True)
 pd.DataFrame(fold_metrics).to_csv(OUT/'fold_ensemble_metrics.csv',index=False)
 rows=[];truth=y.argmax(1);a=probabilities['A'].argmax(1)
 for m in ('B','C','D','E'):
  q=probabilities[m];pred=q.argmax(1)
  for i in np.flatnonzero(pred!=a):rows.append(dict(anonymous_id=f'T{i:05d}',method=m,true_class=int(truth[i]),A_prediction=int(a[i]),new_prediction=int(pred[i]),corrected=bool(a[i]!=truth[i] and pred[i]==truth[i]),regressed=bool(a[i]==truth[i] and pred[i]!=truth[i]),A_confidence=float(probabilities['A'][i].max()),new_confidence=float(q[i].max())))
 pd.DataFrame(rows).to_csv(OUT/'error_transitions.csv',index=False)
 quality=pd.read_csv(OUT/'r_peak_quality.csv');subsets={}
 for name,mask in {'patients':truth<3,'healthy':truth==3,'usable_local_wave':quality.usable.to_numpy()>0,'missing_local_wave':quality.usable.to_numpy()==0}.items():
  ix=np.flatnonzero(mask)
  subsets[name]=dict(records=len(ix),accuracy={m:float(np.mean(q[ix].argmax(1)==truth[ix])) if len(ix) else None for m,q in probabilities.items()},brier={m:float(np.square(q[ix]-y[ix]).sum(1).mean()) if len(ix) else None for m,q in probabilities.items()})
 b.dump(OUT/'clinical_and_coverage_subsets.json',subsets)
 # Mandatory stop rule: clinical non-degradation and direct AUC improvement in every fixed outer fold.
 verdict={}
 for m in ('B','C','D'):
  diffs={k:[next(z[k] for z in fold_metrics if z['fold']==f and z['method']==m)-next(z[k] for z in fold_metrics if z['fold']==f and z['method']=='A') for f in range(3)] for k in ('overall','clinical3','omi_nomi_auc')}
  stable=all(z>0 for z in diffs['omi_nomi_auc']) and all(z>=0 for z in diffs['clinical3']) and boot[m+'_minus_A']['omi_nomi_auc']['ci95'][0]>0 and boot[m+'_minus_E']['omi_nomi_auc']['ci95'][0]>0
  verdict[m]=dict(per_fold_deltas=diffs,stable_clinical_increment=stable,decision='candidate_only_no_auto_followup' if stable else 'STOP_no_expansion_or_structure_search')
 b.dump(OUT/'decision.json',verdict)
 names={'A':'原E29 Ensemble3','B':'R峰附近/QRS代理','C':'R后ST-T代理','D':'扩展心拍','E':'无Raw容量对照'}
 lines=['# E34-Pilot：Raw ECG局部波形残差融合实验','','## 结论','',
  '本次固定四个新增模块对照均完成三折×三seed×30轮。保留原E29三seed集成，所有历史参数冻结；新增CNN每seed为4532参数。是否存在独立增量，按临床三类、每折直接OMI/NOMI AUC、患者实体bootstrap及容量对照综合判断。','',
  '|方法|Overall R²|Clinical-3|条件OMI/NOMI AUC|Macro AUC|Macro F1|','|---|---:|---:|---:|---:|---:|']
 for m,v in recomputed.items():lines.append(f"|{m} {names[m]}|"+'|'.join(f'{v[k]:.6f}' for k in KEYS[:5])+'|')
 lines+=['','## 与原E29及容量对照的配对变化','','2000次患者/健康实体成簇配对bootstrap，点估计按ECG记录计算，重复ECG随实体共同抽样。区间基于固定预测，不覆盖重新训练或预训练随机性。','',
  '|比较|Overall差 [95%CI]|Clinical-3差 [95%CI]|条件AUC差 [95%CI]|','|---|---|---|---|']
 for comparison,v in boot.items():lines.append(f'|{comparison}|'+ '|'.join(f"{v[k]['delta']:+.6f} [{v[k]['ci95'][0]:+.6f}, {v[k]['ci95'][1]:+.6f}]" for k in KEYS[:3])+'|')
 lines+=['','## 三折结果与停止条件','','|方法|折|Overall差|Clinical-3差|条件AUC差|','|---|---|---:|---:|---:|']
 for m,v in verdict.items():
  for f in range(3):lines.append(f'|{m}|{f}|'+ '|'.join(f"{v['per_fold_deltas'][k][f]:+.6f}" for k in KEYS[:3])+'|')

 lines+=['', '；'.join(f"{m}：{v['decision']}" for m,v in verdict.items())+'。']
 if not any(v['stable_clinical_increment'] for v in verdict.values()):lines+=['','**没有任何局部窗口满足稳定临床增量标准。保留原E29 Shared Ensemble3，不扩大CNN、不搜索更多结构，不启动正式多尺度模型训练。**']
 else:lines+=['','存在达到预定门槛的候选，但本次不自动启动正式多尺度模型或后续实验，仍需要独立确认。']
 lines+=['','## 波形及预处理审计','',
  '仅使用当前下游Train的6640条ECG（四类标签顺序OMI、NOMI、Control、Healthy）。所有行逐条核对source index/csv标识与E33-C缓存，概率、标签及fold顺序一致。患者与健康缓存均为9导联I、II、III、V1–V6、1000点；患者三类与健康各抽查原始CSV时间列为500Hz、10秒，对应100Hz缓存。健康原始时间检查只覆盖一条Train记录，不代表所有源档案均已验证。CSV没有可信单位声明；本实验只称源数值单位，不称mV。100Hz不是高分辨率。',
  '', '发现原缓存按来源使用不同global mean/std，因此先逆变换恢复源数值，再在每折outer-development拟合统一per-lead mean/std；B/C/D使用完全相同的统计。不做逐拍缩放、不减去每拍基线，保留相对幅值。来源设备/标定差异仍不能完全排除。新增统计没有使用outer、历史Validation或Test。',
  '', '|方法|R相对窗口|100Hz点数|说明|','|---|---|---:|---|','|B|−80至+120ms|20|QRS附近代理|','|C|+80至+400ms|32|R后代理，非J/ST/T临床标注|','|D|−300至+700ms|100|扩展覆盖，可能包括相邻心拍部分|',
  '', 'WFDB XQRS在导联I检测R峰；三组使用同一中心集合。必须完整覆盖扩展窗口，并要求下一R距当前至少480ms，避免C窗口直接覆盖下一QRS；至多16拍，超过则等间距取样。无有效拍的记录保留，零特征+mask输入，不能因覆盖差而删样本。该筛选可能低估快心率人群收益，覆盖和相邻拍重叠见r_peak_quality.csv。',
  '', '与E33-C的区别：E33-C输入整段1000点并训练Raw-only分类器；本次输入固定R相对局部窗口，跨拍共享CNN，mask后平均聚合，直接学习冻结E29概率的四分类残差。无全窗CNN重训，无Attention、新MLP或预训练。',
  '', '## 冻结与合法cross-fitting','',
  '保留E29-S的三个outer患者/健康实体组及原四类标签。原E29-S fit/select训练出的三seed模型用于outer预测，重算后与原OOF逐条对齐。E12-A、MLM、tokenizer和codebook仅通过已审计的冻结E29-L缓存提供特征，本次不重新实例化或训练。实际加载的E25/E11/Shared及18个既有E32 crossfit头全部只读、eval且requires_grad=False；没有重新训练任何原有头。完整历史checkpoint哈希前后相同。',
  '', '在每个outer development（原fit+select）中，沿用E32两折StratifiedGroupKFold(seed32050+fold)。每个训练offset由完整排除其所属inner-held实体的三seed E29模型生成；该模型的fit与选轮select也不含目标实体。重建原嵌套fit/select划分并核对已有manifest与checkpoint路径，保存逐行分配。禁止使用同一outer模型对其训练记录的in-sample输出拟合融合。',
  '', 'Raw CNN只在上述outer-development的cross-fit offset上拟合标签；固定30轮末轮，不使用原select或outer选轮，不做超参数搜索。因此Raw训练集包含原E29-S的fit+select；这与E29-L的inner选轮协议不同，已预先固定。outer始终仅评估，跨拟合训练概率与outer原E29概率存在训练规模/分布差异，属于保守pilot限制。',
  '', 'CNN结构：Conv1d(9,16,k5)→GELU→Conv1d(16,24,k5)→GELU→Conv1d(24,24,k3)→GELU→时间均值→有效拍均值→Linear(24,4)。同一方法所有拍共享参数，无BatchNorm或Dropout。末层权重和bias初始化零；delta=2*tanh(readout/2)，概率按q*exp(delta)重新归一化。实现使用expm1代数形式，初始化与原q逐值完全相同。B/C/D/E基础权重初始化及batch顺序按seed配对，AdamW lr3e-4 wd0.01，loss=NLL+0.02*mean(delta²)，clip3，batch64，固定30轮。',
  '', 'E使用同一4532参数CNN，只接收重复映射成9通道的E29四类概率，不读波形；是无Raw的额外容量/非线性校准对照，不是与Raw信息相同的输入。B/C/D之间模型参数量完全相同，仅窗口长度不同。每个Raw seed都修正同一个E29 Ensemble3，最终再平均三个Raw seed的修正概率，不能把它误称为三个独立预训练模型。',
  '', '## 类别与训练随机性','','|方法|OMI R²|NOMI R²|Control R²|Healthy R²|患者原始三列R²|患者条件三类R²|','|---|---:|---:|---:|---:|---:|---:|']
 for m,v in recomputed.items():lines.append(f'|{m}|'+ '|'.join(f'{v[k]:.6f}' for k in KEYS[5:])+'|')
 lines+=['','Clinical-3为全体ECG上前三列R²的均值；患者原始三列指标只对真实患者计算，排除恒定Healthy列；条件三类将患者P(OMI/NOMI/Control)归一化。OMI/NOMI AUC只取真实OMI或NOMI记录，分数P(OMI)/(P(OMI)+P(NOMI))。Macro AUC是四类one-v-rest AUC平均；Macro F1按四类argmax。跨折拼接AUC包含不同fold输出之间的排序，必须结合各折变化判断。','',
  '|Raw方法|Overall单seed mean±SD|Clinical-3 mean±SD|条件AUC mean±SD|','|---|---|---|---|']
 for m in ('B','C','D','E'):lines.append(f'|{m}|'+ '|'.join(f"{result[m]['mean'][k]:.6f} ± {result[m]['sd'][k]:.6f}" for k in KEYS[:3])+'|')
 lines+=['','## 覆盖、失败案例及计算成本','']
 for k in range(4):
  z=quality[quality.label==k];lines.append(f"类别{k}：{len(z)}条；无有效拍{int((z.usable==0).sum())}条；有效拍均值{z.usable.mean():.2f}；扩展窗口含下一R的有效拍比例{z.extended_next_r_overlap.sum()/max(1,z.usable.sum()):.3f}。")
 for m in ('B','C','D','E'):
  z=pd.DataFrame([r for r in rows if r['method']==m]);lines.append(f"{m}相对A：类别改变{len(z)}条，纠正{int(z.corrected.sum()) if len(z) else 0}条，退化{int(z.regressed.sum()) if len(z) else 0}条。")
 execution=json.load(open(OUT/'execution_audit.json'));lines+=['', f"36次新增CNN训练，共1080个epoch；每个分支seed4532参数，三seed合计13596新增参数。GPU {json.load(open(OUT/'environment.json'))['gpu']}；数据审计、缓存、冻结预测重算及训练总耗时{execution['seconds']:.1f}秒。统计报告耗时另记report_audit.json。所有checkpoint位于fold*/{{B,C,D,E}}/seed*/checkpoint.pth。",'',
  'error_transitions.csv仅含匿名Train行号、类别及概率，不输出真实患者身份。完整逐折逐seed指标、所有OOF预测、原始概率与crossfit offset、scaler、fold/nested划分均保存；大体积连续波形与局部窗口缓存保留远端；36个新增CNN checkpoint、完整OOF预测、配置代码、统计与报告均同步到本地。',
  '', '## 局限与后续决策','',
  '100Hz、九导联、代理窗口、自动R峰误差和快心率筛选限制结论；负结果不能证明500Hz原始波形或真正临床ST/T边界没有信息。来源标定不确定；预训练已见这些Train无标签ECG，故本次是下游监督划分复核，并非独立预训练泛化。三个Raw seed不能替代独立预训练或外部队列。没有根据outer结果调结构、学习率、窗口或正则。',
  '', '如果窗口AUC没有跨折稳定提升或临床三类退化，按预定规则停止，不追加更大CNN与更多窗口。只有同时超过原E29和无Raw容量对照、且临床表现稳定，才值得建议独立确认后再讨论正式多尺度模型；本次不自动执行后续实验。',
  '', '复现：在远端/home/ubuntu/HeartLang-main运行 .venv-heartlang/bin/python scripts/curriculum_v1/run_e34.py，然后运行report_e34.py。训练支持按COMPLETE跳过已完成CNN，不覆盖历史目录。所有代码副本见code/。']
 (OUT/'E34_PILOT_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
 b.dump(OUT/'report_audit.json',dict(seconds=time.time()-started,weighted_metric_replay_matches_sklearn=True,bootstrap_draws=2000,cluster='patient/healthy entity',test_arrays_read=False))
 (OUT/'status.txt').write_text('complete')
 print('report complete',flush=True)
if __name__=='__main__':main()
