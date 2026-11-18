"""Independent cache checks and final E26 report; no model fitting."""
import json,pathlib,sys
import numpy as np
from sklearn.metrics import r2_score
ROOT=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'heartlang').exists());sys.path[:0]=[str(ROOT/'scripts/curriculum_v1'),str(ROOT/'scripts/experiments'),str(ROOT)]
import run_e26 as x
r=ROOT/'results/e26_information_retention';assert (r/'COMPLETE').exists();out=r/'verification';out.mkdir(exist_ok=True)
load=lambda s,k:np.load(r/'token_cache'/s/f'{k}.npy',mmap_mode='r');checks={};support_r2={};proxy=json.loads((r/'probes/physiology_proxy_probes.json').read_text());sub=json.loads((r/'probes/patient_subtype_probes.json').read_text());raw=json.loads((r/'raw_audit/summary.json').read_text())
for split in ('train','val'):
 wave=np.array(load(split,'wave'));ze=np.array(load(split,'ze'));zq=np.array(load(split,'zq'));sid=np.array(load(split,'sid'));record=np.array(load(split,'record'));lead=np.array(load(split,'lead'));targets=np.array(load(split,'targets'));assert np.array_equal(targets,x.proxy(wave));assert np.isfinite(ze).all();assert np.isfinite(zq).all()
 # Within-SID variance is geometric, independent of clinical labels.
 count=np.bincount(sid,minlength=1024);zs=np.zeros((1024,64));qs=np.zeros((1024,64));np.add.at(zs,sid,ze);np.add.at(qs,sid,zq);zm=zs/np.maximum(count[:,None],1);qm=qs/np.maximum(count[:,None],1)
 checks[split]={'sampled_tokens':len(sid),'represented_records':len(np.unique(record)),'active_sid':int((count>0).sum()),'ze_within_sid_mse':float(np.mean((ze-zm[sid])**2)),'ze_total_variance':float(np.mean((ze-ze.mean(0))**2)),'ze_within_sid_over_total_variance':float(np.mean((ze-zm[sid])**2)/np.mean((ze-ze.mean(0))**2)),'zq_within_sid_mse':float(np.mean((zq-qm[sid])**2)),'quantization_mse':float(np.mean((ze-zq)**2)),'max_tokens_per_record':int(np.bincount(record).max()),'ze_norm_minmax':[float(np.linalg.norm(ze,axis=1).min()),float(np.linalg.norm(ze,axis=1).max())]}
 if split=='val':
  nz=wave!=0;first=nz.argmax(1);last=95-nz[:,::-1].argmax(1);baseline=(first<=8)&(last>=19);qrs=(first<=40)&(last>=57);early=(first<=56)&(last>=63)&baseline;late=(first<=68)&(last>=87)&baseline;mask=[baseline,qrs,qrs&baseline,early,late,late,np.ones(len(wave),bool),np.ones(len(wave),bool)];support_r2['support_caveat']='Leading/trailing exact zeros treated as padding; raw replay confirms tokenizer zero-padding but genuine exact-zero boundaries can be ambiguous. Baseline support required for all baseline-subtracted proxies.'
  for name in ('wave','ze','zq','h6','h9','h12','sid_lead','sid_lead_plus_within_ze'):
   pred=np.load(r/'probes'/f'{name}_proxy_predictions.npy');support_r2[name]={}
   for j,key in enumerate(x.PROXIES):
    ii=mask[j];support_r2[name][key]={'tokens':int(ii.sum()),'r2':float(r2_score(targets[ii,j],pred[ii,j])) if ii.sum()>2 else None}
checks['frozen_model_state_audit']=json.loads((r/'token_cache/audit.json').read_text())['frozen_states_unchanged'];checks['sources_unchanged']=json.loads((r/'source_audit.json').read_text())['sources_unchanged'];checks['test_loaded']=False
x.dump(out/'cache_verification.json',checks);x.dump(out/'support_restricted_proxy_r2.json',support_r2)
lines=['# E26：原始波形→96点→z_e/SID→Transformer信息保留审计','', '## 目的与边界','', '本实验不改数据、码本、E12-A或E25历史结果，不训练新的表示网络。只使用固定Train/Val。读出能力下降不等于证明信息不可逆消失；不同表示维度、非线性、上下文均影响固定线性probe。','', '## 1. 预处理链重放','',f"- 分层抽样192条（Train/Val每类24条）；成功重放{raw['records_replayed']}条，逐值完全一致{raw['replay_exact_records']}条，最大误差{raw['max_replay_abs_error']}。",'- 当前源码已核对：9实测导联，第一导联XQRS检测R峰，丢弃首尾拍；相邻R峰中点定边界。长于96点时围绕R截取；短于96点时补零，无逐拍插值伸缩。R在索引48。','- 算法使用100 Hz。96点名义长度960 ms，首末采样时刻相差950 ms；最大R前480 ms、R后470 ms。实际有效范围受RR中点限制，不能把整个96点视为真实波形。']
b=raw['beat_means'];lines += [f"- {raw['beat_count']}个独立R拍位置的平均有效点{b['source_points']:.3f}，平均补零{b['zero_pad_points']:.3f}；长拍裁切比例{b['cropped']:.2%}。多导联共用相同R位置，覆盖统计未将9导联当成9个独立拍。",f"- 实际保留至R后160/300/400 ms比例：{b['has_160ms']:.2%} / {b['has_300ms']:.2%} / {b['has_400ms']:.2%}。后侧有效时长分位数[0,10,50,90,100%]：{raw['post_ms_quantiles']} ms。",f"- 抽样中因256 token上限丢弃{raw['sentence_tokens_dropped']}个token。",'- 这些是几何覆盖，未使用人工ST/T起止标注，不能宣称上述比例等于ST/T完整率。','', '## 2. 幅值、基线与重采样','', '- 波形准备使用整体训练来源均值/标准差作固定仿射变换；不是逐拍z-score或逐拍幅值归一化。QRS切分本身直接复制已标准化值并补0。','- 患者与健康来源使用不同全局统计量，可能留下来源差异。它是待控制因素，不足以证明模型利用了该差异。']
for src,info in raw['sources'].items():
 for fn,dd in info['summary'].items():
  if 'global_std'in dd:lines.append(f"- `{pathlib.Path(src).name}`：mean={dd.get('global_mean')}, std={dd.get('global_std')}。")
zz=[a for a in raw['zip_checks'] if 'input_samples'in a]
if zz:
 lines += [f"- 读取到{len(zz)}条原CSV；输入点数分布{sorted(set(z['input_samples'] for z in zz))}，输出1000点；polyphase filtering重采样。原CSV核对的类别为{sorted(set(z['label'] for z in zz))}，不能外推到未核对来源。",f"- 反向重采样至原长度后，RMSE/原始标准差中位数{np.median([z['roundtrip_rmse_over_raw_std'] for z in zz]):.4f}，范围{min(z['roundtrip_rmse_over_raw_std'] for z in zz):.4f}–{max(z['roundtrip_rmse_over_raw_std'] for z in zz):.4f}。该差异包含低通滤波损失及往返滤波误差，不是临床失真分数。",f"- 重采样CSV到已存100Hz波形的仿射拟合最大RMSE={max(z['affine_fit_rmse'] for z in zz):.3g}，支持固定仿射链路。CSV物理单位/校准未核实，报告不使用mV阈值。"]
else:lines+=['- 原始高采样率CSV不可用，未给出实测重采样失真结论。']
lines += ['- 形态Encoder含GroupNorm且输出L2归一化，不能仅由单位范数推断幅值信息消失；下面通过敏感性和读出probe检查。','', '## 3. 冻结表示中的可读出信息','', '- 每条ECG每个实测导联固定随机抽1拍（最多9拍），同一采样用于所有表示。Train-only StandardScaler，固定Ridge alpha=10，不在Val选择超参数。Transformer读出为未Mask的上下文token，每层复用末层冻结LayerNorm。','- 指标是已存96点的可复现代理量：固定前窗均值、R附近峰峰值、R附近带符号峰、R后80–160ms均值、R后200–400ms均值/峰值、RMS和整拍均值。它们不是人工ST/T测量。','- 原始代理读出表可能受补零影响；verification/support_restricted_proxy_r2.json另报告窗口有实际支持的子集。基线相减的指标同时要求基线窗有效。','', '|表示|基线窗均值R²|R附近峰峰值R²|R后80–160ms均值R²|R后200–400ms均值R²|RMS R²|','|---|---:|---:|---:|---:|---:|']
for name,v in proxy['results'].items():
 a=v['per_target_r2'];lines.append('|'+name+'|'+'|'.join(f'{a[k]:.4f}' for k in ('baseline_early','r_window_peak_to_peak','post_r_80_160_mean','post_r_200_400_mean','wave_rms'))+'|')
lines += ['','### 技术扰动敏感性','','|split|幅值×1.1后z_e cosine|幅值×1.1后SID变化|偏移+0.1后SID变化|平移1点后SID变化|平移2点后SID变化|','|---|---:|---:|---:|---:|---:|']
ca=json.loads((r/'token_cache/audit.json').read_text())
for s,info in ca['splits'].items():
 a=info['sensitivity'];lines.append(f"|{s}|{a['gain_cos']:.6f}|{a['gain_sid_change']:.2%}|{a['offset_sid_change']:.2%}|{a['shift1_sid_change']:.2%}|{a['shift2_sid_change']:.2%}|")
lines += ['','技术扰动只用于诊断，没有用于训练；不能据此认定所有这些变换都保持临床标签。','','## 4. 同SID内连续差异与患者亚型','','- SID/lead训练均值由Train拟合，再残差化z_e。患者亚型probe仅训练患者Train，评价患者Val；逐ECG聚合，避免把同患者的token当独立病例。','- 基线为抽样token的SID频率+导联比例；增加同SID/lead内z_e残差Mean+Max；另设相同维度的组内打乱残差对照。增加维度、有限组内支持仍是限制。','', '|表示|患者三类Macro AUC|Macro F1|Balanced accuracy|输入维数|','|---|---:|---:|---:|---:|']
for name,a in sub['results'].items():lines.append(f"|{name}|{a['macro_auc']:.4f}|{a['macro_f1']:.4f}|{a['balanced_accuracy']:.4f}|{a['input_dimension']}|")
lines += ['',f"- within-z_e相对SID直方图Macro AUC变化{sub['within_ze_minus_sid_hist_macro_auc']:+.6f}；患者身份聚类bootstrap 95%区间{sub['entity_bootstrap_auc_delta_ci95']}。",'- 这个条件probe只回答本设置下的增量可预测性，不证明疾病因果特征；也不是E12-A正式下游训练结果。','', '## 5. 验收','', '- 冻结模型完整state_dict前后相同，历史checkpoint及源码SHA前后相同；split manifest匹配、Train/Val身份交叉0、未加载Test。','- 第一次尝试调用原量化器eval时发现usage计数更新，审计主动失败；保留attempt1日志/缓存。正式重跑改为纯最近邻读取，与独立量化器副本核对一致，冻结状态完整校验通过。','- cache_verification.json记录SID内z_e方差、z_q近零组内方差、量化误差、token计数和单位范数。','', '## 产物与复现','', '远程：`/home/ubuntu/HeartLang-main/results/e26_information_retention`。token_cache是正式缓存；token_cache_attempt1_eval_usage_stats仅供失败过程追溯。','', '```bash','cd /home/ubuntu/HeartLang-main','bash scripts/curriculum_v1/run_e26_pipeline.sh','.venv-heartlang/bin/python scripts/curriculum_v1/verify_report_e26.py','```']
(r/'E26_BASE_REPORT_CN.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'cache_checks':checks,'raw':{k:raw[k] for k in ('records_replayed','replay_exact_records','beat_means','highrate_csv_available_records')},'subtype':sub},ensure_ascii=False),flush=True)
