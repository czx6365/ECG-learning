"""Write transparent Chinese stage reports and integrated decision."""
from pathlib import Path
import json,shutil
root=Path('/home/ubuntu/HeartLang-main');out=root/'results/pre_e34_evaluation'
def j(p):return json.loads((out/p).read_text())
base=j('00_baseline_audit/baseline_metrics.json');cov=j('02_e33a_tokenizer_coverage/coverage_summary.json')
assoc=j('03_e33b_error_association/adjusted_association.json');boot=j('04_e33c_raw_information/paired_entity_bootstrap.json')
fold=j('04_e33c_raw_information/fold_metrics.json');comp=j('04_e33c_raw_information/complementarity.json')
decision=j('05_integrated_decision/PRE_E34_DECISION.json')
(out/'00_baseline_audit/BASELINE_AUDIT_CN.md').write_text(f'''# E29-S 基线审计

状态：COMPLETE。由三 seed 原始 OOF 概率等权平均重算，与历史报告逐项核对。数据为固定 E12-A 预训练下的下游监督折外评价；原 Train ECG 曾参与无标签预训练。

|指标|重算值|
|---|---:|
|Overall R²|{base['E29']['overall']:.6f}|
|Clinical-3 R²|{base['E29']['clinical3']:.6f}|
|OMI/NOMI AUC|{base['E29']['auc']:.6f}|
|Macro AUC|{base['macro_auc']:.6f}|

外层折与实体映射沿用 `results/e29_shared_generalization/fold_assignment.csv` 和下游 Train metadata。Test 未访问。
''')
shutil.copy2(root/'results/e32_omi_nomi_residual/E32_B_REPORT_CN.md',out/'01_e32b_residual/E32_B_REPORT_CN.md')
(out/'02_e33a_tokenizer_coverage/E33_A_COVERAGE_REPORT_CN.md').write_text(f'''# E33-A Tokenizer 覆盖审计

状态：PARTIAL。全 Train 6,640 条统一 100 Hz 波形按真实 XQRS 峰、相邻 R 中点切拍、96 点裁剪/补零重放；6,640 条保存的 token 序列逐值一致。11 条没有可用内部 R 峰，单列 Unknown。逐点映射示例、完整几何表和失败记录见同目录。患者 3,320 条与健康 3,320 条都可追溯到统一 100 Hz 波形；患者与健康来自不同源目录，任何四分类 raw 模型均需警惕来源捷径。

R 后 0–400 ms 的**几何窗口**，单 token 平均覆盖比例 {cov['single_mean']:.3f}，同导联 SID 序列的并集平均 {cov['sequence_mean']:.3f}。按每条 ECG 的平均覆盖恰为 1 计，单 token 比例 {cov['single_complete']:.3f}，序列比例 {cov['sequence_complete']:.3f}。序列补偿明显；不能以单 token 缺口推断这段波形已从整个 SID 消失。E26 的 32.45% 指单拍保留到检测 R 后 400 ms 的比例，不是临床 T 波完整率。

**医学边界未完成**：现有流程只检测 R 峰。当前环境有 wfdb、无 neurokit2，未得到经质量验证的 QRS onset/offset、J 点、ST 段或 T 末端定位，故不能报告可信 ST/T 覆盖率，也不使用 mV 级临床阈值。原始 500 Hz CSV 仅有患者来源抽样核对；没有覆盖四类的可比高采样率队列。100 Hz 数据不能插值后当成真实高分辨率信息。

切拍代码：`scripts/data_prep/ECG_DSA/tokenize_pretrain_qrs.py`；本次重放：`scripts/curriculum_v1/pre_e34_geometry.py`。该审计证明几何路径，不等同于临床专家标注。
''')
(out/'03_e33b_error_association/E33_B_ERROR_ASSOCIATION_CN.md').write_text(f'''# E33-B 覆盖与错误关联

状态：PARTIAL。使用 E29-S 原 Train OOF 概率，未训练分类器。分析真实 OMI/NOMI 的 2,808 条 ECG；几何覆盖可用 2,802 条。预设 R 后 0–400 ms 单 token 覆盖代理与条件 log loss 的 Spearman 相关约 +0.019。控制类别、RR、有效点比例、token 数和 outer fold 后，覆盖每升高 1 SD 的线性 log loss 系数 {assoc['coef_per_1sd_coverage']:+.4f}，实体聚类稳健 SE {assoc['cluster_se']:.4f}。方向没有支持“覆盖越少错误越多”的简单假设，且估计不精确。

覆盖组的类别组成和病例难度不同，不能用组间 AUC 差异推断因果。医学 ST/T 边界缺失，当前关联仅针对几何代理；无法回答真正 ST/T 覆盖与错误的关系。患者 OMI/NOMI 来源相同，健康来源差异不直接驱动这一二分类相关性；信号质量仍未被独立可靠量化。

明细：`analysis_table.csv`、`coverage_groups.csv`、`adjusted_association.json`。匿名案例使用 T 开头行号，报告不含患者真实身份。
''')
f0=[v['Raw_D2']['auc']-v['E29']['auc'] for v in fold]
(out/'04_e33c_raw_information/E33_C_RAW_INFORMATION_CN.md').write_text(f'''# E33-C 连续波形信息互补性

状态：PARTIAL。统一 100 Hz、9 导联、1000 点全窗输入固定轻量 1D-CNN，沿用 E29-S fit/select/outer 分组；每折三个配对 seed，30 轮，select 选 checkpoint。C1 Raw-only 的 OMI/NOMI AUC {base['Raw_C1']['auc']:.6f}，低于 E29 {base['E29']['auc']:.6f}。C1 Overall R² {base['Raw_C1']['overall']:.6f}，Clinical-3 R² {base['Raw_C1']['clinical3']:.6f}；其四分类整体值不能解释为 OMI/NOMI 增量。两者条件概率相关 {comp['probability_correlation']:.3f}；Raw 正确而 E29 错误 {comp['raw_only_correct']} 条，反向 {comp['e29_only_correct']} 条。错误互补本身不证明独立增量。

D2 使用每个 outer development 内两折 cross-fitting：原 E29 三 seed checkpoint 与新训练 Raw CNN 三 seed 均对 inner held-out 生成无监督标签泄漏的预测，然后只在这些预测上训练二参数条件残差。outer held-out 使用既存 E29-S 和 C1 三 seed 模型。D2 pooled AUC {base['Raw_D2']['auc']:.6f}，相对 E29 {boot['Raw_D2']['auc']['delta']:+.6f}，实体 bootstrap 95% 区间 {boot['Raw_D2']['auc']['ci95']}。三折差值依次 {f0[0]:+.6f}、{f0[1]:+.6f}、{f0[2]:+.6f}。Overall R² 差 {boot['Raw_D2']['overall']['delta']:+.6f}，Clinical-3 差 {boot['Raw_D2']['clinical3']['delta']:+.6f}。没有稳定增量证据。

C2 QRS-only、C3 ST-T-only、C4 Extended-heartbeat 状态：BLOCKED。当前没有经过质量验证的 QRS/J/T 边界，固定 R 后窗口不能冒充生理区。未训练这些模型，也未用 100 Hz 插值冒充高分辨率波形。D2 是预测级诊断融合，不是 E34。

完整 C1 checkpoint、内层 D2 checkpoint、OOF 概率、配置与日志均在本目录。实体 bootstrap 仅覆盖固定预测，不涵盖重新训练、重新划分和独立预训练。
''')
(out/'05_integrated_decision/PRE_E34_FINAL_REPORT_CN.md').write_text(f'''# E34 前置综合结论

## 当前模型

保留 E29 Shared Ensemble3。固定预训练、下游监督 Train-only OOF：Overall R² {base['E29']['overall']:.6f}，Clinical-3 R² {base['E29']['clinical3']:.6f}，OMI/NOMI AUC {base['E29']['auc']:.6f}。历史 Validation 与 Test 未用于本轮新模型选择。

## 七个问题的回答

1. **冻结表示是否已充分利用？** E32-B 的合法 96 维 Shared 线性残差使 AUC 从 0.792372 降至 0.776524，三折均负；当前固定方案没有可复现的剩余线性增量。但不能证明所有下游读出均已穷尽。
2. **Tokenizer 是否有临床相关覆盖不足？** 已证实单 token 的 R 后几何窗口常不完整，整段 SID 序列补偿明显。缺少可信 ST/T 边界，尚不能判断临床 ST/T 是否真正丢失。
3. **覆盖不足与错误有关吗？** 几何代理未出现明确的预期方向关联；真正 ST/T 覆盖与错误的关系仍未知。观察相关不能证明因果。
4. **Raw ECG 有可靠独立增量吗？** 全窗 CNN 单独 AUC 0.739257；合法 D2 融合 AUC 0.793142，仅比 E29 高 0.000771，三折一负两正，95% 实体区间跨零。尚无可靠增量。
5. **足以支持 E34 吗？** 当前不满足 GO 门槛。决定为 **CONDITIONAL GO：先补边界定位、来源与质量审计、独立患者确认，不启动 E34 训练**。
6. **未来应优先新增什么？** 现有证据不足以在完整 ST-T 时间覆盖、连续形态或跨导联局部信息中排序。先建立可靠 J/ST/T 标注与同源队列，再比较容量匹配的区域对照。
7. **论文方法贡献？** 当前更扎实的贡献是可追溯的 token 几何覆盖、序列补偿与严格折外原始波形增量审计；新的多尺度模型方法尚缺必要的机制和独立增量证据。

## 证据层级与缺口

已验证：E29-S 基线复现、6,640 条 token 逐值重放、E32-B 负向、C1/D2 固定 OOF 结果。机制线索：单 token 缺口有序列补偿。未验证：临床 ST/T 边界与质量等级、C2/C3/C4 对照、四类可比高采样率队列、独立预训练未见实体的外部确认。E12-A 无标签预训练已见当前 Train ECG，不能称为 pretraining-unseen 评估。

不将 Attention 权重解释为病灶定位；不将固定 R 后 400 ms 称作 T 波末端；不将 Raw-only 或微小融合差异解释成 Codebook 不可逆丢失。E34 代码和训练均未启动。
''')
for src,dst in [('00_baseline_audit/BASELINE_AUDIT_CN.md','BASELINE_AUDIT_CN.md'),('01_e32b_residual/E32_B_REPORT_CN.md','E32_B_REPORT_CN.md'),('02_e33a_tokenizer_coverage/E33_A_COVERAGE_REPORT_CN.md','E33_A_COVERAGE_REPORT_CN.md'),('03_e33b_error_association/E33_B_ERROR_ASSOCIATION_CN.md','E33_B_ERROR_ASSOCIATION_CN.md'),('04_e33c_raw_information/E33_C_RAW_INFORMATION_CN.md','E33_C_RAW_INFORMATION_CN.md'),('05_integrated_decision/PRE_E34_FINAL_REPORT_CN.md','PRE_E34_FINAL_REPORT_CN.md')]:shutil.copy2(out/src,out/dst)
shutil.copy2(out/'05_integrated_decision/PRE_E34_DECISION.json',out/'PRE_E34_DECISION.json')
print('reports written')
