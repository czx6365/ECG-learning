"""Add fixed decision and reproducibility details to the completed report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_e29_l import OUT,e,s

def main():
    report=OUT/'E29_L_REPORT_CN.md'
    assert report.exists()
    text=report.read_text()
    if '## 最终决策' in text:return
    summary=json.loads((OUT/'summary.json').read_text())
    audit=json.loads((OUT/'code_protocol_audit.json').read_text())
    weights=json.loads((OUT/'weight_statistics.json').read_text())
    rows=pd.read_csv(OUT/'fold_seed_metrics.csv')
    lines=['','## 最终决策','','保留原 E29 Shared Query Ensemble3：第6/9/12层可学习凸组合与第9层Shared Query保持原样。停止继续搜索层组合；本次不启动稀疏选层、独立确认或其他后续实验。',
        '', 'C既没有超过原三层A，也没有超过十二层均匀B。C−A的Overall与Clinical-3区间均为负；C−B的Clinical-3区间为负，Overall区间跨零。因此没有证据支持十二层自动权重带来增量，不能把更多输入层数等同于更好表示。','']
    for ref in ('A','B'):
        c=rows[rows.method=='C'].set_index(['fold','seed'])
        a=rows[rows.method==ref].set_index(['fold','seed'])
        lines.append(f"C−{ref}在9个fold×seed配对中："+'；'.join(f"{k}改善{int(((c[k]-a[k])>0).sum())}/9" for k in ('overall','clinical3','omi_nomi_auc'))+'。')
    lines+=['','## 类别与单模型统计','','以下单模型统计：将每个seed在三个outer fold的预测拼接后计算指标，再对三个seed取均值/样本SD。与固定三seed平均概率的集成指标分开。','',
        '|方案|Overall R² mean±SD|Clinical-3 mean±SD|条件OMI/NOMI AUC mean±SD|','|---|---:|---:|---:|']
    for m in 'ABC':
        v=summary[m]
        lines.append(f"|{m}|"+'|'.join(f"{v['mean'][k]:.6f} ± {v['sd'][k]:.6f}" for k in ('overall','clinical3','omi_nomi_auc'))+'|')
    lines+=['','|方案|集成OMI R²|NOMI R²|Control R²|Healthy R²|','|---|---:|---:|---:|---:|']
    for m in 'ABC':lines.append(f"|{m}|"+'|'.join(f"{summary[m]['ensemble'][k]:.6f}" for k in ('OMI','NOMI','Control','Healthy'))+'|')
    lines+=['','C的下降并非只来自Healthy。C相对A的Healthy几乎持平，而临床三类R²全部下降。OMI/NOMI条件AUC以真实OMI或NOMI记录为对象，分数为P(OMI)/(P(OMI)+P(NOMI))；其C−A区间上界接近零，不能单独声称已确认AUC下降。Clinical-3为四类总体上的前三列R²平均；患者子集另报前三列原始概率与条件归一化R²，排除恒定Healthy列。',
        '', '## 数值、缓存与协议核查','',
        '编号由完整Transformer block显式遍历确定，缓存index0对应block1，不采用未经核对的hidden_states下标。12个block均输出768维，逐层使用同一冻结最终LayerNorm与valid mask，Mean+Max均为1536维。第6/9/12层缓存逐批与旧E25缓存对齐；A的9个最佳checkpoint全部与E29-S逐参数完全相同，OOF预测也通过一致性检查。',
        '', f"C均匀初始化与B初始logits最大差：{audit['C_uniform_initial_logits_max_abs_diff']:.3g}。C融合参数梯度有限且非零；padding token置1000后的logits最大差：{audit['padded_token_perturbation_max_abs_diff']:.3g}。冻结提取使用no_grad，冻结模块eval；原E12-A、tokenizer及历史E29-S checkpoint哈希保持不变。",
        '', '配对训练沿用E29-S的RNG恢复机制，Shared初始化不额外消耗E11训练的dropout随机流；A/B/C仅增删零初始化标量，batch顺序与基础头初始化按seed配对。所有方法保持30轮、AdamW、相同损失及inner select选模；outer只评估。',
        '', f"C的最高权重层计数：{weights['top_layer_counts']}。权重熵范围{min(weights['entropy']):.6f}–{max(weights['entropy']):.6f}，均匀12层为{np.log(12):.6f}；整体仍较分散，不能由权重直接推断病灶或临床重要性。",
        '', '全部十二层池化特征训练侧标准差为1.344–1.416，未发现数量级差异。保留原协议；没有额外拟合标准化，也没有使用历史Validation选择尺度。缓存为float32、无autocast；有效mask随第9层token保存。',
        '', '## 计算成本与可复现路径','',
        f"GPU：{json.loads((OUT/'environment.json').read_text())['gpu']}。27次训练共810个epoch；主实验、缓存与总体bootstrap总耗时{json.loads((OUT/'execution_audit.json').read_text())['seconds']:.1f}秒，附加报告/逐折bootstrap耗时未计入。缓存大小{sum(p.stat().st_size for p in (OUT/'cache').glob('*.npy'))/1024**3:.3f} GiB。不把训练后瞬时显存当作峰值显存，未采集峰值。",
        '', '完整模型与缓存位于远端 /home/ubuntu/HeartLang-main/results/e29_all_layer_fusion/。fold_assignment.csv保留原E29-S逐行分配；fold*/downstream/{A,B,C}/seed*/保留checkpoint、30轮历史及inner预测；oof_A/B/C.npy保存每个seed全部外层预测。所有文件新建于本实验目录，不覆盖历史checkpoint。',
        '', '复现命令（远端项目根目录）：','', '```bash',
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv-heartlang/bin/python scripts/curriculum_v1/run_e29_l.py',
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv-heartlang/bin/python scripts/curriculum_v1/audit_e29_l.py',
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv-heartlang/bin/python scripts/curriculum_v1/report_e29_l.py',
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv-heartlang/bin/python scripts/curriculum_v1/finalize_e29_l.py',
        '```','', '失败案例表prediction_change_cases.csv包含A与B/C预测类别改变的全部匿名记录、真实类别、纠正/退化标志。类别顺序0=OMI、1=NOMI、2=Control、3=Healthy；仅使用匿名Train行号，不含真实患者标识。',
        '', '结果仅针对已固定E12-A与原Train的分组CV。预训练包含这些无标签ECG；三个下游seed不是三个独立预训练seed。配对实体bootstrap反映固定预测的实体采样不确定性，不能替代重复训练或独立队列确认。历史Validation与Test均未读取。']
    lines.append('\n## 失败案例与跨折AUC解释\n\nC相对A共有524条ECG的集成预测类别改变：纠正177条，退化229条，另118条为不同错误类别间的变化。此处按ECG计数，同一患者可有多条记录；不是524名独立患者。\n\nC−A的各折条件OMI/NOMI AUC差为+0.000935、−0.001255、+0.003709，方向混合，与跨折拼接AUC下降不完全一致。拼接AUC还包含不同fold模型输出之间的排序，可能受到跨折分数尺度/校准差异影响；因此不能把拼接AUC下降直接归因为每折临床区分能力均下降。对直接判别能力的判断仍是“未见稳定改善”。Overall与Clinical-3的固定集成差值则在三个fold均为负。\n')
    report.write_text(text+'\n'.join(lines)+'\n')
    print('finalized')

if __name__=='__main__':main()
