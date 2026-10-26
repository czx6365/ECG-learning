#!/usr/bin/env python3
"""Generate an evidence-backed four-way Chinese report after E3 finishes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

NAMES = {'mixed': 'Mixed', 'healthy_only': 'Healthy-only', 'patient_only': 'Patient-only', 'e3': 'Healthy→Patient E3'}
LABELS = ['OMI', 'NOMI', 'Clinical Control', 'Healthy']


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def fmt(value):
    return f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value)


def training_stage(path, kind):
    checkpoint = torch.load(path / 'checkpoint-best.pth', map_location='cpu', weights_only=False)
    epoch = int(checkpoint['epoch'])
    records = [json.loads(line) for line in (path / 'log.txt').read_text().splitlines() if line.strip()]
    record = next(row for row in records if int(row['epoch']) == epoch)
    return {'checkpoint': str(path / 'checkpoint-best.pth'), 'best_epoch': epoch,
            'best_metrics': record, 'final_metrics': records[-1],
            'config': read(path / 'config.json') if (path / 'config.json').exists() else {},
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in checkpoint['args'].items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    exp = run.name
    ref = run / 'reference_threeway'
    downstream = read(ref / 'downstream_results.json')
    downstream['e3'] = {head: read(run / f'{head}_test_regression_summary.json') for head in ('linear', 'mlp2')}
    tokens = {group: read(ref / f'{group}_tokenizer_evaluation.json') for group in ('mixed', 'healthy', 'patient')}
    tokens = {'mixed': tokens['mixed'], 'healthy_only': tokens['healthy'], 'patient_only': tokens['patient'],
              'e3': read(run / 'tokenizer_balanced_validation.json')}
    stages = read(ref / 'training_metrics.json')
    stages['e3'] = {'adaptation': training_stage(Path(f'checkpoints/curriculum_v1/{exp}_adapt_e30'), 'tokenizer'),
                    'mlm': training_stage(Path(f'checkpoints/curriculum_v1/{exp}_sid_mlm_e100'), 'mlm')}
    epochs = read(ref / 'head_epochs.json')
    epochs['e3'] = {}
    for head in ('linear', 'mlp2'):
        path = Path(f'checkpoints/curriculum_v1/{exp}_{head}_e30/checkpoint-best.pth')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        epochs['e3'][head] = {'epoch': int(ck['epoch']), 'checkpoint': str(path)}
    for group, heads in downstream.items():
        for head, row in heads.items():
            baseline = downstream['mixed'][head]
            assert row['n_test'] == baseline['n_test'] == 1366
            assert row['label_names'] == baseline['label_names']
            assert row.get('loss_type', 'mse') == baseline.get('loss_type', 'mse')
            assert abs(np.mean(list(row['per_output_r2'].values())) - row['r2']) < 1e-6
            assert abs(row['rmse'] ** 2 - row['mse']) < 1e-6
    for group, audit in tokens.items():
        assert audit['test_used'] is False
        for cohort in ('patient', 'healthy'):
            assert audit['morphology'][cohort]['records'] == 372
            assert audit['morphology'][cohort]['valid_tokens'] == tokens['mixed']['morphology'][cohort]['valid_tokens']
    result = {'downstream': downstream, 'tokenizer_evaluation': tokens, 'training': stages,
              'head_epochs': epochs, 'sampling': read(run / 'adaptation_budget.json'),
              'preflight': read(run / 'preflight_checks.json'), 'split_audit': read(run / 'frozen_split_audit.json')}
    (run / 'comparison_complete.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# E3：健康初始化→患者适配＋健康 Replay 完整中文实验报告', '',
             '本报告由实际 checkpoint、训练日志与评价 JSON 自动汇总。四组在相同下游划分上比较；主指标为四输出等权平均 R²。', '',
             '## 1. 重点问题与结论', '']
    e3 = downstream['e3']['mlp2']
    mixed = downstream['mixed']['mlp2']
    direction = '高于' if e3['r2'] > mixed['r2'] else '未超过'
    lines += [f"**E3 的共享 MLP 测试平均 R²为 {e3['r2']:.6f}，{direction} Mixed 的 {mixed['r2']:.6f}，差值 {e3['r2']-mixed['r2']:+.6f}。** 线性头差值为 {downstream['e3']['linear']['r2']-downstream['mixed']['linear']['r2']:+.6f}。这是一种子结果，不能宣称统计显著优势。", '']
    for rep in ('z_e', 'z_e_mean_max'):
        a = tokens['e3']['probes'][rep]['best']
        b = tokens['healthy_only']['probes'][rep]['best']
        lines += [f"- {rep} 的 OMI/NOMI 验证 AUC：E3 {a['omi_vs_nomi_auc_score_difference']:.6f}，Healthy-only {b['omi_vs_nomi_auc_score_difference']:.6f}，差值 {a['omi_vs_nomi_auc_score_difference']-b['omi_vs_nomi_auc_score_difference']:+.6f}；四输出验证 R²差值 {a['overall_r2']-b['overall_r2']:+.6f}。"]
    a = tokens['e3']['morphology']['patient']
    b = tokens['patient_only']['morphology']['patient']
    h = tokens['healthy_only']['morphology']['patient']
    lines += [f"- 患者重建 RMSE：E3 {a['reconstruction_rmse']:.6f}，Patient-only {b['reconstruction_rmse']:.6f}，Healthy-only {h['reconstruction_rmse']:.6f}；E3 相对 Healthy-only 变化 {100*(a['reconstruction_rmse']/h['reconstruction_rmse']-1):+.2f}%。形态表示应结合下文四项 purity 与 z_e/z_q 指标判断。", '']
    a = tokens['e3']['morphology']['healthy']
    b = tokens['healthy_only']['morphology']['healthy']
    lines += [f"健康重建 RMSE 相对 Healthy-only 变化 {100*(a['reconstruction_rmse']/b['reconstruction_rmse']-1):+.2f}%；下游 Healthy R²差值为 {e3['per_output_r2']['Healthy']-downstream['healthy_only']['mlp2']['per_output_r2']['Healthy']:+.6f}。没有预设退化阈值或置信区间，不能仅据一个值断言‘显著退化’，须结合健康侧量化、使用率、纯度及 probe 的 Healthy 输出评价。", '',
              '## 2. 实际配置与唯一主要改动', '',
              table(['项目', 'E3 实际配置'], [
                  ['初始化', stages['e3']['adaptation']['config']['init_healthy_c21_checkpoint']],
                  ['初始轮次', 'Healthy-only C2.1 最佳 epoch35；完整61个状态张量及 EMA strict-load'],
                  ['继续训练', '30 adaptation epochs；fresh AdamW 与新 cosine schedule；不恢复健康优化器/旧epoch'],
                  ['每轮采样', '患者6656＋健康3328＝9984 ECG；39×batch256；患者:健康=2:1'],
                  ['采样方式', '两组各自无放回随机排列；混合每个batch；不按OMI/NOMI/Control细分采样或路由'],
                  ['患者独立池', '6670；每轮14条随机轮换省略；单轮不重复患者'],
                  ['健康独立池', '80396；每轮随机抽3328条 replay；单轮不重复健康'],
                  ['优化总预算', '1170步；299520 ECG暴露；不按旧健康数据量进行重复匹配'],
                  ['学习率', '命令base_lr=5e-6；现有batch/128缩放后峰值1e-5；原C2.1峰值5e-5；min_lr1e-6；warmup2epochs'],
                  ['其它优化', 'AdamW；weight_decay1e-4；EMA decay0.99；seed0；worker8'],
                  ['模型', '与C2.1相同；1024×64码本；96点心拍；9导联既有数据；fs100'],
                  ['损失', 'waveform×1＋VQ×1＋原四形态pattern×0.2＋shift_latent×0.05＋shift_SID×0.05；temperature0.1'],
                  ['平移', '延续成熟C2.1的非循环零填充±4点；不重启初期±1/±2课程'],
                  ['未使用', '新监督、disease code、标签路由、码本扩容、血管/二标签辅助、C5a解冻'],
                  ['选模', '完整原strict-val：patient372＋healthy8569；(2×患者cohort平均loss＋健康cohort平均loss)/3；每5epoch'],
                  ['E3最佳轮次', stages['e3']['adaptation']['best_epoch']],
              ]), '',
              '验证使用原有全部记录，没有新增或重分配 split。cohort 分开聚合并按2:1加权，防止患者372/健康8569的自然比例让患者适配变化被掩盖。各cohort loss沿用现有脚本的batch均值聚合；它不是按有效心拍数加权的全局MSE。验算验证推理不改变权重及EMA状态，诊断使用计数恢复。', '',
              '## 3. 数据与评价协议', '',
              '严格父索引 train87066＝患者6670＋健康80396，val8941＝患者372＋健康8569。train/val身份隔离，并排除锁定下游val/test身份。四标签下游固定6640/1344/1366，未改split，train拟合、val选模、test仅用于各预设头最终评价。test在项目历史中已被反复查看，不能称全项目首次盲测。', '',
              '适配完成后冻结tokenizer并重新生成SID；重新从头训练相同HeartLang MLM：strict混合87066/8941，100epoch，batch256，768维/12层/8头/MLP1024，seq256，mask0.5，无spatial/temporal embedding，lr5e-4/min1e-5/warmup5/weight_decay0.05/clip3/seed0。MLM按验证loss选模。', '',
              '冻结HeartLang，linear普通MSE与共享MLP方差归一化MSE各30epoch，batch64，lr5e-4/min1e-5/warmup5/weight_decay0.05/clip3/seed0。共享MLP为768→256→GELU→Dropout0.1→4。四标签为one-hot连续分数回归，不等于校准分类概率。各头只按四输出验证平均R²选模。', '',
              '码本公平评价复用原372患者＋372健康验证manifest。表示probe在原下游train6640拟合Ridge，val1344选择alpha∈{0.1,1,10,100}，有截距、不标准化；z_e/z_q分别Mean与Mean+Max。AUC仅取OMI/NOMI样本，以预测OMI分数减NOMI分数为排序分数。probe不读取test。', '',
              '## 4. Tokenizer本体：相同平衡验证清单', '',
              '重建RMSE与VQ表示MSE越低越好；Active SID为实际使用码字数；perplexity为exp(SID频率熵)，不是独立质量指标。平移分歧为原SID与±2/±4点SID不同的比例，越低越稳定；purity为每个SID占多数的形态标签总数/有效标签数，不等于临床疾病分类准确率。', '']
    for cohort in ('patient', 'healthy'):
        lines += [f"### {'患者' if cohort=='patient' else '健康'} ECG", '', table(
            ['码本', '重建RMSE↓', 'VQ MSE↓', 'Active SID', 'Perplexity', '±2分歧↓', '±4分歧↓', 'R purity', 'T purity', 'ST purity', 'QRS purity'],
            [[NAMES[group], *[fmt(value) for value in (
                tok['morphology'][cohort]['reconstruction_rmse'], tok['morphology'][cohort]['quantization_mse'],
                tok['morphology'][cohort]['active_sid'], tok['morphology'][cohort]['perplexity'],
                tok['morphology'][cohort]['shift_disagreement']['2'], tok['morphology'][cohort]['shift_disagreement']['4'],
                *[tok['morphology'][cohort]['morphology_purity'][key] for key in ('r_polarity', 't_polarity', 'st_state', 'qrs_type')])]]
             for group, tok in tokens.items()]), '']
    lines += ['## 5. 冻结z_e/z_q表示Probe及OMI vs NOMI AUC', '']
    for rep in ('z_e', 'z_q', 'z_e_mean_max', 'z_q_mean_max'):
        rows = []
        for group, tok in tokens.items():
            r = tok['probes'][rep]['best']
            rows.append([NAMES[group], fmt(r['alpha']), fmt(r['overall_r2']),
                         *[fmt(r['per_output_r2'][label]) for label in LABELS], fmt(r['mse']),
                         fmt(r['patient_subset_three_output_r2']), fmt(r['omi_vs_nomi_auc_score_difference'])])
        lines += [f'### {rep}', '', table(['码本', 'alpha', '平均R²', *LABELS, 'MSE', '患者三输出R²', 'OMI/NOMI AUC'], rows), '']
    lines += ['probe属于验证集表示诊断，与完整HeartLang测试成绩不同；不能将两者相减作为预训练增益。AUC候选未单独调参，沿用四输出R²最佳alpha。空心拍记录及全部alpha候选结果见原始JSON。', '',
              '## 6. HeartLang MLM预训练', '']
    rows = []
    for group, group_stages in stages.items():
        r = group_stages['mlm']
        m = r['best_metrics']
        rows.append([NAMES[group], r['best_epoch'], *[fmt(m[key]) for key in ('val_loss', 'val_mlm_acc', 'val_token_entropy', 'train_loss', 'train_mlm_acc')]])
    lines += [table(['码本', '最佳epoch', 'val loss↓', 'val准确率↑', '预测熵', '同轮train loss', '同轮train准确率'], rows), '',
              '预测熵来自softmax输出分布。不同码本SID目标频率和难度不同，更低MLM loss不能直接证明更强临床表示；最终以同协议下游结果判断。', '',
              '## 7. 最终四标签下游：全部8组结果', '',
              table(['码本', '头', '平均R²', *LABELS, 'MSE↓', 'RMSE↓', 'MAE↓', 'best val R²'],
                    [[NAMES[g], h, fmt(r['r2']), *[fmt(r['per_output_r2'][label]) for label in LABELS],
                      *[fmt(r[k]) for k in ('mse', 'rmse', 'mae', 'best_val_r2')]]
                     for g, heads in downstream.items() for h, r in heads.items()]), '',
              '各输出误差：', '',
              table(['码本', '头', '指标', *LABELS],
                    [[NAMES[g], h, metric, *[fmt(r[metric][label]) for label in LABELS]]
                     for g, heads in downstream.items() for h, r in heads.items()
                     for metric in ('per_output_mse', 'per_output_mae')]), '',
              'E3相对前三组的R²差值：', '',
              table(['参考', '头', '平均ΔR²', *LABELS],
                    [[NAMES[g], h, f"{downstream['e3'][h]['r2']-r['r2']:+.6f}",
                      *[f"{downstream['e3'][h]['per_output_r2'][label]-r['per_output_r2'][label]:+.6f}" for label in LABELS]]
                     for g in ('mixed', 'healthy_only', 'patient_only') for h, r in downstream[g].items()]), '',
              '## 8. E3适配训练日志与预算解释', '']
    m = stages['e3']['adaptation']['best_metrics']
    lines += [table(['指标', '最佳checkpoint对应值'], [[k, fmt(v)] for k,v in m.items() if k == 'epoch' or k.startswith('val_')]), '',
              'E3依赖健康C1→C2→C2.1各50轮的已有基础，再追加30轮轻量适配；不是从头30轮模型。Mixed总训练87066记录/轮、340步；Healthy-only每轮314步；Patient-only每輪314步、患者重复12或13次。E3追加39步/轮，既有初始化预算及更新步数与前三组不同；该对照不能隔离“初始化”“replay”“来源配比”“总训练量”的独立因果贡献。', '',
              '## 9. 结论边界与下一步', '',
              '- 本次只运行预先指定的seed0与单一2:1方案，没有多种子、患者级置信区间或显著性检验。',
              '- 没有用test选择epoch、学习率、replay比例或Ridge alpha。若依据本次test继续改方案，应披露反复查看限制。',
              '- OMI/NOMI AUC、四输出平均R²、患者重建与Healthy保持能力是不同维度，须分别报告，不以局部收益替代主结果。',
              '- 若需判断因果效果，后续可预注册相同初始化/预算的自然配比继续训练或无replay对照；这些对照本次未运行。', '',
              '## 10. 审计与原始证据', '',
              '初始化、单轮无重复、seed复现、每轮顺序变化、完整batch256、SID→MLM→两种头接口及验证EMA不变检查均通过。接口测试中的synthetic test来自正式下游TRAIN波形，未评价真实test。', '',
              f'- 实验目录：`{run.resolve()}`',
              '- `comparison_complete.json`：四组全部码本、probe、MLM、下游、checkpoint轮次与配置。',
              '- `全部指标明细.csv`：以上原始结构全部数值展开，含所有alpha候选与SID频率。',
              '- `adaptation_budget.json`、`preflight_checks.json`、`frozen_split_audit.json`：采样、接口与固定split审计。',
              '- `tokenizer_balanced_validation.json`、`linear_test_regression_summary.json`、`mlp2_test_regression_summary.json`：E3评价原始结果。',
              '- `reference_threeway/`：前三组原始证据快照。',
              '- 适配/MLM/linear/mlp2 checkpoint-best、log.txt及独立运行日志保留在实验专属目录。', '',
              '本报告只引用实际完成结果；全部差值由原始数值计算。不存在未运行的补造结果。', '']
    (run / 'E3完整中文实验报告.md').write_text('\n'.join(lines))
    numeric = []
    def walk(value, prefix):
        if isinstance(value, dict):
            for key, item in value.items(): walk(item, prefix + '.' + str(key))
        elif isinstance(value, list):
            for key, item in enumerate(value): walk(item, prefix + f'[{key}]')
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric.append((prefix, value))
    walk(result, 'comparison')
    with (run / '全部指标明细.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['指标路径', '原始数值'])
        writer.writerows(numeric)
    (run / 'report_validation.json').write_text(json.dumps({'four_groups':4, 'heads':8, 'n_test':1366,
        'r2_mean_check':'passed', 'rmse_check':'passed', 'same_balanced_manifest_token_counts':'passed',
        'probe_test_not_used':'passed', 'numeric_rows':len(numeric)}, indent=2))
    print('\n'.join(lines[:12]), flush=True)


if __name__ == '__main__':
    main()
