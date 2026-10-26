#!/usr/bin/env python3
"""E3 full-batch adaptation and tiny SID/MLM/head interface verification.
Synthetic test fixtures use only original downstream TRAIN waveforms.
"""
import json, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path('/home/ubuntu/HeartLang-main')
os.chdir(ROOT)
PY = str(ROOT / '.venv-heartlang/bin/python')
OUT = ROOT / 'outputs/curriculum_v1/e3_full_preflight_20261011'
OUT.mkdir(parents=True, exist_ok=False)
def run(name, script, args, env=None):
    with (OUT / f'{name}.log').open('w') as log:
        subprocess.run([PY, '-u', script, *map(str, args)], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print(f'{name} passed', flush=True)
C21 = OUT / 'adaptation'
env = os.environ.copy()
env['C2_INDEX_DIR'] = 'datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict'
env['C2_INIT_CHECKPOINT'] = env['C21_INIT_CHECKPOINT'] = 'checkpoints/curriculum_v1/healthy_only_c21_full_20261008_c21_shift_e50/checkpoint-best.pth'
run('adaptation', 'scripts/curriculum_v1/train_e3_healthy_patient_adaptation.py', [
    '--pretrain_data_dir', 'datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct',
    '--phys_cache_dir', 'results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct',
    '--output_dir', C21, '--batch_size', 256, '--epochs', 1, '--save_ckpt_freq', 1,
    '--val_freq', 1, '--max_train_steps_per_epoch', 1, '--max_eval_steps', 1,
    '--codebook_n_emd', 1024, '--codebook_emd_dim', 64, '--token_len', 96,
    '--phys_head_source', 'z_q', '--loss_schedule', 'static', '--lambda_corr', 0,
    '--lambda_deriv', 0, '--lambda_phys', .2, '--continuous_alpha', 0,
    '--lr', 5e-6, '--min_lr', 1e-6, '--warmup_epochs', 0, '--weight_decay', 1e-4,
    '--weight_decay_end', 1e-4, '--device', 'cuda', '--seed', 0, '--num_workers', 0], env)
fixture = OUT / 'fourlabel_fixture'
fixture.mkdir()
contract = json.loads((ROOT / 'configs/curriculum_v1/data_contract.json').read_text())
main = ROOT / contract['main_dataset_dir']
labels = np.load(main / 'train_labels.npy')
classes = labels.argmax(1)
selected = {i: np.flatnonzero(classes == i)[:6] for i in range(4)}
for si, split in enumerate(('train', 'val', 'test')):
    rows = np.concatenate([selected[i][2*si:2*si+2] for i in range(4)])
    for suffix in ('data', 'data_in_chans', 'data_in_times', 'labels'):
        source = np.load(main / f'train_{suffix}.npy', mmap_mode='r')
        np.save(fixture / f'{split}_{suffix}.npy', np.array(source[rows]))
    pd.read_csv(main / 'train_metadata.csv').iloc[rows].to_csv(fixture / f'{split}_metadata.csv', index=False)
SID = OUT / 'sid'
run('sid', 'scripts/curriculum_v1/generate_c2_sid_dataset.py',
    ['--source_data_dir', fixture, '--output_dir', SID, '--tokenizer_checkpoint', C21 / 'checkpoint-best.pth',
     '--splits', 'train', 'val', 'test', '--batch_size', 8, '--device', 'cuda', '--symlink_context'])
for split in ('train', 'val', 'test'):
    sid = np.load(SID / f'{split}_sid.npy')
    assert sid.shape == (8, 256) and sid.min() >= 0 and sid.max() <= 1024
MLM = OUT / 'mlm'
backbone = ['--no_spatial_embedding', '--no_temporal_embedding', '--vocab_size', 1024,
            '--pad_sid', 1024, '--seq_len', 256, '--embed_dim', 32, '--depth', 1,
            '--heads', 4, '--dim_head', 8, '--mlp_dim', 64, '--batch_size', 4,
            '--epochs', 1, '--warmup_epochs', 0, '--num_workers', 0, '--device', 'cuda']
run('mlm', 'scripts/experiments/train_heartlang_sid_pretraining.py',
    ['--sid_data_dir', SID, '--output_dir', MLM, '--val_freq', 1, '--save_ckpt_freq', 1] + backbone)
for head, loss in (('linear', 'mse'), ('mlp2', 'variance_normalized_mse')):
    target = OUT / head
    run(head, 'scripts/experiments/train_heartlang_sid_regression.py',
        ['--sid_data_dir', SID, '--pretrained_checkpoint', MLM / 'checkpoint-best.pth',
         '--output_dir', target, '--task_head_type', head, '--task_head_hidden_dim', 256,
         '--loss_type', loss] + backbone)
    r = json.loads((target / 'test_regression_summary.json').read_text())
    assert r['n_test'] == 8 and len(r['per_output_r2']) == 4 and np.isfinite(r['r2'])
report = {'full_batch_adaptation': 'passed', 'strict_healthy_initialization': 'passed',
 'validation_ema_immutable': 'passed', 'sid_generation': 'passed',
 'mlm_linear_mlp_interface': 'passed', 'real_downstream_test_used': False,
 'smoke_backbone': '32-dim/1-layer only for interface; production 768-dim/12-layer',
 'fixture': str(OUT)}
(OUT / 'checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
