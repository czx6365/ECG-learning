#!/usr/bin/env python3
"""Matched-budget cohort sampling pilot; reuse C2.1 network and losses."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, SequentialSampler, Subset

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'heartlang').exists())
sys.path.insert(0, str(ROOT / 'scripts/curriculum_v1'))
INDEX = 'datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict'
os.environ.setdefault('C2_INDEX_DIR', INDEX)
# Required by legacy imports only; build_model below never loads this path.
os.environ.setdefault('C2_INIT_CHECKPOINT', 'unused_fresh_initialization')
os.environ.setdefault('C21_INIT_CHECKPOINT', 'unused_fresh_initialization')
import train_c21_shift_invariant as c21

base = c21.base
MODE = os.environ.get('VOCAB_MODE', 'mixed')
N = int(os.environ.get('VOCAB_SAMPLES_PER_EPOCH', '12288'))
EPOCHS = int(os.environ.get('VOCAB_EPOCHS', '50'))
HEALTHY_EPOCHS = int(os.environ.get('VOCAB_HEALTHY_EPOCHS', '10'))
AUDIT = {}
ORIGINAL_TRAIN = base.train_one_epoch
ORIGINAL_EVAL = c21._original_evaluate
ORIGINAL_WRITE = c21._original_write_config


def cohort_counts(mode, n, epochs, healthy_epochs, natural_fraction):
    if mode == 'healthy_then_patient':
        return [0] * healthy_epochs + [2 * n // 3] * (epochs - healthy_epochs)
    if mode == 'mixed':
        total = (epochs - healthy_epochs) * (2 * n // 3)
        return [(i + 1) * total // epochs - i * total // epochs for i in range(epochs)]
    if mode == 'natural':
        total = int(round(n * epochs * natural_fraction))
        return [(i + 1) * total // epochs - i * total // epochs for i in range(epochs)]
    raise ValueError(mode)


def identity_draws(frame, cohort, count, seed):
    group = frame.loc[frame.cohort.str.lower().eq(cohort)]
    groups = [g.index.to_numpy() for _, g in group.groupby('entity_id', sort=True)]
    if not groups:
        raise RuntimeError(f'Empty {cohort} training cohort')
    rng = np.random.default_rng(seed)
    choices = rng.integers(len(groups), size=count)
    return np.array([groups[k][rng.integers(len(groups[k]))] for k in choices], dtype=np.int64)


class PlannedSampler(Sampler):
    def __init__(self, frame, seed):
        fraction = float(frame.cohort.str.lower().eq('patient').mean())
        self.counts = cohort_counts(MODE, N, EPOCHS, HEALTHY_EPOCHS, fraction)
        ptotal = sum(self.counts)
        # Mixed and sequential curriculum have IDENTICAL within-cohort draws.
        self.patient = identity_draws(frame, 'patient', ptotal, seed + 100)
        self.healthy = identity_draws(frame, 'healthy', N * EPOCHS - ptotal, seed + 200)
        self.seed = seed
        self.epoch = 0
        self.history = []

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return N

    def indices(self, epoch):
        pc = self.counts[epoch]
        po = sum(self.counts[:epoch])
        ho = epoch * N - po
        indices = np.concatenate([self.patient[po:po + pc], self.healthy[ho:ho + N - pc]])
        np.random.default_rng(self.seed + 1000 + epoch).shuffle(indices)
        assert len(indices) == N
        return indices

    def __iter__(self):
        return iter(self.indices(self.epoch).tolist())


def build_loaders(args, data_dir, cache_dir):
    global AUDIT
    if args.epochs != EPOCHS or N % args.batch_size:
        raise RuntimeError('Epoch/batch budget mismatch')
    if args.resume:
        raise RuntimeError('Pilot resume must explicitly preserve best-selection/RNG state; not enabled')
    train = c21.c2.IndexedPhysTargetDataset(data_dir, cache_dir, ROOT / INDEX, 'train')
    val_all = c21.c2.IndexedPhysTargetDataset(data_dir, cache_dir, ROOT / INDEX, 'val')
    frame = pd.read_csv(ROOT / INDEX / 'train_metadata.csv').reset_index(drop=True)
    vf = pd.read_csv(ROOT / INDEX / 'val_metadata.csv').reset_index(drop=True)
    assert len(frame) == len(train) and len(vf) == len(val_all)
    if set(frame.entity_id) & set(vf.entity_id):
        raise RuntimeError('Tokenizer train/val identity overlap')
    excluded = set()
    contract = json.loads((ROOT / 'configs/curriculum_v1/data_contract.json').read_text())
    for split in ('val', 'test'):
        locked = pd.read_csv(ROOT / contract['main_dataset_dir'] / f'{split}_metadata.csv')
        stem = locked.csv_name.astype(str).str.replace(r'\.csv$', '', regex=True)
        excluded.update(locked.cohort.str.lower() + ':' + stem.str.rsplit('_', n=1).str[-1])
    if (set(frame.entity_id) | set(vf.entity_id)) & excluded:
        raise RuntimeError('Locked downstream validation/test leaked into tokenizer pool')
    p = np.flatnonzero(vf.cohort.str.lower().eq('patient'))
    h = np.flatnonzero(vf.cohort.str.lower().eq('healthy'))
    if len(h) < len(p) or len(p) == 0:
        raise RuntimeError('Invalid validation cohort')
    h = np.random.default_rng(20261007).choice(h, len(p), replace=False)
    selected = np.sort(np.concatenate([p, h]))
    val = Subset(val_all, selected.tolist())
    sampler = PlannedSampler(frame, args.seed)
    AUDIT = {'mode': MODE, 'fresh_initialization': True,
             'samples_per_epoch': N, 'epochs': EPOCHS,
             'patient_counts_by_epoch': sampler.counts,
             'patient_exposures': len(sampler.patient), 'healthy_exposures': len(sampler.healthy),
             'patient_draw_sha256': hashlib.sha256(sampler.patient.tobytes()).hexdigest(),
             'healthy_draw_sha256': hashlib.sha256(sampler.healthy.tobytes()).hexdigest(),
             'sampling_unit': 'uniform identity then uniform ECG within identity',
             'tokenizer_val_patient': len(p), 'tokenizer_val_healthy': len(h),
             'strict_train_val_overlap': 0, 'locked_downstream_val_test_overlap': 0,
             'manifest_sha256': hashlib.sha256((ROOT / INDEX / 'assignment_manifest.csv').read_bytes()).hexdigest(),
             'budget_unit': 'ECG records; heartbeat counts separately logged',
             'test_waveforms_used': False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vf.iloc[selected].to_csv(args.output_dir / 'tokenizer_validation_manifest.csv', index=False)
    np.save(args.output_dir / 'sampled_patient_indices.npy', sampler.patient)
    np.save(args.output_dir / 'sampled_healthy_indices.npy', sampler.healthy)
    return (train, val, sampler,
            DataLoader(train, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=args.pin_mem, drop_last=True),
            DataLoader(val, sampler=SequentialSampler(val), batch_size=args.batch_size,
                       num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False))


class CheckedShiftVQ(c21.ShiftInvariantMorphologyVQ):
    def forward(self, *args, **kwargs):
        loss, log = super().forward(*args, **kwargs)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite vocabulary pilot loss')
        return loss, log


def build_model(args, continuous_stats):
    return CheckedShiftVQ(codebook_size=args.codebook_n_emd, codebook_dim=args.codebook_emd_dim,
                          token_len=args.token_len, ema_decay=args.ema_decay,
                          quantize_kmeans_init=args.quantize_kmeans_init, phys_head_source=args.phys_head_source,
                          continuous_mean=continuous_stats['mean'], continuous_std=continuous_stats['std'])


def write_config(args, *rest):
    ORIGINAL_WRITE(args, *rest)
    path = args.output_dir / 'config.json'
    cfg = json.loads(path.read_text())
    cfg.update(experiment='normal_to_patient_vocabulary_pilot', sampling=AUDIT,
               shift_curriculum={'1-10': 1, '11-25': 2, '26-50': 4},
               validation_shift_radius=4, best_checkpoint_eligible_after_epoch=25,
               historical_c21_is_reference_not_budget_matched=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    print(json.dumps(AUDIT, ensure_ascii=False), flush=True)


def evaluate(model, loader, device, args, weights):
    fixed = {**weights, 'shift_radius': 4}
    result = ORIGINAL_EVAL(model, loader, device, args, fixed)
    result['objective_loss'] = result['loss']
    if int(weights.get('shift_radius', 1)) < 4:
        result['loss'] = 1e9
    return result


def train_one_epoch(model, loader, sampler, optimizer, device, epoch, args, *rest):
    result = ORIGINAL_TRAIN(model, loader, sampler, optimizer, device, epoch, args, *rest)
    pc = sampler.counts[epoch - 1]
    result[0].update(sampled_patient_records=pc, sampled_healthy_records=N - pc)
    if MODE == 'healthy_then_patient' and epoch == HEALTHY_EPOCHS:
        base.save_checkpoint(model, optimizer, args, epoch, tag='healthy_stage')
    return result


base.build_loaders = build_loaders
base.build_model = build_model
base.write_config = write_config
base.evaluate = evaluate
base.train_one_epoch = train_one_epoch

if __name__ == '__main__':
    base.main()
