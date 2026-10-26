#!/usr/bin/env python3
"""E3: strict-load healthy C2.1, continue with patient-heavy healthy replay.

Reuse C2.1 architecture, targets, training loop, optimizer creation and losses.
Only initialization, sample composition and adaptation LR/budget change.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'heartlang').exists())
sys.path.insert(0, str(ROOT / 'scripts/curriculum_v1'))
import train_c21_shift_invariant as c21
from e3_replay_sampler import PatientReplaySampler

base = c21.base
original_build_model = c21.build_model
original_write_config = c21.write_config
original_evaluate = c21._original_evaluate


def verify_index(index):
    contract = json.loads((ROOT / 'configs/curriculum_v1/data_contract.json').read_text())
    digest = hashlib.sha256((index / 'assignment_manifest.csv').read_bytes()).hexdigest()
    if digest != contract['c1_assignment_manifest_sha256']:
        raise RuntimeError('Frozen parent index was changed')
    frames = {s: pd.read_csv(index / f'{s}_metadata.csv') for s in ('train', 'val')}
    for split, frame in frames.items():
        assert np.array_equal(frame.source_index, np.load(index / f'{split}_source_index.npy'))
        assert np.array_equal(frame.source_split.map({'train': 0, 'val': 1}), np.load(index / f'{split}_source_code.npy'))
        assert set(frame.cohort.str.lower()) == {'patient', 'healthy'}
    ids = {s: set(frame.entity_id) for s, frame in frames.items()}
    assert not ids['train'] & ids['val']
    for split in ('val', 'test'):
        locked = pd.read_csv(ROOT / contract['main_dataset_dir'] / f'{split}_metadata.csv')
        stem = locked.csv_name.astype(str).str.replace(r'\.csv$', '', regex=True)
        excluded = set(locked.cohort.str.lower() + ':' + stem.str.rsplit('_', n=1).str[-1])
        assert not (ids['train'] | ids['val']) & excluded
    return frames, digest


def build_loaders(args, data_dir, cache_dir):
    index = base.resolve_path(c21.c2.INDEX_DIR)
    frames, digest = verify_index(index)
    train = c21.c2.IndexedPhysTargetDataset(data_dir, cache_dir, index, 'train', lead_id=args.lead_id)
    val = c21.c2.IndexedPhysTargetDataset(data_dir, cache_dir, index, 'val', lead_id=args.lead_id)
    cohorts = frames['train'].cohort.str.lower()
    sampler = PatientReplaySampler(np.flatnonzero(cohorts.eq('patient')),
                                  np.flatnonzero(cohorts.eq('healthy')), args.batch_size, args.seed)
    train_loader = DataLoader(train, sampler=sampler, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    val_loader = DataLoader(val, sampler=SequentialSampler(val), batch_size=int(1.5 * args.batch_size),
                           num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    val_loader.cohort_rows = {c: np.flatnonzero(frames['val'].cohort.str.lower().eq(c))
                             for c in ('patient', 'healthy')}
    audit = {
        'experiment': 'E3 Healthy -> Patient Continue Adaptation',
        'parent_manifest_sha256': digest, 'no_split_change': True,
        'train_val_identity_overlap': 0, 'locked_downstream_val_test_overlap': 0,
        'unique_train_pool': frames['train'].cohort.value_counts().to_dict(),
        'validation_records': frames['val'].cohort.value_counts().to_dict(),
        'patient_sampled_per_epoch': sampler.patient_count,
        'healthy_sampled_per_epoch': sampler.healthy_count,
        'patient_dropped_per_epoch': len(sampler.patient) - sampler.patient_count,
        'samples_per_epoch': len(sampler), 'steps_per_epoch': sampler.steps,
        'batch_size': args.batch_size, 'epochs': args.epochs,
        'total_optimization_steps': sampler.steps * args.epochs,
        'total_ecg_exposures': len(sampler) * args.epochs,
        'sampling': 'each cohort shuffled without replacement per epoch, then mixed within each batch',
        'sampling_seed': args.seed, 'disease_subtype_used_for_sampling_or_routing': False,
        'validation_selection': '(2*patient_cohort_mean_loss + healthy_cohort_mean_loss)/3',
    }
    (args.output_dir / 'adaptation_budget.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    return train, val, sampler, train_loader, val_loader


def build_model(args, continuous_stats):
    if args.resume:
        raise RuntimeError('Do not resume the healthy optimizer/epoch; E3 is model-state initialization with a fresh schedule')
    model = original_build_model(args, continuous_stats)
    path = base.resolve_path(c21.INIT_CHECKPOINT)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    # Verify every tensor, including all EMA buffers, before the first update.
    for key, value in model.state_dict().items():
        if not torch.equal(value, checkpoint['model'][key]):
            raise RuntimeError(f'Initialization differs: {key}')
    model.e3_init_epoch = int(checkpoint['epoch'])
    print(f'E3 strict initialization verified: {len(model.state_dict())} tensors, healthy epoch {model.e3_init_epoch}', flush=True)
    return model


def loss_weights(args, epoch=None):
    # Healthy checkpoint has already reached mature ±4 training (epoch 35).
    # Preserve that loss; do not restart the initial ±1/±2 curriculum.
    weights = c21.loss_weights(args, epoch=35)
    weights['stage'] = 'e3_mature_c21_shift_radius_4'
    return weights


def evaluate(model, loader, device, args, weights):
    if not hasattr(loader, 'cohort_rows'):
        raise RuntimeError('E3 requires the audited cohort validation loader')
    saved = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith('quantize.')}
    rows = {}
    for cohort, indices in loader.cohort_rows.items():
        subset_loader = DataLoader(Subset(loader.dataset, indices), batch_size=loader.batch_size,
                                  num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
        rows[cohort] = original_evaluate(model, subset_loader, device, args, weights)
    # Evaluation only changes the diagnostic cluster_size buffer. Restore it,
    # and assert that inference never changed learned weights or EMA history.
    current = model.state_dict()
    for key, value in saved.items():
        if key != 'quantize.cluster_size' and not torch.equal(current[key], value):
            raise RuntimeError(f'Validation modified tokenizer state: {key}')
        current[key].copy_(value)
    objective = (2 * rows['patient']['loss'] + rows['healthy']['loss']) / 3
    return {'loss': objective, 'objective_loss': objective, 'best_checkpoint_eligible': 1,
            **{f'{cohort}_{key}': value for cohort, stats in rows.items() for key, value in stats.items()}}


def write_config(args, data_dir, cache_dir, train_len, val_len, continuous_stats, base_lr):
    original_write_config(args, data_dir, cache_dir, train_len, val_len, continuous_stats, base_lr)
    path = args.output_dir / 'config.json'
    config = json.loads(path.read_text())
    for key in ('init_c1_checkpoint', 'init_pattern_only_c2_checkpoint', 'c2_stages'):
        config.pop(key, None)
    init = base.resolve_path(c21.INIT_CHECKPOINT)
    config.update({
        'experiment': 'E3 Healthy -> Patient Continue Adaptation', 'stage': 'E3',
        'init_healthy_c21_checkpoint': str(init),
        'init_checkpoint_sha256': hashlib.sha256(init.read_bytes()).hexdigest(),
        'initialization': 'strict full model and EMA state; fresh optimizer; adaptation epoch starts at 1',
        'loss_schedule': 'mature C2.1 constant shift radius 4',
        'shift_curriculum': {'all_adaptation_epochs': 4},
        'main_control_variable': 'healthy C2.1 initialization + patient-heavy sampling + healthy replay',
        'validation_selection': '(2*patient_cohort_mean_loss + healthy_cohort_mean_loss)/3; original full strict val',
        'adaptation_budget': json.loads((args.output_dir / 'adaptation_budget.json').read_text()),
    })
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2))


base.build_loaders = build_loaders
base.build_model = build_model
base.loss_weights = loss_weights
base.evaluate = evaluate
base.write_config = write_config

if __name__ == '__main__':
    base.main()
