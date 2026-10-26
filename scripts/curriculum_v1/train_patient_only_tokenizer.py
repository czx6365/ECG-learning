#!/usr/bin/env python3
"""Patient-only C1/C2/C2.1 with healthy-run matched optimization steps."""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, SequentialSampler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'heartlang').exists())
sys.path.insert(0, str(ROOT / 'scripts/curriculum_v1'))
STAGE = os.environ['PATIENT_STAGE']
N = int(os.environ.get('PATIENT_SAMPLES_PER_EPOCH', '80384'))
module = importlib.import_module({'c1': 'train_c1_word_vq', 'c2': 'train_c2_from_c1',
                                 'c21': 'train_c21_shift_invariant'}[STAGE])
base = module if STAGE == 'c1' else module.base


class RepeatedPermutationSampler(Sampler):
    """All records repeat equally, with a fresh permutation for each pass."""
    def __init__(self, count, samples, seed):
        if count < 1 or samples < 1:
            raise ValueError('Empty dataset/budget')
        self.count, self.samples, self.seed, self.epoch = count, samples, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.samples

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        remaining = self.samples
        while remaining:
            size = min(remaining, self.count)
            yield from torch.randperm(self.count, generator=rng)[:size].tolist()
            remaining -= size


def verify_patient_index(index):
    frames = {s: pd.read_csv(index / f'{s}_metadata.csv') for s in ('train', 'val')}
    for split, frame in frames.items():
        assert set(frame.cohort.str.lower()) == {'patient'}, 'Non-patient tokenizer input'
        codes = np.load(index / f'{split}_source_code.npy')
        rows = np.load(index / f'{split}_source_index.npy')
        assert np.array_equal(frame.source_index.to_numpy(), rows)
        assert np.array_equal(frame.source_split.map({'train': 0, 'val': 1}), codes)
    ids = {s: set(f.entity_id) for s, f in frames.items()}
    assert not ids['train'] & ids['val']
    contract = json.loads((ROOT / 'configs/curriculum_v1/data_contract.json').read_text())
    for split in ('val', 'test'):
        locked = pd.read_csv(ROOT / contract['main_dataset_dir'] / f'{split}_metadata.csv')
        stem = locked.csv_name.astype(str).str.replace(r'\.csv$', '', regex=True)
        excluded = set(locked.cohort.str.lower() + ':' + stem.str.rsplit('_', n=1).str[-1])
        assert not (ids['train'] | ids['val']) & excluded
    return frames


def loaders(args, data_dir, index_dir, cache_dir=None):
    if N % args.batch_size:
        raise RuntimeError('Batch budget must divide exactly')
    frames = verify_patient_index(index_dir)
    if STAGE == 'c1':
        train = module.IndexedC1Dataset(data_dir, index_dir, 'train')
        val = module.IndexedC1Dataset(data_dir, index_dir, 'val')
    else:
        c2 = module if STAGE == 'c2' else module.c2
        train = c2.IndexedPhysTargetDataset(data_dir, cache_dir, index_dir, 'train', lead_id=args.lead_id)
        val = c2.IndexedPhysTargetDataset(data_dir, cache_dir, index_dir, 'val', lead_id=args.lead_id)
    assert len(train) == len(frames['train']) and len(val) == len(frames['val'])
    sampler = RepeatedPermutationSampler(len(train), N, args.seed)
    audit = {'tokenizer_cohort': 'patient_only', 'stage': STAGE, 'sampling_seed': args.seed,
             'unique_train_records': len(train), 'unique_val_records': len(val),
             'healthy_records': 0, 'samples_per_epoch': N,
             'steps_per_epoch': N // args.batch_size, 'epochs': args.epochs,
             'total_optimization_steps': args.epochs * (N // args.batch_size),
             'total_sampled_ecg_exposures': args.epochs * N,
             'mean_record_repetitions_per_epoch': N / len(train),
             'sampling': 'repeated shuffled full ECG passes, then shuffled remainder',
             'healthy_reference': 'healthy_only_c21_full_20261008, floor(80396/256)=314 steps/epoch',
             'budget_unit': 'optimization steps / ECG exposures; heartbeat counts not matched',
             'train_val_identity_overlap': 0, 'locked_downstream_val_test_overlap': 0}
    (args.output_dir / 'patient_budget.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)
    return train, val, sampler, DataLoader(train, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True), DataLoader(val,
        sampler=SequentialSampler(val), batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)


if STAGE == 'c1':
    def build_loaders(args):
        data_dir = base.resolve_path(args.pretrain_data_dir)
        index_dir = base.resolve_path(args.split_index_dir)
        return data_dir, index_dir, *loaders(args, data_dir, index_dir)
else:
    def build_loaders(args, data_dir, cache_dir):
        return loaders(args, data_dir, base.resolve_path(Path(os.environ['C2_INDEX_DIR'])), cache_dir)

base.build_loaders = build_loaders

if __name__ == '__main__':
    base.main()
