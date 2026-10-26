#!/usr/bin/env python3
"""Filter the audited strict tokenizer split to healthy records only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent_index', type=Path, required=True)
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--contract', type=Path, default=Path('configs/curriculum_v1/data_contract.json'))
    a = ap.parse_args()
    contract = json.loads(a.contract.read_text())
    parent_hash = hashlib.sha256((a.parent_index / 'assignment_manifest.csv').read_bytes()).hexdigest()
    if parent_hash != contract['c1_assignment_manifest_sha256']:
        raise RuntimeError('Parent strict manifest differs from frozen contract')
    frames = {}
    arrays = {}
    for split in ('train', 'val'):
        frame = pd.read_csv(a.parent_index / f'{split}_metadata.csv')
        codes = np.load(a.parent_index / f'{split}_source_code.npy')
        rows = np.load(a.parent_index / f'{split}_source_index.npy')
        assert len(frame) == len(codes) == len(rows)
        assert np.array_equal(frame.source_index.to_numpy(), rows)
        assert np.array_equal(frame.source_split.map({'train': 0, 'val': 1}).to_numpy(), codes)
        keep = frame.cohort.astype(str).str.lower().eq('healthy').to_numpy()
        frames[split] = frame.loc[keep].reset_index(drop=True)
        arrays[split] = (codes[keep], rows[keep])
        assert len(frames[split]) > 0
        assert set(frames[split].cohort.str.lower()) == {'healthy'}
    train_entities = set(frames['train'].entity_id)
    val_entities = set(frames['val'].entity_id)
    assert not train_entities & val_entities
    for split in ('val', 'test'):
        locked = pd.read_csv(Path(contract['main_dataset_dir']) / f'{split}_metadata.csv')
        stems = locked.csv_name.astype(str).str.replace(r'\.csv$', '', regex=True)
        ids = set(locked.cohort.str.lower() + ':' + stems.str.rsplit('_', n=1).str[-1])
        assert not (train_entities | val_entities) & ids
    a.output_dir.mkdir(parents=True, exist_ok=False)
    for split in frames:
        frames[split].to_csv(a.output_dir / f'{split}_metadata.csv', index=False)
        np.save(a.output_dir / f'{split}_source_code.npy', arrays[split][0])
        np.save(a.output_dir / f'{split}_source_index.npy', arrays[split][1])
    pd.concat(list(frames.values()), ignore_index=True).to_csv(a.output_dir / 'assignment_manifest.csv', index=False)
    summary = {'version': 'healthy_only_from_curriculum_v1_strict',
               'parent_index': str(a.parent_index), 'parent_manifest_sha256': parent_hash,
               'assignment_manifest_sha256': hashlib.sha256((a.output_dir / 'assignment_manifest.csv').read_bytes()).hexdigest(),
               'train_records': len(frames['train']), 'val_records': len(frames['val']),
               'train_entities': len(train_entities), 'val_entities': len(val_entities),
               'cohort_records': {s: frames[s].cohort.value_counts().to_dict() for s in frames},
               'cross_split_entity_overlap': 0, 'locked_downstream_val_test_overlap': 0,
               'patient_records': 0}
    (a.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
