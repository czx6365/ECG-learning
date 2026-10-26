#!/usr/bin/env python3
"""Validation-only morphology audit and inexpensive frozen ridge probes."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'heartlang').exists())
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
sys.path.insert(0, str(ROOT / 'scripts/curriculum_v1'))
os.environ.setdefault('C2_INDEX_DIR', 'datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict')
os.environ.setdefault('C2_INIT_CHECKPOINT', 'unused_evaluation_only')
import train_morphology_v2_physaware_vq as base
from train_c2_from_c1 import IndexedPhysTargetDataset


def load_model(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['model']
    model = base.MorphologyV2PhysAwareVQ(
        codebook_size=state['quantize.embedding.weight'].shape[0], codebook_dim=64,
        token_len=96, ema_decay=0.99, phys_head_source='z_q',
        continuous_mean=payload['continuous_mean'], continuous_std=payload['continuous_std'])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def pooled(model, directory, split, device, batch_size=32):
    waves = np.load(directory / f'{split}_data.npy', mmap_mode='r')
    chans = np.load(directory / f'{split}_data_in_chans.npy', mmap_mode='r')
    times = np.load(directory / f'{split}_data_in_times.npy', mmap_mode='r')
    outputs = {k: [] for k in ('z_e', 'z_q', 'z_e_mean_max', 'z_q_mean_max')}
    empty = 0
    for start in range(0, len(waves), batch_size):
        x = torch.tensor(np.array(waves[start:start + batch_size]), device=device, dtype=torch.float32)
        c = torch.tensor(np.array(chans[start:start + batch_size]), device=device)
        t = torch.tensor(np.array(times[start:start + batch_size]), device=device)
        valid, _, ze, zq, _, _ = model.encode_valid(x, c, t)
        counts = valid.sum(1)
        empty += int((counts == 0).sum())
        for key, vectors in (('z_e', ze), ('z_q', zq)):
            full = torch.zeros(*valid.shape, 64, device=device)
            full[valid] = vectors
            mean = full.sum(1) / counts.clamp_min(1).unsqueeze(1)
            maximum = full.masked_fill(~valid[..., None], -torch.inf).amax(1)
            maximum[counts == 0] = 0
            outputs[key].append(mean.cpu().numpy())
            outputs[key + '_mean_max'].append(torch.cat([mean, maximum], dim=1).cpu().numpy())
        print(f'features {split} {min(start + batch_size, len(waves))}/{len(waves)}', flush=True)
    return {k: np.concatenate(v) for k, v in outputs.items()}, empty


def metrics(y, pred):
    names = ['OMI', 'NOMI', 'Clinical Control', 'Healthy']
    mask = y[:, 3] == 0
    result = {'overall_r2': float(r2_score(y, pred, multioutput='uniform_average')),
              'per_output_r2': dict(zip(names, r2_score(y, pred, multioutput='raw_values').tolist())),
              'patient_subset_three_output_r2': float(r2_score(y[mask, :3], pred[mask, :3], multioutput='uniform_average')),
              'mse': float(np.mean((y - pred) ** 2))}
    pair = (y[:, 0] == 1) | (y[:, 1] == 1)
    result['omi_vs_nomi_auc_score_difference'] = float(roc_auc_score(y[pair, 0], pred[pair, 0] - pred[pair, 1]))
    return result


@torch.inference_mode()
def morphology(model, dataset, frame, device):
    size = model.codebook_size
    reports = {}
    for cohort in ('patient', 'healthy'):
        indices = frame.index[frame.cohort.str.lower().eq(cohort)].to_numpy()
        usage = np.zeros(size, dtype=np.int64)
        joints = [np.zeros((size, k), dtype=np.int64) for k in (2, 3, 3, 2)]
        sums = {'squared_error': 0., 'quant_error': 0., 'tokens': 0,
                'shift2': 0, 'shift4': 0}
        for start in range(0, len(indices), 16):
            rows = [dataset[int(i)] for i in indices[start:start + 16]]
            x, c, t, labels = [torch.tensor(np.stack([row[j] for row in rows]), device=device)
                               for j in range(4)]
            x = x.float()
            valid, beats, ze, zq, sid, _ = model.encode_valid(x, c, t)
            rec = model.decoder(zq)
            sums['squared_error'] += float((rec - beats).square().sum())
            sums['quant_error'] += float((ze - zq).square().sum())
            sums['tokens'] += len(sid)
            sv = sid.cpu().numpy()
            usage += np.bincount(sv, minlength=size)
            lv = labels[valid].cpu().numpy()
            for fi, table in enumerate(joints):
                k = table.shape[1]
                ok = (lv[:, fi] >= 0) & (lv[:, fi] < k)
                table += np.bincount(sv[ok] * k + lv[ok, fi], minlength=size*k).reshape(size, k)
            for mag in (2, 4):
                for delta in (-mag, mag):
                    shifted = torch.zeros_like(beats)
                    if delta > 0:
                        shifted[:, delta:] = beats[:, :-delta]
                    else:
                        shifted[:, :delta] = beats[:, -delta:]
                    _, _, ss = model.quantize(model.encoder(shifted))
                    sums[f'shift{mag}'] += int((ss != sid).sum())
        n = sums['tokens']
        prob = usage[usage > 0] / max(n, 1)
        reports[cohort] = {
            'records': len(indices), 'valid_tokens': n,
            'reconstruction_rmse': float(np.sqrt(sums['squared_error'] / max(n*96, 1))),
            'quantization_mse': sums['quant_error'] / max(n*64, 1),
            'active_sid': int((usage > 0).sum()),
            'perplexity': float(np.exp(-(prob*np.log(prob)).sum())),
            'shift_disagreement': {str(mag): sums[f'shift{mag}'] / max(2*n, 1) for mag in (2, 4)},
            'morphology_purity': {name: float(table.max(1).sum() / max(table.sum(), 1))
                                  for name, table in zip(('r_polarity', 't_polarity', 'st_state', 'qrs_type'), joints)},
            'sid_usage': usage.tolist()}
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--validation_manifest', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite {args.output}')
    device = torch.device('cuda')
    model = load_model(args.checkpoint, device)
    data = ROOT / 'datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct'
    cache = ROOT / 'results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct'
    index = ROOT / 'datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict'
    dataset = IndexedPhysTargetDataset(data, cache, index, 'val')
    vf = pd.read_csv(index / 'val_metadata.csv')
    chosen = pd.read_csv(args.validation_manifest)
    keys = set(zip(chosen.source_split, chosen.source_index))
    keep = [(s, int(i)) in keys for s, i in zip(vf.source_split, vf.source_index)]
    vf = vf.loc[keep]
    audit = morphology(model, dataset, vf, device)
    # vf retains original strict-val index; morphology indexes refer to dataset rows.
    contract = json.loads((ROOT / 'configs/curriculum_v1/data_contract.json').read_text())
    directory = ROOT / contract['main_dataset_dir']
    features, labels, empty = {}, {}, {}
    for split in ('train', 'val'):
        features[split], empty[split] = pooled(model, directory, split, device)
        labels[split] = np.load(directory / f'{split}_labels.npy').astype(np.float32)
    probes = {}
    for representation in features['train']:
        candidates = []
        for alpha in (0.1, 1., 10., 100.):
            estimator = Ridge(alpha=alpha)
            estimator.fit(features['train'][representation], labels['train'])
            pred = estimator.predict(features['val'][representation])
            candidates.append({'alpha': alpha, **metrics(labels['val'], pred)})
        probes[representation] = {'selected_by': 'four-output validation R2',
                                  'best': max(candidates, key=lambda r: r['overall_r2']),
                                  'all_alpha_results': candidates}
    result = {'checkpoint': str(args.checkpoint), 'test_used': False,
              'probe_protocol': 'frozen Ridge, intercept, no standardization, train fit/val selection',
              'morphology': audit, 'probes': probes, 'empty_records': empty}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v['best'] for k, v in probes.items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
