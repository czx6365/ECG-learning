#!/usr/bin/env python3
"""Compare healthy-only tokenizer full-flow results to corresponding mixed baselines."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=Path, required=True)
    a = ap.parse_args()
    baseline_paths = {
        'linear': Path('checkpoints/curriculum_v1/c21_pattern_shift_fourclass_linear_probe_e30/test_regression_summary.json'),
        'mlp2': Path('checkpoints/curriculum_v1/c22b_variance_mlp2_probe_e30/test_regression_summary.json')}
    results = {}
    rows = ['# Healthy-only tokenizer: full four-label flow', '',
            '| Head | Healthy-only test R² | Mixed-tokenizer baseline R² | Difference |',
            '|---|---:|---:|---:|']
    for head, path in baseline_paths.items():
        result = json.loads((a.run_dir / f'{head}_test_regression_summary.json').read_text())
        baseline = json.loads(path.read_text())
        assert result['n_test'] == baseline['n_test'] == 1366
        assert result['label_names'] == baseline['label_names']
        assert result['loss_type'] == baseline.get('loss_type', 'mse')
        delta = result['r2'] - baseline['r2']
        results[head] = {'healthy_only': result, 'mixed_baseline': baseline,
                         'mixed_baseline_path': str(path), 'overall_r2_delta': delta,
                         'per_output_r2_delta': {k: result['per_output_r2'][k] - baseline['per_output_r2'][k]
                                                  for k in result['per_output_r2']}}
        rows.append(f"| {head} | {result['r2']:.6f} | {baseline['r2']:.6f} | {delta:+.6f} |")
    rows += ['', 'Tokenizer C1/C2/C2.1 uses healthy-only strict train/val records. Sentence MLM uses the original strict mixed patient/healthy split with the tokenizer frozen.',
             'Four-label heads use the frozen patient-level 6640/1344/1366 split and validation R² checkpoint selection. Each predetermined head is evaluated on test once.',
             'No C5a partial unfreezing or binary auxiliary loss is included. This is seed0; it does not establish statistical significance.',
             'Tokenizer training uses fewer records per epoch than the mixed baseline because patient records are excluded.']
    (a.run_dir / 'comparison.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (a.run_dir / 'comparison.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows), flush=True)


if __name__ == '__main__':
    main()
