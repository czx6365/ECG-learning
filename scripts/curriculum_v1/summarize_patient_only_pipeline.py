#!/usr/bin/env python3
"""Produce the mixed / healthy / patient vocabulary four-label comparison."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=Path, required=True)
    a = ap.parse_args()
    paths = {
        'mixed': {'linear': Path('checkpoints/curriculum_v1/c21_pattern_shift_fourclass_linear_probe_e30/test_regression_summary.json'),
                  'mlp2': Path('checkpoints/curriculum_v1/c22b_variance_mlp2_probe_e30/test_regression_summary.json')},
        'healthy_only': {head: Path('results/curriculum_v1/healthy_only_c21_full_20261008') / f'{head}_test_regression_summary.json'
                         for head in ('linear', 'mlp2')},
        'patient_only': {head: a.run_dir / f'{head}_test_regression_summary.json' for head in ('linear', 'mlp2')}}
    result = {cohort: {head: json.loads(path.read_text()) for head, path in heads.items()}
              for cohort, heads in paths.items()}
    for cohort, heads in result.items():
        for head, row in heads.items():
            ref = result['mixed'][head]
            assert row['n_test'] == ref['n_test'] == 1366
            assert row['label_names'] == ref['label_names']
            assert row.get('loss_type', 'mse') == ref.get('loss_type', 'mse')
    lines = ['# Mixed / healthy / patient tokenizer full-flow comparison', '',
             '| Tokenizer source | Linear test R² | Shared MLP test R² |', '|---|---:|---:|']
    for cohort, heads in result.items():
        lines.append(f"| {cohort} | {heads['linear']['r2']:.6f} | {heads['mlp2']['r2']:.6f} |")
    lines += ['', 'Patient and healthy tokenizer runs each use C1/C2/C2.1 50 epochs with 314 updates per epoch and batch256. Patient train records are repeatedly shuffled to match update budget; unique-data sizes still differ.',
              'Sentence MLM remains the same strict mixed patient/healthy pool. Downstream split=6640/1344/1366; seed0; linear MSE and shared MLP variance-normalized MSE, frozen HeartLang, 30 epochs.',
              'All heads selected by validation four-output R². Codebook size1024×64. No C5a unfreezing or binary auxiliary loss.',
              'Step matching does not establish statistical significance or equal heartbeat exposure.']
    (a.run_dir / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (a.run_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
