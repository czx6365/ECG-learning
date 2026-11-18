# HeartLang code snapshot — 2026-11-18

Source is the current training checkout at `/home/ubuntu/HeartLang-main`.
This snapshot updates the HeartLang package and research scripts through E36,
and includes the E30 SID Utility objective-comparison runner.

- `scripts/curriculum_v1/` contains E12 continuous-input SID pretraining,
  E25 layer fusion, E29 Class/Shared Query, grouped evaluation and subsequent audits.
- `scripts/e30_sid_utility/` contains the new SID CE / continuous reconstruction /
  joint-objective experiment, audit and reporting scripts.
- `scripts/experiments/` and `scripts/data_prep/` contain representation experiments
  and preparation utilities.

The default experiment paths and dependencies follow the original training
checkout. E30 can be invoked from the repository root with:

```bash
PYTHONPATH=scripts/curriculum_v1 python scripts/e30_sid_utility/run_e30_sid.py smoke
PYTHONPATH=scripts/curriculum_v1 python scripts/e30_sid_utility/run_e30_sid.py all
PYTHONPATH=scripts/curriculum_v1 python scripts/e30_sid_utility/report_e30.py
```

Validation for this upload consists of syntax parsing of all 213 Python sources
in `heartlang/` and `scripts/`. Full training has not been rerun for this snapshot.
Existing upstream formatting includes trailing whitespace and mixed line endings.
