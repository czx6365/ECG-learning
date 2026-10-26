#!/usr/bin/env python3
"""在同一 E3 checkpoint、患者划分和 probe 预算下比较 E7 token Mean+Max 与 +z_e。"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments")]
import run_representation_e1_fusion as e1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", default="results/ecg_representation_comparison/experiment1_e7_meanmax_fusion")
    args = parser.parse_args()
    e1.FEATURES = {
        "E7_token_meanmax": ("heartlang_mean_max",),
        "E7_token_meanmax_z_e": ("heartlang_mean_max", "z_e_mean_max"),
    }
    e1.RUN = ROOT / args.output
    if e1.RUN.exists():
        raise FileExistsError(f"拒绝覆盖已有结果: {e1.RUN}")
    e1.RUN.mkdir(parents=True)
    data, audit = e1.load_data()
    shapes = {split: {key: list(value.shape) for key, value in data[split]["arrays"].items()}
              for split in ("train", "val", "test")}
    assert shapes["train"]["heartlang_mean_max"][1] == 1536
    assert shapes["train"]["z_e_mean_max"][1] == 128
    e1.dump(e1.RUN / "data_audit.json", {**audit, "feature_shapes": shapes})
    e1.dump(e1.RUN / "source_hashes.json", {str(p): e1.sha(p) for p in (e1.TOKENIZER, e1.SENTENCE, e1.MANIFEST)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    results = []
    for task in ("classification", "stenosis"):
        for method in e1.FEATURES:
            for seed in args.seeds:
                row = e1.train_one(data, method, task, seed, args, device)
                results.append(row)
                print(json.dumps({"task": task, "method": method, "seed": seed,
                                  "best_epoch": row["best_epoch"],
                                  "val": row["val"]["macro_auc_ovr" if task == "classification" else "r2"],
                                  "test": row["test"]["macro_auc_ovr" if task == "classification" else "r2"]}), flush=True)
                e1.dump(e1.RUN / "partial_summary.json", results)
    e1.dump(e1.RUN / "summary.json", results)
    (e1.RUN / "COMPLETE").touch()


if __name__ == "__main__":
    main()
