#!/usr/bin/env python3
"""完全按 E7 CE 头和训练调度重跑强基线及 +z_e 融合。"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/curriculum_v1")]
import run_e4_aggregation as e4
import run_e7_objective as e7

CACHE = ROOT / "outputs/curriculum_v1/e4_e3_aggregation_20261011"
BASE = ROOT / "results/ecg_representation_comparison/experiment1_e7_original_protocol_fusion"


def train(method, seed, arrays, labels, args, device):
    output = BASE / method / f"seed{seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    e4.set_seed(seed)
    dim = arrays["train"].shape[1]
    hidden = 139 if method.endswith("_matched") else 128
    model = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(.1), nn.Linear(hidden, 4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    target = labels["train"].argmax(1)
    steps = len(target) // 64
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = -float("inf"), None, None
    history = []
    started = time.monotonic()
    for epoch in range(1, 31):
        model.train()
        order = torch.randperm(len(target), generator=generator).to(device)
        for step in range(steps):
            ix = order[step * 64:(step + 1) * 64]
            lr = e4.lr_for_step((epoch - 1) * steps + step, 30 * steps, 5 * steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(arrays["train"][ix]), target[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        val, _score, _raw = e7.evaluate(model, arrays["val"], labels["val"], "CE")
        history.append({"epoch": epoch, "val_macro_auc": val["macro_roc_auc"]})
        if val["macro_roc_auc"] > best:
            best, best_epoch, best_state = val["macro_roc_auc"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val, _score, valraw = e7.evaluate(model, arrays["val"], labels["val"], "CE")
    test, _score, testraw = e7.evaluate(model, arrays["test"], labels["test"], "CE")
    np.save(output / "val_raw_outputs.npy", valraw)
    np.save(output / "test_raw_outputs.npy", testraw)
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "method": method,
                "seed": seed, "best_epoch": best_epoch, "input_dim": dim}, output / "checkpoint-best.pth")
    record = {"method": method, "seed": seed, "best_epoch": best_epoch, "val": val, "test": test,
              "input_dim": dim, "parameters": sum(p.numel() for p in model.parameters()),
              "train_seconds": time.monotonic() - started,
              "protocol": "E7 CE: raw cached features, 128 hidden, no LayerNorm, lr 5e-4, 30 ep, val Macro OVR AUC"}
    (output / "summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
    print(json.dumps({"method": method, "seed": seed, "best_epoch": best_epoch,
                      "val_auc": val["macro_roc_auc"], "test_auc": test["macro_roc_auc"]}), flush=True)
    return record


def main():
    global BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--matched-only", action="store_true")
    args = parser.parse_args()
    if args.matched_only:
        BASE = ROOT / "results/ecg_representation_comparison/experiment1_e7_original_protocol_capacity_control"
    if BASE.exists():
        raise FileExistsError(BASE)
    BASE.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = {s: torch.tensor(np.load(CACHE / f"{s}_labels.npy"), device=device, dtype=torch.float32)
              for s in ("train", "val", "test")}
    arrays = {}
    methods = (("E7_token_meanmax_matched",) if args.matched_only
               else ("E7_token_meanmax", "E7_token_meanmax_z_e"))
    for method in methods:
        arrays[method] = {}
        for split in labels:
            x = [np.load(CACHE / f"{split}_heartlang_mean_max.npy")]
            if method.endswith("z_e"):
                x.append(np.load(CACHE / f"{split}_z_e_mean_max.npy"))
            arrays[method][split] = torch.tensor(np.concatenate(x, 1), device=device, dtype=torch.float32)
    training_args = argparse.Namespace(lr=.0005, min_lr=.00001, warmup_lr=.000001, weight_decay=.05)
    results = []
    for method in arrays:
        for seed in args.seeds:
            results.append(train(method, seed, arrays[method], labels, training_args, device))
            (BASE / "partial_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (BASE / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (BASE / "COMPLETE").touch()


if __name__ == "__main__":
    main()
