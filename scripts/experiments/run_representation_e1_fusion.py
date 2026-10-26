#!/usr/bin/env python3
"""E1: 在固定 E3 表示上比较双 Encoder 特征融合。"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_recall_fscore_support,
    r2_score, roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.nn import functional as F


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
CACHE = ROOT / "outputs/curriculum_v1/e4_e3_aggregation_20261011"
META = ROOT / "datasets/ecg_datasets/ECG_DSA/qrs_sid/e3_healthy_patient_replay_20261011"
RUN = ROOT / "results/ecg_representation_comparison/experiment1_dual_encoder"
TOKENIZER = ROOT / "checkpoints/curriculum_v1/e3_healthy_patient_replay_20261011_adapt_e30/checkpoint-best.pth"
SENTENCE = ROOT / "checkpoints/curriculum_v1/e3_healthy_patient_replay_20261011_sid_mlm_e100/checkpoint-best.pth"
MANIFEST = ROOT / "configs/curriculum_v1/data_contract.json"
NAMES = ["OMI", "NOMI", "Clinical Control", "Healthy"]
FEATURES = {
    "E1-0_CLS": ("heartlang",),
    "E1-1_z_e": ("z_e_mean_max",),
    "E1-2_CLS_z_e": ("heartlang", "z_e_mean_max"),
    "E1-3_CLS_z_q": ("heartlang", "z_q_mean_max"),
}


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    contract = json.loads(MANIFEST.read_text())
    assert contract["split_manifest_sha256"] == "0eb883c55b4e4bcf580b8a4c9108b4fb5a94f8665a9db4002529c255dc11d49f"
    result = {}
    identities = {}
    audit = {"manifest_sha256": contract["split_manifest_sha256"], "splits": {}}
    for split, expected in (("train", 6640), ("val", 1344), ("test", 1366)):
        labels = np.load(CACHE / f"{split}_labels.npy")
        meta = pd.read_csv(META / f"{split}_metadata.csv", low_memory=False)
        assert labels.shape == (expected, 4) and len(meta) == expected
        assert np.array_equal(labels.argmax(1), meta["label_id"].to_numpy())
        assert np.allclose(labels.sum(1), 1) and np.isin(labels, [0, 1]).all()
        ids = meta["entity_id"].astype(str).to_numpy()
        identities[split] = set(ids)
        target = pd.to_numeric(meta["overall_max_stenosis"], errors="coerce").to_numpy(dtype=np.float32)
        patient = labels[:, 3] == 0
        assert np.array_equal(np.isfinite(target), patient)
        assert np.all((target[patient] >= 0) & (target[patient] <= 100))
        arrays = {name: np.load(CACHE / f"{split}_{name}.npy").astype(np.float32)
                  for name in {n for group in FEATURES.values() for n in group}}
        assert all(len(x) == expected and np.isfinite(x).all() for x in arrays.values())
        result[split] = {"labels": labels, "target": target, "patient": patient, "meta": meta, "arrays": arrays}
        audit["splits"][split] = {"records": expected, "patients_with_stenosis": int(patient.sum()),
                                   "entities": len(identities[split]), "classes": labels.sum(0).astype(int).tolist()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not identities[a] & identities[b]
    audit["entity_overlap"] = 0
    return result, audit


def build_features(data, names: tuple[str, ...], task: str):
    """每块表示只用本任务训练样本拟合均值和标准差。"""
    train_idx = data["train"]["patient"] if task == "stenosis" else np.ones(len(data["train"]["labels"]), bool)
    stats = {}
    for name in names:
        z = data["train"]["arrays"][name][train_idx].astype(np.float64)
        mean = z.mean(0).astype(np.float32)
        std = np.maximum(z.std(0).astype(np.float32), 1e-4)
        stats[name] = {"mean": mean, "std": std}
    output = {}
    for split in ("train", "val", "test"):
        blocks = [(data[split]["arrays"][name] - stats[name]["mean"]) / stats[name]["std"] for name in names]
        x = np.concatenate(blocks, axis=1)
        assert np.isfinite(x).all()
        output[split] = x.astype(np.float32)
    return output, stats


class FusionHead(nn.Module):
    def __init__(self, dim: int, out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def class_metrics(raw, y):
    truth = y.argmax(1)
    pred = raw.argmax(1)
    probs = np.exp(raw - raw.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    pr, re, f1, support = precision_recall_fscore_support(truth, pred, labels=range(4), zero_division=0)
    auc = [float(roc_auc_score(y[:, k], probs[:, k])) for k in range(4)]
    pair = np.isin(truth, [0, 1])
    return {"accuracy": float(accuracy_score(truth, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=range(4), average="macro")),
            "macro_auc_ovr": float(np.mean(auc)),
            "omi_nomi_auc": float(roc_auc_score(y[pair, 0], raw[pair, 0] - raw[pair, 1])),
            "per_class": {name: {"precision": float(pr[k]), "recall": float(re[k]),
                                 "f1": float(f1[k]), "auc": auc[k], "support": int(support[k])}
                          for k, name in enumerate(NAMES)},
            "confusion_matrix": confusion_matrix(truth, pred, labels=range(4)).tolist(),
            "n": len(truth)}


def regression_metrics(pred, target, groups):
    result = {"r2": float(r2_score(target, pred)),
              "rmse": float(np.sqrt(mean_squared_error(target, pred))),
              "mae": float(mean_absolute_error(target, pred)),
              "pearson": float(pearsonr(target, pred).statistic),
              "spearman": float(spearmanr(target, pred).statistic),
              "n": len(target), "groups": {}}
    for k, name in enumerate(NAMES[:3]):
        selected = groups == k
        yt, yp = target[selected], pred[selected]
        result["groups"][name] = {"n": len(yt),
                                  "r2": float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 0 else None,
                                  "rmse": float(np.sqrt(mean_squared_error(yt, yp))) if len(yt) else None,
                                  "mae": float(mean_absolute_error(yt, yp)) if len(yt) else None}
    return result


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i+512]).to(device)).cpu().numpy()
                           for i in range(0, len(x), 512)])


def train_one(data, method, task, seed, args, device, source_info=None):
    out = RUN / task / method / f"seed{seed}"
    if out.exists():
        raise FileExistsError(f"拒绝覆盖现有实验目录: {out}")
    out.mkdir(parents=True)
    seed_all(seed)
    x, stats = build_features(data, FEATURES[method], task)
    chosen = {split: (data[split]["patient"] if task == "stenosis" else np.ones(len(x[split]), bool))
              for split in x}
    yy = {split: (data[split]["target"] if task == "stenosis" else data[split]["labels"].argmax(1))
          for split in x}
    model = FusionHead(x["train"].shape[1], 1 if task == "stenosis" else 4, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    train_x = torch.from_numpy(x["train"][chosen["train"]]).to(device)
    train_y = torch.from_numpy(yy["train"][chosen["train"]]).to(device)
    steps = max(1, len(train_x) // args.batch)
    total = steps * args.epochs
    warm = steps * min(5, args.epochs)
    gen = torch.Generator().manual_seed(seed)
    best = -float("inf")
    best_epoch = 0
    best_state = None
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=gen).to(device)
        losses = []
        for step in range(steps):
            ix = order[step * args.batch:(step + 1) * args.batch]
            at = (epoch - 1) * steps + step
            lr = args.lr * ((at + 1) / max(1, warm)) if at < warm else 1e-5 + (args.lr - 1e-5) * 0.5 * (1 + np.cos(np.pi * (at - warm) / max(1, total - warm)))
            for group in optimizer.param_groups:
                group["lr"] = float(lr)
            optimizer.zero_grad(set_to_none=True)
            raw = model(train_x[ix])
            loss = (F.smooth_l1_loss(raw[:, 0], train_y[ix].float(), beta=2.0)
                    if task == "stenosis" else F.cross_entropy(raw, train_y[ix].long()))
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_raw = predict(model, x["val"][chosen["val"]], device)
        if task == "stenosis":
            val = regression_metrics(val_raw[:, 0], yy["val"][chosen["val"]],
                                     data["val"]["labels"][chosen["val"]].argmax(1))
            score = val["r2"]
        else:
            val = class_metrics(val_raw, data["val"]["labels"])
            score = val["macro_auc_ovr"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val})
        if score > best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_raw = predict(model, x["test"][chosen["test"]], device)
    if task == "stenosis":
        test = regression_metrics(test_raw[:, 0], yy["test"][chosen["test"]],
                                  data["test"]["labels"][chosen["test"]].argmax(1))
    else:
        test = class_metrics(test_raw, data["test"]["labels"])
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()},
                "feature_stats": {n: {k: torch.from_numpy(v) for k, v in s.items()} for n, s in stats.items()},
                "feature_names": FEATURES[method], "best_epoch": best_epoch, "task": task,
                "seed": seed, "input_dim": x["train"].shape[1]}, out / "checkpoint-best.pth")
    for split in ("val", "test"):
        raw = predict(model, x[split][chosen[split]], device)
        meta = data[split]["meta"].loc[chosen[split]].reset_index(drop=True)
        with (out / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if task == "stenosis":
                writer.writerow(["entity_id", "csv_name", "label_id", "target_overall_max_stenosis", "prediction"])
                for i, row in meta.iterrows():
                    writer.writerow([row["entity_id"], row["csv_name"], row["label_id"], yy[split][chosen[split]][i], raw[i, 0]])
            else:
                writer.writerow(["entity_id", "csv_name", "label_id", *[f"logit_{n}" for n in NAMES]])
                for i, row in meta.iterrows():
                    writer.writerow([row["entity_id"], row["csv_name"], row["label_id"], *raw[i].tolist()])
    summary = {"method": method, "task": task, "seed": seed, "best_epoch": best_epoch,
               "selection": "val R2" if task == "stenosis" else "val macro OVR AUC",
               "val": history[best_epoch - 1]["val"], "test": test,
               "trainable_parameters": sum(p.numel() for p in model.parameters()),
               "input_dim": x["train"].shape[1], "hidden": args.hidden,
               "duration_seconds": time.monotonic() - started,
               "source_tokenizer": str(TOKENIZER), "source_sentence": str(SENTENCE),
               "source_cache": str(CACHE), "hyperparameters": vars(args)}
    if source_info is not None:
        summary["source_info"] = source_info
        summary["source_sentence"] = source_info.get("pretrained_checkpoint")
        summary["source_cache"] = str(RUN)
    dump(out / "summary.json", summary)
    dump(out / "history.json", history)
    print(json.dumps({"method": method, "task": task, "seed": seed,
                      "best_epoch": best_epoch, "test_primary": test["r2"] if task == "stenosis" else test["macro_auc_ovr"]}), flush=True)
    return summary


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=list(FEATURES), default=list(FEATURES))
    parser.add_argument("--run_tag", default="experiment1_dual_encoder")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.seeds = [0]
    run = RUN.parent / "experiment1_smoke" if args.smoke else RUN.parent / args.run_tag
    RUN = run
    if RUN.exists():
        raise FileExistsError(f"拒绝覆盖: {RUN}")
    RUN.mkdir(parents=True)
    data, audit = load_data()
    dump(RUN / "data_audit.json", audit)
    dump(RUN / "source_hashes.json", {str(p): sha(p) for p in (TOKENIZER, SENTENCE, MANIFEST)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    results = []
    for task in ("classification", "stenosis"):
        for method in args.methods:
            for seed in args.seeds:
                results.append(train_one(data, method, task, seed, args, device))
                dump(RUN / "partial_summary.json", results)
    dump(RUN / "summary.json", results)
    (RUN / "COMPLETE").touch()


if __name__ == "__main__":
    main()
