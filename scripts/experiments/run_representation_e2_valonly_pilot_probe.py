#!/usr/bin/env python3
"""E2 中等规模 pilot 的冻结 CLS 小样本验证 probe；绝不读取测试集。"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]
from ecg_representation_e2 import ECGRepresentationMLM
from evaluate_representation_e2 import extract
from evaluate_vocabulary_pilot import load_model
import run_representation_e1_fusion as e1

BASE = ROOT / "results/ecg_representation_comparison"
OUT = BASE / "experiment2_medium_valonly_probe_v2"
MODES = ("waveform", "sid_only", "hybrid", "ze_only")


def sample_indices(split, n_each):
    meta = pd.read_csv(e1.META / f"{split}_metadata.csv", low_memory=False)
    rng = np.random.default_rng(20261016 + (split == "val"))
    chosen = np.concatenate([rng.choice(np.flatnonzero(meta["label_id"].to_numpy() == k), n_each, replace=False)
                             for k in range(4)])
    return np.sort(chosen), meta


def main():
    if OUT.exists():
        raise FileExistsError(f"拒绝覆盖: {OUT}")
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_model(e1.TOKENIZER, device).eval().requires_grad_(False)
    train_ix, train_meta = sample_indices("train", 256)
    val_ix, val_meta = sample_indices("val", 64)
    result = {"note": "5-epoch pretrain diagnostic; train/val only; not architecture ranking",
              "sample": {"train": len(train_ix), "val": len(val_ix),
                         "train_indices_sha256": hashlib.sha256(train_ix.tobytes()).hexdigest(),
                         "val_indices_sha256": hashlib.sha256(val_ix.tobytes()).hexdigest()},
              "modes": {}}
    for mode in MODES:
        ckpt = BASE / f"experiment2_medium_{mode}_seed0_e5/checkpoint-best.pth"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = ECGRepresentationMLM(mode).to(device).eval().requires_grad_(False)
        model.load_state_dict(payload["model"], strict=True)
        x_train = extract(model, teacher, "train", 64, device, train_ix)
        x_val = extract(model, teacher, "val", 64, device, val_ix)
        scaler = StandardScaler().fit(x_train)
        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)
        y_train = train_meta.iloc[train_ix]["label_id"].to_numpy(int)
        y_val = val_meta.iloc[val_ix]["label_id"].to_numpy(int)
        # 固定较强正则与更宽松的优化预算，避免短跑诊断的收敛警告。
        clf = LogisticRegression(C=0.1, max_iter=2000, tol=1e-3, random_state=0)
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_val)
        classification = {"macro_auc_ovr": float(roc_auc_score(y_val, prob, multi_class="ovr", average="macro")),
                          "macro_f1": float(f1_score(y_val, prob.argmax(1), average="macro"))}
        train_patient, val_patient = y_train != 3, y_val != 3
        y_reg_train = pd.to_numeric(train_meta.iloc[train_ix]["overall_max_stenosis"], errors="coerce").to_numpy(float)[train_patient]
        y_reg_val = pd.to_numeric(val_meta.iloc[val_ix]["overall_max_stenosis"], errors="coerce").to_numpy(float)[val_patient]
        assert np.isfinite(y_reg_train).all() and np.isfinite(y_reg_val).all()
        reg = Ridge(alpha=100.0).fit(x_train[train_patient], y_reg_train)
        pred = reg.predict(x_val[val_patient])
        stenosis = {"r2": float(r2_score(y_reg_val, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_reg_val, pred)))}
        result["modes"][mode] = {"checkpoint": str(ckpt), "best_epoch": payload["best_epoch"],
                                 "classification_optimizer_iterations": clf.n_iter_.tolist(),
                                 "val_classification": classification, "val_stenosis": stenosis}
        print(json.dumps({"mode": mode, "class": classification, "stenosis": stenosis}), flush=True)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT / "COMPLETE").touch()


if __name__ == "__main__":
    main()
