#!/usr/bin/env python3
"""E11 medical-prior validation, region ablation, and probability calibration."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]
import run_e4_aggregation as e4
import run_e9_ze_fusion_r2 as e9
import run_e10_lead_aware_ze_fusion as e10
import run_e11_region_residual_fusion as e11


EXP = "e11_prior_ablation_calibration_20261018"
DEFAULT_RUN = ROOT / "results/curriculum_v1" / EXP
E11_RUN = ROOT / "results/curriculum_v1/e11_region_residual_fusion_20261018"
DATA, TOKENIZER, MLM = e9.DATA, e9.TOKENIZER, e9.MLM
LABEL_NAMES, LEAD_NAMES, REGION_NAMES = e9.LABEL_NAMES, e10.LEAD_NAMES, e11.REGION_NAMES
MEDICAL_MEMBERS = e11.REGION_MEMBERS
RANDOM_SEEDS = [1101, 2202, 3303, 4404, 5505]
CALIBRATION_METHODS = ["uncalibrated", "temperature_nll", "temperature_mse", "vector_nll"]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_status(run_dir: Path, value: str) -> None:
    (run_dir / "status.txt").write_text(value, encoding="utf-8")
    print(value, flush=True)


def incidence(members: list[list[int]]) -> np.ndarray:
    m = np.zeros((5, 9), dtype=np.int8)
    for r, cols in enumerate(members):
        m[r, cols] = 1
    return m


def members_from_incidence(m: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(m[r]).astype(int).tolist() for r in range(5)]


def random_partition(seed: int) -> list[list[int]]:
    """Degree-preserving bipartite edge swaps from the medical incidence matrix."""
    rng = np.random.default_rng(seed)
    m = incidence(MEDICAL_MEMBERS)
    successful = 0
    attempts = 0
    while successful < 250 and attempts < 20000:
        attempts += 1
        r1, r2 = rng.choice(5, 2, replace=False)
        c1, c2 = rng.choice(9, 2, replace=False)
        sub = (m[r1, c1], m[r1, c2], m[r2, c1], m[r2, c2])
        if sub == (1, 0, 0, 1):
            m[r1, c1], m[r1, c2], m[r2, c1], m[r2, c2] = 0, 1, 1, 0
            successful += 1
        elif sub == (0, 1, 1, 0):
            m[r1, c1], m[r1, c2], m[r2, c1], m[r2, c2] = 1, 0, 0, 1
            successful += 1
    if successful < 250:
        raise RuntimeError(f"Could not randomize partition seed {seed}")
    medical = incidence(MEDICAL_MEMBERS)
    if np.array_equal(m, medical):
        raise RuntimeError(f"Random partition seed {seed} returned medical partition")
    if not np.array_equal(m.sum(1), medical.sum(1)) or not np.array_equal(m.sum(0), medical.sum(0)):
        raise RuntimeError(f"Random partition seed {seed} broke row/column degrees")
    return members_from_incidence(m)


def make_partitions() -> dict[str, list[list[int]]]:
    result = {"medical": copy.deepcopy(MEDICAL_MEMBERS)}
    seen = {incidence(MEDICAL_MEMBERS).tobytes()}
    for i, seed in enumerate(RANDOM_SEEDS):
        members = random_partition(seed)
        key = incidence(members).tobytes()
        if key in seen:
            raise RuntimeError(f"Duplicate random partition for seed {seed}")
        seen.add(key)
        result[f"random_{i:02d}"] = members
    return result


def regions_for(lead: torch.Tensor, lead_mask: torch.Tensor, members: list[list[int]]):
    membership = torch.zeros(5, 9, dtype=torch.bool, device=lead.device)
    for r, cols in enumerate(members):
        membership[r, cols] = True
    valid = lead_mask[:, None, :] & membership[None, :, :]
    count = valid.sum(2)
    regions = (lead[:, None] * valid.unsqueeze(-1)).sum(2) / count.clamp_min(1).unsqueeze(-1)
    regions = regions * (count > 0).unsqueeze(-1)
    return regions, count > 0


@torch.no_grad()
def evaluate_region(model, arrays: dict, regions: torch.Tensor, region_mask: torch.Tensor):
    model.eval()
    raw, weights = [], []
    for start in range(0, len(arrays["labels"]), 96):
        x, w, _ = model.encode(arrays["mlm"][start:start + 96], arrays["global"][start:start + 96],
                               regions[start:start + 96], region_mask[start:start + 96])
        raw.append(model.classifier(x).cpu().numpy())
        weights.append(w.cpu().numpy())
    raw = np.concatenate(raw); weights = np.concatenate(weights)
    metrics, prob = e11.e7.classification(raw, arrays["labels"].cpu().numpy(), "CE")
    return metrics, prob, raw, weights


def train_partition(partition: str, members: list[list[int]], seed: int, data: dict,
                    run_dir: Path, args, device: torch.device) -> dict:
    out = run_dir / "random_region_experiment" / partition / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    model = e11.E11RegionResidual(seed).to(device)
    e4.set_seed(seed)
    initial_sha = e11.state_sha(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cached_regions = {s: regions_for(data[s]["lead"], data[s]["mask"], members) for s in ("train", "val", "test")}
    target = data["train"]["labels"].argmax(1)
    steps = len(target) // args.batch_size
    generator = torch.Generator().manual_seed(seed)
    best_r2, best_epoch, best_state, history = -float("inf"), None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=generator).to(device)
        losses = []
        for step in range(steps):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            lr = e4.lr_for_step((epoch - 1) * steps + step, args.epochs * steps,
                                args.warmup_epochs * steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            reg, mask = cached_regions["train"]
            main, aux = model.training_outputs(data["train"]["mlm"][idx], data["train"]["global"][idx], reg[idx], mask[idx])
            main_loss = nn.functional.cross_entropy(main, target[idx])
            aux_loss = e9.patient_auxiliary_loss(aux, target[idx])
            loss = main_loss + 0.2 * aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"{partition}/seed{seed}: non-finite loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.clip); optimizer.step()
            losses.append(float(loss))
        val, _, _, _ = evaluate_region(model, data["val"], *cached_regions["val"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val,
                        "residual_alpha_tanh": float(torch.tanh(model.residual_scale).detach())})
        if val["r2"] > best_r2:
            best_r2, best_epoch, best_state = val["r2"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val, val_prob, val_raw, _ = evaluate_region(model, data["val"], *cached_regions["val"])
    # Test is first evaluated here, after validation-only checkpoint selection.
    test, test_prob, test_raw, _ = evaluate_region(model, data["test"], *cached_regions["test"])
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "partition": partition,
                "members": members, "seed": seed, "best_epoch": best_epoch,
                "selection": "validation Overall R2"}, out / "checkpoint-best.pth")
    dump(out / "history.json", history)
    pred = out / "predictions"; pred.mkdir()
    np.save(pred / "val_logits.npy", val_raw); np.save(pred / "val_probabilities.npy", val_prob)
    np.save(pred / "test_logits.npy", test_raw); np.save(pred / "test_probabilities.npy", test_prob)
    result = {"partition": partition, "members": members, "seed": seed,
              "initial_state_sha256": initial_sha, "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": best_epoch, "residual_alpha_tanh": float(torch.tanh(model.residual_scale).detach()),
              "val": val, "test": test}
    dump(out / "summary.json", result)
    print(json.dumps({"partition": partition, "seed": seed, "best_epoch": best_epoch,
                      "val_r2": val["r2"], "test_r2": test["r2"]}), flush=True)
    return result


def aggregate_partitions(results: list[dict]) -> dict:
    partitions = sorted(set(r["partition"] for r in results), key=lambda x: (x != "medical", x))
    output = {"per_seed": results, "partitions": {}}
    for partition in partitions:
        rows = [r for r in results if r["partition"] == partition]
        block = {"members": rows[0]["members"], "trainable_parameters": rows[0]["trainable_parameters"]}
        for split in ("val", "test"):
            block[split] = {}
            for key in ("r2", "mse", "mae", "macro_roc_auc"):
                v = np.array([r[split][key] for r in rows])
                block[split][key] = {"mean": float(v.mean()), "std_ddof1": float(v.std(ddof=1)), "values": v.tolist()}
            block[split]["per_output_r2"] = {}
            for label in LABEL_NAMES:
                v = np.array([r[split]["per_output_r2"][label] for r in rows])
                block[split]["per_output_r2"][label] = {"mean": float(v.mean()), "std_ddof1": float(v.std(ddof=1)), "values": v.tolist()}
        output["partitions"][partition] = block
    medical = output["partitions"]["medical"]
    random_rows = [r for r in results if r["partition"].startswith("random_")]
    output["random_all_runs"] = {}
    for split in ("val", "test"):
        for key in ("r2", "macro_roc_auc"):
            vals = np.array([r[split][key] for r in random_rows])
            output["random_all_runs"].setdefault(split, {})[key] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
    paired = []
    for seed in (0, 1, 2):
        med = next(r for r in results if r["partition"] == "medical" and r["seed"] == seed)
        rnd = [r for r in random_rows if r["seed"] == seed]
        paired.append({"seed": seed,
                       "medical_minus_random_mean_val_r2": med["val"]["r2"] - float(np.mean([r["val"]["r2"] for r in rnd])),
                       "medical_minus_random_mean_test_r2": med["test"]["r2"] - float(np.mean([r["test"]["r2"] for r in rnd]))})
    output["paired_seed_comparison"] = paired
    output["medical_vs_random"] = {
        "medical_val_r2": medical["val"]["r2"]["mean"],
        "random_all_val_r2": output["random_all_runs"]["val"]["r2"]["mean"],
        "delta_val": medical["val"]["r2"]["mean"] - output["random_all_runs"]["val"]["r2"]["mean"],
        "medical_test_r2": medical["test"]["r2"]["mean"],
        "random_all_test_r2": output["random_all_runs"]["test"]["r2"]["mean"],
        "delta_test_descriptive": medical["test"]["r2"]["mean"] - output["random_all_runs"]["test"]["r2"]["mean"],
        "medical_wins_val_runs_same_seed": int(sum(next(r for r in results if r["partition"] == "medical" and r["seed"] == rr["seed"])["val"]["r2"] > rr["val"]["r2"] for rr in random_rows)),
        "total_random_runs": len(random_rows),
    }
    return output


@torch.no_grad()
def region_ablation(data: dict, run_dir: Path, device: torch.device) -> dict:
    val = data["val"]
    regions, region_mask = regions_for(val["lead"], val["mask"], MEDICAL_MEMBERS)
    rows = []
    for seed in (0, 1, 2):
        model = e11.E11RegionResidual(seed).to(device)
        ckpt = torch.load(E11_RUN / f"checkpoints/e11_region_residual/seed{seed}/checkpoint-best.pth", map_location=device)
        model.load_state_dict(ckpt["model"], strict=True); model.eval()
        baseline, _, _, _ = evaluate_region(model, val, regions, region_mask)
        for j, name in enumerate(REGION_NAMES):
            changed_regions = regions.clone(); changed_mask = region_mask.clone()
            changed_regions[:, j] = 0; changed_mask[:, j] = False
            metric, _, _, _ = evaluate_region(model, val, changed_regions, changed_mask)
            rows.append({"seed": seed, "removed_region": name, "baseline": baseline, "ablated": metric,
                         "delta_r2": metric["r2"] - baseline["r2"],
                         "delta_auc": metric["macro_roc_auc"] - baseline["macro_roc_auc"],
                         "per_output_r2_delta": {k: metric["per_output_r2"][k] - baseline["per_output_r2"][k] for k in LABEL_NAMES}})
    aggregate = {}
    for name in REGION_NAMES:
        x = [r for r in rows if r["removed_region"] == name]
        aggregate[name] = {"delta_r2_mean": float(np.mean([r["delta_r2"] for r in x])),
                           "delta_r2_std_ddof1": float(np.std([r["delta_r2"] for r in x], ddof=1)),
                           "delta_auc_mean": float(np.mean([r["delta_auc"] for r in x])),
                           "per_output_r2_delta_mean": {k: float(np.mean([r["per_output_r2_delta"][k] for r in x])) for k in LABEL_NAMES}}
    ranking = sorted(REGION_NAMES, key=lambda n: aggregate[n]["delta_r2_mean"])
    output = {"split": "validation only", "per_seed": rows, "aggregate": aggregate,
              "importance_ranking_most_harmful_to_remove_first": ranking,
              "delta_definition": "ablated minus baseline; negative means model depended on the region"}
    dump(run_dir / "region_ablation/summary.json", output)
    with (run_dir / "region_ablation/results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["region", "delta_r2_mean", "delta_r2_sd", "delta_auc_mean", *[f"delta_{x}_r2" for x in LABEL_NAMES]])
        for name in ranking:
            x = aggregate[name]
            w.writerow([name, x["delta_r2_mean"], x["delta_r2_std_ddof1"], x["delta_auc_mean"], *[x["per_output_r2_delta_mean"][k] for k in LABEL_NAMES]])
    return output


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    x = np.exp(z)
    return x / x.sum(1, keepdims=True)


def probability_metrics(prob: np.ndarray, y_onehot: np.ndarray, bins: int = 15) -> dict:
    true = y_onehot.argmax(1); pred = prob.argmax(1)
    per_r2 = {label: float(r2_score(y_onehot[:, i], prob[:, i])) for i, label in enumerate(LABEL_NAMES)}
    conf = prob.max(1); correct = pred == true
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            ece += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    macro_ece = 0.0
    for c in range(4):
        p, t = prob[:, c], y_onehot[:, c]
        class_ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
            if m.any():
                class_ece += float(m.mean()) * abs(float(t[m].mean()) - float(p[m].mean()))
        macro_ece += class_ece / 4
    return {"r2": float(np.mean(list(per_r2.values()))), "per_output_r2": per_r2,
            "macro_roc_auc": float(roc_auc_score(y_onehot, prob, average="macro", multi_class="ovr")),
            "brier_score": float(np.mean(np.sum((prob - y_onehot) ** 2, axis=1))),
            "ece15_top_label": ece, "ece15_macro_ovr": macro_ece,
            "nll": float(log_loss(true, prob, labels=[0, 1, 2, 3])),
            "accuracy": float(accuracy_score(true, pred)), "macro_f1": float(f1_score(true, pred, average="macro"))}


def fit_temperature(logits: np.ndarray, y: np.ndarray, objective: str) -> float:
    x = torch.tensor(logits, dtype=torch.float64)
    target = torch.tensor(y.argmax(1), dtype=torch.long)
    onehot = torch.tensor(y, dtype=torch.float64)
    log_t = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    opt = torch.optim.LBFGS([log_t], lr=0.2, max_iter=150, tolerance_grad=1e-12,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); t = log_t.exp().clamp(0.05, 20.0); scaled = x / t
        loss = nn.functional.cross_entropy(scaled, target) if objective == "nll" else ((scaled.softmax(1) - onehot) ** 2).mean()
        loss.backward(); return loss
    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def fit_vector(logits: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(logits, dtype=torch.float64)
    target = torch.tensor(y.argmax(1), dtype=torch.long)
    log_scale = nn.Parameter(torch.zeros(4, dtype=torch.float64))
    bias = nn.Parameter(torch.zeros(4, dtype=torch.float64))
    opt = torch.optim.LBFGS([log_scale, bias], lr=0.2, max_iter=200, tolerance_grad=1e-11,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); b = bias - bias.mean(); z = x * log_scale.exp().clamp(0.05, 20.0) + b
        loss = nn.functional.cross_entropy(z, target) + l2 * (log_scale.square().mean() + b.square().mean())
        loss.backward(); return loss
    opt.step(closure)
    return log_scale.detach().exp().clamp(0.05, 20.0).numpy(), (bias.detach() - bias.detach().mean()).numpy()


def apply_vector(logits: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return softmax_np(logits * scale[None] + bias[None])


def calibrate(run_dir: Path) -> dict:
    y_val = np.load(DATA / "val_labels.npy").astype(np.float64)
    y_test = np.load(DATA / "test_labels.npy").astype(np.float64)
    true_val = y_val.argmax(1)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    results = []
    selected = []
    for seed in (0, 1, 2):
        base = E11_RUN / f"predictions/e11_region_residual/seed{seed}"
        val_logits = np.load(base / "val_logits.npy").astype(np.float64)
        test_logits = np.load(base / "test_logits.npy").astype(np.float64)
        params = {"uncalibrated": {}}
        t_nll = fit_temperature(val_logits, y_val, "nll")
        t_mse = fit_temperature(val_logits, y_val, "mse")
        params["temperature_nll"] = {"temperature": t_nll}
        params["temperature_mse"] = {"temperature": t_mse}

        # Five-fold stratified OOF selection of vector-scaling regularization.
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20261018 + seed).split(val_logits, true_val))
        cv = {}
        for l2 in lambdas:
            oof = np.zeros_like(y_val)
            for train_idx, hold_idx in folds:
                scale, bias = fit_vector(val_logits[train_idx], y_val[train_idx], l2)
                oof[hold_idx] = apply_vector(val_logits[hold_idx], scale, bias)
            cv[str(l2)] = probability_metrics(oof, y_val)
        chosen_l2 = max(lambdas, key=lambda z: cv[str(z)]["r2"])
        scale, bias = fit_vector(val_logits, y_val, chosen_l2)
        params["vector_nll"] = {"scale": scale.tolist(), "bias": bias.tolist(),
                                "l2": chosen_l2, "cv": cv,
                                "chosen_by": "highest five-fold OOF validation Overall R2"}

        val_probs = {
            "uncalibrated": softmax_np(val_logits),
            "temperature_nll": softmax_np(val_logits / t_nll),
            "temperature_mse": softmax_np(val_logits / t_mse),
            "vector_nll": apply_vector(val_logits, scale, bias),
        }
        val_metrics = {m: probability_metrics(p, y_val) for m, p in val_probs.items()}
        chosen_method = max(CALIBRATION_METHODS, key=lambda m: val_metrics[m]["r2"])
        # Test is touched only after every parameter and method choice above is frozen.
        test_probs = {
            "uncalibrated": softmax_np(test_logits),
            "temperature_nll": softmax_np(test_logits / t_nll),
            "temperature_mse": softmax_np(test_logits / t_mse),
            "vector_nll": apply_vector(test_logits, scale, bias),
        }
        test_metrics = {m: probability_metrics(p, y_test) for m, p in test_probs.items()}
        out = run_dir / "calibration" / f"seed{seed}"; out.mkdir(parents=True, exist_ok=False)
        dump(out / "parameters.json", params)
        for method in CALIBRATION_METHODS:
            np.save(out / f"val_{method}_probabilities.npy", val_probs[method])
            np.save(out / f"test_{method}_probabilities.npy", test_probs[method])
        row = {"seed": seed, "selected_method_by_val_r2": chosen_method,
               "parameters": params, "val": val_metrics, "test": test_metrics}
        dump(out / "summary.json", row); results.append(row)
        selected.append({"seed": seed, "method": chosen_method,
                         "val": val_metrics[chosen_method], "test": test_metrics[chosen_method]})
    aggregate = {"per_seed": results, "methods": {}, "val_selected_per_seed": selected}
    for method in CALIBRATION_METHODS:
        aggregate["methods"][method] = {}
        for split in ("val", "test"):
            aggregate["methods"][method][split] = {}
            for key in ("r2", "macro_roc_auc", "brier_score", "ece15_top_label", "ece15_macro_ovr", "nll"):
                vals = np.array([r[split][method][key] for r in results])
                aggregate["methods"][method][split][key] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
            aggregate["methods"][method][split]["per_output_r2"] = {}
            for label in LABEL_NAMES:
                vals = np.array([r[split][method]["per_output_r2"][label] for r in results])
                aggregate["methods"][method][split]["per_output_r2"][label] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
    for split in ("val", "test"):
        aggregate.setdefault("val_selected_aggregate", {})[split] = {}
        for key in ("r2", "macro_roc_auc", "brier_score", "ece15_top_label", "ece15_macro_ovr", "nll"):
            vals = np.array([r[split][key] for r in selected])
            aggregate["val_selected_aggregate"][split][key] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
        aggregate["val_selected_aggregate"][split]["per_output_r2"] = {}
        for label in LABEL_NAMES:
            vals = np.array([r[split]["per_output_r2"][label] for r in selected])
            aggregate["val_selected_aggregate"][split]["per_output_r2"][label] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
    dump(run_dir / "calibration/aggregate_summary.json", aggregate)
    return aggregate


def write_csvs(run_dir: Path, partition_summary: dict, ablation: dict, calibration: dict):
    with (run_dir / "random_region_experiment/comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["partition", "split", "overall_r2_mean", "overall_r2_sd", *[f"{x}_r2" for x in LABEL_NAMES], "macro_auc_mean"])
        for name, block in partition_summary["partitions"].items():
            for split in ("val", "test"):
                w.writerow([name, split, block[split]["r2"]["mean"], block[split]["r2"]["std_ddof1"],
                            *[block[split]["per_output_r2"][x]["mean"] for x in LABEL_NAMES], block[split]["macro_roc_auc"]["mean"]])
    with (run_dir / "calibration/comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["method", "split", "overall_r2", *[f"{x}_r2" for x in LABEL_NAMES], "macro_auc", "brier", "ece_top", "ece_macro", "nll"])
        for method, block in calibration["methods"].items():
            for split in ("val", "test"):
                b = block[split]
                w.writerow([method, split, b["r2"]["mean"], *[b["per_output_r2"][x]["mean"] for x in LABEL_NAMES],
                            b["macro_roc_auc"]["mean"], b["brier_score"]["mean"], b["ece15_top_label"]["mean"], b["ece15_macro_ovr"]["mean"], b["nll"]["mean"]])


def write_report(run_dir: Path, partitions: dict, partition_summary: dict, ablation: dict, calibration: dict):
    comp = partition_summary["medical_vs_random"]
    selected = calibration["val_selected_aggregate"]
    unc = calibration["methods"]["uncalibrated"]
    lines = ["# E11 医学区域先验、区域消融与概率校准", "", "## 结论摘要", "",
             f"医学区域在验证集的三种子 Overall R² 为 **{comp['medical_val_r2']:.6f}**，五组随机区域共15次训练均值为 **{comp['random_all_val_r2']:.6f}**，差值 **{comp['delta_val']:+.6f}**。医学区域在同seed的15个随机对照中胜出 {comp['medical_wins_val_runs_same_seed']}/{comp['total_random_runs']} 次。",
             f"测试集描述性差值为 **{comp['delta_test_descriptive']:+.6f}**；随机分组或模型选择没有使用测试集。",
             f"按验证R²选择的校准方法将 Test Overall R² 从 **{unc['test']['r2']['mean']:.6f}** 调整到 **{selected['test']['r2']['mean']:.6f}**，差值 **{selected['test']['r2']['mean']-unc['test']['r2']['mean']:+.6f}**。", "",
             "## 实验一：医学区域 vs 多组随机区域", "",
             "随机区域由医学区域关联矩阵进行二部图换边生成。每个随机划分的区域大小严格保持3/3/3/2/6，每条真实导联的重复次数也逐导联完全保持，因此网络容量、输入槽数量和信息重复度一致。", "",
             "### 固定区域划分", ""]
    for name, members in partitions.items():
        lines.append(f"- **{name}**：" + "；".join("/".join(LEAD_NAMES[j] for j in group) for group in members))
    lines += ["", "### 完整三种子比较", "", "| 分组 | Val R² | Val Macro AUC | Test R²（描述性） | Test Macro AUC |", "|---|---:|---:|---:|---:|"]
    for name, b in partition_summary["partitions"].items():
        lines.append(f"| {name} | {b['val']['r2']['mean']:.6f} ± {b['val']['r2']['std_ddof1']:.6f} | {b['val']['macro_roc_auc']['mean']:.6f} | {b['test']['r2']['mean']:.6f} ± {b['test']['r2']['std_ddof1']:.6f} | {b['test']['macro_roc_auc']['mean']:.6f} |")
    med = partition_summary["partitions"]["medical"]
    lines += ["", "### 各分组 Val 四类别 R²", "", "| 分组 | OMI | NOMI | Clinical Control | Healthy | Overall |", "|---|---:|---:|---:|---:|---:|"]
    for name, b in partition_summary["partitions"].items():
        lines.append("| " + name + " | " + " | ".join([f"{b['val']['per_output_r2'][x]['mean']:.6f}" for x in LABEL_NAMES] + [f"{b['val']['r2']['mean']:.6f}"]) + " |")
    lines += ["", "### 医学分组 Test 四类别 R²", "",
              "| OMI | NOMI | Clinical Control | Healthy | Overall |", "|---:|---:|---:|---:|---:|",
              "| " + " | ".join([f"{med['test']['per_output_r2'][x]['mean']:.6f}" for x in LABEL_NAMES] + [f"{med['test']['r2']['mean']:.6f}"]) + " |", "",
              "## 实验二：E11 单区域移除（验证集）", "",
              "每次仅将一个区域槽的mask置为False并将其表示清零，其他重叠区域保持不变；Attention在剩余区域上重新归一化。负值表示移除后性能下降。", "",
              "| 重要性排序 | 区域 | ΔOverall R² | sd | ΔMacro AUC | OMI ΔR² | NOMI ΔR² | Control ΔR² | Healthy ΔR² |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, name in enumerate(ablation["importance_ranking_most_harmful_to_remove_first"], 1):
        x = ablation["aggregate"][name]
        lines.append(f"| {rank} | {name} | {x['delta_r2_mean']:+.6f} | {x['delta_r2_std_ddof1']:.6f} | {x['delta_auc_mean']:+.6f} | " + " | ".join(f"{x['per_output_r2_delta_mean'][k]:+.6f}" for k in LABEL_NAMES) + " |")
    lines += ["", "区域消融是模型依赖性分析，不能直接解释为医学因果效应。", "",
              "## 实验三：概率校准", "",
              "Temperature Scaling分别以验证集NLL和概率MSE拟合单个温度。Vector Scaling使用每类独立正缩放和中心化偏置，正则强度通过验证集内部的分层五折OOF Overall R²选择，再在完整验证集重拟合。每个seed的最终校准方法也只按Val Overall R²选择。", "",
              "| 方法 | Val R² | Test R² | Test Macro AUC | Test Brier | Test ECE(top) | Test ECE(macro OVR) |", "|---|---:|---:|---:|---:|---:|---:|"]
    display = {"uncalibrated": "未校准", "temperature_nll": "Temperature-NLL", "temperature_mse": "Temperature-MSE", "vector_nll": "Vector-NLL"}
    for method in CALIBRATION_METHODS:
        b = calibration["methods"][method]
        lines.append(f"| {display[method]} | {b['val']['r2']['mean']:.6f} | {b['test']['r2']['mean']:.6f} | {b['test']['macro_roc_auc']['mean']:.6f} | {b['test']['brier_score']['mean']:.6f} | {b['test']['ece15_top_label']['mean']:.6f} | {b['test']['ece15_macro_ovr']['mean']:.6f} |")
    lines += [f"| **每seed按Val R²选择** | **{selected['val']['r2']['mean']:.6f}** | **{selected['test']['r2']['mean']:.6f}** | **{selected['test']['macro_roc_auc']['mean']:.6f}** | **{selected['test']['brier_score']['mean']:.6f}** | **{selected['test']['ece15_top_label']['mean']:.6f}** | **{selected['test']['ece15_macro_ovr']['mean']:.6f}** |", "",
              "### 校准后 Test 四类别 R²", "", "| 方案 | OMI | NOMI | Clinical Control | Healthy | Overall |", "|---|---:|---:|---:|---:|---:|"]
    for method in CALIBRATION_METHODS:
        b = calibration["methods"][method]["test"]
        lines.append("| " + display[method] + " | " + " | ".join([f"{b['per_output_r2'][x]['mean']:.6f}" for x in LABEL_NAMES] + [f"{b['r2']['mean']:.6f}"]) + " |")
    b = selected["test"]
    lines.append("| 每seed按Val选择 | " + " | ".join([f"{b['per_output_r2'][x]['mean']:.6f}" for x in LABEL_NAMES] + [f"{b['r2']['mean']:.6f}"]) + " |")
    lines += ["", "## 是否进入 E12 Region-aware Token-level Fusion", "", "建议依据本轮结果：", "",
              "- 如果医学分组稳定超过全部或绝大多数度数匹配随机分组，且关键区域消融导致一致下降，则值得进入E12；这说明区域结构在record-level聚合之外仍有研究价值。",
              "- E12应先做冻结token级轻量cross-attention或区域token，仅在验证集比较；保留随机区域和全局token等参数对照。",
              "- Clinical Control测试样本仅101例，任何针对该类别的结论仍需更多样本或外部队列确认。", "",
              "## 实验纪律与局限", "", "- 固定patient split、seed0/1/2和E11训练配置；所有新checkpoint按Val Overall R²选择。",
              "- Encoder、Codebook、Sentence MLM全部冻结；历史E11 checkpoint未覆盖。",
              "- Test只在所有训练、正则和校准方法选择完成后评估；报告中的Test对比不用于反向选择。",
              "- 随机划分只有5组；可验证医学分组相对度数匹配随机结构的稳定性，但不能穷尽所有可能分组。", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def smoke(run_dir: Path, device: torch.device):
    run_dir.mkdir(parents=True, exist_ok=False)
    partitions = make_partitions(); medical = incidence(MEDICAL_MEMBERS)
    audit = {}
    for name, members in partitions.items():
        m = incidence(members)
        audit[name] = {"row_sums": m.sum(1).tolist(), "column_sums": m.sum(0).tolist(),
                       "members": [[LEAD_NAMES[j] for j in x] for x in members]}
        if not np.array_equal(m.sum(1), medical.sum(1)) or not np.array_equal(m.sum(0), medical.sum(0)):
            raise RuntimeError("Partition degree audit failed")
    lead = torch.randn(4, 9, 128, device=device)
    mask = torch.tensor([[1] * 9, [0] * 9, [1, 0, 1, 0, 1, 0, 1, 0, 1], [1] * 9], dtype=torch.bool, device=device)
    lead *= mask.unsqueeze(-1)
    for members in partitions.values():
        regions, region_mask = regions_for(lead, mask, members)
        if region_mask[1].any() or not torch.equal(regions[1], torch.zeros_like(regions[1])):
            raise RuntimeError("Empty ECG mask failed")
    logits = np.random.default_rng(0).normal(size=(40, 4))
    y = np.eye(4)[np.arange(40) % 4]
    t1 = fit_temperature(logits, y, "nll"); t2 = fit_temperature(logits, y, "mse")
    scale, bias = fit_vector(logits, y, 0.01)
    for p in [softmax_np(logits), softmax_np(logits / t1), softmax_np(logits / t2), apply_vector(logits, scale, bias)]:
        if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
            raise RuntimeError("Calibration smoke failed")
    dump(run_dir / "smoke.json", {"passed": True, "partition_audit": audit,
                                   "temperature_nll": t1, "temperature_mse": t2,
                                   "vector_scale": scale.tolist(), "vector_bias": bias.tolist(),
                                   "missing_and_empty_masks_passed": True})
    (run_dir / "COMPLETE").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=3.0)
    args = parser.parse_args(); args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_only:
        smoke(args.output_dir, device); return
    run_dir = args.output_dir; run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "pipeline.pid").write_text(str(os.getpid()))
    start = time.time(); partitions = make_partitions()
    partition_spec = {name: {"indices": members, "leads": [[LEAD_NAMES[j] for j in x] for x in members],
                             "row_sizes": incidence(members).sum(1).tolist(),
                             "lead_degrees": dict(zip(LEAD_NAMES, incidence(members).sum(0).astype(int).tolist()))}
                      for name, members in partitions.items()}
    protocol = {"experiment": EXP, "partitions": partition_spec, "random_partition_seeds": RANDOM_SEEDS,
                "partition_constraint": "exact row sizes and exact per-lead repetition degrees",
                "training": {"seeds": [0, 1, 2], "epochs": args.epochs, "batch_size": args.batch_size,
                             "lr": args.lr, "min_lr": args.min_lr, "warmup_epochs": args.warmup_epochs,
                             "weight_decay": args.weight_decay, "clip": args.clip,
                             "selection": "Val Overall R2 only", "test_once_after_selection": True},
                "region_ablation": "existing E11 checkpoints, validation only, mask one region slot",
                "calibration": {"source": "existing E11 val/test logits", "temperature_objectives": ["NLL", "probability MSE"],
                                "vector": "positive class scales + centered bias; L2 selected by 5-fold stratified OOF val R2",
                                "method_selection": "per-seed Val Overall R2"},
                "frozen": ["E3 MorphologyEncoder", "Codebook", "Sentence MLM"]}
    dump(run_dir / "protocol.json", protocol)
    sources = [TOKENIZER, MLM, ROOT / "scripts/curriculum_v1/run_e11_region_residual_fusion.py", Path(__file__)]
    fingerprints = {str(p): sha256(p) for p in sources}; dump(run_dir / "source_fingerprints.json", fingerprints)
    try:
        set_status(run_dir, "loading_and_auditing_e11_features")
        data, feature_audit = e11.load_features(device); dump(run_dir / "feature_audit.json", feature_audit)
        # Exact paired initialization across every partition for each training seed.
        init_audit = {}
        for seed in (0, 1, 2):
            hashes = []
            for _ in partitions:
                model = e11.E11RegionResidual(seed).to(device); e4.set_seed(seed); hashes.append(e11.state_sha(model)); del model
            if len(set(hashes)) != 1:
                raise RuntimeError(f"seed{seed}: partition initial states differ")
            init_audit[f"seed{seed}"] = hashes[0]
        dump(run_dir / "paired_initialization_audit.json", init_audit)

        set_status(run_dir, "experiment2_existing_e11_region_ablation_val")
        ablation = region_ablation(data, run_dir, device)
        set_status(run_dir, "experiment3_existing_e11_probability_calibration")
        calibration = calibrate(run_dir)
        results = []
        for partition, members in partitions.items():
            for seed in (0, 1, 2):
                set_status(run_dir, f"experiment1_training_{partition}_seed{seed}")
                results.append(train_partition(partition, members, seed, data, run_dir, args, device))
                dump(run_dir / "random_region_experiment/partial_results.json", results)
        partition_summary = aggregate_partitions(results)
        dump(run_dir / "random_region_experiment/aggregate_summary.json", partition_summary)
        write_csvs(run_dir, partition_summary, ablation, calibration)
        write_report(run_dir, partitions, partition_summary, ablation, calibration)
        if any(sha256(p) != fingerprints[str(p)] for p in sources):
            raise RuntimeError("Frozen source changed during experiment")
        dump(run_dir / "completion_audit.json", {"source_hashes_unchanged": True, "new_training_runs": len(results),
                                                   "region_ablation_split": "val", "calibration_fit_and_selection_split": "val",
                                                   "test_used_for_selection": False, "test_evaluations_per_training_run": 1,
                                                   "elapsed_seconds": time.time() - start})
        set_status(run_dir, "complete"); (run_dir / "exit_code.txt").write_text("0"); (run_dir / "COMPLETE").touch()
    except BaseException:
        set_status(run_dir, "failed"); (run_dir / "exit_code.txt").write_text("1"); raise


if __name__ == "__main__":
    main()
