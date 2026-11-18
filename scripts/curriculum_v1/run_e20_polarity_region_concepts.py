#!/usr/bin/env python3
"""E20: polarity-aware, location-aware report supervision on exact E12-A downstream.

The four-class inference path and frozen E12-A representation are unchanged.
Only training-patient rows receive a training-only auxiliary concept loss.  The
parser distinguishes asserted, negated, and uncertain findings and maps explicit
lead spans (for example V1-V4) to the five existing E11 ECG regions.  Healthy,
Validation, and Test rows never receive or consume report supervision.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

import run_e12_continuous_input as e12
import run_e18b_downstream_text_auxiliary as e18b


E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
DEFAULT_RUN = ROOT / "results/e20_polarity_region_text_concepts"
CONCEPT_WEIGHT = 0.1
ABSENT_WEIGHT = 0.10
UNCERTAIN_WEIGHT = 0.35
SUGGESTIVE_WEIGHT = 0.60
REGIONS = (
    "global",
    "anteroseptal_v1_v3",
    "anterior_v3_v5",
    "anterolateral_i_v5_v6",
    "inferior_ii_iii",
    "precordial_v1_v6",
)

# Patterns are fixed before Validation is read.  They describe ECG-observable
# morphology or the ECG report's interpretation, not OMI/NOMI/Control labels.
MORPHOLOGIES = {
    "st_elevation": r"ST\s*(?:段)?[^;；。,.，]{0,14}(?:抬高|上抬|弓背(?:向上)?|上移)",
    "st_depression": r"ST\s*(?:段)?[^;；。,.，]{0,14}(?:压低|下移|降低)",
    "q_wave_or_qs": r"(?:(?:异常|病理|坏死性)\s*)?Q\s*波|QS\s*(?:波|型)?|呈\s*QS",
    "t_inversion_or_biphasic": r"T\s*波[^;；。,.，]{0,12}(?:倒置|负向|低平|正负双向|双向)",
    "st_t_change": r"ST\s*[-－— ]?\s*T\s*(?:变化|改变|异常)|ST\s*段与T\s*波异常|复极异常",
    "r_progression_poor": r"[rR]\s*波[^;；。,.，]{0,10}(?:递增不良|进展不良)",
    "mi_interpretation": r"心肌梗死|心梗|(?<![A-Z])STEMI(?![A-Z])|(?<![A-Z])MI(?![A-Z])",
    "ischemia_or_injury": r"心肌(?:缺血|损伤)|缺血性改变|损伤型\s*ST",
}
CONCEPT_NAMES = tuple(f"{m}__{r}" for m in MORPHOLOGIES for r in REGIONS)

NEGATION = re.compile(r"未见(?:明显|急性)?|未发现|未提示|无明显|不支持|排除|否认")
UNCERTAIN = re.compile(r"可能|考虑|疑似|不除外|倾向|待排|首先考虑|可考虑|不能除外|年龄不确定")
SUGGESTIVE = re.compile(r"提示|符合")
DYNAMIC_NO_CHANGE = re.compile(r"(?:无明显|未见明显|没有明显)(?:动态)?(?:变化|改变)|较前(?:无明显|未见明显)(?:变化|改变)")
WEAK_PHYSIOLOGIC = re.compile(r"正常范围内|早期复极|正常变异")

REGION_KEYWORDS = {
    "anteroseptal_v1_v3": ("前间壁", "间隔"),
    "anterior_v3_v5": ("前壁", "广泛前壁"),
    "anterolateral_i_v5_v6": ("侧壁", "高侧壁", "前侧壁"),
    "inferior_ii_iii": ("下壁",),
    "precordial_v1_v6": ("胸前", "胸导联", "左胸"),
}
REGION_LEADS = {
    "anteroseptal_v1_v3": {"V1", "V2", "V3"},
    "anterior_v3_v5": {"V3", "V4", "V5"},
    "anterolateral_i_v5_v6": {"I", "V5", "V6"},
    "inferior_ii_iii": {"II", "III"},
    "precordial_v1_v6": {"V1", "V2", "V3", "V4", "V5", "V6"},
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_status(run: Path, value: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "status.txt").write_text(value + "\n", encoding="utf-8")
    print(value, flush=True)


def normalize_text(text: str) -> str:
    table = str.maketrans({"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "－": "-", "—": "-", "～": "-", "~": "-", "：": ":"})
    return re.sub(r"\s+", " ", str(text).translate(table)).upper()


def expand_leads(fragment: str) -> set[str]:
    """Return only the nine leads that really exist in the HeartLang input."""
    text = normalize_text(fragment)
    leads: set[str] = set()
    for match in re.finditer(r"V([1-9])\s*(?:-|至)\s*V?([1-9])", text):
        a, b = map(int, match.groups())
        for number in range(min(a, b), max(a, b) + 1):
            if 1 <= number <= 6:
                leads.add(f"V{number}")
    for token in re.findall(r"V[1-9]|AVL|AVF|AVR|III|II|I", text):
        if token in {"I", "II", "III"} or (token.startswith("V") and int(token[1:]) <= 6):
            leads.add(token)
    return leads


def nearby_location(clause: str, start: int, end: int) -> tuple[set[str], set[str]]:
    before = clause[max(0, start - 55):start]
    # Prefer the closest phrase after a conjunction when that phrase contains a lead.
    parts = re.split(r"(?:伴|并|及|和|同时)", before)
    lead_fragment = parts[-1] if expand_leads(parts[-1]) else before
    leads = expand_leads(lead_fragment)
    vicinity = clause[max(0, start - 28):min(len(clause), end + 28)]
    regions = {name for name, words in REGION_KEYWORDS.items() if any(word.upper() in vicinity.upper() for word in words)}
    for name, members in REGION_LEADS.items():
        if leads & members:
            regions.add(name)
    return leads, regions


def event_status(clause: str, start: int, end: int, morphology: str) -> tuple[str, float, str]:
    window = clause[max(0, start - 18):min(len(clause), end + 20)]
    tail = clause[start:min(len(clause), end + 24)]
    if morphology == "st_t_change" and DYNAMIC_NO_CHANGE.search(tail):
        return "ignore_dynamic_stability", 0.0, window
    preceding = clause[max(0, start - 16):start]
    if NEGATION.search(preceding) or re.search(r"未见(?:明显|急性)?", window[: max(0, start - max(0, start - 18)) + 4]):
        return "negated", 1.0, window
    if WEAK_PHYSIOLOGIC.search(window):
        return "uncertain", UNCERTAIN_WEIGHT, window
    if UNCERTAIN.search(window):
        return "uncertain", UNCERTAIN_WEIGHT, window
    if morphology in {"mi_interpretation", "ischemia_or_injury"} and SUGGESTIVE.search(window):
        return "suggestive", SUGGESTIVE_WEIGHT, window
    return "asserted", 1.0, window


def parse_report(text: str) -> list[dict]:
    normalized = normalize_text(text)
    events: list[dict] = []
    offset = 0
    for raw_clause in re.split(r"[;；。\n]+", normalized):
        clause = raw_clause.strip()
        if not clause:
            offset += len(raw_clause) + 1
            continue
        for morphology, pattern in MORPHOLOGIES.items():
            for match in re.finditer(pattern, clause, flags=re.IGNORECASE):
                status, weight, context = event_status(clause, match.start(), match.end(), morphology)
                if weight == 0:
                    continue
                leads, regions = nearby_location(clause, match.start(), match.end())
                events.append({
                    "morphology": morphology,
                    "status": status,
                    "weight": weight,
                    "target": 0.0 if status == "negated" else 1.0,
                    "leads": sorted(leads),
                    "regions": sorted(regions),
                    "context": context,
                    "offset": offset + match.start(),
                })
        offset += len(raw_clause) + 1
    return events


def report_targets(text: str) -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]:
    events = parse_report(text)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        keys = [f"{event['morphology']}__global"]
        keys += [f"{event['morphology']}__{region}" for region in event["regions"]]
        for key in keys:
            buckets[key].append(event)
    targets = np.zeros(len(CONCEPT_NAMES), dtype=np.float32)
    weights = np.full(len(CONCEPT_NAMES), ABSENT_WEIGHT, dtype=np.float32)
    conflicts: list[str] = []
    for j, name in enumerate(CONCEPT_NAMES):
        mentions = buckets.get(name, [])
        if not mentions:
            continue
        values = {int(x["target"]) for x in mentions}
        if len(values) > 1:
            weights[j] = 0.0
            conflicts.append(name)
            continue
        targets[j] = float(next(iter(values)))
        weights[j] = max(float(x["weight"]) for x in mentions)
    return targets, weights, events, conflicts


def parser_self_check() -> dict:
    cases = {
        "asserted_v1_v4": "V1-V4导联ST段抬高0.1mV",
        "negated_inferior": "右胸及后壁导联未见明显ST段抬高;II III ST段压低",
        "uncertain_anterior": "V2-V5 ST段抬高，考虑前壁心肌损伤",
        "dynamic_not_negation": "V1-V4 ST段抬高;与前次相比ST-T无明显动态改变",
        "conflicting_history": "V1-V4 ST段抬高;V1-V4未见明显ST段抬高",
    }
    parsed = {}
    for name, text in cases.items():
        targets, weights, events, conflicts = report_targets(text)
        active = {CONCEPT_NAMES[i]: {"target": float(targets[i]), "weight": float(weights[i])}
                  for i in np.flatnonzero(weights > ABSENT_WEIGHT + 1e-6)}
        parsed[name] = {"events": events, "active": active, "conflicts": conflicts}
    required = "st_elevation__anteroseptal_v1_v3"
    if parsed["asserted_v1_v4"]["active"].get(required, {}).get("target") != 1.0:
        raise RuntimeError("Parser failed V1-V4 regional assertion")
    if not any(e["status"] == "negated" for e in parsed["negated_inferior"]["events"]):
        raise RuntimeError("Parser failed explicit negation")
    if not any(e["status"] == "uncertain" for e in parsed["uncertain_anterior"]["events"]):
        raise RuntimeError("Parser failed uncertainty")
    if not parsed["conflicting_history"]["conflicts"]:
        raise RuntimeError("Parser failed longitudinal conflict masking")
    return parsed


def build_concepts(run: Path) -> None:
    out = run / "concepts"
    if (out / "COMPLETE").exists():
        return
    e18b.build_manifest(run)
    out.mkdir(parents=True, exist_ok=False)
    pairs = pd.read_csv(run / "text_manifest/train_report_pairs.csv", low_memory=False)
    labels = np.load(e12.DOWNSTREAM / "train_labels.npy").argmax(1)
    n, c = len(labels), len(CONCEPT_NAMES)
    targets = np.zeros((n, c), dtype=np.float32)
    weights = np.zeros((n, c), dtype=np.float32)
    stats = Counter()
    examples = []
    lead_counts = Counter()
    conflict_counts = Counter()
    for row in pairs.itertuples(index=False):
        i = int(row.downstream_index)
        if labels[i] == 3:
            raise RuntimeError("Healthy record unexpectedly has a report")
        target, weight, events, conflicts = report_targets(str(row.report_text))
        targets[i], weights[i] = target, weight
        stats["patient_reports"] += 1
        stats["reports_with_event"] += bool(events)
        stats["reports_with_conflict"] += bool(conflicts)
        for event in events:
            stats[f"event_{event['status']}"] += 1
            stats["events_with_explicit_leads"] += bool(event["leads"])
            stats["events_with_region"] += bool(event["regions"])
            lead_counts.update(event["leads"])
            if len(examples) < 250 and (event["status"] != "asserted" or event["regions"]):
                examples.append({"downstream_index": i, **event})
        conflict_counts.update(conflicts)
    patient = labels != 3
    if not np.all(weights[patient].sum(1) > 0) or np.any(weights[~patient] != 0):
        raise RuntimeError("Concept weights must exist only for training patients")
    np.save(out / "train_targets.npy", targets)
    np.save(out / "train_weights.npy", weights)
    rows = []
    for j, name in enumerate(CONCEPT_NAMES):
        positive = targets[:, j] > 0.5
        supervised = weights[:, j] > 0
        item = {
            "concept": name,
            "positive": int((positive & supervised).sum()),
            "explicit_or_uncertain_positive_weight": float((weights[:, j] * positive).sum()),
            "supervised_weight": float(weights[:, j].sum()),
            "conflict_reports": int(conflict_counts[name]),
        }
        for class_id, class_name in enumerate(e12.LABEL_NAMES[:3]):
            m = labels == class_id
            item[f"{class_name}_positive"] = int((positive & m).sum())
        rows.append(item)
    pd.DataFrame(rows).to_csv(out / "concept_prevalence.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "parsed_event_examples.csv", index=False)
    checks = parser_self_check()
    dump(out / "parser_self_check.json", checks)
    dump(out / "audit.json", {
        "experiment": "E20 polarity/location-aware patient-only report concepts",
        "source": str(run / "text_manifest/train_report_pairs.csv"),
        "source_fields": "same-row diagnosis_1..6 composite only",
        "inference_architecture": "exact E12-A four-class E11RegionResidual path",
        "text_scope": "Train patient rows only; Healthy/Validation/Test receive no concept loss",
        "polarity": {
            "asserted_weight": 1.0, "negated_target_weight": 1.0,
            "uncertain_positive_weight": UNCERTAIN_WEIGHT,
            "suggestive_interpretation_weight": SUGGESTIVE_WEIGHT,
            "unmentioned_weak_negative_weight": ABSENT_WEIGHT,
            "contradictory_same_row_policy": "mask concept weight to zero",
        },
        "actual_ecg_leads": ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"],
        "regions": list(REGIONS), "morphologies": list(MORPHOLOGIES),
        "concept_count": len(CONCEPT_NAMES), "counts": dict(stats),
        "explicit_lead_counts": dict(lead_counts), "concepts": rows,
        "parser_self_check_passed": True, "validation_test_text_loaded": False,
    })
    if stats["event_negated"] == 0 or stats["event_uncertain"] == 0 or stats["events_with_region"] == 0:
        raise RuntimeError(f"Audit did not exercise polarity/location rules: {stats}")
    (out / "COMPLETE").touch()


class E20ConceptAux(e12.e11.E11RegionResidual):
    def __init__(self, seed: int):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(200000 + int(seed))
            self.concept_head = nn.Linear(128, len(CONCEPT_NAMES))

    def training_outputs_with_concepts(self, mlm, global_ze, regions, region_mask):
        x, _, _ = self.encode(mlm, global_ze, regions, region_mask)
        hidden = self.classifier.shared(x)
        return self.classifier.main(hidden), self.classifier.aux(hidden), self.concept_head(hidden)


def concept_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                 pos_weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none",
                                             pos_weight=pos_weight.float())
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def smoke(run: Path, device: torch.device) -> None:
    path = run / "smoke.json"
    if path.exists():
        return
    build_concepts(run)
    data = e18b.load_data(device)
    targets_np = np.load(run / "concepts/train_targets.npy")
    weights_np = np.load(run / "concepts/train_weights.npy")
    patient = np.flatnonzero(weights_np.sum(1) > 0)[:32]
    healthy = np.flatnonzero(weights_np.sum(1) == 0)[:32]
    idx_np = np.concatenate([patient, healthy])
    idx = torch.from_numpy(idx_np).long().to(device)
    targets = torch.from_numpy(targets_np[idx_np]).float().to(device)
    weights = torch.from_numpy(weights_np[idx_np]).float().to(device)
    model = E20ConceptAux(0).to(device)
    baseline = e12.e11.E11RegionResidual(0).to(device)
    common = baseline.state_dict(); state = model.state_dict()
    common_diff = max(float((state[k] - value).abs().max()) for k, value in common.items())
    if common_diff != 0:
        raise RuntimeError(f"E12-A initialization changed: {common_diff}")
    regions, region_mask = e12.e11.masked_region_means(data["train"]["lead"][idx], data["train"]["lead_mask"][idx])
    main, aux, concept = model.training_outputs_with_concepts(
        data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask)
    pos_weight = torch.ones(len(CONCEPT_NAMES), device=device)
    loss = concept_loss(concept, targets, weights, pos_weight)
    loss.backward()
    grads = {
        "shared": float(model.classifier.shared[0].weight.grad.norm()),
        "concept_head": float(model.concept_head.weight.grad.norm()),
    }
    model.eval(); baseline.eval()
    with torch.no_grad():
        base_logits = baseline(data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask)
        model_logits = model(data["train"]["mlm"][idx], data["train"]["global"][idx], regions, region_mask)
    result = {
        "common_initialization_max_abs_diff": common_diff,
        "initial_inference_logits_max_abs_diff": float((base_logits - model_logits).abs().max()),
        "concept_only_gradient_norms": grads,
        "patient_concept_weight": float(weights[:len(patient)].sum()),
        "healthy_concept_weight": float(weights[len(patient):].sum()),
        "loss_finite": bool(torch.isfinite(loss)), "validation_test_text_used": False,
    }
    if (not result["loss_finite"] or common_diff != 0 or result["initial_inference_logits_max_abs_diff"] != 0
            or result["healthy_concept_weight"] != 0 or not all(math.isfinite(x) and x > 0 for x in grads.values())):
        raise RuntimeError(f"Smoke failed: {result}")
    dump(path, result)


def aggregate(items: list[dict]) -> dict:
    keys = ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy")
    result = {key: {"mean": float(np.mean([row["val"][key] for row in items])),
                    "sd": float(np.std([row["val"][key] for row in items], ddof=1))} for key in keys}
    result["per_output_r2"] = {
        name: {"mean": float(np.mean([row["val"]["per_output_r2"][name] for row in items])),
               "sd": float(np.std([row["val"]["per_output_r2"][name] for row in items], ddof=1))}
        for name in e12.LABEL_NAMES
    }
    result["patient_mean_r2"] = {
        "mean": float(np.mean([np.mean([row["val"]["per_output_r2"][name] for name in e12.LABEL_NAMES[:3]]) for row in items])),
        "sd": float(np.std([np.mean([row["val"]["per_output_r2"][name] for name in e12.LABEL_NAMES[:3]]) for row in items], ddof=1)),
    }
    return result


def train_seed(run: Path, seed: int, device: torch.device) -> dict:
    out = run / "downstream" / f"seed{seed}"
    if (out / "VAL_COMPLETE").exists():
        return json.loads((out / "val_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    data = e18b.load_data(device)
    target = data["train"]["labels"].argmax(1)
    concept_target = torch.from_numpy(np.load(run / "concepts/train_targets.npy")).float().to(device)
    concept_weight = torch.from_numpy(np.load(run / "concepts/train_weights.npy")).float().to(device)
    positive_weight = (concept_weight * concept_target).sum(0)
    negative_weight = (concept_weight * (1 - concept_target)).sum(0)
    pos_weight = (negative_weight / positive_weight.clamp_min(1e-6)).clamp(1.0, 10.0)
    model = E20ConceptAux(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    steps = len(target) // 64
    generator = torch.Generator().manual_seed(seed)
    regions, region_mask = e12.e11.masked_region_means(data["train"]["lead"], data["train"]["lead_mask"])
    history, best, best_epoch, best_state = [], -float("inf"), None, None
    for epoch in range(1, 31):
        model.train(); order = torch.randperm(len(target), generator=generator).to(device)
        sums = {"main": 0.0, "patient": 0.0, "concept": 0.0, "batches": 0}
        for step in range(steps):
            idx = order[step * 64:(step + 1) * 64]
            lr = e12.lr_at((epoch - 1) * steps + step, 30 * steps, 5 * steps, 5e-4, 1e-5, 1e-6)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            main, aux, concepts = model.training_outputs_with_concepts(
                data["train"]["mlm"][idx], data["train"]["global"][idx], regions[idx], region_mask[idx])
            main_loss = F.cross_entropy(main, target[idx])
            patient_loss = e12.e9.patient_auxiliary_loss(aux, target[idx])
            c_loss = concept_loss(concepts, concept_target[idx], concept_weight[idx], pos_weight)
            loss = main_loss + 0.2 * patient_loss + CONCEPT_WEIGHT * c_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"seed{seed}/epoch{epoch}: non-finite loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
            sums["main"] += float(main_loss); sums["patient"] += float(patient_loss)
            sums["concept"] += float(c_loss); sums["batches"] += 1
        val, _, _ = e12.eval_downstream(model, "e12a_region", data["val"])
        row = {"epoch": epoch, "train_main_ce": sums["main"] / sums["batches"],
               "train_patient_ce": sums["patient"] / sums["batches"],
               "train_concept_bce": sums["concept"] / sums["batches"], "val": val}
        history.append(row); print(json.dumps({"seed": seed, **row}, ensure_ascii=False), flush=True)
        if val["r2"] > best:
            best, best_epoch, best_state = val["r2"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val, prob, logits = e12.eval_downstream(model, "e12a_region", data["val"])
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "seed": seed,
                "best_epoch": best_epoch, "selection": "Val Overall R2", "test_run": False},
               out / "checkpoint-best.pth")
    np.save(out / "val_logits.npy", logits); np.save(out / "val_probabilities.npy", prob)
    dump(out / "history.json", history)
    result = {"method": "e20_polarity_region_concepts", "seed": seed, "best_epoch": best_epoch,
              "val": val, "test_run": False, "concept_weight": CONCEPT_WEIGHT,
              "inference_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                    if not n.startswith("concept_head")),
              "training_only_concept_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                      if n.startswith("concept_head"))}
    dump(out / "val_summary.json", result); (out / "VAL_COMPLETE").touch()
    return result


def train_all(run: Path, device: torch.device) -> None:
    path = run / "validation_summary.json"
    if path.exists():
        return
    rows = [train_seed(run, seed, device) for seed in (0, 1, 2)]
    baseline = json.loads((E12_RUN / "downstream/validation_selection.json").read_text())["methods"]["e12a_region"]
    dump(path, {"validation_only": True, "test_run": False,
                "e12a_historical": {"aggregate": aggregate(baseline["per_seed"]), "per_seed": baseline["per_seed"]},
                "e20_text_concepts": {"aggregate": aggregate(rows), "per_seed": rows}})


def report(run: Path) -> None:
    summary = json.loads((run / "validation_summary.json").read_text())
    base = summary["e12a_historical"]["aggregate"]
    model = summary["e20_text_concepts"]["aggregate"]
    rows = []
    for label, key in (("E12-A", "e12a_historical"), ("E20", "e20_text_concepts")):
        for item in summary[key]["per_seed"]:
            val = item["val"]
            rows.append({"method": label, "seed": item.get("seed", item.get("downstream_seed")),
                         "overall_r2": val["r2"],
                         **{f"{name.lower().replace(' ', '_')}_r2": val["per_output_r2"][name]
                            for name in e12.LABEL_NAMES},
                         "macro_auc": val["macro_roc_auc"], "macro_f1": val["macro_f1"], "mse": val["mse"]})
    pd.DataFrame(rows).to_csv(run / "metrics_per_seed.csv", index=False)
    delta = model["r2"]["mean"] - base["r2"]["mean"]
    lines = [
        "# E20：否定与不确定性感知的导联/区域报告概念监督", "", "## 设计", "",
        "- 完整保留E12-A冻结表示、E11区域残差、原始四分类主头和患者CE3辅助头。",
        "- 文本只在Train患者内部作为辅助监督；Healthy、Validation、Test不读取文本。",
        "- 同一Excel行的诊断结论1–6仍拼接给该行全部ECG。",
        "- 区分肯定、明确否定、可能/考虑/不除外、提示；同一行历史描述冲突时屏蔽该概念。",
        "- 将V1–V4等显式导联范围映射到E11现有五个ECG区域；只接受实际输入的I/II/III/V1–V6。",
        "- 总损失：CE4 + 0.2×患者CE3 + 0.1×患者报告概念BCE。",
        "- 每个seed仅按Val Overall R²选择checkpoint；未运行Test。", "", "## Validation", "",
        "| 方法 | Overall R² | Patient mean R² | OMI | NOMI | Control | Healthy | Macro AUC | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| E12-A | {base['r2']['mean']:.6f} ± {base['r2']['sd']:.6f} | {base['patient_mean_r2']['mean']:.6f} | " +
        " | ".join(f"{base['per_output_r2'][name]['mean']:.6f}" for name in e12.LABEL_NAMES) +
        f" | {base['macro_roc_auc']['mean']:.6f} | {base['macro_f1']['mean']:.6f} |",
        f"| E20 | {model['r2']['mean']:.6f} ± {model['r2']['sd']:.6f} | {model['patient_mean_r2']['mean']:.6f} | " +
        " | ".join(f"{model['per_output_r2'][name]['mean']:.6f}" for name in e12.LABEL_NAMES) +
        f" | {model['macro_roc_auc']['mean']:.6f} | {model['macro_f1']['mean']:.6f} |", "",
        f"Overall R²差值：**{delta:+.6f}**。",
        f"患者三类平均R²差值：**{model['patient_mean_r2']['mean'] - base['patient_mean_r2']['mean']:+.6f}**。",
        f"OMI / NOMI / Control差值：**{model['per_output_r2']['OMI']['mean'] - base['per_output_r2']['OMI']['mean']:+.6f} / "
        f"{model['per_output_r2']['NOMI']['mean'] - base['per_output_r2']['NOMI']['mean']:+.6f} / "
        f"{model['per_output_r2']['Clinical Control']['mean'] - base['per_output_r2']['Clinical Control']['mean']:+.6f}**。", "",
    ]
    (run / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "smoke", "train", "report"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args(); run = args.output_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "audit":
        set_status(run, "audit_polarity_region_concepts"); build_concepts(run)
    elif args.stage == "smoke":
        set_status(run, "smoke_exact_e12_inference_and_text_gradients"); smoke(run, device)
    elif args.stage == "train":
        set_status(run, "train_e20_seeds_0_1_2"); train_all(run, device)
    else:
        report(run); set_status(run, "complete")


if __name__ == "__main__":
    main()
