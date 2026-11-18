import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import run_e20_polarity_region_concepts as e20
import run_e18_chief_complaint as chief

ROOT = e20.ROOT / "results/e29_chief_symptoms"
PATTERNS = ("胸痛|胸部疼痛|心前区疼痛", "胸闷", "气促|呼吸困难|气短", "心悸|心慌", "出汗|大汗")
e20.CONCEPT_NAMES = PATTERNS

def parse(text):
    y, w = np.zeros(len(PATTERNS), np.float32), np.zeros(len(PATTERNS), np.float32)
    for j, pattern in enumerate(PATTERNS):
        states = set()
        for clause in re.split(r"[，,。；;、]", text):
            for match in re.finditer(pattern, clause):
                prefix = clause[:match.start()]
                if re.search(r"可能|疑似|不详|不除外|待查", clause):
                    continue
                states.add(0 if re.search(r"否认|未见|无明显|无|未诉", prefix) else 1)
        if len(states) == 1:
            y[j], w[j] = states.pop(), 1
    return y, w

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    e20.set_status(ROOT, "audit")
    tab = pd.read_excel(chief.LABEL_XLSX, usecols=["file_name", "patient_sn", "主诉"])
    meta = pd.read_csv(e20.e12.DOWNSTREAM / "train_metadata.csv", dtype={"patient_sn": str})
    y = np.load(e20.e12.DOWNSTREAM / "train_labels.npy").argmax(1)
    mappings = {}
    multi = set()
    for row in tab.itertuples(index=False, name=None):
        fields = dict(zip(tab.columns, row))
        files = [Path(f.strip()).stem for f in str(fields["file_name"]).split(";") if f.strip()]
        if len(files) != 1:
            multi.update(files)
            continue
        if pd.notna(fields["主诉"]):
            mappings.setdefault(files[0], set()).add((str(fields["patient_sn"]), str(fields["主诉"])))
    targets = np.zeros((len(y),len(PATTERNS)),np.float32)
    weights = np.zeros_like(targets)
    for i, row in enumerate(meta.itertuples()):
        stem = Path(row.csv_name).stem
        matches = mappings.get(stem, set())
        if y[i] != 3 and stem not in multi and len(matches) == 1:
            patient, text = next(iter(matches))
            if patient == str(row.patient_sn):
                targets[i], weights[i] = parse(text)
    eligible = np.flatnonzero(weights.any(1))
    assert len(eligible) > 0, "No usable patient symptom labels"
    assert not weights[y == 3].any()
    e20.dump(ROOT/"audit.json", dict(eligible=len(eligible), positive=(targets*weights).sum(0).tolist(), negative=((1-targets)*weights).sum(0).tolist(), mapping="unambiguous single-ECG row and matching patient ID; encounter timing unverified"))
    device = torch.device("cuda")
    results = {}
    for arm in ("baseline", "real", "shuffled"):
        run = ROOT/arm
        path = run/"concepts"
        path.mkdir(parents=True, exist_ok=True)
        t, w = targets.copy(), weights.copy()
        if arm == "shuffled":
            order = np.random.default_rng(29).permutation(eligible)
            t[eligible], w[eligible] = targets[order], weights[order]
        np.save(path/"train_targets.npy",t)
        np.save(path/"train_weights.npy",w)
        e20.CONCEPT_WEIGHT = 0 if arm == "baseline" else .1
        e20.set_status(ROOT, arm)
        rows = [e20.train_seed(run, seed, device) for seed in range(3)]
        results[arm] = {"aggregate":e20.aggregate(rows),"per_seed":rows}
        e20.dump(ROOT/"validation_summary.json",results)
    e20.set_status(ROOT,"complete")

if __name__ == "__main__":
    main()
