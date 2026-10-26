#!/usr/bin/env python3
"""Create a stratified visual audit of rule-derived ECG physiology labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visual audit for expanded ECG physiology targets")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--phys_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--per_group", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--plots_per_page", type=int, default=10)
    return parser.parse_args()


def choose_spread(candidates: np.ndarray, values: np.ndarray, count: int, rng, excluded: set[int]):
    candidates = candidates[~np.isin(candidates, np.fromiter(excluded, dtype=np.int64))]
    if candidates.size < count:
        raise ValueError(f"Only {candidates.size} unused candidates for requested count={count}")
    order = candidates[np.argsort(values[candidates], kind="stable")]
    bins = np.array_split(order, count)
    selected = np.asarray([rng.choice(group) for group in bins], dtype=np.int64)
    rng.shuffle(selected)
    return selected


def choose_random(candidates: np.ndarray, count: int, rng, excluded: set[int]):
    candidates = candidates[~np.isin(candidates, np.fromiter(excluded, dtype=np.int64))]
    if candidates.size < count:
        raise ValueError(f"Only {candidates.size} unused candidates for requested count={count}")
    return rng.choice(candidates, size=count, replace=False).astype(np.int64)


def category_name(schema: dict, feature: str, value: int) -> str:
    if value < 0:
        return "ignored"
    return schema["discrete_features"][feature][value]


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    phys_dir = args.phys_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads((phys_dir / "physiology_feature_schema.json").read_text(encoding="utf-8"))
    continuous_names = schema["continuous_features"]
    discrete_names = list(schema["discrete_features"])
    continuous_index = {name: idx for idx, name in enumerate(continuous_names)}
    discrete_index = {name: idx for idx, name in enumerate(discrete_names)}

    split = args.split
    waves = np.load(data_dir / f"{split}_data.npy", mmap_mode="r")
    chans = np.load(data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(data_dir / f"{split}_data_in_times.npy", mmap_mode="r")
    continuous = np.load(phys_dir / f"{split}_phys_continuous.npy", mmap_mode="r")
    continuous_mask = np.load(phys_dir / f"{split}_phys_continuous_mask.npy", mmap_mode="r")
    discrete = np.load(phys_dir / f"{split}_phys_discrete.npy", mmap_mode="r")
    landmarks = np.load(phys_dir / f"{split}_phys_landmarks.npy", mmap_mode="r")

    n_records, n_tokens, token_len = waves.shape
    flat_discrete = discrete.reshape(-1, discrete.shape[-1])
    flat_continuous = continuous.reshape(-1, continuous.shape[-1])
    valid = flat_discrete[:, discrete_index["r_polarity"]] >= 0
    qrs_labels = flat_discrete[:, discrete_index["qrs_type"]]
    st_labels = flat_discrete[:, discrete_index["st_state"]]
    t_labels = flat_discrete[:, discrete_index["t_polarity"]]
    qrs_categories = schema["discrete_features"]["qrs_type"]
    st_categories = schema["discrete_features"]["st_state"]
    t_categories = schema["discrete_features"]["t_polarity"]

    wide_id = qrs_categories.index("wide")
    elevation_id = st_categories.index("elevation")
    biphasic_id = t_categories.index("biphasic")
    normal_st_id = st_categories.index("normal")
    all_indices = np.flatnonzero(valid)
    wide_candidates = np.flatnonzero(valid & (qrs_labels == wide_id))
    elevation_candidates = np.flatnonzero(valid & (st_labels == elevation_id))
    biphasic_candidates = np.flatnonzero(valid & (t_labels == biphasic_id))
    control_candidates = np.flatnonzero(
        valid & (qrs_labels != wide_id) & (st_labels == normal_st_id) & (t_labels != biphasic_id)
    )

    rng = np.random.default_rng(args.seed)
    excluded: set[int] = set()
    groups = []
    duration = flat_continuous[:, continuous_index["qrs_duration_ms"]]
    st_deviation = flat_continuous[:, continuous_index["st_deviation"]]
    t_amplitude = np.abs(flat_continuous[:, continuous_index["t_amplitude"]])
    selection_plan = [
        ("wide", wide_candidates, duration, True),
        ("st_elevation", elevation_candidates, st_deviation, True),
        ("t_biphasic", biphasic_candidates, t_amplitude, True),
        ("control", control_candidates, None, False),
    ]
    for group_name, candidates, values, spread in selection_plan:
        if spread:
            picked = choose_spread(candidates, values, args.per_group, rng, excluded)
        else:
            picked = choose_random(candidates, args.per_group, rng, excluded)
        excluded.update(int(item) for item in picked)
        groups.extend((group_name, int(item)) for item in picked)
    rng.shuffle(groups)

    rows = []
    for audit_id, (group, flat_idx) in enumerate(groups, start=1):
        record_idx, token_idx = divmod(flat_idx, n_tokens)
        disc = np.asarray(discrete[record_idx, token_idx])
        cont = np.asarray(continuous[record_idx, token_idx])
        mask = np.asarray(continuous_mask[record_idx, token_idx])
        lm = np.asarray(landmarks[record_idx, token_idx])
        row = {
            "audit_id": audit_id,
            "group": group,
            "split": split,
            "record_index": record_idx,
            "token_index": token_idx,
            "lead_id": int(chans[record_idx, token_idx]),
            "time_block": int(times[record_idx, token_idx]),
        }
        row.update({name: category_name(schema, name, int(disc[idx])) for idx, name in enumerate(discrete_names)})
        for idx, name in enumerate(continuous_names):
            row[name] = float(cont[idx]) if mask[idx] else ""
        for idx, name in enumerate(schema["landmarks"]):
            row[name] = int(lm[idx])
        rows.append(row)

    csv_path = output_dir / f"{split}_physiology_audit_100.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pdf_path = output_dir / f"{split}_physiology_audit_100.pdf"
    palette = {
        "wide": "#d62728",
        "st_elevation": "#ff7f0e",
        "t_biphasic": "#9467bd",
        "control": "#2ca02c",
    }
    per_page = args.plots_per_page
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(rows), per_page):
            page_rows = rows[page_start : page_start + per_page]
            fig, axes = plt.subplots(5, 2, figsize=(16, 19), constrained_layout=True)
            axes = axes.ravel()
            for ax, row in zip(axes, page_rows):
                wave = np.asarray(waves[row["record_index"], row["token_index"]], dtype=np.float32)
                edge_n = min(10, max(1, token_len // 4))
                baseline = float(np.median(np.concatenate([wave[:edge_n], wave[-edge_n:]])))
                signal = wave - baseline
                x_ms = np.arange(token_len) / args.fs * 1000.0
                onset = row["qrs_onset_index"]
                r_peak = row["r_peak_index"]
                offset = row["qrs_offset_index"]
                t_peak = row["t_peak_index"]
                st_start = min(token_len - 1, offset + int(round(0.06 * args.fs)))
                st_end = min(token_len, offset + int(round(0.12 * args.fs)))
                t_start = min(token_len - 1, r_peak + int(round(0.16 * args.fs)))
                t_end = min(token_len, r_peak + int(round(0.40 * args.fs)))

                ax.axhline(0.0, color="#888888", linewidth=0.7)
                ax.axhline(0.10, color="#ff7f0e", linewidth=0.6, linestyle=":", alpha=0.7)
                ax.axhline(-0.10, color="#ff7f0e", linewidth=0.6, linestyle=":", alpha=0.7)
                if st_end > st_start:
                    ax.axvspan(x_ms[st_start], x_ms[st_end - 1], color="#ffbb78", alpha=0.24)
                if t_end > t_start:
                    ax.axvspan(x_ms[t_start], x_ms[t_end - 1], color="#c5b0d5", alpha=0.18)
                ax.plot(x_ms, signal, color="#111111", linewidth=1.2)
                for pos, color, style, label in [
                    (onset, "#1f77b4", "--", "QRS on"),
                    (r_peak, "#d62728", "-", "R"),
                    (offset, "#17becf", "--", "QRS off"),
                    (t_peak, "#9467bd", ":", "T peak"),
                ]:
                    if 0 <= pos < token_len:
                        ax.axvline(x_ms[pos], color=color, linewidth=1.0, linestyle=style, alpha=0.9)
                ax.set_title(
                    f"#{row['audit_id']:03d} {row['group']} | rec={row['record_index']} tok={row['token_index']} "
                    f"lead={row['lead_id']} time={row['time_block']}\n"
                    f"QRS={row['qrs_type']} {row['qrs_duration_ms']:.0f}ms | "
                    f"ST={row['st_state']} {row['st_deviation']:+.3f} | "
                    f"T={row['t_polarity']} amp={row['t_amplitude']:+.3f}",
                    fontsize=9,
                    color=palette[row["group"]],
                )
                ax.set_xlim(0, x_ms[-1])
                ax.grid(alpha=0.14)
                ax.tick_params(labelsize=7)
                ax.set_ylabel("centered ECG", fontsize=7)
            for ax in axes[len(page_rows) :]:
                ax.axis("off")
            fig.suptitle(
                f"ECG physiology audit — {split} — samples {page_start + 1}–{page_start + len(page_rows)}\n"
                "blue/cyan=QRS onset/offset, red=R, orange=ST window, purple=T window/peak",
                fontsize=14,
            )
            page_number = page_start // per_page + 1
            page_path = pages_dir / f"audit_page_{page_number:02d}.png"
            fig.savefig(page_path, dpi=160)
            pdf.savefig(fig)
            plt.close(fig)

    summary = {
        "split": split,
        "seed": args.seed,
        "sample_count": len(rows),
        "groups": {name: sum(row["group"] == name for row in rows) for name in palette},
        "candidate_counts": {
            "all_valid": int(all_indices.size),
            "wide": int(wide_candidates.size),
            "st_elevation": int(elevation_candidates.size),
            "t_biphasic": int(biphasic_candidates.size),
            "control": int(control_candidates.size),
        },
        "artifacts": {
            "pdf": str(pdf_path),
            "csv": str(csv_path),
            "pages": str(pages_dir),
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
