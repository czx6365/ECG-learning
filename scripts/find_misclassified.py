"""Find 20 misclassified ECG samples and plot the original 9-lead waveforms."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 9-lead order matching the waveform data layout
LEAD_NAMES = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]


class FeatureMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FeatureMLP(input_dim=768, hidden_dim=512, dropout=0.1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def plot_full_ecg(waveform, sample_idx, true_label, pred_label, pred_prob, metadata_row, output_path):
    """Plot 9-lead full ECG waveform (no beat segmentation)."""
    waveform = np.asarray(waveform, dtype=np.float64)  # (9, 1000)
    n_leads = len(LEAD_NAMES)
    time_axis = np.arange(waveform.shape[1])

    fig, axes = plt.subplots(n_leads, 1, figsize=(18, 14), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        ax = axes[lead_idx]
        ax.plot(time_axis, waveform[lead_idx], color="tab:blue", linewidth=0.6)
        ax.set_ylabel(lead_name, fontsize=9, rotation=0, labelpad=20)
        ax.set_ylim(waveform[lead_idx].min() - 0.1, waveform[lead_idx].max() + 0.1)
        ax.tick_params(labelsize=7)
        if lead_idx == 0:
            ax.set_title(f"Sample #{sample_idx}  |  True: {int(true_label)}  |  "
                        f"Pred: {int(pred_label)} (p={pred_prob:.3f})  |  "
                        f"WRONG",
                        fontsize=11, fontweight="bold", color="red")

    axes[-1].set_xlabel("Time (samples)", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ckpt_path = Path("runs/compare_dt_vs_mlp/feature_mlp/feature_mlp_best.pth")
    feature_dir = Path("runs/compare_dt_vs_mlp/features")
    waveform_dir = Path("datasets/ecg_datasets/ECG_DSA/waveforms/base")
    output_dir = Path("outputs/misclassified_samples")
    # Clean and recreate output dir
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MLP model
    model = load_model(ckpt_path)

    # Load test features and targets
    test_features = np.load(feature_dir / "test_features.npy")
    test_targets = np.load(feature_dir / "test_targets.npy").reshape(-1)

    # Predict
    with torch.no_grad():
        logits = model(torch.from_numpy(test_features).float())
        probs = torch.sigmoid(logits).squeeze().numpy()
        preds = (probs >= 0.5).astype(np.int64)

    # Find misclassified
    misclassified = np.where(preds != test_targets)[0]
    n_fp = int(((preds == 1) & (test_targets == 0)).sum())
    n_fn = int(((preds == 0) & (test_targets == 1)).sum())
    print(f"Total test samples: {len(test_targets)}")
    print(f"Misclassified: {len(misclassified)} (FP={n_fp}, FN={n_fn})")
    print(f"Accuracy: {1 - len(misclassified)/len(test_targets):.4f}")

    # Load original full waveforms (9, 1000)
    wave_data = np.load(waveform_dir / "test_data.npy", mmap_mode="r")

    # Load metadata
    metadata = pd.read_csv(waveform_dir / "test_metadata.csv")

    # Pick 10 FN + 10 FP (or balanced as much as possible)
    fn_idx = misclassified[test_targets[misclassified] == 1]
    fp_idx = misclassified[test_targets[misclassified] == 0]
    rng = np.random.RandomState(42)
    n_fn_pick = min(10, len(fn_idx))
    n_fp_pick = min(10, len(fp_idx))
    selected_fn = rng.choice(fn_idx, size=n_fn_pick, replace=False) if n_fn_pick > 0 else np.array([])
    selected_fp = rng.choice(fp_idx, size=n_fp_pick, replace=False) if n_fp_pick > 0 else np.array([])

    selected_list = []
    for idx in selected_fn:
        selected_list.append((idx, "FN", int(test_targets[idx]), int(preds[idx]), float(probs[idx])))
    for idx in selected_fp:
        selected_list.append((idx, "FP", int(test_targets[idx]), int(preds[idx]), float(probs[idx])))
    rng.shuffle(selected_list)

    for i, (idx, err_type, true_lbl, pred_lbl, prob) in enumerate(selected_list):
        meta_row = metadata.iloc[int(idx)] if int(idx) < len(metadata) else None
        patient_info = ""
        if meta_row is not None:
            stenosis = meta_row.get("overall_max_stenosis", "N/A")
            patient_info = f" | stenosis={stenosis}"
        print(f"[{i+1:02d}] {err_type} sample {idx}: true={true_lbl}, pred={pred_lbl}, prob={prob:.3f}{patient_info}")

        plot_full_ecg(
            wave_data[int(idx)],
            int(idx),
            true_lbl,
            pred_lbl,
            prob,
            meta_row,
            output_dir / f"misclassified_{i+1:02d}_{err_type}_sample_{idx}.png",
        )

    print(f"\nSaved {len(selected_list)} images to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
