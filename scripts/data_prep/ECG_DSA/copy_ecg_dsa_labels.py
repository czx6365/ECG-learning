"""ECG-DSA QRS 数据集标签复制脚本。

这个工具脚本会把基础波形目录中的 train/val/test 标签文件
复制到指定的 QRS 数据集目录（例如过滤版或掩码版）。

适用场景：在使用 build_filtered_qrs_dataset.py 或
build_masked_qrs_dataset.py 生成新的 QRS 数据集后，
用它补齐对应的标签文件。

输入：来自 ECG_DSA_WAVEFORM_DIR 的 *_labels.npy 文件
输出：复制到 ECG_DSA_QRS_BASE_DIR 或自定义目标目录的标签文件
"""

import argparse
import shutil
from pathlib import Path
import sys

# 允许直接以 `python scripts/data_prep/ECG_DSA/copy_ecg_dsa_labels.py` 方式运行。
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import ECG_DSA_QRS_BASE_DIR, ECG_DSA_WAVEFORM_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy train/val/test label files into the ECG_DSA_QRS directory."
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=ECG_DSA_WAVEFORM_DIR,
        help="Directory containing *_labels.npy files.",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=ECG_DSA_QRS_BASE_DIR,
        help="Directory to receive copied label files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.target_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        source = args.source_dir / f"{split}_labels.npy"
        target = args.target_dir / source.name
        if not source.exists():
            raise FileNotFoundError(f"Missing label file: {source}")
        shutil.copy2(source, target)
        print(f"Copied {source} -> {target}")


if __name__ == "__main__":
    main()
