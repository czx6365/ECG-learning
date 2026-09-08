"""ECG ZIP 处理脚本的共享工具函数。

这个模块抽取了多个 ECG 数据准备脚本都会用到的公共逻辑：
- make_ecg_dsa_dataset.py：ECG-DSA 主流程中的有标签数据集构建
- prepare_ecg_zip_waveforms.py：不做数据划分的无标签波形抽取
- prepare_ecg_zip_pretrain_batch.py：带 train/val 划分的无标签预训练数据准备

主要函数：
- build_member_table：扫描 ZIP 文件并生成 CSV 成员元数据表
- resample_waveforms：使用 polyphase filtering 将波形重采样到目标长度
- load_waveform：从 ZIP 成员中读取并重采样单条波形
- standardize_in_place：按块对 .npy 数组原地做 z-score 标准化

常量：
- CSV_LEADS：标准 9 导联 ECG 配置 ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
"""

import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


CSV_LEADS = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]


def build_member_table(zip_paths: list[Path]) -> pd.DataFrame:
    """Scan ZIP files for CSV members and build a metadata table.

    Args:
        zip_paths: List of ZIP file paths to scan

    Returns:
        DataFrame with columns: csv_name, member_name, source_zip, source_zip_path

    Raises:
        ValueError: If duplicate CSV basenames are found across ZIPs
        RuntimeError: If no CSV files are found
    """
    rows = []
    seen = {}
    duplicates = {}

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zip_file:
            for member_name in zip_file.namelist():
                if not member_name.lower().endswith(".csv"):
                    continue
                if member_name.startswith("__MACOSX/"):
                    continue
                basename = Path(member_name).name
                if basename in seen:
                    duplicates.setdefault(basename, [seen[basename]]).append(zip_path.name)
                    continue
                seen[basename] = zip_path.name
                rows.append(
                    {
                        "csv_name": basename,
                        "member_name": member_name,
                        "source_zip": zip_path.name,
                        "source_zip_path": str(zip_path),
                    }
                )

    if duplicates:
        preview = {
            basename: zip_names
            for basename, zip_names in list(duplicates.items())[:10]
        }
        raise ValueError(
            "Found duplicate CSV basenames across ZIP inputs. "
            f"Preview: {preview}"
        )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise RuntimeError("No ECG CSV files were found inside the provided ZIP inputs.")
    return metadata.sort_values(["source_zip", "csv_name"]).reset_index(drop=True)


def resample_waveforms(waveforms: np.ndarray, target_length: int) -> np.ndarray:
    """Resample ECG waveforms to target length using polyphase filtering.

    Args:
        waveforms: Input waveforms array of shape (time_steps, channels)
        target_length: Desired number of time steps

    Returns:
        Resampled waveforms of shape (target_length, channels)
    """
    if waveforms.shape[0] == target_length:
        return waveforms

    source_length = waveforms.shape[0]
    gcd = math.gcd(source_length, target_length)
    up = target_length // gcd
    down = source_length // gcd
    resampled = resample_poly(
        waveforms,
        up=up,
        down=down,
        axis=0,
        padtype="line",
    ).astype(np.float32, copy=False)

    if resampled.shape[0] > target_length:
        return resampled[:target_length]
    if resampled.shape[0] < target_length:
        pad_rows = target_length - resampled.shape[0]
        return np.pad(resampled, ((0, pad_rows), (0, 0)), mode="edge")
    return resampled


def load_waveform(zip_file: zipfile.ZipFile, member_name: str, target_length: int) -> np.ndarray:
    """Load and resample a single ECG waveform from a ZIP file.

    Args:
        zip_file: Open ZipFile object
        member_name: Path to CSV member within the ZIP
        target_length: Desired number of time steps after resampling

    Returns:
        Waveform array of shape (channels, target_length)
    """
    with zip_file.open(member_name) as handle:
        data_frame = pd.read_csv(handle, usecols=CSV_LEADS)

    waveforms = data_frame[CSV_LEADS].to_numpy(dtype=np.float32, copy=True)
    waveforms = resample_waveforms(waveforms, target_length)
    return waveforms.T.astype(np.float32, copy=False)


def standardize_in_place(data_path: Path, mean: float, std: float, chunk_size: int = 256):
    """Standardize data in-place using memmap for memory efficiency.

    Args:
        data_path: Path to .npy file to standardize
        mean: Mean value for standardization
        std: Standard deviation for standardization
        chunk_size: Number of samples to process per chunk
    """
    array = np.load(data_path, mmap_mode="r+")
    for start in range(0, len(array), chunk_size):
        end = min(start + chunk_size, len(array))
        array[start:end] = (array[start:end] - mean) / std
    array.flush()
    del array
