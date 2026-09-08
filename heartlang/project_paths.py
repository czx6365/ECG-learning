"""HeartLang 项目内部统一使用的路径常量。"""

from pathlib import Path
from typing import Union


RAW_DATA_ROOT = Path("datasets/raw")
PROCESSED_DATA_ROOT = Path("datasets/ecg_datasets")
RESULTS_ROOT = Path("results")

ECG_DSA_RAW_SOURCE_DIR = RAW_DATA_ROOT / "ECG_DSA"
ECG_DSA_WAVEFORM_DIR = PROCESSED_DATA_ROOT / "ECG_DSA" / "waveforms" / "base"
ECG_DSA_QRS_ROOT = PROCESSED_DATA_ROOT / "ECG_DSA" / "qrs"
ECG_DSA_QRS_BASE_DIR = ECG_DSA_QRS_ROOT / "base"
ECG_DSA_QRS_VARIANTS_DIR = ECG_DSA_QRS_ROOT / "variants"
ECG_DSA_QRS_ALLNOISE_CLEAN_DIR = ECG_DSA_QRS_VARIANTS_DIR / "allnoise_clean"
ECG_DSA_QRS_ALLNOISE_MASKED_DIR = ECG_DSA_QRS_VARIANTS_DIR / "allnoise_masked"
ECG_DSA_QRS_NONFLAT_CLEAN_DIR = ECG_DSA_QRS_VARIANTS_DIR / "nonflat_clean"

RESULTS_METRICS_DIR = RESULTS_ROOT / "metrics"
RESULTS_PREDICTIONS_DIR = RESULTS_ROOT / "predictions"
RESULTS_ECG_DSA_PREDICTIONS_DIR = RESULTS_PREDICTIONS_DIR / "ecg_dsa"
RESULTS_EVALUATIONS_DIR = RESULTS_ROOT / "evaluations"
RESULTS_ECG_DSA_EVALUATIONS_DIR = RESULTS_EVALUATIONS_DIR / "ecg_dsa"
RESULTS_ANALYSIS_DIR = RESULTS_ROOT / "analysis"
RESULTS_ECG_DSA_ANALYSIS_DIR = RESULTS_ANALYSIS_DIR / "ecg_dsa"
RESULTS_ECG_DSA_CHANNEL_SCAN_DIR = RESULTS_ECG_DSA_ANALYSIS_DIR / "channel_scan"
RESULTS_ECG_DSA_RANDOM_QRS_SAMPLES_DIR = (
    RESULTS_ECG_DSA_ANALYSIS_DIR / "random_qrs_samples"
)
RESULTS_DEBUG_DIR = RESULTS_ROOT / "debug"
RESULTS_QRS_TRIM_DEBUG_DIR = RESULTS_DEBUG_DIR / "qrs_trim"


def dataset_path_slug(dataset_dir: Union[str, Path]) -> str:
    """把数据集目录压缩成适合文件名的短标识。"""
    parts = [
        part
        for part in Path(dataset_dir).parts
        if part not in {"", ".", "datasets", "ecg_datasets"}
    ]
    if not parts:
        return "dataset"
    return "_".join(parts[-3:])
