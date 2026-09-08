"""把连续 ECG 波形切分成 QRS token 序列的预处理脚本。"""

from wfdb import processing
import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# 允许直接以 `python heartlang/preprocessing/qrs_tokenizer.py` 方式运行。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import ECG_DSA_QRS_BASE_DIR, ECG_DSA_WAVEFORM_DIR


ECG_DSA_CHANNEL_IDS = [1, 2, 3, 7, 8, 9, 10, 11, 12]


class QRSTokenizer(nn.Module):
    """QRS 分词器：检测 R 峰、截取 beat，并整理成固定长度 token 序列。"""

    def __init__(
        self,
        fs,
        max_len,
        token_len,
        save_path,
        stage,
        used_channels,
        channel_ids=None,
    ):
        super().__init__()
        self.fs = fs
        self.max_len = max_len
        self.token_len = token_len
        self.save_path = save_path
        self.device = None
        self.stage = stage
        self.used_channels = list(used_channels)
        if channel_ids is None:
            channel_ids = [channel + 1 for channel in self.used_channels]
        if len(channel_ids) != len(self.used_channels):
            raise ValueError("channel_ids must have the same length as used_channels.")
        self.channel_ids = list(channel_ids)

    def qrs_detection(self, ecg_signal):
        # 用第一导联检测 R 峰位置。
        lead_signal = ecg_signal[0, :]
        qrs_inds = processing.xqrs_detect(sig=lead_signal, fs=self.fs, verbose=False)
        return qrs_inds

    def trim_edge_qrs(self, qrs_inds):
        # 去掉首尾不完整 beat，减少边界误差。
        if len(qrs_inds) <= 2:
            return qrs_inds[:0]
        return qrs_inds[1:-1]

    def extract_qrs_segments(self, ecg_signal, qrs_inds):
        # 按相邻 R 峰中点切分每个 QRS/beat 片段，并裁成固定 token_len。
        channels, _ = ecg_signal.shape
        channel_qrs_segments = []

        for channel_index in range(channels):
            qrs_segments = []
            for i in range(1, len(qrs_inds) - 1):
                center = qrs_inds[i]
                start = (qrs_inds[i] + qrs_inds[i - 1]) // 2
                end = (qrs_inds[i] + qrs_inds[i + 1]) // 2

                start = max(start, 0)
                end = min(end, len(ecg_signal[channel_index]))
                actual_len = end - start

                if actual_len > self.token_len:
                    start = max(center - self.token_len // 2, 0)
                    end = min(start + self.token_len, len(ecg_signal[channel_index]))

                segment = np.zeros(self.token_len)
                segment_start = max(self.token_len // 2 - (center - start), 0)
                segment_end = segment_start + (end - start)

                if segment_end > self.token_len:
                    end -= segment_end - self.token_len
                    segment_end = self.token_len

                segment[segment_start:segment_end] = ecg_signal[channel_index][
                    start:end
                ]
                qrs_segments.append(segment)

            channel_qrs_segments.append(qrs_segments)

        return channel_qrs_segments

    def assign_time_blocks(self, qrs_inds, interval_length=100):
        # 把 R 峰位置映射成粗粒度时间块编号。
        return [(ind // interval_length) + 1 for ind in qrs_inds]

    def qrs_to_sequence(self, channel_qrs_segments, qrs_inds):
        # 把多导联 QRS 片段拉平成一句 ECG sentence，并补齐到固定长度。
        qrs_sequence = []
        in_chans = []
        in_times = []

        for channel_index, channel in enumerate(channel_qrs_segments):
            in_times.extend(self.assign_time_blocks(qrs_inds))
            for segment in channel:
                qrs_sequence.append(segment)
                in_chans.append(self.channel_ids[channel_index])

        current_patch_size = len(qrs_sequence)
        if current_patch_size < self.max_len:
            padding_needed = self.max_len - current_patch_size
            for _ in range(padding_needed):
                qrs_sequence.append(np.zeros(self.token_len))
                in_chans.append(0)
                in_times.append(0)
        elif current_patch_size > self.max_len:
            qrs_sequence = qrs_sequence[: self.max_len]
            in_chans = in_chans[: self.max_len]
            in_times = in_times[: self.max_len]

        return np.stack(qrs_sequence), np.array(in_chans), np.array(in_times)

    def plot_qrs_segments(self, qrs_segments, data_path, index):
        # 把切出来的 QRS 序列可视化，便于人工检查。
        save_path = os.path.dirname(data_path) + f"/figs/QRS_Segments_{index}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        channel = len(qrs_segments)

        if channel == 1:
            fig, axs = plt.subplots(
                channel, figsize=(12, 2 * channel), constrained_layout=True
            )
            axs = [axs]
        else:
            fig, axs = plt.subplots(
                channel, figsize=(12, 2 * channel), constrained_layout=True
            )

        for i, segments in enumerate(qrs_segments):
            offset = 0
            for segment in segments:
                segment_length = len(segment)
                axs[i].plot(
                    range(offset, offset + segment_length), segment, color="blue"
                )
                offset += segment_length

            axs[i].set_title(f"Channel {i + 1}")
            axs[i].set_xlabel("Sample Points")
            axs[i].set_ylabel("Amplitude")

        plt.suptitle("ECG Channels with Sequentially Plotted QRS Segments")
        plt.savefig(save_path)
        plt.close()

    def _initialize_output_arrays(self, batch_size):
        # 使用 memmap 直接写大数组，避免一次性占用过多内存。
        batch_qrs_seq = np.lib.format.open_memmap(
            os.path.join(self.save_path, f"{self.stage}_data.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(batch_size, self.max_len, self.token_len),
        )
        batch_in_chans = np.lib.format.open_memmap(
            os.path.join(self.save_path, f"{self.stage}_data_in_chans.npy"),
            mode="w+",
            dtype=np.int32,
            shape=(batch_size, self.max_len),
        )
        batch_in_times = np.lib.format.open_memmap(
            os.path.join(self.save_path, f"{self.stage}_data_in_times.npy"),
            mode="w+",
            dtype=np.int32,
            shape=(batch_size, self.max_len),
        )
        return batch_qrs_seq, batch_in_chans, batch_in_times

    def forward(self, x, plot):
        # 主流程：按样本执行 R 峰检测、QRS 切分、序列拼接和结果落盘。
        x = x[:, self.used_channels, :]
        bs, _, _ = x.shape
        batch_qrs_seq, batch_in_chans, batch_in_times = self._initialize_output_arrays(
            bs
        )

        if plot and bs > 0:
            plot_count = min(5, bs)
            indexs = np.random.choice(range(bs), size=plot_count, replace=False)
        else:
            indexs = np.array([], dtype=np.int64)

        for batch in tqdm(range(bs), desc="Processing batches"):
            ecg_signal = x[batch]
            qrs_inds = self.qrs_detection(ecg_signal)
            middle_qrs_inds = self.trim_edge_qrs(qrs_inds)
            channel_qrs_segments = self.extract_qrs_segments(ecg_signal, qrs_inds)

            if batch in indexs and plot:
                self.plot_qrs_segments(
                    channel_qrs_segments, self.save_path, index=batch
                )

            qrs_sequence, in_chans, in_times = self.qrs_to_sequence(
                channel_qrs_segments, middle_qrs_inds
            )
            batch_qrs_seq[batch] = qrs_sequence.astype(np.float32, copy=False)
            batch_in_chans[batch] = in_chans.astype(np.int32, copy=False)
            batch_in_times[batch] = in_times.astype(np.int32, copy=False)

        batch_qrs_seq.flush()
        batch_in_chans.flush()
        batch_in_times.flush()


def select_dataset(dataset_name):
    # 不同数据集使用不同 category / split 组合。
    if dataset_name == "PTBXL":
        data_category = [
            "all",
            "diagnostic",
            "form",
            "rhythm",
            "subdiagnostic",
            "superdiagnostic",
        ]
        data_type = ["train", "val", "test"]
    elif dataset_name == "MIMIC-IV":
        data_category = [
            "batch_0",
            "batch_1",
            "batch_2",
            "batch_3",
            "batch_4",
            "batch_5",
            "batch_6",
            "batch_7",
        ]
        data_type = ["train", "val"]
    elif dataset_name == "CSN":
        data_category = ["data"]
        data_type = ["train", "val", "test"]
    elif dataset_name == "CPSC2018":
        data_category = ["data"]
        data_type = ["train", "val", "test"]
    elif dataset_name == "ECG_DSA":
        data_category = ["data"]
        data_type = ["train", "val", "test"]
    elif dataset_name == "PhysioNet2021":
        data_category = ["data"]
        data_type = ["train", "val", "test"]
    else:
        raise ValueError("No such dataset available.")

    return data_category, data_type


def default_channel_ids(dataset_name, available_channels):
    # ECG_DSA 的导联编号并非 1..N 连续，因此单独映射。
    if dataset_name == "ECG_DSA" and available_channels == len(ECG_DSA_CHANNEL_IDS):
        return ECG_DSA_CHANNEL_IDS.copy()
    return list(range(1, available_channels + 1))


def resolve_channel_config(dataset_name, available_channels, used_channels, channel_ids):
    # 统一解析“实际取哪些导联”和“这些导联对应什么 embedding id”。
    base_channel_ids = default_channel_ids(dataset_name, available_channels)
    if used_channels is None:
        used_channels = list(range(available_channels))
    else:
        used_channels = list(used_channels)
        invalid_channels = [
            idx for idx in used_channels if idx < 0 or idx >= available_channels
        ]
        if invalid_channels:
            raise ValueError(
                f"Requested channels out of range for {dataset_name}: "
                f"{invalid_channels}. Available indices: 0 to {available_channels - 1}."
            )

    if channel_ids is None:
        channel_ids = [base_channel_ids[idx] for idx in used_channels]
    else:
        channel_ids = list(channel_ids)
        if len(channel_ids) != len(used_channels):
            raise ValueError("channel_ids must have the same length as used_channels.")

    return used_channels, channel_ids


def resolve_dataset_paths(dataset_name, category, split):
    # 根据数据集名推导原始波形文件路径和输出目录。
    if dataset_name == "ECG_DSA" and category == "data":
        return (
            ECG_DSA_WAVEFORM_DIR / f"{split}_data.npy",
            ECG_DSA_QRS_BASE_DIR,
        )

    data_path = Path(f"./datasets/ecg_datasets/{dataset_name}/{category}/{split}_data.npy")
    save_path = Path(f"./datasets/ecg_datasets/{dataset_name}_QRS/{category}/")
    return data_path, save_path


def resolve_custom_jobs(input_dirs, output_dirs, splits):
    # 支持用户手动指定输入/输出目录批量处理。
    if len(input_dirs) != len(output_dirs):
        raise ValueError("--input_dirs and --output_dirs must have the same length.")

    jobs = []
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        for split in splits:
            jobs.append(
                {
                    "category": input_path.name,
                    "split": split,
                    "data_path": input_path / f"{split}_data.npy",
                    "save_path": output_path,
                }
            )
    return jobs


def main():
    # CLI 入口：遍历任务列表，逐个数据集 split 生成 QRS token。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., PTBXL, MIMIC-IV, CSN, CPSC2018).",
    )
    parser.add_argument(
        "--used_channels",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of channel indices to use. By default, all available channels are used.",
    )
    parser.add_argument(
        "--channel_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional spatial embedding ids for the selected channels. Defaults follow the dataset lead layout.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset categories to process, for example: form rhythm.",
    )
    parser.add_argument(
        "--input_dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Optional custom waveform directories that contain <split>_data.npy files.",
    )
    parser.add_argument(
        "--output_dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Optional custom QRS output directories, paired with --input_dirs.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Optional custom split list used with --input_dirs, for example: train val.",
    )

    args = parser.parse_args()
    dataset_name = args.dataset_name
    selected_categories = args.categories

    if (args.input_dirs is None) != (args.output_dirs is None):
        raise ValueError("--input_dirs and --output_dirs must be provided together.")

    if args.input_dirs is not None:
        custom_splits = args.splits or ["train", "val", "test"]
        jobs = resolve_custom_jobs(args.input_dirs, args.output_dirs, custom_splits)
    else:
        data_category, data_type = select_dataset(dataset_name)
        if selected_categories is not None:
            invalid_categories = sorted(set(selected_categories) - set(data_category))
            if invalid_categories:
                raise ValueError(
                    f"Unsupported categories for {dataset_name}: {invalid_categories}"
                )
            data_category = selected_categories

        jobs = []
        for category in data_category:
            for split in data_type:
                data_path, save_path = resolve_dataset_paths(dataset_name, category, split)
                jobs.append(
                    {
                        "category": category,
                        "split": split,
                        "data_path": data_path,
                        "save_path": save_path,
                    }
                )

    for job in jobs:
        category = job["category"]
        type_ = job["split"]
        data_path = job["data_path"]
        save_path = job["save_path"]

        print(f"Processing data for {category} {type_}...")

        save_path.mkdir(parents=True, exist_ok=True)
        data = np.load(data_path, mmap_mode="r")
        print(data.shape)

        if data.shape[1] > 12:
            data = data.transpose(0, 2, 1)

        resolved_used_channels, resolved_channel_ids = resolve_channel_config(
            dataset_name=dataset_name,
            available_channels=data.shape[1],
            used_channels=args.used_channels,
            channel_ids=args.channel_ids,
        )

        tokenizer = QRSTokenizer(
            fs=100,
            max_len=256,
            token_len=96,
            save_path=str(save_path),
            stage=type_,
            used_channels=resolved_used_channels,
            channel_ids=resolved_channel_ids,
        )

        tokenizer(data, plot=True)


if __name__ == "__main__":
    main()
