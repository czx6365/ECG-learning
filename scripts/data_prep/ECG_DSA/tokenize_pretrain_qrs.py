"""轻量版 ECG_DSA 预训练 QRS tokenizer。

只依赖 numpy / wfdb / tqdm，不依赖 torch。
输入目录需要包含：
- train_data.npy
- val_data.npy

输出目录会生成：
- train_data.npy
- train_data_in_chans.npy
- train_data_in_times.npy
- val_data.npy
- val_data_in_chans.npy
- val_data_in_times.npy
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from wfdb import processing


ECG_DSA_CHANNEL_IDS = [1, 2, 3, 7, 8, 9, 10, 11, 12]


class QRSTokenizerLite:
    def __init__(self, fs=100, max_len=256, token_len=96, used_channels=None, channel_ids=None):
        self.fs = fs
        self.max_len = max_len
        self.token_len = token_len
        self.used_channels = list(range(9)) if used_channels is None else list(used_channels)
        self.channel_ids = ECG_DSA_CHANNEL_IDS.copy() if channel_ids is None else list(channel_ids)
        if len(self.used_channels) != len(self.channel_ids):
            raise ValueError("used_channels and channel_ids must have the same length.")

    def qrs_detection(self, ecg_signal):
        return processing.xqrs_detect(sig=ecg_signal[0, :], fs=self.fs, verbose=False)

    def trim_edge_qrs(self, qrs_inds):
        if len(qrs_inds) <= 2:
            return qrs_inds[:0]
        return qrs_inds[1:-1]

    def extract_qrs_segments(self, ecg_signal, qrs_inds):
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

                segment = np.zeros(self.token_len, dtype=np.float32)
                segment_start = max(self.token_len // 2 - (center - start), 0)
                segment_end = segment_start + (end - start)

                if segment_end > self.token_len:
                    end -= segment_end - self.token_len
                    segment_end = self.token_len

                segment[segment_start:segment_end] = ecg_signal[channel_index][start:end]
                qrs_segments.append(segment)
            channel_qrs_segments.append(qrs_segments)
        return channel_qrs_segments

    @staticmethod
    def assign_time_blocks(qrs_inds, interval_length=100):
        return [(ind // interval_length) + 1 for ind in qrs_inds]

    def qrs_to_sequence(self, channel_qrs_segments, qrs_inds):
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
                qrs_sequence.append(np.zeros(self.token_len, dtype=np.float32))
                in_chans.append(0)
                in_times.append(0)
        elif current_patch_size > self.max_len:
            qrs_sequence = qrs_sequence[: self.max_len]
            in_chans = in_chans[: self.max_len]
            in_times = in_times[: self.max_len]

        return (
            np.stack(qrs_sequence).astype(np.float32, copy=False),
            np.asarray(in_chans, dtype=np.int32),
            np.asarray(in_times, dtype=np.int32),
        )

    def tokenize_split(self, data_path: Path, output_dir: Path, split: str):
        data = np.load(data_path, mmap_mode="r")
        if data.shape[1] > 12:
            data = data.transpose(0, 2, 1)
        data = data[:, self.used_channels, :]

        batch_size = len(data)
        qrs_seq = np.lib.format.open_memmap(
            output_dir / f"{split}_data.npy",
            mode="w+",
            dtype=np.float32,
            shape=(batch_size, self.max_len, self.token_len),
        )
        in_chans = np.lib.format.open_memmap(
            output_dir / f"{split}_data_in_chans.npy",
            mode="w+",
            dtype=np.int32,
            shape=(batch_size, self.max_len),
        )
        in_times = np.lib.format.open_memmap(
            output_dir / f"{split}_data_in_times.npy",
            mode="w+",
            dtype=np.int32,
            shape=(batch_size, self.max_len),
        )

        for batch_index in tqdm(range(batch_size), desc=f"Tokenizing {split}"):
            ecg_signal = np.asarray(data[batch_index])
            qrs_inds = self.qrs_detection(ecg_signal)
            middle_qrs_inds = self.trim_edge_qrs(qrs_inds)
            channel_qrs_segments = self.extract_qrs_segments(ecg_signal, qrs_inds)
            sequence, chans, times = self.qrs_to_sequence(channel_qrs_segments, middle_qrs_inds)
            qrs_seq[batch_index] = sequence
            in_chans[batch_index] = chans
            in_times[batch_index] = times

        qrs_seq.flush()
        in_chans.flush()
        in_times.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize ECG_DSA pretraining waveforms into QRS sequences.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--token_len", type=int, default=96)
    return parser.parse_args()


def main():
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = QRSTokenizerLite(fs=args.fs, max_len=args.max_len, token_len=args.token_len)
    for split in args.splits:
        tokenizer.tokenize_split(
            data_path=args.input_dir / f"{split}_data.npy",
            output_dir=args.output_dir,
            split=split,
        )


if __name__ == "__main__":
    main()
