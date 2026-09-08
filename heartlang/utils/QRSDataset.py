"""HeartLang 预训练与微调阶段用到的数据集封装。"""

import os
import bisect
import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

folder_path = List[Path]


def build_ECGpretrain_dataset(
    datasets_2dlist: list,
    stage,
    aux_regression_label_cols: Optional[Sequence[str]] = None,
    aux_regression_target_scale=1.0,
):
    # 输入是二维目录列表，每个一维列表会拼成一个 ECGDataset。
    dataset_list = []
    for dataset_1dlist in datasets_2dlist:
        dataset = ECGDataset(
            dataset_1dlist,
            stage,
            aux_regression_label_cols=aux_regression_label_cols,
            aux_regression_target_scale=aux_regression_target_scale,
        )
        dataset_list.append(dataset)
    return dataset_list


class ECGSingleDataset(Dataset):
    def __init__(
        self,
        folder_path: Path,
        stage: str,
        aux_regression_label_cols: Optional[Sequence[str]] = None,
        aux_regression_target_scale=1.0,
    ):
        self.__folder_path = folder_path
        self.__stage = stage
        self.__aux_regression_label_cols = list(aux_regression_label_cols or [])
        self.__aux_regression_target_scale = float(aux_regression_target_scale)
        self.__init_dataset()

    def __init_dataset(self) -> None:
        # 预训练数据可能很大，使用 memmap 按需读取，避免把整份 .npy 一次性载入内存。
        self.data = np.load(
            os.path.join(str(self.__folder_path), self.__stage + "_data.npy"),
            mmap_mode="r",
        )
        self.in_time_matrix = np.load(
            os.path.join(str(self.__folder_path), self.__stage + "_data_in_times.npy"),
            mmap_mode="r",
        )
        self.in_chan_matrix = np.load(
            os.path.join(str(self.__folder_path), self.__stage + "_data_in_chans.npy"),
            mmap_mode="r",
        )
        self.aux_regression_labels = None
        self.aux_regression_mask = None
        if self.__aux_regression_label_cols:
            aux_path = os.path.join(
                str(self.__folder_path),
                self.__stage + "_aux_regression_labels.npy",
            )
            if not os.path.exists(aux_path):
                raise FileNotFoundError(
                    f"Missing auxiliary regression labels for pretraining: {aux_path}"
                )
            aux_values = np.load(aux_path, allow_pickle=True).astype(np.float32)
            if aux_values.ndim == 1:
                aux_values = aux_values.reshape(-1, 1)
            if aux_values.shape[0] != self.data.shape[0]:
                raise ValueError(
                    f"Auxiliary regression label count {aux_values.shape[0]} does not "
                    f"match sample count {self.data.shape[0]} in {aux_path}."
                )
            if aux_values.shape[1] != len(self.__aux_regression_label_cols):
                raise ValueError(
                    f"Auxiliary regression label dim {aux_values.shape[1]} does not "
                    f"match requested columns {len(self.__aux_regression_label_cols)} "
                    f"in {aux_path}."
                )
            if self.__aux_regression_target_scale <= 0:
                raise ValueError("aux_regression_target_scale must be positive.")
            self.aux_regression_mask = np.isfinite(aux_values).astype(np.float32)
            self.aux_regression_labels = (
                np.nan_to_num(aux_values, nan=0.0)
                / self.__aux_regression_target_scale
            ).astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        # memmap 以只读方式打开，按样本复制一份可写数组，避免 DataLoader 告警。
        item = (
            np.array(self.data[idx], copy=True),
            np.array(self.in_chan_matrix[idx], copy=True),
            np.array(self.in_time_matrix[idx], copy=True),
        )
        if self.aux_regression_labels is None:
            return item
        return item + (
            np.array(self.aux_regression_labels[idx], copy=True),
            np.array(self.aux_regression_mask[idx], copy=True),
        )


class ECGDataset(Dataset):
    def __init__(
        self,
        folder_paths: folder_path,
        stage,
        aux_regression_label_cols: Optional[Sequence[str]] = None,
        aux_regression_target_scale=1.0,
    ):
        self.__folder_paths = folder_paths
        self.__datasets = []
        self.__dataset_start_idxes = []
        self.__stage = stage
        self.__aux_regression_label_cols = list(aux_regression_label_cols or [])
        self.__aux_regression_target_scale = float(aux_regression_target_scale)

        self.__length = None
        self.__init_dataset()

    def __init_dataset(self) -> None:
        # 把多个目录的数据拼成一个逻辑上的大数据集。
        self.__datasets = [
            ECGSingleDataset(
                folder_path,
                self.__stage,
                aux_regression_label_cols=self.__aux_regression_label_cols,
                aux_regression_target_scale=self.__aux_regression_target_scale,
            )
            for folder_path in self.__folder_paths
        ]

        sample_num = 0
        for dataset in self.__datasets:
            self.__dataset_start_idxes.append(sample_num)
            sample_num += len(dataset)
        self.__length = sample_num

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        # 使用前缀长度数组把全局索引映射回具体子数据集。
        dataset_idx = bisect.bisect(self.__dataset_start_idxes, idx) - 1
        item_idx = idx - self.__dataset_start_idxes[dataset_idx]
        return self.__datasets[dataset_idx][item_idx]


class ECGDatasetFinetune(Dataset):
    """分类微调数据集，支持随机/分层抽样缩小训练集。"""

    def __init__(
        self,
        data_folder_path,
        stage,
        split_ratio=1.0,
        sampling_method="random",
        task_type="classification",
        aux_regression_metadata_dir="",
        aux_regression_label_cols=None,
        aux_regression_target_scale=1.0,
    ):
        self.stage = stage
        self.split_ratio = split_ratio
        self.sampling_method = sampling_method
        self.task_type = task_type
        self.aux_regression_metadata_dir = aux_regression_metadata_dir
        self.aux_regression_label_cols = list(aux_regression_label_cols or [])
        self.aux_regression_target_scale = float(aux_regression_target_scale)
        self.data_path = os.path.join(data_folder_path, stage + "_data.npy")
        self.label_path = os.path.join(data_folder_path, stage + "_labels.npy")
        self.aux_regression_label_path = os.path.join(
            data_folder_path, stage + "_aux_regression_labels.npy"
        )
        self.chan_matrix_path = os.path.join(
            data_folder_path, stage + "_data_in_chans.npy"
        )
        self.time_matrix_path = os.path.join(
            data_folder_path, stage + "_data_in_times.npy"
        )
        (
            self.data,
            self.labels,
            self.in_chan_matrix,
            self.in_time_matrix,
            self.aux_regression_labels,
            self.aux_regression_mask,
        ) = self.load_data()
        self.prepare_data()
        self.print_class_distribution()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.aux_regression_labels is not None:
            label = {
                "main": label,
                "aux_regression": self.aux_regression_labels[idx],
                "aux_regression_mask": self.aux_regression_mask[idx],
            }
        in_chan_matrix = self.in_chan_matrix[idx]
        in_time_matrix = self.in_time_matrix[idx]
        return sample, label, in_chan_matrix, in_time_matrix

    def random_sampling(
        self,
        data,
        labels,
        in_chan_matrix,
        in_time_matrix,
        aux_regression_labels=None,
        aux_regression_mask=None,
    ):
        # 按比例随机抽样训练样本。
        sample_indices = random.sample(
            range(len(data)), int(len(data) * self.split_ratio)
        )
        data = data[sample_indices]
        labels = labels[sample_indices]
        in_chan_matrix = in_chan_matrix[sample_indices]
        in_time_matrix = in_time_matrix[sample_indices]
        if aux_regression_labels is not None:
            aux_regression_labels = aux_regression_labels[sample_indices]
        if aux_regression_mask is not None:
            aux_regression_mask = aux_regression_mask[sample_indices]
        return (
            data,
            labels,
            in_chan_matrix,
            in_time_matrix,
            aux_regression_labels,
            aux_regression_mask,
        )


    def stratified_sampling(
        self,
        data,
        labels,
        in_chan_matrix,
        in_time_matrix,
        aux_regression_labels=None,
        aux_regression_mask=None,
    ):
        if self.task_type == "regression":
            raise ValueError("Regression task does not support stratified sampling.")
        # 依据标签主类做近似分层抽样，尽量保持类别分布。
        y = np.argmax(labels.numpy(), axis=1)
        unique_classes = np.unique(y)

        # Initialize lists to collect the selected indices
        train_indices = []

        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            np.random.shuffle(class_indices)

            # Calculate the number of samples to select from this class
            n_samples = max(1, int(len(class_indices) * self.split_ratio))

            # Add selected indices to the train_indices list
            train_indices.extend(class_indices[:n_samples])

        # Convert train_indices to a numpy array and shuffle it
        train_indices = np.array(train_indices)
        np.random.shuffle(train_indices)

        # Subset the data, labels, in_chan_matrix, and in_time_matrix
        data = data[train_indices]
        labels = labels[train_indices]
        in_chan_matrix = in_chan_matrix[train_indices]
        in_time_matrix = in_time_matrix[train_indices]
        if aux_regression_labels is not None:
            aux_regression_labels = aux_regression_labels[train_indices]
        if aux_regression_mask is not None:
            aux_regression_mask = aux_regression_mask[train_indices]

        return (
            data,
            labels,
            in_chan_matrix,
            in_time_matrix,
            aux_regression_labels,
            aux_regression_mask,
        )

    def load_data(self):
        # 微调阶段同时读取标签文件。
        data = torch.from_numpy(np.load(self.data_path,allow_pickle=True)).float()
        label_array = np.load(self.label_path, allow_pickle=True)
        if self.task_type == "regression":
            labels = torch.from_numpy(label_array.astype(np.float32)).float()
        else:
            labels = torch.from_numpy(label_array.astype(np.int64))
        aux_regression_labels, aux_regression_mask = self.load_aux_regression_labels(labels)
        in_chan_matrix = torch.from_numpy(np.load(self.chan_matrix_path,allow_pickle=True))
        in_time_matrix = torch.from_numpy(np.load(self.time_matrix_path,allow_pickle=True))

        return (
            data,
            labels,
            in_chan_matrix,
            in_time_matrix,
            aux_regression_labels,
            aux_regression_mask,
        )

    def load_aux_regression_labels(self, main_labels: torch.Tensor):
        if not self.aux_regression_label_cols:
            return None, None
        if os.path.exists(self.aux_regression_label_path):
            aux_values = np.load(self.aux_regression_label_path, allow_pickle=True).astype(
                np.float32
            )
            if aux_values.ndim == 1:
                aux_values = aux_values.reshape(-1, 1)
            if aux_values.shape[0] != len(main_labels):
                raise ValueError(
                    f"Auxiliary regression label count {aux_values.shape[0]} does not match "
                    f"main label count {len(main_labels)} for split '{self.stage}'."
                )
            if aux_values.shape[1] != len(self.aux_regression_label_cols):
                raise ValueError(
                    f"Auxiliary regression label dim {aux_values.shape[1]} does not match "
                    f"requested columns {len(self.aux_regression_label_cols)} for split '{self.stage}'."
                )
            aux_mask = np.isfinite(aux_values).astype(np.float32)
            aux_values = np.nan_to_num(aux_values, nan=0.0)
            if self.aux_regression_target_scale <= 0:
                raise ValueError("aux_regression_target_scale must be positive.")
            aux_values = aux_values / self.aux_regression_target_scale
            print(
                f"Loaded auxiliary regression labels from {self.aux_regression_label_path}"
            )
            return torch.from_numpy(aux_values).float(), torch.from_numpy(aux_mask).float()

        if not self.aux_regression_metadata_dir:
            raise ValueError(
                "Using auxiliary regression labels requires either split-level "
                "*_aux_regression_labels.npy files in dataset_dir or --aux_regression_metadata_dir."
            )

        metadata_path = os.path.join(
            self.aux_regression_metadata_dir, self.stage + "_metadata.csv"
        )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Auxiliary regression metadata not found: {metadata_path}"
            )

        metadata = pd.read_csv(metadata_path)
        missing_cols = [
            col for col in self.aux_regression_label_cols if col not in metadata.columns
        ]
        if missing_cols:
            raise KeyError(
                f"Missing auxiliary regression columns in {metadata_path}: {missing_cols}"
            )
        if len(metadata) != len(main_labels):
            raise ValueError(
                f"Metadata row count {len(metadata)} does not match label count {len(main_labels)} "
                f"for split '{self.stage}'."
            )

        if "label" in metadata.columns:
            metadata_main = metadata["label"].to_numpy()
            expected_main = main_labels.reshape(-1).cpu().numpy()
            if not np.array_equal(metadata_main, expected_main):
                raise ValueError(
                    f"Main labels in {metadata_path} are not aligned with {self.label_path}."
                )

        aux_values = metadata[self.aux_regression_label_cols].to_numpy(dtype=np.float32)
        aux_mask = np.isfinite(aux_values).astype(np.float32)
        aux_values = np.nan_to_num(aux_values, nan=0.0)

        if self.aux_regression_target_scale <= 0:
            raise ValueError("aux_regression_target_scale must be positive.")
        aux_values = aux_values / self.aux_regression_target_scale

        return torch.from_numpy(aux_values).float(), torch.from_numpy(aux_mask).float()

    def print_class_distribution(self):
        if self.task_type == "regression":
            labels = self.labels.float()
            print(f"==================== stage {self.stage}'s regression target summary ====================")
            print(f"Label shape: {tuple(labels.shape)}")
            print(f"Mean: {labels.mean().item():.6f}")
            print(f"Std: {labels.std(unbiased=False).item():.6f}")
            print(f"Min: {labels.min().item():.6f}")
            print(f"Max: {labels.max().item():.6f}")
            return
        # 打印每个标签维度的样本数，便于检查标签稀疏度。
        label_counts = torch.sum(self.labels, dim=0)
        print(f"==================== stage {self.stage}'s class distribution ====================")
        for class_index, count in enumerate(label_counts):
            print(f"Class {class_index}: {int(count.item())} samples")
        if self.aux_regression_labels is not None:
            print(
                f"==================== stage {self.stage}'s auxiliary regression summary ===================="
            )
            for idx, col in enumerate(self.aux_regression_label_cols):
                values = self.aux_regression_labels[:, idx]
                mask = self.aux_regression_mask[:, idx] > 0
                valid_values = values[mask]
                valid_count = int(mask.sum().item())
                if valid_count == 0:
                    print(f"{col}: no valid auxiliary labels")
                    continue
                print(
                    f"{col} (scaled / {self.aux_regression_target_scale:g}): "
                    f"count={valid_count}, mean={valid_values.mean().item():.6f}, "
                    f"std={valid_values.std(unbiased=False).item():.6f}, "
                    f"min={valid_values.min().item():.6f}, max={valid_values.max().item():.6f}"
                )

    def prepare_data(self):
        # 只在训练集上执行抽样；验证/测试集保持完整。
        if self.stage == "train" and self.split_ratio < 1.0:
            if self.sampling_method == "random":
                (
                    self.data,
                    self.labels,
                    self.in_chan_matrix,
                    self.in_time_matrix,
                    self.aux_regression_labels,
                    self.aux_regression_mask,
                ) = (
                    self.random_sampling(
                        self.data,
                        self.labels,
                        self.in_chan_matrix,
                        self.in_time_matrix,
                        self.aux_regression_labels,
                        self.aux_regression_mask,
                    )
                )
            elif self.sampling_method == "stratified":
                if self.task_type == "regression":
                    raise ValueError("Regression task only supports random sampling.")
                (
                    self.data,
                    self.labels,
                    self.in_chan_matrix,
                    self.in_time_matrix,
                    self.aux_regression_labels,
                    self.aux_regression_mask,
                ) = (
                    self.stratified_sampling(
                        self.data,
                        self.labels,
                        self.in_chan_matrix,
                        self.in_time_matrix,
                        self.aux_regression_labels,
                        self.aux_regression_mask,
                    )
                )


def prepare_finetune_dataset(
    data_folder_path,
    split_ratio,
    sampling_method,
    task_type="classification",
    aux_regression_metadata_dir="",
    aux_regression_label_cols: Optional[Sequence[str]] = None,
    aux_regression_target_scale=1.0,
):
    # 统一构建 train / val / test 三个数据集对象。
    train_dataset = ECGDatasetFinetune(
        data_folder_path=data_folder_path,
        stage="train",
        split_ratio=split_ratio,
        sampling_method=sampling_method,
        task_type=task_type,
        aux_regression_metadata_dir=aux_regression_metadata_dir,
        aux_regression_label_cols=aux_regression_label_cols,
        aux_regression_target_scale=aux_regression_target_scale,
    )
    val_dataset = ECGDatasetFinetune(
        data_folder_path=data_folder_path,
        stage="val",
        task_type=task_type,
        aux_regression_metadata_dir=aux_regression_metadata_dir,
        aux_regression_label_cols=aux_regression_label_cols,
        aux_regression_target_scale=aux_regression_target_scale,
    )
    test_dataset = ECGDatasetFinetune(
        data_folder_path=data_folder_path,
        stage="test",
        task_type=task_type,
        aux_regression_metadata_dir=aux_regression_metadata_dir,
        aux_regression_label_cols=aux_regression_label_cols,
        aux_regression_target_scale=aux_regression_target_scale,
    )
    return train_dataset, val_dataset, test_dataset
