"""Classification dataset for time series classification tasks."""

import numpy as np
import torch
from typing import Optional

from .base import BaseTimeSeriesDataset


class ClassificationDataset(BaseTimeSeriesDataset):
    """分类任务数据集

    支持两种标签模式：
    1. per_step_labels: 每个时间步一个标签 [T]，需要指定 label_mode
    2. per_sample_labels: 每个样本一个标签 [num_windows]

    Args:
        data: 原始数据 [T, F]
        labels: 标签 [T] 或 [num_windows]
        window_size: 窗口大小
        stride: 滑窗步长
        scale: 是否标准化
        offset: 在原始序列中的起始偏移量
        x_mark: 时间特征 [T, time_dim]
        label_mode: 标签模式 ("last" | "majority")

    Returns:
        dict: {
            "x": Tensor [F, window_size],
            "y": int,
            "mask": None,
            "x_mark": Tensor 或 None,
            "y_mark": None,
            "idx": int,
        }
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        stride: int = 1,
        scale: bool = True,
        offset: int = 0,
        x_mark: Optional[np.ndarray] = None,
        label_mode: str = "last",
    ):
        super().__init__(data, window_size, stride, scale, offset)
        self.x_mark = x_mark
        self.label_mode = label_mode

        # 判断标签模式
        if len(labels) == len(data):
            self.per_step_labels = True
            self.labels = labels.astype(np.int64)
        elif len(labels) == len(self):
            self.per_step_labels = False
            self.labels = labels.astype(np.int64)
        else:
            raise ValueError(
                f"Labels length {len(labels)} doesn't match data length {len(data)} "
                f"or number of windows {len(self)}"
            )

    def _get_label(self, local_idx: int) -> int:
        """获取窗口对应的标签"""
        if self.per_step_labels:
            window_labels = self.labels[local_idx:local_idx + self.window_size]

            if self.label_mode == "last":
                return int(window_labels[-1])
            elif self.label_mode == "majority":
                return int(np.bincount(window_labels).argmax())
            else:
                raise ValueError(f"Unknown label mode: {self.label_mode}")
        else:
            return int(self.labels[local_idx])

    def __getitem__(self, logical_index: int) -> dict:
        idx = self._get_actual_index(logical_index)
        local_idx = logical_index * self.stride

        # 输入序列 [window_size, F] -> [F, window_size]
        x = self.data[local_idx:local_idx + self.window_size].T
        y = self._get_label(local_idx)

        # 时间特征
        x_mark = self.x_mark[local_idx:local_idx + self.window_size] if self.x_mark is not None else None

        return {
            "x": torch.FloatTensor(x),
            "y": y,
            "mask": None,
            "x_mark": torch.FloatTensor(x_mark) if x_mark is not None else None,
            "y_mark": None,
            "idx": idx,
        }


__all__ = ["ClassificationDataset"]
