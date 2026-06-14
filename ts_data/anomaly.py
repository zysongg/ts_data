"""Anomaly detection dataset for time series anomaly detection tasks."""

import numpy as np
import torch
from typing import Optional

from .base import BaseTimeSeriesDataset


class AnomalyDataset(BaseTimeSeriesDataset):
    """异常检测任务数据集

    支持两种模式：
    1. train: 只返回数据，标签为占位符（全0）
    2. test: 返回数据和真实异常标签

    说明：异常检测通常使用自编码器，训练时只用正常数据，不需要标签

    Args:
        data: 原始数据 [T, F]
        labels: 异常标签 [T] (0=正常, 1=异常)
        window_size: 窗口大小
        stride: 滑窗步长
        scale: 是否标准化
        offset: 在原始序列中的起始偏移量
        x_mark: 时间特征 [T, time_dim]
        mode: 模式 ("train" | "test")

    Returns:
        dict: {
            "x": Tensor [F, window_size],
            "y": Tensor [window_size],
            "mask": None,
            "x_mark": Tensor 或 None,
            "y_mark": None,
            "idx": int,
        }
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 96,
        stride: int = 1,
        scale: bool = True,
        offset: int = 0,
        x_mark: Optional[np.ndarray] = None,
        mode: str = "train",
    ):
        super().__init__(data, window_size, stride, scale, offset)
        self.x_mark = x_mark
        self.mode = mode

        if mode == "test":
            assert labels is not None, "Test mode requires labels"
            self.labels = labels.astype(np.float32)
        else:
            # 训练模式：标签为占位符
            self.labels = None

    def __getitem__(self, logical_index: int) -> dict:
        idx = self._get_actual_index(logical_index)
        local_idx = logical_index * self.stride

        # 输入序列 [window_size, F] -> [F, window_size]
        x = self.data[local_idx:local_idx + self.window_size].T

        # 获取标签
        if self.mode == "train":
            y = np.zeros(self.window_size, dtype=np.float32)  # 占位符
        else:
            y = self.labels[local_idx:local_idx + self.window_size]

        # 时间特征
        x_mark = self.x_mark[local_idx:local_idx + self.window_size] if self.x_mark is not None else None

        return {
            "x": torch.FloatTensor(x),
            "y": torch.FloatTensor(y),
            "mask": None,
            "x_mark": torch.FloatTensor(x_mark) if x_mark is not None else None,
            "y_mark": None,
            "idx": idx,
        }


__all__ = ["AnomalyDataset"]
