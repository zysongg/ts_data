"""Generation dataset for time series generation tasks."""

import numpy as np
import torch
from typing import Optional

from .base import BaseTimeSeriesDataset


class GenerationDataset(BaseTimeSeriesDataset):
    """生成任务数据集

    Args:
        data: 原始数据 [T, F]
        window_size: 窗口大小
        stride: 滑窗步长
        scale: 是否标准化
        offset: 在原始序列中的起始偏移量
        x_mark: 时间特征 [T, time_dim]

    Returns:
        dict: {
            "x": Tensor [F, window_size],
            "y": None,
            "mask": None,
            "x_mark": Tensor 或 None,
            "y_mark": None,
            "idx": int,
        }
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        scale: bool = True,
        offset: int = 0,
        x_mark: Optional[np.ndarray] = None,
    ):
        super().__init__(data, window_size, stride, scale, offset)
        self.x_mark = x_mark

    def __getitem__(self, logical_index: int) -> dict:
        idx = self._get_actual_index(logical_index)
        local_idx = logical_index * self.stride

        # 输入序列 [window_size, F] -> [F, window_size]
        x = self.data[local_idx:local_idx + self.window_size].T

        # 时间特征
        x_mark = self.x_mark[local_idx:local_idx + self.window_size] if self.x_mark is not None else None

        return {
            "x": torch.FloatTensor(x),
            "y": None,
            "mask": None,
            "x_mark": torch.FloatTensor(x_mark) if x_mark is not None else None,
            "y_mark": None,
            "idx": idx,
        }


__all__ = ["GenerationDataset"]
