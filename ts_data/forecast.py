"""Forecast dataset for time series prediction tasks."""

import math
import numpy as np
import torch
from typing import Optional

from .base import BaseTimeSeriesDataset


class ForecastDataset(BaseTimeSeriesDataset):
    """预测任务数据集

    Args:
        data: 原始数据 [T, F]
        input_len: 输入序列长度（历史窗口）
        pred_len: 预测序列长度
        label_len: 标签序列长度（teacher forcing）
        stride: 滑窗步长
        scale: 是否标准化
        offset: 在原始序列中的起始偏移量
        x_mark: 输入时间特征 [T, time_dim]
        y_mark: 目标时间特征 [T, time_dim]

    Returns:
        dict: {
            "x": Tensor [F, input_len],
            "y": Tensor [F, label_len + pred_len],
            "mask": None,
            "x_mark": Tensor 或 None,
            "y_mark": Tensor 或 None,
            "idx": int,
        }
    """

    def __init__(
        self,
        data: np.ndarray,
        input_len: int,
        pred_len: int,
        label_len: int = 0,
        stride: int = 1,
        scale: bool = True,
        offset: int = 0,
        x_mark: Optional[np.ndarray] = None,
        y_mark: Optional[np.ndarray] = None,
    ):
        super().__init__(data, input_len + pred_len, stride, scale, offset)
        self.input_len = input_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.x_mark = x_mark
        self.y_mark = y_mark

    def __len__(self) -> int:
        return math.ceil(
            (len(self.data) - self.input_len - self.pred_len + 1) / self.stride
        )

    def __getitem__(self, logical_index: int) -> dict:
        idx = self._get_actual_index(logical_index)
        local_idx = logical_index * self.stride

        # 输入序列 [input_len, F] -> [F, input_len]
        x = self.data[local_idx:local_idx + self.input_len].T

        # 目标序列 [label_len + pred_len, F] -> [F, label_len + pred_len]
        r_start = local_idx + self.input_len - self.label_len
        r_end = local_idx + self.input_len + self.pred_len
        y = self.data[r_start:r_end].T

        # 时间特征
        x_mark = self.x_mark[local_idx:local_idx + self.input_len] if self.x_mark is not None else None
        y_mark = self.y_mark[r_start:r_end] if self.y_mark is not None else None

        return {
            "x": torch.FloatTensor(x),
            "y": torch.FloatTensor(y),
            "mask": None,
            "x_mark": torch.FloatTensor(x_mark) if x_mark is not None else None,
            "y_mark": torch.FloatTensor(y_mark) if y_mark is not None else None,
            "idx": idx,
        }


__all__ = ["ForecastDataset"]
