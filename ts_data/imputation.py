"""Imputation dataset for time series imputation tasks."""

import math
import numpy as np
import torch
from typing import Optional

from .base import BaseTimeSeriesDataset


class ImputationDataset(BaseTimeSeriesDataset):
    """插补任务数据集

    Args:
        data: 原始数据 [T, F]
        window_size: 窗口大小
        mask_ratio: 缺失比例
        mask_mode: 缺失模式 ("random" | "block" | "forecast")
        seed: 随机种子（用于可复现的 mask）
        stride: 滑窗步长
        scale: 是否标准化
        offset: 在原始序列中的起始偏移量
        x_mark: 时间特征 [T, time_dim]

    Returns:
        dict: {
            "x": Tensor [F, window_size] (mask 后),
            "y": Tensor [F, window_size] (完整),
            "mask": Tensor [F, window_size],
            "x_mark": Tensor 或 None,
            "y_mark": None,
            "idx": int,
        }
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        mask_ratio: float = 0.25,
        mask_mode: str = "random",
        seed: int = 42,
        stride: int = 1,
        scale: bool = True,
        offset: int = 0,
        x_mark: Optional[np.ndarray] = None,
    ):
        super().__init__(data, window_size, stride, scale, offset)
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.seed = seed
        self.x_mark = x_mark

    def _get_mask(self, idx: int) -> np.ndarray:
        """生成可复现的 mask

        Args:
            idx: 实际索引（连续）

        Returns:
            mask: [F, window_size] (1=观测, 0=缺失)
        """
        rng = np.random.RandomState(self.seed + idx)
        n_features = self.data.shape[1]

        if self.mask_mode == "random":
            mask = rng.random((n_features, self.window_size)) > self.mask_ratio

        elif self.mask_mode == "block":
            mask = np.ones((n_features, self.window_size))
            block_len = max(1, int(self.window_size * self.mask_ratio))
            start = rng.randint(0, self.window_size - block_len + 1)
            mask[:, start:start + block_len] = 0

        elif self.mask_mode == "forecast":
            mask = np.ones((n_features, self.window_size))
            mask[:, -max(1, int(self.window_size * self.mask_ratio)):] = 0

        else:
            raise ValueError(f"Unknown mask mode: {self.mask_mode}")

        return mask.astype(np.float32)

    def __getitem__(self, logical_index: int) -> dict:
        idx = self._get_actual_index(logical_index)
        local_idx = logical_index * self.stride

        # 完整序列 [window_size, F] -> [F, window_size]
        seq = self.data[local_idx:local_idx + self.window_size].T
        mask = self._get_mask(idx)  # 使用连续的实际索引

        # 应用 mask
        x = seq * mask

        # 时间特征
        x_mark = self.x_mark[local_idx:local_idx + self.window_size] if self.x_mark is not None else None

        return {
            "x": torch.FloatTensor(x),
            "y": torch.FloatTensor(seq),
            "mask": torch.FloatTensor(mask),
            "x_mark": torch.FloatTensor(x_mark) if x_mark is not None else None,
            "y_mark": None,
            "idx": idx,
        }


__all__ = ["ImputationDataset"]
