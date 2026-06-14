"""Base dataset class for time series tasks."""

import math
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class BaseTimeSeriesDataset(Dataset):
    """时序数据集基类

    Args:
        data: 原始数据 [T, F]
        window_size: 窗口大小
        stride: 滑窗步长
        scale: 是否标准化
        offset: 在原始序列中的起始偏移量（用于保持 idx 连续）
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        scale: bool = True,
        offset: int = 0,
    ):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.stride = stride
        self.scale = scale
        self.offset = offset

        self.scaler = StandardScaler()
        if scale:
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)

    def __len__(self) -> int:
        return math.ceil(
            (len(self.data) - self.window_size + 1) / self.stride
        )

    def _get_actual_index(self, logical_index: int) -> int:
        """将逻辑索引转换为实际索引（连续）"""
        return logical_index * self.stride + self.offset

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """逆标准化"""
        return self.scaler.inverse_transform(data)


__all__ = ["BaseTimeSeriesDataset"]
