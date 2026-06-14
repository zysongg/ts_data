"""Conditional generation dataset for time series generation tasks with conditions."""

import numpy as np
import torch
from typing import Optional, Dict, Any, Union

from .base import BaseTimeSeriesDataset


class ConditionalGenerationDataset(BaseTimeSeriesDataset):
    """条件生成任务数据集

    支持三种条件类型：
    1. 文本条件 (text): 预计算的文本嵌入
    2. 属性条件 (attribute): 离散或连续属性
    3. 标签条件 (label): 类别标签

    Args:
        data: 原始数据 [T, F] 或预分割的 [N, L, F]
        window_size: 窗口大小（如果 data 是 [T, F]）
        stride: 滑窗步长
        scale: 是否标准化
        offset: 在原始序列中的起始偏移量
        x_mark: 时间特征 [T, time_dim]
        text_emb: 文本嵌入 [N, D] 或 [T, D]
        attrs: 属性 [N, A] 或 [T, A]
        labels: 标签 [N] 或 [T]

    Returns:
        dict: {
            "x": Tensor [F, window_size],
            "x_mark": Tensor 或 None,
            "text_emb": Tensor [D] 或 None,
            "attrs": Tensor [A] 或 None,
            "label": int 或 None,
            "idx": int,
        }
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 128,
        stride: int = 1,
        scale: bool = True,
        offset: int = 0,
        x_mark: Optional[np.ndarray] = None,
        text_emb: Optional[np.ndarray] = None,
        attrs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ):
        super().__init__(data, window_size, stride, scale, offset)
        self.x_mark = x_mark

        # 处理条件信息
        self.text_emb = None
        self.attrs = None
        self.labels = None

        if text_emb is not None:
            self.text_emb = text_emb.astype(np.float32)
        if attrs is not None:
            self.attrs = attrs
        if labels is not None:
            self.labels = labels.astype(np.int64)

    def _get_condition(self, local_idx: int, logical_index: int) -> Dict[str, Any]:
        """获取对应样本的条件信息

        Args:
            local_idx: 在当前分割中的局部索引
            logical_index: 逻辑索引（用于窗口采样）

        Returns:
            包含 text_emb, attrs, label 的字典
        """
        result = {"text_emb": None, "attrs": None, "label": None}

        # 判断条件是与样本对齐还是与时间步对齐
        n_samples = len(self)

        if self.text_emb is not None:
            if len(self.text_emb) == n_samples:
                # 与样本对齐
                result["text_emb"] = torch.FloatTensor(self.text_emb[logical_index])
            elif len(self.text_emb) >= local_idx + self.window_size:
                # 与时间步对齐，取窗口均值
                result["text_emb"] = torch.FloatTensor(
                    self.text_emb[local_idx:local_idx + self.window_size].mean(axis=0)
                )

        if self.attrs is not None:
            if len(self.attrs) == n_samples:
                result["attrs"] = torch.LongTensor(self.attrs[logical_index])
            elif len(self.attrs) >= local_idx + self.window_size:
                # 取窗口起始位置的属性
                result["attrs"] = torch.LongTensor(self.attrs[local_idx])

        if self.labels is not None:
            if len(self.labels) == n_samples:
                result["label"] = int(self.labels[logical_index])
            elif len(self.labels) >= local_idx + self.window_size:
                # 取窗口内的多数标签
                window_labels = self.labels[local_idx:local_idx + self.window_size]
                result["label"] = int(np.bincount(window_labels).argmax())

        return result

    def __getitem__(self, logical_index: int) -> dict:
        idx = self._get_actual_index(logical_index)
        local_idx = logical_index * self.stride

        # 输入序列 [window_size, F] -> [F, window_size]
        x = self.data[local_idx:local_idx + self.window_size].T

        # 时间特征
        x_mark = None
        if self.x_mark is not None:
            x_mark = self.x_mark[local_idx:local_idx + self.window_size]

        # 获取条件信息
        conditions = self._get_condition(local_idx, logical_index)

        return {
            "x": torch.FloatTensor(x),
            "x_mark": torch.FloatTensor(x_mark) if x_mark is not None else None,
            "text_emb": conditions["text_emb"],
            "attrs": conditions["attrs"],
            "label": conditions["label"],
            "idx": idx,
        }


class PreSplitGenerationDataset:
    """预分割的条件生成数据集

    直接加载预分割的 npy 文件，适用于 ConTSG-Bench 格式的数据。

    Args:
        ts: 时序数据 [N, L, F]
        text_emb: 文本嵌入 [N, D]
        attrs: 属性 [N, A]
        labels: 标签 [N]
        caps: 原始文本描述 [N]
        normalize: 是否标准化

    Returns:
        dict: {
            "x": Tensor [F, L],
            "tp": Tensor [L],
            "text_emb": Tensor [D] 或 None,
            "attrs": Tensor [A] 或 None,
            "label": int 或 None,
            "cap": str 或 None,
            "idx": int,
        }
    """

    def __init__(
        self,
        ts: np.ndarray,
        text_emb: Optional[np.ndarray] = None,
        attrs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        caps: Optional[np.ndarray] = None,
        normalize: bool = True,
    ):
        self.ts = ts.astype(np.float32)
        self.text_emb = text_emb.astype(np.float32) if text_emb is not None else None
        self.attrs = attrs
        self.labels = labels.astype(np.int64) if labels is not None else None
        self.caps = caps

        # 标准化
        if normalize:
            self._normalize()

    def _normalize(self):
        """标准化时序数据"""
        self.ts_mean = self.ts.mean(axis=(0, 1), keepdims=True)
        self.ts_std = self.ts.std(axis=(0, 1), keepdims=True)
        self.ts_std = np.where(self.ts_std == 0, 1.0, self.ts_std)
        self.ts = (self.ts - self.ts_mean) / self.ts_std

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, idx: int) -> dict:
        ts_tensor = torch.from_numpy(self.ts[idx])  # [L, F]
        seq_len = ts_tensor.shape[0]

        # 转置为 [F, L]
        x = ts_tensor.transpose(0, 1)

        result = {
            "x": x,
            "tp": torch.arange(seq_len, dtype=torch.float32),
            "text_emb": torch.from_numpy(self.text_emb[idx]) if self.text_emb is not None else None,
            "attrs": torch.LongTensor(self.attrs[idx]) if self.attrs is not None else None,
            "label": int(self.labels[idx]) if self.labels is not None else None,
            "cap": str(self.caps[idx]) if self.caps is not None else None,
            "idx": idx,
        }

        return result


__all__ = ["ConditionalGenerationDataset", "PreSplitGenerationDataset"]
