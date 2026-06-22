"""DataModule for time series data management."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .forecast import ForecastDataset
from .imputation import ImputationDataset
from .generation import GenerationDataset
from .classification import ClassificationDataset
from .anomaly import AnomalyDataset
from .conditional_generation import ConditionalGenerationDataset
from .utils import load_data


class DataModule:
    """数据模块，负责数据分割和 Dataset 创建

    支持三种分割模式：
    1. ratio: 按比例分割
    2. standard: 标准分割（ETT 等常用数据集）
    3. fixed: 固定分割点

    Args:
        data: 原始数据 [T, F]，或文件路径字符串
        labels: 标签（用于分类/异常检测）
        text_emb: 文本嵌入 [N, D] 或 [T, D]（用于条件生成）
        attrs: 属性 [N, A] 或 [T, A]（用于条件生成）
        split_ratio: 分割比例 (train, val, test)
        split_mode: 分割模式 ("ratio" | "standard")
        dataset_name: 数据集名称（用于标准分割）
        scale: 是否标准化
        file_format: 文件格式（当 data 是文件路径时使用）
        **load_kwargs: 传递给文件加载函数的参数
    """

    # 标准数据集分割点
    STANDARD_SPLITS = {
        "etth1": (12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24),
        "etth2": (12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24),
        "ettm1": (12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4),
        "ettm2": (12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4),
    }
    
    # ETT 数据集名称集合（用于自动识别）
    ETT_DATASETS = {"etth1", "etth2", "ettm1", "ettm2"}
    
    # 默认分割比例
    DEFAULT_SPLIT_RATIO = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data: Union[np.ndarray, str],
        labels: Optional[np.ndarray] = None,
        text_emb: Optional[np.ndarray] = None,
        attrs: Optional[np.ndarray] = None,
        split_ratio: Optional[Tuple[float, float, float]] = None,
        split_mode: Optional[str] = None,
        dataset_name: Optional[str] = None,
        scale: bool = True,
        file_format: Optional[str] = None,
        **load_kwargs,
    ):
        # 加载数据
        if isinstance(data, str):
            data_array, time_marks, dates = load_data(data, file_format=file_format, **load_kwargs)
            self.data = data_array
            self.time_marks = time_marks
            self.dates = dates
        else:
            self.data = data.astype(np.float32)
            self.time_marks = None
            self.dates = None

        self.labels = labels
        self.text_emb = text_emb.astype(np.float32) if text_emb is not None else None
        self.attrs = attrs
        self.dataset_name = dataset_name.lower() if dataset_name else None
        self.scale = scale

        # 自动推断分割模式
        if split_mode is None:
            if self.dataset_name in self.ETT_DATASETS:
                split_mode = "standard"
            else:
                split_mode = "ratio"
        self.split_mode = split_mode

        # 设置分割比例
        if split_ratio is None:
            split_ratio = self.DEFAULT_SPLIT_RATIO
        self.split_ratio = split_ratio

        # 计算分割点
        self._compute_splits()

        # 标准化（只在训练集上 fit）
        self.scaler = StandardScaler()
        if scale:
            train_data = self.data[:self.train_end]
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.data)

    def _compute_splits(self):
        """计算训练/验证/测试的分割点"""
        T = len(self.data)

        if self.split_mode == "standard" and self.dataset_name in self.STANDARD_SPLITS:
            self.train_end, self.val_end, self.test_end = self.STANDARD_SPLITS[self.dataset_name]
            # 确保不超过数据长度
            self.test_end = min(self.test_end, T)
        else:
            train_ratio, val_ratio, test_ratio = self.split_ratio
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

            self.train_end = int(T * train_ratio)
            self.val_end = int(T * (train_ratio + val_ratio))
            self.test_end = T

    def get_split_info(self) -> dict:
        """获取分割信息"""
        return {
            "train": (0, self.train_end),
            "val": (self.train_end, self.val_end),
            "test": (self.val_end, self.test_end),
            "train_size": self.train_end,
            "val_size": self.val_end - self.train_end,
            "test_size": self.test_end - self.val_end,
        }

    def create_dataset(
        self,
        flag: str,
        task: str,
        **kwargs,
    ):
        """创建指定任务和分割的 Dataset

        Args:
            flag: "train" | "val" | "test"
            task: "forecast" | "imputation" | "generation" | "conditional_generation" | "classification" | "anomaly"
            **kwargs: 传递给 Dataset 的参数

        Returns:
            Dataset 实例
        """
        if flag == "train":
            start, end = 0, self.train_end
        elif flag == "val":
            start, end = self.train_end, self.val_end
        elif flag == "test":
            start, end = self.val_end, self.test_end
        else:
            raise ValueError(f"Unknown flag: {flag}")

        data = self.data[start:end]
        labels = self.labels[start:end] if self.labels is not None else None
        offset = start  # 关键：设置偏移量

        # 自动获取时间特征
        x_mark = self.time_marks[start:end] if self.time_marks is not None else None
        y_mark = x_mark  # 预测任务中 y_mark 通常与 x_mark 相同

        # 条件信息（用于条件生成任务）
        text_emb = self.text_emb[start:end] if self.text_emb is not None else None
        attrs = self.attrs[start:end] if self.attrs is not None else None

        # 通用参数
        common_kwargs = {
            "scale": False,  # 已经在 DataModule 中标准化
            "offset": offset,
        }

        # 如果用户没有显式指定 x_mark/y_mark，则自动传入
        if "x_mark" not in kwargs and x_mark is not None:
            common_kwargs["x_mark"] = x_mark
        if "y_mark" not in kwargs and y_mark is not None and task == "forecast":
            common_kwargs["y_mark"] = y_mark

        # 条件生成任务需要特殊处理
        if task == "conditional_generation":
            if "text_emb" not in kwargs and text_emb is not None:
                common_kwargs["text_emb"] = text_emb
            if "attrs" not in kwargs and attrs is not None:
                common_kwargs["attrs"] = attrs
            if "labels" not in kwargs and labels is not None:
                common_kwargs["labels"] = labels

        common_kwargs.update(kwargs)
        common_kwargs["stride"] = self._resolve_split_stride(
            flag, int(common_kwargs.get("stride", 1))
        )

        if task == "forecast":
            return ForecastDataset(data=data, **common_kwargs)
        elif task == "imputation":
            return ImputationDataset(data=data, **common_kwargs)
        elif task == "generation":
            return GenerationDataset(data=data, **common_kwargs)
        elif task == "conditional_generation":
            return ConditionalGenerationDataset(data=data, **common_kwargs)
        elif task == "classification":
            return ClassificationDataset(data=data, labels=labels, **common_kwargs)
        elif task == "anomaly":
            mode = "train" if flag == "train" else "test"
            return AnomalyDataset(data=data, labels=labels, mode=mode, **common_kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

    @staticmethod
    def _resolve_split_stride(flag: str, stride: int) -> int:
        """Map signed stride semantics to the positive Dataset stride.

        Positive stride subsamples only the test split. Negative stride applies
        ``abs(stride)`` to train, val, and test.
        """
        if stride == 0:
            raise ValueError("stride must be non-zero")
        if stride < 0:
            return abs(stride)
        return stride if flag == "test" else 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """逆标准化"""
        return self.scaler.inverse_transform(data)

    @staticmethod
    def from_contsg_folder(
        data_folder: Union[str, Path],
        split: str = "train",
        normalize: bool = True,
    ):
        """从 ConTSG-Bench 格式的文件夹加载数据

        期望的文件结构：
            data_folder/
            ├── {split}_ts.npy          # (N, L, C)
            ├── {split}_caps.npy        # (N,) 文本描述 [可选]
            ├── {split}_cap_emb.npy     # (N, D) 文本嵌入 [可选]
            ├── {split}_attrs_idx.npy   # (N, A) 属性 [可选]
            └── {split}_labels.npy      # (N,) 标签 [可选]

        Args:
            data_folder: 数据文件夹路径
            split: "train" | "valid" | "test"
            normalize: 是否标准化

        Returns:
            PreSplitGenerationDataset 实例
        """
        from .conditional_generation import PreSplitGenerationDataset

        data_folder = Path(data_folder)

        # 加载时序数据
        ts_path = data_folder / f"{split}_ts.npy"
        if not ts_path.exists():
            raise FileNotFoundError(f"Time series file not found: {ts_path}")
        ts = np.load(ts_path)

        # 加载可选的条件信息
        caps_path = data_folder / f"{split}_caps.npy"
        caps = np.load(caps_path, allow_pickle=True) if caps_path.exists() else None

        cap_emb_path = data_folder / f"{split}_cap_emb.npy"
        cap_emb = np.load(cap_emb_path) if cap_emb_path.exists() else None

        attrs_path = data_folder / f"{split}_attrs_idx.npy"
        attrs = np.load(attrs_path) if attrs_path.exists() else None

        labels_path = data_folder / f"{split}_labels.npy"
        labels = np.load(labels_path) if labels_path.exists() else None

        return PreSplitGenerationDataset(
            ts=ts,
            text_emb=cap_emb,
            attrs=attrs,
            labels=labels,
            caps=caps,
            normalize=normalize,
        )


__all__ = ["DataModule"]
