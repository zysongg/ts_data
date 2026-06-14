"""DataModule for time series data management."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler

from .forecast import ForecastDataset
from .imputation import ImputationDataset
from .generation import GenerationDataset
from .classification import ClassificationDataset
from .anomaly import AnomalyDataset
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

    def __init__(
        self,
        data: Union[np.ndarray, str],
        labels: Optional[np.ndarray] = None,
        split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        split_mode: str = "ratio",
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
        self.split_ratio = split_ratio
        self.split_mode = split_mode
        self.dataset_name = dataset_name.lower() if dataset_name else None
        self.scale = scale

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
            task: "forecast" | "imputation" | "generation" | "classification" | "anomaly"
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

        common_kwargs.update(kwargs)

        if task == "forecast":
            return ForecastDataset(data=data, **common_kwargs)
        elif task == "imputation":
            return ImputationDataset(data=data, **common_kwargs)
        elif task == "generation":
            return GenerationDataset(data=data, **common_kwargs)
        elif task == "classification":
            return ClassificationDataset(data=data, labels=labels, **common_kwargs)
        elif task == "anomaly":
            mode = "train" if flag == "train" else "test"
            return AnomalyDataset(data=data, labels=labels, mode=mode, **common_kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """逆标准化"""
        return self.scaler.inverse_transform(data)


__all__ = ["DataModule"]
