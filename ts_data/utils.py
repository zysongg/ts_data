"""Data loading utilities."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union


# TSFLib 数据集元信息
DATASET_INFO = {
    # ============== common 类别 ==============
    # ETT 系列
    "etth1": {"num_features": 7, "frequency": "h", "length": 17420, "category": "common"},
    "etth2": {"num_features": 7, "frequency": "h", "length": 17420, "category": "common"},
    "ettm1": {"num_features": 7, "frequency": "15min", "length": 69680, "category": "common"},
    "ettm2": {"num_features": 7, "frequency": "15min", "length": 69680, "category": "common"},
    # 天气/环境
    "weather": {"num_features": 21, "frequency": "10min", "length": 52696, "category": "common"},
    "global_temp": {"num_features": 3850, "frequency": "h", "length": 17544, "category": "common"},
    "solar": {"num_features": 137, "frequency": "10min", "length": 52560, "category": "common"},
    # 交通
    "traffic": {"num_features": 862, "frequency": "h", "length": 17544, "category": "common"},
    "metr_la": {"num_features": 207, "frequency": "5min", "length": 34272, "category": "common"},
    "pems03": {"num_features": 358, "frequency": "5min", "length": 26208, "category": "common"},
    "pems04": {"num_features": 307, "frequency": "5min", "length": 16992, "category": "common"},
    "pems07": {"num_features": 883, "frequency": "5min", "length": 28224, "category": "common"},
    "pems08": {"num_features": 170, "frequency": "5min", "length": 17856, "category": "common"},
    "taxi30": {"num_features": 1214, "frequency": "10min", "length": 20477, "category": "common"},
    # 能源/电力
    "electricity": {"num_features": 321, "frequency": "h", "length": 26304, "category": "common"},
    "ecl": {"num_features": 321, "frequency": "h", "length": 26304, "category": "common"},  # electricity 别名
    # 金融/经济
    "exchange": {"num_features": 8, "frequency": "d", "length": 7588, "category": "common"},
    # 健康
    "illness": {"num_features": 7, "frequency": "w", "length": 966, "category": "common"},
    # 网络/流量
    "wiki2000": {"num_features": 2000, "frequency": "d", "length": 1704, "category": "common"},
    "honeypot_fyi": {"num_features": 6, "frequency": "h", "length": 2161, "category": "common"},
    # 合成数据 (AOS - Adaptive Oscillator System)
    "aos_amplitude": {"num_features": 16, "frequency": "h", "length": 12000, "category": "common"},
    "aos_frequency": {"num_features": 16, "frequency": "h", "length": 12000, "category": "common"},
    "aos_mixed": {"num_features": 16, "frequency": "h", "length": 12000, "category": "common"},
    "aos_phase": {"num_features": 16, "frequency": "h", "length": 12000, "category": "common"},
    "aos_stationary": {"num_features": 16, "frequency": "h", "length": 12000, "category": "common"},
    # ============== tfb 类别 ==============
    # 空气质量
    "aq_shunyi": {"num_features": 11, "frequency": "h", "length": 35064, "category": "tfb"},
    "aq_wan": {"num_features": 11, "frequency": "h", "length": 35064, "category": "tfb"},
    # 天气/环境
    "weather_nor": {"num_features": 21, "frequency": "10min", "length": 52696, "category": "tfb"},
    "weather_ab": {"num_features": 21, "frequency": "10min", "length": 52696, "category": "tfb"},
    "wind": {"num_features": 7, "frequency": "10min", "length": 48673, "category": "tfb"},
    "zaf_noo": {"num_features": 11, "frequency": "10min", "length": 19225, "category": "tfb"},
    "cze_lan": {"num_features": 11, "frequency": "10min", "length": 19934, "category": "tfb"},
    # 交通
    "pems_bay": {"num_features": 325, "frequency": "5min", "length": 52116, "category": "tfb"},
    # 金融/经济
    "fred_md": {"num_features": 107, "frequency": "w", "length": 728, "category": "tfb"},
    "nasdaq": {"num_features": 5, "frequency": "d", "length": 1244, "category": "tfb"},
    "nyse": {"num_features": 5, "frequency": "d", "length": 1243, "category": "tfb"},
    # 健康/社会
    "covid19": {"num_features": 948, "frequency": "d", "length": 1392, "category": "tfb"},
    # ============== workload 类别 ==============
    # 云计算负载
    "faas": {"num_features": 226, "frequency": "5min", "length": 2305, "category": "workload"},
    "iaas": {"num_features": 93, "frequency": "5min", "length": 3456, "category": "workload"},
    "paas": {"num_features": 426, "frequency": "5min", "length": 7776, "category": "workload"},
    "rds": {"num_features": 1113, "frequency": "5min", "length": 6624, "category": "workload"},
}


def load_csv(
    filepath: str,
    target_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    date_col: Optional[str] = "date",
    freq: Optional[str] = "h",
    start_date: Optional[str] = None,
    infer_time: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """加载 CSV 格式的时序数据

    Args:
        filepath: CSV 文件路径
        target_col: 目标列名（单变量预测时使用）
        feature_cols: 特征列名列表（多变量时使用）
        date_col: 日期列名，设置为 None 时不使用日期列
        freq: 频率（用于时间特征提取或自动推断）
        start_date: 起始日期字符串（如 "2020-01-01"），用于自动推断时间
        infer_time: 是否自动推断时间（即使没有日期列）

    Returns:
        (data, time_marks, dates) 元组
        - data: [T, F] numpy 数组
        - time_marks: [T, time_dim] 时间特征，如果没有日期列则为 None
        - dates: DatetimeIndex，如果没有日期列则为 None
    """
    df = pd.read_csv(filepath)

    # 提取日期列
    dates = None
    time_marks = None

    if date_col is not None and date_col in df.columns:
        # 从 CSV 中读取日期
        dates = pd.to_datetime(df[date_col])
        time_marks = _time_features(dates, freq=freq)
        df = df.drop(columns=[date_col])
    elif infer_time or start_date is not None:
        # 自动推断时间
        if start_date is None:
            start_date = "2020-01-01"
        if freq is None:
            freq = "h"
        num_timesteps = len(df)
        dates = pd.date_range(start=start_date, periods=num_timesteps, freq=freq)
        time_marks = _time_features(dates, freq=freq)

    # 提取特征列
    if feature_cols is not None:
        df = df[feature_cols]
    elif target_col is not None:
        df = df[[target_col]]

    data = df.values.astype(np.float32)
    return data, time_marks, dates


def load_npy(
    filepath: str,
    time_file: Optional[str] = None,
    freq: Optional[str] = None,
    start_date: Optional[str] = None,
    infer_time: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """加载 NPY 格式的时序数据

    Args:
        filepath: NPY 文件路径
        time_file: 时间特征文件路径（可选，支持 npy/npz/csv）
        freq: 频率（用于自动推断时间特征）
        start_date: 起始日期字符串（如 "2020-01-01"），用于自动推断时间
        infer_time: 是否自动推断时间（即使没有指定其他参数）

    Returns:
        (data, time_marks, dates) 元组
    """
    data = np.load(filepath).astype(np.float32)
    time_marks = None
    dates = None

    # 加载时间信息
    if time_file is not None:
        # 从单独的时间文件加载
        time_marks, dates = _load_time_marks(time_file, freq=freq)
    elif infer_time or start_date is not None or (freq is not None):
        # 自动推断时间特征
        num_timesteps = data.shape[0]
        if start_date is None and freq is None:
            start_date = "2020-01-01"
            freq = "h"
        elif freq is None:
            freq = "h"

        dates = pd.date_range(
            start=start_date or "2020-01-01",
            periods=num_timesteps,
            freq=freq,
        )
        time_marks = _time_features(dates, freq=freq)

    return data, time_marks, dates


def load_data(
    filepath: str,
    file_format: Optional[str] = None,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """自动加载时序数据

    Args:
        filepath: 文件路径
        file_format: 文件格式 ("csv" | "npy" | "npy_dict")，如果不指定则根据扩展名推断
        **kwargs: 传递给具体加载函数的参数

    Returns:
        (data, time_marks, dates) 元组
    """
    if file_format is None:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            file_format = "csv"
        elif ext in [".npy", ".npz"]:
            # 尝试检测是否为 TSFLib 格式
            try:
                data_dict = np.load(filepath, allow_pickle=True).item()
                if isinstance(data_dict, dict) and "data" in data_dict:
                    file_format = "npy_dict"
                else:
                    file_format = "npy"
            except (ValueError, AttributeError):
                file_format = "npy"
        else:
            raise ValueError(f"Unknown file format: {ext}")

    if file_format == "csv":
        return load_csv(filepath, **kwargs)
    elif file_format == "npy":
        return load_npy(filepath, **kwargs)
    elif file_format == "npy_dict":
        return load_npy_dict(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def _time_features(dates: pd.DatetimeIndex, freq: Optional[str] = "h") -> np.ndarray:
    """提取时间特征

    Args:
        dates: DatetimeIndex
        freq: 频率

    Returns:
        time_features: [T, time_dim] numpy 数组
    """
    if freq is None:
        freq = "h"
    freq = freq.lower()

    features = []
    features.append(dates.month.values)
    features.append(dates.day.values)
    features.append(dates.weekday.values)
    features.append(dates.hour.values)

    if freq in ["min", "t"]:
        features.append(dates.minute.values)

    return np.stack(features, axis=-1).astype(np.float32)


def _load_time_marks(
    time_file: str,
    freq: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
    """加载时间特征文件

    Args:
        time_file: 时间特征文件路径
        freq: 频率（用于从 csv 解析时间特征）

    Returns:
        (time_marks, dates) 元组
    """
    ext = os.path.splitext(time_file)[1].lower()
    dates = None

    if ext == ".npy":
        time_marks = np.load(time_file).astype(np.float32)
    elif ext == ".npz":
        npz = np.load(time_file)
        time_marks = npz["time_marks"].astype(np.float32)
    elif ext == ".csv":
        # CSV 格式：读取为时间序列
        df = pd.read_csv(time_file)
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            time_marks = _time_features(dates, freq=freq)
        else:
            time_marks = df.values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported time file format: {ext}")

    return time_marks, dates


def load_npy_dict(
    filepath: str,
    freq: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """加载 TSFLib 格式的 npy 文件（字典格式）

    TSFLib 数据集使用 np.save(..., allow_pickle=True) 保存字典，包含：
    - data: [T, F] numpy 数组
    - time_date: pd.DatetimeIndex

    Args:
        filepath: NPY 文件路径
        freq: 频率（用于时间特征提取，如果不指定则尝试推断）

    Returns:
        (data, time_marks, dates) 元组
    """
    data_dict = np.load(filepath, allow_pickle=True).item()

    # 提取数据
    data = data_dict["data"].astype(np.float32)
    dates = data_dict.get("time_date", None)
    time_marks = None

    if dates is not None and isinstance(dates, pd.DatetimeIndex):
        # 从日期提取时间特征
        if freq is None:
            # 尝试从数据推断频率
            freq = _infer_freq(dates)
        time_marks = _time_features(dates, freq=freq)
    elif freq is not None:
        # 自动推断时间特征
        num_timesteps = data.shape[0]
        dates = pd.date_range(start="2020-01-01", periods=num_timesteps, freq=freq)
        time_marks = _time_features(dates, freq=freq)

    return data, time_marks, dates


def _infer_freq(dates: pd.DatetimeIndex) -> str:
    """从时间索引推断频率"""
    if len(dates) < 2:
        return "h"

    delta = dates[1] - dates[0]
    if delta >= pd.Timedelta(days=7):
        return "w"
    elif delta >= pd.Timedelta(days=1):
        return "d"
    elif delta >= pd.Timedelta(hours=1):
        return "h"
    elif delta >= pd.Timedelta(minutes=1):
        return "min"
    else:
        return "s"


def get_data_path(
    dataset_name: str,
    category: str = "common",
    data_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """获取 TSFLib 数据集路径

    Args:
        dataset_name: 数据集名称（如 "etth1"）
        category: 数据类别 ("common" | "tfb" | "workload")
        data_dir: 数据根目录，如果不指定则从环境变量 TSDATADIR 读取

    Returns:
        数据集文件路径
    """
    if data_dir is None:
        data_dir = os.environ.get("TSDATADIR")
        if data_dir is None:
            raise ValueError(
                "data_dir is not provided. Set TSDATADIR environment variable or pass data_dir."
            )

    root = Path(data_dir).expanduser()
    category_dir = root / category
    data_path = category_dir / f"{dataset_name}.npy"

    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    return data_path


def get_dataset_info(name: str) -> dict:
    """获取数据集元信息

    Args:
        name: 数据集名称

    Returns:
        包含 num_features, frequency, length 的字典
    """
    name_lower = name.lower()
    if name_lower not in DATASET_INFO:
        available = ", ".join(DATASET_INFO.keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available}")
    return DATASET_INFO[name_lower].copy()


__all__ = [
    "load_csv",
    "load_npy",
    "load_npy_dict",
    "load_data",
    "get_data_path",
    "get_dataset_info",
    "DATASET_INFO",
]
