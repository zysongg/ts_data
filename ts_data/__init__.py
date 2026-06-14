"""ts_data: Time series data loading and processing library."""

from .base import BaseTimeSeriesDataset
from .forecast import ForecastDataset
from .imputation import ImputationDataset
from .generation import GenerationDataset
from .classification import ClassificationDataset
from .anomaly import AnomalyDataset
from .conditional_generation import ConditionalGenerationDataset, PreSplitGenerationDataset
from .datamodule import DataModule
from .utils import (
    load_data,
    load_csv,
    load_npy,
    load_npy_dict,
    get_data_path,
    get_dataset_info,
    DATASET_INFO,
)

__version__ = "0.2.0"

__all__ = [
    "BaseTimeSeriesDataset",
    "ForecastDataset",
    "ImputationDataset",
    "GenerationDataset",
    "ClassificationDataset",
    "AnomalyDataset",
    "ConditionalGenerationDataset",
    "PreSplitGenerationDataset",
    "DataModule",
    "load_data",
    "load_csv",
    "load_npy",
    "load_npy_dict",
    "get_data_path",
    "get_dataset_info",
    "DATASET_INFO",
]
