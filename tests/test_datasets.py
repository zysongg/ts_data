"""Tests for ts_data package."""

import numpy as np
import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ts_data import (
    ForecastDataset,
    ImputationDataset,
    GenerationDataset,
    ClassificationDataset,
    AnomalyDataset,
    DataModule,
)


@pytest.fixture
def sample_data():
    """生成测试数据"""
    np.random.seed(42)
    return np.random.randn(1000, 5).astype(np.float32)


@pytest.fixture
def sample_labels():
    """生成测试标签"""
    np.random.seed(42)
    return np.random.randint(0, 3, size=1000)


class TestForecastDataset:
    """测试预测数据集"""

    def test_basic(self, sample_data):
        ds = ForecastDataset(
            data=sample_data,
            input_len=96,
            pred_len=96,
            scale=False,
        )

        assert len(ds) == len(sample_data) - 96 - 96 + 1
        sample = ds[0]
        assert "x" in sample
        assert "y" in sample
        assert "idx" in sample
        assert sample["x"].shape == (5, 96)
        assert sample["y"].shape == (5, 96)
        assert sample["idx"] == 0

    def test_with_label_len(self, sample_data):
        ds = ForecastDataset(
            data=sample_data,
            input_len=96,
            pred_len=96,
            label_len=48,
            scale=False,
        )

        sample = ds[0]
        assert sample["y"].shape == (5, 48 + 96)

    def test_stride(self, sample_data):
        ds = ForecastDataset(
            data=sample_data,
            input_len=96,
            pred_len=96,
            stride=2,
            scale=False,
        )

        sample = ds[0]
        assert sample["idx"] == 0
        sample = ds[1]
        assert sample["idx"] == 2


class TestImputationDataset:
    """测试插补数据集"""

    def test_basic(self, sample_data):
        ds = ImputationDataset(
            data=sample_data,
            window_size=96,
            mask_ratio=0.25,
            scale=False,
        )

        sample = ds[0]
        assert "x" in sample
        assert "y" in sample
        assert "mask" in sample
        assert sample["x"].shape == (5, 96)
        assert sample["y"].shape == (5, 96)
        assert sample["mask"].shape == (5, 96)

    def test_mask_reproducible(self, sample_data):
        ds = ImputationDataset(
            data=sample_data,
            window_size=96,
            mask_ratio=0.25,
            seed=42,
            scale=False,
        )

        # 同一索引的 mask 应该相同
        mask1 = ds[0]["mask"]
        mask2 = ds[0]["mask"]
        assert torch.allclose(mask1, mask2)

    def test_mask_modes(self, sample_data):
        for mode in ["random", "block", "forecast"]:
            ds = ImputationDataset(
                data=sample_data,
                window_size=96,
                mask_ratio=0.25,
                mask_mode=mode,
                scale=False,
            )
            sample = ds[0]
            assert sample["mask"] is not None


class TestGenerationDataset:
    """测试生成数据集"""

    def test_basic(self, sample_data):
        ds = GenerationDataset(
            data=sample_data,
            window_size=96,
            scale=False,
        )

        sample = ds[0]
        assert "x" in sample
        assert sample["x"].shape == (5, 96)
        assert sample["y"] is None
        assert sample["mask"] is None


class TestClassificationDataset:
    """测试分类数据集"""

    def test_basic(self, sample_data, sample_labels):
        ds = ClassificationDataset(
            data=sample_data,
            labels=sample_labels,
            window_size=96,
            scale=False,
        )

        sample = ds[0]
        assert "x" in sample
        assert "y" in sample
        assert sample["x"].shape == (5, 96)
        assert isinstance(sample["y"], int)

    def test_label_modes(self, sample_data, sample_labels):
        for mode in ["last", "majority"]:
            ds = ClassificationDataset(
                data=sample_data,
                labels=sample_labels,
                window_size=96,
                label_mode=mode,
                scale=False,
            )
            sample = ds[0]
            assert isinstance(sample["y"], int)


class TestAnomalyDataset:
    """测试异常检测数据集"""

    def test_train_mode(self, sample_data):
        ds = AnomalyDataset(
            data=sample_data,
            window_size=96,
            mode="train",
            scale=False,
        )

        sample = ds[0]
        assert "x" in sample
        assert "y" in sample
        assert sample["x"].shape == (5, 96)
        assert sample["y"].shape == (96,)
        # 训练模式标签为全0
        assert torch.all(sample["y"] == 0)

    def test_test_mode(self, sample_data):
        labels = np.zeros(1000)
        labels[100:200] = 1  # 异常区间

        ds = AnomalyDataset(
            data=sample_data,
            labels=labels,
            window_size=96,
            mode="test",
            scale=False,
        )

        sample = ds[0]
        assert sample["y"].shape == (96,)


class TestDataModule:
    """测试数据模块"""

    def test_ratio_split(self, sample_data):
        dm = DataModule(
            data=sample_data,
            split_ratio=(0.6, 0.2, 0.2),
            scale=True,
        )

        info = dm.get_split_info()
        assert info["train_size"] == 600
        assert info["val_size"] == 200
        assert info["test_size"] == 200

    def test_create_dataset(self, sample_data):
        dm = DataModule(
            data=sample_data,
            split_ratio=(0.6, 0.2, 0.2),
            scale=True,
        )

        train_ds = dm.create_dataset("train", "forecast", input_len=96, pred_len=96)
        val_ds = dm.create_dataset("val", "forecast", input_len=96, pred_len=96)
        test_ds = dm.create_dataset("test", "forecast", input_len=96, pred_len=96)

        # 验证 idx 连续性
        last_train_idx = train_ds[len(train_ds) - 1]["idx"]
        first_val_idx = val_ds[0]["idx"]
        last_val_idx = val_ds[len(val_ds) - 1]["idx"]
        first_test_idx = test_ds[0]["idx"]

        assert first_val_idx > last_train_idx
        assert first_test_idx > last_val_idx

    def test_positive_stride_only_affects_test_split(self, sample_data):
        dm = DataModule(
            data=sample_data,
            split_ratio=(0.6, 0.2, 0.2),
            scale=False,
        )

        train_ds = dm.create_dataset("train", "forecast", input_len=24, pred_len=12, stride=4)
        val_ds = dm.create_dataset("val", "forecast", input_len=24, pred_len=12, stride=4)
        test_ds = dm.create_dataset("test", "forecast", input_len=24, pred_len=12, stride=4)

        assert train_ds.stride == 1
        assert val_ds.stride == 1
        assert test_ds.stride == 4
        assert train_ds[1]["idx"] - train_ds[0]["idx"] == 1
        assert val_ds[1]["idx"] - val_ds[0]["idx"] == 1
        assert test_ds[1]["idx"] - test_ds[0]["idx"] == 4

    def test_negative_stride_affects_all_splits(self, sample_data):
        dm = DataModule(
            data=sample_data,
            split_ratio=(0.6, 0.2, 0.2),
            scale=False,
        )

        train_ds = dm.create_dataset("train", "forecast", input_len=24, pred_len=12, stride=-4)
        val_ds = dm.create_dataset("val", "forecast", input_len=24, pred_len=12, stride=-4)
        test_ds = dm.create_dataset("test", "forecast", input_len=24, pred_len=12, stride=-4)

        assert train_ds.stride == 4
        assert val_ds.stride == 4
        assert test_ds.stride == 4
        assert train_ds[1]["idx"] - train_ds[0]["idx"] == 4
        assert val_ds[1]["idx"] - val_ds[0]["idx"] == 4
        assert test_ds[1]["idx"] - test_ds[0]["idx"] == 4

    def test_etth_split(self, sample_data):
        # ETT 标准分割需要足够长的数据
        long_data = np.random.randn(20000, 5).astype(np.float32)

        dm = DataModule(
            data=long_data,
            split_mode="standard",
            dataset_name="etth1",
            scale=True,
        )

        info = dm.get_split_info()
        assert info["train_size"] == 12 * 30 * 24  # 8640


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
