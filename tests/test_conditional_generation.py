"""Tests for conditional generation dataset."""

import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ts_data import ConditionalGenerationDataset, PreSplitGenerationDataset, DataModule


@pytest.fixture
def sample_data():
    """生成测试数据"""
    np.random.seed(42)
    return np.random.randn(1000, 5).astype(np.float32)


@pytest.fixture
def sample_conditions():
    """生成测试条件"""
    np.random.seed(42)
    return {
        "text_emb": np.random.randn(1000, 1024).astype(np.float32),  # 与时间步对齐
        "attrs": np.random.randint(0, 5, size=(1000, 3)),  # 与时间步对齐
        "labels": np.random.randint(0, 3, size=1000),  # 与时间步对齐
    }


class TestConditionalGenerationDataset:
    """测试条件生成数据集"""

    def test_basic(self, sample_data, sample_conditions):
        ds = ConditionalGenerationDataset(
            data=sample_data,
            window_size=96,
            scale=False,
            text_emb=sample_conditions["text_emb"],
            attrs=sample_conditions["attrs"],
            labels=sample_conditions["labels"],
        )

        assert len(ds) == len(sample_data) - 96 + 1
        sample = ds[0]
        assert "x" in sample
        assert "text_emb" in sample
        assert "attrs" in sample
        assert "label" in sample
        assert "idx" in sample
        assert sample["x"].shape == (5, 96)
        assert sample["text_emb"].shape == (1024,)
        assert sample["attrs"].shape == (3,)
        assert isinstance(sample["label"], int)

    def test_without_conditions(self, sample_data):
        ds = ConditionalGenerationDataset(
            data=sample_data,
            window_size=96,
            scale=False,
        )

        sample = ds[0]
        assert sample["text_emb"] is None
        assert sample["attrs"] is None
        assert sample["label"] is None


class TestPreSplitGenerationDataset:
    """测试预分割生成数据集"""

    def test_basic(self):
        np.random.seed(42)
        ts = np.random.randn(100, 128, 5)  # (N, L, F)
        text_emb = np.random.randn(100, 1024)
        attrs = np.random.randint(0, 5, size=(100, 3))
        labels = np.random.randint(0, 3, size=100)
        caps = np.array([f"Sample {i}" for i in range(100)])

        ds = PreSplitGenerationDataset(
            ts=ts,
            text_emb=text_emb,
            attrs=attrs,
            labels=labels,
            caps=caps,
            normalize=True,
        )

        assert len(ds) == 100
        sample = ds[0]
        assert "x" in sample
        assert "tp" in sample
        assert "text_emb" in sample
        assert "attrs" in sample
        assert "label" in sample
        assert "cap" in sample
        assert "idx" in sample

        assert sample["x"].shape == (5, 128)  # [F, L]
        assert sample["tp"].shape == (128,)
        assert sample["text_emb"].shape == (1024,)
        assert sample["attrs"].shape == (3,)
        assert isinstance(sample["label"], int)
        assert isinstance(sample["cap"], str)

    def test_without_conditions(self):
        np.random.seed(42)
        ts = np.random.randn(100, 128, 5)

        ds = PreSplitGenerationDataset(
            ts=ts,
            normalize=False,
        )

        sample = ds[0]
        assert sample["text_emb"] is None
        assert sample["attrs"] is None
        assert sample["label"] is None
        assert sample["cap"] is None


class TestDataModuleConditionalGeneration:
    """测试 DataModule 的条件生成功能"""

    def test_conditional_generation_task(self, sample_data, sample_conditions):
        dm = DataModule(
            data=sample_data,
            text_emb=sample_conditions["text_emb"],
            attrs=sample_conditions["attrs"],
            labels=sample_conditions["labels"],
            split_ratio=(0.6, 0.2, 0.2),
            scale=True,
        )

        train_ds = dm.create_dataset("train", "conditional_generation", window_size=96)
        val_ds = dm.create_dataset("val", "conditional_generation", window_size=96)

        # 验证 Dataset 创建成功
        assert len(train_ds) > 0
        assert len(val_ds) > 0

        # 验证样本格式
        sample = train_ds[0]
        assert sample["x"].shape == (5, 96)
        assert sample["text_emb"] is not None
        assert sample["attrs"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
