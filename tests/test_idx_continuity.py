"""验证 idx 在不同 stride 下的连续性"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ts_data import DataModule


def test_idx_continuity_across_splits():
    """测试 train/val/test 的 idx 连续性"""
    np.random.seed(42)
    data = np.random.randn(1000, 5).astype(np.float32)

    for stride in [-1, -2, -3, -5, -10]:
        dm = DataModule(
            data=data,
            split_ratio=(0.6, 0.2, 0.2),
            scale=False,  # 方便观察
        )

        train_ds = dm.create_dataset("train", "forecast", input_len=96, pred_len=96, stride=stride)
        val_ds = dm.create_dataset("val", "forecast", input_len=96, pred_len=96, stride=stride)
        test_ds = dm.create_dataset("test", "forecast", input_len=96, pred_len=96, stride=stride)

        # 获取每个分割的最后一个 idx
        last_train_idx = train_ds[len(train_ds) - 1]["idx"]
        first_val_idx = val_ds[0]["idx"]
        last_val_idx = val_ds[len(val_ds) - 1]["idx"]
        first_test_idx = test_ds[0]["idx"]

        print(f"\n=== stride={stride} ===")
        print(f"train: idx range [0, {last_train_idx}]")
        print(f"val:   idx range [{first_val_idx}, {last_val_idx}]")
        print(f"test:  idx range [{first_test_idx}, ...]")

        # 验证 idx 表示窗口在原始序列中的起始位置
        # train: [0, train_end)
        # val: [train_end, val_end)
        # test: [val_end, test_end)
        train_end = dm.train_end
        val_end = dm.val_end

        # val 的第一个 idx 应该是 train_end（因为窗口从 train_end 开始）
        assert first_val_idx == train_end, \
            f"stride={stride}: val first idx should be {train_end}, got {first_val_idx}"

        # test 的第一个 idx 应该是 val_end（因为窗口从 val_end 开始）
        assert first_test_idx == val_end, \
            f"stride={stride}: test first idx should be {val_end}, got {first_test_idx}"

        print(f"✓ idx 连续性验证通过")


def test_idx_within_split():
    """测试单个分割内 idx 的连续性"""
    np.random.seed(42)
    data = np.random.randn(1000, 5).astype(np.float32)

    for stride in [-1, -2, -3, -5, -10]:
        dm = DataModule(
            data=data,
            split_ratio=(0.6, 0.2, 0.2),
            scale=False,
        )

        for flag in ["train", "val", "test"]:
            ds = dm.create_dataset(flag, "forecast", input_len=96, pred_len=96, stride=stride)

            # 验证相邻样本的 idx 差为 stride
            for i in range(min(10, len(ds) - 1)):
                idx_i = ds[i]["idx"]
                idx_next = ds[i + 1]["idx"]
                expected_stride = abs(stride)
                assert idx_next - idx_i == expected_stride, \
                    f"stride={stride}, {flag}: idx[{i}]={idx_i}, idx[{i+1}]={idx_next}, diff={idx_next - idx_i}"

            print(f"stride={stride}, {flag}: ✓ 内部 idx 连续性验证通过")


def test_idx_matches_original_sequence():
    """验证 idx 确实指向原始序列的正确位置"""
    np.random.seed(42)
    data = np.random.randn(100, 3).astype(np.float32)

    stride = -5
    dm = DataModule(
        data=data,
        split_ratio=(0.6, 0.2, 0.2),
        scale=False,
    )

    train_ds = dm.create_dataset("train", "forecast", input_len=10, pred_len=5, stride=stride)

    # 验证几个样本的 idx
    for i in range(min(5, len(train_ds))):
        sample = train_ds[i]
        idx = sample["idx"]
        expected_idx = i * abs(stride)
        assert idx == expected_idx, f"sample[{i}]: idx should be {expected_idx}, got {idx}"

    print("✓ idx 与原始序列位置匹配验证通过")


def test_all_tasks_idx():
    """测试所有任务的 idx 连续性"""
    np.random.seed(42)
    data = np.random.randn(1000, 5).astype(np.float32)
    labels = np.random.randint(0, 3, size=1000)
    anomaly_labels = np.zeros(1000)
    anomaly_labels[100:200] = 1

    stride = -3

    # 预测任务
    dm = DataModule(data=data, split_ratio=(0.6, 0.2, 0.2), scale=False)
    train_ds = dm.create_dataset("train", "forecast", input_len=96, pred_len=96, stride=stride)
    val_ds = dm.create_dataset("val", "forecast", input_len=96, pred_len=96, stride=stride)
    test_ds = dm.create_dataset("test", "forecast", input_len=96, pred_len=96, stride=stride)
    _check_continuity(train_ds, val_ds, test_ds, stride, "forecast")

    # 插补任务
    train_ds = dm.create_dataset("train", "imputation", window_size=96, stride=stride)
    val_ds = dm.create_dataset("val", "imputation", window_size=96, stride=stride)
    test_ds = dm.create_dataset("test", "imputation", window_size=96, stride=stride)
    _check_continuity(train_ds, val_ds, test_ds, stride, "imputation")

    # 生成任务
    train_ds = dm.create_dataset("train", "generation", window_size=96, stride=stride)
    val_ds = dm.create_dataset("val", "generation", window_size=96, stride=stride)
    test_ds = dm.create_dataset("test", "generation", window_size=96, stride=stride)
    _check_continuity(train_ds, val_ds, test_ds, stride, "generation")

    # 分类任务
    dm_cls = DataModule(data=data, labels=labels, split_ratio=(0.6, 0.2, 0.2), scale=False)
    train_ds = dm_cls.create_dataset("train", "classification", window_size=96, stride=stride)
    val_ds = dm_cls.create_dataset("val", "classification", window_size=96, stride=stride)
    test_ds = dm_cls.create_dataset("test", "classification", window_size=96, stride=stride)
    _check_continuity(train_ds, val_ds, test_ds, stride, "classification")

    # 异常检测任务
    dm_anom = DataModule(data=data, labels=anomaly_labels, split_ratio=(0.6, 0.2, 0.2), scale=False)
    train_ds = dm_anom.create_dataset("train", "anomaly", window_size=96, stride=stride)
    val_ds = dm_anom.create_dataset("val", "anomaly", window_size=96, stride=stride)
    test_ds = dm_anom.create_dataset("test", "anomaly", window_size=96, stride=stride)
    _check_continuity(train_ds, val_ds, test_ds, stride, "anomaly")


def _check_continuity(train_ds, val_ds, test_ds, stride, task_name):
    """检查三个分割的 idx 连续性
    
    idx 表示窗口在原始序列中的起始位置，即 offset + i * stride
    """
    first_val = val_ds[0]["idx"]
    first_test = test_ds[0]["idx"]

    # val 的第一个 idx 应该是 val_ds.offset
    assert first_val == val_ds.offset, \
        f"{task_name}: val first idx should be {val_ds.offset}, got {first_val}"

    # test 的第一个 idx 应该是 test_ds.offset
    assert first_test == test_ds.offset, \
        f"{task_name}: test first idx should be {test_ds.offset}, got {first_test}"

    print(f"✓ {task_name}: idx 连续性验证通过")


if __name__ == "__main__":
    print("=" * 60)
    print("验证 idx 在不同 stride 下的连续性")
    print("=" * 60)

    test_idx_continuity_across_splits()
    print()
    test_idx_within_split()
    print()
    test_idx_matches_original_sequence()
    print()
    test_all_tasks_idx()
    print()
    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
