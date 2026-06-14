# ts_data

时序数据加载和处理库，为时序任务提供标准化的 Dataset 接口。

## 安装

```bash
cd TSLib-tool/ts_data
pip install -e .
```

## 功能特性

- 支持 5 种时序任务：预测、插补、生成、分类、异常检测
- 支持多种数据格式：CSV、NPY
- 支持自动/手动时间戳推断
- 支持自定义滑窗步长（stride）
- 支持 train/val/test 分割
- 支持标准化（StandardScaler）
- 返回统一的字典格式

## 快速开始

### 基础用法

```python
import numpy as np
from ts_data import DataModule
from torch.utils.data import DataLoader

# 方式 1: 传入 numpy 数组
data = np.random.randn(10000, 7)  # [T, F]
dm = DataModule(data=data, split_ratio=(0.6, 0.2, 0.2), scale=True)

# 方式 2: 传入文件路径
dm = DataModule(data="path/to/data.csv", date_col="date", freq="h")

# 创建 Dataset
train_ds = dm.create_dataset("train", "forecast", input_len=96, pred_len=96)
val_ds = dm.create_dataset("val", "forecast", input_len=96, pred_len=96)
test_ds = dm.create_dataset("test", "forecast", input_len=96, pred_len=96)

# 创建 DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
```

### 从文件加载

```python
from ts_data import DataModule

# CSV 文件（有日期列）
dm = DataModule(data="data.csv", date_col="date", freq="h")

# CSV 文件（无日期列，自动推断）
dm = DataModule(data="data.csv", start_date="2020-01-01", freq="h")

# NPY 文件（自动推断时间）
dm = DataModule(data="data.npy", freq="h", start_date="2020-01-01")

# NPY 文件（从单独的时间文件加载）
dm = DataModule(data="data.npy", time_file="time_marks.npy")

# TSFLib 格式 NPY 文件（自动检测）
dm = DataModule(data="path/to/etth1.npy", split_ratio=(0.6, 0.2, 0.2))
```

## 支持的任务

### 1. 预测任务 (Forecast)

```python
ds = dm.create_dataset(
    flag="train",
    task="forecast",
    input_len=96,      # 输入序列长度
    pred_len=96,       # 预测序列长度
    label_len=48,      # 标签序列长度（用于 teacher forcing）
    stride=1,          # 滑窗步长
)

sample = ds[0]
# sample = {
#     "x": Tensor [F, 96],         # 输入序列
#     "y": Tensor [F, 144],        # 目标序列（label_len + pred_len）
#     "mask": None,
#     "x_mark": Tensor [96, 4],    # 输入时间特征
#     "y_mark": Tensor [144, 4],   # 目标时间特征
#     "idx": int,                  # 窗口起始位置
# }
```

### 2. 插补任务 (Imputation)

```python
ds = dm.create_dataset(
    flag="train",
    task="imputation",
    window_size=96,
    mask_ratio=0.25,        # 缺失比例
    mask_mode="random",     # "random" | "block" | "forecast"
    seed=42,                # 随机种子（用于可复现的 mask）
    stride=1,
)

sample = ds[0]
# sample = {
#     "x": Tensor [F, 96],         # mask 后的序列
#     "y": Tensor [F, 96],         # 完整序列
#     "mask": Tensor [F, 96],      # mask (1=观测, 0=缺失)
#     "x_mark": Tensor [96, 4],
#     "y_mark": None,
#     "idx": int,
# }
```

### 3. 生成任务 (Generation)

```python
ds = dm.create_dataset(
    flag="train",
    task="generation",
    window_size=96,
    stride=1,
)

sample = ds[0]
# sample = {
#     "x": Tensor [F, 96],
#     "y": None,
#     "mask": None,
#     "x_mark": Tensor [96, 4],
#     "y_mark": None,
#     "idx": int,
# }
```

### 4. 分类任务 (Classification)

```python
# 需要传入标签
labels = np.random.randint(0, 3, size=10000)  # [T] 或 [num_windows]
dm = DataModule(data=data, labels=labels, split_ratio=(0.6, 0.2, 0.2))

ds = dm.create_dataset(
    flag="train",
    task="classification",
    window_size=96,
    stride=1,
    label_mode="last",      # "last" | "majority"
)

sample = ds[0]
# sample = {
#     "x": Tensor [F, 96],
#     "y": int,                    # 类别标签
#     "mask": None,
#     "x_mark": Tensor [96, 4],
#     "y_mark": None,
#     "idx": int,
# }
```

### 5. 异常检测任务 (Anomaly Detection)

```python
# 需要传入异常标签
labels = np.zeros(10000)
labels[1000:1500] = 1  # 异常区间
dm = DataModule(data=data, labels=labels, split_ratio=(0.6, 0.2, 0.2))

train_ds = dm.create_dataset("train", "anomaly", window_size=96)
test_ds = dm.create_dataset("test", "anomaly", window_size=96)  # test 模式需要真实标签

sample = train_ds[0]
# sample = {
#     "x": Tensor [F, 96],
#     "y": Tensor [96],            # 训练时为全0占位符
#     "mask": None,
#     "x_mark": Tensor [96, 4],
#     "y_mark": None,
#     "idx": int,
# }
```

## 频率格式说明

`freq` 参数用于指定数据的时间频率，支持以下格式：

| 频率 | 别名 | 说明 | 示例 |
|------|------|------|------|
| `"h"` | `"H"` | 小时 | 每小时一条记录 |
| `"min"` | `"T"` | 分钟 | 每分钟一条记录 |
| `"d"` | `"D"` | 天 | 每天一条记录 |
| `"w"` | `"W"` | 周 | 每周一条记录 |
| `"m"` | `"MS"` | 月 | 每月一条记录 |
| `"s"` | `"S"` | 秒 | 每秒一条记录 |
| `"q"` | `"Q"` | 季度 | 每季度一条记录 |
| `"y"` | `"Y"`, `"A"` | 年 | 每年一条记录 |

### 组合频率

支持数字前缀表示倍数：

| 格式 | 说明 |
|------|------|
| `"2h"` | 每 2 小时 |
| `"15min"` | 每 15 分钟 |
| `"30T"` | 每 30 分钟 |
| `"7d"` | 每 7 天 |
| `"2w"` | 每 2 周 |

### 使用示例

```python
# 每小时数据
dm = DataModule(data="etth1.csv", freq="h")

# 每 15 分钟数据
dm = DataModule(data="ettm1.csv", freq="15min")

# 每天数据
dm = DataModule(data="daily.csv", freq="d")

# 每周数据
dm = DataModule(data="weekly.csv", freq="w")
```

## 时间特征

根据 `freq` 自动提取的时间特征维度：

| 特征 | 范围 | 所有频率 | 分钟级 |
|------|------|----------|--------|
| month | 1-12 | ✓ | ✓ |
| day | 1-31 | ✓ | ✓ |
| weekday | 0-6 | ✓ | ✓ |
| hour | 0-23 | ✓ | ✓ |
| minute | 0-59 | - | ✓ |

**注意**：当 `freq="min"` 或 `freq="T"` 时，会额外包含 `minute` 特征。

## 分割模式

### 1. 比例分割

```python
dm = DataModule(
    data=data,
    split_ratio=(0.6, 0.2, 0.2),  # train, val, test
    split_mode="ratio",
)
```

### 2. 标准分割（ETT 数据集）

```python
dm = DataModule(
    data=data,
    split_mode="standard",
    dataset_name="etth1",  # 自动使用 ETT 标准分割点
)
```

### 3. 获取分割信息

```python
info = dm.get_split_info()
# {
#     "train": (0, 8640),
#     "val": (8640, 11520),
#     "test": (11520, 14400),
#     "train_size": 8640,
#     "val_size": 2880,
#     "test_size": 2880,
# }
```

## idx 连续性

`idx` 表示窗口在原始序列中的起始位置，保证 train/val/test 之间连续：

```python
# 数据长度 1000，分割 60%/20%/20%
train_ds = dm.create_dataset("train", "forecast", input_len=96, pred_len=96)
val_ds = dm.create_dataset("val", "forecast", input_len=96, pred_len=96)
test_ds = dm.create_dataset("test", "forecast", input_len=96, pred_len=96)

# train: idx ∈ [0, 408]
# val:   idx ∈ [600, 608]
# test:  idx ∈ [800, ...]
```

## API 参考

### DataModule

```python
class DataModule:
    def __init__(
        self,
        data: Union[np.ndarray, str],     # 数据或文件路径
        labels: Optional[np.ndarray] = None,  # 标签（分类/异常检测）
        split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        split_mode: str = "ratio",        # "ratio" | "standard"
        dataset_name: Optional[str] = None,  # 用于标准分割
        scale: bool = True,               # 是否标准化
        file_format: Optional[str] = None,  # "csv" | "npy"
        # 文件加载参数
        freq: Optional[str] = None,       # 频率
        start_date: Optional[str] = None,  # 起始日期
        infer_time: bool = False,         # 是否自动推断时间
        date_col: Optional[str] = "date",  # CSV 日期列
        **kwargs,
    ):
        ...
    
    def create_dataset(self, flag: str, task: str, **kwargs) -> Dataset:
        ...
    
    def get_split_info(self) -> dict:
        ...
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        ...
```

### Dataset 基类

```python
class BaseTimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,           # [T, F]
        window_size: int,
        stride: int = 1,
        scale: bool = True,
        offset: int = 0,            # 在原始序列中的偏移量
    ):
        ...
    
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, index: int) -> dict:
        ...
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        ...
```

## 文件结构

```
ts_data/
├── ts_data/
│   ├── __init__.py
│   ├── base.py           # 基类
│   ├── forecast.py       # 预测任务
│   ├── imputation.py     # 插补任务
│   ├── generation.py     # 生成任务
│   ├── classification.py # 分类任务
│   ├── anomaly.py        # 异常检测任务
│   ├── datamodule.py     # 数据模块
│   └── utils.py          # 工具函数
├── tests/
│   ├── test_datasets.py
│   └── test_idx_continuity.py
├── setup.py
└── README.md
```

## License

MIT

## TSFLib 数据格式支持

`ts_data` 支持 TSFLib 的数据格式，这种格式使用 `.npy` 文件存储字典，包含数据和日期索引。

### 数据格式

```python
# TSFLib 格式 npy 文件内容
data_dict = {
    "data": np.ndarray,      # [T, F] 数据数组
    "time_date": pd.DatetimeIndex,  # 时间索引
}
```

### 加载 TSFLib 数据

```python
from ts_data import DataModule, get_data_path, get_dataset_info

# 方式 1: 直接加载
dm = DataModule(data="path/to/etth1.npy", split_ratio=(0.6, 0.2, 0.2))

# 方式 2: 使用 get_data_path
data_path = get_data_path("etth1", category="common", data_dir="/path/to/tsdata")
dm = DataModule(data=data_path, split_ratio=(0.6, 0.2, 0.2))

# 方式 3: 使用环境变量 TSDATADIR
# export TSDATADIR="/path/to/tsdata"
dm = DataModule(data="etth1.npy", category="common")
```

### 获取数据集信息

```python
from ts_data import get_dataset_info, DATASET_INFO

# 获取单个数据集信息
info = get_dataset_info("etth1")
# {'num_features': 7, 'frequency': 'h', 'length': 17420, 'category': 'common'}

# 查看所有数据集
print(f"总共 {len(DATASET_INFO)} 个数据集")
```

### 支持的数据集（共 41 个）

#### common 类别（25 个）

| 数据集 | 特征数 | 频率 | 长度 | 说明 |
|--------|--------|------|------|------|
| etth1 | 7 | h | 17420 | ETT 小时数据 |
| etth2 | 7 | h | 17420 | ETT 小时数据 |
| ettm1 | 7 | 15min | 69680 | ETT 15分钟数据 |
| ettm2 | 7 | 15min | 69680 | ETT 15分钟数据 |
| weather | 21 | 10min | 52696 | 天气数据 |
| global_temp | 3850 | h | 17544 | 全球温度 |
| solar | 137 | 10min | 52560 | 太阳能 |
| traffic | 862 | h | 17544 | 交通流量 |
| metr_la | 207 | 5min | 34272 | 洛杉矶交通 |
| pems03 | 358 | 5min | 26208 | PEMS 交通 |
| pems04 | 307 | 5min | 16992 | PEMS 交通 |
| pems07 | 883 | 5min | 28224 | PEMS 交通 |
| pems08 | 170 | 5min | 17856 | PEMS 交通 |
| taxi30 | 1214 | 10min | 20477 | 出租车 |
| electricity | 321 | h | 26304 | 电力消耗 |
| ecl | 321 | h | 26304 | electricity 别名 |
| exchange | 8 | d | 7588 | 汇率 |
| illness | 7 | w | 966 | 流感数据 |
| wiki2000 | 2000 | d | 1704 | 网络流量 |
| honeypot_fyi | 6 | h | 2161 | 安全数据 |
| aos_amplitude | 16 | h | 12000 | 合成数据 |
| aos_frequency | 16 | h | 12000 | 合成数据 |
| aos_mixed | 16 | h | 12000 | 合成数据 |
| aos_phase | 16 | h | 12000 | 合成数据 |
| aos_stationary | 16 | h | 12000 | 合成数据 |

#### tfb 类别（12 个）

| 数据集 | 特征数 | 频率 | 长度 | 说明 |
|--------|--------|------|------|------|
| aq_shunyi | 11 | h | 35064 | 空气质量 |
| aq_wan | 11 | h | 35064 | 空气质量 |
| weather_nor | 21 | 10min | 52696 | 天气数据 |
| weather_ab | 21 | 10min | 52696 | 天气数据 |
| wind | 7 | 10min | 48673 | 风速数据 |
| zaf_noo | 11 | 10min | 19225 | 环境数据 |
| cze_lan | 11 | 10min | 19934 | 环境数据 |
| pems_bay | 325 | 5min | 52116 | 湾区交通 |
| fred_md | 107 | w | 728 | 宏观经济 |
| nasdaq | 5 | d | 1244 | 股票数据 |
| nyse | 5 | d | 1243 | 股票数据 |
| covid19 | 948 | d | 1392 | 疫情数据 |

#### workload 类别（4 个）

| 数据集 | 特征数 | 频率 | 长度 | 说明 |
|--------|--------|------|------|------|
| faas | 226 | 5min | 2305 | FaaS 负载 |
| iaas | 93 | 5min | 3456 | IaaS 负载 |
| paas | 426 | 5min | 7776 | PaaS 负载 |
| rds | 1113 | 5min | 6624 | RDS 负载 |

### 数据目录结构

```
TSDATADIR/
├── common/
│   ├── etth1.npy
│   ├── etth2.npy
│   ├── weather.npy
│   └── ...
├── tfb/
│   ├── aq_shunyi.npy
│   ├── covid19.npy
│   └── ...
└── workload/
    ├── faas.npy
    ├── iaas.npy
    └── ...
```
