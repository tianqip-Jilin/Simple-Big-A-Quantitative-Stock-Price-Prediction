# THU-BDC2026 Stock Ranking Baseline

本项目是一个面向沪深 300 成分股的量化选股基线方案。模型使用过去一段时间的量价与技术因子序列，为同一交易日内的候选股票打分排序，并输出权重合计不超过 1 的 Top 5 股票组合。

## 项目特点

- **任务形式**：学习排序，而不是单只股票二分类或回归。
- **输入数据**：沪深 300 成分股日线行情，默认使用过去 60 个交易日。
- **特征工程**：支持 `39` 组技术指标，以及默认的 `158+39` 组合特征。
- **核心模型**：`StockTransformer`，同时建模单股时间序列特征和同日股票之间的横截面关系。
- **输出结果**：`output/result.csv`，包含最多 5 只股票及其配置权重。

## 目录结构

```text
.
├── code/src/
│   ├── config.py          # 训练和预测配置
│   ├── model.py           # StockTransformer 模型
│   ├── train.py           # 训练入口
│   ├── predict.py         # 预测入口
│   └── utils.py           # 特征工程、数据集构造等工具函数
├── data/
│   ├── stock_data.csv     # 原始行情数据
│   ├── train.csv          # 训练集
│   ├── test.csv           # 本地评分测试集
│   └── split_train_test.py
├── model/                 # 训练产物
├── output/                # 预测输出
├── test/                  # 本地评分和 Docker 验证脚本
├── get_stock_data.py      # 使用 baostock 获取沪深 300 数据
├── train.sh               # 训练脚本
├── test.sh                # 预测脚本
├── Dockerfile
├── docker-compose.yml
└── GUIDE.md               # 更完整的运行说明
```

## 快速开始

建议使用 Python 3.10 到 3.12，并通过 `uv` 安装依赖。

```bash
uv sync
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

训练模型：

```bash
sh train.sh
```

Windows 也可以直接运行：

```powershell
python code/src/train.py
```

生成预测结果：

```bash
sh test.sh
```

Windows 也可以直接运行：

```powershell
python code/src/predict.py
```

预测完成后会生成：

```text
output/result.csv
```

文件字段为：

```text
stock_id,weight
```

## 数据准备

项目默认读取：

```text
data/train.csv
data/test.csv
```

如需重新获取行情数据，可以修改 `get_stock_data.py` 中的日期范围后运行：

```bash
python get_stock_data.py
```

该脚本会从 baostock 获取沪深 300 成分股及日线行情，并保存到：

```text
data/stock_data.csv
data/hs300_stock_list.csv
```

获取原始数据后，可按日期切分训练集和测试集：

```bash
python data/split_train_test.py \
  --input data/stock_data.csv \
  --output-dir data \
  --train-start 2024-01-02 \
  --train-end 2026-03-06 \
  --test-start 2026-03-09 \
  --test-end 2026-03-13
```

## 本地评分

预测完成后，可以用测试集估算本地收益分数：

```bash
python test/score_self.py
```

结果会写入：

```text
temp/tmp.csv
```

## Docker 打包与验证

构建镜像：

```bash
docker buildx build --platform linux/amd64 --build-arg IMAGE_NAME=nvidia/cuda -t bdc2026 .
```

使用 `docker compose` 验证镜像可运行性：

```bash
docker compose up
```

验证成功后，预测结果会出现在：

```text
test/output/result.csv
```

导出提交镜像：

```bash
docker save -o team_name.tar bdc2026:latest
```

## 常见问题

### TA-Lib 安装失败

项目特征工程依赖 `ta-lib`。如果直接安装 Python 包失败，需要先安装系统层面的 TA-Lib C 库。Dockerfile 中已经包含 Linux 下的安装步骤，可作为参考。

### 没有 GPU 能不能运行

可以。代码会按 `CUDA -> MPS -> CPU` 的顺序自动选择设备。没有 GPU 时会回退到 CPU，只是训练速度会更慢。

### 输出结果有什么限制

`result.csv` 最多包含 5 只股票，权重之和需要在 0 到 1 之间。当前默认策略为选择排序分数最高的 5 只股票，并赋予等权重 `0.2`。

## 更多说明

完整的环境配置、数据获取、训练、预测、Docker 和赛事验证流程见 [GUIDE.md](./GUIDE.md)。
