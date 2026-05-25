# 使用指南

本文档说明如何配置环境、准备数据、训练模型、生成预测结果，并完成 Docker 打包和本地验证。

## 1. 环境配置

### 1.1 安装 uv

项目使用 `uv` 管理 Python 依赖。若本机已有 conda 或 Python 环境，可以直接安装：

```bash
pip install uv
```

也可以参考 uv 官方安装方式。

### 1.2 安装依赖

在项目根目录执行：

```bash
uv sync
```

项目要求 Python 版本：

```text
>=3.10,<3.13
```

主要依赖包括：

- `torch`
- `pandas`
- `scikit-learn`
- `ta-lib`
- `baostock`
- `akshare`
- `tensorboardX`

### 1.3 激活虚拟环境

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. 数据准备

### 2.1 获取沪深 300 行情数据

数据获取脚本为：

```text
get_stock_data.py
```

脚本会：

- 登录 baostock；
- 获取沪深 300 成分股列表；
- 下载指定日期区间内的日线行情；
- 将结果保存为 `data/stock_data.csv`；
- 将成分股列表保存为 `data/hs300_stock_list.csv`。

如需调整数据区间，修改 `get_stock_data.py` 中的：

```python
start_date = "2024-01-01"
end_date = "2026-03-15"
```

然后运行：

```bash
python get_stock_data.py
```

### 2.2 数据字段

训练和预测默认使用以下中文字段：

```text
股票代码
日期
开盘
收盘
最高
最低
成交量
成交额
振幅
涨跌额
换手率
涨跌幅
```

当前代码文件中部分中文字符串可能因历史编码问题显示为乱码，但数据列名需要与代码中读取的列名保持一致。若重新生成数据，请确认 `train.py`、`predict.py` 和 `utils.py` 中使用的列名与 CSV 文件一致。

### 2.3 切分训练集和测试集

使用：

```bash
python data/split_train_test.py \
  --input data/stock_data.csv \
  --output-dir data \
  --train-start 2024-01-02 \
  --train-end 2026-03-06 \
  --test-start 2026-03-09 \
  --test-end 2026-03-13
```

生成：

```text
data/train.csv
data/test.csv
```

参数含义：

- `--input`：原始行情 CSV；
- `--output-dir`：输出目录；
- `--train-start` / `--train-end`：训练集日期范围；
- `--test-start` / `--test-end`：测试集日期范围。

## 3. 模型训练

训练入口：

```text
code/src/train.py
```

快捷脚本：

```bash
sh train.sh
```

Windows 可直接运行：

```powershell
python code/src/train.py
```

训练会读取：

```text
data/train.csv
```

默认配置位于：

```text
code/src/config.py
```

核心参数：

- `sequence_length = 60`：每只股票使用过去 60 个交易日；
- `feature_num = "158+39"`：默认特征组；
- `batch_size = 4`；
- `num_epochs = 50`；
- `learning_rate = 1e-5`；
- `output_dir = ./model/60_158+39`。

训练产物包括：

```text
model/60_158+39/best_model.pth
model/60_158+39/scaler.pkl
model/60_158+39/config.json
model/60_158+39/final_score.txt
model/60_158+39/log/
```

## 4. 生成预测结果

预测入口：

```text
code/src/predict.py
```

快捷脚本：

```bash
sh test.sh
```

Windows 可直接运行：

```powershell
python code/src/predict.py
```

预测过程会加载：

```text
model/60_158+39/best_model.pth
model/60_158+39/scaler.pkl
```

输出文件：

```text
output/result.csv
```

默认输出格式：

```csv
stock_id,weight
600000,0.2
000001,0.2
```

赛事格式要求：

- 最多 5 只股票；
- 权重之和在 0 到 1 之间；
- 当前默认策略为 Top 5 等权重。

## 5. 本地评分

预测完成后运行：

```bash
python test/score_self.py
```

脚本会读取：

```text
output/result.csv
data/test.csv
```

并将参考分数写入：

```text
temp/tmp.csv
```

## 6. Docker 打包

### 6.1 构建镜像

```bash
docker buildx build --platform linux/amd64 --build-arg IMAGE_NAME=nvidia/cuda -t bdc2026 .
```

镜像中会安装：

- Python 运行环境；
- TA-Lib C 库；
- uv；
- 项目 Python 依赖；
- 当前项目代码和模型文件。

### 6.2 docker compose 验证

```bash
docker compose up
```

`docker-compose.yml` 会挂载：

```text
./data        -> /app/data
./test/output -> /app/output
./temp        -> /app/temp
```

默认执行：

```bash
/bin/bash /app/data/run.sh
```

`data/run.sh` 当前会先运行初始化脚本，再运行预测脚本。

验证成功后，结果文件位于：

```text
test/output/result.csv
```

### 6.3 导出镜像

```bash
docker save -o team_name.tar bdc2026:latest
```

将生成的 `.tar` 文件作为提交物。

## 7. 模拟赛事方批量测试

将导出的镜像 tar 文件放到：

```text
test/tars/
```

并在下面文件中写入 tar 文件名：

```text
test/tar_files_list.txt
```

Linux:

```bash
python test/test.py
```

Windows:

```powershell
python test/test_windows.py
```

成功后会生成：

```text
test/result.csv
```

示例：

```csv
Team Name,Final Score
team_name,0.018867553640330992
```

## 8. 常见问题

### TA-Lib 安装失败

`ta-lib` 需要系统层面的 TA-Lib C 库。Linux 下可参考 Dockerfile 中的安装方式：

```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make -j1
make install
```

然后再安装 Python 依赖。

### baostock 下载失败

常见原因是网络不稳定或代理配置冲突。可以关闭代理后重试，或稍后重新运行。脚本支持增量更新，中断后重新运行通常不会重复下载完整数据。

### 训练速度很慢

训练会自动按 `CUDA -> MPS -> CPU` 选择设备。CPU 可以运行，但速度明显慢于 GPU。可适当减小特征数量、缩短训练轮数或使用已有模型权重进行预测。

### 预测时报模型文件不存在

请先完成训练，或确认以下文件已经存在：

```text
model/60_158+39/best_model.pth
model/60_158+39/scaler.pkl
```

### 输出文件校验失败

请检查：

- `output/result.csv` 是否存在；
- 是否包含 `stock_id` 和 `weight` 两列；
- 股票数量是否不超过 5；
- 权重和是否在 0 到 1 之间。
