# 第 03 阶段：让 Transformer 从历史轨迹预测未来

适用项目：`/home/user/YANG/00trajdff`，Conda 环境 `trajdff`，A5000。

这一步的交付物是一个**可以训练、保存、恢复、评测和画图的单轨迹 Transformer 基线**。
它只看目标 vehicle 的历史，不使用邻车和地图。之后加入地图与交互编码，再接条件扩散；当前先建立可信的训练流程。
现有 CV 是比较对象：固定 100 个验证场景，ADE 5.406 m、FDE 14.111 m、MR 92.0%。
这些是你之前的结果，Transformer 的结果需要在你的机器实际训练后产生。

## 0. 放置代码：不要改数据文件名

把 ZIP 里的整个 `stage03_history_transformer` 文件夹放进项目根目录。
解压后应能找到 `00trajdff/stage03_history_transformer/train.py`。
原来的 `scripts/01_visualize_cv.py`、`scripts/02_evaluate.py` 和原始数据继续保留。

所有命令在项目根目录执行：

```bash
cd /home/user/YANG/00trajdff
conda activate trajdff
```

需要已有：

| 路径 | 用途 |
|---|---|
| `data/raw/train/<场景ID>/scenario_<场景ID>.parquet` | 训练数据 |
| `data/raw/val/<场景ID>/scenario_<场景ID>.parquet` | 验证数据 |
| `splits/val_cv100.txt` | 上一步固定下来的 100 个验证场景 ID，一行一个 |
| `reports/cv_val100.csv` | 上一步逐场景 CV 结果；存在时会自动核对 |

`val_cv100.txt` 保存的是 `0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca` 这种 ID，
不包含 `scenario_` 前缀、`.parquet` 后缀或路径。场景名保持原样，不需要重新编号。
如果你之前保存的划分文件位置不同，使用 `prepare.py --val-ids 实际路径`。
如果没有保存划分文件，先找回上一步真正评测的 100 个 ID；不要重新随机抽一批来比较。

## 1. 检查 Python 和 GPU

```bash
python -m pip install -r stage03_history_transformer/requirements-data.txt
python stage03_history_transformer/check_environment.py
```

应看到当前解释器属于 `trajdff`，`CUDA available: True`，GPU 名称含 `A5000`，以及 GPU 矩阵乘法输出。
只看到 `nvidia-smi` 正常，不能证明这个 Python 的 PyTorch 能使用 GPU。
如果没有 torch 或 CUDA 为 False，先完成 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 中 Linux / Pip / Python / CUDA 对应的安装；
CUDA 选项需要结合这台机器的驱动检查。不要为运行这个包直接覆盖已正常工作的 GPU 环境，也不要装 CPU wheel 来解决 A5000 的问题。
代码使用标准 PyTorch 2.2+ API；本包没有锁定你尚未核实的 CUDA 版本。

## 2. 制作小缓存：5,000 个训练场景 + 原来的 100 个验证场景

```bash
python stage03_history_transformer/prepare.py --train-count 5000
```

程序在 train 划分内排序后用 seed=42 打乱，选择前 5,000 个符合协议的 focal vehicle 场景。
验证集直接读取旧的 ID 列表，不重新抽样；缺失或不合格就报错，避免悄悄换评测对象。
如果旧 CSV 存在，还会检查局部坐标下重新计算的 CV 与旧结果一致。

输出位置 `data/processed/history_v1/`，包含：

- `train.npz`、`val.npz`：数组缓存，训练时无需每轮重新读取 Parquet。
- `train_ids.txt`、`val_ids.txt`：实际使用的场景列表。
- `train_scan.csv`：训练样本选择与排除记录。
- `metadata.json`：坐标尺度、特征、样本数、CV 结果及缓存校验值。

缓存存在时拒绝覆盖。已经成功准备过就跳过本步；需要新版本时用 `--cache-name history_v2`，后续各命令也加同一个参数。
数据仍在原来的 train / val 划分里，绝不混合后重新切分。

## 3. 先确认一个 batch 能前向和反向传播

```bash
python stage03_history_transformer/check_batch.py --device cuda
```

通常会看到：

```text
hist (4, 50, 6) torch.float32
hist_valid (4, 50) torch.bool
future (4, 60, 2) torch.float32
future_valid (4, 60) torch.bool
cv_m (4, 60, 2) torch.float32
First current normalized position: [0.0, 0.0]
First current heading sin/cos: [0.0, 1.0]
Prediction: (4, 60, 2) normalized positions
Finite loss: True
```

这一步只证明数据与计算能接起来。随机初始化模型的误差不代表训练效果。

## 4. 16 个训练场景过拟合检查

```bash
python stage03_history_transformer/train.py --device cuda --run history_overfit16 --overfit 16 --batch-size 16 --epochs 300
```

`--overfit 16` 只取缓存中前 16 个训练场景，并自动关闭 dropout。
日志里的 **replayFDE** 是在这 16 个训练场景上重新预测得到的误差，诊断时主要看它与训练损失。
`valFDE` 仍来自固定验证集，此阶段不要求它变好。

通过标准：训练损失明显下降，训练场景的 ADE/FDE 能降到很小，并检查预测与标签贴合。
例如 replayFDE 低于约 0.5～1 m 可作为实用的排查参考，不是理论保证或正式成绩。
如果 300 轮仍明显不收敛，先检查坐标、尺度、学习率和数据，不要急着扩大训练。

`checkpoints/history_overfit16/training.csv` 中有 replay_ADE_m / replay_FDE_m。
`best.pt` 按最低训练 replayFDE 保存；这个模型只用于诊断。
它和正式训练使用不同的 run 目录，避免把诊断权重当作基线结果。

画出第一个训练样本，检查模型是否学会贴合标签：

```bash
python stage03_history_transformer/visualize.py --device cuda --split train --checkpoint checkpoints/history_overfit16/best.pt
```

图保存在 `figures/history_overfit16/train/`。默认第一个缓存训练样本属于这 16 个样本。

## 5. 正式小规模训练

过拟合检查通过后，从随机初始化开始：

```bash
python stage03_history_transformer/train.py --device cuda --run history_tf_v1 --batch-size 64 --epochs 50
```

默认模型：128 维隐藏特征、4 个注意力头、3 层 Transformer、dropout=0.1。
优化器为 AdamW，学习率 0.001，梯度范数裁剪为 1；先使用 FP32。
每轮训练后，在同一批 100 个验证场景上评测；正式 `best.pt` 按验证 FDE 选择。

输出 `checkpoints/history_tf_v1/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 模型参数、学习率、种子、数据版本及环境 |
| `training.csv` | 每轮 loss、验证 ADE/FDE/MR 和时间 |
| `best.pt` | 验证 FDE 最低的权重及训练状态 |
| `last.pt` | 最近一轮权重及训练状态，用于恢复 |

同名 run 已存在时会拒绝覆盖；新实验使用新名字，如 `--run history_tf_lr3e4 --lr 0.0003`。
显存不足时先改 `--batch-size 32`，这是新实验，要使用新的 run 名称。
这里不承诺特定运行时长或一定超过 CV，以你的训练日志为准。

中断后恢复，例如继续训练到总计 100 轮：

```bash
python stage03_history_transformer/train.py --device cuda --batch-size 64 --epochs 100 --resume checkpoints/history_tf_v1/last.pt
```

`--epochs 100` 表示总轮数达到 100，不是再加 100。恢复时使用原 batch-size 和原缓存。
恢复会读入模型、优化器和随机数状态；不同 GPU、驱动或 PyTorch 版本仍可能导致数值差异。

## 6. 和 CV 在同一批验证场景上比较

```bash
python stage03_history_transformer/evaluate.py --device cuda --checkpoint checkpoints/history_tf_v1/best.pt
```

输出 `reports/transformer_history_tf_v1.csv` 和同名 `.json`。
CSV 每行一个场景：`scenario_id, ADE, FDE, miss, CV_ADE, CV_FDE, CV_miss`。
JSON 中 `miss` 是比例，例如 0.92 代表 92%。所有位置误差都还原为米。

| 指标 | 计算 |
|---|---|
| ADE | 每场景 60 个未来位置误差取均值，再对场景平均 |
| FDE | 每场景第 60 个未来位置误差，再对场景平均 |
| MR | `FDE > 2.0 m` 的场景比例，等于 2.0 m 不算 miss |

当前 CV 和 Transformer 都只有一条预测，即 K=1，此时 minADE/minFDE 等于 ADE/FDE。
这套 vehicle-only、100 场景协议是项目内部对比，不是官方完整 AV2 排行榜成绩。
这 100 个场景已用于选择 checkpoint，应视为开发验证集；后面要报告更可靠的结论，
需要更大的验证集以及预先固定、未用于调参的留出集合，不能反复调参后称它为独立测试成绩。

## 7. 画出失败案例

先执行上一步评测，然后：

```bash
python stage03_history_transformer/visualize.py --device cuda --checkpoint checkpoints/history_tf_v1/best.pt
```

默认从评测报告选择 Transformer FDE 最大的验证场景，画出历史、真实未来、CV、Transformer 和逐时刻误差。
也可以添加 `--scenario-id 你要看的完整场景ID` 来固定某个验证场景。
图片保存到 `figures/history_tf_v1/<场景ID>.png`。
默认选择最差场景属于事后失败分析，不能把这个过程用于选择模型输入或替换验证样本。

先回答三件事：直行是否稳定？转弯时是否切弯？减速或停车时是否继续向前冲？
没有地图和交互条件时，模型可能根据平均模式预测，甚至比 CV 差；这正是后续条件建模要解决的问题。

## 8. 你现在需要理解的核心

单场景历史 50 帧，每帧 6 个特征：

| 列 | 含义 |
|---|---|
| 0、1 | 局部 x、y，除以 50 m |
| 2、3 | 局部 vx、vy，除以 20 m/s |
| 4、5 | `sin(heading - heading49)`、`cos(heading - heading49)` |

局部原点固定在 timestep=49 的位置；局部 x 朝当前车头、y 朝左。
坐标轴在整个预测过程中固定，不会随未来车头转动。
行向量公式：`local_xy = (map_xy - origin) @ rotation`；速度只旋转，不平移。
旋转角、原点、归一化尺度全部来自历史或预先定义的常数，没有从未来估计。

`hist` 的 shape 是 `[B,50,6]`；模型输出 `[B,60,2]`，直接表示归一化的未来位置，不是位移增量。
`dataset.py` 把未来标签也除以 50；评测时再乘 50 还原成米。
训练损失是每个有效未来点的 `dx²+dy²` 的平均，单位是归一化距离平方，不等于 ADE。
因此不能把日志 `loss=0.01` 直接解释成 `0.01 m`。

模型把每帧特征映射到 128 维，加时间位置编码，编码全部历史，取第 49 帧的输出，经 MLP 一次预测全部 60 帧。
历史中的每一帧都已观测，所以历史编码器允许双向注意力；这不构成使用未来标签。
未来数组只进入损失或评测，不作为模型输入。

Dataset 中 mask 的 True 表示有效；PyTorch padding mask 的 True 表示忽略，所以代码里使用 `~hist_valid`。
本阶段只选 110 帧齐全的 focal vehicle，两个 mask 目前都是 True；
代码尚未支持任意缺失历史、缺失未来、多个参与者或官方 test 隐藏未来标签。

## 9. 文件学习顺序

先运行，不需要逐字手敲。优先读这四个位置：

1. `data.py / encode_dataframe`：怎么从 Parquet 生成历史特征与未来标签。
2. `dataset.py / __getitem__`：怎么给 DataLoader 返回一个样本。
3. `model.py / forward`：`[B,50,6]` 怎样变成 `[B,60,2]`。
4. `train.py` 的训练循环：前向、loss、清零梯度、反向、更新。

命令行、保存文件和记录日志先会用即可。此时还不需要研究扩散采样公式。

可选自检命令：

```bash
python -m unittest discover -s stage03_history_transformer/tests -v
```

## 10. 做完这一阶段后再做什么

保留训练配置、CV/Transformer 对比表、至少几个成功和失败案例。
下一阶段加入地图与邻车条件，在同一编码条件下比较确定性 Transformer 与 Conditional Diffusion，
再统一候选条数 K、计算 minADE/minFDE/MR，并单独测量预处理与推理成本。
当前尚未实现扩散、多模态预测或推理计时；本阶段也不能证明闭环规划安全。
