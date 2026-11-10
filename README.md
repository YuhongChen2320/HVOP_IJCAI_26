# HVOP：Flower17 运行与复现记录

本目录已使用现有环境完整跑通 Flower17：7 个视图依次训练，每个视图 800 轮，共 5600 轮。原始模型、数据加载和训练代码未修改。

## 本次结果

实验日期：2026-11-10。使用 CPU，单次固定随机种子运行，训练循环耗时 **1 小时 4 分 48 秒**，进程正常结束（退出码 0）。

| 最终评估指标 | 结果 |
| --- | ---: |
| Accuracy | **67.81%** |
| Macro-F1 | **67.62%** |
| Micro-F1 | **67.81%** |
| Macro-Precision | 68.71% |
| Macro-Recall | 67.81% |

上述指标取自全部训练完成后的最终评估，保留原日志的两位小数。它们不是训练过程中的最高分，也不是多次运行的均值。目录内没有提供论文目标指标，因此本记录仅说明本地代码的实际运行结果，尚未核对是否达到论文效果。

- [完整训练日志](results/flower17_full.log)
- [结构化结果汇总](results/flower17_summary.json)
- [环境完整依赖快照](results/environment.lock.txt)
- [补充运行记录](results/reproduction_notes.txt)

## 环境

本次复用了已有虚拟环境，无需新建：

```text
Python: /data2/chyh/.venvs/GNN/bin/python
Python 版本: 3.10
PyTorch: 2.5.1+cu121
NumPy: 1.26.4
SciPy: 1.13.1
scikit-learn: 1.5.2
Matplotlib: 3.9.4
tqdm: 4.67.1
texttable: 1.7.0
```

本次执行环境中 `torch.cuda.is_available()` 为 `False`，`nvidia-smi` 无法正常工作，因此实际训练使用 CPU。安装了 CUDA 版 PyTorch 不代表当前环境可以访问 GPU。

## 快速运行

在服务器终端执行：

```bash
cd /data2/chyh/ICLR_code_HVOP

# 短跑检查：每个视图 1 轮，完整遍历 7 个视图
./run_flower17.sh --num_epoch 1 > results/flower17_smoke_new.log 2>&1

# 正式训练：每个视图 800 轮，使用新文件名保留本次已有日志
./run_flower17.sh > results/flower17_full_new.log 2>&1
```

另开终端查看进度：

```bash
tail -f /data2/chyh/ICLR_code_HVOP/results/flower17_full_new.log
```

[run_flower17.sh](run_flower17.sh) 会自动切换到项目目录，调用上述 Python 环境，并默认指定 `--device cpu --dataset flower17`。脚本默认使用 4 个 OpenMP/MKL 线程，OpenBLAS 使用 1 个线程，并将 Matplotlib 缓存设为可写的临时目录。

可在命令末尾追加参数，例如：

```bash
./run_flower17.sh --num_epoch 100 --seed 42 > results/flower17_100ep.log 2>&1
```

修改轮数或其他超参数后的结果，应与本次默认参数结果分别记录。

当前执行工具未能保留脱离会话的 `nohup` 进程，本次完整实验通过保持执行会话运行完成。使用 IDE 中的代理执行任务时，应避免中途停止执行会话。原代码没有实现模型检查点保存或断点续训。

## 数据与实验设置

数据和预计算邻接矩阵已包含在目录中，不需要重新下载或构图：

```text
datasets/flower17.mat
adj_matrix/flower17/v0_knn15_adj.npz
...
adj_matrix/flower17/v6_knn15_adj.npz
```

Flower17 包含 1360 个样本、17 个类别、7 个视图；此处每个视图的输入特征矩阵均为 `1360 × 1360`。代码先对视图特征归一化，再读取对应的预计算图，并进行带自环的邻接归一化。

| 参数 | 本次取值 |
| --- | --- |
| 视图顺序 | 0 → 1 → 2 → 3 → 4 → 5 → 6 |
| `num_epoch` | 每个视图 800 轮 |
| `hidden` | 256 |
| `lr` | 0.006 |
| `weight_decay` | 0.0005 |
| `ratio` | 0.1 |
| 标注 / 未标注样本数 | 136 / 1224 |
| `seed` | 42 |
| `sf_seed` | 2021 |
| `knns` | 15 |
| `n_repeated` | 1 |

这是全图上的半监督设置：训练损失中的分类监督使用标注样本，评估使用未标注样本；全部样本的特征和图结构参与计算。划分由代码按类别比例和固定划分种子生成。

完整参数定义见 [args.py](args.py)。部分命令行参数虽然存在于参数表中，但不一定在当前训练实现中使用；本次保持原代码默认行为。

## 文件说明

| 文件 / 目录 | 用途 |
| --- | --- |
| `train.py` | 多视图顺序训练与最终评估入口 |
| `args.py` | 命令行参数与默认值 |
| `Dataloader.py` | 数据读取、划分和邻接归一化 |
| `model.py` | 模型定义 |
| `hlop_module.py` | HLOP 子空间更新与投影 |
| `proj_linear.py` | 带投影的线性层 |
| `utils.py` | 指标计算及辅助函数 |
| `run_flower17.sh` | 本次新增的 CPU 运行脚本 |
| `results/flower17_smoke.log` | 已通过的短跑检查日志 |
| `results/flower17_interrupted.log` | 首次被会话中断的未完成实验 |
| `results/flower17_full.log` | 本次完成全部 5600 轮的正式日志 |
| `results/flower17_summary.json` | 根据正式日志整理的最终指标 |

原代码中的 `save_results` 等参数没有实际写出结果文件；本次通过重定向标准输出和错误输出保留训练日志，另行整理结果汇总。训练进度条的总量设置为 800，但会跨视图累计，因此日志中出现超过 800 的计数属于原代码显示行为。

## 迁移到其他环境

运行脚本写有本机 Python 的绝对路径，迁移时需要修改该路径。可使用 Python 3.10 新建环境，并参考 [依赖快照](results/environment.lock.txt) 配置依赖；其中 `torch==2.5.1+cu121` 需要使用提供对应构建的安装源。依赖快照记录了原环境全部已安装包，并非项目最小依赖清单。

若要在可用 GPU 上运行，可从项目目录直接调用训练入口：

```bash
/path/to/python -u train.py --device 0 --dataset flower17 --num_epoch 800
```

该 GPU 命令未在本次环境中验证；本文报告的结果均来自 CPU 运行。
