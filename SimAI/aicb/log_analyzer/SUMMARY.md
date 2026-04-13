# aicb/log_analyzer 模块设计与使用总结

## 目录功能概述

`log_analyzer` 是 AICB 的**日志分析与性能统计模块**，负责解析通信基准测试日志、计算带宽指标、生成统计报告和可视化图表。支持 DeepSpeed 格式日志和 CSV 结果文件的分析。

## 目录结构

```
log_analyzer/
├── __init__.py              # 空文件，标记为 Python 包
├── log.py                   # 核心数据模型（LogItem, Log, Workload）
├── ds_comm_log_analyzer.py  # DeepSpeed 通信日志解析器
├── analyze_res_csv.py       # CSV 结果文件分析器
├── plot.py                  # 可视化绘图工具
└── utils.py                 # 工具函数（单位转换、带宽计算）
```

## 文件详细说明

### `log.py` — 核心数据模型

#### `LogItem` 数据类
通信日志的最小单元，字段包括：

| 字段 | 类型 | 说明 |
|------|------|------|
| `comm_type` | CommType | 通信类型（all_reduce, all_gather 等） |
| `comm_group` | CommGroup | 通信组（tp_group, dp_group 等） |
| `comm_group_size` | int | 通信组大小 |
| `msg_size` | float | 消息大小（字节） |
| `stage` | str | 训练阶段标记 |
| `elapsed_time` | float | 耗时（ms），设置时自动计算 algbw/busbw |
| `algbw` / `busbw` | float | 算法带宽 / 总线带宽（GB/s） |

关键方法：
- `view_as_ds_log()` — 格式化为 DeepSpeed 风格日志字符串
- `view_as_csv_line()` — 格式化为 CSV 行

#### `Log` 类
管理完整的通信日志记录：
- `add_comm_log()` — 添加日志项，按 epoch 分组
- `analyze()` — 按阶段（init/train）聚合统计，输出详细通信信息表
- `dump()` — 导出为 CSV 文件（保存到 `results/comm_logs/`）
- `analyze_time()` — 计算迭代时间的 max/min/avg/P90/std

#### `Workload` 类
管理工作负载数据：
- `append()` — 添加工作负载项（支持 LogItem 或 Dict）
- `dump()` — 导出为 CSV 文件（保存到 `results/mocked_workload/`）
- `load()` — 从 pickle 文件加载

### `ds_comm_log_analyzer.py` — DeepSpeed 日志解析器

- `parse_ds_log_item(line)` — 解析单行 DeepSpeed 日志，提取 comm_op、msg_size、time 等字段
- `parse_ds_comm_log(filename)` — 解析完整日志文件，识别 epoch 边界（ZeRO 初始化/microstep），返回 `Log` 对象
- `string2comm_type(s)` — 字符串到 CommType 枚举的映射

### `analyze_res_csv.py` — CSV 结果分析

- `analyze_csv(file_path)` — 读取 CSV 结果，按 (comm_type, comm_group, msg_size) 分组，计算 busbw 的 mean/max/min/std（排除最小的2个异常值）

### `plot.py` — 可视化

- `log_boxplot()` — 按通信类型和消息大小绘制耗时箱线图
- `log_time_plotter()` — 绘制 epoch 时间折线图

### `utils.py` — 工具函数

| 函数 | 功能 |
|------|------|
| `convert_size_to_msg(size_bytes)` | 字节数转可读字符串（如 1048576 → "1.0 MB"） |
| `convert_msg_to_size(msg)` | 可读字符串转字节数 |
| `calc_bw_log(comm_type, size, duration, group_size)` | 根据通信类型计算算法带宽和总线带宽 |

带宽计算规则：
- **all_gather / reduce_scatter**：busbw = throughput × (n-1)/n
- **all_reduce**：busbw = throughput × 2(n-1)/n
- **broadcast / reduce / send / recv**：busbw = throughput

## 依赖关系

- **内部依赖**：`utils.utils`（CommType, CommGroup 枚举）、`utils.benchmark_logger`
- **外部依赖**：numpy, matplotlib, pandas
- **被调用方**：`aicb.py` 主入口、`workload_applyer.py`

## 使用方式

```python
# 解析 DeepSpeed 日志
from log_analyzer.ds_comm_log_analyzer import parse_ds_comm_log
comm_log = parse_ds_comm_log("comm_log.txt")
comm_log.analyze()  # 输出统计表

# 分析 CSV 结果
from log_analyzer.analyze_res_csv import analyze_csv
grouped = analyze_csv("results.csv")

# 命令行使用
python -m log_analyzer.analyze_res_csv <path_to_csv>
```
