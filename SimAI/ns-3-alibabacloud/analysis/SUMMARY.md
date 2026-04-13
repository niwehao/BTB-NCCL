# ns-3-alibabacloud/analysis 模块设计与使用总结

## 目录功能概述

`analysis` 是 NS-3 网络仿真的**结果分析工具集**，提供对仿真输出数据的分析和可视化，覆盖流完成时间（FCT）、带宽利用率、队列长度、QP 速率和 CNP 等关键网络指标。

## 目录结构

```
analysis/
├── fct_analysis.py        # 流完成时间（FCT）分析
├── bw_analysis.py         # 带宽利用率分析与可视化
├── qlen_analysis.py       # 交换机队列长度分析
├── qp_rate_analysis.py    # QP（Queue Pair）速率分析
└── qp_cnp_analysis.py     # CNP（Congestion Notification Packet）分析
```

## 文件详细说明

### `fct_analysis.py` — 流完成时间分析

分析 NS-3 输出的 FCT 文件，计算流完成时间的 slowdown 统计：
- 支持按流类型过滤：normal (100)、incast (200)、all
- 按百分位分组统计：中位数、P95、P99
- 可对比多种拥塞控制算法（HPCC-PINT、HPCC 等）

```bash
python fct_analysis.py -p fct_fat -s 5 -t 0 -b 25
```

### `bw_analysis.py` — 带宽分析

从 NS-3 带宽监控文件中提取数据，生成每个节点各端口的带宽利用率时序图：
- 使用 pandas 处理数据，按 node_id 和 port_id 分组
- 计算 tx_bytes 差值转换为 Gbps
- 使用 matplotlib 和 plotly 生成静态/交互式图表
- 输出到 `simulation/monitor_output/figs/`

### `qlen_analysis.py` — 队列长度分析

分析交换机端口的队列占用情况，用于检测网络拥塞热点。

### `qp_rate_analysis.py` — QP 速率分析

分析 RDMA Queue Pair 的发送速率变化，用于评估拥塞控制算法的效果。

### `qp_cnp_analysis.py` — CNP 分析

分析拥塞通知报文的产生频率和分布，帮助理解网络拥塞状态。

## 依赖关系

- **外部依赖**：numpy, pandas, matplotlib, plotly
- **数据来源**：NS-3 仿真输出文件（`simulation/monitor_output/` 和 `simulation/mix/`）

## 使用方式

```bash
# 分析 FCT
python analysis/fct_analysis.py -p fct_fat -t 0

# 分析带宽
python analysis/bw_analysis.py <bw_file_name>

# 分析队列长度
python analysis/qlen_analysis.py <qlen_file_name>
```

## 在项目中的角色

提供 NS-3 网络仿真结果的**后处理分析能力**，帮助用户理解网络性能瓶颈、拥塞控制效果和带宽利用率，为网络配置优化提供数据支持。
