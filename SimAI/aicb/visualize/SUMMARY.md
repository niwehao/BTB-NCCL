# aicb/visualize 模块设计与使用总结

## 目录功能概述

`visualize` 是 AICB 的**可视化报告生成模块**，将通信基准测试结果或工作负载 CSV 数据转换为交互式 HTML 报告，包含饼图、散点图、CDF 图和时间线图等多种图表。

## 目录结构

```
visualize/
├── __init__.py       # 空文件
├── generate.py       # 可视化报告生成主逻辑（~545行）
└── example.html      # HTML 模板文件（Jinja2 模板）
```

## 文件详细说明

### `generate.py` — 可视化生成引擎

#### 数据读取与处理

| 函数 | 功能 |
|------|------|
| `custom_csv_reader()` | 自定义 CSV 读取器，处理元组格式的 msg_size |
| `read_csv_and_structure_data()` | 读取 CSV 文件并转换为 `LogItem` 对象列表 |
| `split_data_by_epoch()` | 按 epoch 分割数据，可选过滤计算操作 |
| `count_by_epoch()` | 按 epoch 统计各通信类型的次数 |
| `extract_data_from_log_items()` | 提取训练阶段的通信数据（类型、大小、组、busbw、次数） |
| `extract_iteration()` | 从 epoch 数据中提取单次迭代（以 broadcast 为分隔） |

#### 图表生成函数

| 函数 | 图表类型 | 说明 |
|------|---------|------|
| `create_pie_chart_for_epoch()` | 饼图 | 展示各通信类型的次数占比 |
| `create_scatter_chart("commtype")` | 散点图 | X 轴为通信类型，Y 轴为 log(msg_size)，颜色按通信组 |
| `create_scatter_chart("group")` | 散点图 | X 轴为通信组，Y 轴为 log(msg_size)，颜色按通信类型 |
| `calculate_cdf_by_commtype()` + `create_cdf_chart_by_commtype()` | CDF 线图 | 按通信类型绘制消息大小的累积分布函数 |
| `create_timeline_chart()` | 时间线图 | 展示计算与通信交替执行的时间线，附计算/通信时间占比饼图 |
| `create_ratio_pie()` | 饼图 | 整体计算 vs 通信时间占比 |

#### 主入口函数

`visualize_output(filepath, only_workload)` — 完整流程：
1. 读取 CSV 数据
2. 生成所有图表
3. 使用 Jinja2 渲染 `example.html` 模板
4. 输出到 `results/visual_output/` 目录

#### 颜色方案

```python
colors_group = {
    "tp_group": "#376AB3",    # 蓝色
    "dp_group": "#87C0CA",    # 青色
    "ep_group": "#E8EDB9",    # 浅黄
    "pp_group": "#8cc540",    # 绿色
    "all_reduce": "#376AB3",
    "all_to_all": "#009f5d",
    ...
}
```

## 依赖关系

- **外部依赖**：pyecharts（图表库）、jinja2（模板引擎）、numpy
- **内部依赖**：`log_analyzer.log`（LogItem, CommType, CommGroup）
- **被调用方**：`aicb.py` 主入口（当 `--enable_visual` 时）和 `workload_applyer.py`

## 使用方式

```bash
# 可视化已有通信日志
python visualize/generate.py results/comm_logs/xxx_log.csv

# 可视化工作负载（无实际运行时间）
python visualize/generate.py results/mocked_workload/xxx_workload.csv only_workload

# 通过 AICB 主流程自动生成
./aicb.py --enable_visual ...
```

输出文件保存于 `results/visual_output/<filename>.html`，可在浏览器中打开交互查看。

## 生成的报告内容

1. **通信类型分布饼图** — 各通信操作（all_reduce/all_gather 等）的占比
2. **按通信类型的散点图** — 展示不同通信类型下的消息大小分布
3. **按通信组的散点图** — 展示不同通信组下的消息大小分布
4. **消息大小 CDF 图** — 按通信类型分别绘制消息大小的累积分布
5. **计算/通信时间线** — 每次迭代的计算与通信交替执行模式（仅实际运行时可用）
6. **计算 vs 通信时间比例** — 整体和每次迭代的计算/通信时间占比
