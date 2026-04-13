# vidur-alibabacloud/vidur 模块设计与使用总结

## 目录功能概述

`vidur` 是一个**LLM 推理服务仿真器**，基于离散事件仿真（DES）架构，模拟 LLM 推理请求的到达、调度、批处理和执行过程，支持多种调度策略（vLLM、Orca、Sarathi、Splitwise 等）和执行时间预测方法。阿里巴巴在此基础上扩展了 SimAI 网络仿真集成、DAG 请求建模、Prefill-Decode 分离（PD 分离）等能力。

## 目录结构

```
vidur/
├── main.py                     # 主入口
├── simulator.py                # 离散事件仿真器核心
├── logger.py                   # 日志初始化
├── config/                     # 配置系统
├── config_optimizer/           # 配置优化器（Pareto 分析）
├── entities/                   # 仿真实体定义
├── events/                     # 事件系统
├── execution_time_predictor/   # 执行时间预测器
├── request_generator/          # 请求生成器
├── scheduler/                  # 多级调度器
├── metrics/                    # 指标统计与输出
├── profiling/                  # GPU 性能分析数据
├── types/                      # 枚举类型定义
└── utils/                      # 工具函数
```

## 核心架构设计

### `simulator.py` — 离散事件仿真器

基于**最小堆**的事件驱动仿真：

- 事件按 `(时间, ID, 事件类型)` 优先级排序
- 事件处理可产生新事件，形成事件链
- 支持时间限制终止（`time_limit`）
- 输出 JSON 事件 trace 和 Chrome trace（可用于 `chrome://tracing` 可视化）

仿真初始化流程：

1. 创建 `Cluster`（集群）和多个 `Replica`（模型副本）
2. 通过 `RequestGenerator` 生成请求
3. 将请求转化为 `RequestArrivalEvent` 加入事件队列
4. 创建 `GlobalScheduler` 管理调度

### `entities/` — 仿真实体

| 实体              | 说明                                                                           |
| ----------------- | ------------------------------------------------------------------------------ |
| `Cluster`       | 集群，包含多个 Replica                                                         |
| `Replica`       | 模型副本（一个 DP 单位），包含 TP/PP 配置                                      |
| `Request`       | 推理请求，含 prefill/decode token 数，支持**DAG 属性**（`nx.DiGraph`） |
| `Batch`         | 批次，聚合多个 Request，计算 token 总数                                        |
| `BatchStage`    | 批次在 PP stage 上的执行阶段                                                   |
| `Task`          | 计算节点（COMPUTE/PROMPT/TOKEN 类型）                                          |
| `Flow`          | 通信节点（KVCache 传输等），在 Link 上执行                                     |
| `Link`          | 网络链路（PCIe/Ethernet/IB/NVLink/RDMA/Dummy）                                 |
| `ExecutionTime` | 执行时间封装                                                                   |
| `Processor`     | CPU/GPU 处理器                                                                 |

**Request 扩展**：

- 支持 `RequestType`：MIXED、PREFILL、DECODE（PD 分离）
- 基于 `networkx.DiGraph` 的 DAG 请求建模
- 跟踪 prefill 完成时间、调度延迟、重启次数

**Replica 扩展**：

- 支持 `ReplicaType`：MIXED、PREFILL、DECODE
- PD 分离参数：`pd_p2p_comm_bandwidth`、`pd_node_ratio`
- NVLink/RDMA 带宽配置

### `events/` — 事件系统

| 事件                          | 说明            |
| ----------------------------- | --------------- |
| `RequestArrivalEvent`       | 请求到达        |
| `GlobalScheduleEvent`       | 全局调度触发    |
| `ReplicaScheduleEvent`      | Replica 级调度  |
| `ReplicaStageScheduleEvent` | PP Stage 级调度 |
| `BatchStageArrivalEvent`    | 批次阶段到达    |
| `BatchStageEndEvent`        | 批次阶段完成    |
| `BatchEndEvent`             | 批次完成        |

事件按 `BaseEvent` 基类统一管理，通过 `handle_event()` 方法处理并返回后续事件列表。

### `scheduler/` — 多级调度器

**三级调度架构**：

1. **GlobalScheduler（全局调度器）**

   - `RoundRobinGlobalScheduler`：轮询分配
   - `LORGlobalScheduler`：最少已排队请求
   - `RandomGlobalScheduler`：随机分配
   - `SplitwiseGlobalScheduler`：PD 分离调度（Prefill 和 Decode 分配到不同 Replica）
2. **ReplicaScheduler（Replica 调度器）**

   - `VLLMReplicaScheduler`：vLLM 调度策略
   - `OrcaReplicaScheduler`：Orca 连续批处理
   - `SarathiReplicaScheduler`：Sarathi 分块 prefill
   - `SplitwiseReplicaScheduler`：Splitwise PD 分离
   - `FasterTransformerReplicaScheduler`：FasterTransformer
   - `LightLLMReplicaScheduler`：LightLLM
3. **ReplicaStageScheduler**：PP Stage 级调度

### `execution_time_predictor/` — 执行时间预测

| 预测器                                     | 说明                                                                  |
| ------------------------------------------ | --------------------------------------------------------------------- |
| `LinearRegressionExecutionTimePredictor` | 线性回归预测                                                          |
| `RandomForrestExecutionTimePredictor`    | 随机森林预测                                                          |
| `SklearnExecutionTimePredictor`          | sklearn 基类                                                          |
| `TPTimePredictor`                        | **SimAI 集成**：生成 SimAI 工作负载文件并调用仿真器预测通信时间 |
| `SimAIWorkload`                          | SimAI 工作负载格式封装（WorkItem 数据类）                             |

**SimAI 集成**（关键扩展）：

- `TPTimePredictor` 生成 SimAI 格式的工作负载文件
- 调用 `SimAI_simulator`（NS-3）或 `SimAI_analytical`（解析模型）获取通信时间
- 缓存机制：基于参数哈希避免重复仿真
- 结合 NCCL CPU 开销估算最终时间

### `request_generator/` — 请求生成器

**请求间隔生成**：

- `PoissonRequestIntervalGenerator`：泊松分布
- `GammaRequestIntervalGenerator`：Gamma 分布
- `StaticRequestIntervalGenerator`：固定间隔
- `TraceRequestIntervalGenerator`：基于真实 trace

**请求长度生成**：

- `UniformRequestLengthGenerator`：均匀分布
- `ZipfRequestLengthGenerator`：Zipf 分布
- `FixedRequestLengthGenerator`：固定长度
- `TraceRequestLengthGenerator`：基于 trace

**请求生成器**：

- `SyntheticRequestGenerator`：合成请求（间隔 + 长度组合）
- `TraceReplayRequestGenerator`：trace 回放

### `config/` — 配置系统

基于 `dataclass` 的层次化配置：

- `SimulationConfig`：顶级仿真配置
- `ClusterConfig`、`ReplicaConfig`：集群/副本配置
- `BaseModelConfig`：模型参数（num_layers, embedding_dim, num_q_heads 等）
- `BaseDeviceSKUConfig`：设备 SKU（GPU 型号、显存）
- `BaseNodeSKUConfig`：节点 SKU（GPU 数量）
- 支持 CLI 参数自动解析（`create_from_cli_args()`）

### `config_optimizer/` — 配置优化器

- `ConfigExplorer`：参数空间探索（使用 Ray 并行）
- `BottleneckAnalyzer`：瓶颈分析
- `generate_pareto_curves.py`：Pareto 曲线生成
- `StatsExtractor`：统计数据提取

### `metrics/` — 指标系统

- `MetricsStore`：统一指标存储
- `CDFSketch`：CDF 分位数统计
- `DataSeries`：时序数据
- `SeriesAverageMeter`：滑动平均

### `profiling/` — GPU 性能分析

按组件收集 GPU 计算时间数据：

- `attention/`：注意力计算 profiling
- `mlp/`：MLP 层 profiling
- `collectives/`：集合通信 profiling
- `cpu_overhead/`：CPU 开销测量
- `common/`：公共 profiling 工具

### `types/` — 类型枚举

定义所有枚举类型：EventType、GlobalSchedulerType、ReplicaSchedulerType、ExecutionTimePredictorType、RequestGeneratorType、DeviceSKUType、NodeSKUType 等。

## 使用方式

```bash
# 基本推理仿真
python -m vidur.main \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --request_generator_config_type synthetic

# 使用 SimAI 网络仿真预测通信时间
python -m vidur.main \
  --execution_time_predictor_config_simai_enable true \
  --execution_time_predictor_config_simai_dir /path/to/simai
```

## 依赖关系

- **外部依赖**：numpy, pandas, scipy, scikit-learn, networkx, matplotlib, ray（配置优化）
- **SimAI 集成**：调用 astra-sim 的 SimAI_simulator/SimAI_analytical 二进制文件
- **数据来源**：Azure Functions trace、profiling 数据

## 在项目中的角色

作为 SimAI 的 **LLM 推理仿真前端**，模拟推理请求的完整生命周期（到达→调度→批处理→执行→完成），通过与 SimAI 网络仿真集成获取精确的通信时间预测。支持 PD 分离、DAG 请求建模和多种调度策略评估，为推理服务配置优化提供数据支持。

## 和其他组件组合

┌────────────────────────────────────────────────────────┐
│ 训练场景:                                              │
│   AICB → ASTRA-sim → ns-3                              │
│   关心: 一个 training step 的端到端时间                 │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ 推理场景:                                              │
│                                                         │
│  ┌──────────────────────────────────────────────┐     │
│  │  Vidur (推理服务层模拟)                       │     │
│  │   • 请求到达 (Poisson / trace)                │     │
│  │   • Global scheduler (SplitWise)              │     │
│  │   • Replica scheduler (Sarathi / vLLM-like)   │     │
│  │   • PD 解耦 (Prefill 节点 ↔ Decode 节点)       │     │
│  │   • KV cache 传输                              │     │
│  │   • 输出: TTFT / TBT / e2e latency             │     │
│  └─────────────────┬────────────────────────────┘     │
│                    │ "这一批 prefill 要多久?"            │
│                    ▼                                    │
│  ┌──────────────────────────────────────────────┐     │
│  │  Execution Time Predictor (插件式)            │     │
│  │   ├─ vidur (RF 模型)                          │     │
│  │   ├─ aicb (AICB 推理 workload)                │     │
│  │   ├─ simai_analytical (ASTRA-sim α-β)         │     │
│  │   └─ simai_simulation (ASTRA-sim + ns-3)      │     │
│  └──────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────┘

使用
