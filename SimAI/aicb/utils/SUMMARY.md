# aicb/utils 模块设计与使用总结

## 目录功能概述

`utils` 是 AICB 的**核心工具模块**，提供通信类型/通信组枚举定义、命令行参数解析、计时器、日志记录、AIOB 计算时间解析和 DeepGEMM 测试工具等基础功能。几乎被所有其他模块依赖。

## 目录结构

```
utils/
├── utils.py              # 核心工具：枚举定义、参数解析、计算结果处理（~920行）
├── benchmark_logger.py   # 基准测试日志记录器（~107行）
├── timer.py              # CUDA/CPU 计时器（~60行）
└── deepgemm_utils.py     # DeepGEMM FP8 分组 GEMM 测试工具（~236行）
```

## 文件详细说明

### `utils.py` — 核心工具集

#### 枚举定义

| 枚举类 | 说明 | 值 |
|--------|------|-----|
| `CommType` | 通信操作类型 | all_reduce, all_gather, reduce_scatter, broadcast, isend, irecv, barrier, all_to_all, computation, epoch_end 等 |
| `CommGroup` | 通信组类型 | dp_group, tp_group, pp_group, ep_group, ep_dp_group, ep_tp_group, embedding_group, all |
| `ReduceOp` | 归约操作类型 | SUM, PRODUCT, MIN, MAX, AVG 等 |
| `Strategy` | MoE token 分配策略 | RoundRobin, UniformRandom |

#### 并行分组算法

- `generate_masked_orthogonal_rank_groups()` — 根据并行度和掩码生成正交并行组（支持 TP/DP/PP/EP/CP 任意组合）
- `RankGenerator` 类 — 管理多维并行分组，通过 `get_ranks(token)` 获取任意并行组的 rank 列表

#### 参数解析系统

`get_params()` → `get_args()` 单例模式，包含以下参数组：

| 参数组函数 | 覆盖内容 |
|-----------|---------|
| `get_model_params()` | hidden_size, num_layers, seq_length, vocab_size, flash_attn, swiglu |
| `get_ds_params()` | ZeRO stage, bucket sizes, persistence thresholds |
| `get_megatron_params()` | sequence_parallel, distributed_optimizer |
| `get_moe_params()` | num_experts, moe_router_topk, expert_model_parallel_size |
| `get_deepseek_params()` | MLA 相关参数（q_lora_rank, kv_lora_rank, qk_rope_dim 等） |
| `get_collective_test_params()` | begin_size, end_size, test_comm, iter_num |
| `get_aiob_params()` | aiob_enable, comp_filepath, recompute_activations |
| `get_simAI_workload_params()` | overlap_version |

#### AIOB 计算时间处理

- `get_comp_out(args)` — 在 GPU 上运行 MegatronModel 获取真实计算时间
- `extract_averages(file_path)` — 解析 AIOB 输出文件，提取各算子平均计算时间（attention_forward/backward, mlp_forward/backward 等）
- `extract_inference_averages()` — 推理场景的计算时间提取
- `Comp_with_aiob()` — 将 AIOB 计算时间填充到工作负载中
- `process_all_keys()` — 处理 JSON 格式的计时数据，计算 avg/min/max

#### 其他工具

- `get_padded_vocab_size()` — 将 vocab_size 对齐到 TP 友好的大小
- `cuda_timing_decorator` — CUDA Event 计时装饰器
- `WorkloadWriter` — 工作负载 CSV/PKL 读写
- `get_ep_expected_m_per_group()` — 计算 MoE EP 场景下每组期望 token 数

### `benchmark_logger.py` — 基准测试日志

#### `BenchLogger` 类
- `log_timing(name)` — 装饰器，自动计时通信操作并记录日志
- `end_epoch()` — 标记 epoch 结束，记录迭代时间
- `dump_log()` — 导出日志到 CSV
- `analyze_comm_log()` / `analyze_comm_time()` — 分析通信性能

全局实例 `bench_logger` 供全项目使用。

#### `LoggerFactory` 类
- `create_logger()` — 创建标准格式的 Python Logger

### `timer.py` — 计时器

#### `CudaEventTimer` 类
使用 `torch.cuda.Event` 精确测量 GPU 操作耗时（毫秒）。

#### `Timer` 类
- `use_host_timer=False`（默认）：使用 CUDA Event 计时，精确测量 GPU 操作
- `use_host_timer=True`：使用 CPU `time.time()` 计时，适用于 epoch 级别

### `deepgemm_utils.py` — DeepGEMM 测试工具

提供 FP8 分组 GEMM 的测试和基准测试函数：

| 函数 | 功能 |
|------|------|
| `generate_normal()` | 生成标准 GEMM 测试数据（FP8/BF16） |
| `generate_m_grouped_masked()` | 生成 M 轴掩码分组 GEMM 测试数据 |
| `generate_m_grouped_contiguous()` | 生成 M 轴连续分组 GEMM 测试数据 |
| `test_func_masked()` / `test_func_contiguous()` | 执行分组 GEMM 正确性测试 |
| `bench_masked()` / `bench_contiguous()` | 使用 Kineto 进行性能基准测试 |

## 依赖关系

- **外部依赖**：torch, pandas, numpy, argparse, deep_gemm
- **内部被依赖**：被 `aicb.py`、`workload_generator`、`workload_applyer`、`log_analyzer` 等几乎所有模块引用
- `CommType` 和 `CommGroup` 是整个项目的核心数据类型

## 使用方式

```python
from utils.utils import get_args, CommType, CommGroup, RankGenerator

# 获取解析后的参数
args = get_args()

# 使用通信类型枚举
comm_type = CommType.all_reduce

# 生成并行分组
rg = RankGenerator(tp=8, ep=1, dp=4, pp=2, cp=1, order="tp-dp-pp")
tp_groups = rg.get_ranks("tp")

# 使用日志记录器
from utils.benchmark_logger import bench_logger
bench_logger.analyze_comm_log()
```
