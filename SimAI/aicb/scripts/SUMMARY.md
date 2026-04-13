# aicb/scripts 模块设计与使用总结

## 目录功能概述

`scripts` 是 AICB 的**启动脚本和配置目录**，提供各种模型（Megatron/DeepSpeed/MoE/推理）的一键运行脚本和推理模型配置文件。它是用户与 AICB 交互的主要入口层。

## 目录结构

```
scripts/
├── megatron_gpt.sh                   # Megatron GPT 模型运行脚本（支持物理集群）
├── megatron_workload_with_aiob.sh    # Megatron 工作负载生成脚本（带 AIOB 计算模拟）
├── deepspeed_llama.sh                # DeepSpeed LLaMA 模型运行脚本
├── coll_comm_check.sh                # 集合通信性能测试脚本
├── inference_workload_with_aiob.sh   # 推理工作负载生成脚本
├── run_in_cluster.py                 # 集群分布式运行管理脚本
└── inference_configs/                # 推理模型配置文件
    ├── deepseek_default.json         # DeepSeek-671B 默认配置
    ├── qwen3_moe_default.json        # Qwen3-MoE-235B 默认配置
    ├── qwen3_next_default.json       # Qwen3-Next-80B 默认配置
    └── mistral                       # Mixtral-8x7B 配置
```

## 文件详细说明

### `megatron_gpt.sh` — Megatron 集群运行脚本

在物理 GPU 集群上使用 `torchrun` 运行 AICB 基准测试。

**支持的模型预设**：

| 参数 | 模型名 | 层数 | 隐藏维度 | 注意力头 |
|------|--------|------|---------|---------|
| `7` | gpt_7B | 36 | 4096 | 32 |
| `13` | gpt_13B | 40 | 5120 | 40 |
| `22` | gpt_22B | 48 | 6144 | 64 |
| `175` | gpt_175B | 96 | 12288 | 96 |
| `405` | llama_405B | 128 | 16384 | 128 |
| `moe` | Mixtral_8*7B | 32 | 4096 | 32 |
| `deepseek671` | DeepSeek_671B | 61 | 18432 | 128 |
| `deepseek236` | DeepSeek_236B | 60 | 12288 | 128 |
| `deepseek16` | DeepSeek_16B | 27 | 10944 | 16 |

**关键参数**：`--aiob_enable`（开启计算模拟）、`--moe_enable`（MoE 模式）、`--sp`（序列并行）、`--workload_only`（仅生成工作负载）

### `megatron_workload_with_aiob.sh` — 工作负载生成脚本

调用 `workload_generator.SimAI_training_workload_generator` 离线生成训练工作负载文件，无需 GPU 集群。支持与 `megatron_gpt.sh` 相同的模型预设，额外支持 `--gpu_type` 和 `--recompute_activations` 参数。

### `deepspeed_llama.sh` — DeepSpeed ZeRO 运行脚本

使用 DeepSpeed ZeRO（Stage 1/2/3）框架运行 LLaMA 基准测试。

**特有参数**：
- `--zero_stage`：ZeRO 阶段（1/2/3）
- `--reduce_bucket_size`、`--allgather_bucket_size`：通信桶大小
- `--param_persistence_threshold`：参数持久化阈值
- `--contiguous_gradients`：使用 reduce 代替 all_reduce

### `coll_comm_check.sh` — 集合通信测试脚本

测试单个集合通信操作的性能（如 all_reduce），支持指定消息大小范围（`--begin_size` 到 `--end_size`）和迭代次数，支持多 all_reduce 并发测试。

### `inference_workload_with_aiob.sh` — 推理工作负载生成

调用 `workload_generator.SimAI_inference_workload_generator` 生成推理场景工作负载。

**支持模型**：DeepSeek-671B、Qwen3-MoE-235B、Qwen3-Next-80B、Mixtral-8x7B

**关键参数**：`--phase`（prefill/decode）、`--aiob_enable`（AIOB 计算模拟）、`--ep_size`（专家并行度）

### `run_in_cluster.py` — 集群部署管理

通过 Docker 容器在多节点 GPU 集群上运行 AICB。自动检测本机 IP 确定节点 rank，启动 Docker 容器并挂载代码目录。

**使用前需修改**：`IMAGE_NAME`（Docker 镜像名）、`IPLIST`（IP 列表路径）、`AICB_DIR`（代码路径）

### `inference_configs/` — 推理模型配置

JSON 格式的模型架构配置文件，包含层数、隐藏维度、注意力头数、MoE 专家数等参数。DeepSeek 配置包含 MLA（Multi-head Latent Attention）特有参数如 `d_kv_c`、`d_q_c`、`d_r` 等。

## 使用方式

```bash
# 在集群上运行 Megatron GPT-13B
bash scripts/megatron_gpt.sh -m 13 --aiob_enable --nnodes 4

# 仅生成 DeepSeek-671B 训练工作负载
bash scripts/megatron_workload_with_aiob.sh -m deepseek671

# 生成推理工作负载
bash scripts/inference_workload_with_aiob.sh -m deepseek-671B -p decode -s 1024

# 运行集合通信测试
bash scripts/coll_comm_check.sh --test_comm all_reduce --begin_size 4096 --end_size 1073741824

# DeepSpeed ZeRO-3 LLaMA
bash scripts/deepspeed_llama.sh -m 13 --zero_stage 3
```

## 依赖关系

- 所有脚本最终调用 `aicb.py`（集群运行）或 `workload_generator` 模块（工作负载生成）
- 集群运行脚本依赖 `torchrun` 和分布式环境变量（`WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT`）
