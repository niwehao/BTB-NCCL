# aicb/training 模块设计与使用总结

## 目录功能概述

`training` 目录是 AICB 的**用户教程文档目录**，包含完整的使用指南，涵盖环境配置、物理集群运行、工作负载生成、自定义模型等内容。

## 目录结构

```
training/
└── tutorial.md    # AICB 完整使用教程
```

## 教程内容概要

### 1. 环境配置
- **Docker 方式**：使用 Dockerfile 或 NGC PyTorch 容器（>= 23.08）
- **本地方式**：Python >= 3.8, CUDA >= 11.8, PyTorch >= 2.0.0, NVIDIA APEX
- 仅生成工作负载无需额外依赖

### 2. 物理集群运行

#### 单节点运行
通过 `scripts/megatron_gpt.sh` 启动，需设置环境变量 `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`。

#### 多节点运行
通过 `scripts/run_in_cluster.py` + pssh/pscp 批量部署运行。

### 3. 日志与结果分析
- 每次通信完成后输出：通信类型、通信组、消息大小、耗时、吞吐量
- 迭代时间汇总分析，可检测 jitter
- CSV 文件保存于 `results/comm_logs/`

### 4. 工作负载生成
- 通过 `scripts/megatron_workload_with_aiob.sh` 生成 SimAI 可用的工作负载文件
- 支持 `--aiob_enable` 获取真实 GPU 计算时间
- 生成文件保存于 `results/mocked_workload/`

### 5. 参数说明
教程中详细列出了所有参数分类：
- **训练参数**：world_size, global_batch, micro_batch, epoch_num
- **模型参数**：num_layers, hidden_size, num_attention_heads, seq_length 等
- **并行参数**：tensor_model_parallel_size, pipeline_model_parallel, enable_sequence_parallel
- **MoE 参数**：num_experts, moe_router_topk, expert_model_parallel_size, moe_grouped_gemm
- **DeepSeek MLA 参数**：q_lora_rank, kv_lora_rank, qk_rope_dim, qk_nope_dim, v_head_dim
- **DeepSpeed 参数**：zero_stage, reduce_bucket_size, allgather_bucket_size 等

### 6. 自定义模型
基于 `MockedParam` 和 `MockedModel` 基类创建自定义模型架构。训练过程抽象为 `init → forward → backward → step` 四个阶段，每阶段包含对应的通信工作负载项。

## 使用方式

此文件为纯文档，不包含可执行代码，供开发者参考学习 AICB 的使用方法。
