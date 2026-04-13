# aicb/workload_generator 模块设计与使用总结

## 目录功能概述

`workload_generator` 是 AICB 的**核心工作负载生成模块**，通过模拟（Mocked）模型架构和训练/推理流程，自动生成通信和计算工作负载文件，用于物理集群基准测试或 SimAI 网络仿真。

## 目录结构

```
workload_generator/
├── __init__.py
├── workload_generator.py                    # 工作负载生成器基类
├── SimAI_training_workload_generator.py     # SimAI 训练工作负载生成器
├── SimAI_inference_workload_generator.py    # SimAI 推理工作负载生成器
├── Vidur_workload_generator.py              # Vidur 推理仿真工作负载生成器
├── generate_megatron_workload.py            # Megatron 通用工作负载生成
├── generate_deepspeed_stage1_2_workload.py  # DeepSpeed ZeRO-1/2 工作负载生成
├── generate_deepspeed_stage3_workload.py    # DeepSpeed ZeRO-3 工作负载生成
├── generate_collective_test.py              # 集合通信测试工作负载生成
├── generate_ds_trace_replay_workload.py     # DeepSpeed 日志回放工作负载生成
├── analysis_pytorch_trace.py                # PyTorch Trace 分析工具
└── mocked_model/                            # 模拟模型架构
    ├── MockedModel.py                       # 基类定义
    ├── training/                            # 训练场景模拟模型
    │   ├── MockedMegatron.py                # Megatron 框架模型
    │   ├── MockedDeepspeed.py               # DeepSpeed 框架模型
    │   ├── MockedDeepSeek.py                # DeepSeek MLA+MoE 模型
    │   ├── AiobMegatron.py                  # Megatron AIOB 计算测量模型
    │   └── AiobDeepSeek.py                  # DeepSeek AIOB 计算测量模型
    └── inference/                           # 推理场景模拟模型
        ├── MockedDeepSeek.py                # DeepSeek 推理模型
        ├── MockedQwen3Moe.py                # Qwen3-MoE 推理模型
        ├── MockedQwen3Next.py               # Qwen3-Next 推理模型
        ├── AiobDeepSeek.py                  # DeepSeek 推理 AIOB 测量
        ├── AiobQwen3Moe.py                  # Qwen3-MoE 推理 AIOB 测量
        └── AiobQwen3Next.py                 # Qwen3-Next 推理 AIOB 测量
```

## 核心架构设计

### 基类层

#### `MockedParam` — 模拟参数
- 模拟 `torch.nn.Parameter`，记录 shape 和 element size
- `msg_size()` 返回参数传输的字节数

#### `MockedModel` — 模拟模型
- 模拟 `torch.nn.Module`，支持 `parameters()` 和 hook 注册
- `Linear` 子类模拟线性层

#### `MockedParamsBase` — 模拟参数配置基类
- 支持从 JSON 配置文件或命令行参数加载模型配置
- 用于推理场景的参数管理

#### `WorkloadGenerator` — 工作负载生成器基类
- 模拟训练流程：`init → [forward → backward] × microbatches → step` × epochs
- 支持 pipeline 并行的 `with_pipeline_forward_backward()`
- 子类重写 `forward()`, `backward()`, `step()` 填充具体通信操作

### SimAI 工作负载格式

#### `Work_Item` 数据类
每个工作项包含：
- `name`：操作名称
- `forward_compute_time`：前向计算时间（微秒）
- `forward_comm` / `forward_comm_size`：前向通信类型和大小
- `backward_compute_time` / `backward_comm` / `backward_comm_size`：反向传播
- `dp_compute_time` / `dp_comm` / `dp_comm_size`：数据并行通信
- `process_time`：处理时间

### 训练工作负载生成

#### `SimAI_training_workload_generator.py`
主入口，根据框架类型（Megatron/DeepSpeed/DeepSeek）选择对应的模拟模型，生成 SimAI 格式的工作负载文件。支持 AIOB 计算时间填充。

#### `generate_megatron_workload.py`
继承 `WorkloadGenerator`，实现 Megatron 框架的通信模式：
- Forward: TP all-gather/reduce-scatter（SP 启用时）、MoE all-to-all
- Backward: 与 forward 对应的反向通信
- Step: DP all-reduce / reduce-scatter + all-gather（分布式优化器）

#### `generate_deepspeed_stage1_2_workload.py` / `stage3`
实现 DeepSpeed ZeRO-1/2/3 的通信模式（bucket-based reduce/all-gather）

### 推理工作负载生成

#### `SimAI_inference_workload_generator.py`
生成推理场景（prefill/decode）的工作负载，支持 DeepSeek-671B、Qwen3-MoE、Qwen3-Next 等模型。

### 模拟模型实现

| 类 | 模拟内容 |
|-----|---------|
| `MockedMegatron` | Megatron GPT/LLaMA 的 Transformer 层、注意力、MLP、MoE |
| `MockedDeepspeed` | DeepSpeed ZeRO 的 LLaMA 模型架构 |
| `MockedDeepSeek` | DeepSeek 的 MLA（Multi-head Latent Attention）+ MoE 架构 |
| `AiobMegatron` | 真实 PyTorch 模型，用于在 GPU 上测量计算时间 |
| `AiobDeepSeek` | DeepSeek 的 GPU 计算时间测量模型 |

## 依赖关系

- **内部依赖**：`utils.utils`（参数解析、枚举）、`log_analyzer.log`（Workload, LogItem）、`core`（grouped_gemm）
- **外部依赖**：torch（AIOB 模式时需要）
- **被调用方**：`aicb.py`、`scripts/` 中的启动脚本

## 使用方式

```bash
# 生成 Megatron 训练工作负载（SimAI 格式）
python -m workload_generator.SimAI_training_workload_generator \
  --frame=Megatron --model_name=GPT-13B --world_size=16 \
  --tensor_model_parallel_size=2 --pipeline_model_parallel=1 \
  --global_batch=16 --micro_batch=1 --num_layers=40 --seq_length=2048 \
  --hidden_size=5120 --num_attention_heads=40

# 生成推理工作负载
python -m workload_generator.SimAI_inference_workload_generator \
  DeepSeek-671B configs/deepseek_default.json --phase decode

# 生成通用 Megatron 工作负载
python -m workload_generator.generate_megatron_workload
```

## 设计说明

采用**模拟模型 + 模板方法**模式：
1. `MockedModel` 构建模型架构（无真实权重），只记录参数 shape
2. `WorkloadGenerator` 定义训练循环模板
3. 具体生成器在 forward/backward/step 中填充通信操作
4. AIOB 模式下使用真实 PyTorch 模型测量 GPU 计算时间
5. 输出标准化的工作负载文件供仿真或回放使用
