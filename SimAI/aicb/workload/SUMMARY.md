# aicb/workload 模块设计与使用总结

## 目录功能概述

`workload` 是 AICB 的**预生成工作负载数据目录**，包含多种模型和框架组合的工作负载文件，分为物理集群可用格式和 SimAI 仿真器可用格式两类。同时包含工作负载规格定义和 AIOB 计算时间输入文件。

## 目录结构

```
workload/
├── Workload_spec_v1.1.csv                 # 工作负载规格定义表
├── simAI/                                  # SimAI 仿真器格式工作负载
│   ├── model_workload/                     # 模型训练工作负载
│   │   ├── G13B-M1-C01_GPT13B_megatron_tp8_pp1_mbs1_A100.txt
│   │   ├── G13B-M1-C02_GPT13B_megatron_tp8_pp1_mbs1_sp_A100.txt
│   │   ├── G175B-M1-C03_GPT175B_megatron_tp8_pp1_mbs1_A100.txt
│   │   ├── L7B-M1-C04_Llama7B_megatron_tp2_pp1_mbs1_A100.txt
│   │   ├── L65B-M1-C05_Llama65B_megatron_tp8_pp1_mbs1_A100.txt
│   │   ├── L7B-D1-C02_Llama7B_deepspeed_zero3_A100.txt
│   │   └── L65B_D1_C08_Llama65B_deepspeed_zero3_A100.txt
│   └── micro_test/                         # 微型集合通信测试
│       ├── all_reduce.txt
│       ├── all_gather.txt
│       ├── all_to_all.txt
│       └── muti_all_reduce.txt
├── physical/                               # 物理集群格式工作负载（CSV）
│   ├── model_workload/                     # 模型训练工作负载
│   │   ├── G13B-M1-C01_GPT13B_megatron_tp8_pp1_mbs1.csv
│   │   ├── G175B-M1-C03_GPT175B_megatron_tp8_pp16_mbs1.csv
│   │   ├── L7B-D1-C01_Llama7B_zero2_mbs1.csv
│   │   └── ... (更多模型配置)
│   └── micro_test/                         # 微型通信测试
└── aiob_inputs/                            # AIOB 计算时间输入文件
```

## 文件格式说明

### `Workload_spec_v1.1.csv` — 工作负载规格表

TSV 格式，定义了所有预设工作负载的参数配置：

| 列 | 说明 |
|-----|------|
| Name | 模型名称（LLaMA_7B, GPT_13B, Mistral_8*7B 等） |
| Parameter_size | 参数规模 |
| Hidden_size, Num_of_layers, Attention_heads | 模型架构参数 |
| World_size, TP, DP, PP, SP | 并行配置 |
| Zero_level | DeepSpeed ZeRO 级别 |
| Expert num, TopK, group_gemm | MoE 相关参数 |
| reduce_bucket_size 等 | DeepSpeed 通信参数 |

### SimAI 格式工作负载（`.txt`）

用于 SimAI（astra-sim + ns-3）网络仿真的工作负载格式。文件头包含模型并行信息，后续每行表示一个训练操作，包含通信类型、消息大小、计算时间等字段。

### 物理集群格式工作负载（`.csv`）

CSV 格式，用于在物理 GPU 集群上通过 `workload_applyer.py` 回放执行。包含 comm_type, comm_group, msg_size, elapsed_time 等字段。

### 命名规则

文件名编码了模型和配置信息：
- `G13B-M1-C01`：GPT-13B, Megatron 框架, 配置编号 01
- `L7B-D1-C02`：LLaMA-7B, DeepSpeed 框架, 配置编号 02
- 后缀 `_sp`：启用序列并行
- 后缀 `_A100`：针对 A100 GPU 的计算时间

## 覆盖的模型

| 模型 | 框架 | 配置 |
|------|------|------|
| GPT-13B | Megatron | tp8/pp1, 含/不含 SP |
| GPT-175B | Megatron | tp8/pp1, tp8/pp16 |
| LLaMA-7B | Megatron / DeepSpeed ZeRO-2/3 | tp2/pp1 |
| LLaMA-13B | DeepSpeed | ZeRO-2/3 |
| LLaMA-30B | DeepSpeed | ZeRO-2/3 |
| LLaMA-65B | Megatron / DeepSpeed ZeRO-3 | tp8/pp1 |
| Mistral 8×7B | Megatron MoE | ep8, topk2 |

## 使用方式

```bash
# SimAI 仿真使用
# 工作负载文件直接作为 astra-sim 的输入

# 物理集群回放
python workload_applyer.py --workload_path workload/physical/model_workload/xxx.csv

# 生成新的工作负载
bash scripts/megatron_workload_with_aiob.sh -m 13
# 生成文件保存到 results/mocked_workload/
```

## 在项目中的角色

作为 AICB 的**预置数据集**，为用户提供即用的基准测试工作负载，无需从头生成。SimAI 格式文件直接用于网络仿真，物理格式文件用于真实集群测试验证。
