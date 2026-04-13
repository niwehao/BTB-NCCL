# Vidur 与其他组件结合使用：完整命令清单

Vidur 通过 `--random_forrest_execution_time_predictor_config_backend` 参数选择**底层时间预测后端**，与 SimAI 其他组件结合。共有 **4 种 backend**，每种对应不同的组件组合。

---

## 一、四种使用模式总览

| 模式 | backend 值 | 组合的组件 | 需要预编译 | 速度 | 精度 |
|---|---|---|---|---|---|
| **Mode 1：纯 Vidur** | `vidur` | 仅 Vidur（Random Forest 模型） | ❌ | ⚡⚡⚡ | 中 |
| **Mode 2：Vidur + AICB** | `aicb` | Vidur + AICB | ❌ | ⚡⚡ | 高（支持 MoE） |
| **Mode 3：Vidur + ASTRA-sim 解析** | `simai_analytical` | Vidur + ASTRA-sim analytical | ✅ analytical | ⚡⚡ | 中高 |
| **Mode 4：Vidur + ASTRA-sim + ns-3** | `simai_simulation` | Vidur + ASTRA-sim + ns-3 | ✅ ns3 | ⚡ | 最高 |

---

## 二、Mode 1：纯 Vidur（最快入门）

### 何时用
- 快速验证调度策略、PD 解耦比例
- 模型在 Vidur 预训练数据集里（Llama / Qwen / CodeLlama 等）
- 不关心通信细节

### 命令

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 1 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --random_forrest_execution_time_predictor_config_backend vidur
```

---

## 三、Mode 2：Vidur + AICB（推荐用于 MoE/DeepSeek）

### 何时用
- 跑 **DeepSeek-V3-671B / Qwen3-MoE-235B / Qwen3-Next-80B** 这类有 MoE/EP 的模型
- 需要最准确的算子和通信开销
- 这是 vidur-alibabacloud 主推的模式

### 前置条件
需要先有 SimAI + AICB 的 Docker 环境（Vidur 会调用 AICB 生成 trace）。

### 命令（DeepSeek-671B + Fixed Length）

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 8 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 5 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 1024 \
  --fixed_request_length_generator_config_decode_tokens 10 \
  --random_forrest_execution_time_predictor_config_backend aicb
```

### 命令（DeepSeek-671B + Trace Length）

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 8 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 1024 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --random_forrest_execution_time_predictor_config_backend aicb
```

---

## 四、Mode 3：Vidur + ASTRA-sim Analytical（中等精度）

### 何时用
- 需要比 Mode 1 更准的通信开销
- 不想跑 ns-3 那么慢
- 只关心 TP（不支持 PP/EP）

### 前置条件：编译 analytical 后端

```bash
cd SimAI
./scripts/build.sh -c analytical
```

### 命令（Llama-3-8B）

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 1 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --random_forrest_execution_time_predictor_config_backend simai_analytical
```

---

## 五、Mode 4：Vidur + ASTRA-sim + ns-3（最高精度）

### 何时用
- 论文级实验，需要包级精度
- 研究网络拓扑对推理服务的影响
- 评估 RDMA 拥塞、PFC、HPCC 等

### 前置条件：① 编译 ns3 后端

```bash
cd SimAI
./scripts/build.sh -c ns3
```

### 前置条件：② 生成网络拓扑文件

```bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X \
  -g 128 \
  -gt A100 \
  -bw 100Gbps \
  -nvbw 2400Gbps
```

参数说明：

| 参数 | 含义 |
|---|---|
| `-topo Spectrum-X` | 拓扑类型（Spectrum-X / Fat-Tree / DragonFly） |
| `-g 128` | GPU 总数 |
| `-gt A100` | GPU 类型 |
| `-bw 100Gbps` | 网络带宽 |
| `-nvbw 2400Gbps` | NVLink 带宽 |

会生成 `Spectrum-X_128g_8gps_100Gbps_A100` 这个拓扑文件。

### 命令（Llama-3-8B + ns-3）

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 1 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --random_forrest_execution_time_predictor_config_backend simai_simulation \
  --random_forrest_execution_time_predictor_config_simai_dir ../ \
  --random_forrest_execution_time_predictor_config_simai_simulation_topo ../Spectrum-X_128g_8gps_100Gbps_A100 \
  --random_forrest_execution_time_predictor_config_simai_simulation_config ../astra-sim-alibabacloud/inputs/config/SimAI.conf
```

---

## 六、参数完整字典

### A. 模型与并行（所有 Mode 通用）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--replica_config_model_name` | `meta-llama/Llama-2-7b-hf` | 模型名（HuggingFace 路径或别名 `deepseek-671B`、`Qwen3-Moe-235B`、`Qwen3-Next-80B`） |
| `--replica_config_tensor_parallel_size` | `1` | TP 并行度 |
| `--replica_config_num_pipeline_stages` | `1` | PP 级数 |
| `--replica_config_expert_model_parallel_size` | `1` | EP 并行度（仅 MoE 模型） |

### B. 集群与 PD 解耦

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--cluster_config_num_replicas` | `1` | 总 replica 数（即 DP 数） |
| `--replica_config_pd_node_ratio` | `0.5` | P 节点占比，0.5=P:D=1:1，1.0=全 prefill |
| `--global_scheduler_config_type` | `round_robin` | 全局调度器：`round_robin` / `split_wise` / `random` |
| `--replica_scheduler_config_type` | `sarathi` | replica 内调度器：`sarathi`/`vllm`/`split_wise`/`orca`/`faster_transformer` |

### C. 带宽与通信

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--replica_config_pd_p2p_comm_bandwidth` | `800` | P→D 节点之间 KV cache 传输带宽（Gbps） |
| `--replica_config_nvlink_bandwidth` | `1600` | NVLink 带宽（Gbps），用于 TP/EP |
| `--replica_config_rdma_bandwidth` | `800` | 跨节点 RDMA 带宽（Gbps） |
| `--replica_config_pd_p2p_comm_dtype` | `float16` | KV cache 传输数据类型：`float16`/`float32`/`bfloat16` |

### D. 请求生成器

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--synthetic_request_generator_config_num_requests` | `128` | 生成多少个请求 |
| `--interval_generator_config_type` | `poisson` | 到达间隔类型：`poisson`/`gamma`/`static`/`trace` |
| `--poisson_request_interval_generator_config_qps` | `0.5` | Poisson 到达率（QPS） |
| `--length_generator_config_type` | `fixed` | 请求长度类型：`fixed`/`trace`/`uniform`/`zipf` |

### E. 固定长度模式

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--fixed_request_length_generator_config_prefill_tokens` | `2048` | 每个请求的 prefill 长度 |
| `--fixed_request_length_generator_config_decode_tokens` | `512` | 每个请求的 decode 长度 |

### F. Trace 长度模式

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--trace_request_length_generator_config_trace_file` | `data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv` | trace 文件路径 |
| `--trace_request_length_generator_config_max_tokens` | `4096` | 截断阈值 |

### G. Backend 选择（核心）

| 参数 | 默认值 | 可选值 |
|---|---|---|
| `--random_forrest_execution_time_predictor_config_backend` | `vidur` | `vidur` / `aicb` / `simai_analytical` / `simai_simulation` |

### H. SimAI 后端专用（仅 Mode 3、4 需要）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--random_forrest_execution_time_predictor_config_simai_dir` | `'../'` | SimAI 根目录 |
| `--random_forrest_execution_time_predictor_config_simai_simulation_topo` | `'../example/topo'` | 拓扑文件路径（仅 Mode 4） |
| `--random_forrest_execution_time_predictor_config_simai_simulation_config` | `'../astra-sim-alibabacloud/inputs/config/SimAI.conf'` | SimAI 配置文件（仅 Mode 4） |

---

## 七、调度器选项详解

### Global Scheduler（全局调度器）

| 值 | 含义 |
|---|---|
| `round_robin` | 轮询分配请求到 replica |
| `random` | 随机分配 |
| `split_wise` | **PD 解耦专用**：prefill 请求送 P 节点，decode 送 D 节点 |
| `lor` | Least Outstanding Requests，挑负载最低的 replica |

### Replica Scheduler（单 replica 调度器）

| 值 | 含义 |
|---|---|
| `sarathi` | Sarathi-Serve：chunked prefill + 连续 batching |
| `vllm` | vLLM 风格：迭代级调度 + PagedAttention |
| `split_wise` | SplitWise：纯 prefill 或纯 decode 节点 |
| `orca` | Orca：迭代级 batching |
| `faster_transformer` | FasterTransformer 风格：固定 batch |

---

## 八、结合不同 backend 的"决策树"

```
你要模拟什么？
│
├─ MoE 模型 (DeepSeek/Qwen3-Moe/Qwen3-Next)?
│  → 用 backend=aicb （Mode 2）
│
├─ Dense 模型 (Llama/Qwen)?
│  ├─ 关心通信但不要太慢?
│  │  → 用 backend=simai_analytical （Mode 3）
│  ├─ 关心包级网络细节 (拥塞、PFC)?
│  │  → 用 backend=simai_simulation （Mode 4）
│  └─ 只想快速 sweep 配置?
│     → 用 backend=vidur （Mode 1）
```

---

## 九、查看完整参数列表

```bash
cd SimAI/vidur-alibabacloud
python -m vidur.main -h
```

---

## 十、输出位置

所有 Mode 跑完都生成在：

```
./simulator_output/YYYY-MM-DD_HH-MM-SS-XXXXXX/request_metrics.csv
```

关键列：
- `request_e2e_time`：端到端延迟
- `prefill_e2e_time`：TTFT
- `pd_p2p_comm_size` / `pd_p2p_comm_time`：PD 通信开销
- `tbt`：每 token 间隔（TPOT）

---

## 十一、典型实验配方

### 实验 1：评估 PD 解耦收益

```bash
# 不解耦
python -m vidur.main ... \
  --replica_config_pd_node_ratio 1.0 \
  --global_scheduler_config_type round_robin \
  --replica_scheduler_config_type sarathi

# 解耦 (P:D=1:1)
python -m vidur.main ... \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise
```

对比两次的 `request_e2e_time` 和 `tbt`。

### 实验 2：找最佳 TP 切分

```bash
for tp in 1 2 4 8; do
  python -m vidur.main ... \
    --replica_config_tensor_parallel_size $tp \
    --random_forrest_execution_time_predictor_config_backend aicb
done
```

### 实验 3：找最佳 EP（MoE 模型）

```bash
for ep in 1 2 4 8 16; do
  python -m vidur.main ... \
    --replica_config_model_name deepseek-671B \
    --replica_config_expert_model_parallel_size $ep \
    --random_forrest_execution_time_predictor_config_backend aicb
done
```

### 实验 4：评估网络拓扑（需 ns-3）

```bash
# 100Gbps
python3 gen_Topo_Template.py -topo Spectrum-X -g 128 -bw 100Gbps -nvbw 2400Gbps
python -m vidur.main ... --backend simai_simulation --topo Spectrum-X_128g_8gps_100Gbps_A100

# 200Gbps
python3 gen_Topo_Template.py -topo Spectrum-X -g 128 -bw 200Gbps -nvbw 2400Gbps
python -m vidur.main ... --backend simai_simulation --topo Spectrum-X_128g_8gps_200Gbps_A100
```

---

## 十二、一句话速记

> **Vidur 的命令永远是 `python -m vidur.main <一堆 --replica/--scheduler/--generator 参数> --random_forrest_execution_time_predictor_config_backend <vidur|aicb|simai_analytical|simai_simulation>`**——backend 决定了和哪个组件结合，其他参数描述了"模拟什么样的服务场景"。

你可以从 **Mode 1（vidur backend）** 开始快速跑通，再根据需要切换到 Mode 2/3/4 获取更准确的结果。