# SimAI-mixnet 使用手册

## cache

./run.sh conf/topo/os_fattree.json conf/topo/mixnet.json conf/topo/fattree.json conf/workload/mistral-8-7B-train.json
./run.sh conf/topo/mixnet.json conf/topo/os_fattree.json conf/topo/fattree.json conf/workload/deepseek-671b-decode.json

## 文件结构

```
SimAI-mixnet/
├── simai_htsim                        # 模拟器二进制
├── build.sh                           # 编译脚本
├── run.sh                             # 一站式运行脚本 (读 JSON 配置)
├── build_htsim/                       # 编译中间产物
├── conf/                              # JSON 配置文件
│   ├── topo/                          # 拓扑配置
│   │   ├── mixnet.json                # OCS-ECS 混合拓扑
│   │   ├── fattree.json               # 全带宽 fat-tree
│   │   ├── os_fattree.json            # 过量订阅 fat-tree
│   │   ├── agg_os_fattree.json        # 聚合过量订阅 fat-tree
│   │   ├── fc.json                    # Full circuit 全连接
│   │   └── flat.json                  # Flat 扁平拓扑
│   └── workload/                      # workload 配置
│       ├── deepseek-671b-decode.json  # DeepSeek 推理 decode
│       └── deepseek-671b-train.json   # DeepSeek 训练
├── log/                               # 运行日志
│   └── {topo}_{MMDD_HHMMSS}/
│       ├── trace.log                  # 完整运行日志
│       ├── stats.txt                  # 统计摘要
│       ├── fct_output.txt             # TCP flow 完成时间
│       ├── topo.json                  # 本次运行的拓扑配置副本
│       ├── workload.json              # 本次运行的 workload 配置副本
│       └── ncclFlowModel_EndToEnd.csv # 通信层分维度时间分解
├── SimAI/
│   ├── aicb/                          # AICB workload 生成器
│   │   ├── scripts/
│   │   │   ├── inference_workload_with_aiob.sh
│   │   │   └── inference_configs/deepseek_default.json
│   │   └── results/workload/          # 生成的 workload 文件
│   └── astra-sim-alibabacloud/
│       └── astra-sim/
│           ├── network_frontend/htsim/
│           │   ├── AstraSimNetwork.cc # 主入口 (CLI、拓扑工厂、日志)
│           │   └── entry.h            # 多拓扑选路 + SendFlow
│           └── workload/Workload.cc   # workload 解析
└── mixnet-sim/
    └── mixnet-htsim/src/clos/
        └── datacenter/
            ├── topology.h             # 拓扑基类
            ├── mixnet.cpp/.h          # OCS-ECS 混合拓扑
            ├── fat_tree_topology.cpp  # Fat-tree
            ├── os_fattree.cpp         # Oversubscribed fat-tree
            ├── agg_os_fattree.cpp     # Aggregated oversubscribed fat-tree
            ├── fc_topology.cpp        # Full circuit
            └── flat_topology.cpp      # Flat
```

## 使用方法

### 编译

```bash
./build.sh              # 单线程编译
./build.sh -j 8         # 8 线程并行
./run.sh --build-only   # 通过 run.sh 编译
```

### 运行 (推荐: 通过 JSON 配置)

```bash
# 单拓扑运行
./run.sh conf/topo/fattree.json conf/workload/deepseek-671b-decode.json
./run.sh conf/topo/fattree.json conf/workload/deepseek-671b-decode.json --skip-build

# 多拓扑批量运行 (同一个 workload, 指定多个拓扑)
./run.sh conf/topo/fattree.json conf/topo/mixnet.json conf/topo/fc.json conf/workload/deepseek-671b-decode.json

# 全部拓扑测试
./run.sh --test-all                                        # 用第一个 workload
./run.sh --test-all conf/workload/deepseek-671b-train.json # 指定 workload

# 强制重新生成 workload
./run.sh conf/topo/fc.json conf/workload/deepseek-671b-decode.json --gen-workload
```

### 运行 (直接 CLI)

```bash
./simai_htsim -w workload.txt --topo fattree --nodes 64 \
  --tp_degree 8 --pp_degree 2 --ep_degree 4 --dp_degree 1 \
  --gpus_per_server 8 --speed 100000 --iterations 1
```

## JSON 配置格式

配置分为两个文件：**拓扑配置** (`conf/topo/`) 和 **workload 配置** (`conf/workload/`)。

### 拓扑配置 (conf/topo/*.json)

```json
{
  "topology": {
    "type": "fattree",
    "speed": 100000,
    "queuesize": 8
  },
  "simulation": {
    "iterations": 1
  }
}
```

### Workload 配置 (conf/workload/*.json)

```json
{
  "model": {
    "name": "DeepSeek-671B",
    "model_size": "deepseek-671B",
    "world_size": 64,
    "tp_degree": 8,
    "pp_degree": 2,
    "dp_degree": 1,
    "ep_degree": 4,
    "gpus_per_server": 8,
    "phase": "decode",
    "seq_length": 1024,
    "micro_batch": 1,
    "workload": "SimAI/aicb/results/workload/xxx.txt"
  }
}
```

### model 参数

| 参数               | 说明                     | 用途                                  |
| ------------------ | ------------------------ | ------------------------------------- |
| name               | 模型名称                 | 标识                                  |
| world_size         | 总 GPU 数                | 传递给 --nodes                        |
| tp/pp/dp/ep_degree | 并行度                   | 传递给 simai_htsim                    |
| gpus_per_server    | 每台服务器 GPU 数        | 机器数 = world_size / gpus_per_server |
| phase              | decode / prefill / train | 决定使用哪个 workload 生成器          |
| seq_length         | 序列长度                 | workload 生成                         |
| micro_batch        | 微批大小                 | workload 生成                         |
| global_batch       | 全局批大小 (仅 train)    | workload 生成, 默认 32                |
| workload           | workload 文件路径        | 为空则自动生成                        |

### train 参数 (仅 phase=train 的 workload 配置)

训练 workload 需要额外的 `train` 字段指定模型结构，所有参数均有 DeepSeek-671B 默认值：

| 参数                     | 说明                 | 默认值        |
| ------------------------ | -------------------- | ------------- |
| frame                    | 训练框架             | DeepSeek      |
| model_name               | 模型标识             | DeepSeek_671B |
| num_layers               | 层数                 | 61            |
| hidden_size              | 隐藏层维度           | 18432         |
| num_attention_heads      | 注意力头数           | 128           |
| ffn_hidden_size          | FFN 隐藏层维度       | 2048          |
| vocab_size               | 词表大小             | 50257         |
| max_position_embeddings  | 最大位置编码长度     | 4096          |
| num_experts              | MoE 专家数           | 256           |
| n_shared_expert          | 共享专家数           | 1             |
| n_dense_layer            | 非 MoE 密集层数      | 3             |
| q_lora_rank              | Q LoRA 秩            | 1536          |
| kv_lora_rank             | KV LoRA 秩           | 512           |
| qk_nope_dim              | QK NoPE 维度         | 128           |
| qk_rope_dim              | QK RoPE 维度         | 64            |
| v_head_dim               | V Head 维度          | 128           |
| moe_router_topk          | 路由 Top-K           | null (不传)   |
| epoch_num                | 训练轮数             | 1             |
| moe_enable               | 启用 MoE             | true          |
| enable_sequence_parallel | 启用序列并行         | true          |
| swiglu                   | 启用 SwiGLU          | false         |
| use_flash_attn           | 启用 Flash Attention | false         |
| recompute_activations    | 启用激活重计算       | false         |
| moe_grouped_gemm         | 启用分组 GEMM        | false         |

推理 workload (decode/prefill) 不需要 `train` 字段，仅依赖 `model` 中的 `model_size` 预设（如 `deepseek-671B`）。

### topology 参数

| 参数         | 适用拓扑                    | 说明                                                       | 默认值  |
| ------------ | --------------------------- | ---------------------------------------------------------- | ------- |
| type         | 全部                        | mixnet / fattree / os_fattree / agg_os_fattree / fc / flat | fattree |
| speed        | 全部                        | 链路速率 Mbps                                              | 100000  |
| queuesize    | 全部                        | 队列大小 (packets)                                         | 8       |
| alpha        | mixnet                      | 每台机器 OCS 电路数                                        | 4       |
| reconf_delay | mixnet                      | OCS 重配延迟 (us)                                          | 10      |
| ecs_only     | mixnet                      | 强制 ECS 模式                                              | false   |
| os_ratio     | os_fattree / agg_os_fattree | 过量订阅比                                                 | 2       |

### simulation 参数

| 参数       | 说明         | 默认值 |
| ---------- | ------------ | ------ |
| iterations | 模拟迭代次数 | 1      |
| rto_ms     | TCP 超时重传时间 (ms) | 1      |

## 拓扑说明

| 拓扑                        | --topo         | 队列类型                          | 有效带宽                                 |
| --------------------------- | -------------- | --------------------------------- | ---------------------------------------- |
| OCS-ECS 混合                | mixnet         | OCS: ECN, ECS: LOSSLESS_INPUT_ECN | OCS: alpha×speed, ECS: (8-alpha)×speed |
| Fat-tree                    | fattree        | LOSSLESS_INPUT_ECN                | 8×speed                                 |
| Oversubscribed fat-tree     | os_fattree     | LOSSLESS_INPUT_ECN                | speed (有过量订阅)                       |
| Agg oversubscribed fat-tree | agg_os_fattree | LOSSLESS_INPUT_ECN                | speed (有过量订阅)                       |
| Full circuit                | fc             | ECN                               | speed (点对点)                           |
| Flat                        | flat           | ECN                               | speed (点对点)                           |

## 拓扑架构生成

所有拓扑的机器数由并行参数推导：`num_machines = world_size / gpus_per_server`。

以下以默认 decode 配置为例：64 GPU, 8 GPU/server = **8 台机器**, tp=8, pp=2, ep=4, dp=1。

### Fat-tree (fattree)

**K 值计算**：最小偶数 K 使得 K³/4 ≥ num_machines。8 台机器 → K=4（4³/4=16 ≥ 8）。

**交换机结构**（三层 Clos）：

| 层级          | 数量 | 公式  |
| ------------- | ---- | ----- |
| ToR (接入层)  | 8    | K²/2 |
| Agg (汇聚层)  | 8    | K²/2 |
| Core (核心层) | 4    | K²/4 |

**连接方式**：

- 每个 ToR 连接 K/2=2 台机器（下行）+ K/2=2 个 Agg 交换机（上行）
- 每个 Agg 连接 K/2=2 个 ToR（下行）+ K/2=2 个 Core（上行）
- Pod 数 = K = 4，每 Pod 含 K/2=2 个 ToR + K/2=2 个 Agg

**链路速率**：speed × 8 = 800 Gbps（8 端口聚合带宽）

**路由**：同 ToR 直达；同 Pod 经 Agg ECMP；跨 Pod 经 Core 转发。

```
          [Core 0] [Core 1] [Core 2] [Core 3]
            / \      / \      / \      / \
      ┌────┘   └──┐ ...                  
   [Agg0] [Agg1] [Agg2] [Agg3] ... [Agg6] [Agg7]
    / \    / \    / \    / \          / \    / \
 [ToR0][ToR1][ToR2][ToR3]  ...   [ToR6][ToR7]
   |  |   |  |   |  |   |  |       |  |   |  |
  M0  M1  M2  M3  M4  M5  M6  M7  ...  (8台机器使用前8个位置)
```

### Oversubscribed Fat-tree (os_fattree)

与 Fat-tree 相同的 K=4 三层结构，但引入过量订阅。

**参数**：Nhpr = os_ratio = 2（每 ToR 下挂主机数）

| 层级 | 数量 | 公式            |
| ---- | ---- | --------------- |
| ToR  | 8    | K²/2           |
| Agg  | 8    | K × (K - Nhpr) |
| Core | 4    | Nup/2           |

**过量订阅**：每 ToR 有 Nhpr=2 个下行端口连主机，(K-Nhpr)=2 个上行端口连 Agg。下行总带宽 = 上行总带宽，实际无过量订阅（os_ratio=2 时刚好平衡）。os_ratio 越大，下行端口越多，上行比例越低。

**链路速率**：所有链路 speed = 100 Gbps。

### Aggregated Oversubscribed Fat-tree (agg_os_fattree)

**参数**：Nhpr = K/2 = 2（固定），up_port = os_ratio = 2

| 层级 | 数量 | 公式         |
| ---- | ---- | ------------ |
| ToR  | 8    | K²/2        |
| Agg  | 8    | K × up_port |
| Core | 4    | Nup/2        |

**与 os_fattree 的区别**：Agg 层分配方式不同。agg_os_fattree 每 Pod 分配 Nup/K=2 个 Agg 交换机，更均匀地分配汇聚层资源。

**链路速率**：所有链路 speed = 100 Gbps。

### Full Circuit (fc)

**结构**：完全图，所有机器两两直连。

- 节点数：8 台机器
- 双向链路数：C(8,2) = 28 条
- 每对机器之间一条独立链路

**链路速率**：speed = 100 Gbps。

**路由**：单跳直达，无中间交换机。

```
   M0 ──── M1
  /|╲\    /|╲\
 M2  M3──M4  M5
  ╲|╱/    ╲|╱/
   M6 ──── M7
 (所有节点两两直连, 简化示意)
```

### Flat

与 FC 结构相同（完全图），但支持从文件加载自定义拓扑矩阵。无自定义文件时退化为 FC。

**链路速率**：speed = 100 Gbps。

### OCS-ECS 混合 (mixnet)

**区域划分**（基于并行度）：

```
region_size = (ep_degree × tp_degree) / gpus_per_server
            = (4 × 8) / 8 = 4 台机器
region_num  = num_machines / region_size
            = 8 / 4 = 2 个区域
```

区域 0：机器 0-3，区域 1：机器 4-7

**双层网络**：

| 层       | 类型           | 覆盖范围       | 链路速率                        |
| -------- | -------------- | -------------- | ------------------------------- |
| OCS 光层 | 区域内直连电路 | 同区域机器对   | speed × alpha = 400 Gbps       |
| ECS 电层 | Fat-tree 骨干  | 全局（跨区域） | speed × (8 - alpha) = 400 Gbps |

- **OCS**：每台机器最多 alpha=4 条光电路连接同区域其他机器。连接矩阵 `conn[i][j]` 动态生成，仅区域内连接。
- **ECS**：完整 Fat-tree（K=4），作为所有非 OCS 流量的回退路径。
- 总带宽：OCS + ECS = 400 + 400 = 800 Gbps（与纯 Fat-tree 一致）。

**路由决策**（entry.h）：

- All-to-All 流量 + OCS 连接存在 → 走 OCS 光层
- 其他流量（All-Reduce 等）→ 走 ECS Fat-tree
- 区域正在重配置 → 延迟发送，等待重配完成

**动态重配置**：All-to-All 层完成后，根据流量矩阵贪心重新分配 OCS 电路。

```
区域 0 (OCS 域)          区域 1 (OCS 域)
┌─────────────────┐      ┌─────────────────┐
│ M0 ⇄ M1         │      │ M4 ⇄ M5         │
│ ↕╲ ╱↕           │      │ ↕╲ ╱↕           │
│ M2 ⇄ M3         │      │ M6 ⇄ M7         │
│ (alpha=4 OCS)   │      │ (alpha=4 OCS)   │
└────────┬────────┘      └────────┬────────┘
         │         ECS Fat-tree          │
         └──────── (K=4 骨干) ──────────┘
```

## AICB Workload 生成

run.sh 根据 `phase` 自动选择生成器：

| phase            | 生成脚本                                    | model_size 示例 |
| ---------------- | ------------------------------------------- | --------------- |
| decode / prefill | `scripts/inference_workload_with_aiob.sh` | deepseek-671B   |
| train            | `scripts/megatron_workload_with_aiob.sh`  | deepseek671     |

workload 文件不存在或指定 `--gen-workload` 时自动生成。也可手动：

```bash
cd SimAI/aicb
# 推理 (decode / prefill)
bash scripts/inference_workload_with_aiob.sh \
  --model_size deepseek-671B --phase decode \
  --world_size 64 --tensor_model_parallel_size 8 \
  --pipeline_model_parallel 2 --expert_model_parallel_size 4 \
  --seq_length 1024 --micro_batch 32 \
  --result_dir results/workload/

# 训练
bash scripts/megatron_workload_with_aiob.sh \
  --model_size deepseek671 \
  --world_size 64 --tensor_model_parallel_size 2 \
  --pipeline_model_parallel 2 --expert_model_parallel_size 8 \
  --seq_length 4096 --micro_batch 1 --global_batch 32 \
  --moe_enable
```
