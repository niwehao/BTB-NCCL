# SimAI + htsim OCS-ECS Mixnet 使用手册

## 项目路径

```
/Users/apple/Project/nccl/SimAI-mixnet/
├── simai_htsim                      # 编译好的模拟器二进制
├── build_htsim/                     # 编译产出的 .o 文件
├── log/                             # 运行日志输出目录
│   └── EP_x_TP_x_DP_x_PP_x_{OCS|ECS}_GPUx_iterX_MMDD_HHMMSS/
│       ├── trace.log                # 完整运行日志 (含 pass 时间、事件计数)
│       ├── stats.txt                # 统计摘要 (配置、流量分布、字节占比)
│       └── fct_output.txt           # 每条 TCP flow 的完成时间
├── SimAI/
│   ├── aicb/                        # AICB workload 生成器
│   │   ├── workload_generator/
│   │   │   ├── SimAI_inference_workload_generator.py   # 推理(decode/prefill)
│   │   │   └── SimAI_training_workload_generator.py    # 训练
│   │   ├── scripts/inference_configs/
│   │   │   └── deepseek_default.json                   # DeepSeek-671B 模型参数
│   │   └── results/workload/                           # 生成的 workload 输出目录
│   └── astra-sim-alibabacloud/
│       └── astra-sim/
│           ├── network_frontend/htsim/
│           │   ├── AstraSimNetwork.cc   # 网络前端(CLI参数解析、日志目录创建、统计输出)
│           │   └── entry.h              # OCS/ECS 选路逻辑 + 流量统计计数器
│           ├── system/Sys.cc            # 系统层(PP_size修复)
│           └── workload/Workload.cc     # workload解析
└── mixnet-sim/
    └── mixnet-htsim/src/clos/
        ├── datacenter/
        │   ├── mixnet.cpp               # OCS 拓扑 + get_paths() 路由
        │   ├── mixnet.h
        │   └── fat_tree_topology.cpp    # ECS fat-tree 拓扑
        ├── mixnet_topomanager.cpp       # OCS 重配置管理 (贪心分配、immediate reconfig)
        ├── ecnqueue.cpp                 # ECN 队列 (OCS 链路使用)
        ├── queue_lossless_output.cpp    # Lossless+ECN 队列 (ECS fat-tree 使用)
        └── tcp.h / tcp.cpp              # TCP/DCTCP 传输层
```

---

## 一、编译

### 增量编译 (推荐)

只重编译改动的文件，然后重新链接：

```bash
HTSIM_SRC="/Users/apple/Project/nccl/SimAI-mixnet/mixnet-sim/mixnet-htsim/src/clos"
FLATBUF_INC="/opt/homebrew/Cellar/flatbuffers/25.12.19/include"
SIMAI_SRC="/Users/apple/Project/nccl/SimAI-mixnet/SimAI/astra-sim-alibabacloud"
BUILD="/Users/apple/Project/nccl/SimAI-mixnet/build_htsim"

# 编译改动的源文件 (按需替换文件名)
g++ -Wall -std=c++17 -O0 -g \
    -I${HTSIM_SRC} -I${HTSIM_SRC}/datacenter -I${FLATBUF_INC} -I${SIMAI_SRC} \
    -c ${SIMAI_SRC}/astra-sim/network_frontend/htsim/AstraSimNetwork.cc \
    -o ${BUILD}/SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/htsim/AstraSimNetwork.cc.o

# 链接 (排除 main_tcp_mixnet.cpp.o，它是独立测试入口)
g++ -std=c++17 -O0 -g \
    $(find $BUILD -name "*.o" ! -name "main_tcp_mixnet.cpp.o") \
    -o /Users/apple/Project/nccl/SimAI-mixnet/simai_htsim
```

**注意**: entry.h 是头文件，修改后需手动重编译 AstraSimNetwork.cc.o

产出: `./simai_htsim`

---

## 二、AICB 生成 Workload

**环境**: conda myenv (torch 2.0.1，仅用于模型参数计算，不跑 GPU)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
cd /Users/apple/Project/nccl/SimAI-mixnet/SimAI/aicb
```

### 2.1 推理 Decode Workload

```bash
python -m workload_generator.SimAI_inference_workload_generator \
    DeepSeek-671B \
    scripts/inference_configs/deepseek_default.json \
    --world_size=64 \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel=2 \
    --expert_model_parallel_size=4 \
    --micro_batch=32 \
    --seq_length=1
```

- 输出: `results/workload/DeepSeek-671B-world_size64-tp8-pp2-ep4-bs32-seq1-decode.txt`
- 238 layers, mode=1 (inference), 无 wg 通信
- `--phase=prefill` 可切换到 prefill 阶段

### 2.2 训练 Training Workload

```bash
python -m workload_generator.SimAI_training_workload_generator \
    --frame=DeepSeek \
    --model_name=DeepSeek-671B \
    --world_size=1024 \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel=8 \
    --expert_model_parallel_size=8 \
    --num_experts=288 \
    --moe_router_topk=8 \
    --num_layers=61 \
    --hidden_size=7168 \
    --num_attention_heads=128 \
    --vocab_size=129280 \
    --micro_batch=1 \
    --global_batch=64 \
    --seq_length=4096 \
    --enable_sequence_parallel \
    --moe_enable \
    --n_shared_expert=1 \
    --n_dense_layers=3 \
    --qk_rope_dim=64 \
    --qk_nope_dim=128 \
    --q_lora_rank=1536 \
    --kv_lora_rank=512 \
    --v_head_dim=128 \
    --workload_only
```

### 2.3 关键注意事项

- **不加 `--aiob_enable`**: 不跑 GPU，compute_time 用默认值 1 (仅测网络通信)
- **必须 `--enable_sequence_parallel`**: MoE 模式强制要求
- **训练生成器 bug fix**: 原代码 `SIMAI_workload(model, args, None)` 改为 `SIMAI_workload(model, args, {})` (第986行)

---

## 三、运行模拟

### 3.1 CLI 参数说明

```
./simai_htsim [options]
  -w, --workload FILE     Workload 文件路径 (必须)
  --nodes N               总 GPU 数 (default: 8)
  --alpha N               每台机器最大 OCS 电路数 (default: 4)
  --speed N               链路速率 Mbps (default: 100000, 即 100Gbps)
  --reconf_delay N        OCS 重配延迟 us (default: 10)
  --dp_degree N           DP 并行度 (default: 1)
  --tp_degree N           TP 并行度 (default: 1)
  --pp_degree N           PP 并行度 (default: 1)
  --ep_degree N           EP 并行度 (default: 8)
  --gpus_per_server N     每台服务器 GPU 数 (default: 8)
  --queuesize N           队列大小 packets (default: 8)
  --iterations N          迭代次数 (default: 1)
  --ecs_only              强制所有流量走 ECS (禁用 OCS)
```

### 3.2 运行示例

```bash
BIN=/Users/apple/Project/nccl/SimAI-mixnet/simai_htsim
WL=/Users/apple/Project/nccl/SimAI-mixnet/SimAI/aicb/results/workload/DeepSeek-671B-world_size64-tp8-pp2-ep4-bs32-seq1-decode.txt
COMMON="--nodes=64 --tp_degree=8 --pp_degree=2 --ep_degree=4 --dp_degree=1 --gpus_per_server=8 --alpha=6 --speed=100000 --iterations=8"

# OCS+ECS mixnet
$BIN -w "$WL" $COMMON

# ECS-only 对比
$BIN -w "$WL" $COMMON --ecs_only
```

### 3.3 日志输出

每次运行自动创建日志目录：
```
log/EP_4_TP_8_DP_1_PP_2_OCS_GPU64_iter8_0413_103406/
├── trace.log        # 完整运行日志 (stdout)
│                    #   - pass: X finished at time: Y
│                    #   - [htsim] Events processed: ...
│                    #   - [SendFlow] OCS/ECS/NVLink 前几条
│                    #   - [RECONF] 重配置触发记录
├── stats.txt        # 统计摘要
│                    #   - 运行配置 (GPU数、并行度、网络模式、队列类型)
│                    #   - 流统计 (OCS/ECS/NVLink 流数量、字节、MB)
│                    #   - 流数量占比、字节占比
│                    #   - 跨机器字节占比 (OCS vs ECS)
│                    #   - 事件总数、模拟结束时间
└── fct_output.txt   # 每条 TCP flow 完成记录
```

stderr 输出日志目录路径: `[LOG] Output directory: ...`

---

## 四、网络拓扑架构

### 4.1 队列类型

- **ECS (fat-tree)**: `LOSSLESS_INPUT_ECN` — 无损 + ECN 标记，PFC 反压，不丢包
- **OCS (直连)**: `ECN` — ECN 标记 + tail drop，点对点直连基本无拥塞

### 4.2 OCS-ECS 选路规则

```
if (同机器):
    走 NVLink (900Gbps)
elif (comm_type == ALLTOALL_EP && OCS有可用电路 && !ecs_only模式):
    走 OCS (直连, alpha×100G 带宽, 3跳)
else:
    走 ECS (fat-tree, (8-alpha)×100G 带宽, 9跳)
```

- 在 `entry.h` 中实现, 通过 `g_force_ecs_only` 全局变量控制 ECS-only 模式
- OCS 电路按 EP region 动态分配: region_size = ep_degree × tp_degree / gpus_per_server
- 当前只有 EP AllToAll (com_type==4) 走 OCS，PP 等其他跨机器流量走 ECS

### 4.3 OCS 重配置机制

- **触发条件**: 一层 AllToAll 的所有跨机器 flow 在某个 region 内全部传完
- **流量矩阵**: 用当前层实际传输的 (src_machine, dst_machine, bytes) 构建
- **贪心分配**: 每次选流量最大的机器对分配一条 OCS 电路，直到 alpha 条用完
- **Immediate reconfig**: 暂停队列 → 直接重配置 → 恢复队列，不等排空 (避免死锁)
- **路由保持**: 运行中的 TCP flow 保留旧路由，只有新 flow 使用新拓扑

### 4.4 通信组到网络的映射

| 通信组 | CommType | 跨机器? | 网络路径 |
|--------|----------|---------|---------|
| TP | ALLREDUCE/REDUCESCATTER/ALLGATHER | 否 (同机) | NVLink |
| PP | pp_comm | 可能跨机 | ECS |
| EP | ALLTOALL_EP | 是 | OCS 或 ECS |
| DP | REDUCESCATTER/ALLGATHER | 是 | ECS |
| DP_EP | REDUCESCATTER_DP_EP/ALLGATHER_DP_EP | 是 | ECS |

### 4.5 关键参数关系

- `region_size = ep_degree × tp_degree / gpus_per_server`
- NVLink group = 8 GPU = 1台机器 (所有机内通信走 NVLink)
- OCS 带宽 = alpha × speed (直连点对点)
- ECS 带宽 = (8 - alpha) × speed (fat-tree 共享)

---

## 五、Workload 文件格式

### Header
```
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: <TP> ep: <EP> pp: <PP> [vpp: <N>] [ga: <N>] all_gpus: <total> [mode: 1] checkpoints: 0 checkpoint_initiates: 0 pp_comm: <bytes>
<num_layers>
```
- `mode: 1` = inference (仅 forward)
- 无 `mode` = training (forward + backward)
- `ga` = gradient accumulation steps
- `vpp` = 训练时为 num_layers_per_stage, 推理时为 1

### Layer 格式 (12 fields, tab-separated)
```
name  placeholder  fp_compute  fp_comm_type  fp_comm_size  ig_compute  ig_comm_type  ig_comm_size  wg_compute  wg_comm_type  wg_comm_size  wg_update_time
```

### CommType 映射
- fp/ig 后缀: `ALLREDUCE` → TP组, `ALLTOALL_EP` → EP组, `REDUCESCATTER`/`ALLGATHER` → TP组
- wg 后缀: `REDUCESCATTER`/`ALLGATHER` → DP组, `REDUCESCATTER_DP_EP`/`ALLGATHER_DP_EP` → DP_EP组
