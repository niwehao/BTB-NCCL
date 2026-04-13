## 1.执行流

### 1.AICB 生成工作负载文件

 **做了什么** ：把"我要用 8 张 A100、TP=8 训练 GPT-13B"这样的配置，翻译成具体的通信操作序列。

 **怎么做的** ：通过 MockedModel 模式——不真正运行模型，只模拟模型的前向/反向传播过程，记录每一步会产生什么通信。

以 `MockedMegatron` 为例，`forward()` 阶段：

* 经过每个 attention_layer → 产生一次 AllReduce（TP 组内同步 activation）
* 经过每个 mlp_layer → 产生一次 AllReduce
* `backward()` 阶段同理，但还有梯度聚合
* `step()` 阶段 → 可能有 AllGather/ReduceScatter（取决于 ZeRO 优化级别）

### 2.astra-sim 读取并调度

本质就是模拟一个系统

 **做了什么** ：读取 workload 文件，按照训练循环（forward → backward → step）的顺序，驱动通信操作的执行。

 **核心类 `Sys`** ：每个 GPU 一个 `Sys` 实例

```
Workload.call() → 根据 parallelismPolicy 选择迭代策略
    │
    ├─ iterate_hybrid_parallel_Transformer_fwd_in_bckwd()
    │   遍历每一层:
    │     1. 计算阶段: 等待 forward_compute_time 纳秒
    │     2. 通信阶段: 调用 generate_collective()
    │        │
    │        ▼
    │   generate_all_reduce() / generate_reduce_scatter() / ...
    │        │
    │        ▼
    │   MockNcclChannel 将集合通信分解为点对点传输
    │        │
    │        ▼
    │   sim_send() / sim_recv() → 交给网络后端
    │
    └─ 所有层完成后 → report() 输出统计
```

 **关键转换——MockNccl** ：

这是 SimCCL 的基础版本，负责将高层集合通信翻译为具体的点对点传输：

| 集合通信             | MockNccl 分解方式                                                    |
| -------------------- | -------------------------------------------------------------------- |
| AllReduce(8卡, 20MB) | → Ring 算法：7 步 ReduceScatter + 7 步 AllGather，每步传 20/8=2.5MB |
| AllGather(8卡, 10MB) | → Ring 算法：7 步，每步每卡发 10/8=1.25MB 给下一个卡                |
| AllToAll(8卡, 16MB)  | → 每卡向其他 7 卡各发 16/8=2MB，共 56 个点对点传输                  |

### 3.网络后端执行

**`AstraSimNetwork`** 是 astra-sim 和 NS-3 之间的 **桥梁** ：

```cpp
// AstraSimNetwork.cc
sim_send(dst, count, type, tag){
// 创建 RdmaClient 应用
// 设置: 源IP=本节点, 目的IP=dst节点, 传输大小=count字节
// 挂载完成回调 → 通知 astra-sim 这笔传输完成了
}

sim_get_time(){
returnSimulator::Now().GetNanoSeconds();// NS-3 的仿真时钟
}
```

 **NS-3 内部发生什么** （以一笔 2.5MB 的 send 为例）：

```
RdmaClient 发起 QP 连接
    │
    ▼
RdmaHw.GetNxtPacket() → 按 MTU 切分成 ~1700 个包
    │
    ▼
每个包经过:
    ┌──────────────────────────────────────────────┐
    │ 发送端 NIC                                    │
    │ ├─ 速率控制 (DCQCN/HPCC/TIMELY)               │
    │ └─ 按当前速率 m_rate 控制发送间隔               │
    │                                               │
    │ 交换机 (SwitchNode)                            │
    │ ├─ ECMP 选路 (多路径负载均衡)                    │
    │ ├─ SwitchMmu 缓冲区管理                        │
    │ │   ├─ 队列超过 kmin → 标记 ECN (CE)            │
    │ │   └─ 队列超过 headroom → 触发 PFC 暂停        │
    │ └─ 转发到出口端口                               │
    │                                               │
    │ 接收端 NIC                                     │
    │ ├─ 检查序列号，发送 ACK/NACK                    │
    │ └─ 如果 ECN 标记 → 发送 CNP 给发送端             │
    │                                               │
    │ 发送端收到反馈                                   │
    │ ├─ 收到 ACK → 滑动窗口前移                      │
    │ ├─ 收到 CNP → 降速 (DCQCN: rate *= (1-α/2))   │
    │ ├─ 收到 INT → 精确调速 (HPCC: 基于链路利用率)    │
    │ └─ 超时无 ACK → NACK 重传                      │
    └──────────────────────────────────────────────┘
```

### 4.结果返回

NS-3 中一笔传输完成后（所有包都 ACK 了）：

```
QpComplete 回调 → AstraSimNetwork 的 notifyAppFinish
    → astra-sim Sys 收到通知 → 标记这个 chunk 完成
    → 如果所有 chunk 都完成 → 标记整个集合通信完成
    → 如果所有层的通信都完成 → Workload.report() 输出统计
```

 **输出** ：

* `ncclFlowModel_EndToEnd.csv`：每个集合通信的端到端延迟
* FCT 文件、带宽/队列长度监控文件


如果不需要包级精度，可以用解析模型替代 NS-3：

```
astra-sim → AnalyticalNetwork（不经过 NS-3）
    → 用带宽公式估算: latency = size / bandwidth + alpha * hops
    → 速度快 100-1000 倍，但不模拟拥塞
```

## 2.推理场景：Vidur 路径详解

### 5.请求生成

```python
# Vidur 的请求生成器
RequestGenerator.generate()
    → 按泊松/Gamma/trace 生成请求到达时间
    → 每个请求:(arrived_at, num_prefill_tokens, num_decode_tokens)
    → 包装为 RequestArrivalEvent 加入事件队列
```

### 6.离散事件仿真循环

```python
while event_queue:
    event = heappop(event_queue)# 取出最早的事件
    new_events = event.handle_event()# 处理，可能产生新事件
  
# 事件链:
# RequestArrival → GlobalSchedule → ReplicaSchedule
#   → ReplicaStageSchedule → BatchStageArrival
#   → BatchStageEnd → BatchEnd → (下一个 Decode 迭代)
```

 **调度器决定** ：

* 哪些请求合成一个 Batch？（continuous batching vs static batching）
* Batch 分配到哪个 Replica？（轮询 / 最少排队 / PD 分离）

### 7.SimAI 通信预测（关键集成点）

当需要计算 Batch 的 TP AllReduce 时间时：

```python
# TPTimePredictor.get_execution_time()

# 1. 计算通信量
all_reduce_bytes = hidden_size × num_tokens × 2# FP16

# 2. 生成 SimAI 格式的 workload 文件
workload.append_work_item(WorkItem(
    name="layer0",
    forward_comm="ALLREDUCE",
    forward_comm_size=all_reduce_bytes
))
workload.dump_file("allreduce_xxx.txt")

# 3. 调用 SimAI 仿真器（子进程）
command =f'SimAI_simulator -t 16 -w {workload_file} -n {topo} -c {conf}'
subprocess.run(command)

# 4. 读取结果 CSV → 提取通信延迟(μs→ms)
# 5. 加上 NCCL CPU 开销 → 返回给 Vidur
return latency_ms + nccl_overhead
```

这里 Vidur 实际上**调用了完整的 astra-sim + NS-3 流程**作为子进程，只是输入简化为单层 AllReduce。

### 8.一句话总结每个组件的职责

| 组件                      | 一句话                                                  |
| ------------------------- | ------------------------------------------------------- |
| **AICB**            | "告诉你训练一个模型**需要哪些通信** "             |
| **astra-sim (Sys)** | "按训练循环**调度这些通信的执行顺序** "           |
| **MockNccl**        | "把 AllReduce**拆成具体的点对点包** "             |
| **NS-3 RDMA**       | "模拟每个包**在真实网络中的传输过程** "           |
| **analytical**      | "用公式**快速估算**传输时间，不走网络"            |
| **Vidur**           | "模拟推理请求的**到达-调度-执行全过程** "         |
| **TPTimePredictor** | "Vidur 和 SimAI 的**桥梁** ，调 SimAI 算通信时间" |

## 3.SimAI 的 MockNccl 模拟到了什么级别

MockNccl 模拟到了 **NCCL 的通信拓扑构建 + 算法选择 + Flow 分解** 这一层，但**没有模拟 NCCL 的 GPU kernel 执行、内存拷贝、CUDA stream 调度**。

### 已模拟（通信调度层）

```
┌─────────────────────────────────────────────────────────┐
│                   NCCL 真实实现                          │
│                                                         │
│  ┌──────────────────────────┐                           │
│  │ 1. 通信组 (Communicator)  │ ✅ MockNcclGroup         │
│  │    TP/DP/PP/EP/DP_EP 分组 │    完整模拟5种组类型       │
│  │    rank→group 映射        │    含 NVSwitch 分配       │
│  ├──────────────────────────┤                           │
│  │ 2. 拓扑发现 + Channel 构建│ ✅ MockNcclChannel        │
│  │    Ring channel          │    模拟节点内/节点间环构建   │
│  │    Tree channel (双二叉树)│    模拟 Double Binary Tree │
│  │    NVLS channel          │    模拟 NVLink Switch 通道 │
│  │    NVLS-Tree channel     │    模拟 NVLS+Tree 混合通道  │
│  ├──────────────────────────┤                           │
│  │ 3. 算法选择               │ ✅ get_algo_proto_info     │
│  │    根据 GPU 型号选算法     │    A100→Ring              │
│  │    根据组大小选算法        │    H100+8卡→NVLS          │
│  │    根据数据量选算法        │    其余→Ring              │
│  ├──────────────────────────┤                           │
│  │ 4. Flow 分解              │ ✅ genXxxFlowModels       │
│  │    集合操作→点对点 flow    │    精确到每个 chunk 的      │
│  │    含依赖关系 (DAG)       │    src, dst, size, deps   │
│  │    含 PXN 代理发送        │                           │
│  ├──────────────────────────┤                           │
│  │ 5. 硬件延迟/带宽参数      │ ✅ MockNccl.h 常量表       │
│  │    baseLat[algo][proto]   │    直接搬 NCCL 源码的表     │
│  │    hwLat[NVL/PCI/NET]     │    含 treeCorrectionFactor │
│  │    perChMaxBw 表          │    区分 Volta/Ampere/Hopper│
│  └──────────────────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

### 未模拟（GPU 执行层 + 传输协议层）

```
┌─────────────────────────────────────────────────────────┐
│                   未模拟的部分                            │
│                                                         │
│  ❌ GPU Kernel 执行                                      │
│     NCCL 真实用 GPU kernel 做 reduce 计算                 │
│     MockNccl 只关心"传了多少数据"，不模拟计算                │
│                                                         │
│  ❌ 传输协议 (LL / LL128 / Simple)                        │
│     NCCL 有3种协议，影响实际吞吐和延迟：                     │
│     - LL (Low Latency): 8B 有效 + 8B flag，小消息快        │
│     - LL128: 120B 有效 + 8B flag，中等消息                 │
│     - Simple: 大消息，全靠 DMA                            │
│     MockNccl 的 protocol 字段设为 UNDEF，未实际使用         │
│                                                         │
│  ❌ Proxy Thread + CUDA Stream 调度                       │
│     NCCL 真实通过 proxy thread 异步驱动网络发送              │
│     通过 CUDA stream 管理 kernel 和 DMA 的重叠              │
│     MockNccl 不模拟这一层                                  │
│                                                         │
│  ❌ 内存注册 (GDR / IPC / mmap)                           │
│     NCCL 对 NVLink 用 IPC，对 RDMA 用 GDRCopy              │
│     MockNccl 不区分内存访问路径                             │
│                                                         │
│  ❌ 动态拓扑探测 (ncclTopoGetSystem)                       │
│     NCCL 真实读 PCIe/NVLink 拓扑来决定 channel 路径         │
│     MockNccl 用硬编码的 gpus_per_node 和 NVSwitch 列表      │
│                                                         │
│  ❌ CollNet (SHARP) 支持                                  │
│     定义了常量但未实现 flow 生成                             │
│  ❌ NVLS-Tree flow 生成                                   │
│     genAllReduceFlowModels 中 NVLS_TREE 分支 return {}     │
│  ❌ 分块重叠 (Pipeline within collective)                  │
│     NCCL 实际会把大消息切成多个 chunk 并流水线重叠             │
│     MockNccl 只做了 chunk 切分但未模拟 kernel 级流水线        │
└─────────────────────────────────────────────────────────┘
```

MockNccl 的选择逻辑 vs 真实 NCCL：

| 条件                                  | MockNccl 选择       | 真实 NCCL                        |
| :------------------------------------ | :------------------ | :------------------------------- |
| TP AllReduce + A100/A800              | Ring                | 取决于消息大小，小→Tree，大→Ring |
| TP AllReduce + H100/H800 + 8卡 + NVLS | **NVLS**            | NVLS（一致）                     |
| TP AllReduce + H100 + <8卡            | Ring                | 取决于消息大小和拓扑             |
| DP/EP AllReduce                       | 一律 Ring           | 根据带宽模型选 Tree 或 Ring      |
| AllGather/ReduceScatter               | 一律 Ring           | 可选 Ring/NVLS                   |
| AllToAll                              | 一律 Ring（全互连） | 类似                             |
| 协议选择                              | **UNDEF（不选）**   | 根据消息大小选 LL/LL128/Simple   |

**简化程度**：真实 NCCL 的 `ncclTunerPlugin` 会基于 `bandwidths[func][algo][proto]` 选最优（算法, 协议）组合。MockNccl 只硬编码了算法，协议一律未定义。

### Flow 分解的精确度

以 **Ring AllReduce (8卡, 20MB)** 为例：

```
MockNccl 实际生成的 flow:
├─ chunksize = 20MB / 8卡 / nChannels
├─ 阶段1: ReduceScatter（7步 Ring）
│   step 0: rank0→rank1 (chunksize), rank1→rank2, ..., rank7→rank0
│   step 1: rank0→rank1, rank1→rank2, ...
│   ...共 7 步，每步 8 个 flow
├─ 阶段2: AllGather（7步 Ring）
│   同上 7 步
└─ 每个 SingleFlow 包含:
    ├─ flow_id, src, dst, flow_size
    ├─ parent_flow_id / child_flow_id  ← 依赖关系 DAG
    ├─ channel_id, chunk_id, chunk_count
    └─ conn_type: "RING" / "NVLS" / "NVL" / "NET"
```

这些 `SingleFlow` 最终被交给 astra-sim 的 `sim_send()/sim_recv()` → NS-3 做包级仿真。

### 总结：在 NCCL 架构中的位置

```
真实 NCCL 分层:
┌─────────────────────────┐
│   用户 API              │  ncclAllReduce()
├─────────────────────────┤
│   算法选择 + 拓扑        │  ← MockNccl 模拟到这里
│   Channel 构建           │  ← 这里
│   Flow/Chunk 分解        │  ← 这里
├═════════════════════════╡  ══ 分界线 ══
│   Proxy Thread          │  ← 未模拟
│   GPU Kernel (LL/Simple)│  ← 未模拟
│   CUDA Stream 调度       │  ← 未模拟
├─────────────────────────┤
│   传输层                 │  ← NS-3 在这里接管
│   (NVLink / RDMA / PCIe)│     用包级仿真替代真实硬件
└─────────────────────────┘
```

所以 MockNccl 本质上是：**用 NCCL 的通信拓扑和算法选择逻辑，生成一个 "谁给谁发多少数据、什么顺序" 的 DAG，然后把实际传输交给 NS-3 去精确模拟**。它跳过了 GPU kernel 层面的细节（因为那部分对网络仿真不重要），专注于"网络上会看到什么样的流量模式"。

### NCCL和RDMA关系

**NCCL 是上层的集合通信库，RDMA 是它底层可以选择的传输方式之一。**

**NCCL（NVIDIA Collective Communications Library）是什么：** 它是 NVIDIA 开发的一个库，专门用来做多 GPU 之间的集合通信操作，比如 AllReduce、AllGather、Broadcast、ReduceScatter 等。你写分布式训练代码时（PyTorch DDP、Megatron 等），框架底层调用的就是 NCCL 来完成 GPU 之间的数据同步。

**NCCL 要解决的问题是：** 给定一堆 GPU（可能在同一台机器上，也可能分布在不同机器上），怎么高效地把梯度汇总、把参数广播出去。它需要决定通信的拓扑（ring、tree、还是其他）、数据怎么切分、走哪条通路。

**NCCL 底层的传输通路有几种选择：**

同一台机器内的 GPU 之间，NCCL 优先走 NVLink 或 NVSwitch，这是 GPU 之间直连的高速通道，带宽最大、延迟最低。

跨机器的 GPU 之间，数据必须走网络。这时候 NCCL 支持两种方式：一种是走传统的 TCP/IP Socket，另一种就是走 RDMA。如果用 RDMA，又分两种具体实现——InfiniBand Verbs 或者 RoCE，它们都通过 libibverbs 这个接口来调用。

**为什么跨机通信几乎都选 RDMA：** 大模型训练中跨机通信的数据量非常大（每次迭代可能要同步几个 GB 的梯度），而且通信和计算是 pipeline 起来的，网络稍微慢一点就会成为瓶颈，GPU 就得空等。RDMA 的高带宽、低延迟、CPU 零开销这些特性在这里就非常关键。如果用 TCP，光 CPU 处理协议栈的开销就会吃掉不少性能，延迟也高很多。

**所以整个调用链大致是这样的：**

训练框架（PyTorch）→ NCCL（决定通信拓扑和算法）→ 同机走 NVLink，跨机走 libibverbs（RDMA）或 Socket（TCP）→ 网卡 → 网络

**一个类比：** NCCL 就像一个物流调度系统，它决定货物怎么拆分、走什么路线。RDMA 就像高速公路，TCP 就像普通公路。NCCL 可以选择走哪种路，但在大规模训练这种"运大量货、要求快"的场景下，自然会选高速公路。

## 4.NS-3 如何模拟 NCCL 的传输层

NS-3 **并不直接模拟 NCCL 的传输层**。它模拟的是 **RDMA 网络硬件本身**——NIC、交换机、链路、拥塞控制。NCCL 传输层在真实系统中的工作（创建 QP、注册内存、调用 ibv_post_send）被一个**适配层**（`entry.h` + `AstraSimNetwork`）替代，直接把 MockNccl 生成的 flow 转化为 NS-3 的 RDMA QP。

### 完整链路：从 MockNccl flow 到 NS-3 数据包

```
MockNccl SingleFlow {src=0, dst=1, size=2.5MB, conn_type="RING"}
    │
    │  astra-sim Sys::sim_send(dst=1, count=2.5MB)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ ASTRASimNetwork::sim_send()         [AstraSimNetwork.cc]│
│                                                         │
│  1. 记录 sentHash[tag, (src,dst)] = {count, callback}   │
│  2. 调用 SendFlow(src=0, dst=1, 2.5MB, callback, tag)   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ SendFlow()                                [entry.h:107] │
│                                                         │
│  // 计算每个 QP 的分包大小                                │
│  PacketCount = maxPacketCount / _QPS_PER_CONNECTION_     │
│                                                         │
│  // 创建 NS-3 的 RdmaClient 应用                        │
│  RdmaClientHelper clientHelper(                         │
│      pg=3,                    // 优先级组                 │
│      serverAddress[src],      // 源 IP                   │
│      serverAddress[dst],      // 目的 IP                 │
│      port,                    // 源端口（递增分配）         │
│      dport=100,               // 目的端口                │
│      real_PacketCount,        // 传输字节数               │
│      maxBdp,                  // 窗口大小 = max(BDP)     │
│      pairRtt[src][dst],       // 基准 RTT                │
│      msg_handler, fun_arg,    // 完成回调                 │
│      tag, src, dst            // flow 标识               │
│  );                                                     │
│  if (nvls_on) clientHelper.SetAttribute("NVLS_enable")  │
│                                                         │
│  // 安装到源节点，延迟 send_lat(默认6μs) 后启动           │
│  appCon = clientHelper.Install(n.Get(src));              │
│  appCon.Start(Time(send_lat));   // ← 模拟 NCCL 发送延迟 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼  NS-3 内部接管
┌─────────────────────────────────────────────────────────┐
│ RdmaClient::StartApplication()                          │
│                                                         │
│  // 通过 RdmaDriver 创建 Queue Pair                     │
│  rdmaHw->AddQueuePair(                                  │
│      src, dst, tag,                                     │
│      size=2.5MB,                                        │
│      pg=3,           // 优先级                           │
│      sip, dip,       // IP 地址                          │
│      sport, dport,   // 端口号                           │
│      win=maxBdp,     // 发送窗口                         │
│      baseRtt         // 基准 RTT                         │
│  );                                                     │
│  // 从这里开始，NS-3 完全按包级模拟                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         NS-3 RDMA 包级仿真 (rdma-hw.cc)                 │
│                                                         │
│  while (snd_nxt < m_size) {                             │
│    // 1. 生成数据包                                      │
│    pkt = GetNxtPacket(qp)  // MTU 切分，~1000B/包        │
│                                                         │
│    // 2. 速率控制 - 什么时候发下一个包                     │
│    nextAvail = now + pktSize / m_rate                    │
│                                                         │
│    // 3. 包进入网络拓扑...                               │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
```

### NS-3 网络初始化：构建"虚拟数据中心"

`SetupNetwork()` 函数（`common.h:694`）构建了一个完整的虚拟 RDMA 数据中心：

```
SetupNetwork() 做了什么:

1. 读拓扑文件 → 创建节点
   ┌─────────────────────────────────────────────┐
   │ node_type = 0 → GPU 节点 (CreateObject<Node>)│
   │ node_type = 1 → 交换机 (CreateObject<SwitchNode>)
   │ node_type = 2 → NVSwitch (CreateObject<NVSwitchNode>)
   └─────────────────────────────────────────────┘

2. 创建链路 (QbbHelper)
   每条链路配置: DataRate, Delay, ErrorRate
   安装 QbbNetDevice (支持 PFC 的网卡)
   ↓
3. 配置交换机 MMU
   每个端口: ECN 阈值 (kmin/kmax/pmax)
             PFC headroom = rate × delay × 3
             缓冲区大小 (默认 16MB)
   ↓
4. 创建 RDMA 硬件 (每个 GPU 节点)
   RdmaHw 配置:
   ├─ cc_mode (拥塞控制算法选择)
   │   1=DCQCN, 3=HPCC, 7=TIMELY, 8=DCTCP, 10=HPCC-PINT
   ├─ RateAI / RateHAI (加性增/超加性增速率)
   ├─ AlphaResumInterval (DCQCN α 更新间隔)
   ├─ RPTimer (速率恢复定时器)
   ├─ TargetUtil = 0.95 (HPCC 目标利用率)
   ├─ Mtu = 1000B (单包有效载荷)
   └─ GPUsPerServer (用于 NVSwitch 路由判断)
   ↓
5. 注册回调
   QpComplete → qp_finish()  // QP 传输完成
   SendComplete → send_finish()  // 发送完成
   ↓
6. 计算路由表 + 全局 RTT/BDP
   CalculateRoutes() → ECMP 最短路径
   计算每对节点的 pairRtt, pairBw, pairBdp
```

### 每个包在 NS-3 中经历什么

```
GPU 0 发送 2.5MB 给 GPU 1 (跨机, Ring AllReduce 的一步)

时间线:
────────────────────────────────────────────────────────────

t=6μs    RdmaClient 启动 (send_lat=6μs，模拟 NCCL CPU 开销)
         RdmaHw::AddQueuePair() 创建 QP
         初始速率 = 线速 (如 100Gbps)

t=6.008μs 第1个包 (1000B) 从 NIC 发出
          ┌──────────────────────────────────┐
          │ 包结构:                           │
          │ ├─ PPP Header                    │
          │ ├─ IPv4 Header (ECN 字段)         │
          │ ├─ UDP Header                    │
          │ ├─ INT Header (如果 HPCC 模式)    │
          │ │   └─ 每跳: lineRate, time,     │
          │ │          bytes, qlen           │
          │ └─ Payload (1000B)               │
          └──────────────────────────────────┘

t=6.01μs  包到达 ToR 交换机
          SwitchNode::SwitchReceiveFromDevice()
          ├─ ECMP 选路 → 选出口端口
          ├─ SwitchMmu::CheckIngressAdmission() → 缓冲区够不够
          │   ├─ 够 → 入队
          │   └─ 不够 → 丢包 (实际走 PFC 暂停)
          ├─ SwitchMmu::ShouldSendCN() → 需要标 ECN 吗？
          │   └─ 队列 > kmin → 概率标记 ECN CE
          ├─ CheckAndSendPfc() → 需要 PFC 暂停吗？
          │   └─ 队列 > headroom threshold → 发 PFC PAUSE 帧
          └─ SwitchNotifyDequeue() → 包出队发往下一跳

t=6.02μs  包到达目的 NIC
          RdmaHw::Receive()
          ├─ ReceiverCheckSeq() → 序列号对不对？
          │   ├─ 对 → 发 ACK
          │   └─ 不对 → 发 NACK
          └─ 如果 ECN CE → 发 CNP (拥塞通知) 回源端

t=6.03μs  源端收到 ACK
          RdmaHw::ReceiveAck()
          ├─ 窗口滑动 → snd_una 前移
          └─ 根据 cc_mode 调速:

          cc_mode=1 (DCQCN):
          ├─ 收到 CNP → rate *= (1 - α/2), 重置定时器
          ├─ 定时器到期无 CNP → 进入恢复阶段
          │   Fast Recovery → Active Increase → Hyper Increase
          └─ α 周期性衰减: α = (1-g)×α + g×(收到CNP?1:0)

          cc_mode=3 (HPCC):
          ├─ 从 INT Header 提取每跳利用率 u
          │   u = (bytes_delta × 8) / (time_delta × lineRate)
          ├─ 找最拥塞跳: u_max = max(u[hop])
          ├─ 目标速率: Rc_new = Rc × u_target / u_max
          └─ 平滑调整: rate = rate×(1-1/N) + Rc_new/N

t=...     所有 ~2500 个包传完 (2.5MB / 1000B)
          RdmaHw::QpComplete(qp)
```

### 回调返回：NS-3 → astra-sim

```
NS-3 QP 完成
    │
    ▼
qp_finish()                              [entry.h:299]
├─ 记录 FCT (Flow Completion Time) 到文件
├─ 通过 sender_src_port_map 找回 flowTag
├─ 累加 received_chunksize
├─ is_receive_finished() → 所有 QP 都完成了？
│   └─ 如果多 QP (_QPS_PER_CONNECTION_>1)
│      需要等所有 QP 都回调才算完成
└─ notify_receiver_receive_data()
    ├─ 查 expeRecvHash → astra-sim 注册的接收回调
    ├─ 字节数匹配 → 触发 msg_handler 回调
    └─ astra-sim Sys 收到通知
       → 这笔 flow 完成
       → 检查 DAG 依赖 → 触发后续 flow
       → 所有 flow 完成 → 集合通信完成

send_finish()                             [entry.h:347]
├─ 通过 sender_src_port_map 找回 flowTag
├─ is_sending_finished() → 所有 QP 发送完毕？
└─ notify_sender_sending_finished()
    └─ 通知 astra-sim 发送侧完成
```

### 对应关系总结

| NCCL 真实传输层            | NS-3 模拟对应                                |
| :------------------------- | :------------------------------------------- |
| `ibv_reg_mr()` 注册内存    | 不模拟（无需）                               |
| `ibv_create_qp()` 创建 QP  | `RdmaHw::AddQueuePair()`                     |
| `ibv_post_send()` 发送 WR  | `RdmaHw::GetNxtPacket()` 按 MTU 生成包       |
| NIC DMA 读取数据           | 不模拟（直接生成包）                         |
| 包经过 PCIe → NIC → 线缆   | `QbbNetDevice → QbbChannel` 链路仿真         |
| 交换机转发 + 缓冲          | `SwitchNode` + `SwitchMmu` 完整模拟          |
| NVSwitch 转发              | `NVSwitchNode` 完整模拟                      |
| ECN 标记                   | `SwitchMmu::ShouldSendCN()`                  |
| PFC PAUSE/RESUME           | `CheckAndSendPfc()` / `CheckAndSendResume()` |
| 拥塞控制 (DCQCN 等)        | `RdmaHw` 中 5 种 CC 算法完整实现             |
| ACK/NACK 机制              | `RdmaHw::ReceiveAck()` 完整实现              |
| `ibv_poll_cq()` 完成通知   | `QpComplete` trace 回调 → `qp_finish()`      |
| Proxy thread 异步发送      | 不模拟（NS-3 事件驱动天然异步）              |
| CUDA kernel 内 reduce 计算 | 不模拟（只关心网络延迟）                     |
| GDRCopy / IPC 内存访问     | 不模拟                                       |

**一句话**：NS-3 模拟的是 **RDMA NIC 以下的一切**（包括 NIC 的速率控制、交换机转发和缓冲、链路传播）。NCCL 传输层在 NIC 以上做的事（proxy thread、CUDA kernel、内存注册）被直接跳过，由适配层（`SendFlow()` → `RdmaClientHelper`）把 MockNccl 的 flow 直接注入 NS-3 的 RDMA QP。

## 5.RDMA 模拟 vs TCP 模拟：到底差在哪

差别不在"传数据"本身，而在 **丢包处理机制完全相反**——RDMA 网络是**无损（lossless）**的，TCP 网络是**有损（lossy）**的。这导致拥塞控制、交换机行为、性能特征全都不同，模拟结果会有本质差异。

### 核心差异对比

#### 1. 丢包 vs 不丢包：最根本的区别

```
TCP 网络（有损）:
    交换机缓冲区满了 → 丢包 → 发送端超时/快速重传 → 降速

RDMA 网络（无损）:
    交换机缓冲区快满了 → 发 PFC PAUSE 帧 → 上游停发 → 不丢包
```

NS-3 中 PFC 的实现（`switch-node.cc:92`）：

```cpp
void SwitchNode::CheckAndSendPfc(uint32_t inDev, uint32_t qIndex) {
    if (m_mmu->CheckShouldPause(inDev, qIndex)) {
        device->SendPfc(qIndex, 0);    // 发送 PAUSE 帧
        m_mmu->SetPause(inDev, qIndex); // 标记该端口该优先级暂停
    }
}
```

**这在 TCP 模拟里完全不存在。** TCP 交换机就是 tail drop 或 RED/WRED，满了直接扔。

#### 2. 拥塞控制哲学完全不同

```
┌─────────────────────────┬──────────────────────────────┐
│       TCP                │         RDMA                  │
├─────────────────────────┼──────────────────────────────┤
│ 信号: 丢包 / RTT 变化    │ 信号: ECN 标记 / INT 遥测     │
│                          │                               │
│ 在哪里做: 内核协议栈       │ 在哪里做: NIC 硬件             │
│  (软件, ~μs 级响应)       │  (硬件, ~ns 级响应)            │
│                          │                               │
│ 算法: CUBIC / BBR / Reno │ 算法: DCQCN / HPCC / TIMELY   │
│                          │                               │
│ 窗口 or 速率: 窗口控制    │ 窗口 or 速率: 速率控制          │
│  cwnd 个包              │  m_rate (bps) 精确控速         │
│                          │                               │
│ 反馈路径:                 │ 反馈路径:                      │
│  丢包→超时→重传           │  ECN→交换机标记→CNP→NIC降速    │
│  (粗粒度, ~RTT 级)       │  (细粒度, 每个 ACK)            │
└─────────────────────────┴──────────────────────────────┘
```

NS-3 里 RDMA 的速率控制（`rdma-hw.cc:546`）:

```cpp
// 收到 ACK 后，根据 cc_mode 调速
if (cnp) {
    if (m_cc_mode == 1)  // DCQCN
        cnp_received_mlx(qp);  // 立即降速: rate *= (1 - α/2)
}
if (m_cc_mode == 3)       // HPCC
    HandleAckHp(qp, p, ch);   // 从 INT 头读每跳利用率，精确调速
else if (m_cc_mode == 7)  // TIMELY
    HandleAckTimely(qp, p, ch); // 基于 RTT 梯度调速
else if (m_cc_mode == 8)  // DCTCP
    HandleAckDctcp(qp, p, ch); // 基于 ECN 比例调窗口
```

TCP 根本没有 CNP 这个东西，也没有 INT 头。

#### 3. 交换机行为完全不同

```
TCP 交换机 (NS-3 标准模块):
┌──────────────────────────────────┐
│ 包到达 → 查路由 → 入队           │
│ 队列满 → tail drop (直接丢)      │
│ 可选 RED: 概率早期丢包            │
│ 没有 PFC, 没有 headroom          │
│ 出队 → 转发，完事                 │
└──────────────────────────────────┘

RDMA 交换机 (SimAI 的 SwitchNode):
┌──────────────────────────────────┐
│ 包到达 → ECMP 选路               │
│                                   │
│ 入口准入控制:                      │
│  CheckIngressAdmission()          │
│  └─ 共享缓冲区 + headroom 检查    │
│                                   │
│ 出口准入控制:                      │
│  CheckEgressAdmission()           │
│  └─ 出口队列深度检查               │
│                                   │
│ ECN 标记:                         │
│  ShouldSendCN()                   │
│  └─ 队列 ∈ [kmin, kmax] → 概率标记│
│  └─ 队列 > kmax → 必定标记        │
│                                   │
│ PFC 流控:                         │
│  CheckAndSendPfc()                │
│  └─ 队列 > threshold → PAUSE 帧  │
│  CheckAndSendResume()             │
│  └─ 队列 < threshold → RESUME 帧 │
│                                   │
│ INT 遥测 (HPCC 模式):             │
│  每跳写入: 链路速率, 字节数,       │
│            时间戳, 队列长度         │
└──────────────────────────────────┘
```

`switch-node.cc:126` 的入口/出口双重准入控制:

```cpp
if (m_mmu->CheckIngressAdmission(inDev, qIndex, p->GetSize())
    && m_mmu->CheckEgressAdmission(idx, qIndex, p->GetSize())) {
    // 通过 → 入队
} else {
    return; // 丢弃 (但正常情况下 PFC 会在之前暂停，不会走到这里)
}
CheckAndSendPfc(inDev, qIndex);  // 每个包都检查是否需要暂停
```

#### 4. 传输层协议差异

```
TCP:                              RDMA (NS-3 模拟):
─────────                         ─────────────────
三次握手建连                        无连接建立 (QP 直接创建)
字节流语义                          消息语义 (固定大小传输)
内核协议栈处理                      NIC 硬件处理 (RdmaHw)
慢启动 + 拥塞避免                   初始即线速，被动降速
ACK 时钟驱动                       速率时钟驱动 (m_rate)
超时重传 (RTO ~200ms)              NACK 快速重传 (~μs)
Nagle/Delayed ACK                  无 (每 N 个包 ACK 一次)
拥塞窗口 cwnd (离散)               发送速率 m_rate (连续)
```

代码里的速率控制 vs TCP 窗口控制:

```cpp
// RDMA: 精确的速率定时器控制下一个包何时发出
void RdmaHw::UpdateNextAvail(Ptr<RdmaQueuePair> qp, Time interframeGap, uint32_t pkt_size) {
    Time sendingTime = interframeGap + Seconds(qp->m_rate.CalculateTxTime(pkt_size));
    qp->m_nextAvail = Simulator::Now() + sendingTime;
    // → 包间隔由 m_rate 精确控制
}

// TCP: 窗口控制，有多少空位就发多少
// cwnd 个包在飞 → 收到 ACK → cwnd++ (加性增) 或 cwnd/=2 (乘性减)
```

#### 5. 对训练性能模拟的影响

```
为什么 RDMA 训练场景不能用 TCP 模拟?

场景: 128 卡 AllReduce, 每卡发 64MB

TCP 模拟会看到:
├─ 慢启动阶段: 前几个 RTT 发得很慢 (~ms 级额外开销)
├─ 多流竞争 → 丢包 → cwnd 砍半 → 吞吐震荡
├─ 尾延迟: 重传导致个别 flow 延迟 10x-100x
├─ Incast 问题: 128→1 的 AllReduce 归约阶段
│  大量丢包 → 大量超时 → 性能崩塌
└─ 总延迟: 显著偏高

RDMA 模拟会看到:
├─ 无慢启动: 初始即线速发送
├─ 无丢包: PFC 保证，但可能有 PFC 暂停
├─ 拥塞反馈快: ECN/CNP ~μs 级响应
├─ Incast 问题: 128→1 时
│  交换机标 ECN → CNP → 所有发送端同时降速
│  PFC 兜底 → 不丢包但有延迟抖动
│  可能触发 PFC 风暴 (headroom 被打满)
└─ 总延迟: 低得多，但有 PFC 导致的尾延迟

差距有多大？
  同样 Incast 场景:
  TCP:  延迟可能 10ms+ (重传)
  RDMA: 延迟可能 100μs (PFC 暂停一下就恢复)
  → 差 100 倍
```

#### 6. 模拟出来但现实中独有的 RDMA 问题

NS-3 的 RDMA 模拟能捕捉到一些 **TCP 模拟永远看不到的问题**：

| RDMA 特有问题       | NS-3 怎么模拟                 | TCP 里对应什么          |
| :------------------ | :---------------------------- | :---------------------- |
| **PFC 死锁**        | 环形依赖的 PFC PAUSE 互锁     | 不存在 (TCP 丢包不暂停) |
| **PFC 风暴**        | 一个端口暂停 → 传播到整个网络 | 不存在                  |
| **ECMP 负载不均**   | 同样有，但 RDMA 下更敏感      | 也有，但丢包能"自平衡"  |
| **Victim flow**     | 无辜流被 PFC 暂停             | 不存在（各流独立丢包）  |
| **NVSwitch 内通信** | NVSwitchNode 单独模拟         | 完全不涉及              |
| **INT 遥测精度**    | IntHeader 每跳写入利用率      | 不存在                  |

#### 总结

```
如果把 RDMA 换成 TCP 模拟:

1. 缺少 PFC → 无损变有损 → Incast 性能预测完全错误
2. 缺少 DCQCN/HPCC → 拥塞响应差 100 倍 (ms vs μs)
3. 多了慢启动 → 短 flow 延迟被高估
4. 缺少 INT 遥测 → HPCC 这种算法根本跑不了
5. 缺少 NVSwitch → 机内通信无法建模
6. 训练的 AllReduce 延迟预测可能偏差 10x-100x
```

**所以 SimAI 不是"换个协议名字"，而是从交换机缓冲区管理、流控机制、拥塞控制算法到 NIC 行为，整套都是专门为 RDMA 数据中心网络写的。** 如果用标准 NS-3 的 TCP 模块去模拟，得到的训练通信延迟数据基本不可用。

## 6.SimAI NS-3 数据中心网络拓扑

### 拓扑文件格式

拓扑配置文件第一行：

```
node_num  gpus_per_server  nvswitch_num  switch_num  link_num  gpu_type
```

后续是 NVSwitch/交换机 ID 列表，然后是逐条链路定义：

```
src  dst  bandwidth  latency  error_rate
```

### 三种预定义拓扑模板

#### 1. AlibabaHPN (阿里巴巴高性能网络)

```
默认: 15360 GPU, 200Gbps, Rail-Optimized + Dual-ToR
```

**结构: 3 层** — GPU → NVSwitch (机内) → ASW (接入交换机) → PSW (汇聚交换机)

```
           PSW × 120 (汇聚层)
          ╱  │  │  │  ╲
    ASW × 240 (接入层, Dual-ToR 双上联)
     │  │         │  │
   NVSwitch      NVSwitch     ← 机内 NVLink (2880Gbps)
   ┌┴┬┴┐        ┌┴┬┴┐
   GPU×8        GPU×8         ← 每服务器 8 GPU
```

- **Rail-Optimized**: 同一 GPU 槽位共享同一 ASW (减少跨 ASW 流量)
- **Dual-ToR**: 每 GPU 连两个 ASW (冗余 + 带宽翻倍)
- **Dual-Plane**: 可选双平面 (ASW 分两组各连不同 PSW 集合)
- 每 ASW 下挂 128 NIC，ASW-PSW 间 400Gbps

#### 2. Spectrum-X (NVIDIA)

```
默认: 4096 GPU, 400Gbps, Rail-Optimized + Single-ToR
```

- 和 AlibabaHPN 类似的 Rail-Optimized 结构
- 单 ToR (每 GPU 只连一个 ASW)
- 默认更高带宽 (400Gbps)
- 每 ASW 下挂 64 NIC

#### 3. DCN+ (传统数据中心网络)

```
默认: 512 GPU, 400Gbps, Non-Rail-Optimized
```

- **非 Rail-Optimized**: GPU 不按槽位聚合，类似传统胖树
- 可选 Dual-ToR (此时带宽降为 200Gbps, NIC/ASW=128)
- 8 ASW + 8 PSW，规模较小

### 共同的网络层级

所有拓扑都包含 **3 层节点**：

| 层级     | 节点类型          | NS-3 实现                   | 连接                          |
| :------- | :---------------- | :-------------------------- | :---------------------------- |
| **机内** | NVSwitch (type=2) | `NVSwitchNode`              | NVLink 2880Gbps, 延迟 0.025μs |
| **接入** | ASW (type=1)      | `SwitchNode` (ECMP+PFC+ECN) | 200/400Gbps, 延迟 0.5μs       |
| **汇聚** | PSW (type=1)      | `SwitchNode`                | 400Gbps                       |
| **计算** | GPU (type=0)      | `Node` + `RdmaHw`           | —                             |

### 关键特性

- **Rail-Optimized** 是核心设计思想：同一服务器内的第 k 个 GPU 都连到同一个 ASW_k，使 AllReduce 等集合通信在单一 ASW 内闭合
- 所有拓扑都支持 **NVSwitch** 用于机内 GPU 互联
- ASW 层使用 **ECMP 哈希路由** + **PFC 无损** + **ECN/DCQCN 拥塞控制**
- 拓扑完全参数化，通过命令行参数可自由调整 GPU 数量、带宽、交换机数量等

## 7.nccl生成逻辑拓扑

### 整体架构：三层转换

```
┌──────────────────────────────────────────────────────────┐
│  第1层: 通信组划分 (逻辑分组)                              │
│  GPU rank → TP/DP/PP/EP/DP_EP 组                         │
├──────────────────────────────────────────────────────────┤
│  第2层: 逻辑拓扑生成 (集合通信算法)                        │
│  Ring / Tree / NVLS / NVLS_Tree 通道                     │
├──────────────────────────────────────────────────────────┤
│  第3层: 流分解 → SingleFlow (src, dst, size)              │
│  逻辑流 → 物理 GPU rank 对                                │
├──────────────────────────────────────────────────────────┤
│  第4层: NS-3 物理网络传输                                  │
│  rank → IP 地址 → RDMA QP → 拓扑文件中的物理链路           │
└──────────────────────────────────────────────────────────┘
```

------

### 第1层：通信组划分（逻辑分组）

`MockNcclGroup` 构造函数根据并行策略参数，将所有 GPU rank 划分到不同的通信组：

```
输入: ngpus=64, gpus_per_node=8, TP=8, DP=8, PP=1, EP=4, DP_EP=2

TP 组 (张量并行, 机内):
  Group 0: [0,1,2,3,4,5,6,7]     ← 一台机器的 8 GPU
  Group 1: [8,9,10,11,12,13,14,15]
  ...

DP 组 (数据并行, 跨机):
  Group 8: [0,8,16,24,32,40,48,56]  ← 每台机器的第0号GPU
  Group 9: [1,9,17,25,33,41,49,57]  ← 每台机器的第1号GPU
  ...

EP 组 (专家并行):
  Group 16: [0,8,16,24]  ← 4个TP组中对应位置的rank
  ...
```

**关键数据结构：**

- `GroupIndex[{rank, type}] → group_id` — 查询某 rank 属于哪个组
- `AllGroups[group_id] → GroupInfo{Ranks, nNodes, NVSwitchs}` — 组内所有 rank + 跨几个节点 + NVSwitch ID

每个 GroupInfo 还存储了该组涉及的 **NVSwitch** 节点 ID（来自物理拓扑文件），这是逻辑与物理的第一个连接点。

------

### 第2层：逻辑拓扑生成（通道构建）

根据 `get_algo_proto_info()` 选择的算法，生成不同的逻辑通信拓扑：

#### 算法选择逻辑 (`get_algo_proto_info`, line 2028)

```
AllReduce:
  TP 组 + A100/A800       → Ring
  TP 组 + H100/H800 + ≥8rank + NVLS_ENABLE → NVLS
  其他                     → Ring

AllGather / ReduceScatter / AllToAll → 全部用 Ring
```

#### Ring 通道生成 (`genringchannels`)

```
输入: GroupInfo{Ranks=[0,1,2,3,4,5,6,7], nNodes=1, nlocalRanks=8}

Step 1: gen_local_ring() — 生成本地环排列
  Channel 0: [0,1,2,3,4,5,6,7]   ← 从 rank 0 开始轮转
  Channel 1: [1,2,3,4,5,6,7,0]   ← 从 rank 1 开始轮转
  ...共 nlocalRanks 个 channel

Step 2: generateringchannels() — 展开为多节点环
  对每个 channel, 按 delta 偏移复制到各节点:
  环中每个 rank 记录: [prev, next, node_recv_rank, node_send_rank]

结果 (单节点例子, Channel 0):
  ringchannels[0][0] = {7, 1, 0, 7}  ← rank 0 的前驱=7, 后继=1
  ringchannels[0][1] = {0, 2, 0, 7}
  ...
  ringchannels[0][7] = {6, 0, 0, 7}  ← 环闭合: 7→0
```

跨节点时（如 DP 组），ring 通过 `delta` 跨越节点边界：

```
DP 组: Ranks=[0,8,16,24], nNodes=4, nlocalRanks=1
  Channel 0: 0 → 8 → 16 → 24 → 0 (环)
  delta = 8 (节点间偏移)
```

#### Tree 通道生成 (`gettreechannels`)

```
Step 1: genInterDouBinTree() — 生成双二叉树（节点间）
  4 个节点 [0,1,2,3]:
  
  Tree 1:          Tree 2 (shift+1):
       1                2
      / \              / \
     0   2            1   3
         |            |
         3            0

Step 2: ConnInterIntraTree() — 将节点内 rank 串成链
  节点 1 内: rank 8→9→10→...→15 (链式)
  节点内根 rank 连接到节点间树的上/下游

结果: ncclTree 结构
  treechannel[rank=8]  = {up=-1, down=[0,10]}  ← 根节点
  treechannel[rank=0]  = {up=8,  down=[]}      ← 左子树
  treechannel[rank=10] = {up=8,  down=[16]}    ← 右子树→下一节点
```

#### NVLS 通道生成 (`get_nvls_channels`)

```
单节点, Ranks=[0,1,...,7], NVSwitch=64:

  nvlschannel[0][rank_0] = Tree{up=NVSwitch_64, down=[]}
  nvlschannel[0][rank_1] = Tree{up=NVSwitch_64, down=[]}
  ...
  nvlschannel[0][8]      = Tree{rank=NVSwitch_64, up=-1, down=[0,1,...,7]}

  即: 所有 GPU → NVSwitch → 所有 GPU (星形拓扑)
```

#### NVLS_Tree 通道生成 (`get_nvls_tree_channels`)

```
跨节点 NVLS, 结合 Tree + NVLink:

机内 (gen_nvls_tree_intra_channels):
  选定 rank → NVSwitch → 其他 rank  (NVLink 星形)

机间 (gen_nvls_tree_inter_channels):
  用双二叉树连接各节点的 "代表 rank"

结果: ncclChannelNode 树
  GPU_0 ←NVLink→ NVSwitch_0 ←NVLink→ GPU_1..7
                     ↑ (机间)
  GPU_8 ←NVLink→ NVSwitch_1 ←NVLink→ GPU_9..15
```

------

### 第3层：流分解（逻辑拓扑 → SingleFlow）

通道建好后，根据集合操作生成具体的 **SingleFlow** 数据流：

#### Ring AllReduce 流分解 (`genAllReduceRingFlowModels`)

```
Ring: 0→1→2→3→0, data_size=1MB, nranks=4
chunksize = 1MB/4/nChannels, chunk_count = 2*(4-1) = 6

ReduceScatter 阶段 (chunk 0~2):
  Flow 0: src=0, dst=1, size=chunksize, prev={3}, child={Flow 4}
  Flow 1: src=1, dst=2, size=chunksize, prev={0}, parent={Flow 0}
  Flow 2: src=2, dst=3, size=chunksize, prev={1}, parent={Flow 1}

AllGather 阶段 (chunk 3~5):
  Flow 3: src=0, dst=1, size=chunksize, prev={3}, parent={Flow 2}
  Flow 4: src=1, dst=2, ...
  ...

每个 flow 包含 DAG 依赖:
  parent_flow_id: 必须等这些 flow 完成才能开始
  child_flow_id:  本 flow 完成后触发这些 flow
```

#### NVLS AllReduce 流分解 (`genAllreduceNVLSFlowModels`)

```
单节点, 8 GPU, NVSwitch=64, chunk_count=4:

每个 chunk:
  Reduce 阶段 (上行):
    Flow 0: src=GPU_0, dst=NVSwitch_64, conn_type="NVLS"
    Flow 1: src=GPU_1, dst=NVSwitch_64, conn_type="NVLS"
    ...
    Flow 7: src=GPU_7, dst=NVSwitch_64, conn_type="NVLS"

  Broadcast 阶段 (下行, 依赖上行全部完成):
    Flow 8:  src=NVSwitch_64, dst=GPU_0, parent=[0,1,...,7]
    Flow 9:  src=NVSwitch_64, dst=GPU_1, parent=[0,1,...,7]
    ...
    Flow 15: src=NVSwitch_64, dst=GPU_7, parent=[0,1,...,7]
```

#### AllToAll 流分解 (`genAlltoAllFlowModels`)

```
4 rank, 最简单的全对全:
  Flow 0: src=0, dst=1, prev=[1,2,3]  (等所有peer准备好)
  Flow 1: src=0, dst=2
  Flow 2: src=0, dst=3
  Flow 3: src=1, dst=0
  ...共 4×3=12 个 flow
```

------

### 第4层：SingleFlow → NS-3 物理网络

这是逻辑到物理的最后一步，发生在 `entry.h` 的 `SendFlow()` 函数：

```
SingleFlow{src=GPU_0, dst=GPU_8, flow_size=65536}
    │
    ▼
astra-sim 的 sim_send(rank=0, dst=8, count=65536)
    │
    ▼
SendFlow(src=0, dst=8, maxPacketCount=65536)
    │
    ├── serverAddress[0] → IP 10.0.0.1  (由 node_id_to_ip 映射)
    ├── serverAddress[8] → IP 10.0.0.9
    │
    ▼
RdmaClientHelper(pg=3, srcIP, dstIP, port, dport, packetCount, BDP, RTT)
    │
    ▼
NS-3 物理模拟:
  srcIP 10.0.0.1 在拓扑文件中对应 node 0 (GPU 节点)
  dstIP 10.0.0.9 在拓扑文件中对应 node 8
  
  物理路径 (由拓扑文件 + CalculateRoutes 确定):
    Node 0 → QbbNetDevice → QbbChannel → SwitchNode(ASW_0) 
    → SwitchNode(PSW_x) → SwitchNode(ASW_1) → Node 8
```

**拓扑文件的 rank→物理节点 映射:**

```
拓扑文件第一行: 72 8 8 16 200 H100
                │  │ │  │   │   │
         总节点数 │ │ 交换机│  链路数 GPU型号
         GPU/server NVSwitch数

节点编号:
  0~47:  GPU 节点 (type=0), 每个对应一个 rank
  48~55: NVSwitch 节点 (type=2), 机内 NVLink
  56~71: 交换机节点 (type=1), ASW/PSW

链路定义:
  0 48 2880Gbps 0.000025ms 0    ← GPU 0 到 NVSwitch 48 (NVLink)
  0 56 400Gbps  0.0005ms   0    ← GPU 0 到 ASW 56 (以太网)
  56 64 400Gbps  0.0005ms  0    ← ASW 56 到 PSW 64
```

### 总结图

```
┌─────────────────────────────────────────────────────────┐
│           MockNcclGroup 构造函数                         │
│  输入: TP=8, DP=8, gpus_per_node=8                      │
│  输出: AllGroups, GroupIndex                             │
│        TP Group 0: [0,1,...,7] (nNodes=1)               │
│        DP Group 8: [0,8,16,...,56] (nNodes=8)           │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   genringchannels  gettreechannels  get_nvls_channels
   (Ring 环)        (双二叉树)       (NVSwitch 星形)
        │               │               │
        ▼               ▼               ▼
   ringchannels     treechannels     nvlschannels
   rank→[prev,next] rank→{up,down}   rank→{up=NVSwitch}
        │               │               │
        └───────┬───────┘───────┬───────┘
                ▼               ▼
         get_algo_proto_info()
         选择: Ring/NVLS/Tree
                │
                ▼
        genXxxFlowModels()
        遍历通道, 按 chunk 分割
                │
                ▼
        SingleFlow{flow_id, src, dst, flow_size,
                   parent_flow_id, child_flow_id,
                   conn_type="RING"/"NVLS"/"TREE"}
                │
                ▼
        astra-sim: sim_send(src=rank, dst=rank)
                │
                ▼
        entry.h: SendFlow()
          rank → serverAddress[rank] (IP)
          RdmaClientHelper(srcIP, dstIP, packetCount)
                │
                ▼
        NS-3 物理网络模拟
          IP → 拓扑文件中的 Node
          Node → QbbNetDevice → SwitchNode → ...
          ECMP 路由 + PFC + ECN + DCQCN
```

**核心思路**: MockNccl 只关心 **"rank 之间的逻辑关系"**（谁给谁发数据、发多少、顺序依赖），不关心物理网络怎么连接。`rank` 编号既是逻辑标识也是物理拓扑文件中的节点编号（node ID），这个 **编号一致性** 就是逻辑到物理的桥梁——NS-3 通过 `serverAddress[rank]` 将 rank 映射为 IP 地址，再由拓扑文件定义的物理链路完成实际传输。

## 8.SimAI 如何模拟网络延迟和计算时间

整个 SimAI 是一个 **双时钟域** 的离散事件仿真系统：astra-sim 驱动计算+调度（Tick 时钟），NS-3 驱动网络传输（纳秒时钟），两者通过回调函数同步。

------

### 一、网络延迟模拟（NS-3 侧）

NS-3 模拟的端到端延迟由 **4 个分量** 自然叠加：

```
端到端延迟 = 注入延迟 + Σ(序列化延迟 + 传播延迟 + 排队延迟) per hop
```

#### 1. 注入延迟 (Host Processing Latency)

```cpp
// entry.h, SendFlow()
int send_lat = 6000;  // 默认 6μs (纳秒单位)
const char* send_lat_env = std::getenv("AS_SEND_LAT");  // 可配置
// ...
appCon.Start(Time(send_lat * 1000));  // 转换为皮秒,延迟注入
```

模拟 GPU 发起 RDMA Write 到 NIC 实际开始发送的延迟（包括 PCIe 延迟、驱动开销等），默认 **6μs**，可通过环境变量 `AS_SEND_LAT` 调整。

#### 2. 序列化延迟 (Transmission/Serialization Delay)

```cpp
// qbb-net-device.cc, TransmitStart()
Time txTime = m_bps.CalculateBytesTxTime(p->GetSize());
//          = packet_size_bytes × 8 / link_bandwidth_bps

// 例: 1KB 包在 400Gbps 链路上:
//     1024 × 8 / 400e9 = 20.48 ns
```

每经过一条链路，包都需要被"序列化"到线路上，时间 = 包大小 / 链路带宽。

#### 3. 传播延迟 (Propagation Delay)

```cpp
// qbb-channel.cc, TransmitStart()
Simulator::ScheduleWithContext(
    m_link[wire].m_dst->GetNode()->GetId(),
    txTime + m_delay,    // ← 序列化 + 传播
    &QbbNetDevice::Receive,
    m_link[wire].m_dst, p);
```

`m_delay` 来自拓扑文件中每条链路的 `link_delay` 字段：

```
// 拓扑文件链路定义:
// src dst bandwidth  delay      error_rate
   0   56  400Gbps    0.0005ms   0      ← NIC 到 ASW: 500ns
   48  0   2880Gbps   0.000025ms 0      ← NVLink: 25ns
```

NS-3 的调度器在 `txTime + m_delay` 后触发接收事件，**精确到纳秒级**。

#### 4. 排队延迟 (Queueing Delay)

这是 NS-3 模拟中**最关键也最复杂**的延迟分量，由交换机缓冲区自然产生：

```cpp
// switch-node.cc, SendToDev()
// 入口准入检查
if (m_mmu->CheckIngressAdmission(inDev, outDev, qIndex, p->GetSize())
    && m_mmu->CheckEgressAdmission(inDev, outDev, qIndex, p->GetSize())) {
    m_mmu->UpdateIngressAdmission(inDev, outDev, qIndex, p->GetSize());
    m_mmu->UpdateEgressAdmission(inDev, outDev, qIndex, p->GetSize());
    // 包进入出口队列排队
    m_bytes[inDev][outDev][qIndex] += p->GetSize();
    // ...
}
```

**排队延迟不是显式设置的**，而是通过事件驱动仿真自然涌现：

- 多个入口端口同时向同一出口端口发包 → 包在出口队列中排队
- 前面的包还在序列化时 → 后面的包必须等待
- PFC 触发时 → 上游被暂停，额外延迟
- ECN/DCQCN 降速时 → 发送间隔增大

#### 5. RTT 预计算（用于 DCQCN 窗口初始化）

```cpp
// common.h, SetupNetwork()
// BFS 遍历路由路径,累加每跳延迟
uint64_t delay = pairDelay[src][dst];     // 单向传播延迟总和
uint64_t txDelay = pairTxDelay[src][dst]; // 路径上序列化延迟总和
uint64_t rtt = delay * 2 + txDelay;       // RTT = 2×传播 + 序列化
uint64_t bdp = rtt * bw / 1e9 / 8;       // BDP = RTT × 带宽

// 传给 RdmaClientHelper 作为初始窗口
RdmaClientHelper clientHelper(pg, srcIP, dstIP, port, dport,
    packetCount,
    has_win ? pairBdp[src][dst] : 0,  // 初始 BDP 窗口
    pairRtt[src][dst],                 // RTT
    ...);
```

**注意**: 这个预计算的 RTT 仅用于初始化 DCQCN 的 BDP 窗口，实际传输中的延迟是由 NS-3 事件驱动仿真**自然产生**的。

#### 完整数据路径的延迟示意

```
GPU 0 发送 1KB 到 GPU 8 (跨节点, Rail-Optimized):

时刻 0:        astra-sim 调用 sim_send()
                    │
时刻 +6μs:     注入延迟 (send_lat)
                    │
                ▼ NIC 开始发送
时刻 +6μs+20ns: 序列化完成 (1KB/400Gbps)
                    │ 传播 500ns
时刻 +6.52μs:  到达 ASW (Leaf Switch)
                    │ 排队延迟 (0~数ms, 取决于拥塞)
                    │ 序列化 20ns
                    │ 传播 500ns
时刻 +7.04μs:  到达 PSW (Spine Switch) [无拥塞]
                    │ 排队延迟
                    │ 序列化 20ns  
                    │ 传播 500ns
时刻 +7.56μs:  到达目标 ASW
                    │ 排队延迟
                    │ 序列化 20ns
                    │ 传播 500ns
时刻 +8.08μs:  到达 GPU 8 的 NIC
                    │
                ▼ 触发 qp_finish → notify_receiver
```

------

### 二、Kernel 计算时间模拟（astra-sim 侧）

#### 1. 工作负载文件格式

AICB 生成的 workload 文件定义了每层的计算时间和通信需求：

```
# 每行格式:
# layer_id  依赖  fwd_compute  fwd_comm_type  fwd_comm_size  
#                  ig_compute   ig_comm_type   ig_comm_size
#                  wg_compute   wg_comm_type   wg_comm_size   wg_update_time

attention_column  -1  1750840  ALLGATHER  50331648  875420  REDUCESCATTER  0  875420  NONE  0  100
                      ───────                       ──────                    ──────
                      前向计算                       反向输入梯度              反向权重梯度
                      1,750,840 Tick                 875,420 Tick              875,420 Tick
```

**关键字段：**

- `fwd_compute_time` = **1750840** Tick — 前向传播的 kernel 计算时间
- `ig_compute_time` = **875420** Tick — 输入梯度计算时间
- `wg_compute_time` = **875420** Tick — 权重梯度计算时间
- `wg_update_time` = **100** Tick — 参数更新时间
- 通信类型和大小（ALLGATHER 50MB 等）

#### 2. 计算时间的来源（AICB 的 AIOB 模块）

```
真实 GPU 上运行 AIOB (AI Operation Benchmark)
         │
         ▼
    测量每个 kernel 的执行时间
    (matmul, attention, layernorm, ...)
         │
         ▼
    汇总为每层 fwd/ig/wg 的 Tick 数
         │
         ▼
    写入 workload.txt
```

**Tick 单位**: 通常对应 GPU 时钟周期。AIOB 在真实 GPU 上跑 microbenchmark 得到精确的 kernel 执行时间。

#### 3. 计算时间在仿真中的调度

```cpp
// Workload.cc, 加载时乘以 compute_scale
Layer* l = new Layer(...,
    fp_compute_time * generator->compute_scale,  // 前向计算
    ...,
    ig_compute_time * generator->compute_scale,  // 反向输入梯度
    ...,
    wg_compute_time * generator->compute_scale,  // 反向权重梯度
    ...);
```

**compute_scale** 是全局缩放因子，可用于模拟不同 GPU 速度（如用 A100 的 benchmark 数据模拟 H100 的计算速度）。

#### 4. 计算-通信交替的状态机

```
Layer.call(EventType event) 状态转换:

┌──────────────┐   fwd_compute_time    ┌──────────────┐
│ Forward Pass │ ────────────────────→ │ Fwd Comm     │
│ Compute      │   register_event()    │ (AllGather)  │
└──────────────┘                       └──────┬───────┘
                                              │ 通信完成回调
                                              ▼
                                  ┌───────────────────┐
                                  │ Fwd_Comm_Finished  │
                                  │ + fwd_update_time  │
                                  └─────────┬─────────┘
                                            │
                                            ▼
┌──────────────┐  ig_compute_time   ┌──────────────┐
│ Input Grad   │ ←────────────────  │ Next Layer   │
│ Compute      │                    │ or Backward  │
└──────┬───────┘                    └──────────────┘
       │ register_event()
       ▼
┌──────────────┐
│ IG Comm      │   通信完成 + ig_update_time
│ (ReduceScat) │ ──→ Weight Grad Compute ──→ WG Comm ──→ ...
└──────────────┘
// Layer.cc
void Layer::call(EventType event, CallData* mdata) {
    if (event == EventType::Fwd_Comm_Finished) {
        // 通信完成后,调度 update 延迟
        generator->register_event(this,
            EventType::Fwd_Comm_Finished_After_Delay,
            mdata, fwd_update_time);  // ← 这个延迟模拟参数更新
    }
    // ...
}
```

#### 5. 计算与通信的重叠 (Pipeline)

```cpp
// Sys.hh 中的流调度器
std::list<BaseStream*> ready_list;         // 就绪的流
int active_Streams[kMaxStreamsPerSch];      // 每队列活跃流数
```

astra-sim 支持多流并行：一层的通信和下一层的计算可以重叠执行。这就是为什么 **compute_time 和 comm_time 不是简单相加**，而是通过事件驱动调度自然重叠。

#### 6. 其他延迟常量

```cpp
// Sys.cc
processing_latency = 10;                  // 处理延迟 (Tick)
communication_delay = 10 * injection_scale; // 端点延迟 (可由配置文件覆盖)
local_reduction_delay = 1;                 // 本地归约延迟 (Tick)
```

------

### 三、两个时钟域的同步

```
astra-sim (Tick 域)              NS-3 (纳秒域)
    │                                │
    │  sim_send(rank, dst, size)     │
    ├───────────────────────────────→│
    │                                │ 创建 RDMA QP
    │                                │ 包级模拟传输...
    │                                │ (排队/ECN/PFC/CC)
    │                                │
    │  qp_finish() 回调              │
    │←───────────────────────────────┤
    │                                │
    │  notify_receiver() 回调        │
    │←───────────────────────────────┤
    │                                │
    │  通信完成 → 调度下一个          │
    │  compute_time Tick 后触发       │
    │  下一次通信                     │
```

**关键**: NS-3 中的实际传输时间（包含所有排队/拥塞/PFC 延迟）通过回调函数返回给 astra-sim。astra-sim 不需要知道具体的网络延迟是多少——它只需要等 NS-3 告诉它"传输完成了"。这种设计使得：

- **计算时间** = workload 文件指定 (AICB/AIOB 在真实 GPU 上测量)
- **通信时间** = NS-3 逐包仿真自然产生 (传播+序列化+排队+拥塞控制)
- **总训练时间** = 通过事件驱动调度器将两者自然交织

## 9.完整运行 SimAI 的指令

### Step 0: 克隆 & 初始化子模块

```bash
git clone https://github.com/aliyun/SimAI.git
cd SimAI
git submodule update --init --recursive
git submodule update --remote
```

### Step 1: 编译

```bash
# 编译分析模式 (快速, 不需要 NS-3)
./scripts/build.sh -c analytical

# 编译仿真模式 (完整 NS-3 包级仿真)
# 注意: NGC 镜像中需先移除 ninja
# apt remove ninja-build && pip uninstall ninja
./scripts/build.sh -c ns3
```

### Step 2: 生成 Workload 文件

```bash
# 方式 A: 无 GPU — 用自带的示例文件 (计算时间为分析值)
# 不需要生成, 直接用 example/workload_analytical.txt

# 方式 B: 无 GPU — 用 AICB 生成 (计算时间=1, 只测通信)
cd aicb
python3 -m workload_generator.SimAI_training_workload_generator \
    --frame Megatron \
    --model_name "GPT-7B" \
    --num_layers 32 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --world_size 32 \
    --tensor_model_parallel_size 8 \
    --pipeline_model_parallel 1 \
    --global_batch 24
cd ..
或者
bash scripts/megatron_workload_with_aiob.sh \
    -m 7 \
    --world_size 32 \
    --tp 8 \
    --pp 1 \
    --global_batch 24

# 方式 C: 有 GPU — 用 AICB + AIOB (真实 kernel 计时)
cd aicb
python3 -m workload_generator.SimAI_training_workload_generator \
    --frame Megatron -m 7 \
    --world_size 32 \
    --tensor_model_parallel_size 8 \
    --aiob_enable
cd ..

# 方式 D: 无 GPU — 复用别人的测量文件
cd aicb
python3 -m workload_generator.SimAI_training_workload_generator \
    --frame Megatron -m 7 \
    --world_size 32 \
    --tensor_model_parallel_size 8 \
    --aiob_enable \
    --comp_filepath workload/aiob_inputs/Example.txt
cd ..
```

##### 参数

###### 一、基础参数（get_params）

| 参数                           | 类型    | 默认值     | 含义                                                         |
| :----------------------------- | :------ | :--------- | :----------------------------------------------------------- |
| `--frame`                      | choices | `Megatron` | 训练框架：`Megatron`、`DeepSpeed`、`DeepSeek`、`collective_test` |
| `--gpu_type`                   | str     | None       | GPU 型号（如 A100、H800），影响算法选择                      |
| `--world_size`                 | int     | 1          | **总 GPU 数量**                                              |
| `--tensor_model_parallel_size` | int     | 1          | **TP 并行度**（张量模型并行），通常 = 单机 GPU 数，通信走 NVLink |
| `--pipeline_model_parallel`    | int     | 1          | **PP 并行度**（流水线并行），模型按层切分到多组 GPU          |
| `--context-parallel-size`      | int     | 1          | **CP 并行度**（上下文并行），将长序列切分到多个 GPU 并行处理 |
| `--pp_rank`                    | int     | -1         | PP 切分位置（encoder/decoder 在哪一层切开），-1 表示自动     |
| `--global_batch`               | int     | 4          | 全局 batch size（一次迭代所有 GPU 总共处理的样本数）         |
| `--micro_batch`                | int     | 1          | 微 batch size（单个 GPU 单次前向/反向处理的样本数）          |
| `--epoch_num`                  | int     | 1          | 训练迭代轮数（生成几轮 workload）                            |
| `--computation_enable`         | flag    | False      | 是否启用计算时间模拟（不启用则 compute time = 0）            |
| `--dtype`                      | str     | `bfloat16` | 数据类型，影响通信量大小（bf16 = 2字节/参数，fp32 = 4字节）  |
| `--ffn_hidden_size`            | int     | None       | FFN 隐藏层大小，默认 = 4 × hidden_size；SwiGLU 时通常设为 hidden_size × 8/3 |
| `--enable_visual`              | flag    | False      | 启用可视化输出                                               |
| `--workload_only`              | flag    | False      | 仅生成 workload 文件，不运行训练                             |

> **DP（数据并行）不需要手动设，自动计算**：`DP = world_size / (TP × PP × EP)`

###### 二、模型结构参数（get_model_params）

| 参数                        | 类型 | 默认值 | 含义                                                         |
| :-------------------------- | :--- | :----- | :----------------------------------------------------------- |
| `--model_name`              | str  | None   | 模型名称（如 GPT-7B），用于输出文件命名                      |
| `--hidden_size`             | int  | 1024   | Transformer 隐藏层维度（决定每层参数量和通信量）             |
| `--num_layers`              | int  | 24     | Transformer 层数（模型深度，直接影响 workload 行数）         |
| `--seq_length`              | int  | 2048   | 最大序列长度（影响 Attention 计算量和激活值大小）            |
| `--num_attention_heads`     | int  | None   | 注意力头数（默认 = hidden_size / 128）                       |
| `--vocab_size`              | int  | 32000  | 词表大小（影响 Embedding 层的通信量）                        |
| `--max_position_embeddings` | int  | 4096   | 最大位置编码长度                                             |
| `--add_bias_linear`         | flag | False  | 线性层是否加 bias（加了会略增通信量）                        |
| `--use_flash_attn`          | flag | False  | 使用 Flash Attention（减少激活值内存，改变计算模式）         |
| `--swiglu`                  | flag | False  | 使用 SwiGLU 激活函数（替代 GELU，FFN 结构变为门控式，参数量增加） |

常见模型参数对照表

| 模型       | num_layers | hidden_size | num_attention_heads |
| :--------- | :--------- | :---------- | :------------------ |
| GPT-7B     | 32         | 4096        | 32                  |
| GPT-13B    | 40         | 5120        | 40                  |
| GPT-22B    | 48         | 6144        | 48                  |
| GPT-175B   | 96         | 12288       | 96                  |
| LLaMA-405B | 126        | 16384       | 128                 |

###### 三、MOE 专家混合参数（get_moe_params）

| 参数                           | 类型 | 默认值 | 含义                                                         |
| :----------------------------- | :--- | :----- | :----------------------------------------------------------- |
| `--moe_enable`                 | flag | False  | **开启 MOE 模式**（不开则所有 MOE 参数无效）                 |
| `--expert_model_parallel_size` | int  | 1      | **EP 并行度**（专家并行），专家分布到几组 GPU 上             |
| `--num_experts`                | int  | 1      | 专家总数（如 Mixtral=8，DeepSeek=256）                       |
| `--moe_router_topk`            | int  | 1      | 每个 token 激活几个专家（topk=2 表示每个 token 选 2 个专家） |
| `--moe_grouped_gemm`           | flag | False  | 对同一 rank 上的多个专家使用 Grouped GEMM（合并矩阵乘法，提高效率） |
| `--activation_func`            | str  | None   | MLP 激活函数类型                                             |

> **MOE 通信特点**：EP 组之间需要 All-to-All 通信（token 发送到对应专家所在的 GPU）

###### 四、DeepSeek 专用参数（get_deepseek_params）

| 参数                | 类型 | 默认值 | 含义                                                         |
| :------------------ | :--- | :----- | :----------------------------------------------------------- |
| `--n_dense_layers`  | int  | 3      | 前几层为 Dense 层（不用 MOE），之后的层才是 MOE 层           |
| `--n_shared_expert` | int  | 2      | 共享专家数量（每个 token 都会经过的专家，不参与路由）        |
| `--qk_rope_dim`     | int  | 64     | QK 带旋转位置编码的维度（RoPE 维度）                         |
| `--qk_nope_dim`     | int  | 128    | QK 不带位置编码的维度                                        |
| `--q_lora_rank`     | int  | 1536   | Q 矩阵的 LoRA 降维大小（MLA 注意力机制的低秩压缩）           |
| `--kv_lora_rank`    | int  | 512    | KV 矩阵的 LoRA 降维大小（MLA 的 KV 缓存压缩，大幅减少 KV cache） |
| `--v_head_dim`      | int  | 128    | Value 投影的每头维度                                         |

> DeepSeek 使用 **MLA（Multi-head Latent Attention）**，通过低秩投影压缩 KV cache，这些参数控制压缩率。

###### 五、Megatron 框架专用参数（get_megatron_params）

| 参数                             | 类型 | 默认值 | 含义                                                         |
| :------------------------------- | :--- | :----- | :----------------------------------------------------------- |
| `--enable_sequence_parallel`     | flag | False  | 开启序列并行（在 TP 组内沿序列维度切分 LayerNorm/Dropout，减少激活值内存） |
| `--use-distributed-optimizer`    | flag | False  | 分布式优化器（将优化器状态按 DP 组分片，减少内存，增加一次 All-Gather） |
| `--make_vocab_size_divisible_by` | int  | 128    | 将词表大小补齐到此值的倍数（方便张量并行切分）               |
| `--overlap_grad_reduce`          | flag | False  | 梯度通信与计算重叠（边算反向传播边做 AllReduce）             |

###### 六、DeepSpeed 框架专用参数（get_ds_params）

| 参数                            | 类型 | 默认值 | 含义                                                      |
| :------------------------------ | :--- | :----- | :-------------------------------------------------------- |
| `--stage`                       | int  | 3      | ZeRO 优化阶段：1=优化器状态分片，2=+梯度分片，3=+参数分片 |
| `--amp_enabled`                 | flag | False  | 自动混合精度训练                                          |
| `--reduce_bucket_size`          | int  | 5×10⁸  | 梯度 AllReduce 的桶大小（字节），多个小梯度合并后一起通信 |
| `--allgather_bucket_size`       | int  | 5×10⁸  | AllGather 桶大小（ZeRO Stage 1/2 使用）                   |
| `--contiguous_gradients`        | flag | False  | 梯度连续存储（Stage 1/2，减少内存碎片）                   |
| `--param_persistence_threshold` | int  | 10⁵    | 参数持久化阈值（Stage 3，小于此值的参数常驻 GPU 不分片）  |
| `--model_persistence_threshold` | int  | MAX    | 模型持久化阈值（Stage 3）                                 |
| `--max_live_parameters`         | int  | 10⁹    | 同时驻留 GPU 的最大参数量（Stage 3）                      |
| `--prefetch_bucket_size`        | int  | 10⁹    | 参数预取桶大小（Stage 3，提前加载下一层参数）             |

------

###### 七、AIOB 计算时间参数（get_aiob_params）

| 参数                      | 类型 | 默认值 | 含义                                                         |
| :------------------------ | :--- | :----- | :----------------------------------------------------------- |
| `--aiob_enable`           | flag | False  | 启用 AIOB 真实计算 profiling（需要 torch + GPU）             |
| `--comp_filepath`         | str  | None   | 预计算的 kernel 时间文件路径（如 `Example.txt`），避免需要真实 GPU |
| `--gated_linear_unit`     | bool | False  | 使用门控线性单元                                             |
| `--bias_gelu_fusion`      | flag | False  | Bias + GELU 融合算子                                         |
| `--openai_gelu`           | flag | False  | 使用 OpenAI 版 GELU 实现                                     |
| `--onnx_safe`             | flag | False  | ONNX 导出安全模式                                            |
| `--squared_relu`          | flag | False  | 使用 Squared ReLU 激活函数                                   |
| `--recompute_activations` | flag | False  | 激活值重计算（用时间换显存，减少峰值内存）                   |

------

###### 八、通信测试参数（get_collective_test_params）

当 `--frame collective_test` 时使用：

| 参数                        | 类型 | 默认值       | 含义                                                         |
| :-------------------------- | :--- | :----------- | :----------------------------------------------------------- |
| `--begin_size`              | int  | 1048576      | 通信测试起始数据大小（字节）                                 |
| `--end_size`                | int  | 1048576      | 通信测试结束数据大小（字节）                                 |
| `--test_comm`               | str  | `all_reduce` | 测试的通信原语类型（all_reduce / all_gather / reduce_scatter / all_to_all） |
| `--iter_num`                | int  | 500          | 测试迭代次数                                                 |
| `--multi_all_reduce_enable` | int  | 0            | 多次 AllReduce 测试                                          |

------

###### 九、其他参数

| 参数                | 类型 | 默认值 | 含义                                  |
| :------------------ | :--- | :----- | :------------------------------------ |
| `--overlap_version` | flag | False  | 启用通信-计算重叠版本的 workload 生成 |

------

###### 十, 推理参数

推理 workload 生成器完整参数

| 参数                           | 类型             | 默认值              | 含义                                                         |
| :----------------------------- | :--------------- | :------------------ | :----------------------------------------------------------- |
| `model_name`                   | 位置参数（必填） | —                   | 模型名，必须包含 `DeepSeek`、`Qwen3-Moe` 或 `Qwen3-Next` 之一 |
| `config_file`                  | 位置参数（可选） | None                | JSON 配置文件路径，用于自定义模型结构参数                    |
| `--world_size`                 | int              | 1                   | 总 GPU 数量                                                  |
| `--tensor_model_parallel_size` | int              | 1                   | TP 并行度                                                    |
| `--pipeline_model_parallel`    | int              | 1                   | PP 并行度                                                    |
| `--expert_model_parallel_size` | int              | 1                   | EP 并行度                                                    |
| `--seq_length`                 | int              | 1                   | 序列长度（prefill 时影响大，decode 时通常为 1）              |
| `--micro_batch`                | int              | 1                   | 微 batch 大小                                                |
| `--phase`                      | choices          | `decode`            | 推理阶段：`decode`（逐 token 生成）或 `prefill`（处理输入序列） |
| `--moe_enable`                 | flag             | True                | 是否启用 MOE（推理默认开启）                                 |
| `--aiob_enable`                | flag             | False               | 启用 AIOB 真实计算 profiling（需要 torch + GPU）             |
| `--aiob_forward_loops`         | int              | 1                   | AIOB 前向传播循环次数                                        |
| `--moe_routing_strategy`       | str              | RoundRobin          | MOE 路由策略                                                 |
| `--result_dir`                 | str              | `results/workload/` | 输出目录                                                     |

```bash
直接跳过sh//python3 -m workload_generator.SimAI_inference_workload_generator \
    Qwen3-Moe-Mixtral mixtral_8x7b.json \ #模型名是 Qwen3-Moe-Mixtral，其中包含了 Qwen3-Moe 这个子串。
    --world_size 16 \
    --tensor_model_parallel_size 8 \
    --expert_model_parallel_size 2 \
    --phase decode
```

###### Shell 脚本 `-m` 预设模型对照

| `-m` 值       | 模型          | num_layers | hidden_size | heads | 其他                |
| :------------ | :------------ | :--------- | :---------- | :---- | :------------------ |
| `7`           | GPT-7B        | 32         | 4096        | 32    |                     |
| `13`          | GPT-13B       | 40         | 5120        | 40    |                     |
| `22`          | GPT-22B       | 48         | 6144        | 48    |                     |
| `175`         | GPT-175B      | 96         | 12288       | 96    |                     |
| `405`         | LLaMA-405B    | 126        | 16384       | 128   |                     |
| `moe`         | Mixtral 8×7B  | 32         | 4096        | 32    | 8 experts, topk=2   |
| `deepseek671` | DeepSeek-671B | 61         | 7168        | 128   | 256 experts, topk=8 |
| `deepseek236` | DeepSeek-236B | 60         | 5120        | 128   | 160 experts, topk=6 |
| `deepseek16`  | DeepSeek-16B  | 28         | 2048        | 16    | 64 experts, topk=6  |

##### 自定义模型



### Step 3: 生成网络拓扑文件

```bash
# Spectrum-X 拓扑, 32 GPU, H100, 400Gbps
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo Spectrum-X -g 32 -psn 1

# 或: AlibabaHPN 双ToR, 64 GPU
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo AlibabaHPN --dp -g 64 -asn 16 -psn 16

# 或: DCN+ 单ToR, 256 GPU
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo DCN+ -g 256 -psn 64 -bw 400Gbps

# 或: 自定义 Rail-Optimized, 32 GPU, A100, 200Gbps
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -g 32 -bw 200Gbps -gt A100 -psn 8 --ro
```

### Step 4a: 运行 SimAI-Analytical（快速分析）

```bash
./bin/SimAI_analytical \
    -w example/workload_analytical.txt \
    -g 9216 \
    -g_p_s 8 \
    -r test- \
    -busbw example/busbw.yaml
```

### Step 4b: 运行 SimAI-Simulation（完整 NS-3 仿真）

```bash
# 用示例 workload + Spectrum-X 32GPU 拓扑
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 \
./bin/SimAI_simulator \
    -t 8 \
    -w ./example/microAllReduce.txt \
    -n ./Spectrum-X_32g_8gps_400Gbps_H100 \
    -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

### 参数速查

| 参数               | 含义                         |
| :----------------- | :--------------------------- |
| `-t 8`             | 多线程加速 (8~16 线程)       |
| `-w`               | workload 文件路径            |
| `-n`               | 拓扑文件路径 (Step 3 生成的) |
| `-c`               | 网络配置文件 (拥塞控制等)    |
| `AS_SEND_LAT=3`    | 注入延迟 3μs                 |
| `AS_NVLS_ENABLE=1` | 启用 NVLS 算法               |
| `AS_PXN_ENABLE=1`  | 启用 PXN 跨节点优化          |

### 最小可运行的一键命令（用自带示例）

```bash
# 编译
./scripts/build.sh -c ns3

# 生成拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    --ro -g 32 -gt H100 -bw 400Gbps -nvbw 1360Gbps

# 运行仿真
AS_SEND_LAT=12 AS_NVLS_ENABLE=1 \
./bin/SimAI_simulator -t 8 \
    -w ./example/microAllReduce.txt \
    -n ./Rail_Opti_SingleToR_32g_8gps_400Gbps_H100 \
    -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

输出结果在 `./results/` 目录下，CSV 文件包含每层的计算时间、通信时间、端到端延迟等。
