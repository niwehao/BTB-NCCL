# astra-sim-alibabacloud/astra-sim 模块设计与使用总结

## 目录功能概述

`astra-sim` 是 SimAI 的**核心仿真系统层**，基于 ASTRA-Sim（MIT 开源项目）修改，实现了分布式 AI 训练/推理的系统级仿真。它处理工作负载调度、集合通信算法模拟、拓扑管理，并通过 Network Frontend 接口与不同的网络后端（NS-3、Analytical、PhyNet）对接。

## 目录结构

```
astra-sim/
├── system/                          # 核心系统模块
│   ├── Sys.hh/cc                    # 系统主类，调度和协调所有组件
│   ├── Common.hh                    # 公共枚举和类型定义
│   ├── AstraNetworkAPI.hh           # 网络抽象接口
│   ├── AstraComputeAPI.hh           # 计算抽象接口
│   ├── AstraMemoryAPI.hh            # 内存抽象接口
│   ├── AstraSimDataAPI.hh           # 数据 API
│   ├── AstraParamParse.hh           # 参数解析
│   ├── BaseStream.hh/cc             # 流基类
│   ├── StreamBaseline.hh/cc         # 基线流实现
│   ├── DataSet.hh/cc                # 数据集管理
│   ├── CollectivePhase.hh/cc        # 集合通信阶段
│   ├── LogGP.hh/cc                  # LogGP 网络模型
│   ├── MemBus.hh/cc                 # 内存总线模型
│   ├── MyPacket.hh/cc               # 数据包定义
│   ├── PacketBundle.hh/cc           # 数据包捆绑
│   ├── MockNcclChannel.h/cc         # 模拟 NCCL Channel
│   ├── MockNcclGroup.h/cc           # 模拟 NCCL Group
│   ├── SimAiFlowModelRdma.hh/cc     # SimAI RDMA 流模型
│   ├── PhyMultiThread.hh/cc         # 物理多线程支持
│   ├── QueueLevels.hh/cc            # 队列层级管理
│   ├── collective/                  # 集合通信算法实现
│   │   ├── Algorithm.hh/cc          # 算法基类
│   │   ├── Ring.hh/cc               # Ring AllReduce
│   │   ├── HalvingDoubling.hh/cc    # Halving-Doubling
│   │   ├── DoubleBinaryTreeAllReduce.hh/cc  # 双二叉树 AllReduce
│   │   ├── NcclTreeFlowModel.hh/cc  # NCCL Tree 流模型
│   │   └── AllToAll.hh/cc           # All-to-All 通信
│   ├── topology/                    # 拓扑管理
│   │   ├── LogicalTopology.hh       # 逻辑拓扑基类
│   │   ├── RingTopology.hh/cc       # 环形拓扑
│   │   ├── BinaryTree.hh/cc         # 二叉树拓扑
│   │   ├── DoubleBinaryTreeTopology.hh/cc  # 双二叉树
│   │   ├── Torus3D.hh/cc            # 3D Torus
│   │   ├── GeneralComplexTopology.hh/cc    # 通用复合拓扑
│   │   ├── ComputeNode.hh/cc        # 计算节点
│   │   └── Node.hh/cc               # 节点基类
│   ├── memory/
│   │   └── SimpleMemory.hh/cc       # 简单内存模型
│   ├── scheduling/
│   │   └── OfflineGreedy.hh/cc      # 离线贪心调度
│   └── fast-backend/
│       └── FastBackEnd.hh/cc        # 快速后端
├── network_frontend/                # 网络前端接口
│   ├── ns3/                         # NS-3 网络后端接口
│   │   ├── AstraSimNetwork.hh/cc    # NS-3 网络适配
│   │   ├── entry.h                  # NS-3 入口
│   │   └── common.h                 # 公共定义
│   ├── analytical/                  # 解析网络模型
│   │   ├── AnalyticalNetwork.hh/cc  # 解析网络实现
│   │   ├── AnaSim.hh/cc             # 解析仿真入口
│   │   └── AnalyticalAstra.cc       # Astra 适配
│   └── phynet/                      # 物理网络接口
│       ├── SimAiEntry.hh/cc         # PhyNet 入口
│       ├── SimAiMain.cc             # 主程序入口
│       ├── PhySimAi.hh/cc           # 物理仿真
│       └── SimAiPhyNetwork.hh/cc    # 物理网络适配
└── workload/                        # 工作负载管理
    ├── Workload.hh/cc               # 工作负载解析和执行
    ├── Layer.hh/cc                   # 模型层定义
    └── CSVWriter.hh/cc              # CSV 输出
```

## 核心架构设计

### `Sys` 类 — 系统核心

- 管理调度器（SchedulerUnit）、通信组、拓扑
- 支持并行策略：TP、DP、PP、EP、DP_EP
- 处理事件驱动的仿真循环

### 网络抽象层

- `AstraNetworkAPI` — 定义网络后端接口
- 三种后端实现：NS-3（精确包级仿真）、Analytical（解析模型）、PhyNet（物理网络）

### 集合通信算法

- Ring、HalvingDoubling、DoubleBinaryTreeAllReduce、NcclTreeFlowModel、AllToAll
- 支持 NCCL 风格的 Channel/Group 模拟

### 工作负载系统

- `Workload` 解析 AICB 生成的工作负载文件
- 支持 `ParallelismPolicy`：MicroBenchmark、Data、Transformer、TransformerFwdInBckwd、DLRM、DistributedInference 等
- `Layer` 描述模型层的计算和通信特征

## 代码规模

约 143 个 C++ 源文件/头文件。

## 依赖关系

- **上游**：接收 AICB 生成的工作负载文件
- **下游网络后端**：ns-3-alibabacloud（NS-3 仿真）或 analytical 解析模型
- **构建依赖**：CMake、C++17

## 在项目中的角色

作为 SimAI 仿真的**系统层核心**，连接工作负载输入和网络仿真后端，模拟分布式训练的调度、集合通信、内存访问等行为，输出性能统计数据。

# `astra-sim/system/` 目录文件作用详解

这是 ASTRA-sim 的**系统层(system layer)**核心,负责把工作负载(训练循环)翻译成"计算 + 通信 + 内存访问"的事件流,再调度到网络后端。我按功能模块分组介绍。

---

## 一、顶层调度(系统主控)

### `Sys.hh / Sys.cc`

**整个系统的"内核"**。每个 GPU rank 对应一个 `Sys` 实例。职责:

- 持有当前 rank 的 ID、并行配置(TP/DP/PP/EP);
- 提供 `call(EventType, CallData)` 事件分发,驱动整个 collective 生命周期;
- 维护 stream 队列、调度策略(FIFO / LIFO / HighestFirst);
- 把 collective 拆成 `CollectivePhase` 列表后丢给 stream 层;
- 与 `AstraNetworkAPI` / `AstraMemoryAPI` / `AstraComputeAPI` 三个抽象接口对接。

→ 可以理解为"每张 GPU 上的运行时调度器"。

---

## 二、对外抽象接口(可插拔后端的关键)

| 文件                         | 作用                                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `AstraNetworkAPI.hh`       | **网络后端抽象**。定义 `sim_send / sim_recv / sim_schedule` 等纯虚接口,analytical / ns3 / phynet 三种后端各自实现一份。 |
| `AstraComputeAPI.hh`       | 计算抽象,声明"这一层的前向/反向需要多长 GPU 时间",通常返回从 trace 中读到的值。                                                 |
| `AstraMemoryAPI.hh`        | 内存抽象,声明 read/write 时延,通常配合 `MemBus` 用。                                                                          |
| `AstraSimDataAPI.hh`       | 上层(Workload)和系统层之间的数据契约,定义 `LayerData`、`AstraSimDataAPI` 这种结构。                                         |
| `AstraParamParse.hh / .cc` | 解析命令行 / json 配置(`-w` workload 文件、`-n` system 配置等)。                                                            |

→ ASTRA-sim 之所以能切换 analytical/ns3/phynet,就是因为这一层是纯虚接口。

---

## 三、Stream 与集合通信生命周期

NCCL 一次集合通信被拆成:**Stream(队列里的工作单元) → Phase(算法的一个阶段) → Packet(底层 P2P)**。

| 文件                         | 作用                                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `BaseStream.hh / .cc`      | 流(Stream)的抽象基类。一条 stream = 一次 collective 的执行过程,内含 phase 链表、计数器、回调。                           |
| `StreamBaseline.hh / .cc`  | 默认的 stream 实现。负责按 phase 顺序推进、收发包、统计带宽和耗时。                                                      |
| `DataSet.hh / .cc`         | 把多条 stream 组合成一个"数据集合",用于多算法并行(比如同时 RING+TREE)的整体完成判断。                                    |
| `CollectivePhase.hh / .cc` | **集合通信的一个阶段**。一次 AllReduce 在 ring 上是 `2(N-1)` 个 phase,phase 内部记录算法对象、发送对象、依赖等。 |

整体生命周期:

```
generate_collective(All_Reduce)
   → 创建 DataSet
   → push 多条 BaseStream(每个算法一条)
   → 每条 stream 包含若干 CollectivePhase
   → phase 推进时调用 sim_send/sim_recv,封装成 MyPacket
```

---

## 四、底层数据包与调度队列

| 文件                      | 作用                                                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `MyPacket.hh / .cc`     | ASTRA-sim 内部的"逻辑包"。包含 src/dst、size、stream_id 等,送进 `AstraNetworkAPI::sim_send` 时会转成后端特定的格式。 |
| `PacketBundle.hh / .cc` | 把多个 `MyPacket` 打包成一个 bundle,用于"等若干个包都到了再触发回调"——经典 reduce_scatter 的合并语义。             |
| `QueueLevels.hh / .cc`  | **多级队列管理**。NCCL 会把不同优先级、不同算法的通信放到不同 channel/queue 上并行,这里建模这种多通道并行。      |
| `LogGP.hh / .cc`        | **LogGP 网络模型实现**(latency/overhead/gap/bandwidth 4 参数)。analytical 后端常用,把 size 转换成时延。          |

---

## 五、内存与总线模型

| 文件                | 作用                                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `MemBus.hh / .cc` | **内存总线模型**。每次集合通信在 GPU 上要经过 SM↔HBM↔NVLink 的搬运,`MemBus` 模拟这段 read/write 的开销和带宽竞争。 |

→ 如果你只关注网络,可以把它理解成"在每条 flow 进出 GPU 时叠加的一段固定/比例时延"。

---

## 六、NCCL 算法与拓扑模拟(MockNccl 系列)

| 文件                        | 作用                                                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MockNccl.h`              | NCCL 常量表(算法/协议数、带宽表 `perChMaxRingLL128Bws`、延迟表 `baseLat/hwLat`),从真实 NCCL 源码移植。                                              |
| `MockNcclGroup.h / .cc`   | **核心**。管理 TP/DP/PP/EP 通信组,生成 Ring/Tree/NVLS 通道,实现 `gen*FlowModels()` 把集合通信展开成 `SingleFlow` 列表(我上一轮回答详细讲过)。 |
| `MockNcclChannel.h / .cc` | 每个 rank 的"NCCL 通信器"`MockNcclComm`,持有该 rank 在各通道里的位置;`SingleFlow` 等数据结构定义在这里。                                            |
| `MockNcclQps.h`           | RDMA QP(Queue Pair)的轻量定义,被 RDMA 流模型使用。                                                                                                      |
| `MockNcclLog.h / .cc`     | NCCL 行为的日志输出。                                                                                                                                   |

---

## 七、SimAI 自研的 RDMA 流模型

### `SimAiFlowModelRdma.hh / .cc`

SimAI 区别于原版 ASTRA-sim 的关键之一。它不再用 LogGP 这种粗糙模型,而是把上面 `MockNccl` 生成的 `SingleFlow` 列表**直接转换成 RDMA QP 上的 send/recv 事件**,送到 ns-3-alibabacloud 后端做包级仿真。

→ 这个文件是"SimCCL 输出 → 真实包级网络仿真"的桥梁。

---

## 八、多线程支持(物理后端用)

### `PhyMultiThread.hh / .cc`

`SimAiPhyCommon.hh`、`SimSendCaller / SimRecvCaller`、`PhyMultiThread` 一起构成 **phynet 后端**:在真实多机环境上跑 SimAI,用 pthread 并行管理收发线程。analytical/ns3 模式下基本不用到。

---

## 九、子目录

| 子目录            | 作用                                                                                                                                                                                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `collective/`   | **集合通信算法实现**。每个文件一个算法: `Ring.cc`、`DoubleBinaryTreeAllReduce.cc`、`AllToAll.cc`、`HalvingDoubling.cc` 等。这些是"算法层",负责按 NCCL 拓扑生成 phase 序列(MockNccl 是"展开后的 flow 层",collective/ 是"算法语义层")。 |
| `topology/`     | 物理拓扑定义:`Torus`、`Ring`、`Switch`、`HierarchicalTopology` 等。给 collective/ 算法提供 ring/tree 形状。                                                                                                                                 |
| `scheduling/`   | 调度策略(FIFO/LIFO/HighestFirst 等),决定多个 collective 同时存在时谁先发。                                                                                                                                                                          |
| `memory/`       | 内存子系统的具体实现(配合 `MemBus` 用)。                                                                                                                                                                                                          |
| `fast-backend/` | 加速版后端,跳过部分细节做粗粒度模拟。                                                                                                                                                                                                               |

---

## 十、辅助/事件载荷类(数量多但都是小工具)

这些是事件回调时携带的数据载荷,理解一两个就够,基本是"event payload struct + 简单方法":

| 文件                                        | 作用                                                                              |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| `Callable.hh`                             | 抽象基类,定义 `void call(EventType, CallData*)`。所有能接收事件的对象都继承它。 |
| `CallData.hh`                             | 事件载荷的基类。                                                                  |
| `BasicEventHandlerData.hh/cc`             | 通用事件载荷(stream_id、phase 信息)。                                             |
| `RecvPacketEventHadndlerData.hh/cc`       | "收到一个包"事件的载荷。                                                          |
| `SendPacketEventHandlerData.hh/cc`        | "发出一个包"事件的载荷。                                                          |
| `RendezvousSendData / RendezvousRecvData` | 大消息 rendezvous 协议(先握手再传)的状态。                                        |
| `IntData.hh/cc`                           | 单 int 包装的载荷,用于简单回调。                                                  |
| `MemMovRequest.hh/cc`                     | 一次内存搬运的描述(与 `MemBus` 配套)。                                          |
| `DMA_Request.hh/cc`                       | DMA 请求结构。                                                                    |
| `BootStrapnet.hh/cc`                      | 启动阶段所有 rank 之间的"握手网络",建立通信组前的对齐。                           |

---

## 十一、统计与监控

| 文件                                    | 作用                                                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `StatData.hh/cc`                      | 统计数据基类。                                                                                                                       |
| `StreamStat.hh/cc`                    | 单条 stream 的耗时、带宽统计。                                                                                                       |
| `NetworkStat.hh/cc`                   | 网络层统计(每条链路收发字节、链路利用率)。                                                                                           |
| `SharedBusStat.hh/cc`                 | 共享总线(MemBus)的占用统计。                                                                                                         |
| `Usage.hh/cc`, `UsageTracker.hh/cc` | 全局资源利用率追踪。                                                                                                                 |
| `calbusbw.h / calbusbw.cc`            | **算总线带宽的算式**。给定拓扑/连接类型,计算实际可用带宽(考虑 NVLink 数量、PCIe lane 等),供 LogGP 和 SimAiFlowModelRdma 调用。 |

---

## 十二、串起来看一次 AllReduce 的代码路径

```
Workload 层 (训练 trace) 
   ↓ 调用 Sys::generate_collective(All_Reduce, size=4MB, group=DP)
Sys.cc
   ↓ 找到这个 rank 在 DP 组里的 MockNcclComm
MockNcclComm::get_flow_model
   ↓ MockNcclGroup::genAllReduceFlowModels → genAllReduceRingFlowModels
   ↓ 返回 List<SingleFlow>(带依赖图)
   ↓
collective/Ring.cc (算法层)
   ↓ 把 SingleFlow 包成 CollectivePhase 序列
StreamBaseline (stream 层)
   ↓ 推进每个 phase: 调用 MemBus 算 GPU 内搬运 + 调用 sim_send/sim_recv
MemBus.cc                 (算 GPU 内时延)
SimAiFlowModelRdma.cc     (转 RDMA QP 事件)
   ↓
AstraNetworkAPI 实现 (analytical / ns3 / phynet)
   ↓ 包级仿真,完成时回调
Sys::call(EventType::PacketReceived, ...)
   ↓ 触发依赖该包的下一个 SingleFlow
   ↓
全部 SingleFlow 完成 → DataSet 完成 → Workload 进入下一层
```

---

## 总结

- **`Sys.cc`** = 内核调度器
- **`AstraNetworkAPI.hh`** = 后端可插拔接口
- **`MockNccl*` + `collective/`** = NCCL 算法 + 拓扑展开
- **`StreamBaseline / DataSet / CollectivePhase / MyPacket / QueueLevels`** = collective 生命周期管理
- **`MemBus / LogGP / calbusbw`** = 计时模型
- **`SimAiFlowModelRdma`** = 接 ns-3 包级后端的桥
- **`PhyMultiThread / SimSendCaller`** = 接真实硬件 phynet 的桥
- **其余 `*Data.cc / *Stat.cc`** = 事件载荷 + 统计

整体哲学:**把一次集合通信沿"算法 → 阶段 → 流 → 包"四级逐级展开,每一级用一个 .cc 文件管理,最后落到一个抽象网络接口上**。这样换网络后端只需重写一个 `AstraNetworkAPI` 实现,其它代码完全复用。

# ASTRA-sim 的开销模型清单

按"代码所在层次"自上而下分类。 **🔵 = 真实排队/状态机** ,  **🟡 = 公式化(参数+size)** ,  **🔴 = 写死的常数** ,  **⚫️ = 未建模(用户填或忽略)** 。

---

## 1. 工作负载层(Workload)

| 项目                                        | 类型 | 来源/值                                        | 说明                                                    |
| ------------------------------------------- | ---- | ---------------------------------------------- | ------------------------------------------------------- |
| **GPU 计算时间(matmul/attention 等)** | ⚫️ | 来自 `.txt`workload 文件的 `comp_time`字段 | 仿真器**不算**矩阵乘的 FLOPs,直接读用户给的 ns 数 |
| 每层前向/后向时间                           | ⚫️ | 同上                                           | 一般来自 SimAI workload generator 或 profiler           |
| 算子之间的依赖                              | 🔵   | workload 里的 `parent_ids`链表               | 由 `Workload::call`驱动                               |

> **重要** :这意味着 ASTRA-sim  **不模拟"compute 和 comm overlap"的真实瓶颈** ,只模拟时间线上的重叠 —— 计算时间是黑盒。

---

## 2. 系统层 / Stream 调度(Sys.cc)

| 项目                              | 类型 | 值/位置                                             | 说明                                                                |
| --------------------------------- | ---- | --------------------------------------------------- | ------------------------------------------------------------------- |
| `processing_latency`            | 🔴   | `10`(`Sys.cc:172`)                              | Stream 调度处理一个事件的固定 tick                                  |
| `communication_delay`(端点延迟) | 🔴   | `10 * injection_scale`(`Sys.cc:188`)            | 每次 inject 的固定开销;可被 system 配置文件 `endpoint-delay:`覆写 |
| `local_reduction_delay`         | 🔴   | `1`(`Sys.cc:174`)                               | 本地 reduce 单位时间,被 LogGP 用                                    |
| `preferred_dataset_splits`      | 🔴   | `1`                                               | 拆分流的颗粒度                                                      |
| `num_channels`(legacy 路径)     | 🔴   | `1`                                               | 仅老 Ring 路径用,MockNccl 路径忽略                                  |
| **DataSet/BaseStream 队列** | 🔵   | `ready_list`/`running_list`/`running_streams` | LIFO/FIFO 调度,真排队                                               |
| **per-vnet 队列**           | 🔵   | `active_Streams[vnet]`+`queues_per_dim`         | 每个虚拟通道一个队列,有占用上限                                     |

---

## 3. 内存层(SimpleMemory.cc)

| 项目                   | 类型 | 公式                                                                                          | 说明                                                    |
| ---------------------- | ---- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `npu_mem_read/write` | 🟡   | `delay = size / npu_access_bw_GB`                                                           | 纯除法,**无队列** ,可以无限并发                   |
| `nic_mem_read/write` | 🔵   | `delay = size / nic_access_bw_GB`,但**串行化**到 `last_read_request_serviced`时间戳 | 这是个**轻量队列** :同一 NIC 的请求会被串成时间线 |
| `access_latency`     | 🔴   | 配置文件参数,固定常量                                                                         | 加在 `nic_mem_*`的开头                                |

注:`PacketBundle::call` 里的 `delay = mem_write(size) + 2 * mem_read(size)` 这个 3-访存模型是用来模拟"reduce 一次 = 写一次 + 读两次"的固定开销公式。

---

## 4. MemBus + LogGP(GPU 总线建模,系统层最复杂的一块)

`LogGP.cc` 实现的是经典 **LogGP** 性能模型(Latency, overhead, gap, Gap-per-byte):

| 参数                      | 类型 | 含义                       |
| ------------------------- | ---- | -------------------------- |
| `L`(Latency)            | 🔴   | 链路传播延迟,常数          |
| `o`(overhead)           | 🔴   | 每次发送/接收的 CPU 端开销 |
| `g`(gap)                | 🔴   | 两次连续传输之间的最小间隔 |
| `G`(Gap per byte)       | 🔴   | 每字节延迟 = 1/带宽        |
| `local_reduction_delay` | 🔴   | reduce 计算单位 ns/100B    |
| `THRESHOLD = 8`         | 🔴   | 切换发送/接收方向的阈值    |

 **真排队的部分** (`LogGP.hh:42-49`):

```cpp
std::list<MemMovRequest> sends;        // 🔵 发送队列
std::list<MemMovRequest> receives;     // 🔵 接收队列
std::list<MemMovRequest> processing;   // 🔵 reduce 计算队列
std::list<MemMovRequest> retirements;  // 🔵 完成等待队列
std::list<MemMovRequest> pre_send;     // 🔵 等总线队列
std::list<MemMovRequest> pre_process;  // 🔵 等处理队列
```

 **state machine** (`LogGP.hh:30`):

```cpp
enum class State { Free, waiting, Sending, Receiving };  // 🔵
enum class ProcState { Free, Processing };
```

 **关键计算公式** (`LogGP.cc:77-79, 116`):

```cpp
// 发送完成时间
register_event(Send_Finished, offset + G * (size - 1));   // 🟡
// 接收完成时间
register_event(Rec_Finished, offset + (size-1)*G + L + o); // 🟡
// reduce 处理时间
register_event(Processing_Finished, (size/100) * local_reduction_delay + 50); // 🟡 + 🔴(+50 是常数)
```

| 项目               | 类型 | 说明                                       |
| ------------------ | ---- | ------------------------------------------ |
| 发送序列化(g 间隔) | 🔵🟡 | 状态机强制 g 内只能发一次,**真排队** |
| 单次发送时长       | 🟡   | LogGP 公式                                 |
| 跨 bus 半双工竞争  | 🔵   | `subsequent_reads > THRESHOLD`切换方向   |
| reduction 处理     | 🟡🔴 | 线性 size + 固定 50 ns 偏移                |
| MemBus 是否启用    | 🔴   | `model_shared_bus = 0`(默认关)           |

> ⚠️  **默认配置 `inp_model_shared_bus = 0`** :意味着 LogGP 这套队列模型在默认 SimAI 跑出来的结果里 **几乎没有效果** 。要真的让它生效,需要在 system 配置文件里把 `model-shared-bus:` 设为 1。

---

## 5. NCCL 选路 / Channel 数(MockNccl)

| 项目                                                         | 类型 | 说明                                                                      |
| ------------------------------------------------------------ | ---- | ------------------------------------------------------------------------- |
| 算法选择(Ring/Tree/NVLS)                                     | 🔴   | `MockNcclGroup.cc:2028`写死的 if-else 规则,**没有 cost model**    |
| Channel 数                                                   | 🔴   | `gen_local_ring`: **= nlocalranks** ,与 message size 无关         |
| Chunk size                                                   | 🟡   | `data_size / nranks / nChannels`,纯除法                                 |
| `baseLat[]`、`hwLat[][][]`、`perChMaxRingLL128Bws[][]` | ⚫️ | **dead code** :在 `MockNccl.h`中声明,**C++ 里没有任何调用** |
| `nccl_infos`缓存                                           | 🔵   | `std::map`,首次决策后命中缓存                                           |
| 总线带宽 `calbusbw.cc`                                     | 🔴   | 经验查表 `(GPU type, nNodes, conn type) → BW`,完全静态                 |

---

## 6. Collective 流执行(NcclTreeFlowModel.cc)

| 项目                                       | 类型 | 说明                                                     |
| ------------------------------------------ | ---- | -------------------------------------------------------- |
| Step 间依赖                                | 🔵   | `indegree_mapping[flow_id]`计数器,真等待               |
| 同 step 内 N 条 flow 并行                  | 🔵   | 同时入 ns-3 队列                                         |
| `free_packets`per (channel, sender_node) | 🔵   | 真在记每个 channel 上谁还没发完                          |
| `_stream_count[i]`per channel            | 🔵   | 每个 channel 流计数                                      |
| 完成 → 发下一条                           | 🔵   | `--indegree_mapping[next] == 0`触发 `insert_packets` |
| ⚠ launch overhead / SM warmup             | ⚫️ | **完全不模拟**                                     |
| ⚠ kernel 启动延迟                         | ⚫️ | 不模拟                                                   |
| ⚠ DMA engine 排队                         | ⚫️ | 不模拟                                                   |

---

## 7. 网络层 ns-3 backend(开销建模最重的一层)

进了 `ns-3-alibabacloud`,从 `simai_ibv_post_send` 投递后:

| 项目                                  | 类型 | 说明                                                              |
| ------------------------------------- | ---- | ----------------------------------------------------------------- |
| 链路带宽                              | 🔵🟡 | 拓扑文件给的 `bw`,QbbNetDevice 真模拟传输时间                   |
| 链路传播延迟                          | 🔴   | 拓扑文件 `link.delay`,常量加到每个 packet                       |
| **交换机端口队列**              | 🔵   | `BEgressQueue`(8 优先级),真 FIFO 排队                           |
| **PFC 暂停/恢复**               | 🔵   | watermark + pause frame,真状态机                                  |
| **ECMP 路由**                   | 🔵   | 5-tuple hash,跨多路径分发                                         |
| **拥塞控制**                    | 🔵   | DCQCN / HPCC / Timely / DCTCP,可选,真 RTT/CNP 反馈                |
| **IRN / 选择性重传**            | 🔵   | 真模拟丢包重发                                                    |
| **MTU 分包**                    | 🔵🟡 | 大消息按 MTU 切,逐包入队                                          |
| Switch 处理时延                       | 🔴   | 通常 0 或固定常数,**不模拟 store-and-forward 队列处理时间** |
| 网卡处理时延                          | 🔴   | RdmaHw 里有少量固定开销                                           |
| **NIC 速率匹配 / inject limit** | 🔵   | RdmaQueuePair 状态机限速                                          |

---

## 8. 一张总览图(谁是常数,谁是队列)

```
┌──────────────────────────────────────────────────────────┐
│ 工作负载层      ⚫️  comp_time 完全外部输入                  │
├──────────────────────────────────────────────────────────┤
│ Sys 调度        🔴 processing_latency=10 / endpoint=10     │
│                 🔵 BaseStream 队列, vnet 队列              │
├──────────────────────────────────────────────────────────┤
│ NCCL 算法选择   🔴 if-else 写死, 不算 cost                 │
│                 🔴 channel 数 = nlocalranks 写死           │
│                 🔵 nccl_infos 缓存                         │
├──────────────────────────────────────────────────────────┤
│ Collective 执行 🔵 indegree 依赖图, free_packets, 真排队   │
│                 ⚫️ 不模拟 SM/launch overhead              │
├──────────────────────────────────────────────────────────┤
│ MemBus/LogGP    🔴 L/o/g/G 都是常数                        │
│                 🟡 公式 = offset + (size-1)*G + L + o      │
│                 🔵 6 条队列 + 状态机(默认配置不启用)        │
│                 ⚠️  默认 model_shared_bus=0,基本没效果      │
├──────────────────────────────────────────────────────────┤
│ Memory          🔴 access_latency 常数                     │
│                 🟡 size/bw 公式                            │
│                 🔵 NIC 端 last_serviced 串行化(轻队列)     │
├──────────────────────────────────────────────────────────┤
│ ns-3 fabric     🔵🔵🔵 真排队、真 PFC、真 ECMP、真 CC      │
│                 🔴 link.delay 常数                         │
│  ← 这一层是 SimAI 模拟精度的核心                            │
└──────────────────────────────────────────────────────────┘
```

---

## 9. 关键陷阱(决定结果可信度的几个点)

1. **算法/channel 数都是写死的** → 不要用 SimAI 比较 NCCL 不同算法/不同 nChannels 的影响,它没这能力。
2. **GPU 计算时间是黑盒** → comm/comp overlap 的"comp"部分没有真实模型,改 GPU 型号需要你自己换 workload。
3. **MemBus 默认关闭** → LogGP 那一坨队列模型在 default config 下基本是摆设。想研究 PCIe/NVLink 内部争用要主动开 `model-shared-bus`。
4. **Switch 处理时延 = 0** → 不模拟 store-and-forward,小消息延迟会偏低。
5. **没有 kernel launch / SM warmup 抖动** → 小消息 AllReduce 的尾延迟会偏低。
6. **`baseLat / hwLat` 是 dead code** → 想做 NCCL tuner 复刻就只能自己加。
7. **真正决定结果的是 ns-3 那一层** → 想得到准的网络结果,关键是拓扑文件 + CC 算法选对。

---

## 10. 一句话总结

ASTRA-sim 的开销模型是个  **"上窄下宽"的金字塔** :

* **上层(系统、调度、NCCL 决策)** 几乎全是常数和写死的规则;
* **中层(MemBus/LogGP)** 有完整的队列+状态机,但 **默认关闭** ;
* **下层(ns-3 fabric)** 是真正下了功夫的地方 —— 链路、队列、PFC、CC 全部真模拟。

所以  **SimAI 的模拟精度本质上 ≈ ns-3 模拟的精度** ;上面那些层主要负责"把 collective 翻译成正确的 P2P 流和正确的依赖"。
