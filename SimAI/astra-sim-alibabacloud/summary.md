# ASTRA-sim 的完整工作流

ASTRA-sim 是一个**事件驱动的分层模拟器**。它的工作流可以概括为：**读 workload → 翻译成集合通信调度 → 拆解成 P2P send/recv → 调用网络后端模拟 → 收集统计**。

下面我从架构、初始化、运行时事件循环、各层职责四个角度完整讲解。

---

## 一、ASTRA-sim 的三层架构

```
┌──────────────────────────────────────────────────────┐
│ Layer 1: Workload Layer                              │
│   - 读 trace 文件 (来自 AICB)                         │
│   - 维护事件序列和依赖关系                             │
│   - 区分 forward / backward / optimizer 阶段          │
└──────────────────┬───────────────────────────────────┘
                   │ "执行 all_reduce(256MB, tp_group)"
                   ▼
┌──────────────────────────────────────────────────────┐
│ Layer 2: System Layer                                │
│   - SimCCL (MockNccl*): 集合通信算法选择               │
│   - 拓扑感知 (机内/机间)                              │
│   - 调度器 (DataSet, BaseStream)                      │
│   - 内存总线模拟 (MemBus)                              │
│   - 把 collective 拆成 send/recv 序列                  │
└──────────────────┬───────────────────────────────────┘
                   │ send(rank0, rank1, 36MB, t=100us)
                   ▼
┌──────────────────────────────────────────────────────┐
│ Layer 3: Network Frontend (可插拔)                    │
│   - analytical: α-β 模型解析                          │
│   - ns3: 调用 ns-3-alibabacloud                      │
│   - phynet: 真实网络                                  │
└──────────────────────────────────────────────────────┘
```

每层只和相邻层通信，**通过 callback 上报完成事件**。

---

## 二、启动阶段：初始化流程

### 步骤 1：读配置和拓扑

```
$ AstraSim 启动参数:
  --workload <workload.csv>           ← AICB 生成的 trace
  --system-config <SimAI.conf>         ← 系统配置
  --network-config <topology.txt>      ← 网络拓扑
  --num-npus 64                        ← GPU 数量
```

ASTRA-sim 解析这些文件，构建：

```cpp
struct SystemConfig {
    int num_npus = 64;
    int local_mem_bw = 600;         // GB/s, HBM 带宽
    int boost_mode = 0;
    string collective_algorithm = "ring";
    string scheduling_policy = "LIFO";
    ...
};

struct NetworkTopology {
    int num_dim = 3;                // 3D Torus
    vector<int> units_per_dim = {8, 4, 2};
    vector<float> link_bandwidth = {300, 200, 100}; // GB/s 每维度
    vector<float> link_latency = {0.5, 1.0, 2.0};   // us
};
```

### 步骤 2：为每个 NPU 创建 Sys 对象

ASTRA-sim 给**每个 GPU rank** 都创建一个 `Sys` 对象：

```cpp
for (int rank = 0; rank < num_npus; rank++) {
    Sys* sys = new Sys(
        network_backend,    // 共享的网络后端
        rank,               // 自己的全局 rank
        system_config,
        topology,
        ...
    );
    sys->initialize();
}
```

每个 `Sys` 实例就是"一张 GPU 的视角"，里面包含：
- 自己的 rank 号
- 自己所在的 collective groups（TP/DP/PP）
- 自己的事件队列
- 自己的内存总线状态

### 步骤 3：解析 workload trace

```cpp
Workload* workload = new Workload(workload_file, sys);
workload->parse();
```

workload trace（来自 AICB）的格式大致是：

```
HYBRID_TRANSFORMER 1024 4096 ... 32 ...
forward.start
  all_reduce  tp_group  256MB  forward.embedding
  computation 50us               forward.matmul
  all_gather  tp_group  64MB    forward.attention
  ...
backward.start
  ...
```

workload 层把每行解析成一个 `LogItem`/`CollectivePhase` 对象，构成一个**事件列表**。

### 步骤 4：构建集合通信组

通过 `RankGenerator` 之类的逻辑，每个 Sys 对象计算出自己属于哪些 group：

```cpp
sys->tp_group = {16, 17, 18, 19, 20, 21, 22, 23};   // 我和谁一组做 TP
sys->dp_group = {17, 25};                            // 我和谁一组做 DP
sys->pp_group = {1, 17, 33, 49};                     // 我和谁一组做 PP
```

---

## 三、运行时：事件驱动循环

ASTRA-sim 是一个**离散事件模拟器**。整个仿真就是一个全局事件队列的不断处理：

```cpp
EventQueue global_queue;   // 按时间排序的优先队列

while (!global_queue.empty()) {
    Event* e = global_queue.pop();
    sim_time = e->time;
    e->call();   // 触发回调
}
```

每个 `Event` 包含：
- 触发时间
- 触发对象（哪个 Sys / Stream / DataSet）
- 回调函数

---

## 四、一次集合通信的完整生命周期

让我们追踪 **`all_reduce(256MB, tp_group)`** 在 ASTRA-sim 里的完整流程。

### 阶段 1：Workload 层提交

```cpp
// Workload 层读到 trace 的下一行
LogItem item = trace[index++];
// item = {comm_type: ALL_REDUCE, group: tp_group, size: 256MB}

// 构造 CollectivePhase
CollectivePhase* phase = new CollectivePhase(
    sys, rank, ALL_REDUCE, tp_group, 256MB
);

// 提交给 system 层
sys->generate_collective(phase);
```

### 阶段 2：System 层选择算法（SimCCL）

```cpp
// MockNcclGroup 决定用什么算法
Algorithm algo = MockNcclGroup::choose_algorithm(
    ALL_REDUCE, 256MB, tp_group_size=8, topology
);
// 返回: RING (大消息走 Ring)
```

### 阶段 3：算法展开成 P2P 序列

Ring AllReduce 包含两个阶段：
- **Reduce-Scatter** (N-1 = 7 步)
- **AllGather** (N-1 = 7 步)

每一步都是一次 send + recv：

```cpp
// 创建一个 BaseStream 来管理这次 collective
BaseStream* stream = new RingAllReduceStream(...);

// 为每一步创建 P2P 操作
for (int step = 0; step < 14; step++) {
    int src = (rank - 1 + group_size) % group_size;
    int dst = (rank + 1) % group_size;
    int chunk_size = 256MB / 8;   // 32MB per chunk
    
    MyPacket* pkt = new MyPacket(src, dst, chunk_size, step);
    stream->packets.push_back(pkt);
}
```

### 阶段 4：调度到内存总线

GPU 不能瞬间发数据——要先经过 HBM → NIC 的 DMA。System 层用 `MemBus` 模拟这个：

```cpp
// 把数据从 HBM 拷贝到 NIC buffer
MemMovRequest* mem_req = new MemMovRequest(
    chunk_size,
    sys->local_mem_bw   // 600 GB/s
);
membus->enqueue(mem_req);

// 内存搬运完成后触发网络发送
mem_req->set_callback([=]() {
    sys->network->send(src, dst, chunk_size, callback);
});
```

模拟出来的时间：
```
chunk = 32MB, mem_bw = 600 GB/s
mem_time = 32 MB / 600 GB/s = 53 us
```

### 阶段 5：调用网络后端

```cpp
// 委托给 network frontend
network_backend->sim_send(
    src=17, dst=18,
    size=32MB,
    callback=stream->on_send_complete
);
```

不同的后端有不同的实现：

#### Analytical 后端（α-β 模型）

```cpp
double time = latency + size / bandwidth;
// = 1us + 32MB / 200 Gbps
// = 1us + 1280us = 1281us
schedule_event(callback, sim_time + time);
```

#### ns-3 后端

```cpp
// 把这个 send 翻译成 ns-3 的 application 调用
ns3::Send(src_node, dst_node, size, ns3_callback);
// ns-3 内部把 32MB 切成 MTU 包
// 模拟 RDMA、DCQCN、PFC 等
// 最后回调通知 sys 时间到了
```

### 阶段 6：完成事件回调

当 ns-3 / analytical 通知 send 完成：

```cpp
void Stream::on_send_complete(int step) {
    completed_steps++;
    if (completed_steps == 14) {
        // 整个 AllReduce 完成
        sys->workload->on_collective_complete(phase_id);
    } else {
        // 触发下一个 step
        execute_next_step();
    }
}
```

### 阶段 7：Workload 层推进

```cpp
void Workload::on_collective_complete(int phase_id) {
    // 标记这个 phase 完成
    completed_phases++;
    
    // 检查依赖：这个 collective 完成后，是否能触发新事件？
    for (Phase* next : dependency_graph[phase_id]) {
        if (all_deps_satisfied(next)) {
            schedule(next);
        }
    }
}
```

---

## 五、计算事件 vs 通信事件

Workload trace 里既有通信操作也有计算操作：

```
all_reduce  tp_group  256MB  forward.embedding
computation 50us               forward.matmul     ← 计算事件
all_gather  tp_group  64MB
```

### 计算事件的处理

**ASTRA-sim 自己不模拟计算**——计算时间是 AICB 的 **AIOB 模块**预先实测的。Workload 层只是读取这个时间然后推进：

```cpp
case CommType::computation:
    // 直接从 trace 读取时间
    double comp_time = item.elapsed_time;   // 50us
    schedule_event(on_complete, sim_time + comp_time);
    break;
```

这就是为什么 AICB 里有个 AIOB 子模块——它专门负责实测各种 op 的时间，写进 trace，让 ASTRA-sim 直接用。

---

## 六、计算和通信的重叠（Overlap）

ASTRA-sim 的一个重要能力是模拟**计算和通信的重叠**：

```
时间轴:
  T=0:    forward.layer1 (50us 计算)
  T=0:    后台启动 all_reduce(layer0_grad)   ← 同时进行
  T=50:   forward.layer1 完成
  T=50:   forward.layer2 开始 (50us)
  T=80:   all_reduce 完成
  T=100:  forward.layer2 完成
  
真实墙钟时间: 100us (而不是 50+30+50 = 130us)
```

ASTRA-sim 用**多个并行的 BaseStream** 来管理可以重叠的事件：

```cpp
// Stream 1: 计算流
compute_stream->add(layer1_compute);
compute_stream->add(layer2_compute);

// Stream 2: 通信流
comm_stream->add(layer0_grad_allreduce);

// 两个 stream 并行推进，事件循环交错处理它们
```

最终的 sim_time 由**最慢的 stream** 决定。

---

## 七、调度策略和依赖

每个 GPU 的事件不是无脑顺序的，而是有**依赖关系**：

```
forward.layer1  ←─── 必须在 forward.layer0 之后
  │
  └── forward.layer2 必须在 forward.layer1 之后

backward.layer2 必须在 forward.layer2 之后（链式法则）

optimizer.step 必须在所有 backward 完成后
```

ASTRA-sim 用 **DataSet** 和 **CollectivePhase** 维护这些依赖：

```cpp
class DataSet {
    int id;
    vector<DataSet*> dependencies;     // 我依赖谁
    vector<DataSet*> dependents;       // 谁依赖我
    bool is_complete;
    
    void check_and_schedule() {
        for (DataSet* dep : dependencies) {
            if (!dep->is_complete) return;
        }
        // 所有依赖都满足了，可以执行
        execute();
    }
};
```

---

## 八、最终输出

仿真结束后，ASTRA-sim 输出：

### 全局统计

```
Total simulation time: 245.3 ms
Compute time: 180.0 ms (73%)
Communication time: 110.5 ms (45%)
Overlap time: 45.2 ms (18%)
```

### 每个阶段细分

```
forward.embedding:    12.5 ms
forward.attention:    45.0 ms
forward.mlp:          35.0 ms
backward.attention:   60.0 ms
backward.mlp:         50.0 ms
optimizer.step:       42.8 ms
```

### 每个 collective 的开销

```
all_reduce (TP, 256MB) avg: 8.5 ms × 32 occurrences = 272 ms
all_gather (DP, 64MB)  avg: 4.2 ms × 8  occurrences = 33.6 ms
send/recv  (PP, 4MB)   avg: 1.5 ms × 16 occurrences = 24 ms
```

### 带宽利用率

```
TP group bandwidth utilization: 78%
DP group bandwidth utilization: 45%
PP group bandwidth utilization: 22%
```

---

## 九、完整工作流图

```
┌──────────────────────────────────────────────────────────┐
│ Phase 1: 启动                                             │
│   ① 解析 CLI 参数和配置文件                                │
│   ② 创建网络后端 (ns3 / analytical)                       │
│   ③ 为每个 NPU 创建 Sys 对象                              │
│   ④ 解析 workload trace                                   │
│   ⑤ 构建集合通信组                                        │
│   ⑥ 把第一批事件入队                                       │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 2: 事件循环 (主循环)                                 │
│                                                           │
│   while event_queue not empty:                            │
│     event = pop_min_time()                                │
│     sim_time = event.time                                 │
│                                                           │
│     if event is COMPUTE:                                  │
│       直接读取预测时间，schedule completion ★ AIOB         │
│                                                           │
│     elif event is COLLECTIVE:                             │
│       SimCCL 选择算法 (Ring/Tree/HD)                       │
│       展开成 P2P send/recv 序列                            │
│       逐个 step 启动:                                      │
│         MemBus 模拟 HBM→NIC 拷贝                          │
│         调用 network backend 模拟 send                    │
│         注册 send_complete 回调                            │
│                                                           │
│     elif event is SEND_COMPLETE:                          │
│       推进 stream 状态                                     │
│       触发下一个 step 或 collective 完成                   │
│                                                           │
│     elif event is COLLECTIVE_COMPLETE:                    │
│       通知 workload 层                                     │
│       检查依赖，激活新事件                                  │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 3: 收尾                                             │
│   ① 收集统计数据                                           │
│   ② 输出文本/CSV/json 报告                                 │
│   ③ 析构所有对象                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 十、和其他组件的交互时序

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  AICB   │    │ ASTRA-sim│    │ SimCCL   │    │ ns-3     │
└────┬────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
     │              │               │               │
     │ workload.csv │               │               │
     │─────────────→│               │               │
     │              │               │               │
     │              │ collective    │               │
     │              │──────────────→│               │
     │              │               │               │
     │              │               │ select algo   │
     │              │               │ (Ring)         │
     │              │               │               │
     │              │ send/recv list│               │
     │              │←──────────────│               │
     │              │               │               │
     │              │ for each step:│               │
     │              │ sim_send()    │               │
     │              │──────────────────────────────→│
     │              │               │               │
     │              │               │               │ pkt sim
     │              │               │               │ DCQCN
     │              │               │               │
     │              │ on_complete   │               │
     │              │←──────────────────────────────│
     │              │               │               │
     │              │ next step ... │               │
     │              │               │               │
     │              │ all done      │               │
     │              │               │               │
     │              │ stats output  │               │
     │              │ (csv/json)    │               │
```

---

## 十一、ASTRA-sim 关心的核心问题

回到工作流的本质——ASTRA-sim 用整个流程来回答以下问题：

| 问题 | 通过什么机制回答 |
|---|---|
| 一次 collective 用什么算法？ | SimCCL 算法选择 |
| 算法展开后有多少个 P2P send？ | SimCCL 算法展开 |
| 每个 send 在网络上多久？ | Network backend (analytical/ns3) |
| 内存总线的开销多少？ | MemBus 模拟 |
| 计算和通信能不能重叠？ | 多 Stream 并行调度 |
| 哪些事件依赖哪些事件？ | DataSet 依赖图 |
| 整个 step 多少时间？ | sim_time 累积 |

---

## 十二、一句话总结

| 阶段 | 做什么 |
|---|---|
| **启动** | 读 workload trace、配置、拓扑；为每个 NPU 创建 Sys；构建通信组 |
| **事件循环** | 弹出最早事件 → 触发回调 → 产生新事件 → 推进 sim_time |
| **遇到 collective** | SimCCL 选算法 → 展开成 send/recv 序列 → MemBus 模拟 HBM→NIC → network backend 模拟传输 → 完成后回调 |
| **遇到 compute** | 直接读 AIOB 实测时间，推进 sim_time |
| **依赖管理** | DataSet 图保证 forward → backward → optimizer 顺序，多 Stream 支持计算/通信重叠 |
| **结束** | 输出全局时间、各阶段细分、带宽利用率 |

**核心洞察**：ASTRA-sim 的工作流可以浓缩成**"workload trace 驱动的事件循环 + 三层翻译"**——上层（workload）说"做什么"，中层（system + SimCCL）说"怎么做"，下层（network backend）说"做了多久"。整个系统是分层解耦的：换一个 workload trace 就能模拟新模型，换一个 network backend 就能切换精度/速度的折中——这正是 ASTRA-sim 的设计精髓。