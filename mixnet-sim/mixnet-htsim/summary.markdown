# mixnet-htsim 的 workload / NCCL 模拟分析

核心实现集中在 `src/clos/ffapp.{h,cpp}`，配合 flatbuffer 任务图 (`taskgraph_generated.h`，schema 见 `taskgraph.proto`)。整体架构是 **DAG 任务图驱动 + 事件队列调度 + 集合通信算法展开为 DCTCP flow**。

## 1. 任务图输入

- 外部 (mixnet-flexflow / FlexFlow) 生成 `.fbuf`，`FFApplication::load_taskgraph_flatbuf()` 加载 (ffapp.cpp:184)。
- 顶层元数据：`nnodes/ngpupernode/dp_degree/tp_degree/pp_degree/ep_degree`，每节点默认 `NUM_GPU_PER_NODE = 8` (ffapp.h:20)。
- 三类实体：
  - **Device** (`FFDevice`)：GPU / CPU / NVLINK / PCI / NW_COMM。NW_COMM 记 `from_node → to_node`。
  - **Task** (`FFTask` 及子类)：`TASK_FORWARD/BACKWARD/UPDATE/BARRIER/COMM/P2P/ALLREDUCE/REDUCESCATTER/ALLGATHER/ALLTOALL/…` (见 `SimTaskType`)。
  - **依赖边**：`nexttasks` + 前驱计数 `counter`。

## 2. 调度模型 (DAG + EventList)

- `FFTask` 继承 `EventSource`，用 `eventlist().sourceIsPending(*t, time)` 入队。

- `start_init_tasks()`：所有 `counter==0` 的根任务立即 schedule (ffapp.cpp:770)。

- ```
  doNextEvent() → taskstart()
  ```

  ：

  - 普通计算任务 → `execute_compute()`：检查设备空闲，占用 `device->busy_up_to`，用 flatbuffer 里的 `runtime` 作为 GPU/CPU 解析执行时间 (ffapp.cpp:969)。
  - 通信任务 → `start_flow()`，走真实网络仿真。

- `cleanup()` (ffapp.cpp:1050)：递减后继 `counter`，归零就 `sourceIsPending`。跑完整图后调用 `reset_and_restart()` 进入下一 iteration。

这个模型等价于把 FlexFlow 的算子级 DAG「按顺序触发」一遍，各 GPU 在本地是串行独占的。

## 3. NCCL 集合通信的模拟方式

每种集合操作被「算法展开」成若干 TCP flow，而不是用一个黑盒延迟模型：

### Ring AllReduce (`FFRingAllreduce`, ffapp.cpp:1305)

最接近 NCCL ring 的实现：

- `doNextEvent`：一次发起 `node_group.size()` 条并行 flow，每条从 `node_group[i]` → `node_group[(i+1) % N]`，每条传 `operator_size / N` 字节。
- `ar_finish_ring()` 回调：当本 round 的 N 条 flow 全部完成 → `curr_round++`；当 `curr_round == 2*(N-1)`（NCCL ring 的 reduce-scatter + all-gather 总轮数）→ 任务完成。
- 小包处理：若 `operator_size < MTU*N`，直接 `operator_size *= 2*(N-1)/N` 一轮结束 (early terminate)。
- Intra-node (同一 8 卡节点)：不走网络，`finish_time = start_time + size / nvlink_bandwidth`（`NVLINK_BANDWIDTH=600 GB/s`, ffapp.cpp:19）。

### 其它 AllReduce 策略 (可通过 `FFAllReduceStrategy` 选)

- `FFPSAllreduce`：1 个 PS (node_group[0])，2 轮 (scatter → gather)。
- `FFDPSAllreduce`：每对节点间直接 all-to-all 互发 (fully connected)。
- `FFNewRingAllreduce`：多 ring + 自定义 `jumps`，对应非相邻邻居的多路并行 ring。

### ReduceScatter / AllGather (ffapp.cpp:1853, 1985)

结构与 ring AR 一致，但总轮数是 `N-1`（不是 `2(N-1)`）。

### AllToAll (`FFAlltoAll`, ffapp.cpp:2119)

- 没有「分轮」，而是一次性为所有 `(src, dst)` pair 发一条 flow。`total_rounds = from*to*tp_degree`，每条完成时 `curr_round++`。

- MoE 权重矩阵

  ：flow 大小不是均匀的，而是用外部加载的 

  ```
  weight_matrix
  ```

   (

  ```
  load_weight_matrix
  ```

  , ffapp.cpp:153) 做 top-k 路由权重：

  ```
  xfer_size[fn,tn] = total_xfer_size * weight_matrix[fn_exp%8][tn_exp%8] / 32768
  ```

  根据 info 里的 

  ```
  GROUP_BY forward
  ```

   / 

  ```
  AGGREGATE backward
  ```

   决定矩阵方向 (ffapp.cpp:395-405)。这是本仓库模拟 MoE expert dispatch / combine 的核心。

- Mixnet 特有：根据 `topology->conn[src_node][dst_node]` 判断这条 flow 走 OCS 光路 (`is_elec=false`) 还是电 fabric (`is_elec=true`)；并统计 `node_ocs_volume` / `node_eps_volume`。

- 正向 `GROUP_BY` 的 AllToAll 在 `cleanup()` 路径 (ffapp.cpp:1066) 会调 `updatetrafficmatrix()` 把流量写入 `all2all_traffic_matrix`，触发 `RegionalTopoManager` 光路重配。

### P2P (`FFP2P`, ffapp.cpp:2337)

把 `(src_indices[i], dst_indices[i])` 成对的 flow 并发发出，PP 的点对点用。

## 4. flow 层：真实网络协议仿真

所有 `start_flow()` 路径都是同一套：

```cpp
DCTCPSrc* src = new DCTCPSrc(..., eventlist(), src_gpu, dst_gpu, <callback>, <ctx>);
TcpSink*  snk = new TcpSink();
src->set_flowsize(...);
src->set_ssthresh(ssthresh * data_packet_size());

// 选路径（多路径就 rand 选一条）
srcpaths = topology->get_paths(src, dst);      // 纯电
// or get_eps_paths(src, dst);                 // mixnet: 强制走电 fabric
routeout = new Route(*srcpaths->at(choice));
routeout->push_back(snk);

src->connect(routeout, routein, snk, start_time);
```

- 用 `DCTCP`（带 ECN）承载集合通信的每一条 flow，链路仿真交给 htsim 的 `Queue/Pipe/Switch`。
- `tcpRtxScanner` 负责 RTO、tcp_info 打点（带有 `micro_batch_id / layer_id / target_*` 便于和训练图对齐）。
- flow 完成回调 (`ar_finish_ring` / `finish_alltoall` / `finish_p2p` / …) 再推进集合通信的下一 round 或标记任务结束。

## 5. 和 Mixnet (OCS+EPS) 的耦合

- ```
  is_mixnet=1
  ```

   时：

  - AllReduce 总是走 EPS (`is_elec=true`，`get_eps_paths`)；
  - AllToAll 按 `conn` 矩阵决定 OCS/EPS；
  - `TASK_BACKWARD` 里包含 "Aggregate" 名字时，在 `execute_compute()` 里触发本 region 的重配 (ffapp.cpp:997)；
  - `TASK_ALLTOALL GROUP_BY forward` 在 `cleanup()` 里根据新流量矩阵触发重配 (ffapp.cpp:1066)，并 `sourceIsPending(task, ready_time + reconf_delay + 10)` 让流量等光路切完再发。

## 总结

mixnet-htsim 不是「一条 NCCL = 一个封闭公式」，而是：

1. **上层**：FlexFlow 风格的算子 DAG 任务图（.fbuf），计算靠固定 runtime + device 互斥模拟。
2. **集合通信层**：按 NCCL 算法手写展开 —— Ring AR 是 `2(N-1)` 轮、每轮 N 条并发 flow、每条 `size/N` 字节；RS/AG 是 `N-1` 轮；AllToAll 是 pairwise flow、大小按 MoE weight 矩阵；P2P 是直接配对。
3. **flow 层**：每条 flow 真的用 `DCTCPSrc/TcpSink` 走 htsim 的 queue/pipe/switch，享受 ECN、多路径、OCS 重配逻辑。
4. **节点内**：同 8 卡直接按 `size / NVLINK_BW (600 GB/s)` 解析计算，不进网络。

这种设计的好处是：能同时评估 NCCL 算法选择（ring/PS/DPS）、真实 DCTCP 动态，和 OCS 动态重配三者的交互——这也是 mixnet 项目关心的核心问题。

# 拓扑选择

现在看得很清楚了。NCCL 算法选择是**一个全局命令行开关**，在 `load_taskgraph_flatbuf` 加载任务图时按 `allreduce_strategy` 字段直接实例化对应子类。

## 1. 入口：命令行 `-ar` 参数

`main_tcp_flat.cpp:186-191` 解析 `-ar ring|ps|dps`，再在 `:277-290` 映射成枚举 `FFAllReduceStrategy`：

```cpp
FFApplication::FFAllReduceStrategy ar_strategy = FFApplication::FF_DEFAULT_AR;
if      (allreduce_strategy == "ring") ar_strategy = FFApplication::FF_RING_AR;
else if (allreduce_strategy == "ps")   ar_strategy = FFApplication::FF_PS_AR;
else if (allreduce_strategy == "dps")  ar_strategy = FFApplication::FF_DPS_AR;
```

`FFAllReduceStrategy` 定义见 ffapp.h:460:

```cpp
enum FFAllReduceStrategy { FF_RING_AR, FF_PS_AR, FF_DPS_AR, FF_DEFAULT_AR };
```

这个枚举通过构造函数传给 `FFApplication` 并存入 `allreduce_strategy` 成员 (ffapp.cpp:100-118)。

## 2. 实际选择点：`load_taskgraph_flatbuf` 建任务时

算法选择并不是运行时每个 AR 自己决定，而是 **加载任务图的阶段**，遇到 `TASK_ALLREDUCE / SUB / DP / TP_ALLREDUCE` 时，根据 `allreduce_strategy` 直接 `new` 不同的 `FFTask` 子类 (ffapp.cpp:247-327)：

```cpp
if (this_task.type() == SimTaskType_TASK_ALLREDUCE ||
    this_task.type() == SimTaskType_TASK_SUB_ALLREDUCE ||
    this_task.type() == SimTaskType_TASK_DP_ALLREDUCE ||
    this_task.type() == SimTaskType_TASK_TP_ALLREDUCE) {

    if (fancy_ring) {                         // 另一个开关：多 ring + jumps
        tasks[id] = new FFNewRingAllreduce(...);
    }
    else if (allreduce_strategy == FF_RING_AR ||
             allreduce_strategy == FF_DEFAULT_AR) {
        tasks[id] = new FFRingAllreduce(...);      // 对应 NCCL ring
    }
    else if (allreduce_strategy == FF_PS_AR) {
        tasks[id] = new FFPSAllreduce(...);        // 参数服务器
    }
    else if (allreduce_strategy == FF_DPS_AR) {
        tasks[id] = new FFDPSAllreduce(...);       // distributed PS / full-mesh
    }
}
```

每个子类都是 `FFTask` 的派生类，`doNextEvent()` / `start_flow()` / 对应的 `ar_finish_*` 回调各自实现了一套通信模式。从这一刻起，运行时就不再做算法选择——任务图里的每个 AR 对象已经是 ring / PS / DPS 之一。

## 3. 关键特征 & 注意点

- **全局一致**：整个 app 只有一个 `allreduce_strategy`，**所有 AllReduce 任务（不论 DP/TP/SUB）共用同一算法**。没有按任务粒度切换的机制。
- **`FF_DEFAULT_AR` == `FF_RING_AR`**：不传 `-ar` 就是 ring。
- **`fancy_ring` 优先级最高**，但它是 `FFApplication` 构造时硬编码 `fancy_ring = false` (ffapp.cpp:107)，正常路径下不会走 `FFNewRingAllreduce`。`selected_jumps` 也没人填，相关代码基本是 dead path。
- **`main_tcp_mixnet.cpp` 硬编码为 `FF_DEFAULT_AR`**（ffapp.cpp 调用处 ffapp.cpp:402），mixnet 实验里 AllReduce 只会走 ring。
- **ReduceScatter / AllGather / AllToAll / P2P 不参与这个选择**，它们没有多实现可选，固定走各自的类 (`FFReduceScatter` / `FFAllGather` / `FFAlltoAll` / `FFP2P`)。
- 选择发生在 **load 阶段**，意味着想对比 ring vs ps vs dps 得重新加载任务图并构造新的 `FFApplication`。

## 4. 三种实现的算法差别（启发式 NCCL 模型）

| 策略         | 类                                 | 轮数            | 每轮 flow 数            | 每 flow 大小 | 完成条件                |
| ------------ | ---------------------------------- | --------------- | ----------------------- | ------------ | ----------------------- |
| `FF_RING_AR` | `FFRingAllreduce` (ffapp.cpp:1305) | `2*(N-1)`       | `N` 条 (ring 邻居)      | `size/N`     | `curr_round == 2*(N-1)` |
| `FF_PS_AR`   | `FFPSAllreduce` (ffapp.cpp:1622)   | `2` (push→pull) | `N-1` 条 (节点↔`ng[0]`) | `size`       | `curr_round == 2`       |
| `FF_DPS_AR`  | `FFDPSAllreduce` (ffapp.cpp:1743)  | `1`             | `N*(N-1)` 条 (全互联)   | `size/N`     | 所有 flow 结束          |

这三种都是「算法展开成 TCP flow → 由 htsim 网络层真实仿真拥塞」的模式，差异只在**通信拓扑/轮数**上，并不是调用一个 NCCL 黑盒延迟公式。

## 总结

NCCL 算法选择 = **命令行 `-ar` → 枚举 → `load_taskgraph_flatbuf` 里 `if/else` 实例化不同的 `FFTask` 子类**。一次选定，全局生效，任务图里所有 AllReduce 节点都用这一种算法；其它集合操作没有多实现可选。