# MixNet 模拟器中 OCS/EPS 动态选择与 OCS 重配置机制分析

本文对应 `paper.pdf`（*MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed MoE Training*, SIGCOMM'25）在 `mixnet-htsim` 中的实现映射。重点回答两个问题：
1. 模拟器如何在每条流上**动态**选择 OCS 与 EPS 交换机
2. 模拟器如何对 **区域 OCS** 进行重配置

---

## 1. 论文架构回顾

- **双平面互联**：每台服务器有 8 张 NIC，拆分后分别接入 **区域 OCS**（Regional OCS domain）和 **全局 EPS**（Global EPS fabric）。见 paper Fig. 6。
- **区域划分**：整个集群按 `region_size` 划分为若干 region；每个 region 内部只有一个 OCS，跨 region 必须走 EPS。
- **每个 region 的 OCS** 有 `alpha` 条可用端口/节点，决定了单个节点最多可建的 OCS 直连数。
- **触发条件**：MoE 层的 All-to-All 具有强 local pattern，可以根据 gate unit 的输出预测 expert 通信需求，从而在执行前重配 OCS。
- **算法 1 (ReconfigureOCS)**：基于贪心，按需求矩阵中"最长完成时间"迭代分配 OCS 电路，直到耗尽每个节点的 alpha 条端口，再生成 NIC-level 映射并下发。
- **隐藏开销**：把重配置时间与前一阶段 attention / expert 计算并行，尽量不阻塞 critical path。

---

## 2. 模拟器类映射

| 论文组件 | 模拟器类 | 路径 |
|---|---|---|
| 整体 OCS+EPS 拓扑 | `Mixnet` | `mixnet-htsim/src/clos/datacenter/mixnet.{h,cpp}` |
| 全局 EPS fabric | `Mixnet::elec_topology`（另一个 `Topology*`） | 同上 |
| TopologyController（全局） | `MixnetTopoManager` | `mixnet-htsim/src/clos/mixnet_topomanager.{h,cpp}` |
| TopologyController（region） | `RegionalTopoManager`（每个 region 一个） | 同上 |
| TrafficMonitor | `All2AllTrafficRecorder` | 同上 |
| CollectiveCommunicationsManager | `FFAlltoAll` / `FFRingAllreduce` / `FFP2P` 等 | `ffapp.{h,cpp}` |

关键状态：
- `Mixnet::conn[i][j]`：region 内节点 i→j 当前的 OCS 电路数；**0 表示无直连**，需走 EPS。
- `Mixnet::queues[i][j]`：对应的 queue 对象；`_bitrate = SPEED * conn[i][j]`。
- `RegionalTopoManager::status ∈ {TOPO_INIT, TOPO_LIVE, TOPO_RECONF}`，配合 `reconfig_end_time` 控制新流是否立即放行。

---

## 3. 动态 OCS / EPS 选择——**per-flow** 的决策

所有"动态选择"都发生在 `FFAlltoAll::start_flow()`（`ffapp.cpp`，约 2185–2319 行）——**每一条 A2A flow 单独判断**，而不是统一拨一个开关。

核心代码（简化）：

```cpp
int src_node = src_gpu / NUM_GPU_PER_NODE;
int dst_node = dst_gpu / NUM_GPU_PER_NODE;
bool is_elec;
if (ffapp->topology->conn[src_node][dst_node] > 0) {
    // 该 (src,dst) 目前存在 OCS 电路 → 走光网络
    srcpaths = ffapp->topology->get_paths(src_gpu, dst_gpu);
    is_elec = false;
} else {
    // 无 OCS 电路 → 退回 EPS
    srcpaths = ffapp->topology->get_eps_paths(src_gpu, dst_gpu);
    is_elec = true;
}
tcpsrc->set_is_elec(is_elec);
```

其中：
- `Mixnet::get_paths()`（`mixnet.cpp:186`）在 `conn[src][dst] > 0` 时返回 region 内直连路径；否则走 fallback。
- `Mixnet::get_eps_paths()`（`mixnet.cpp:246`）**无条件**委托给 `elec_topology->get_paths()`，即全局 EPS fabric 自己的路由逻辑。

### 哪些流**强制** EPS？
- `FFRingAllreduce` / `FFPSAllreduce` / `FFDPSAllreduce`（即所有 AllReduce 变体）在创建 TCP flow 时**直接**调用 `get_eps_paths()`（`ffapp.cpp:1416` 等），对应 paper §5.3 "DP via hierarchical EPS AllReduce"。
- `FFP2P`（pipeline 通信）类似。
- 只有 **A2A（EP 流量）** 才会根据 `conn` 矩阵按流决定走 OCS 还是 EPS —— 这正是 paper Fig. 7 中 "EP via OCS direct-connect, fall back to EPS if no circuit" 的实现。

### 重配置窗口内的新流
`FFAlltoAll::start_flow()` 尾部（约 2296–2315 行）还判断对应 region 是否处于 `TOPO_RECONF`：若是，则把 `tcpsrc->connect()` 调度延后到 `regional_topo_manager->reconfig_end_time`，避免在新电路尚未建好时就把包灌进去。

---

## 4. 流量矩阵的采集与下发

### 4.1 计算阶段
`FFAlltoAll::updatetrafficmatrix()`（`ffapp.cpp:2133`）遍历 `operator_sizes`（MoE weight matrix 加权后的 per-(src,dst) 传输量），累加出该 region 的 `Matrix2D<double> tm`。MoE 权重来源于：

```cpp
xfer_size_tmp = total_xfer_size * 1.0 *
                weight_matrix[fn_exp%8][tn_exp%8] / 32768;
```

即用离线 profile 的 expert-to-expert 概率近似 gate unit 的预测。

### 4.2 登记阶段
计算完的 `tm` 通过 `demandrecorder->append_traffic_matrix(layer_id, region_id, tm)` 存入 `All2AllTrafficRecorder::traffic_matrix[layer][region]`。
- 第一个 iteration 原样保存；后续 iteration 会 `check_traffic_matrix()` 核对一致性（MoE 训练稳态下层间 pattern 稳定，这是 paper §5.1 的前提）。

---

## 5. 重配置触发点

模拟器里有**两个**地方会触发 `eventlist().sourceIsPending(regional_topo_manager, now)`：

1. **前向 GROUP_BY A2A**：`FFTask::cleanup()`（`ffapp.cpp:1066` 附近）
   
   - 当一条 A2A 任务 cleanup(**一个 task 跑完之后的收尾工作**) 时，如果它属于 "GROUP_BY"（MoE 层 dispatch）且是 forward，就立即请求重配对应 region 的 OCS。
   - 后续依赖该 A2A 的 ready task 会被推迟 `reconf_delay + 10 ps` 再 ready，以确保进入 RECONF 窗口期。
   - **不同 layer 的 GROUP_BY 流量模式不一样**
     每层 MoE 的 router 把 token 分给哪些 expert 是不同的 → traffic matrix 不同 → 需要的 OCS 直连对也不同。
     - 模拟器中每个 iteration 同一层 A2A 的 traffic matrix **完全一样**
   
2. **反向 Aggregate**：`FFTask::execute_compute()`（`ffapp.cpp:997` 附近）
   - 当遇到 "Aggregate" 的 TASK_BACKWARD 时（即 MoE 层 backward 阶段还没开始跑反向 A2A），就**提前**触发下一个前向 layer 的 OCS 重配。
   - 这是 paper §5.2/§5.3 中"把 reconfig 时间和 attention/expert compute 并行"的实现：反向计算 attention 的同时，后台已经把下一个 A2A 用的电路调整好。
   
   解释:
   
   正确的前向时间线：
   
   ```
   Attention fwd ─→ [Dispatch A2A fwd]       ← 这个藏不了，必须等 reconf_delay
                          ↓                     这就是 cleanup() 路径 + reconf_delay 的原因
                   Expert fwd (>100ms)        ← OCS 在这里后台切到 Combine 需要的拓扑
                          ↓
                   [Combine A2A fwd]          ← 这个的 reconfig 已经藏在 Expert fwd 里了 ✓
                          ↓
                   下一层 Attention fwd
   ```
   
   **正确的反向顺序是**：
   
   ```
   进入 layer L backward:
     Aggregate bwd compute (小，只是本地梯度)
            ↓
     [Combine A2A backward]        ← 第一个 A2A（时间上先发生）
            ↓
     Expert FFN backward (很长)
            ↓
     [Dispatch A2A backward]       ← 第二个 A2A 反向 compute 更长
            ↓
     Attention backward (很长)
            ↓
   进入 layer L-1 backward
   ```

1. **"first all-to-all" = Dispatch (GROUP_BY)**，**"second all-to-all" = Combine (AGGREGATE)** —— 按 forward 时间顺序命名，反向里"first/second"沿用这个命名不变
2. **Forward 其实能藏一半**：第二个 A2A（Combine fwd）的重配可以藏在 expert fwd compute 里
3. **Forward 藏不了的是第一个 A2A**（Dispatch fwd），这个必须等, Attention fwd 太短（只有几十 ms 甚至更短）
4. **Backward 两个都能藏**：第一个 A2A（Combine bwd，时间上先发生）藏在"later layer 的 attention compute"里，第二个 A2A（Dispatch bwd，时间上后发生）藏在"expert computation period"里

## 6. `RegionalTopoManager` 状态机

`RegionalTopoManager` 本身是 `EventSource`，整个生命周期靠 `doNextEvent()`（`mixnet_topomanager.cpp:114`）驱动：

```
TOPO_LIVE  --start_reconf()-->  TOPO_RECONF  --finish_reconf()-->  TOPO_LIVE
             ^                                                      |
             +------------------------------------------------------+
```

- `start_reconf()`（line 132）
  1. 调用 `set_regional_tcp_pause()`：遍历 `rtx_scanner->_tcps`，对该 region 内**还在走 OCS** 的 `TcpSrc`（`is_in_region()` 判定：`!_finished && !is_elec && src/dst gpu ∈ [8*start_node, 8*end_node)`）调用 `pause_flow_in_region()`。
  2. 检查 region 内队列是否为空；若非空，延迟到队列排空再触发 `_do_reconf()`（防止丢包）。

- `_do_reconf()`（line 174）
  1. `update_regional_queue_bandwidth()`：调用 `regional_topo_reconfig()` 得到新 `conn` 矩阵，然后对 region 内所有 `queues[i][j]->_bitrate` 重新赋值 `SPEED * conn[i][j]`——0 的队列相当于关闭。
  2. `update_regional_route()`（line 265）：扫描该 region 内正在跑的 TCP flow，根据新 `conn` 判定：
     - 原来走 OCS 但新 `conn=0` → 切到 EPS，`set_is_elec(true)` + `update_route(eps_paths)`。
     - 原来走 EPS 但新 `conn>0` → 切到 OCS，反之。
     - 保持 OCS/EPS 不变但路径变了 → 仅更新路径。
  3. 设置 `status = TOPO_RECONF`、`reconfig_end_time = now + reconf_delay + 1`，然后把自己 `sourceIsPending(..., now + reconf_delay)` 重新挂回 event queue。

- `finish_reconf()`（line 184）
  - `status = TOPO_LIVE`，把 queue 状态置回 READY，调用 `resume_regional_tcp_flows()` 让被 pause 的 TCP 继续发送。
  - 新来的 A2A flow（在第 3 节尾部描述的 "重配窗口内的新流"）也会在 `reconfig_end_time` 触发 `connect()`。

---

## 7. 贪心算法 `regional_topo_reconfig()`（mixnet_topomanager.cpp:370）

对应 paper 的 **Algorithm 1 ReconfigureOCS**。伪代码：

```
input:
  D  = traffic matrix (copy)              // 模拟器里直接取 demandrecorder 里最近一次 append 的 tm
  alpha                                   // 每节点最大 OCS 端口数
  conn = zeros(region_size, region_size)  // 新的连接矩阵
  avail_conn = [alpha, alpha, ..., alpha] // 每节点剩余端口

loop:
  (i, j) = argmax(D), skip diagonal        // 取当前最大 demand
  if D[i][j] == 0: break
  if avail_conn[i] > 0 and avail_conn[j] > 0:
      conn[i][j] += 1
      conn[j][i] += 1
      avail_conn[i] -= 1
      avail_conn[j] -= 1
      D[i][j] = D[i][j] / (conn[i][j] * 100)  // 模拟"多一条电路后完成时间下降"
      D[j][i] = D[i][j]
  if avail_conn[i] == 0: D[i, :] = 0          // 端口用光，行列清零
  if avail_conn[j] == 0: D[:, j] = 0

写回: topo->conn[i+start_node][j+start_node] = conn[i][j]
```

与论文 Algorithm 1 的差异：
- 论文用的是"估计完成时间"（带宽 + 链路数），模拟器用一个 `demand / (conn*100)` 的粗糙近似。目的是在迭代里把已经被满足的 (i,j) 排到后面，不是精确复现 completion time。
- 论文还要做 **NUMA-aware NIC mapping**（把 OCS 电路分配到具体的 NIC），模拟器省掉了这一层，只到节点级粒度——因为 htsim 内部对 NIC 级并不做独立排队。

---

## 8. 端到端流程（前向一个 MoE 层的视角）

1. FlexFlow 生成 DAG，导出为 `.fbuf`，其中包含 MoE 层的两个 A2A（Dispatch/GROUP_BY 和 Aggregate）。
2. `main_tcp_mixnet` 启动时构造 `Mixnet`（初始 `conn` 由 `random_connect()` 或 `weighted_connect()` 给出），创建 `All2AllTrafficRecorder` 和 `MixnetTopoManager`，`MixnetTopoManager` 为每个 region 生成一个 `RegionalTopoManager`。
3. 任务调度遇到**反向 Aggregate**（上一个 step）时，`execute_compute()` 就对 *本次前向* 的那个 region 调度一次重配置事件（隐藏在 attention/expert compute 里）。
4. `RegionalTopoManager::doNextEvent()` → `start_reconf()` → pause 在走 OCS 的 flow → 等待队列清空 → `_do_reconf()` 贪心算新 `conn` → 更新 queue bitrate → 重新路由在跑的流 → 置 `TOPO_RECONF` → 延迟 `reconf_delay` 后 `finish_reconf()` → 解除 pause、置 `TOPO_LIVE`。
5. 前向 `FFAlltoAll::start_flow()` 到来，对每一条 (src,dst) 单独查询新 `conn[src][dst]`：
   - `>0` → 走 OCS（`get_paths`，`is_elec=false`）
   - `=0` → 走 EPS（`get_eps_paths`，`is_elec=true`）
   若到达时 region 仍处于 RECONF，flow 的 `connect()` 会推迟到 `reconfig_end_time`。
6. A2A 结束后 `FFTask::cleanup()` 触发 GROUP_BY 的又一次重配（供下一个前向 layer 用），但常见路径是靠第 3 步提前完成的那次隐藏式重配。
7. `FFAlltoAll::updatetrafficmatrix()` 把本次用到的 demand 写回 `All2AllTrafficRecorder`，供下一个 iteration 做一致性检查 / 下一次重配输入。

---

## 9. 几点落差与简化

- 模拟器**没有**真正建模光路交换机的物理时延模型，直接用一个常量 `reconf_delay` 代表整段切换时间。
- 贪心的 "链路完成时间" 公式是 `demand / (conn * 100)`，跟论文式(2)(3) 的 completion time 只是定性一致。
- 没有 NUMA-aware NIC 映射（第 7 节末尾）。
- `weight_matrix` 是离线固定的 8×8 expert 概率，用来近似 gate 输出；真实系统中会基于 runtime gate output，模拟器用这种方式保证每次 iteration 的 demand 可以重复。
- AllReduce/P2P 永远强制走 EPS；只有 A2A 真正体现"动态选择"。

---

## 10. 关键代码索引

| 说明 | 文件 : 行 |
|---|---|
| Per-flow OCS/EPS 选择 | `ffapp.cpp` : 2219-2242 (FFAlltoAll::start_flow) |
| 重配窗口内 flow 延迟 connect | `ffapp.cpp` : 2296-2315 |
| AllReduce 强制 EPS | `ffapp.cpp` : 1416 附近 (FFRingAllreduce) |
| 前向 GROUP_BY 触发重配 | `ffapp.cpp` : 1066 附近 (FFTask::cleanup) |
| 反向 Aggregate 触发重配 | `ffapp.cpp` : 997 附近 (FFTask::execute_compute) |
| MoE weight matrix 应用 | `ffapp.cpp` : 395-405 |
| 流量矩阵登记 | `mixnet_topomanager.cpp` : 47 (append_traffic_matrix) |
| 状态机入口 | `mixnet_topomanager.cpp` : 114 (doNextEvent) |
| start_reconf / pause flows | `mixnet_topomanager.cpp` : 132, 245 |
| `_do_reconf` 核心序列 | `mixnet_topomanager.cpp` : 174 |
| finish_reconf | `mixnet_topomanager.cpp` : 184 |
| 路由切换 OCS↔EPS | `mixnet_topomanager.cpp` : 265 (update_regional_route) |
| 队列带宽重写 | `mixnet_topomanager.cpp` : 327 (update_regional_queue_bandwidth) |
| is_in_region 判定 | `mixnet_topomanager.cpp` : 351 |
| **贪心重配算法** | `mixnet_topomanager.cpp` : 370 (regional_topo_reconfig) |
| Mixnet::get_paths (OCS) | `mixnet.cpp` : 186 |
| Mixnet::get_eps_paths (EPS) | `mixnet.cpp` : 246 |
| init_network 用 conn 定 bitrate | `mixnet.cpp` : 115 |

### 举个 MoE 层的例子，DAG 大致长这样（简化）

```
[Attention_fwd on GPU0..N]                      type=FORWARD, runtime=20ms
           ↓ next_tasks
[Router_gate on GPU0..N]                        type=FORWARD, runtime=1ms
           ↓
[Dispatch A2A]                                  type=ALLTOALL, info="GROUP_BY forward"
  from_node_ids=[0..127], to_node_ids=[0..127]
  operator_sizes={(0,3):4MB, (0,7):2MB, ...}   ← 这就是 traffic matrix
           ↓
[Expert_fwd on each expert GPU]                 type=FORWARD, runtime=100ms
           ↓
[Combine A2A]                                   type=ALLTOALL, info="AGGREGATE forward"
           ↓
[Attention_fwd of layer L+1]                    ...
  
... 一路走到最后一层 ...

[Loss compute]
           ↓
[Attention_bwd of last layer]                   type=BACKWARD
           ↓
[Aggregate_bwd compute]                         type=BACKWARD, name="Aggregate"  ← 就是代码 :998 判断的那个！
           ↓
[Combine A2A bwd]                               type=ALLTOALL, info="AGGREGATE backward"
           ↓
[Expert_bwd]                                    type=BACKWARD
           ↓
[Dispatch A2A bwd]                              type=ALLTOALL, info="GROUP_BY backward"
           ↓
... 层层往上 ...

[DP AllReduce for gradients]                    type=DP_ALLREDUCE
           ↓
[Weight update]                                 type=UPDATE
```

## 1. OCS vs EPS 的本质区别

| 维度           | EPS（电交换，如传统 Clos）     | OCS（光交换，MEMS/硅光）               |
| -------------- | ------------------------------ | -------------------------------------- |
| 交换粒度       | 包（per-packet）               | 电路（port-to-port 直连）              |
| 端口带宽       | 固定，SerDes 限制（400G/800G） | 光域透明，**带宽随光模块升级免费增长** |
| 跳数           | Leaf→Spine→Leaf，多跳，有排队  | **1 跳直连**，无排队、无 buffer        |
| 每 bit 成本    | 高（ASIC + SerDes）            | 低 2~5×                                |
| 能耗           | 高                             | 低（MEMS 镜片几乎不耗电）              |
| 重配时间       | 无（天然 any-to-any）          | ms 级（MEMS）或 μs 级（硅光）          |
| 通信模式适应性 | **任意 pattern**               | **只适合稀疏/可预测 pattern**          |

一句话：**OCS 便宜、带宽大、延迟低，但它"笨"——不能像包交换那样应付任意突发和多对一**。

## 2. 为什么 A2A (EP) 正好是 OCS 的"甜点"

MoE 的 A2A 有几个关键性质，让它和 OCS 天生契合：

### ① Pattern **稀疏 + 强 locality**

MoE 的 gate unit 会把 token 路由到 top-k 个 expert（通常 k=1 或 2），加上 load-balancing loss 的约束，导致：

- 一个 GPU 的 token 只会发给**少数**几个 expert（不是均匀打到所有 N-1 个节点）
- 每个 expert 在全集群里只在**少数**几个 GPU 上有副本

结果是 N×N 的 traffic matrix **非常稀疏**。paper §5.1 的测量显示 production MoE 里 A2A demand 的 top-10% 元素占了大部分流量。

👉 稀疏 ⇒ 不需要 full mesh，只需要给 "热" 的 (i,j) 对建 OCS 直连就够 —— 正好是 OCS 能提供的。

### ② Pattern **可预测 + 稳定**

- 层间：训练稳态下，每个 MoE 层的 routing 分布相对稳定（这也是 paper Fig. 4 的关键观察）
- 迭代间：同一层在连续 iteration 里 pattern 基本一致

👉 可预测 ⇒ 可以**提前**基于 gate output（甚至历史均值）算好 demand，有时间去重配 OCS。

### ③ 数据量**巨大**

MoE A2A 一次传 **数百 MB ~ 数 GB**（和模型规模、batch、sequence length 成正比）。这种粗粒度传输：

- 重配开销（ms 级）相对传输时间（几十~几百 ms）可以摊薄
- 对 OCS "一次建链、长时间用" 的模式非常友好

### ④ **带宽瓶颈**集中在这里

在大模型训练里，A2A 已经成为 critical path 的主要占比（paper 里报告占到 step time 的 30%~60%）。谁先把 A2A 做快，谁就能省钱。

## 3. 为什么 AllReduce、P2P **不**走 OCS

这正是 paper §5.3 的设计理由，也是模拟器里强制 `get_eps_paths()` 的原因：

- **DP AllReduce**（Ring / Tree）：
  - pattern 是**环**或**树**，每个节点都要和所有人参与 —— 相当于 full mesh demand
  - 稀疏度低，OCS 优势发挥不出来
  - 对 tail latency 敏感，EPS 的 per-packet 多路径 ECMP 反而更稳
  - 改用 hierarchical AllReduce（region 内 NVLink + region 间 EPS）已经足够好
- **PP 的 P2P**：
  - 流量很小（只有 activation，不像 A2A 那样 GB 级别）
  - 两个相邻 stage 之间的通信在整个 step 里占比低
  - 用 OCS 建链的重配开销反而盖过收益
- **TP**：
  - 限在单节点内走 NVLink / NVSwitch，根本不出节点，和 fabric 无关

所以 MixNet 的设计哲学是：**用便宜的 OCS 去扛最大头的 A2A，把剩下的杂项流量留给灵活但贵的 EPS**。这是一种**按通信 pattern 做流量分层**的思路。

## 4. 具体收益（paper §7 数值，记忆中）

- OCS 单端口带宽可以做到 800G+，且随光模块升级 0 成本扩容
- MixNet 对比纯 EPS fat-tree：**相同预算下 A2A 带宽提升 1.5~2×**，或**相同性能下 fabric 成本降 30~50%**
- 重配开销 (~ms) 能被 attention/expert 计算完全 hide 掉，对 step time 几乎零影响

## 5. 一句话总结

> **OCS 提供"廉价的大带宽直连"，A2A (EP) 提供"稀疏且可预测的需求"**——两者匹配，才让 runtime 重配 OCS 这种 "看起来很贵" 的操作在 MoE 训练场景下变得值得。换成任意 pattern 的通信或任意模型，这个设计都不成立。

这也解释了为什么 MixNet 不是"all-optical"，而是**混合** fabric：OCS 的"笨"必须用 EPS 的"灵活"来兜底，才能在真实训练 workload 下 work。
