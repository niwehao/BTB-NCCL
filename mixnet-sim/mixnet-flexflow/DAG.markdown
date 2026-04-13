# 结论：mixnet-flexflow 在生成 DAG 时 **并不选择 NCCL 算法**

答案很明确：**导出给 mixnet-htsim 的任务图里完全没有算法信息**。算法选择是 htsim 那一侧单独做的事。

## 1. 任务图导出端：只写"语义"，不写"算法"

`substitution.cc:1193` 的 `Graph::get_taskgraph_flatbuf()` 是 mixnet-flexflow 产生 `.fbuf` 的唯一入口。对 AllReduce 相关任务，它只做一件事：按 `SimTask::type` 做一对一枚举映射 (substitution.cc:1282-1303)：

```cpp
case SimTask::TASK_SUB_ALLREDUCE:  → SimTaskType_TASK_SUB_ALLREDUCE;
case SimTask::TASK_DP_ALLREDUCE:   → SimTaskType_TASK_DP_ALLREDUCE;
case SimTask::TASK_TP_ALLREDUCE:   → SimTaskType_TASK_TP_ALLREDUCE;
case SimTask::TASK_ALLREDUCE:      → SimTaskType_TASK_ALLREDUCE;
case SimTask::TASK_REDUCESCATTER:  → SimTaskType_TASK_REDUCESCATTER;
case SimTask::TASK_ALLGATHER:      → SimTaskType_TASK_ALLGATHER;
case SimTask::TASK_ALLTOALL:       → SimTaskType_TASK_ALLTOALL;
```

然后写入 `to_node_ids / from_node_ids / xfer_size / info / micro_batch_id / layer_id`。**没有任何字段表达"用 ring 还是 PS"**。

## 2. 任务图构建端：只说"谁和谁做 AllReduce，多大"

上游 `task_manager->new_allreduce_task(...)` 的两个重载 (simulator.cc:1931, 2067, 2079) 只记录：

- `to_node_ids`：参与 AR 的 device id 列表
- `xfer_size`：总传输量（weight / output 的 fp16 分片）
- `info`：字符串标签 (`"forward tp allreduce"` / `"backward dp allreduce"` 等)
- `layer_id / micro_batch_id`：训练流程位置

调用点也只关心"该在哪里插入一次 AR"：

- **DP AllReduce** (substitution.cc:1578-1587)：对每个 TP rank 构造一组 `dp_device_ids`（沿 DP 维），在反向传播后挂一个 `TASK_ALLREDUCE`，大小 = `weights_memory / (2*tp_degree_local)`。
- **TP AllReduce** (substitution.cc:1608-1625)：仅在 `OP_MULTIHEAD_ATTENTION` / `OP_LINEAR` 且 `tp_degree_local > 1` 时插入前向 + 反向各一次，大小 = `outputs_memory / (2*tp_degree_local)`。

这两处生成的是**逻辑 AR 节点**——"一组 device 之间要做一次 allreduce，传 X 字节"——不关心底层怎么实现。

## 3. FlexFlow 内部自己的 simulator 有算法分支，但那是另一条路

`simulator.cc:1741 expand_allreduce()` 确实有算法分支，但它是 **FlexFlow 搜索 parallel strategy 时的 cost model**，不参与 `.fbuf` 导出：

```cpp
#ifdef FF_USE_NCCL
  // Ring: 把 AR 展开成 n_participants 段链式 comm，每段 xfer_size = 2*(N-1)/N * total
  ...
#else
  // Parameter server: gather + scatter
  ...
#endif
```

- 这是 **编译期宏**，不是运行时参数。
- 它只在 `LogicalTaskgraphBasedSimulator::simulate_...()` 内部调度 `ready_queue` 时被调用 (simulator.cc:1457)，把 AR "展开"成若干 `TASK_NOMINAL_COMM`，供 FlexFlow 内部估算一个 `sim_time` 浮点数，用于图搜索/策略选择。
- 对应的 flatbuf 导出 `LogicalTaskgraphBasedSimulator::get_taskgraph_flatbuf()` (simulator.cc:1907) 和 `TaskgraphSimulator::get_taskgraph_flatbuf()` (simulator.cc:1926) 都是 `// TODO()` **空函数**。
- 实际被 `searlize_logical_taskgraph` (substitution.cc:1465) 调的是 **`Graph::get_taskgraph_flatbuf`**，那里没有 `expand_allreduce`，也不会看 `FF_USE_NCCL`。

也就是说，`expand_allreduce` 的 ring/PS 分支 **只影响 FlexFlow 自己找 strategy 时的内部估算**，而**落盘的 `.fbuf` 里依然是未展开的逻辑 AR 节点**。

## 4. 端到端流水线

```
mixnet-flexflow
   │
   ├─ 图搜索阶段 (内部成本模型)
   │     LogicalTaskgraphBasedSimulator::simulate...
   │       └── expand_allreduce()   ← 这里才看 FF_USE_NCCL (ring vs PS)
   │           用来给一个 float sim_time，供 strategy search 打分
   │
   └─ 导出阶段
         Graph::searlize_logical_taskgraph
           └── Graph::get_taskgraph_flatbuf
                 └── 只写 TASK_*ALLREDUCE 节点 (node 列表 + size + info)
                 NO algorithm info
                         │
                         ▼
                   taskgraph.fbuf
                         │
                         ▼
mixnet-htsim (main_tcp_flat.cpp:186 -ar ring|ps|dps)
   └── FFApplication::load_taskgraph_flatbuf
         └── 对每个 TASK_*ALLREDUCE 按 allreduce_strategy 实例化
             FFRingAllreduce / FFPSAllreduce / FFDPSAllreduce
```

## 一句话总结

**mixnet-flexflow 生成 DAG 时不做 NCCL 算法选择** —— 它只标注"这里要一次 AllReduce，参与者是谁，大小多少"。真正的算法选择发生在 mixnet-htsim 加载任务图时，通过 `-ar ring|ps|dps` 命令行参数决定实例化哪种子类。这也是为什么 `main_tcp_mixnet.cpp` 写死 `FF_DEFAULT_AR`——mixnet 实验默认就是 ring，和 flatbuf 里的内容无关。

如果想让 FlexFlow 侧的 strategy search 和 htsim 侧的实际仿真用同一种算法，你需要**手动**确保：编译 flexflow 时 `FF_USE_NCCL` 开关和启动 htsim 时的 `-ar` 参数保持一致（`FF_USE_NCCL=on ↔ -ar ring`，`off ↔ -ar ps`）。目前这是两个互不通气的配置。

# Ring的选择

## 1. 真正一刀切的部分

| 集合操作                                   | 可选实现                          | 结论                        |
| ------------------------------------------ | --------------------------------- | --------------------------- |
| `ALLREDUCE` / `DP_AR` / `TP_AR` / `SUB_AR` | ring / ps / dps / newring         | **全局一种**，由 `-ar` 决定 |
| `REDUCESCATTER`                            | 只有 `FFReduceScatter` (ring)     | 唯一实现                    |
| `ALLGATHER`                                | 只有 `FFAllGather` (ring)         | 唯一实现                    |
| `ALLTOALL`                                 | 只有 `FFAlltoAll` (pairwise flat) | 唯一实现                    |
| `P2P`                                      | 只有 `FFP2P`                      | 唯一实现                    |

所以：**一次仿真里，所有 AR 的逻辑拓扑是同一种模式**，所有 AllToAll 都是同一种模式，依此类推。不存在"这个 AR 用 ring，那个 AR 用 tree"、"小 AR 切 ring 大 AR 切 tree"这类 size-aware 或 topology-aware 的切换。

这和真实 NCCL 有显著差距。真实 NCCL 会按 message size、rank 数、网络拓扑在 **Ring / Double-Binary-Tree / CollNet** 之间自动切换，并且 intra-node 和 inter-node 会分层（2-level 甚至 3-level）；`mixnet-htsim` 里都没有。

## 2. 还有变化的部分（别全抹平）

"同一种算法"不等于"同一张物理图"：

### 2.1 每个 AR 的 `node_group` 不同

ring 算法是固定的，但 ring **上的成员**来自 flatbuf 里每个 AR 任务的 `to_node_ids`：

- 一个 TP-AR 的 node_group 可能是 `{g0,g1,g2,g3}` (某个 attention/linear 的 TP 维)
- 一个 DP-AR 的 node_group 可能是 `{g0, g8, g16, ..., g(8*(dp-1))}` (跨 DP rank)
- 不同 layer / micro-batch 的 AR，node_group 集合也不同

所以虽然**算法一样**，每个 AR 任务**实际走的 ring 是不同的一组节点**。

### 2.2 每条 flow 的网络路径仍是独立选的

在 `start_flow()` 里：

```cpp
srcpaths = topology->get_paths(src, dst);   // 或 get_eps_paths
choice = rand() % srcpaths->size();
```

多路径拓扑下，每条 flow **随机挑一条 ECMP 路径**，两个 ring 段可能走完全不同的物理链路。这就是为什么整个仿真还有意义——拥塞模式不是复制的。

### 2.3 AllToAll 的 per-pair size 来自 MoE 权重矩阵

`FFAlltoAll` 虽然只有一个实现，但每次任务的 `operator_sizes[(fn,tn)]` 是按 `weight_matrix` 生成的 (ffapp.cpp:395-405)：

```cpp
operator_sizes[(fn,tn)] = total * weight_matrix[fn_exp%8][tn_exp%8] / 32768;
```

不同 expert-to-expert 对的流量大小不一样，所以整体流量矩阵是**不均匀**的——这是 MoE 仿真关心的点。

### 2.4 Mixnet 场景下 OCS/EPS 的选择是按 flow 的

每条 AllToAll flow 在 `start_flow()` 里查 `topology->conn[src_node][dst_node]` 决定走光 (OCS) 还是电 (EPS) (ffapp.cpp:2223)。AllReduce / P2P 固定走 EPS。所以**同一个 AllToAll 任务内部**，不同 pair 可能走不同物理平面。

## 3. 所以正确的理解是

> 一次仿真里，**算法/通信模式**被每类操作的**唯一实现**（或 `-ar` 选中的那个实现）锁死；变化只发生在：
>
> - **谁参与**（每个任务的 node_group / from-to 集合）
> - **流量大小**（MoE 权重矩阵）
> - **物理路径**（ECMP 随机选路、Mixnet OCS vs EPS）

从算法视角看，你说的没错——**所有 AR 共享同一个拓扑逻辑**，所有 AllToAll 共享同一个拓扑逻辑；这个仿真器不做真实 NCCL 的"按 size / 按层级"自适应算法切换。

## 4. 这是一个需要注意的限制

如果你的研究关心的是：

- **算法选择对训练时间的影响**（ring vs tree vs hierarchical）
- **小消息 latency-bound vs 大消息 BW-bound 的 crossover**
- **intra-node NVLink + inter-node IB 的两级 ring**
- **不同算法对 OCS reconfig 机会窗口的影响**

那这套框架目前**模拟不了**。想做这类对比，要么：

1. 跑多次仿真，每次换 `-ar` 得不同数据点；
2. 扩展 `FFRingAllreduce` 加一个"按 message size 阈值切换算法"的上层 wrapper；
3. 或者实现一个 `FFTreeAllreduce` / `FFHierRingAllreduce`，然后在 `load_taskgraph_flatbuf` 里按规则选类。

但只要你承认"仿真关心的是 DCTCP 拥塞 + OCS 重配的网络行为，而非 NCCL 算法细节"，现在这个"一刀切"是合理的简化。

# 如何生成

找到完整链路了。**FlexFlow 生成 DAG 的核心就是：你用 C++ API 写一个"模型定义 + 并行配置"的程序，`ff.compile_taskgraph()` 就会帮你把完整的 DAG 吐出来。** 具体分三步。

## Step 1：命令行参数决定模型尺寸和并行策略

看 `moe.cc:25-73` 的 `parse_input_args()`，它接收这些参数：

```cpp
// 模型架构参数（写入 TransformerConfig）
--num-layers N           // MoE block 层数
--embedding-size E       // 词嵌入维度
--hidden-size H          // 隐藏层宽度
--expert-hidden-size EH  // expert FFN 内部维度
--num-heads NH           // attention heads
--sequence-length S      // 序列长度

// MoE 特有参数（写入 MoeConfig）
--expnum E               // 每层多少个 expert
--topk K                 // 每个 token 激活几个 expert

// 训练参数
--batchsize B
--microbatchsize MB
```

另外 `FFConfig` 还有 **并行维度**：

```cpp
ffConfig.train_dp        // Data Parallel degree
ffConfig.train_tp        // Tensor Parallel degree
ffConfig.train_pp        // Pipeline Parallel degree
```

所以**模型规模 + 并行策略完全是参数化的**。比如想模拟 Mixtral 8×7B（paper Table 1），就传：

```
--num-layers 32 --num-heads ? --hidden-size ? --expnum 8 --topk 2 --sequence-length 4096
ffConfig.train_dp=1, train_tp=4, train_pp=4, train_ep=8
```

## Step 2：C++ API 符号化地描述模型结构

看 `moe.cc:141-188` 的 `create_moe_decoder()`：

```cpp
Tensor create_moe_decoder(FFModel *model, ...) {
  Tensor x = input;
  for (int i = 0; i < tfConfig->num_layers; i++) {
    // === Attention + residual + LN ===
    x = model->layer_norm(
          model->add(
            model->multihead_attention(x, x, x,
                                       tfConfig->hidden_size,
                                       tfConfig->num_heads,
                                       ... ,
                                       ffConfig->train_dp,
                                       ffConfig->train_tp,
                                       ffConfig->train_pp,
                                       moeConfig->num_exp),
            x),
          axes, true, 1e-05);

    // === MoE FFN + residual + LN ===
    x = model->layer_norm(
          model->add(
            model->moe(x,
                       moeConfig->num_exp,
                       moeConfig->num_select,
                       tfConfig->expert_hidden_size,
                       tfConfig->hidden_size,
                       moeConfig->alpha,
                       moeConfig->lambda),
            x),
          axes, true, 1e-05);
  }
  x = model->dense(x, tfConfig->hidden_size, AC_MODE_RELU, false);
  return x;
}
```

几个要点：

1. **模型是一个循环展开**：你显式写一个 for 循环跑 `num_layers` 次，每次调 `multihead_attention` + `moe`。所以 num_layers 不是"让 FlexFlow 智能决定"的，是你在代码里亲手展开的。
2. **`model->moe(...)` 是一个复合 op**：它内部会展开成 `group_by` + `expert FFN` + `aggregate` 三个子 op（对应 `src/ops/group_by.cc`、`aggregate.cc`）。这就是为什么 DAG 里看得到 `GROUP_BY` 和 `AGGREGATE` 的名字。
3. **每个符号化调用只是"声明一个层"**，不真正算东西。这一步只是在内部建了一张**计算图**（Layer graph）。

## Step 3：`compile_taskgraph()` 把模型图编译成 DAG

看 `model.cc:2849-2877`：

```cpp
void FFModel::compile_taskgraph(CompMode comp_mode) {
  ...
  create_operators_from_layers();          // ① Layer → Operator 展开
  
  TaskLauncher launcher(GRAPH_OPTIMIZE_TASK_ID, ...);
  Future future = runtime->execute_task(ctx, launcher);
  
  PCG::GraphOptimalViewSerialized ret =    // ② 图搜索 + 策略优化
      future.get_result<...>();
}
```

这一步 FlexFlow 做了几件很重的事：

**① 把 Layer 展开成 Operator**

- 复合 op（比如 `moe`）被拆成 `group_by` + `experts` + `aggregate` 等原子 op
- 为每个 forward op 自动生成对应的 **backward op**
- 插入 **权重 gradient update** op

**② 根据并行策略切图**

- 按 DP 把 batch 维度分片
- 按 TP 把 attention heads / hidden size 分片
- 按 PP 把 layer 切成 stage
- 按 EP 把 experts 分到不同 GPU
- 在切割边界自动插入 **collective comm op**：A2A、AllReduce、ReduceScatter、AllGather 等

**③ 运行图优化搜索（`GRAPH_OPTIMIZE_TASK_ID`）**

- FlexFlow 的招牌能力：用 MCMC/beam search 等算法找 **"最优的并行策略组合"**
- 输出一张 `best_graph` + 每个 op 的 `MachineView`（即"放在哪组 GPU 上")
- `moe.cc:244` 里 `ff.compile_taskgraph()` 就是触发这套流程

**④ 最后**：遍历 `best_graph`，把每个 op 的前向/后向任务、数据量（根据切分后的 tensor shape 算出来）、预估 runtime、依赖边（由 operator 间的 tensor producer-consumer 关系决定）序列化成 flatbuffer 文件（`.flatbuf`）。模拟器再读取这个文件。

## 所以回答你的问题

> **"FlexFlow 离线生成是怎么生成，规定模型的参数和架构吗？"**

**两边都要**，分工是：

| 谁规定                         | 规定什么                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| **命令行参数 / Config 结构体** | 数值类的东西：`num_layers`、`hidden_size`、`num_exp`、`topk`、`batchsize`、`DP/TP/PP/EP degree` |
| **C++ 代码里的符号化调用**     | 结构类的东西：先 attention 还是先 MoE、是否加 residual、是否 LayerNorm、用几层循环 |
| **`compile_taskgraph()` 内部** | 自动推导的东西：每个 op 的 backward、切分后的 tensor 形状、集合通信 op 插入位置、op 放在哪些 GPU、每个 task 的 runtime 估计 |

换句话说：

- **你不写 forward/backward 的细节**（只写 forward 的 symbolic call，backward 让 FlexFlow 自动 autodiff 生成）
- **你不写通信的细节**（A2A、AllReduce、P2P 这些由切分策略自动插入）
- **你不写 task 的 runtime**（FlexFlow 用 cost model 估出来）
- **你要写的**：模型骨架 + 尺寸超参 + 并行度

然后 `compile_taskgraph()` 吐出一个 `.flatbuf` 文件，这个文件就是模拟器读进来的那张 DAG。整个流程是**纯离线**的 —— 不需要真的跑训练、不需要真的有 GPU，就能生成出描述完整训练 iteration 的任务图。

## 题外话：为什么要用 FlexFlow 而不是手写 DAG

因为一个 MoE 训练 iteration 的 DAG 轻松上万个 task（几十层 × 每层多个 op × 各种 comm × 多个 micro-batch）。手写不现实。FlexFlow 已经有：

1. **完整的 op 库**（`src/ops/`下一堆 `.cc`）
2. **成熟的 autograd 机制**（forward 自动推 backward）
3. **并行策略搜索**（能自动找 TP/DP/PP 组合）
4. **runtime cost model**（能估每个 op 在某个 GPU 上跑多久）

把这些现成的能力用起来，只要改改 `compile_taskgraph` 的输出路径让它吐 flatbuf 给模拟器吃，就得到了"能评估任意 MoE 模型 × 任意并行配置 × 任意网络拓扑"的完整仿真管线。这比直接在真实集群上跑实验便宜几个数量级，而且能测各种光交换机参数扫描。