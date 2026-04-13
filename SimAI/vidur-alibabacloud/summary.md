# Vidur 的工作原理 + 推理服务入门

我会**从零开始**讲，假设你不了解推理服务。

---

## 第一部分：什么是 LLM 推理服务

### 1. 什么是"推理"

**训练**：教模型怎么说话（一次性的，做完就停）
**推理（inference）**：让训练好的模型回答问题（不停地服务用户）

当你在 ChatGPT 网页输入问题、它流式吐出回答——背后就是 GPU 集群在做"推理服务"。

### 2. 推理服务面临的问题

不像训练只跑一次，推理服务要**同时处理成千上万个用户的请求**：

```
用户 A: "今天天气怎么样?"          ← 短输入、短输出
用户 B: "帮我写一篇 5000 字论文"    ← 短输入、超长输出
用户 C: "总结这篇 10000 字的文章"   ← 超长输入、短输出
用户 D: "你好"                     ← 极短
...同时来 1000 个请求
```

服务系统要决定：

- 哪个请求先处理？
- 怎么 batch 在一起省 GPU 时间？
- GPU 内存满了怎么办？
- 有的请求要等很久（队头阻塞），怎么避免？

这些"决定"就是**调度器（scheduler）**的工作。

### 3. 推理的两个阶段：Prefill 和 Decode

```
用户输入: "今天天气" (4 个 token)
模型输出: "怎么样啊很好的" (一个一个吐)

阶段 1: Prefill（预填充）
  ─────────────────────────
  把 "今天天气" 4 个 token **一次性** 喂进模型
  模型计算这 4 个 token 的所有 attention
  生成第 1 个输出 token "怎"
  特点: 计算量大、并行度高、时间长 ← 决定 TTFT

阶段 2: Decode（解码）
  ─────────────────────────
  把 "怎" 喂进模型 → 生成 "么"
  把 "么" 喂进模型 → 生成 "样"
  把 "样" 喂进模型 → 生成 "啊"
  ...一个一个生成
  特点: 每次只算 1 个 token、内存带宽密集、时间短但次数多 ← 决定 TBT
```

**关键差异**：

- **Prefill**：一次性吃 N 个 token → 计算瓶颈（GPU 算力满）
- **Decode**：每次吃 1 个 token → 内存瓶颈（HBM 带宽满）

---

## 第二部分：Vidur 怎么"模拟"这一切

### Vidur 不是真的跑模型！

**关键理解**：Vidur 不加载任何模型权重、不做任何 CUDA 计算、不发任何网络包。它是一个**事件驱动的离散时间模拟器**——所有"发生"的事情都只是数字游戏。

```
真实世界：              Vidur 模拟：
GPU 算了 200ms     →   把 simulation_time += 0.2 秒
NIC 传了 50ms      →   把 simulation_time += 0.05 秒
请求完成           →   写一行 CSV
```

### Vidur 需要回答两个问题

**问题 1**："这次 forward 要多久？"
**问题 2**："然后该做什么？"

回答问题 1 → **Execution Time Predictor**
回答问题 2 → **Scheduler + Event Queue**

---

## 第三部分：Mode 1（纯 Vidur）怎么不用 trace 和网络模拟

### 核心秘诀：**Random Forest 预测器**

微软原版 Vidur 在真实 GPU 上跑了**几万次** profiling 数据：

```
在 A100 上跑 Llama-7B, TP=1, batch=1, seq_len=128 → 实测 23ms #每个 profiling 数据点对应一次 forward iteration（GPU 跑一次模型 forward 的时间），不是整个请求。 1.实测的是：一个 Replica一次  forward 处理 prefill_chunk_size 个新 prompt token 的时间。
2.一个 Replica一次  forward 处理 为 batch_size 个请求各生成 1 个新 token 的时间。(只关心 Replica 这个抽象单位的耗时)
在 A100 上跑 Llama-7B, TP=1, batch=4, seq_len=512 → 实测 87ms
在 A100 上跑 Llama-7B, TP=2, batch=8, seq_len=1024 → 实测 145ms
... (几万条这样的数据) 
```

把这些数据训练成一个 **Random Forest 机器学习模型**：

```
输入: (model, tp_size, batch_size, total_tokens, kv_cache_size, ...)
输出: 这次 forward 大概多少 ms
```

这个 RF 模型很小（几 MB），加载到内存里，**每次预测只要几微秒**。

```python
# 伪代码
def predict_forward_time(batch):
    features = [batch.size, batch.total_tokens, tp_size, model_dim, ...]
    return random_forest_model.predict(features)  # 纯 ML 推理，没 GPU
```

### 所以"纯 Vidur"的工作流程

```
1. Vidur 创建一个虚拟集群（4 个 replica）replica是一组协同 GPU，它们共同持有完整模型，能独立处理推理请求
2. Vidur 生成 100 个虚拟请求（按 Poisson 分布到达）
3. 调度器把请求 batching 起来
4. 对每个 batch：
   ├─ 调用 RF 模型: "这个 batch 要多少 ms?" → 23ms
   ├─ 把 simulation_time += 23ms
   └─ 把 batch 里的请求标记为"完成一个 token"
5. Decode 阶段循环上面的步骤
6. 全部请求完成 → 输出 CSV
```

**全程没有 trace 文件、没有 ns-3、没有 GPU**。Random Forest 模型就是预先"封装好的 GPU 时间知识"。

### 那 Mode 2 (AICB) 有什么不同？

| Mode 1 (vidur backend)    | Mode 2 (aicb backend)            |
| ------------------------- | -------------------------------- |
| 用预训练 RF 模型          | 用 AICB 实时计算 / AIOB 算子时间 |
| 仅支持 RF 数据集里的模型  | 支持 DeepSeek/Qwen3 等新模型     |
| 不支持 EP                 | 支持 EP（MoE 关键）              |
| 准确度依赖 profiling 数据 | 更精确，有真实算子模型           |

---

## 第四部分：调度器在 Vidur 里是什么

### 调度器要解决什么问题

假设 GPU 已经在跑一个 batch，这时来了 5 个新请求，调度器要决定：

1. **等当前 batch 跑完，再处理新的吗？** → orca 风格
2. **下个迭代立即把新请求加进来吗？** → vLLM / Sarathi 风格
3. **强制中断当前的，先跑紧急的吗？** → preemption
4. **prefill 和 decode 混在一起跑吗？** → split_wise 不混

不同调度器有不同策略。Vidur 把它们都模拟出来，让你对比效果。

### 调度器的两个层级

```
        ┌─────────────────────────────────┐
        │ Global Scheduler                │
        │ (集群级，决定请求送到哪个 replica) │
        └────────────┬────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Replica0│ │Replica1│ │Replica2│
    │ ┌────┐ │ │ ┌────┐ │ │ ┌────┐ │
    │ │ Re │ │ │ │ Re │ │ │ │ Re │ │  ← Replica Scheduler
    │ │plic│ │ │ │plic│ │ │ │plic│ │  (单 replica 内决定哪些请求 batch 在一起)
    │ │a S │ │ │ │a S │ │ │ │a S │ │
    │ │ched│ │ │ │ched│ │ │ │ched│ │
    │ └────┘ │ │ └────┘ │ │ └────┘ │
    └────────┘ └────────┘ └────────┘
```

| 层级              | 决策                           | Vidur 参数                          |
| ----------------- | ------------------------------ | ----------------------------------- |
| **Global**  | 路由：新请求送到哪台机器？     | `--global_scheduler_config_type`  |
| **Replica** | Batching：本机的请求怎么打包？ | `--replica_scheduler_config_type` |

### 几种典型调度器

**1. round_robin（最简单）**

```
请求 1 → Replica 0
请求 2 → Replica 1
请求 3 → Replica 2
请求 4 → Replica 0
... 轮询
```

**2. SplitWise（PD 解耦）**

```
请求来了：
  ├─ Prefill 阶段送给 P 节点（计算密集型 GPU）
  └─ Prefill 完成 → KV cache 传给 D 节点 → Decode 在那边跑
```

##### detail

step1:把长 Prefill 切成小块（Chunked Prefill）

```
原始: 1000 token 的 Prefill，一次跑完 → 200ms

Sarathi: 切成 8 块 × 125 token，每块单独 forward
  chunk 1: 25ms
  chunk 2: 25ms
  ...
  chunk 8: 25ms
  总时间: 200ms (略长一点点，因为每次启动开销)
```

切完之后，每一块的耗时和一次 Decode iteration 差不多。

step2:每块 Prefill **混进** Decode batch 一起跑

```
iteration 1: [prefill_chunk_1, decode_R1, decode_R2, decode_R3, decode_R4]
             ↑                    ↑
             125 prefill tokens   4 个 decode 请求各 1 个 token
           
             总共算: 125 + 4 = 129 tokens
             耗时: ~30ms
```

 **关键** ：这个混合 batch 一次 forward 同时完成了：

* 125 个 prefill token 的计算
* 4 个用户的下一个 decode token 生成

**3. Sarathi（chunked prefill + 连续 batching）**

```
长 prefill 切成小块，混进 decode batch 一起跑
避免长 prefill 阻塞 decode 用户
```

**4. vLLM（PagedAttention + 迭代级调度）**

```
每个迭代结束后重新决定 batch 内容
KV cache 用页表管理避免碎片化
```

PagedAttention 在 Vidur 里怎么模拟

**Vidur 模拟 vLLM 调度器时（**`--replica_scheduler_config_type vllm`），PagedAttention 主要影响 **KV cache 内存管理**：

```
# 概念性代码
class VLLMReplicaScheduler:
    def __init__(self):
        self.total_blocks = total_gpu_memory / block_size
        self.free_blocks = self.total_blocks
        self.block_table = {}  # request_id → list of block ids
    
    def can_accept(self, request):
        # 检查是否有足够的 free block
        needed = ceil(request.prefill_tokens / block_size)
        return self.free_blocks >= needed
    
    def add_request(self, request):
        needed = ceil(request.prefill_tokens / block_size)
        self.free_blocks -= needed
        self.block_table[request.id] = list(range(needed))
    
    def step(self):
        # decode 一步，可能要给某些请求分配新 block
        for request in active_requests:
            if request.needs_new_block():
                if self.free_blocks > 0:
                    self.free_blocks -= 1
                else:
                    # OOM → preempt 一些请求
                    self.preempt_some_requests()
```

**Vidur 不真的做矩阵运算，但**精确模拟内存分配/回收**，所以能反映 PagedAttention 带来的"更高 batch size 容纳能力"和"preemption 行为"。

随机森林模拟时间:

但是除了时间维度以外,我们要考虑：资源管理（Spatial）

```
GPU 显存: 80GB
当前用了: 60GB
剩余:    20GB
能装下: 200 个 KV cache block
```

→  **由 PagedAttention 模拟驱动** ，回答"现在能干什么"。

---

## 第五部分：解释 entities/ 里的"仿真实体"

这些类是 Vidur 内部的**数据结构**——把"集群、请求、计算"这些抽象概念变成 Python 对象。我一个个解释。

### `Cluster`（集群）

**真实对应**：你的整个机房
**Vidur 里**：包含多个 Replica 的容器对象

```python
cluster = Cluster(num_replicas=4)
# cluster.replicas = [Replica0, Replica1, Replica2, Replica3]
```

---

### `Replica`（副本）

**真实对应**：一组协同工作的 GPU（比如 8 张 A100 + TP=4 + PP=2）
**Vidur 里**：一个能"端到端跑模型"的最小单位

DP=4 意思就是有 4 个 Replica，每个 Replica 内部又可以有自己的 TP/PP 配置。

```python
replica = Replica(
    model="Llama-3-8B",
    tensor_parallel_size=4,
    num_pipeline_stages=2,
)
# 这一个 replica 实际占 4×2 = 8 张 GPU
```

---

### `Request`（请求）

**真实对应**：一个用户的提问"今天天气怎么样"
**Vidur 里**：

```python
request = Request(
    arrived_at=10.5,          # 第 10.5 秒到达系统
    num_prefill_tokens=128,   # 输入是 128 个 token
    num_decode_tokens=512,    # 期望生成 512 个 token
)
```

**DAG 属性 (`nx.DiGraph`)** 是什么？

DAG = Directed Acyclic Graph（有向无环图），用 networkx 库表达。

普通的请求是线性的：

```
prefill → decode → decode → decode → ... → 完成
```

但**有 PD 解耦**或**复杂场景**时，一个请求要做的事情形成一张图：

```
       ┌────────────┐
       │ Prefill 在 │
       │ P 节点跑   │
       └─────┬──────┘
             │
       ┌─────▼──────┐
       │ KV cache   │  ← 这是 Flow（通信）
       │ 传到 D 节点 │
       └─────┬──────┘
             │
       ┌─────▼──────┐
       │ Decode 在  │
       │ D 节点跑   │
       └────────────┘
```

这张图里有 3 个**节点**（2 个 Task + 1 个 Flow）和 2 条**边**（依赖关系）。`nx.DiGraph` 就是表达这种依赖结构的工具。Vidur 调度器靠它知道"什么时候该做什么"。

---

### `Batch`（批次）

**真实对应**：GPU 一次 forward 处理的多个请求合并
**Vidur 里**：

```python
batch = Batch(
    requests=[request1, request2, request3, request4],
    total_tokens=512 + 256 + 128 + 64,  # 4 个请求总共 960 个 token
)
```

为什么要 batch？因为 GPU 一次 forward 不管处理 1 个还是 8 个请求，开销差不多。把多个请求合并能大幅提高吞吐。

---

### `BatchStage`（批次在 PP stage 上的执行阶段）

**真实对应**：PP=4 时，一个 batch 在 stage 0 → stage 1 → stage 2 → stage 3 流水线传递
**Vidur 里**：

```python
batch_stage = BatchStage(
    batch=batch,
    stage_id=2,   # 当前在第 2 个 PP stage
)
```

只有 PP > 1 才有意义。表示"这个 batch 现在跑到了哪个 stage"。

---

### `Task`（计算节点）

**真实对应**：GPU 上的一段计算
**Vidur 里**：表达一次"要花时间的计算事件"，分三种类型：

| 类型        | 含义                          |
| ----------- | ----------------------------- |
| `COMPUTE` | 通用计算                      |
| `PROMPT`  | Prefill 计算（处理用户输入）  |
| `TOKEN`   | Decode 计算（生成一个 token） |

调度器会根据 Task 类型选择不同的执行时间预测。

---

### `Flow`（通信节点）

**真实对应**：数据从一个 GPU 传到另一个 GPU
**Vidur 里**：

```python
flow = Flow(
    src=replica_p,          # 源
    dst=replica_d,          # 目的
    size_bytes=1.5 * 1024 * 1024 * 1024,  # 1.5GB KV cache
    link=link_rdma,         # 走哪条链路
)
```

Flow 主要用于 PD 解耦时**传 KV cache**——这是 Vidur 模拟的一种通信操作。它和 ASTRA-sim 模拟的"集合通信"不同：

|      | Vidur 的 Flow | ASTRA-sim 的 Collective |
| ---- | ------------- | ----------------------- |
| 类型 | 点对点        | 集合（all_reduce 等）   |
| 用途 | KV cache 传输 | TP/DP/PP 同步           |
| 粒度 | 整个数据块    | 拆成 send/recv 序列     |

---

### `Link`（网络链路）

**真实对应**：物理链路
**Vidur 里**：

| 类型         | 含义                         |
| ------------ | ---------------------------- |
| `PCIe`     | CPU↔GPU                     |
| `Ethernet` | 机间普通以太网               |
| `IB`       | InfiniBand                   |
| `NVLink`   | NVIDIA 机内高速互联          |
| `RDMA`     | RDMA over Converged Ethernet |
| `Dummy`    | 占位符（不真实通信）         |

每个 Link 有带宽和延迟属性。Flow 在某个 Link 上跑，时间 = `size / bandwidth + latency`。

---

### `ExecutionTime`（执行时间封装）

**真实对应**：一段时间长度
**Vidur 里**：把"这次 forward 多久"封装成对象，包含细分：

```python
execution_time = ExecutionTime(
    total_time=23.5,            # 总共 23.5ms
    compute_time=18.0,          # 计算 18ms
    communication_time=5.5,     # 通信 5.5ms
    attention_time=8.0,         # attention 占 8ms
    mlp_time=10.0,              # MLP 占 10ms
)
```

Random Forest 后端预测出来的就是这个对象。Vidur 用它来推进 simulation_time。

---

### `Processor`（处理器）

**真实对应**：单张 GPU 或 CPU
**Vidur 里**：抽象了一个能"按时间执行 Task"的资源。每个 Replica 内部有多个 Processor。

---

## 第六部分：把这一切串起来的事件循环

```python
# 伪代码
sim_time = 0
event_queue = PriorityQueue()  # 按时间排序

# 初始化：把所有请求的"到达事件"放进队列
for request in requests:
    event_queue.push(ArriveEvent(request, time=request.arrived_at))

while not event_queue.empty():
    event = event_queue.pop()
    sim_time = event.time
  
    if isinstance(event, ArriveEvent):
        # 请求到达 → 全局调度器分配 replica
        replica = global_scheduler.assign(event.request)
        replica.scheduler.add_request(event.request)
  
    elif isinstance(event, BatchScheduledEvent):
        # 一个 batch 准备好了 → 预测执行时间
        batch = event.batch
        execution_time = predictor.predict(batch)  # ★ 这里调 RF / AICB / SimAI
      
        # 在未来某个时间点产生"完成事件"
        event_queue.push(BatchCompleteEvent(batch, time=sim_time + execution_time))
  
    elif isinstance(event, BatchCompleteEvent):
        # batch 跑完了 → 更新请求状态
        for request in event.batch.requests:
            request.completed_tokens += 1
            if request.is_done():
                metrics.record(request)
            else:
                # 还没生成完，把它放回调度队列
                replica.scheduler.add_request(request)
      
        # 可能有新的 batch 可以调度
        new_batch = replica.scheduler.try_schedule()
        if new_batch:
            event_queue.push(BatchScheduledEvent(new_batch, time=sim_time))

# 输出 metrics
write_csv(metrics)
```

**关键洞察**：整个模拟器是**纯 Python**，没有任何"真实"的东西。所有 GPU 时间都来自 `predictor.predict(batch)` 的调用。这就是 Vidur 不需要 trace 和网络模拟也能跑的根本原因。

---

## 第七部分：一个完整的"虚拟运行"例子

假设你跑：

```bash
python -m vidur.main \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --cluster_config_num_replicas 2 \
  --synthetic_request_generator_config_num_requests 3 \
  --random_forrest_execution_time_predictor_config_backend vidur
```

### Vidur 内部发生了什么

```
T=0.0s   生成 3 个虚拟请求:
         R1: arrived=0.1s, prefill=100, decode=20
         R2: arrived=0.2s, prefill=200, decode=10
         R3: arrived=0.3s, prefill=50,  decode=30
       
         创建 2 个 Replica (Llama-7B, TP=1, PP=1)
       
T=0.1s   ArriveEvent(R1)
         GlobalScheduler: round_robin → R1 送 Replica0
         Replica0.scheduler: 没有正在跑的 batch，立即调度
         调用 RF 模型: 100 prefill tokens → 18ms
         产生 BatchCompleteEvent(R1, time=0.118s)
       
T=0.118s BatchCompleteEvent(R1)
         R1 prefill 完成，进入 decode 阶段
         metrics: R1.prefill_completed_at = 0.118
         立即调度下一个: R1 的第一个 decode token
         调用 RF 模型: 1 decode token → 5ms
         产生 BatchCompleteEvent(R1_decode_1, time=0.123s)
       
T=0.2s   ArriveEvent(R2)
         GlobalScheduler: round_robin → R2 送 Replica1
         Replica1.scheduler: 立即调度
         调用 RF 模型: 200 prefill tokens → 35ms
         产生 BatchCompleteEvent(R2, time=0.235s)
       
T=0.123s BatchCompleteEvent(R1_decode_1)
         R1 已完成 1/20 个 decode token
         继续调度下一个 decode
         ...
       
T=0.3s   ArriveEvent(R3)
         GlobalScheduler: round_robin → R3 送 Replica0
         Replica0.scheduler: 当前还在跑 R1 的 decode
         如果是 sarathi 调度器: R3 prefill + R1 decode 合并 batch
         如果是 vllm 调度器: 同上
         如果是 orca: 等 R1 完全跑完
         ...
       
... 一直循环直到所有请求完成

T=2.5s   全部完成
         metrics 写入 request_metrics.csv:
         R1: arrived=0.1, completed=0.32, e2e=0.22, ttft=0.018
         R2: arrived=0.2, completed=0.45, e2e=0.25, ttft=0.035
         R3: arrived=0.3, completed=0.52, e2e=0.22, ttft=0.012
```

整个过程**没有跑模型、没有发包、没有用 GPU**。所有数字都是 RF 模型预测出来的。

---

## 第八部分：一句话总结

| 问题                                 | 答案                                                                                                  |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Vidur 是什么？**             | LLM 推理服务的**离散事件模拟器**，模拟"请求→调度→执行→完成"的全流程                          |
| **没有 trace 怎么模拟？**      | 用**预训练的 Random Forest 模型**预测每次 forward 的 GPU 时间，不需要真实运行                   |
| **没有网络模拟怎么算通信？**   | Mode 1 用简化模型 `time = size / bandwidth + latency`；如果要精确模拟才切到 Mode 4 (ns-3)           |
| **调度器是干什么的？**         | 决定"什么请求 什么时候 在哪台机器 和谁一起 跑"                                                        |
| **entities/ 里那些类是什么？** | Vidur 内部的**数据结构**——把"集群/请求/批次/链路"等概念变成 Python 对象，让事件循环能操纵它们 |
| **DAG 是什么？**               | 用图表达请求内部的步骤依赖（特别是 PD 解耦时 prefill→KV传输→decode 的依赖）                         |
| **Flow 和 Task 区别？**        | Task 是计算（消耗 GPU 时间），Flow 是通信（消耗带宽时间）——两种不同的"耗时事件"                     |

**核心洞察**：Vidur 把"运行 GPU"这件事**抽象成一个数学函数** `time = predict(batch)`，然后用**事件驱动模拟**去走完整个推理服务流程。所有"实体类"都是为了支撑这个事件循环——它们就像剧本里的"角色卡"，事件循环像导演按时间顺序触发每个角色的动作。
