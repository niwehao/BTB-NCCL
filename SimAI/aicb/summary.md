## AICB：模型计算 → 通信操作 的完整转换逻辑

### 一、整体架构

```
模型参数 + 并行策略
        ↓
   构建 MockedModel（模拟模型结构）
        ↓
   逐层调用 forward() / backward()
        ↓
   每个组件（Attention/MLP/MoE）根据并行策略
   插入对应的通信操作（LogItem）
        ↓
   收集所有 LogItem → 输出 workload 文件
```

核心思想：**AICB 不真正计算，只是根据模型结构和并行策略，推算出每层需要哪些通信、通信多大**。

------

### 二、基本数据结构

#### LogItem（一次通信操作）

```python
LogItem:
    comm_type          # 通信类型：all_reduce / all_gather / reduce_scatter / all_to_all / isend / irecv
    comm_group         # 通信组：tp_group / dp_group / pp_group / ep_group
    comm_group_size    # 通信组大小（几个 GPU 参与）
    msg_size           # 消息大小（字节）
    stage              # 来源阶段名（如 "forward.MegatronRowLinear"）
```

#### Workload（一次完整训练/推理的通信序列）

```python
Workload = [LogItem, LogItem, LogItem, ...]  # 按执行顺序排列
```

------

### 三、训练：模型结构 → 通信操作

#### 3.1 模型结构拆解

```
MegatronModel
├── Embedding                              → AllReduce (TP)
├── TransformerLayer × num_layers
│   ├── Attention
│   │   ├── QKV 投影: ColumnLinear         → AllGather (TP+SP) 上一层可能是SP 或 无通信
│   │   └── 输出投影: RowLinear            → ReduceScatter (TP+SP) 或 AllReduce (TP)
│   └── MLP 或 MoE
│       ├── MLP (Dense):
│       │   ├── up 投影: ColumnLinear       → AllGather (TP+SP)
│       │   └── down 投影: RowLinear        → ReduceScatter (TP+SP)
│       └── MoE (Sparse):
│           ├── dispatch                    → AllToAll (EP)
│           ├── expert 计算                 → AllGather (TP)
│           └── combine                     → AllToAll (EP)
└── FinalNorm
```

#### 3.2 两种关键线性层的通信规则

**ColumnLinear**（参数按列切分到 TP 组各 GPU）：

| 条件             | Forward 通信             | Backward 通信            |
| ---------------- | ------------------------ | ------------------------ |
| TP=1             | 无                       | 无                       |
| TP>1，无序列并行 | 无                       | AllReduce (tp_group)     |
| TP>1，有序列并行 | **AllGather** (tp_group) | ReduceScatter (tp_group) |

**RowLinear**（参数按行切分到 TP 组各 GPU）：

| 条件             | Forward 通信                 | Backward 通信        |
| ---------------- | ---------------------------- | -------------------- |
| TP=1             | 无                           | 无                   |
| TP>1，无序列并行 | **AllReduce** (tp_group)     | 无                   |
| TP>1，有序列并行 | **ReduceScatter** (tp_group) | AllGather (tp_group) |

> **核心逻辑**：ColumnLinear 的输出在 TP 组内是**部分结果**，需要 RowLinear 做**聚合**。序列并行把 AllReduce 拆成更高效的 ReduceScatter + AllGather。

#### 3.3 完整训练一个 step 的通信序列

```
一次训练迭代:
│
├── 1. init()                     # 初始化同步
│   ├── AllReduce (dp_group) × 3
│   ├── AllGather (dp_group)
│   └── Broadcast (tp_group)
│
├── 2. forward()                  # 前向传播
│   ├── Broadcast (tp_group)      # 元数据同步
│   ├── 逐层 forward:
│   │   ├── Attention: ColumnLinear → RowLinear 各一次通信
│   │   └── MLP/MoE: ColumnLinear → RowLinear 各一次通信（MoE 加 AllToAll）
│   └── AllReduce (dp_group)      # Loss 同步
│
├── 3. backward()                 # 反向传播（通信与 forward 对称，顺序相反）
│   └── 逐层 backward（从最后一层到第一层）
│
├── 4. step()                     # 优化器更新
│   ├── 非分布式优化器: AllReduce (dp_group)           # 梯度同步
│   └── 分布式优化器:  ReduceScatter + AllGather (dp_group)  # 梯度分片
│   └── AllReduce (tp_group)      # LayerNorm 梯度同步
│
└── 5. PP 通信（如果 PP > 1）
    └── isend / irecv (pp_group)  # 流水线阶段间传递激活值
```

------

### 四、通信大小计算公式

#### 4.1 TP 组通信（每层都有）

| 操作                    | 公式                            | 说明                           |
| ----------------------- | ------------------------------- | ------------------------------ |
| ColumnLinear AllGather  | `2 × seq × batch × input_size`  | 收集完整输入，2 是 bf16 字节数 |
| RowLinear ReduceScatter | `2 × seq × batch × output_size` | 聚合部分结果                   |
| RowLinear AllReduce     | `2 × seq × batch × output_size` | 无 SP 时的聚合                 |
| Embedding AllReduce     | `2 × batch × seq × hidden_size` | 词嵌入聚合                     |

#### 4.2 DP 组通信（每个 step 一次）

| 操作                       | 公式                             | 说明                   |
| -------------------------- | -------------------------------- | ---------------------- |
| 梯度 AllReduce             | `4 × total_params / pp_size`     | FP32 梯度，4 字节/参数 |
| 分布式优化器 ReduceScatter | `4 × total_params / pp_size`     | 梯度分片               |
| 分布式优化器 AllGather     | `2 × total_params / pp_size`     | 参数收集，BF16         |
| LayerNorm 梯度             | `2 × layernorm_params / pp_size` | 小量，但必须同步       |

#### 4.3 PP 组通信（每个 microbatch 一次）

| 操作          | 公式                                  | 说明            |
| ------------- | ------------------------------------- | --------------- |
| isend / irecv | `2 × hidden_size × seq × micro_batch` | 激活值/梯度传递 |

#### 4.4 EP 组通信（MoE 层才有）

| 操作                      | 公式                                        | 说明               |
| ------------------------- | ------------------------------------------- | ------------------ |
| MoE dispatch AllToAll     | `2 × seq × hidden × batch × topk / tp_size` | token 路由到专家   |
| MoE combine AllToAll      | `2 × seq × hidden × batch × topk / tp_size` | 专家结果收回       |
| MoE permutation AllGather | `2 × hidden × topk × batch × seq`           | TP 组内 token 排列 |

------

### 五、推理 vs 训练的区别

|                  | 训练                      | 推理                                       |
| ---------------- | ------------------------- | ------------------------------------------ |
| **阶段**         | forward + backward + step | 仅 forward                                 |
| **DP 通信**      | 有（梯度同步）            | **无**（不更新参数）                       |
| **PP 通信**      | isend/irecv 双向          | 仅单向                                     |
| **计算精度**     | BF16（elem_size=2）       | DeepSeek 用 FP8（elem_size=1），通信量减半 |
| **MoE dispatch** | `topk` 个专家             | DeepSeek 用 `topk-1`（1 个共享专家本地算） |

DeepSeek 推理还有 **FP8 量化系数**：

```python
FP8_FACTOR = (1 + 4/128) / 2 ≈ 0.515
# dispatch 通信量 = msg_size × FP8_FACTOR（约减半）
# combine 通信量 = msg_size（全精度输出）
```

------

### 六、举例：GPT-7B, TP=8, DP=4, 32 GPU

```
模型参数：num_layers=32, hidden_size=4096, seq=2048, micro_batch=1

每层 forward 通信（TP 组内，开启序列并行时）：
├── Attention:
│   ├── QKV AllGather:      2 × 2048 × 1 × 4096 = 16 MB
│   └── Output ReduceScatter: 2 × 2048 × 1 × 4096 = 16 MB
├── MLP:
│   ├── Up AllGather:       2 × 2048 × 1 × 4096 = 16 MB
│   └── Down ReduceScatter: 2 × 2048 × 1 × 4096 = 16 MB
└── 每层合计: 64 MB

32 层 forward: 64 MB × 32 = 2048 MB
32 层 backward: 同上 ≈ 2048 MB

DP 梯度同步（每 step）:
├── total_params ≈ 7B → AllReduce: 4 × 7B / 1 = 28 GB（最大通信）

每次迭代 TP 通信: ~4 GB
每次迭代 DP 通信: ~28 GB  ← 瓶颈
```

------

### 七、总结：转换链路

```
hidden_size, num_layers, seq_length     ← 决定每层参数量
        ↓
TP / SP                                 ← 决定每层用什么通信原语、通信多大
        ↓
PP                                      ← 决定层间 P2P 通信
        ↓
DP                                      ← 决定梯度同步大小
        ↓
EP + num_experts + topk                 ← 决定 MoE AllToAll 大小
        ↓
micro_batch × epoch_num                 ← 决定通信次数
        ↓
[LogItem, LogItem, ...] → workload 文件
```

**本质就是：模型结构决定通信量，并行策略决定通信方式，两者组合生成完整的通信序列。**