# ASTRA-sim 如何模拟 NCCL

## 一、代码位置

所有 NCCL 模拟逻辑集中在一个目录:

```
astra-sim-alibabacloud/astra-sim/system/MockNccl*
```

共 8 个文件,约 2856 行代码:

| 文件                   | 行数           | 职责                                |
| ---------------------- | -------------- | ----------------------------------- |
| `MockNccl.h`         | 225            | NCCL 常量表(算法、协议、延迟、带宽) |
| `MockNcclGroup.h`    | 179            | 全局通信组管理类声明                |
| `MockNcclGroup.cc`   | **2102** | 核心: 通道生成 + Flow Model 生成    |
| `MockNcclChannel.h`  | 121            | 单 rank 通信器、SingleFlow 结构体   |
| `MockNcclChannel.cc` | 69             | 单 rank 通信器实现                  |
| `MockNcclQps.h`      | 38             | QP(Queue Pair)定义                  |
| `MockNcclLog.h/cc`   | 122            | 日志                                |

> 命名 `Mock` 含义: 不真正运行 NCCL,但**完全复刻 NCCL 的算法选择逻辑和拓扑构造**,产出"如果真跑 NCCL 会发生哪些点对点通信"的清单。

---

## 二、整体逻辑(三阶段)

```
[阶段 1] 组划分          [阶段 2] 通道构建            [阶段 3] Flow 展开
   构造函数          ──→     genringchannels        ──→  genAllReduceFlowModels
 TP/DP/PP/EP                gettreechannels             genReduceScatterFlowModels
 → AllGroups                get_nvls_channels           genAllGatherFlowModels
                                                        genAlltoAllFlowModels
                                                              ↓
                                                   List<SingleFlow>(src,dst,size,deps)
                                                              ↓
                                                   交给网络后端 (analytical/ns3) 计时
```

---

## 三、阶段 1: 通信组构建(构造函数)

`MockNcclGroup::MockNcclGroup(ngpus, gpus_per_node, TP, DP, PP, EP, DP_EP, NVSwitch, gpu_type)` 在系统启动时被调用一次。

它根据并行配置(TP/DP/PP/EP)枚举出所有可能的通信组,并填入:

```cpp
std::map<std::pair<int,GroupType>,int> GroupIndex;   // (rank, TP/DP/...) → 组号
std::map<int,GroupInfo> AllGroups;                   // 组号 → {成员rank列表, NVSwitch列表, ...}
```

例如 8 卡 TP=4,DP=2:

- TP 组: {0,1,2,3}, {4,5,6,7}
- DP 组: {0,4}, {1,5}, {2,6}, {3,7}

每个 rank 会查 `GroupIndex` 知道自己属于哪个 TP 组、哪个 DP 组。

---

## 四、阶段 2: 通道(Channel)生成

NCCL 在真实运行时会先构造**逻辑拓扑通道**(Ring / Tree / NVLS),Mock 完全照搬。

### Ring 通道

```cpp
RingChannels MockNcclGroup::genringchannels(int rank, GroupType type);
```

为每个组构造若干环(NCCL 默认多通道并行),返回:

```
channel_id → rank → [前一节点, 后一节点]
```

### Tree 通道

```cpp
TreeChannels MockNcclGroup::gettreechannels(int rank, GroupType type);
```

节点间构造 Double Binary Tree(节点内仍然是链式),数据结构:

```cpp
struct ncclTree { int depth, rank, up; std::vector<int> down; };
```

### NVLS / NVLS_Tree 通道

针对 NVLink Switch(H100/B200)的 SHARP 加速:节点内 NVS 上做 reduce,节点间再走 tree。

`MockNcclComm` 构造时一次性把三类通道全部生成并存好:

```cpp
MockNcclComm::MockNcclComm(int _rank,GroupType _type,MockNcclGroup* _GlobalGroup) {
  this->ringchannels = GlobalGroup->genringchannels(rank,type);
  this->treechannels = GlobalGroup->gettreechannels(rank,type);
  this->nvlschannels = GlobalGroup->get_nvls_channels(rank,type);
}
```

---

## 五、阶段 3: 算法选择 + Flow 展开(关键)

### 5.1 算法选择 — 复刻 NCCL Tuner

`MockNccl.h` 里把 NCCL 真实使用的常量表完整搬了过来:

```cpp
#define NCCL_NUM_ALGORITHMS 6   // Tree/Ring/CollNet_Direct/CollNet_Chain/NVLS/NVLS_Tree
#define NCCL_NUM_PROTOCOLS  3   // Simple / LL / LL128

extern float baseLat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
extern float hwLat  [3 /*NVLINK,PCI,NET*/][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
extern float llMaxBws        [3][3];
extern float perChMaxRingLL128Bws[3][3];
extern float perChMaxTreeBws[3][3];
```

`get_algo_proto_info(type, rank, op, data_size)` 会:

1. 用每条算法 × 每条协议算一次 `time = baseLat + hwLat + nBytes / bw`;
2. 选 `time` 最小的组合 → 输出 `ncclInfo{ algorithm, protocol, nChannels, nBytes }`。

→ **这就是真实 NCCL 中"小消息走 Tree LL,大消息走 Ring SIMPLE"那个 tuner 的代码级复刻**。

### 5.2 算法分发

```cpp
genAllReduceFlowModels(...) {
  ncclInfo* info = get_algo_proto_info(...);
  switch (info->algorithm) {
    case NCCL_ALGO_TREE:
    case NCCL_ALGO_RING:        return genAllReduceRingFlowModels(...);
    case NCCL_ALGO_NVLS:        return genAllreduceNVLSFlowModels(...);
    case NCCL_ALGO_NVLS_TREE:   return genallReduceNVLSTreeFlowModels(...);
  }
}
```

不同 collective(AllReduce / ReduceScatter / AllGather / AllToAll / Reduce)各有独立生成器:

- `genAllReduceRingFlowModels`
- `genReduceScatterFlowModels`
- `genAllGatherFlowModels`
- `genAlltoAllFlowModels`
- `genAllreduceNVLSFlowModels`
- `genallReduceNVLSTreeFlowModels`

### 5.3 Flow 展开 — 把"集合通信"拆成"P2P + 依赖"

输出统一格式 `FlowModels = std::map<(step,index), SingleFlow>`:

```cpp
struct SingleFlow {
  int flow_id;
  int src, dest;                        // P2P 端点
  uint64_t flow_size;                   // 这一段传几字节
  std::vector<int> prev;                // 必须等待的前置 flow_id(同步)
  std::vector<int> parent_flow_id;      // 父依赖
  std::vector<int> child_flow_id;       // 后续依赖
  int channel_id;                       // 哪个 NCCL 通道
  int chunk_id, chunk_count;            // 数据分块编号
  std::string conn_type;                // "RING"/"PXN_INIT"/"NVLS_TREE"...
};
```

### 5.4 例: Ring AllReduce 展开

NCCL 的 Ring AllReduce 在 N 个 rank 上需要 `2(N-1)` 步: N-1 步 ReduceScatter + N-1 步 AllGather。

`genAllReduceRingFlowModels` 做的事:

1. 把 `data_size` 按 chunk 切;
2. 对每个 channel × 每个 chunk × 每一步,生成一个 `SingleFlow(src=ring[i], dst=ring[i+1], size=chunk_size, prev=[上一步对应flow])`;
3. 跨节点的 NVLink+IB 通信会被替换成 **PXN(PCIe X NIC)** 的两段:先 GPU→本地 NIC GPU(`PXN_INIT`),再 NIC GPU→远端 GPU,这里也对应到 SingleFlow 链。

最终一次 AllReduce 会被展开成几十~几百条带依赖图的 SingleFlow,完整描述了"哪一步谁发给谁、依赖于谁完成"。

### 5.5 NVLS Tree 展开

`generate_flow_model_nvls_tree_allreduce_up/down` 用 BFS 遍历树结构:

- up 阶段: 子节点→父节点的 reduce flow,记录 `inDegree` 和 `nodeprevs`,父节点的 flow `prev` 设成所有子节点的 flow_id;
- down 阶段: 父节点→子节点的 broadcast flow,依赖 up 阶段的根节点 flow。

→ 自动构造出一棵带依赖的 DAG。

---

## 六、缓存与对外接口

`MockNcclComm::get_flow_model(data_size, op, layer_num, loopstate)` 是工作负载层调用入口:

```cpp
return GlobalGroup->getFlowModels(type, rank, op, data_size, layer_num, loopstate);
```

`getFlowModels` 内部用 `flow_models` map 做缓存(同一 `(type,op,size)` 只生成一次):

```cpp
std::map<std::string, std::map<int, std::shared_ptr<FlowModels>>> flow_models;
```

工作负载层每遇到一条 collective(例如训练第 N 层的 AllReduce 梯度),就调一次 `get_flow_model`,拿到一张 SingleFlow 列表。

---

## 七、Flow 列表如何变成"时间"

MockNccl 只负责"产出 P2P 通信清单和依赖图",**不算时间**。算时间是后面这一步:

```
SingleFlow 列表
   ↓
system 层 MemBus / Workload (按 prev 依赖逐个发起)
   ↓
network frontend(三选一):
   - analytical:  α-β 模型按 size/bw 直接算
   - ns3:         真实包级 packet simulation
   - phynet:      跑在真硬件上
   ↓
回调通知"这条 flow 完成" → 触发依赖它的 child flow → ...
   ↓
最后一条 flow 完成 → collective 完成 → 工作负载继续
```

---

## 八、为什么这套 Mock 是"真"的

1. **拓扑生成**和真实 NCCL 一样: Ring / Double Binary Tree / NVLS / NVLS_Tree 都按 NCCL 源码逻辑构造。
2. **算法/协议选择**完整复刻 NCCL Tuner — 同样的 `baseLat[][]` / `hwLat[][][]` / `perChMaxBws[][]` 表(数值就是从 NCCL 源码里抄出来的)。
3. **PXN、多 channel 并行、chunk 切分、依赖图**这些 NCCL 里影响实测带宽的细节都保留。
4. **不实际跑 GPU**,所以可以在一台机器上模拟成千上万张卡,这正是 ASTRA-sim 存在的意义。

# 简化
简而言之:`MockNccl*` = 一个**"无 GPU 版的 NCCL 控制平面"**,负责吐出"这次集合通信由哪些点对点通信组成、它们的依赖关系是什么",剩下的物理时延交给 `network_frontend` 真实仿真。
简化 1: 不算 time,只查规则表
真 NCCL 会枚举 6 算法 × 3 协议 = 18 种组合,套公式 time = baseLat + hwLat × hops + size/bw,选 time 最小的。

SimAI 没做这件事。它就是几个 if-else 写死:

条件	选什么
TP + A100/A800	RING
TP + H100/H800 + 8卡+ NVLS 开	NVLS
TP + H100/H800 + 其他情况	RING
DP / EP / DP_EP + 任意	RING
AllGather / ReduceScatter / AllToAll	RING
简化 2: 协议字段直接置为 NCCL_PROTO_UNDEF
真 NCCL 要在 LL / LL128 / Simple 三种协议里选(影响小消息延迟和大消息带宽)。

SimAI 一行代码:info->protocol = NCCL_PROTO_UNDEF; —— 完全不选协议。

为什么? 因为 SimAI 的 flow 模型生成只关心"用哪个算法的拓扑展开",协议层的差异已经被折算到底层 LogGP 参数里了。

简化 3: nChannels = 0
这里设为 0,真正的 nChannels 是后面别的地方决定的(在 gen*FlowModels 里直接按算法的固定 channel 数生成,或者从配置读)。

三、baseLat / hwLat / perChMaxRingLL128Bws 这些表到底用不用?
我搜了一下整个 system 目录:

baseLat / hwLat 出现在:
  MockNccl.h        ← 声明
  collective.md     ← 文档

→ 它们在 .h 里声明了,但 .cc 代码里没人调用。这些表是从真 NCCL 源码搬过来的"摆设",当前版本的 SimAI 实际上没用上。

我之前以为 SimAI 用这些表算 time 来选算法,这是错的。这些表更像是"以防未来需要"的预留。真正在 SimAI 里负责"算时间"的是:

calbusbw.cc: 根据 GPU 类型 + 节点数 + 连接类型,直接给出"等效总线带宽"(经验值)
LogGP: 4 参数 (L, o, g, G) 模型,套 time = L + (size-1)×g + size×G
网络后端: analytical / ns3 自己算包级时延
算法选择(get_algo_proto_info)和时间计算是分开的: 选完算法后,在 flow 生成阶段才用 calbusbw 等模型估算每条 flow 的时间。
