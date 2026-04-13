# ASTRA-sim 如何调用 ns-3

## 架构总览:一张图

```
┌────────────────────────────────────────────────────┐
│  ASTRA-sim (system layer)                          │
│                                                    │
│  NcclTreeFlowModel::insert_packets(channel, flow)  │
│           ↓                                        │
│  Sys::sim_send(buffer, count, dst, tag, req,       │
│                msg_handler, fun_arg)                │
│  Sys::sim_recv(buffer, count, src, tag, req,       │
│                msg_handler, fun_arg)                │
│           ↓                                        │
├────────────────── 虚函数接口 ──────────────────────┤
│  AstraNetworkAPI (纯虚基类)                        │
│    virtual sim_send() = 0                          │
│    virtual sim_recv() = 0                          │
│    virtual sim_schedule() = 0                      │
│    virtual sim_get_time() = 0                      │
├────────────────── 实现层 ──────────────────────────┤
│  ASTRASimNetwork (ns3 实现, 继承 AstraNetworkAPI)  │
│    sim_send() → sentHash + SendFlow()              │
│    sim_recv() → expeRecvHash (登记等待)             │
│    sim_get_time() → Simulator::Now()               │
│    sim_schedule() → Simulator::Schedule()          │
│           ↓                                        │
├────────────────── 胶水层 (entry.h) ────────────────┤
│  SendFlow()                                        │
│    → RdmaClientHelper + Install + Start            │
│           ↓                                        │
├────────────────── ns-3 内部 ───────────────────────┤
│  RdmaClient::StartApplication()                    │
│    → RdmaDriver::AddQueuePair()                    │
│      → RdmaHw: 真正的网卡/交换机/PFC/CC 仿真       │
│        → QbbNetDevice: 交换机端口排队               │
│           ↓ 完成事件                                │
│  qp_finish() / send_finish()                       │
│    → notify_receiver_receive_data()                │
│    → notify_sender_sending_finished()              │
│      → t2.msg_handler(t2.fun_arg)  ← 回调回上层    │
└────────────────────────────────────────────────────┘
```

------

## 阶段 1：发起——从 SingleFlow 到 `sim_send`/`sim_recv`

`NcclTreeFlowModel::insert_packets` 把一个 SingleFlow 翻译成一对 `sim_send + sim_recv`。在 `Sys.cc` 里:

```
Sys::sim_send(buffer, count=flow_size, dst, tag=tag_id, request, handler, fun_arg)
Sys::sim_recv(buffer, count=flow_size, src, tag=tag_id, request, handler, fun_arg)
```

这两个函数透传给 `AstraNetworkAPI` 的虚函数。

------

## 阶段 2：胶水——`ASTRASimNetwork` 做了什么

### `sim_send`（`AstraSimNetwork.cc:107-133`）

```cpp
int sim_send(..., int dst, int tag, sim_request *request,
             void (*msg_handler)(void*), void *fun_arg) {
  // 1. 在 sentHash 登记这次发送（用于日后回调匹配）
  sentHash[make_pair(tag, make_pair(rank, dst))] = t;
  
  // 2. 立刻调 SendFlow 注入 ns-3
  SendFlow(rank, dst, count, msg_handler, fun_arg, tag, request);
}
```

### `sim_recv`（`AstraSimNetwork.cc:134-208`）

```cpp
int sim_recv(..., int src, int tag, sim_request *request,
             void (*msg_handler)(void*), void *fun_arg) {
  // 先查 recvHash：数据是不是已经到了？
  if (recvHash 已有 [tag, src→rank]) {
    // 数据早到了（ns-3 先完成的）→ 立刻回调
    t.msg_handler(t.fun_arg);     // ← 直接触发上层
  } else {
    // 数据还没到 → 登记到 expeRecvHash 等着
    expeRecvHash[make_pair(tag, make_pair(src, rank))] = t;
  }
}
```

**关键设计**：`sim_recv` **不发起任何网络操作**,它只是登记"我在等来自 src 的 tag 数据"。

------

## 阶段 3：注入 ns-3——`SendFlow()`（`entry.h:107-164`）

```cpp
void SendFlow(int src, int dst, uint64_t maxPacketCount,
              void (*msg_handler)(void*), void *fun_arg, int tag,
              sim_request *request) {
  
  uint32_t port = portNumber[src][dst]++;   // 每个 flow 一个唯一端口号
  
  // 1. 记录 port → flowTag 映射（日后 qp_finish 回查）
  sender_src_port_map[make_pair(port, make_pair(src, dst))] = request->flowTag;
  
  // 2. 构造 ns-3 RdmaClient 应用
  int send_lat = 6000;  // 🔴 默认 6μs 注入延迟（可被 AS_SEND_LAT 环境变量覆写）
  RdmaClientHelper clientHelper(
      pg,                              // 优先级组
      serverAddress[src],              // 源 IP
      serverAddress[dst],              // 目的 IP
      port, dport,
      real_PacketCount,                // 字节数
      has_win ? BDP : 0,               // 窗口（拥塞控制）
      pairRtt[src][dst],               // RTT
      msg_handler, fun_arg, tag,
      src, dst);
  
  // 3. 安装到 ns-3 节点并调度
  ApplicationContainer appCon = clientHelper.Install(n.Get(src));
  appCon.Start(Time(send_lat));        // 在 send_lat 后启动
  
  // 4. 计数器 +1（多 QP 场景需要所有都完成才算完）
  waiting_to_sent_callback[flow_id, (src,dst)]++;
  waiting_to_notify_receiver[flow_id, (src,dst)]++;
}
```

从这里开始，**控制权交给 ns-3 的事件循环**。ns-3 的 `RdmaClient::StartApplication` 会创建 QueuePair,往 `RdmaHw` 里注入数据包,经过交换机端口队列、PFC、ECMP、DCQCN 等真实仿真。

------

## 阶段 4：返回——ns-3 完成后怎么通知 ASTRA-sim

### 发送完成路径

```
ns-3 NIC 把最后一个包发出去
  → RdmaDriver 的 "SendComplete" TracedCallback
  → send_finish() (entry.h:347)
    → is_sending_finished()           // 入度计数 -1
    → 计数到 0:
      notify_sender_sending_finished()
        → sentHash 查回 msg_handler
        → t2.msg_handler(t2.fun_arg)  // ← 回调 ASTRA-sim
```

这个回调在 `NcclTreeFlowModel::run` 里被接住,触发 `EventType::PacketSentFinshed`,然后调用 `reduce()` 做本地归约。

### 接收完成路径（**更关键**）

```
ns-3 目的端收完最后一个 ACK
  → RdmaDriver 的 "QpComplete" TracedCallback
  → qp_finish() (entry.h:299)
    → sender_src_port_map 反查 flowTag
    → is_receive_finished()           // 入度计数 -1
    → 计数到 0:
      notify_receiver_receive_data()
        → expeRecvHash 查回 msg_handler
        → ehd->flowTag = flowTag      // ← 把 child_flow_id 传回去!
        → t2.msg_handler(t2.fun_arg)  // ← 回调 ASTRA-sim
```

这个回调触发 `EventType::PacketReceived`,带着 `tree_flow_list`(child 列表)回到 `NcclTreeFlowModel`,然后:

```cpp
for (int next_flow_id : next_flow_list) {
  if (--indegree_mapping[next_flow_id] == 0) {
    insert_packets(channel_id, next_flow_id);  // ← 发下一步
  }
}
```

形成完整的循环。

------

## 阶段 5：启动——`main()` 怎么把它们焊在一起

`AstraSimNetwork.cc:260-335` 的 `main()`:

```cpp
// 1. 初始化 ns-3 网络拓扑
main1(network_topo, network_conf);
  → ReadConf() → SetConfig() → SetupNetwork(qp_finish, send_finish)
  //                                       ↑ 把回调函数指针传给 ns-3

// 2. 每个 GPU 节点创建一对 (Network, Sys)
for (int j = 0; j < nodes_num; j++) {
  networks[j] = new ASTRASimNetwork(j, 0);        // ns-3 适配器
  systems[j]  = new AstraSim::Sys(networks[j], ...); // Sys 持有 network 指针
}

// 3. 启动工作负载
for (int i = 0; i < nodes_num; i++) {
  systems[i]->workload->fire();   // workload 发起第一个 collective
}

// 4. ns-3 事件循环接管
Simulator::Run();   // ← 所有事情在这里面发生
```

**整个仿真就是 ns-3 的 `Simulator::Run()` 事件循环**。ASTRA-sim 的所有"动作"都是通过 ns-3 event 驱动的——`sim_schedule` 也是直接调用 `Simulator::Schedule(NanoSeconds(...))`。

------

## 三个关键数据结构（配对的"信箱"）

| Hash Map                     | key                    | 用途                                          |
| ---------------------------- | ---------------------- | --------------------------------------------- |
| `sentHash`                   | `(tag, (src, dst))`    | 发送端登记的回调,等 `send_finish` 来取        |
| `expeRecvHash`               | `(tag, (src, dst))`    | 接收端登记的"我在等",等 `qp_finish` 来匹配    |
| `recvHash`                   | `(tag, (src, dst))`    | 数据先于 `sim_recv` 到达时,暂存已收到的字节数 |
| `sender_src_port_map`        | `(port, (src, dst))`   | port → flowTag,供 `qp_finish` 反查 flow 归属  |
| `waiting_to_notify_receiver` | `(flow_id, (src,dst))` | 一个 flow 拆成多个 QP 时的入度计数器          |

**乱序匹配机制**:因为 `sim_send` 和 `sim_recv` 可能不同步调用(send 先/recv 先),这三个 hash 形成了一个**消息信箱**:

- `sim_recv` 调用时先查 `recvHash`,如果数据已到就直接回调(不等)。
- `qp_finish` 调用时先查 `expeRecvHash`,如果 recv 还没登记就先存 `recvHash`(等 recv 来取)。

------

## 一句话总结

ASTRA-sim 通过 **虚函数接口** `AstraNetworkAPI` 把 `sim_send/sim_recv` 映射到 ns-3 的 RDMA QueuePair;ns-3 的事件循环驱动整个仿真;数据包走完交换机/链路后,ns-3 通过 **TracedCallback** (`qp_finish` / `send_finish`) 回调回 ASTRA-sim;中间靠 **4 个全局 hash map** 做发送/接收的配对匹配。整个系统是**单进程、单事件循环**,ASTRA-sim 是"事件生产者+消费者",ns-3 是"事件调度器+网络模拟器"。

Here is the complete summary of what was done:

------

## 实现完成报告

### 目标

用 htsim 替换 NS-3 作为 SimAI 的网络后端，实现 OCS-ECS 混合拓扑，复用 mixnet 的光电选择逻辑。

### 编译与测试

- **编译**: `make -f Makefile.htsim -j8` ✅ 成功
- **二进制**: `simai_htsim` (arm64, 1.3MB)
- **测试运行**: ✅ 成功完成，工作负载正常解析和执行

运行示例：

```bash
./simai_htsim -w SimAI/aicb/results/workload/DeepSeek-671B-world_size8-tp1-pp1-ep8-bs32-seq1024-decode.txt \
  --nodes 8 --alpha 4 --speed 100000 --reconf_delay 10 \
  --dp_degree 1 --tp_degree 1 --pp_degree 1 --ep_degree 8 --gpus_per_server 1
```

### 修改的 SimAI 原文件（2个）

| 文件                                               | 修改内容                                                     |
| -------------------------------------------------- | ------------------------------------------------------------ |
| `astra-sim/system/AstraNetworkAPI.hh`              | `ncclFlowTag` 结构体新增 `int com_type` 字段，用于传递集合通信类型（AllToAll=4）到网络后端 |
| `astra-sim/system/collective/NcclTreeFlowModel.cc` | 两处 `flowTag` 构造处添加 `snd_req.flowTag.com_type = static_cast<int>(this->comType)` |

### 修改的 mixnet 文件（1个）

| 文件                                          | 修改内容                                                     |
| --------------------------------------------- | ------------------------------------------------------------ |
| `mixnet-htsim/src/clos/datacenter/mixnet.cpp` | 注释掉 `#include "taskgraph_generated.h"`（未使用，且与本地 flatbuffers 版本不兼容） |

### 新建文件（3个）

| 文件                                        | 说明                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| `network_frontend/htsim/entry.h`            | htsim 版 SendFlow + 回调机制 + OCS/ECS选择逻辑 + NVLink处理 + CallbackEvent适配器 |
| `network_frontend/htsim/AstraSimNetwork.cc` | ASTRASimNetwork 类 + main() + 拓扑创建 + 事件循环            |
| `Makefile.htsim`                            | 联合编译 astra-sim + htsim 的构建文件                        |

### 核心设计

- **`CallbackEvent`**: 桥接 astra-sim 的 `void(*)(void*)` 回调到 htsim 的 `EventSource` 继承体系
- **OCS/ECS 选择**: `com_type==4 && conn[src][dst]>0` → OCS (Mixnet直连)，否则 → ECS (FatTree)
- **重配置感知**: 若 OCS 区域正在重配置 (`TOPO_RECONF`)，延迟流到重配完成
- **NVLink**: 同机器 GPU 通信模拟为 900Gbps 固定带宽传输
- **终止机制**: `sim_finish()` 设置 `g_simulation_done` 标志，事件循环检查后退出