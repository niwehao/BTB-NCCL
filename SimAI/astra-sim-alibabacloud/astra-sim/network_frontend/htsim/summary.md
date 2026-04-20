# htsim 前端:从 main 到 workload 解析的调用链

本文件记录 htsim 前端启动后,aicb 生成的 workload 文件是如何被 astra-sim 读入并解析的完整路径。所有行号基于当前仓库状态。

## 调用链概览

```
htsim main                        AstraSimNetwork.cc:372
  └─ parse_params → params.workload        AstraSimNetwork.cc:379
  └─ for (j = 0; j < nodes_num; j++)       AstraSimNetwork.cc:702
       └─ new AstraSim::Sys(..., params.workload, ...)   AstraSimNetwork.cc:704
            └─ Sys::Sys(..., std::string my_workload, ...)  Sys.cc:119
                 ├─ post_process_inputs()                    Sys.cc:197
                 ├─ scheduler_unit / vLevels / logical_topologies  Sys.cc:247–262
                 ├─ NI->sim_init(MEM)                        Sys.cc:273
                 ├─ new MemBus(...)                          Sys.cc:274
                 └─ new Workload(run_name, this, my_workload, ...)  Sys.cc:285
                      └─ Workload::Workload(..., std::string name, ...)  Workload.cc:59
                           └─ this->initialized = initialize_workload(name)  Workload.cc:87
                                └─ Workload::initialize_workload(name)     Workload.cc:1345
                                     ├─ ifstream.open(name)                Workload.cc:1349
                                     ├─ 第一行 header → 解析并行策略/TP/EP/PP/VPP/GA/all_gpus/pp_comm/checkpoints   Workload.cc:1363–1480
                                     ├─ 第二行 → 层数 lines                 Workload.cc:1482–1490
                                     ├─ layers = new Layer*[SIZE]          Workload.cc:1491
                                     └─ for (i = 0; i < lines; i++)
                                          按字段读每一层:
                                          id / depen / fp_compute / fp_comm_type / fp_comm_size
                                          / ig_compute / ig_comm_type / ig_comm_size
                                          / wg_compute / wg_comm_type / wg_comm_size / wg_update_time
                                          然后按字符串前缀映射到 ComType + MockNccl::GroupType
                                          (ALLREDUCE / ALLTOALL / ALLREDUCEALLTOALL / ALLGATHER / ...)
                                          Workload.cc:1492+
```

## 分步说明

### 1. 进程入口
- 文件: `SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/htsim/AstraSimNetwork.cc`
- `int main(int argc, char *argv[])` 位于 `AstraSimNetwork.cc:372`
- `parse_params(argc, argv, &params)` (`AstraSimNetwork.cc:379`) 把命令行的 `-w <workload_path>` 写入 `params.workload`
- 随后完成全局 SPEED / GPU 数 / 拓扑类型解析、创建 log 目录、构建拓扑、准备 NVSwitch 映射 (`AstraSimNetwork.cc:680–700`),这些与 workload 解析无直接关系

### 2. 为每个 GPU 构造 Sys
- `AstraSimNetwork.cc:702–730`:
  ```cpp
  for (int j = 0; j < nodes_num; j++) {
    networks[j] = new ASTRASimNetwork(j, 0);
    systems[j] = new AstraSim::Sys(
        networks[j], nullptr, j, 0,
        params.iterations,
        {nodes_num}, {1}, "",
        params.workload,        // ← my_workload (workload 文件路径)
        ...);
  }
  ```
- **同一个 workload 文件会被 `nodes_num` 个 Sys 各自独立打开并 parse 一次**,每个 Sys 持有自己的 `Workload` 对象和 `layers[]` 数组。

### 3. Sys::Sys 构造函数里 new Workload
- 文件: `SimAI/astra-sim-alibabacloud/astra-sim/system/Sys.cc`
- `Sys::Sys(... std::string my_workload ...)` 位于 `Sys.cc:119`
- 先做:`post_process_inputs()` (`Sys.cc:197`)、建 `scheduler_unit` / `vLevels` / `logical_topologies` (`Sys.cc:247–262`)、`NI->sim_init(MEM)` (`Sys.cc:273`)、`new MemBus(...)` (`Sys.cc:274`)
- 然后 `Sys.cc:285–293`:
  ```cpp
  workload = new Workload(
      run_name, this, my_workload, num_passes,
      total_stat_rows, stat_row, path, this->seprate_log);
  ```
- `Sys.cc:294–298`:若 `workload->initialized == false` 则 `sys_panic`

### 4. Workload::Workload 触发解析
- 文件: `SimAI/astra-sim-alibabacloud/astra-sim/workload/Workload.cc`
- `Workload::Workload(..., std::string name, ...)` 位于 `Workload.cc:59`
- `Workload.cc:87` 调用 `this->initialized = initialize_workload(name);` —— 这是 `initialize_workload` **唯一的调用点**

### 5. Workload::initialize_workload 本体
- 位于 `Workload.cc:1345`
- 打开文件:`Workload.cc:1348–1362`
- **第一行 header**(对应 aicb dump 出来的 meta 行):`Workload.cc:1363–1380` 先 tokenize,然后 `decode_parallelsim(tokens[0])` 取并行策略
- 按策略提取字段:`model_parallel_NPU_group:` / `ep:` / `pp:` / `vpp:` / `ga:` / `all_gpus:` / `pp_comm:` / `checkpoints:` / `checkpoint_initiates:` / `DLRM_LAST_BOTTOM_LAYER:`(`Workload.cc:1382–1480`)
- **第二行**:层数 `lines`(`Workload.cc:1482–1490`),分配 `layers = new Layer*[SIZE]`(`Workload.cc:1491`)
- **每层字段**(`Workload.cc:1492+`):
  `id` / `depen` /
  `fp_compute_time` `fp_comm_type_s` `fp_comm_size` /
  `ig_compute_time` `ig_comm_type_s` `ig_comm_size` /
  `wg_compute_time` `wg_comm_type_s` `wg_comm_size` / `wg_update_time` / ...
- 通信字符串前缀映射(`Workload.cc:1529+`):
  `ALLREDUCE*` → `ComType::All_Reduce`,
  `ALLTOALL*` → `ComType::All_to_All`,
  `ALLREDUCEALLTOALL*` → `ComType::All_Reduce_All_to_All`,
  `ALLGATHER*` → `ComType::All_Gather`,
  后缀 `_EP` / `_DP_EP` 决定 `MockNccl::GroupType`(DP / EP / DP_EP)

### 6. fire
- 回到 main:`AstraSimNetwork.cc:734–736` 对每个 Sys 调用 `systems[i]->workload->fire();`,进入事件循环 `eventlist.doNextEvent()` (`AstraSimNetwork.cc:742`)。

## 相关文件速查

| 文件 | 作用 |
| --- | --- |
| `network_frontend/htsim/AstraSimNetwork.cc` | htsim 前端 main,构造 Sys、fire workload、跑事件循环 |
| `network_frontend/htsim/entry.h` / `entry_prenet.h` | ASTRASimNetwork 的发送/接收入口 |
| `system/Sys.cc` | `Sys::Sys` 构造时 new `Workload` |
| `workload/Workload.cc` | `Workload::Workload` → `initialize_workload` 读文件、建 `Layer[]` |
| `workload/Layer.cc` / `Layer.hh` | 每层的运行时状态与回调 |

## 与 aicb 的对应关系

aicb 的 `SimAI/aicb/workload_generator/SimAI_training_workload_generator.py` 里把 `Work_Item` 逐个 append 到 `self.workload`,然后 dump 成文本:第一行是并行策略 header,第二行是层数,之后每行是一层的若干字段。`Workload::initialize_workload` 读取时的字段顺序与 aicb dump 时写的字段顺序一一对应;若不一致会触发 `Workload.cc:1478` 的 "Input workload format mismatch" 警告。

## tcp与RDMA仿真

  这套组合"像" RoCE 的哪些地方                          
                                                           
  ┌─────────────────────┬──────────────────────────────┐   
  │      RoCE 特征      │       htsim 的模拟方式       │   
  ├─────────────────────┼──────────────────────────────┤
  │ Lossless fabric     │ LosslessInputQueue 的 PFC    │
  │                     │ watermark 反压               │
  ├─────────────────────┼──────────────────────────────┤   
  │ ECN-based CC        │ ecnqueue 做 marking + DCTCP  │
  │                     │ 算 α 调 cwnd                 │   
  ├─────────────────────┼──────────────────────────────┤
  │ PFC pause frame     │ EthPausePacket + sendPause   │   
  ├─────────────────────┼──────────────────────────────┤   
  │ 拥塞不丢包,靠       │ ✓                            │
  │ ECN+PFC 调速        │                              │   
  └─────────────────────┴──────────────────────────────┘   
   
  这是一个合理的替身模型 —— 在链路级行为(utilization, queue
   depth, ECN 触发率, FCT)上和 RoCE 差不多,所以对
  collective 性能评估有参考价值。                          
                  
   但它不是真 RDMA,以下都没有                            
   
  真 RDMA 特征: Message-oriented(非字节流)                 
  htsim 里: 仍是 byte stream + TCP 切片
  差别: TCP 分段开销/乱序重组在模型里仍然存在              
  ────────────────────────────────────────
  真 RDMA 特征: QP(RC/UC/UD) + QPN                         
  htsim 里: 无,只有 port number                            
  差别: 并发连接管理、QP 状态机全没                        
  ────────────────────────────────────────                 
  真 RDMA 特征: verbs 语义(READ/WRITE/SEND/RECV)           
  htsim 里: 只有 byte stream send
  差别: RDMA 读远端 / 远端直写内存都模拟不了               
  ────────────────────────────────────────
  真 RDMA 特征: DCQCN(NP / RP Rate-based CC)               
  htsim 里: 用 DCTCP cwnd-based 代替
  差别: DCQCN rate 调节动力学 ≠ DCTCP cwnd 动力学          
  ────────────────────────────────────────
  真 RDMA 特征: 3-way handshake-less 建连                  
  htsim 里: TCP 仍然有 SYN/slow-start(htsim 简化了但不是零)
  差别: RDMA 建连开销可忽略,htsim 模型里有                 
  ────────────────────────────────────────
  真 RDMA 特征: PFC-induced HoL blocking / victim flow     
  htsim 里: 部分体现(pause 会波及同一上游)
  差别: htsim 没建模 priority class,无法研究 priority      
  间干扰          
  ────────────────────────────────────────
  真 RDMA 特征: Go-Back-N / selective retransmit(RDMA 风格)
  htsim 里: TCP 重传(sack/go-back-n 按 TCP)
  差别: 重传细节不同      
### 拥塞算法   
类型: ECN                                            
  底层 class: ECNQueue                                 
  丢包?: 会丢(ecnqueue.cpp:52-60 超 maxsize drop)      
  PFC 反压?: 不主动发(能识别收到的)                    
  ECN 打标?: 有 (:105-106 出队时 _queuesize > K 打 CE)
  ────────────────────────────────────────             
  类型: LOSSLESS                                       
  底层 class: LosslessQueue
  丢包?: 不丢(靠 Switch 级 PFC)                        
  PFC 反压?: sendPause(queue_lossless.cpp:78-81      
    high_threshold 触发)                             
  ECN 打标?: 无
  ────────────────────────────────────────
  类型: LOSSLESS_INPUT                                 
  底层 class: LosslessOutputQueue + LosslessInputQueue
  丢包?: 不丢                                          
  PFC 反压?: input 虚队列发                          
    PAUSE(queue_lossless_input.cpp:47-50)            
  ECN 打标?: 无(ECN=0)
  ────────────────────────────────────────
  类型: LOSSLESS_INPUT_ECN                             
  底层 class: 同上 + ECN=1, K=16
  丢包?: 不丢                                          
  PFC 反压?: 同上                                    
  ECN 打标?: 有 (queue_lossless_output.cpp:110-111   
    出队时 _queuesize > K 打 CE)
ECN queue 的 K = 50 pkts,LOSSLESS_INPUT_ECN queue 的 
  K = 16 pkts
queuesize(用户可配的)     对于LIE   
看 LOSSLESS_INPUT_ECN 中maxsize 是写死的        
  memFromPkt(10000),完全忽略传入的 queuesize。你设     
  queuesize=8 对 LOSSLESS_INPUT_ECN queue 的 maxsize
  没有效果,也跟 K 无关。
对于ECN -ocs  
这里 queuesize 已经是字节(从构造函数传进来),再  
  memFromPkt 一次等于字节数当 pkt count 乘 1500,结果   
  maxsize = memFromPkt(12000) = 12,000 pkts ≈ 18 MB ——
  远超你写 queuesize=8 的意图
实际:                      
                                                     
  ┌───────┬─────────────────┬───────────────┬─────┐    
  │ 位置  │   queue_type    │    maxsize    │  K  │  
  ├───────┼─────────────────┼───────────────┼─────┤    
  │ Mixne │                 │ 12000         │ 50  │  
  │ t OCS │ ECN             │ pkts(单位 bug │ pkt │
  │  over │                 │  放大)        │ s   │    
  │ lay   │                 │               │     │
  ├───────┼─────────────────┼───────────────┼─────┤    
  │ FatTr │                 │ 10000 pkts(硬 │ 16  │    
  │ ee    │ LOSSLESS_INPUT_ │ 编码,忽略     │ pkt │
  │ ECS   │ ECN             │ queuesize)    │ s   │    
  │ 底座  │                 │               │     │  
  └───────┴─────────────────┴───────────────┴─────┘



## 对astra-sim 的数据流
在整体数据流里的位置

   astra-sim 核心(Sys / Workload / NcclTreeFlowModel)
           │  collective 算法决定 "rank A 向 rank B 发 N
  bytes tag T"
           │
           ▼  通过 AstraNetworkAPI 接口(多态)
   ASTRASimNetwork::sim_send(dst, count, tag, ...)     ←
  htsim 前端的实现
           │
           ▼  sentHash 登记 + 调用 SendFlow
   entry.h::SendFlow(src, dst, ...)
           │
           ▼  构造 DCTCPSrc + 查 g_mixnet_topo 路径 + 建
  TCP flow
   htsim 事件循环 + TcpRtxTimerScanner +
  FatTreeTopology/Mixnet/Prenet
           │
           ▼  flow 完成后 htsim_flow_finish 回调
   接收端 ASTRASimNetwork::sim_recv 或已登记的 recvHash
  被命中
           │
           ▼  调用 msg_handler(fun_arg) 通知 astra-sim 核心

### 详细解释recv流: sim_recv()
  astra-sim                                                
  底层其实是三个独立的函数,它们可能按任意先后发生:
                                                           
  ① A 侧:       sim_send(dst=B, count=100, tag=5,          
  handler_A, ...)                                          
                 → 建 TCP flow,开始传输,A 侧登记 sentHash  
                                                           
  ② B 侧:       sim_recv(src=A, count=100, tag=5,          
  handler_B, arg_B)                       
                 → B 的 Sys 告诉 AstraSimNetwork           
  "我准备好收数据了"                                       
                                                           
  ③ 网络完成:    notify_receiver_receive_data(A→B, 100,    
  flowTag)                                                 
                 → 真正的 TCP 数据传完,从 entry.h
  那边回调过来 
    B 侧的 ②(sim_recv)和 ③(数据真到)哪个先发生,没保证。      
  - ② 先:Sys 调度 recv 时,flow 还没传完                    
  - ③ 先:flow 传完了,但 Sys 还没调 sim_recv
                                            
  Hash: recvHash recv                                          
    key: (tag, src, dst)                                     
    value: uint64_t(字节数)                                  
    语义: 已到达但还没被 recv 消费的数据量                 
  ────────────────────────────────────────                 
  Hash: expeRecvHash   post                                 
  key: (tag, src, dst)                                     
  value: task1{ count, msg_handler, fun_arg }              
  语义: 已 post 但数据还没到的 recv 请求                   
  
                                                            
  recvHash = "到了货但没人领"的缓冲池(存数量) 
  expeRecvHash = "挂了号但货没来"的等待队列(存回调)  

   //   匹配逻辑(简化):
  // if (recvHash 已有 tag 的数据)
  //     if (已到量 == 预期量) → 立刻回调 msg_handler,删条目
  //     if (已到量  > 预期量) → 分一部分走,剩的留在 recvHash
  //     if (已到量  < 预期量) → 把已到的消掉,剩的挂到
  // expeRecvHash 等
  // else
  //     把 recv 请求挂到 expeRecvHash,等数据到

    ① "分一部分走":消费掉 recv 请求需要的 100 字节,扣减      
  recvHash
  ② "剩的留在 recvHash":300 − 100 = 200,这 200 继续躺在    
  recvHash 里,等下一次 sim_recv 来领   

  ## CallbackEvent
  两个使用场景(entry.h 里只有两处 new)                 
                                                       
  用法 1:schedule_callback(entry.h:155-158)—           
  sim_schedule 的底层实现                                                                  
  void schedule_callback                                                  
  被 ASTRASimNetwork::sim_schedule                   
  调(AstraSimNetwork.cc:99-106):                                                               
  astra-sim 说 "X ns 后调这个函数",htsim 前端就 new  
  一个 CallbackEvent,把函数打包进去,挂 _eventlist      
  上延迟 X ps 触发。这是 astra-sim 所有"延迟回调"(比如
  Workload 的 Workload_Wait、register_event(...,       
  delay))的底层通道。  

  用法 2:NVLink 同机 flow 的"完成回调"(entry.h:837-839)
  跨机 TCP 路径的完成不用 CallbackEvent(TcpSink      
  自己在接收到尾包时调 htsim_flow_finish),只有同机
  NVLink 这种没有真实包、只有时间延迟的情形,需要显式造
  一个定时事件来"假装 flow 结束"。  

## sys状态机
State: "等待 step k 的 chunk 接收完成"             
    ↓ 事件:某个 flow 收到数据                        
    ↓ handler 被调,更新 "已到 chunk 数 +=1"          
    ↓ 如果凑齐了这一步所有 chunk:             
       ├─ 做本步的 reduce(local compute)             
       ├─ 发起 step k+1 的 sim_send                  
       └─ 进入下一状态 "等待 step k+1 的 chunk"      
                

## 模块交互
● 四者调用关系总结

  四个文件是四层同心圆:外层包内层,上行调用、下行回调。

  ┌─────────────────────────────────────────────────────┐
  │  AstraSimNetwork.cc  (main + 每rank一个网络适配器)  │   ← 最外层:装配+启动
  │  ┌───────────────────────────────────────────────┐  │
  │  │  Sys.cc  (每rank一个系统单例,管 collective)  │  │   ← 编排层
  │  │  ┌─────────────────────────────────────────┐  │  │
  │  │  │  Workload.cc (状态机 + layer 列表驱动)  │  │  │   ← 业务层
  │  │  └─────────────────────────────────────────┘  │  │
  │  └───────────────────────────────────────────────┘  │
  │  ┌───────────────────────────────────────────────┐  │
  │  │  entry.h  (AstraSim ↔ htsim 胶水,全局函数)   │  │   ← 适配层
  │  └───────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────┘


  文件: AstraSimNetwork.cc
  是什么: main() + ASTRASimNetwork 类(实现 AstraNetworkAPI)
  关键产物: EventList、拓扑、systems[]、networks[]
  ────────────────────────────────────────
  文件: entry.h
  是什么: 全局函数+哈希表,桥接 AstraSim 的 send/recv 语义和 htsim 的
    TCP flow
  关键产物: SendFlow、notify_*、sentHash/recvHash/expeRecvHash
  ────────────────────────────────────────
  文件: Sys.cc
  是什么: 每 rank 一个 Sys 对象,负责 collective 的分解/调度
  关键产物: Sys、generate_*、front_end_sim_send/recv
  ────────────────────────────────────────
  文件: Workload.cc
  是什么: 每 rank 一个 Workload,跑 forward/backward 状态机
  关键产物: Workload、fire()、iterate_*、issue_pp_communication

    AstraSimNetwork.cc 对外:
  - class ASTRASimNetwork : public AstraNetworkAPI 实现
  sim_send/sim_recv/sim_schedule/sim_finish/sim_get_time/sim_time_resolution。
  - 还拥有 main()、CLI 解析、拓扑构造、EventList 循环。

  entry.h 对外(都是自由函数/全局变量):
  - SendFlow(...) —— 被 ASTRASimNetwork::sim_send 调。
  - notify_sender_sending_finished(...) / notify_receiver_receive_data(...) —— 被 htsim
   flow 完成钩子调。                                                                   
  - is_sending_finished / is_receive_finished —— flow 完成去重计数器。
  - 三张哈希:sentHash / recvHash / expeRecvHash(跨文件共享状态的"总线")。
  - 还有 Mixnet/Prenet 需要的 g_moe_reconfig_mgr、reroute_flow_if_dead 等。            
                                                                                       
  Sys.cc 对外:                                                                         
  - Sys(NI, ...) 构造,接收 AstraNetworkAPI*,内部 new Workload。                        
  - front_end_sim_send/recv(...) —— Workload 和 CollectivePhase 发包统一入口,内部转调  
  NI->sim_send/recv。                                                                
  - generate_all_to_all / all_reduce / all_gather / reduce_scatter /                   
  generate_collective —— Layer 发起 collective 的入口。              
                                                                                       
  Workload.cc 对外:                                         
  - Workload(...) 构造,解析 workload.txt,产出 layers[]。                               
  - fire() —— 启动状态机,由 AstraSimNetwork 在全部 rank 都建好后统一调。
  - 状态机内部用 generator 指针回调 Sys 的 generate_* 和 front_end_sim_send/recv。  
## msg
### SchedulerUnit
### Ring的同步
真正发送是 Ring::ready(),它前面有个 4 条件门禁                                          
  只有下面全满足才真发:
  1. enabled:本 rank 没被 boost_mode 禁掉。       
  2. packets.size() > 0:队列里有待发的包。        
  3. stream_count > 0:phase 还没跑完。            
  4. free_packets > 0:本地 MemBus       
  A. packets:"有没有包要发"                       
                                                  
Ring::ready() 里挂 recv(ehd.event = PacketReceived)
       ↓ 存进 expeRecvHash(entry.h)
       ↓
       ↓  ── htsim 没收到左邻居数据,Ring 冻结 ──
       ↓  (此时 packets 可能已空,ready() 返回 false,没人往前推)
       ↓
  htsim 里左邻居的 TCP flow 完成
       ↓ tcp.cpp 的完成钩子
       ↓ → entry.h::notify_receiver_receive_data
       ↓    (从 expeRecvHash 取出 task1,调 handler)
       ↓ → Sys::handleEvent(ehd)
       ↓    (读 ehd.event == PacketReceived,走对应分支)
       ↓ → StreamBaseline::consume(ehd)
       ↓    (调 algorithm->run)
       ↓ → Ring::run(PacketReceived)
       ↓    ├─ total_packets_received++
       ↓    └─ insert_packet(nullptr)         ← 这里!
       ↓         └─ packets.push_back(MyPacket)
       ↓         └─ process_max_count → release_packets → PacketBundle
       ↓              └─ MemBus 跑延迟 → 回调 Ring::run(General)
       ↓                    ├─ free_packets++
       ↓                    └─ ready() → 终于能发下一轮
