# 生成拓扑以后
1. ECS 底座 FatTree(和 mixnet 结构一样)                                                                 
                                                                                                          
  先生成ecs的fattree的底座
                                                                                                          
  1. OCS overlay:Prenet 拓扑                                                                              
                                                                                                                                                                                              
                                                            
  2. 把参数灌进 PrenetConfig
                                                                                                          
  g_prenet_cfg.variant_pool_k        = params.prenet_variant_k;
  g_prenet_cfg.probe_ratio           = params.prenet_probe_ratio;                                         
  g_prenet_cfg.arbiter_window        = timeFromUs((double)params.prenet_arbiter_window_us);               
  g_prenet_cfg.confidence_init       = params.prenet_confidence_init;                                     
  g_prenet_cfg.confidence_max        = params.prenet_confidence_max;                                      
  g_prenet_cfg.predictor_log_every   = params.prenet_predictor_log_every;                                 
  g_prenet_cfg.link_speed_mbps       = params.speed;                                                      
  g_prenet_cfg.alpha                 = params.alpha;                                                      
  PrenetConfig 字段前面讲过(prenet_predictor.h:63-78)—— variant 池大小 K、probe 流比例、arbiter           
  时间窗、confidence counter 范围、log 频率、link 速度、α 等,都从 user_param 灌进来。这些参数只给         
  predictor / arbiter 用,不影响 Prenet 拓扑本身。                                                 
                                                                                                          
  4. 预计算 OCS 连接变体池 PrenetVariantPool                
                                                                                                          
    PrenetVariantPool(prenet_variant_pool.h:18)= 离线预计算的方案库                                         
  - 内部存 _variants[region_id][variant_id],每个元素是 ConnVariant{ conn_local[i][j] },就是 region 内部的 
  0/1 邻接矩阵                                                                                            
  - 构造时一次性 build_random_variant(...) 生成 region_num × K 张矩阵,之后只读                            
  - 不绑定 eventlist,不建 pipe/queue,纯数据结构                                                           
                                                                                                          
  Prenet(prenet.h:31 class Prenet : public Topology)= 运行时拓扑对象                                      
  - 持有真正的 conn(当前生效的邻接矩阵)+ pipes/queues/switchs(真实 htsim 网络组件)                        
  - 继承自 Topology,get_paths / get_eps_paths 是对外接口                                                  
  - 初始化时 prenet.cpp:98 所有 queue 带宽先设 0,"等 apply_variant 来填"                                         
                                                            
  5. 应用 index-0 作为初始 variant                                                                        
                                                            
  for (int r = 0; r < g_prenet_topo->region_num; r++) {                                                   
    g_prenet_topo->apply_variant(r, g_prenet_variants->default_variant(r));
  }                                                                                                       
  - prenet_variant_pool.h:24 default_variant(region_id) 返回该 region 的 index-0 方案
  - Prenet::apply_variant(region_id, ConnVariant)(prenet.h:45)把这张邻接矩阵装配到 OCS 上(建 pipe /       
  queue,就是第一张生效的 OCS 连接)                                                                 
                                                                                                          
  启动时 OCS 不是"空",而是按每个 region 的第 0 号 variant 联通起来 —— 这样第一个 flow 到来时已经有路径。
                                                                                                          
  6. Arbiter(时间窗仲裁)                                                                                  
                                                                                                          
  g_prenet_arbiter = new PrenetArbiter(eventlist, g_prenet_cfg.arbiter_window);                           
  PrenetArbiter 在 prenet_arbiter.h:12 class PrenetArbiter : public EventSource。职责:在一个              
  arbiter_window(默认 2µs,见 PrenetConfig.arbiter_window = 2·1000·1000                                    
  ps)里收集所有候选重配请求,挑一个赢家。                                                                  
                                                                                                          
  为什么要仲裁:多个 flow 可能同时建议重配同一 region,但实际只能重配一次(OCS 交换机物理限制),arbiter 做
  per-region per-window 的去冲突。败者在 prenet_predictor.cpp 里会被 erase_pending 清掉,避免错误学习。    
  
  7. TopoManager(实际驱动重配)                                                                            
                                                            
  g_prenet_topomanager = new PrenetTopoManager(
      g_prenet_topo, g_prenet_arbiter,                                                                    
      timeFromUs((double)params.reconf_delay_us), eventlist);
  g_prenet_topo->topomanager = g_prenet_topomanager;                                                      
  prenet_topomanager.h:44 class PrenetTopoManager。收 arbiter 选出的 winner → 调 Prenet::apply_variant
  换连接矩阵 → 用 reconf_delay_us 模拟 OCS 光交换的切换延迟 → 切换期间流量被 hold。                       
                                                                                   
  TopoManager 和 Arbiter 拆开:arbiter 做决策(谁赢),topomanager 执行决策(装配新 variant + 加 reconf        
  delay)。经典的 policy/mechanism 分离。                                                                  
  
  8. Predictor(决定下一步动作)                                                                            
                                                            
  g_prenet_predictor = new PrenetPredictor(g_prenet_cfg, g_prenet_topo, g_prenet_ecs_underlay);           
  前面那个问题讨论过的 TAGE 风格 3 分类动作预测器(STAY_ECS / USE_OCS_ASIS / RECONFIG_OCS)。每次跨机 flow  
  开始时 predict,flow 结束时 update 反馈。它需要同时看到 OCS 拓扑(决定 variant 选择)和 ECS                
  底座(counterfactual 对比)。                                                                             
                                                                                                          
  9. 把 TCP 重传卡死时的自救钩子指向 prenet 版本            
                                                                                                          
  TcpSrc::on_rtx_stuck = &reroute_flow_if_dead_prenet;
  - entry_prenet.h:98 inline bool reroute_flow_if_dead_prenet(TcpSrc* tcp) { ... }                        
  - mixnet 那边用的是 reroute_flow_if_dead(AstraSimNetwork.cc:676)                                        
  - 语义:某条 TCP 流重传多次仍没进展(大概率是 OCS                                                         
  重配把原路径切断了),直接重新计算路由、重挂新路径。两个拓扑要用不同的 reroute                            
  函数是因为路由查询方式不同(prenet 的 OCS overlay 结构和 mixnet 不同),作者注释里把 prenet / mixnet 用的  
  lambda 分离,正好也对齐到这条钩子                                                                        
                                                                                                          
  整体数据流                                                
                                                                                                          
  Prenet (OCS overlay, g_topology)
    ├─ elec_topology → FatTreeTopology (ECS underlay, 独立实例)                                           
    ├─ variant_pool  → PrenetVariantPool (预计算 K 个 region-level OCS 邻接)                              
    └─ topomanager   → PrenetTopoManager                                                                  
                         └─ 用 PrenetArbiter 选赢家 → 调 apply_variant 换 OCS 连接                        
                                                                                                          
  PrenetPredictor(独立对象,靠全局指针被 entry_prenet.h 调用)                                              
    ├─ 看 Prenet 拓扑和 ECS 底座                                                                          
    ├─ 每个 flow 开始 predict 3 分类动作 + 决策 id                                                        
    ├─ flow 结束 update 反馈(给 counter 打分)                                                             
    └─ 仲裁败者通过 erase_pending 丢弃                                                                    
                                                                                                          
  所以这 55 行完成的是:建好 OCS+ECS 双层拓扑 → 预计算 K 个 OCS 连接方案 → 装上初始方案 → 搭起 "predict →  
  arbitrate → apply" 三段式决策流水线。在 mixnet 里这一套是静态的("一次性随机连好,不再动"),在 prenet
  里是运行时动态的。        


## minxnet重新配置的触发:
### 这个管理器不是主动跑的,靠三处调用驱动:

  A. on_a2a_flow_start(每条跨机 a2a flow 启动时,从 SendFlow 调)

  SendFlow 里那行 g_moe_reconfig_mgr.on_a2a_flow_start(...) 就是这里。每条 a2a 进来做三件事:

  1. 累加到 layer_tm[(pass, layer, region)][local_src][local_dst] += flow_size
  2. pass 0 时:record_fwd_layer(layer)  收集 block 结构
  3. 如果是 dispatch 层的第一次(reconf_done 没见过):
        ├─ 查 prediction(上一 pass 同 block 的 TM)
        ├─ should_skip_reconfig? — top-N 热点是否已连通
        │      是 → skip,继续 append 到 demand_recorder,return
        │      否 → trigger_proactive_reconfig(...)
        │              └─ 把 prediction 塞给 topomanager
        │              └─ 让对应 region 的 OCS 开始切换,记录 reconfig_end_time
        │           return ReconfigResult{should_defer=true, reconfig_end_time}
  返回的 should_defer 决定当前这条 flow 是否被 park 到重配结束(前面讨论的 DeferredSendEvent 机制)。

  B. on_rank_pass_end(每个 rank pass 结束时,从 Workload.cc 的 hook 调)

  由前面讨论过的 on_rank_pass_end_hook 触发(Workload.cc:730):
  1. pass_end_ranks[pass].insert(rank)
  2. 凑够 total_ranks → pass_fully_drained.insert(pass)
  3. pass 0 first drained → freeze_structure()
  4. 对这 pass 的每个 block 调 close_block(pass, blk, region_size)
  同步点的语义很关键:全员报完 pass P 意味着 pass P 所有 a2a flow 都进过 on_a2a_flow_start 累加了,TM 才完整。

  C. close_block(被 B 内部调,也做幂等)

  1. block_closed[(pass, block)]? → skip
  2. 找出这个 block 的 2 个 layer(dispatch + combine)
  3. 对每个 region:
        合并这 2 个 layer 的 layer_tm → 存到 block_tm[(pass, block, rid)]
        如果这个 (block, rid) 的 last_block_tm 是更老的 pass → 覆盖
  4. 删掉已合并的 layer_tm 条目(释放内存)
  last_block_tm 只保留最新,下 pass 预测直接用它,不需要穿越历史。

  5. 完整流水(pass 1 预测 pass 0 的 dispatch0)

  [Pass 0 Forward 进行中]
     rank 42 发 dispatch0 的 a2a flow:
        on_a2a_flow_start(pass=0, layer=3, ...)
           ├─ record_fwd_layer(3)         ← 发现 layer 3 是 a2a
           ├─ layer_tm[(0,3,r)] += flow_size
           └─ structure_frozen==false → 不触发 reconfig

     (所有 rank 都在往 layer_tm[(0,3,r)] 累加)

     rank 42 发 combine0 的 a2a flow(layer=5):
        on_a2a_flow_start(pass=0, layer=5, ...)
           ├─ record_fwd_layer(5)
           └─ layer_tm[(0,5,r)] += flow_size

  [Pass 0 结束,每个 rank 陆续报告]
     on_rank_pass_end(rank=0, pass=0, ...)
     on_rank_pass_end(rank=1, pass=0, ...)
     ...
     on_rank_pass_end(rank=127, pass=0, ...)
        └─ 全员到齐:
           ├─ freeze_structure()          ← layer 3 → block 0 dispatch; layer 5 → block 0 combine
           ├─ close_block(pass=0, block=0, region_size):
           │     block_tm[(0,0,r)] = layer_tm[(0,3,r)] + layer_tm[(0,5,r)]
           │     last_block_tm[(0,r)] = block_tm[(0,0,r)].copy()
           │     last_block_tm_pass[(0,r)] = 0
           └─ 删 layer_tm 里 pass 0 的条目
  目的只有一个:做多 rank 同步栅栏,告诉
  manager"这个 rank 的 pass P 所有 a2a flow
  都已经 issue 完了"。

  等所有 rank 都报到齐,manager 才知道 "pass P
  的 layer_tm
  已经不会再有新字节进来",此时才安全地做两件跨
  rank 聚合的事:

  1. 关闭这个 pass 的所有 block:把 dispatch +
  combine 两个 layer 的 TM 合并成 block
  TM,晋升为 last_block_tm 供下一 pass
  的预测使用
  2. (仅 pass 0)冻结 block 结构:保证所有 rank
  都遍历过一次 forward,fwd_a2a_layer_order
  才完整,才能冻结
  ### 完整过程

  Pass 1 Forward 开始]
     fwd dispatch (layer=3, pair_pos=0):
        on_a2a_flow_start → accumulate + 触发重配 → defer
        DeferredSendEvent 挂在 reconfig_end_time
        重配完成后 replay SendFlow → 吃 OCS 大带宽 ✅

     fwd combine (layer=5, pair_pos=1):
        on_a2a_flow_start → 只 accumulate 到 layer_tm[(1,5,r)],pair_pos!=0 return
        flow 正常建流,走当前 OCS 配置(刚切完的那张,针对 dispatch 的热点)

  [Pass 1 Backward 开始]
     bwd combine (layer=5, pair_pos=1):
        on_a2a_flow_start → 只 accumulate 到 layer_tm[(1,5,r)](和 fwd combine 合流)
        不触发重配,走当前 OCS 配置
        ⚠ 设计文档期望这里用 fwd 的 real TM 重新重配,但代码没这么做

     bwd dispatch (layer=3, pair_pos=0):
        on_a2a_flow_start → accumulate + reconf_done.count((1,3)) 已存在 → return
        不触发重配,走当前 OCS 配置

  [Pass 1 全部结束 → on_rank_pass_end 凑齐]
     close_block(pass=1, block=0):
        block_tm[(1,0,r)] = layer_tm[(1,3,r)] + layer_tm[(1,5,r)]
                         ↑ fwd dispatch + bwd dispatch 合并    ↑ fwd combine + bwd combine 合并
        晋升为 last_block_tm[(0,r)]  供 pass 2 的 fwd dispatch 用
## a2a flow_start的完整过程
流程概览(8 步)                              
                                                  
  时间线示意(pass 1 的一个典型 block 的 dispatch)

  t=0           某 rank 发 pass=1 layer=3 的 a2a flow,SendFlow 调 on_a2a_flow_start
                │
                ├─ 累加 layer_tm[(1,3,r)] += size
                ├─ !frozen? 否(pass 0 drain 完了)它特指 "MoE block  结构的映射表已经建好
                ├─ layer 3 在 layer_to_block、pair_pos==0  ✓
                ├─ reconf_done 没见过 (1,3) → insert
                ├─ has_pred(查 last_block_tm[(0,*)]) → true
                ├─ should_skip_reconfig → false(top-N 缺)
                ├─ trigger_proactive_reconfig:
                │     for each region rid:
                │        rtm[rid].reconfig_end_time = 0 + reconf_delay_rid + 1
                │        rtm[rid] 挂到事件队列 now=0
                │     max_end = max(reconf_delay_rid) + 1
                └─ return {should_defer=true, reconfig_end_time=max_end}

  t=0           SendFlow 收到 should_defer:
                DeferredSendEvent * ev → sourceIsPending(ev, max_end)

  t=0..max_end  [重配窗口期]
                │ rtm[rid]->doNextEvent() 在 now=0 被调:算新 conn 矩阵,下发到 pipe/queue
                │ 所有新来的 a2a flow 走 SendFlow B 分支:
                │     rtm.reconfig_end_time > now → 打包新 DeferredSendEvent,挂 max_end
                │ 所有 a2a 都堆积在 max_end 那一刻

  t=max_end     所有 DeferredSendEvent 同时到期,replay SendFlow:
                g_replaying_deferred_flow=true → 跳过 manager、跳过 B 分支 →
                正常建 TCP flow,走切换完后的新 OCS 连接 → 吃大带宽

  t=max_end..   a2a flow 正常传输,flow_finish 时触发 notify_*
                pass 1 继续跑 combine / backward ...

  t=pass 1 drain 全员 on_rank_pass_end(*, 1) 到齐 → close_block(1, 0..) → last_block_tm 更新给 pass 2

  ### 所以"包在途遇到拓扑切换"具体会发生什么                 
                                          
  ┌─────────────────┬────────────────────────────────┐   
  │    包的位置     │        重配发生时的命运        │   
  ├─────────────────┼────────────────────────────────┤
  │ 还没 post 到    │ SendFlow defer,重配完再发      │   
  │ TCP             │                                │
  ├─────────────────┼────────────────────────────────┤   
  │ 在 TcpSrc 本地  │ TCP 被 pause_flow_in_region,停 │
  │ buffer(还没     │ 发;重配完 resume               │
  │ send_packets)   │                                │   
  ├─────────────────┼────────────────────────────────┤   
  │ 已经进 Pipe(链  │ 继续按旧路径到达目的地 —— 这条 │   
  │ 路传输中)       │  pipe 对象没变,只是对应的 OCS  │   
  │                 │ 映射变了                       │
  ├─────────────────┼────────────────────────────────┤
  │                 │ 留在 queue 里;重配后如果 queue │
  │ 已经进 Queue    │  的 bitrate=0,reroute 把 flow  │   
  │                 │ 改路;bitrate>0 则继续 drain    │
  ├─────────────────┼────────────────────────────────┤   
  │ 已经到          │                                │   
  │ TcpSink、待发   │ 正常返程                       │
  │ ACK             │                                │   
  └─────────────────┴────────────────────────────────┘
   Pipe —— 模拟"链路传播延迟"
   Queue —— 模拟"交换机端口的发送队列 + 带宽" 
   所以 queues[i][j] 实际表达的是 "假设 i-j 的 OCS        
  光路通了,从 i 侧 NIC 发往 j 侧 NIC 这条端到端链路的带宽
   + 排队行为"。OCS 重配时改 queues[i][j]._bitrate
  就是在说 "现在这条 NIC-to-NIC 的虚拟管道有没有带宽"。
  1. 基本拼装规则:每条单向链路 = Queue + Pipe             
  一个物理单向 hop 由两个组件串联构成:                                          
  Sender ──► [ Queue ] ──► [ Pipe ] ──► Next sink
              排队+串行化      传播延迟
  mixnet 的 OCS 是 直接                                
  machine-to-machine(经一次光交换就到对方):                                   
  Route: [ Queue_Mi→Mj, Pipe_Mi→Mj, TcpSink_at_Mj ]   

   已经在 Pipe 里的包不会莫名消失,因为 pipe    
    对象本身在、它的事件队列挂钩在、到达时间戳已经记录过了 
    。你拔光纤两端的插头,正在纤芯里飞的那个光子也会按原速到
    达对端(虽然物理上真拔光纤的话它会出纤芯漏出去,但这种建 
    模选择是"正在传的包视作已经上路")。                                                     
  所以代码对"tensor 在途时拓扑变了"的处理是:  
  - 拦得住的(新 flow)→ defer 延后                        
  - 拦不住的(已在 TCP buffer)→ pause 冻在 sender 侧
  - 已经 escape 的(进 pipe/queue)→                       
  要么继续跑,要么重配后判断"队列死了"就 reroute 到 ECS


  ### 对于重新配置:
    1. 代码证据 —— reroute_to 注释说得很直接    
                                                         
  tcp.cpp:195-213:                                       
  void TcpSrc::reroute_to(const Route *newfwd, const     
  }                                                      
                                          
  关键语义:
  - _route 换成新的 ECS 路径                             
  - 旧路径上已经在途(包括 OCS queue           
  里排队)的包:它们各自持有 const Route* 指针指向旧路径(在
   set_route 时捕获),不受 swap 影响                      
  - future sends / retransmits:用新的 _route,也就是 ECS  
  路径                                                 
                                                         
  1. 所以那些"困在 OCS queue 里的旧包"的真实命运
                                                         
  T = 重配前:                                            
    rank A 已经把 K 个包塞进 queues[i][j](OCS queue),
    它们的 pkt->_route 指针指向 OCS Route                
                                              
  T = 重配时:                                            
    queues[i][j]._bitrate = 0,永远不服务 → K 个包永久留在
   OCS queue          下次被重新接通,作为僵尸包被发送,数量很小可以忽略                                  
                                                         
  T = 重配后:                                            
    reroute_to → tcp->_route = ECS Route                 
                                              
  T = 过一段时间,TCP 收不到 ACK → RTO → retransmit       
    send_packets 重传,新建的包 set_route(tcp->_route) →
  指向 ECS Route                                         
    这些重传包从 host 侧 ECS egress queue 开始 enqueue,走
   fat-tree 路径到达   
  ### 僵尸包
  会被接收者自动丢弃,这是 TCP标准的重复包去重。
   ### sent flow
   SendFlow
    │
    ├─ TOPO_PRENET? → SendFlowPrenet return
    │
    ├─ portNumber++ / 统计累加
    │
    ├─ 同机 (src_m == dst_m)?
    │   └─ NVLink CallbackEvent 延迟 transfer_time → htsim_flow_finish
   [阶段3 return]
    │
    ├─ 主动重配 defer 检查(MoE a2a first flow)?
    │   └─ defer 到 reconfig_end_time  [阶段4 return]
    │
    ├─ 撞上重配窗口?
    │   └─ defer 到 reconfig_end_time  [阶段5 return]
    │
    ├─ 路由决策:conn[src_m][dst_m] > 0 且是 a2a?→ use_ocs=true
    │
    ├─ 建 DCTCPSrc + TcpSink + registerTcp
    │
    ├─ 取 Route:
    │   ├─ OCS:get_paths (双向都通则用)
    │   ├─ ECS:get_eps_paths (走 fat-tree)
    │   └─ 空就降级 ECS
    │
    ├─ 复制 Route + append endpoint
    ├─ flowSrc->connect(routeout, routein, sink, now)
    └─ waiting_to_{sent_callback,notify_receiver}[...]++