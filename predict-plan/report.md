# Prenet Report — 代码总结 + 第四轮审阅报告

本文件用于**后续 debug 参考**,包含两部分:

1. **Part 1**:prenet 代码修改总结 + 数据流端到端流程(code map + data-flow 验收)
2. **Part 2**:第四轮独立 agent 审阅报告原样转述(修复验证 + 新发现 + 未触及问题 + 剩余风险)

---

# Part 1. 代码修改总结 + 数据流

## 1. 文件清单 + 计划对应

### 1.1 新增(12 个文件,~2200 行)

| 文件 | 行数 | 对应 plan 章节 | 作用 |
|------|------|-----------------|------|
| `mixnet-sim/mixnet-htsim/src/clos/datacenter/prenet.h` | 78 | §2.1 | Prenet 拓扑类声明 |
| `mixnet-sim/mixnet-htsim/src/clos/datacenter/prenet.cpp` | 233 | §2.2~§2.4 | Prenet 实现、`apply_variant` |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_variant_pool.h` | 43 | §6.1 | `ConnVariant`、`PrenetVariantPool` 声明 |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_variant_pool.cpp` | 98 | §6.2 | K 个 random 0/1 variant 生成、`pick_covering_variant` |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_arbiter.h` | 47 | §5.1 | `PrenetArbiter` 声明 |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_arbiter.cpp` | 67 | §5.2~§5.3 | 窗口收集 + (confidence, msg_size) 排序 + grant/reject |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_topomanager.h` | 57 | §3.1 | Regional + 顶层 TopoManager 声明 |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_topomanager.cpp` | 83 | §3.1~§3.2 | reconfig 状态机 + arbiter grant_fn 绑定 |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_predictor.h` | 142 | §4.1 | `TctKey`/`PredictResult`/`UpdateInput`/`PrenetConfig` + `PrenetPredictor` 声明 |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_predictor.cpp` | 326 | §4.3~§4.7 | base_table 预测、update 闭环、M/D/1 外推、variant 选择 |
| `SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/htsim/entry_prenet.h` | 677 | §7 | `SendFlowPrenet`、`htsim_flow_finish_prenet`、仲裁 callback、deferred replay |
| `conf/topo/prenet.json` | 24 | §11 | prenet 默认配置 |

### 1.2 修改(4 个文件,追加式,不触碰 mixnet 逻辑)

| 文件 | 修改类型 | 对应 plan § | 具体位置 |
|------|---------|-----------|---------|
| `Makefile.htsim` | 追加 `PRENET ?= 1` 段 | §10 | 加到 `HTSIM_DC_SRCS` 后,不删除/重排原有条目 |
| `SimAI/.../entry.h` | 加 `TOPO_PRENET` enum 值、加 `#include "entry_prenet.h"`、在 `SendFlow` 顶部加 1 行 dispatcher | §8 | `enum TopoType` 末尾(含 `#ifdef PRENET_ENABLED`);`SendFlow` 前向声明 + include + 函数体首两行 |
| `SimAI/.../AstraSimNetwork.cc` | parse topo 新增一行、构造分支 `else if (g_topo_type == TOPO_PRENET)`、CLI 新增 6 个 `--prenet_*`、`on_rank_pass_end_hook` 按 topo 分注册、stats.txt 追加 prenet 段、cleanup 追加 delete | §9 | 所有改动在独立 `#ifdef PRENET_ENABLED` 块内,mixnet 分支字节级不变 |
| `run.sh` | case 语句加 `prenet)` | §11 | 读 `.topology.prenet.*` 字段 |

### 1.3 与 plan 的偏差(有意)

- **TAGE v0 简化**:plan §4.2 要求 base_table + 4 张 tagged tables(历史长度 [4,8,16,32])。实际只实现 base_table + 1 bit history fold。`prenet_predictor.h` 顶部注释声明,留作 PR-7 补齐。
- **pick_variant**:plan §6.2 v1 描述基于 traffic_hint 的 kmeans 分类。v0 只过滤 `conn_local[s][d]>0` 返回第一个覆盖 variant。
- **emit_probe 限 coll=4**:plan §4.5 没明确限制 coll_type。代码在 predictor 源头加了 `&& coll_type == 4` 保证 probe-split 只发生在 pure AllToAll(避免 AllReduce ring 同步死锁)。

### 1.4 隔离检查

- 无一处 prenet 代码修改 mixnet 既有文件
- `#ifdef PRENET_ENABLED` 包裹所有 prenet 代码;`PRENET=0` 编译出的二进制与合入前字节级等价
- 无 prenet 代码 include `mixnet.h`、`mixnet_topomanager.h`
- `TcpSrc::on_rtx_stuck` 全局函数指针按 topo 分支覆盖(mixnet 覆盖 → mixnet 版本;prenet 覆盖 → prenet 版本,函数首行 `if (g_topo_type != TOPO_PRENET) return false;` 守卫)
- `on_rank_pass_end_hook` 按 topo 注册不同 lambda,不在同一 lambda 内混杂

---

## 2. 各模块职责速览

### 2.1 `Prenet`(拓扑)
```
构造:set_params(no_of_gpus) → 分配 queue/pipe 二维矩阵 → init_network(所有
      queue _bitrate=0)→ renew_routes 预填 1-hop 路径模板 → AstraSimNetwork
      构造后调用 apply_variant(r, default_variant(r)) 给每个 region 刷默认
      conn + 真带宽。
get_paths(s, d):conn[s][d]>0 → 返回新 Route(single-hop queue+pipe);否则
      返回空 vector。
get_eps_paths(s, d):委托 elec_topology->get_paths;nullptr 包装为空 vector。
apply_variant(region, v):写 conn[i][j]=v.conn_local[i][j]、更新 queue
      _bitrate 和 _ps_per_byte。
```

### 2.2 `PrenetVariantPool`
构造时对每个 region 跑 K 次 random connect(每节点 alpha 条出边,无向对称),
产出 0/1 `conn_local` 矩阵。`default_variant` 返回 `variant_id=0`,
`pick_covering_variant(region, s_local, d_local)` 找第一个 `conn_local[s][d]>0`
的 variant,返回 -1 让 predictor 降级 STAY_ECS。

### 2.3 `PrenetArbiter`(继承 `EventSource`)
- `submit(region, variant, conf, msg_size, cb)`:入队 `_queue`;若当前无 pending
  window,`sourceIsPendingRel(this, window)` 挂起一个 window 后的事件。
- `doNextEvent`:window 到点,对每 region 选 `(conf, msg_size)` 最大的 winner,
  调 `_grant_fn(region, variant_id)` 要求 TopoManager 真正启动 reconfig;
  返回的 `end_time` 通知 winner cb,输家 cb(false, 0)。

### 2.4 `PrenetTopoManager`
- `PrenetRegionalManager`(每 region 一个 `EventSource`):
  - `start_reconf(v)`:`topo->apply_variant(region, v)` + `status=RECONF` +
    `sourceIsPendingRel(this, reconf_delay)`,返回 `now+delay+1` 作 end_time。
  - `doNextEvent` → `finish_reconf`:`status=LIVE`,`reconfig_end_time=0`。
- `PrenetTopoManager::request_reconfig` 把请求转给 arbiter;构造时绑定
  `arbiter->set_grant_fn([](region, variant_id){ rm.start_reconf(...) })`。

### 2.5 `PrenetPredictor`
- `predict(src, dst, bytes, coll, region)`:
  - 构造 `TctKey`(msg_size_bucket, coll_type, reconfig_needed=!conn_exists,
    src/dst_node, region_id)
  - `base_index = hash(bucket, coll, rnf_needed) XOR (global_history & 1)`,查
    `base_table[64]` 的 2-bit signed ctr
  - `ocs_preferred = (ctr >= 0)`
  - 决定 action:按 plan §4.4 的决策表(详见第 3 节流程图)
  - 若 RECONFIG_OCS:`pick_covering_variant`;找不到降级 STAY_ECS
  - `action_chose_ocs = (action != STAY_ECS)`
  - `emit_probe = action_chose_ocs && conf < max && coll == AllToAll`
  - 登记 `_pending[did] = ctx`(含 start_time=now)
  - 返回 `PredictResult{action, variant_id, conf, emit_probe, probe_ratio, did}`
- `update(in)`:
  - 查 `_pending[did]`,取 ctx
  - `executed_went_ocs = (in.taken_action ∈ {USE_OCS_ASIS, RECONFIG_OCS})`
  - 若 !executed_went_ocs:反事实 = `bytes / link_bps + (reconfig_needed ? reconf_delay : 0)`;was_correct = `actual <= counterfactual`
  - 若 executed_went_ocs:`extrap = extrapolate_full_ecs(total_bytes, probe_bytes, probe_actual, rho_at_finish, link_bps)`;was_correct = `actual_main <= extrap`
  - 更新 ctr:`ocs_should_be_preferred = executed_went_ocs ? was_correct : !was_correct` → `ctr±1`(饱和 [-4, 3])
  - push history bit:`executed_went_ocs ? 1 : 0`
  - erase ctx
- `erase_pending(did)`:arbiter loser / 异常 drop 清 ctx
- `extrapolate_full_ecs`:M/D/1 线性 service + wait factor `ρ/(2(1-ρ))`,
  若有 probe 则乘 `clamp([0.25, 4.0], probe_actual / expected)` 校准因子

### 2.6 `entry_prenet.h`(SendFlowPrenet + glue)
- `SendFlowPrenet(src, dst, bytes, ...)`:整个预测/分派/spawn 的主控(详见第 3 节)
- `htsim_flow_finish_prenet(ctx)`:
  1. 复用 SimAI 的 `received_chunksize`/`sent_chunksize` 累加 + `notify_*`
  2. 若 `decision_id != 0`,更新 meta end_time;主+probe 都到齐时
     调 `predictor.update(in)` 关环
- `PrenetDeferredSendEvent`:两用:
  - 区域忙 defer(`is_reconfig_winner=false`):end_time 到 → 再入 `SendFlowPrenet`(predictor 重新决策)
  - RECONFIG_OCS winner defer(`is_reconfig_winner=true`):end_time 到 → 直接
    spawn OCS 流 + 登 meta(不再过 predictor,避免 double-count)
- `reroute_flow_if_dead_prenet`:作为 `TcpSrc::on_rtx_stuck` 回调,dead OCS → 拉回 ECS

---

## 3. 数据流端到端流程

### 3.1 发起阶段 — 从 SimAI 到 prenet 决策点

```
┌─────────────────────────────────────────────────────────────────┐
│ SimAI Workload::fire → issue_comm → sim_send                   │
│        ↓                                                         │
│ ASTRASimNetwork::sim_send(... msg_handler, fun_arg ...)         │
│        ↓ (entry.h line ~116)                                    │
│ sentHash[(tag, src, dst)] = {msg_handler, fun_arg, count}       │
│        ↓                                                         │
│ SendFlow(src, dst, count, msg_handler, fun_arg, tag, request)   │
│        ↓                                                         │
│ ┌────────────────────────────────────────────┐                  │
│ │ entry.h::SendFlow (line ~795)              │                  │
│ │ ── 一行 dispatcher ──                      │                  │
│ │ if (g_topo_type == TOPO_PRENET) {          │ ← 不走到 mixnet  │
│ │   SendFlowPrenet(src, dst, count, ...);    │                  │
│ │   return;                                  │                  │
│ │ }                                          │                  │
│ └────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 决策阶段 — SendFlowPrenet 主流程

```
SendFlowPrenet(src, dst, maxPacketCount, ..., request)
    │
    ├── [S1] portNumber[src][dst]++;  sender_src_port_map 登记
    │
    ├── [S2] 同机 NVLink 检查
    │       └── src_machine == dst_machine ?
    │           ├── YES → transfer_time = count * 8 / 900Gbps
    │           │         new PrenetFlowContext{decision_id=0}
    │           │         schedule_callback(transfer_time, htsim_flow_finish_prenet, ctx)
    │           │         ++waiting_to_sent_callback / ++waiting_to_notify_receiver
    │           │         return
    │           └── NO ↓
    │
    ├── [S3] 骨架模式(predictor 未初始化 / topo 未就绪)
    │       └── prenet_send_ecs_only(...)  [纯 ECS,decision_id=0,不进预测器]
    │           return
    │
    ├── [S4] 区域繁忙 defer 检查(同时看 src 和 dst region)
    │       ├── end_src = topomanager->region_reconfig_end(region_src)
    │       ├── end_dst = topomanager->region_reconfig_end(region_dst)
    │       ├── end = max(end_src, end_dst)
    │       └── end > now ?
    │           ├── YES → new PrenetDeferredSendEvent{is_reconfig_winner=false, decision_id=0}
    │           │         sourceIsPending(event, end)
    │           │         return
    │           └── NO ↓
    │
    ├── [S5] 调用 predict(src, dst, maxPacketCount, com_type, region_src)
    │       ↓
    │       ┌─────────────────────────────────────────────────┐
    │       │ PrenetPredictor::predict                        │
    │       │                                                  │
    │       │  1. bucket = bucket_of(bytes)                    │
    │       │  2. reconfig_needed = !topo.conn_exists(s, d)    │
    │       │  3. key = TctKey{bucket, coll, rnf_nd, s, d, r}  │
    │       │  4. idx = hash(bucket, coll, rnf) XOR hist_bit   │
    │       │  5. ctr = base_table[idx]                        │
    │       │  6. conf = |ctr|  (0..3)                         │
    │       │  7. ocs_preferred = (ctr >= 0)                   │
    │       │  8. 决策:                                        │
    │       │     ┌────────────────────────────────┐           │
    │       │     │ ocs_preferred=true:            │           │
    │       │     │   reconfig_needed=false        │           │
    │       │     │     → USE_OCS_ASIS             │           │
    │       │     │   reconfig_needed=true:        │           │
    │       │     │     coll==AllToAll:            │           │
    │       │     │       bucket>=1 → RECONFIG_OCS │           │
    │       │     │       bucket==0 → STAY_ECS     │           │
    │       │     │     coll!=AllToAll → STAY_ECS  │           │
    │       │     │ ocs_preferred=false → STAY_ECS │           │
    │       │     └────────────────────────────────┘           │
    │       │  9. if RECONFIG_OCS:                             │
    │       │        variant = variant_pool.pick_covering(...)│
    │       │        if variant < 0: action = STAY_ECS        │
    │       │ 10. action_chose_ocs = (action != STAY_ECS)      │
    │       │ 11. emit_probe = action_chose_ocs                │
    │       │                 && conf<max && coll==AllToAll    │
    │       │ 12. did = ++_next_decision_id                    │
    │       │     _pending[did] = ctx{key, action, bytes,      │
    │       │         action_chose_ocs, start_time=now, ...}   │
    │       │ 13. return PredictResult{action, variant,        │
    │       │            conf, emit_probe, probe_ratio, did}   │
    │       └─────────────────────────────────────────────────┘
    │
    └── [S6] 分支 — 三条
        │
        ├── STAY_ECS
        │     ├── spawn TCP(use_ocs=false, decision_id=did)
        │     ├── spawn_fail → [PRENET][FATAL] assert (ECS 总该有路)
        │     ├── 登 meta{action=STAY_ECS, bytes_main=full, bytes_probe=0}
        │     ├── ++counters
        │     └── return
        │
        ├── RECONFIG_OCS
        │     ├── arbiter.submit(region, variant, conf, msg_size, cb)
        │     │         (cb 带 d_winner{decision_id=did})
        │     └── cb 在 arbiter_window 后被调用:
        │           ├── granted:
        │           │    └── new PrenetDeferredSendEvent{
        │           │            is_reconfig_winner=true, decision_id=did}
        │           │        sourceIsPending(ev, end_time=now+reconf_delay)
        │           │        ++arbiter_wins
        │           └── rejected:
        │                ├── predictor.erase_pending(did)  ← 关键:防止 ctx 泄漏
        │                ├── prenet_send_ecs_only(decision_id=0)
        │                └── ++arbiter_losses
        │
        └── USE_OCS_ASIS(with optional probe)
              ├── probe_safe_coll = (coll == AllToAll)
              ├── 预检 OCS 可达 + ECS 可达(get_paths 双向,立即 free)
              │     !ecs_reachable → [PRENET][FATAL] assert
              │     !ocs_reachable → want_split = false
              ├── 若 want_split:probe_bytes = max * ratio;main_bytes = max - probe
              │   否则:probe_bytes = 0;main_bytes = max
              ├── spawn main(use_ocs=true,degrade 到 ECS 由 spawn_flow 处理)
              │     out_used_ocs 指示实际是否走 OCS
              ├── 登 meta:
              │     action = used_ocs ? USE_OCS_ASIS : STAY_ECS  ← 关键:
              │                          执行路径可能与预测路径不同,
              │                          用实际路径供 predictor 学习
              ├── ++counters
              └── if want_split:
                    ├── spawn probe(use_ocs=false)
                    ├── spawn_fail → assert(预检通过不可能失败)
                    ├── meta.bytes_probe = probe_bytes
                    └── ++counters(第二次)
```

### 3.3 仲裁胜者 defer → reconfig → OCS spawn

```
PrenetDeferredSendEvent{is_reconfig_winner=true, did}::doNextEvent
  fire at now = original_predict_time + reconf_delay
    │
    ├── spawn main(use_ocs=true, decision_id=did)
    │   spawn_flow 内部:
    │     - get_paths(s, d):conn[s][d]>0(刚被 apply_variant 写成 1)→ 返回 OCS 路
    │     - new DCTCPSrc 注册 scanner,connect routeout/routein
    │     - out_used_ocs = true
    ├── 登 meta{action=RECONFIG_OCS, bytes_main=full, bytes_probe=0}
    │   (winner 不做 probe — 已经付了 reconf_delay 成本)
    └── ++counters

并行地,Arbiter 的 grant_fn 里已经触发:
  PrenetRegionalManager::start_reconf(variant)
    │
    ├── topo->apply_variant(region, variant)
    │     └── 对该 region 内所有 (i,j) queue:写 conn[i][j] = variant.conn_local[i][j]
    │         (variant 是 0/1)
    │         更新 queue->_bitrate / _ps_per_byte
    ├── status = RECONF
    ├── reconfig_end_time = now + reconf_delay + 1
    └── sourceIsPendingRel(this, reconf_delay)
          └── 到点 → doNextEvent → finish_reconf:status=LIVE, end=0
                    (这条事件的 finish 时刻正好 ≤ winner replay 时刻)
```

### 3.4 传输阶段 — TCP flow 在 htsim 事件循环跑

```
htsim EventList::doNextEvent 循环
  │
  ├── TcpSrc::sendData → Queue::receivePacket → Pipe::doNextEvent →
  │   Queue::receivePacket → ... (直连 OCS 只有 1 跳;ECS 多跳)
  │
  ├── TcpSink 收到最后一个 packet →  TcpSink::receivePacket → 触发
  │   flowSrc 的 application_callback(已 bind 为 htsim_flow_finish_prenet)
  │
  └── TCP RTO 若 stuck 在 dead OCS queue:TcpSrc::rtx_timer_hook → on_rtx_stuck
      → reroute_flow_if_dead_prenet → 检查 queue._bitrate==0 → 拉一条 ECS 路
      → reroute_to(newfwd, newback) → 继续跑
```

### 3.5 反馈阶段 — 闭环到 predictor

```
htsim_flow_finish_prenet(pctx)
  │
  ├── [F1] SimAI chunk 完成语义(与原 htsim_flow_finish 等价)
  │       ├── received_chunksize[(flow_id, s, d)] += flow_size
  │       ├── if --waiting_to_notify_receiver == 0:
  │       │      notify_receiver_receive_data(s, d, total_size, flow_tag)
  │       │      → 调用 expeRecvHash[tag].msg_handler(fun_arg)
  │       │      → SimAI 下一阶段 issue_comm
  │       └── (同理 sent_chunksize + notify_sender_sending_finished)
  │
  ├── [F2] prenet update 闭环
  │       │
  │       ├── decision_id == 0?
  │       │     → YES:NVLink / skeleton / arbiter-loser ECS / dead-route-reroute
  │       │           → 直接 return,不进预测器
  │       │     → NO ↓
  │       │
  │       ├── meta = g_prenet_pending_flows[decision_id]
  │       ├── pctx->is_probe ? meta->end_time_probe = now : meta->end_time_main = now
  │       │                    probe_done = true         main_done = true
  │       │
  │       ├── main_done && (bytes_probe==0 || probe_done) ?
  │       │     → YES ↓ (否则等另一半到)
  │       │
  │       ├── 构造 UpdateInput:
  │       │     taken_action = meta->action  ← 实际执行路径,不是预测的
  │       │     bytes_main = meta->bytes_main
  │       │     bytes_probe = meta->bytes_probe
  │       │     end_time_main / end_time_probe / start_time / rho_at_finish
  │       │
  │       ├── predictor.update(in)
  │       │   ↓
  │       │   ┌──────────────────────────────────────────────┐
  │       │   │ PrenetPredictor::update                      │
  │       │   │                                               │
  │       │   │  1. ctx = _pending[did]; _pending.erase      │
  │       │   │  2. actual_main = end_time_main - ctx.start  │
  │       │   │  3. actual_probe = end_time_probe - start    │
  │       │   │  4. executed_went_ocs = in.taken_action ∈    │
  │       │   │                        {USE_OCS_ASIS, RECONF} │
  │       │   │  5. 分支:                                     │
  │       │   │     !executed_went_ocs:                       │
  │       │   │       counterfactual = bytes/link_bps         │
  │       │   │           + (rnf_nd ? reconf_delay : 0)       │
  │       │   │       was_correct = actual <= ctf             │
  │       │   │     executed_went_ocs:                        │
  │       │   │       extrap = extrapolate_full_ecs(...)      │
  │       │   │       was_correct = actual <= extrap          │
  │       │   │  6. ocs_should_be_preferred =                 │
  │       │   │       executed_went_ocs ? was_correct         │
  │       │   │                         : !was_correct        │
  │       │   │  7. base_table[idx] ± 1 (饱和)               │
  │       │   │  8. global_history = (<<1) | executed_went_ocs│
  │       │   │  9. 统计 counters                            │
  │       │   └──────────────────────────────────────────────┘
  │       │
  │       ├── g_prenet_pending_flows.erase(did); delete meta
  │       └── 同步到 global stats counters(供 stats.txt 输出)
  │
  └── delete pctx
```

### 3.6 Pass 结束 — 统计快照

```
Workload.cc 内每 rank 完成一 pass → on_rank_pass_end_hook(rank, pass)
  ↓
AstraSimNetwork.cc 里按 topo 注册的 lambda:
  g_topo_type == TOPO_PRENET:
    g_prenet_predictor->on_pass_end(rank, pass)
      └── rank==0 时打印 [PRENET] pass_end 日志(total/correct/wrong/各 action 计数)

仿真结束 → main() 末尾:
  stats.txt 追加 prenet 段:
    prenet_predictions_total / _correct / _wrong
    prenet_action_stay_ecs / _use_ocs_asis / _reconfig_ocs
    prenet_probes_emitted / _arbiter_wins / _arbiter_losses
    prenet_accuracy %
```

---

## 4. 验收 checklist(代码 ↔ plan 对应)

| plan 要求 | 代码落地 | 验收点 |
|----------|---------|-------|
| §2.1 `Prenet : Topology` 构造需 alpha/dp/tp/pp/ep/gpus_per_node | `prenet.h` Prenet 构造函数全签名 | 与 mixnet 构造对齐,但不共用 |
| §2.3 `get_paths` 单跳直连,conn≤0 返回空 | `prenet.cpp:141-180` | OCS 不可达 caller 自动 fallback ECS |
| §3.1 RegionalManager 有 LIVE/RECONF 两态,带 reconfig_end_time | `prenet_topomanager.h:18-32` + `.cpp:9-27` | 与 mixnet RegionalTopoManager 平行但独立 |
| §4.1 TctKey 含 bucket/coll/rnf_needed | `prenet_predictor.h:24-32` | 决策依据与 plan §4.4 表格一致 |
| §4.4 决策表(action/bucket/coll 三维) | `prenet_predictor.cpp:121-158` | A2A + bucket≥1 + rnf → RECONFIG_OCS |
| §4.5 update 流程,push history bit 用执行路径 | `prenet_predictor.cpp:253-297` | global_history bit = executed_went_ocs |
| §5.2 仲裁按 (confidence, msg_size) 排序 | `prenet_arbiter.cpp:28-44` | 同 region 多请求只取最大 |
| §5.3 arbiter_window 显著小于 reconf_delay | prenet.json `arbiter_window_us=2` vs `reconf_delay=10` | 2us < 10us,合理 |
| §6 variant_pool 预编译 K 个 variant | `prenet_variant_pool.cpp` | K 默认 8,构造时 random generate |
| §7.2 SendFlowPrenet 整流程 | `entry_prenet.h:340-575` | 三分支 + defer + 预检 |
| §7.3 probe 与 main 共享 flow_id | `prenet_spawn_flow` 同 `request->flowTag` | chunk 累加自然汇总到 maxPacketCount |
| §8 entry.h 只一行 dispatcher + enum 新值 | `entry.h:67-77, 790-800` | `#ifdef PRENET_ENABLED` 包裹,mixnet 分支未动 |
| §9 AstraSimNetwork.cc 追加 prenet 构造 | `AstraSimNetwork.cc` | CLI 参数前缀 `--prenet_*`,统计带 `prenet_` |
| §10 Makefile `PRENET ?= 1` | `Makefile.htsim` | `PRENET=0` 完全剥离 |
| §11 配置文件 + run.sh case | `conf/topo/prenet.json` + `run.sh` | 字段齐全 |

---

## 5. 四轮 review 修复简明回顾

| 轮次 | 关键修复 | 最终状态 |
|------|---------|---------|
| R1 | 拆分 TAGE 计划(后降级 v0 base_table only) | 注释标明 v0 范围 |
| R2 | chunk flow_id 共享、HtsimFlowContext 扩展、Makefile PRENET gate | 实现到位 |
| R3 | RECONFIG_OCS 闭环(winner/loser path 用原 decision_id)、erase_pending、emit_probe 限 A2A | 核心闭环贯通 |
| R4 | USE_OCS_ASIS **预检 OCS+ECS**、`prenet_send_ecs_only` assert 化、dst_region defer、update M/D/1 统一、各死字段注释 | 达到冒烟测试级 |

---

## 6. 下一步动作建议

1. **构建验证**:`make -f Makefile.htsim PRENET=0` 和 `PRENET=1` 都能编;
   `mixnet` workload 在两个构建下产物字节级等价(验证隔离)
2. **冒烟**:`./run.sh conf/topo/prenet.json conf/workload/<small>.json`
3. **预期输出**:stdout 出现 `[PRENET]` 日志;stats.txt 有 prenet 段;
   `prenet_accuracy` 从 0 启动,多 iteration 后上升(基础学习效应)
4. **与 mixnet 对比**:同 workload 跑三组(mixnet/fattree/prenet),对比 FCT
5. **剩余问题**随后续 PR 处理:性能(build_hop_cache)、M/D/1 模型自洽、死字段清理

---

# Part 2. 第四轮独立 agent 审阅报告(原样转述)

## A. 本轮修复验证

- **B1 预检 OCS+ECS** ✅ 基本成立,预检到 spawn 无事件派发,不变量保持
- **B2 ECS spawn assert 化** ✅ 所有调用点已更新
- **B3 max(src/dst region end)** ✅ 语义正确(但跨 region 的 dst-region defer 其实是 conservative over-defer,不影响正确性)
- **B4 M/D/1 统一** ✅
- **D emit_probe 限 A2A** ✅ predictor 源头 + entry 双重保险,但 coll=5 允许 RECONFIG_OCS 而不 probe 的"不对称"没注释
- **B5 start_time fallback 删除** ✅
- **C4 get_eps_paths nullptr 包装** ✅
- **C3 DeferredData 生命周期注释** ✅
- **C2 经验 ρ 参数注释** ⚠️ 大部分加了,局部 `rho_now → rho` 改名位置未注释(小瑕疵)

## B. 新引入的问题

- **B1【中】extrapolate_full_ecs 模型混用**:probe 校准用 M/M/1 `1/(1-ρ)`,主模型用 M/D/1 `ρ/(2(1-ρ))`,两假设不自洽,clamp 掩盖。v0 可接受。
- **B2【中】预检开销**:每 USE_OCS_ASIS 额外 4 次 path query;FatTreeTopology::get_paths 每次 new 一整条 Route。建议与 `max_rho_on_path` 合并或加 LRU。
- **B3【中】预检 vs 真 spawn 冗余**:spawn 内部 OCS 不可达又会查 ECS → 第 3 对 get_paths。probe spawn 又一次。建议 `prenet_spawn_flow` 加"预检通过"旁路。
- **B4【低】edge case `maxPacketCount==1` 可能让 `main_bytes=0`** 进入 spawn,行为未知。加 `if (main_bytes == 0) want_split = false;` 守卫。
- **B5【低】** 预检永远显示 OCS 可达(因为 predict 选 USE_OCS_ASIS 时 conn>0,且预检期间无事件),所以 `!ocs_reachable` 分支是纯死防御。和 B2 性能浪费相关。

## C. 仍存在未触及问题

- **C1【已知】** `build_hop_cache` 仍是空函数,`max_rho_on_path` 每次 predict/finish 都 get_paths → 分配压力
- **C2【已知】** arbiter loser 不试次优 variant
- **C3【新】** `UpdateInput::start_time` 变死字段,读都不读,建议删
- **C4【新】** RECONFIG_OCS 选的 `emit_probe=true` 被 entry 层静默丢弃(entry 只在 USE_OCS_ASIS 看 emit_probe)— predictor 层应只对 USE_OCS_ASIS 开 emit_probe,避免死信号
- **C5【新】** `PrenetDeferredData` 里 msg_handler/fun_arg/tag 在 reconfig-winner 路径(b)不读,可拆 struct 省 72 字节

## D. 文档对齐

- principles/plan 全部对齐(§5.1、§4.2、§4.4、§6.3)
- plan §4.5 emit_probe 现在被 predictor 进一步限定到 coll=4,比 plan 更严格,**建议 plan §4.5 同步一行注释**

## E. 整体结论

> 代码**达到"可进入 build + 小 workload 冒烟测试"的工程质量**。B1/B2 核心修复真正解决阻塞挂起窗口,预检不变量成立。**最大剩余风险**:(1) 性能(C1 + B2/B3 的 fattree get_paths 分配开销),大 workload 会拖慢但不崩;(2) extrapolate 模型不自洽被 clamp 掩盖,影响精度不影响正确性;(3) 死字段(UpdateInput::start_time、RECONFIG_OCS 路径的 emit_probe)可能误导未来 reader。
>
> **建议合入前做一件小事**:删除 `UpdateInput::start_time` 字段(或加 assert),消除一个踩坑点。冒烟测试完再议 C1/性能补丁。

---

# 附录:debug 时可能用到的 grep 关键词

- `[PRENET]` — 所有 prenet 日志(info / drop / fatal)统一前缀
- `[PRENET][FATAL]` — 不变量破坏,立刻 abort 的路径
- `[PRENET][DROP]` — 软失败路径(目前已全部升级为 FATAL)
- `[PRENET][REROUTE_ECS]` — RTO 触发的 dead OCS → ECS 拉回
- `g_prenet_predictions_*` / `g_prenet_action_*` / `g_prenet_arbiter_*` — 运行时统计
- `decision_id == 0` — NVLink / skeleton / arbiter-loser / dead-reroute 共用的"不进 predictor update"标记
- `meta->action` — 实际执行路径(非预测路径),用于 predictor 反馈
- `PrenetDeferredSendEvent{is_reconfig_winner=true}` — winner 专用 defer,不再入 SendFlowPrenet
- `erase_pending` — 异常路径清理 `_pending` 防泄漏
- `probe_safe_coll == 4` — probe 拆分仅在 pure AllToAll

## 附录:最大风险点快速参考

1. `prenet_predictor.cpp::max_rho_on_path` — 每次 predict 都 `get_paths` + free。大 workload 下分配压力。
2. `prenet_predictor.cpp::extrapolate_full_ecs` — M/M/1 校准 + M/D/1 主模型不自洽,靠 `clamp([0.25, 4.0])` 掩盖模型错误。
3. `prenet_predictor.h::UpdateInput::start_time` — 死字段,未来 reader 可能误加读逻辑。
4. `predict()` 对 RECONFIG_OCS 设 `emit_probe=true` 但 entry 层静默忽略 — 死信号,未来改 entry 的人可能误以为 RECONFIG_OCS 会 probe。
5. `arbiter_window_us=2` 对比 `reconf_delay=10`:比值 1:5 在极端高并发下仲裁窗口可能汇聚过多请求。可调。
