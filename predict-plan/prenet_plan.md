# Prenet 实现计划

> 配套文件:`prenet_principles.md`(隔离/低耦合准则)、`plan.md`(SpecRing 原始构想)、
> `11_tage.md`(TAGE 预测器原理)。
>
> **本计划写出每个文件、每个类、每个函数的接口和职责;具体函数体保持伪代码,
> 真正落地时按当前代码风格(htsim C++17)编写。先按本计划与用户对齐,再动代码。**

---

## 0. 与现状的对照

### 0.1 已经存在(直接复用)
- `Topology` 抽象基类:`mixnet-htsim/src/clos/datacenter/topology.h:7`
  - 接口 `get_paths(src_gpu, dst_gpu)`、`get_eps_paths(src_gpu, dst_gpu)`、
    成员 `conn`(node × node 矩阵)。
- `FatTreeTopology`:作为 ECS 底座(prenet 同样需要)。
- `EventList` / `EventSource` / `simtime_picosec` 时间体系。
- `TcpRtxTimerScanner`、`DCTCPSrc`、`TcpSink`:发起 TCP flow 的标准方式(参考
  `entry.h::SendFlow`)。
- `Workload.cc` 通过 `on_pass_end_hook` / `on_rank_pass_end_hook` 通知一遍训练结束。
- CLI 解析 / log 目录建立 / `stats.txt` 输出框架(`AstraSimNetwork.cc`)。

### 0.2 新拓扑 prenet 与 mixnet 的关键差异
| 维度 | mixnet | prenet |
|------|--------|--------|
| OCS 触发条件 | 仅 `com_type == All_to_All` 且 `conn[s][d] > 0` | 任意 collective,基于 (size, type, conflict, history) 预测 |
| 决策粒度 | per-(pass, dispatch_layer),每对一次 reconfig | per-flow 三态:`STAY_ECS / USE_OCS_ASIS / RECONFIG_OCS` |
| 预测器 | 朴素:复用上一 pass 同 block 的 traffic matrix | TAGE 风格多表预测(短/长历史并存) |
| 验证 | 无,reconfig 后只看 throughput 总值 | 5% probe 拆分 + M/D/1 排队论外推 → 更新 saturating counter |
| 重构仲裁 | 无,每 region 顺序触发 | 仲裁器:同一时间窗内多 region 请求按 (confidence, msg_size) 排序 |
| Variant pool | 临时根据当下 traffic matrix 算一次 | 启动期 + warmup 预编译 K 个 variant,runtime 只 select |
| 失败保底 | reroute_dead_flows + RTO self-heal | 同上 + 任何环节失败都 fallback ECS |

### 0.3 预测器 hooks 的位置
- 决策点:`SendFlowPrenet`(替代 `SendFlow` 中的 mixnet 路径),在
  生成 TCP src 之前调用 `g_prenet_predictor.predict(...)`。
- 更新点:`htsim_flow_finish` 的 prenet 版本(`htsim_flow_finish_prenet`),
  flow 完成时调用 `g_prenet_predictor.update(...)`。
- 仲裁点:决策若是 `RECONFIG_OCS`,先把请求投到
  `g_prenet_arbiter.submit(...)`,等它在 arbiter_window_us 后回调 winner。
- 训练 pass 结束:用 `on_rank_pass_end_hook` 让预测器衰减 usefulness、
  打印一轮 confidence/accuracy 报表。

---

## 1. 文件清单

### 1.1 新增文件
| 文件 | 行数估计 | 主要内容 |
|------|---------|---------|
| `mixnet-sim/mixnet-htsim/src/clos/datacenter/prenet.h` | ~80 | `class Prenet : public Topology` 声明 |
| `mixnet-sim/mixnet-htsim/src/clos/datacenter/prenet.cpp` | ~400 | `Prenet` 实现,init/get_paths/get_eps_paths/apply_variant |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_topomanager.h` | ~80 | `class PrenetTopoManager`、`PrenetRegionalManager` |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_topomanager.cpp` | ~400 | reconfig 状态机、queue bw 更新、defer 队列 |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_predictor.h` | ~120 | `struct TctKey`、`class PrenetPredictor`(TAGE) |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_predictor.cpp` | ~500 | TAGE table、tag/index hash、predict/update |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_arbiter.h` | ~50 | `class PrenetArbiter` |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_arbiter.cpp` | ~150 | submit/window/排序 |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_variant_pool.h` | ~60 | `struct ConnVariant`、`class VariantPool` |
| `mixnet-sim/mixnet-htsim/src/clos/prenet_variant_pool.cpp` | ~250 | precompute K variants、select、replace |
| `SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/htsim/entry_prenet.h` | ~600 | `SendFlowPrenet`、`htsim_flow_finish_prenet`、`g_prenet_*` 全局 |
| `conf/topo/prenet.json` | ~25 | 预设 prenet 配置 |
| `predict-plan/prenet_plan.md` | (本文) | — |
| `predict-plan/prenet_principles.md` | (已写) | 隔离准则 |

### 1.2 修改文件(只追加,不改原有行)
| 文件 | 修改 | 注意 |
|------|------|------|
| `Makefile.htsim` | 新增 `PRENET ?= 1` 段、prenet 源文件追加 | 见 `prenet_principles.md` §6 |
| `SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/htsim/entry.h` | (a) `enum TopoType` 末尾加 `TOPO_PRENET`;(b) `SendFlow` 顶部加唯一一行 dispatcher | 不动 mixnet 任何分支 |
| `SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/htsim/AstraSimNetwork.cc` | (a) `topo_name == "prenet"` 解析;(b) 新建 prenet 拓扑分支;(c) 新建 prenet 的 `on_rank_pass_end_hook` lambda(条件注册);(d) 新参数 `--prenet_*`;(e) `stats.txt` 中 prenet 段 | 不改 mixnet 已有分支 |
| `run.sh` | 无需改(读 `.topology.type` 已经通用) | 仅添加 `conf/topo/prenet.json` 即可 |

---

## 2. `prenet.h` / `prenet.cpp`

### 2.1 类声明草图

```cpp
// prenet.h
#pragma once
#ifdef PRENET_ENABLED
#include "topology.h"
#include "fat_tree_topology.h"
#include "eventlist.h"
#include "switch.h"
#include <vector>
#include <unordered_map>

class PrenetVariantPool;       // forward
class PrenetTopoManager;       // forward

class Prenet : public Topology {
public:
  // 构造参数与 Mixnet 类似 + variant pool 大小、predictor 配置
  Prenet(int no_of_gpus, mem_b queuesize, EventList& ev,
         queue_type qt, simtime_picosec reconf_delay,
         FatTreeTopology* elec_topology,
         int alpha, int dp, int tp, int pp, int ep,
         int gpus_per_node, int variant_pool_k);

  // Topology 接口
  vector<const Route*>* get_paths(int src_gpu, int dst_gpu) override;
  vector<const Route*>* get_eps_paths(int src_gpu, int dst_gpu) override;
  vector<int>* get_neighbours(int src) override { return nullptr; }
  int no_of_nodes() const override { return _no_of_nodes; }

  // prenet 专用接口
  void apply_variant(int region_id, const ConnVariant& v);     // 切换 conn + 路由 + 队列带宽
  bool circuit_exists(int src_node, int dst_node) const;        // O(1) 查 conn
  uint64_t get_link_bitrate(int src_node, int dst_node) const;
  uint64_t get_queue_size(int src_node, int dst_node) const;    // 给 predictor 估排队论用

  // 数据
  int _no_of_nodes;
  vector<vector<Pipe*>> pipes;
  vector<vector<Queue*>> queues;
  vector<Switch*> switchs;
  FatTreeTopology* elec_topology;            // ECS underlay,prenet **新建独立实例**
  int alpha, dp_degree, tp_degree, pp_degree, ep_degree, gpus_per_node;
  int region_size, region_num;
  simtime_picosec reconf_delay;
  EventList& eventlist;

  PrenetVariantPool*  variant_pool  = nullptr;   // 注入,Prenet 持有指针不持有所有权
  PrenetTopoManager*  topomanager   = nullptr;   // 同上,反向指针给 ECNQueue::set_*

private:
  void init_network();         // 镜像 Mixnet::init_network 的结构,但用独立 routes map
  void renew_routes(int s, int e);
  std::unordered_map<uint64_t, std::vector<std::vector<size_t>*>> _routes;
  mem_b _queuesize;
  queue_type _qt;
};
#endif // PRENET_ENABLED
```

### 2.2 init 与 mixnet 的差别
- mixnet 启动会跑一次 `random_connect` / `weighted_connect` 算 conn。
  prenet 启动改为:**先创建空 conn**,然后立即 `apply_variant(region, variant_pool->default_variant(region))`。
  `default_variant` 由 variant_pool 在自己构造里跑 random connect 给出。
- 这样 prenet 不跟 mixnet 共享 random/weighted connect 的实现,
  代码重复但完全独立。

### 2.3 `get_paths` / `get_eps_paths`
- 与 mixnet 几乎相同(直连 OCS, 单跳;ECS 走 elec_topology),
  唯一区别:任何 `conn[s][d] == 0` 时直接返回**空 vector**,
  调用方(`SendFlowPrenet`)负责改走 `get_eps_paths`。
- **不抛 assert**,以满足 `principles §5`。

### 2.4 `apply_variant`
```text
1. for (i,j) in region: 把目标 conn 写入 topo->conn[i][j]
2. 调用 update_regional_queue_bandwidth(region_id):仿 mixnet 的同名函数
3. renew_routes(start, end):仿 mixnet renew_routes_ocs
4. 不主动 reroute 现存 flows;与 mixnet 一致,by design 让旧 flow 跑完
   (PrenetTopoManager 在 reconfig 期间用 defer 机制阻止新 flow 产生)
```

---

## 3. `prenet_topomanager.h` / `.cpp`

### 3.1 类
- `PrenetRegionalManager : public EventSource`
  - 生命周期:每 region 一个,与 mixnet `RegionalTopoManager` 平行
  - 状态机:`LIVE` ↔ `RECONF`
  - 接口:
    - `void start_reconf(const ConnVariant& target)`
    - `void finish_reconf()` (`doNextEvent` 切换状态)
    - `simtime_picosec reconfig_end_time` 给 entry_prenet.h 用做 defer 截止
  - **不引入 mixnet 的 traffic matrix 概念**:reconf 的"目标 conn"由 caller
    (`PrenetArbiter`)直接传 variant。
  - **复制粘贴并改写**:从 `mixnet_topomanager.cpp` 的 `start_reconf`、
    `update_regional_queue_bandwidth`、`reroute_dead_flows`、
    `set_regional_tcp_pause` 借鉴写法,但所有指针/类型换成 prenet 的。
    遵循 `principles §9`:不 include / 不继承 mixnet 头文件。

- `PrenetTopoManager`
  - 持有 `vector<PrenetRegionalManager*> regional_managers`
  - 接口:
    - `void request_reconfig(int region_id, const ConnVariant& v, double confidence, uint64_t msg_size, std::function<void(bool granted, simtime_picosec end_time)> cb)`
    - 内部:把请求转给 `PrenetArbiter`,arbiter 决定胜者后再 `start_reconf`,
      cb(true, end_time) 给胜者,cb(false, 0) 给败者。
  - **不**直接管 traffic matrix,与 mixnet 完全解耦。

### 3.2 reroute_dead_flows
- 同 mixnet:flow 路径上某 hop `_bitrate == 0` 视为死路 → reroute 到 ECS。
- 用 prenet 自己持有的 fattree,而不是 mixnet 的。

---

## 4. `prenet_predictor.h` / `.cpp`

### 4.1 数据结构

```cpp
// prenet_predictor.h
struct TctKey {
  uint8_t  msg_size_bucket;   // 2 bit (0..3)
  uint8_t  coll_type;         // 3 bit (ComType enum cast)
  uint8_t  reconfig_needed;   // 1 bit (=0 if conn[s][d]>0, else 1)
  uint16_t src_node;          // 仅参与 hash 不参与桶分类
  uint16_t dst_node;
  uint8_t  region_id;
  // 12 bit 索引基础;TAGE 的 history 在 hash 时叠加
};

enum class PrenetAction : uint8_t {
  STAY_ECS         = 0,
  USE_OCS_ASIS     = 1,
  RECONFIG_OCS     = 2,
};

struct PredictResult {
  PrenetAction action;
  int          variant_id;     // 若 action == RECONFIG_OCS,目标 variant
  int          confidence;     // 0..3 (saturating)
  bool         emit_probe;     // 若 action != STAY_ECS 且 conf < CONF_MAX 时可设 true
  double       probe_ratio;    // = g_prenet_cfg.probe_ratio if emit_probe
  uint64_t     decision_id;    // 64bit 唯一 ID,update 时用来回查
};

struct UpdateInput {
  uint64_t  decision_id;
  PrenetAction taken_action;
  uint64_t  bytes_main;
  uint64_t  bytes_probe;        // 0 if no probe
  simtime_picosec start_time;
  simtime_picosec end_time_main;     // 主 flow 完成时间
  simtime_picosec end_time_probe;    // probe 完成时间(若有)
  uint64_t  measured_link_bw;        // 主路径平均带宽(用 measured)
  double    fattree_rho_at_decision; // 决策时 ECS 利用率
  double    fattree_rho_at_finish;   // 完成时 ECS 利用率
};
```

### 4.2 TAGE 表布局
- `base_table`:64 entry,每 entry 是 2-bit signed counter(>=0 → OCS, <0 → ECS)。
  index = `hash(coll_type, msg_size_bucket, reconfig_needed)`。
- `t1..t4`:tagged tables,history lengths `[4, 8, 16, 32]`,可由配置覆盖。
  每张表 256 entry,entry = `{tag(10 bit), ctr(3 bit signed), u(2 bit)}`。
- `global_history`:32 bit ring(每个 collective 完成后 push 1 if "OCS 选对" else 0)。
- 索引/tag 用 `xor folded compressed history`(参考 `11_tage.md` 的伪代码)。

### 4.3 接口

```cpp
class PrenetPredictor {
public:
  PrenetPredictor(const PrenetConfig& cfg, Prenet* topo, FatTreeTopology* ecs);

  // 决策入口:flow 起始时调用
  PredictResult predict(int src_gpu, int dst_gpu, uint64_t bytes,
                        AstraSim::ComType coll_type,
                        int region_id);

  // 完成回调:flow finish 时调用
  void update(const UpdateInput& in);

  // pass 结束时被 hook 调用,做 usefulness 衰减、打印统计
  void on_pass_end(int rank, int pass);

  // 调试/统计快照
  void dump_stats(std::ostream& out) const;

private:
  // ---- key 构造 ----
  uint8_t bucket_of(uint64_t bytes) const;       // 用 cfg.msg_size_buckets_kb 边界
  TctKey  build_key(int src_gpu, int dst_gpu, uint64_t bytes,
                    AstraSim::ComType ct, int region_id) const;

  // ---- TAGE 内部 ----
  uint32_t base_index(const TctKey& k) const;
  uint32_t tagged_index(int t, const TctKey& k) const;
  uint16_t tagged_tag  (int t, const TctKey& k) const;

  PredictResult tage_predict(const TctKey& k);
  void          tage_update (const TctKey& k, PrenetAction was, bool was_correct);

  // ---- variant 选择 ----
  // 决定 RECONFIG_OCS 时,从 variant_pool 选一个 variant_id
  int  pick_variant(int region_id, const TctKey& k);

  // ---- 排队论外推 ----
  // 给定 probe 实测延迟 + ECS 利用率,算"全部走 ECS 的延迟"
  simtime_picosec extrapolate_full_ecs(uint64_t main_bytes, uint64_t probe_bytes,
                                       simtime_picosec probe_actual,
                                       double rho_now, double link_capacity_bps);

  // ---- 决策 ID -> 上下文,update 时回查 ----
  std::unordered_map<uint64_t, struct DecisionCtx> _pending;
  uint64_t _next_decision_id = 1;

  // ---- 表 ----
  std::vector<int8_t>  base_table;                 // [64] signed counter
  struct TaggedEntry { uint16_t tag; int8_t ctr; uint8_t u; };
  std::vector<std::vector<TaggedEntry>> tagged;    // [num_tables][table_size]
  uint32_t global_history = 0;                     // 32 bit
  uint64_t history_bits   = 32;

  // ---- 配置 / 引用 ----
  const PrenetConfig& cfg;
  Prenet* topo;                  // 用来查 conn / 链路带宽 / 队列长度
  FatTreeTopology* ecs;          // 用来查 ECS 队列利用率

  // ---- 统计 ----
  uint64_t total_predictions = 0;
  uint64_t correct_predictions = 0;
  std::array<uint64_t, 3> action_counts {0,0,0};
};
```

### 4.4 `predict()` 流程
```text
1. k = build_key(src_gpu, dst_gpu, bytes, coll_type, region_id)
2. (provider, alternate, base_pred) = tage_predict(k)
3. 决定 action:
   - 如果 base_pred 对应"OCS 优",且 reconfig_needed == 0 → USE_OCS_ASIS
   - 如果 base_pred 对应"OCS 优",且 reconfig_needed == 1:
       - **v0 安全约束:仅 coll_type == AllToAll 才允许 RECONFIG_OCS**;
         其它 collective(AllReduce/AllGather/ReduceScatter)→ STAY_ECS。
         理由:AllReduce ring 算法跨 region 同步点会被 defer 卡住。
       - 满足上面约束时:
         - msg_size_bucket >= 2 → RECONFIG_OCS
         - msg_size_bucket == 1 → RECONFIG_OCS (A2A 灰色地带也尝试)
         - msg_size_bucket == 0 → STAY_ECS (小消息不值得)
   - 否则 → STAY_ECS
4. confidence = saturated counter 的绝对值
5. emit_probe = (confidence < cfg.confidence_max) && (action == USE_OCS_ASIS || action == RECONFIG_OCS)
   - 注:仅在选 OCS 时 probe 才有意义(probe 是少量 ECS 副本,验证 ECS 替代方案)
6. variant_id = pick_variant(region_id, k, src_node, dst_node) if action == RECONFIG_OCS else -1
   - **pick_variant 必须只考虑 conn_local[src_local][dst_local] > 0 的 variant**;
     若 K 个 variant 都没覆盖 src→dst 这条边,降级 action = STAY_ECS,decision_id 仍登记
     以便后续 update 学到"这条 pair 没办法 reconfig"。
7. 把 (k, action, src_gpu, dst_gpu, bytes, coll_type, decision_id, ts, action_chose_ocs) 存进 _pending
8. 返回 PredictResult
```

### 4.5 `update()` 流程
```text
1. ctx = _pending[in.decision_id]; _pending.erase(...)
2. 计算 truth:
   - case STAY_ECS:
       counterfactual_ocs = bytes / (alpha * SPEED_bps) + (reconfig_needed ? reconf_delay : 0)
       was_correct = (ecs_actual <= counterfactual_ocs)
   - case USE_OCS_ASIS / RECONFIG_OCS:
       若有 probe(probe_bytes > 0):
         extrapolated_full_ecs = extrapolate_full_ecs(bytes, probe_bytes, probe_actual, rho_at_finish, link_bw)
       否则:
         extrapolated_full_ecs = bytes / link_bw   // 退化到无拥塞 ECS,保守
       was_correct = (ocs_actual <= extrapolated_full_ecs)
3. tage_update(ctx.key, ctx.action, was_correct)
4. **global_history = (global_history << 1) | (ctx.action_chose_ocs ? 1 : 0)**
   - push 的是"这次行动是否选了 OCS",这是 TAGE 学习的"taken bit"
   - was_correct 用来更新 ctr 符号,不进 history
5. 累计统计 + 每 cfg.predictor_log_every 次打印一次 [PRENET] 摘要
```

### 4.6 `extrapolate_full_ecs` 排队论
```text
- 输入:bytes(主流量), probe_bytes, probe_actual_ps, rho_now, link_capacity_bps
- 模型(M/D/1 近似):
       service_time = bytes / link_bw                                # 静态发送时间
       rho_full     = min(0.99, rho_now + main_bytes / window / link_bw)
       wait_factor  = rho_full / (2 * (1 - rho_full))                # M/D/1 期望等待
       T_full       = service_time * (1 + wait_factor)
- 若有 probe:校准因子 cal = probe_actual / (probe_bytes / link_bw / (1 - rho_now + eps))
  T_full *= cal   # 用 probe 把模型偏差校准回来
- 返回 simtime_picosec
```
> v0 工程近似;`prenet_principles.md §10` 允许后续替换为更准的模型,接口签名稳定。

### 4.7 ECS 利用率获取(v0 取路径上 max ρ)
- prenet 持有自己的 `FatTreeTopology* ecs`。
- `PrenetPredictor` 构造时通过 `ecs->get_paths(0,1) ... get_paths(N-1,N-2)` 缓存
  典型路径上经过的 `Queue*` 集合。
- 计算 ρ 时:对 src→dst 一条 ECS 路径上的 **每一跳 Queue 取 max** ρ
  (`enqueued bytes / queue_capacity`),作为瓶颈 ρ。
  - 理由:fattree 真正瓶颈在 agg/core 层,单 access 端口低估太多。
- 实现细节:为避免每次 `predict` 都重新查路径,prenet 启动时一次性建立
  `pair_to_hop_queues[src_machine][dst_machine] = vector<Queue*>` 缓存。

### 4.7.1 `pick_variant`
```text
输入:region_id, key, src_local, dst_local
1. 遍历 variant_pool->size_per_region() 个 variant
2. 仅保留 conn_local[src_local][dst_local] > 0 的候选
3. 候选为空 → 返回 -1(caller 把 action 降级 STAY_ECS)
4. 候选不为空 → 用 base_table 的 confidence 作 hint,选第一个候选
   (v1 可改为基于 traffic_hint 选最匹配的)
```

---

## 5. `prenet_arbiter.h` / `.cpp`

### 5.1 接口

```cpp
class PrenetArbiter : public EventSource {
public:
  PrenetArbiter(EventList& ev, simtime_picosec window);

  // submit 一个 reconfig 请求;hub 在 window 后调用 cb 通知 winner/loser
  void submit(int region_id, const ConnVariant& v,
              double confidence, uint64_t msg_size,
              std::function<void(bool granted, simtime_picosec end_time)> cb);

  void doNextEvent() override;   // window 到点 → 排序 → 通知

private:
  struct Pending {
    int region_id;
    ConnVariant variant;
    double confidence;
    uint64_t msg_size;
    std::function<void(bool, simtime_picosec)> cb;
  };
  std::vector<Pending> _queue;
  bool _scheduled = false;
  simtime_picosec _window;
};
```

### 5.2 排序规则(`plan.md §4.2`)
1. 比 confidence(高赢)
2. 比 msg_size(大赢)
3. 同 region_id 的请求合并:取 confidence/msg_size 最大那条,弃其余
4. 不同 region 之间互不冲突 → 都赢

如果 winner 区域已在 reconfig(`reconfig_end_time > now`),winner 这次也回 cb(false)
(避免叠加),由 caller 自己 fallback ECS。

### 5.3 排队论的"window"
- `cfg.arbiter_window_us`:典型 1~5 us(光开关切换尺度)
- 太大 → flow 等不起;太小 → 没机会聚合
- 必须显著小于 `reconf_delay`,否则 arbiter 没意义。

---

## 6. `prenet_variant_pool.h` / `.cpp`

### 6.1 类

```cpp
struct ConnVariant {
  int variant_id;
  std::vector<std::vector<int>> conn_local;  // [region_size][region_size]
  std::array<uint64_t, 4> expected_bw_summary;  // (avg, p50, p95, max) for stats
};

class PrenetVariantPool {
public:
  PrenetVariantPool(int region_size, int region_num, int alpha, int K);

  const ConnVariant& default_variant(int region_id) const;
  const ConnVariant* get(int region_id, int variant_id) const;  // null if id 越界
  int size_per_region() const { return _K; }

  // 选最匹配 traffic_hint 的 variant
  // (predictor.pick_variant 的实现 fallback)
  int select_best(int region_id, const std::vector<std::vector<double>>& traffic_hint) const;

  // warmup 阶段在线增删 variant(`plan.md §11.3`)
  // v0:不支持运行时增删,只在构造里固定 K 个 variant
  // v1:on_pass_end 时调用 try_add_variant() 收集尾段 traffic 加新 variant
  void try_add_variant(int region_id, const std::vector<std::vector<double>>& tm);

private:
  int _region_size, _region_num, _alpha, _K;
  // [region_id][variant_id] → ConnVariant
  std::vector<std::vector<ConnVariant>> _variants;

  // 复制粘贴 mixnet 的 random_connect 到这里(隔离准则)
  ConnVariant build_random_variant(int region_id, int variant_id, std::mt19937& rng);
};
```

### 6.2 v0 实现
- 构造时为每个 region 生成 K 个 random_connect variant(K=8 作为默认)。
- variant 0 = 默认 variant,Prenet 启动时 apply。
- `select_best` v0 用贪心:对 traffic_hint 取 top-N pairs,看每个 variant
  覆盖了多少,选覆盖最多的。

### 6.3 v1 优化(后续扩展,本计划不实现)
- 启动 warmup phase(前 1 个 pass)记录 traffic matrix → kmeans 出 K class →
  每个 class 跑一次 `regional_topo_reconfig` 算法生成 variant。
- runtime 增量:用 `try_add_variant` 替换最少使用的 variant(LRU)。

---

## 7. `entry_prenet.h`

### 7.1 全局
```cpp
#pragma once
#ifdef PRENET_ENABLED
#include "prenet.h"
#include "prenet_topomanager.h"
#include "prenet_predictor.h"
#include "prenet_arbiter.h"
#include "prenet_variant_pool.h"
#include "tcp.h"
#include "dctcp.h"

// 拓扑指针
Prenet*               g_prenet_topo        = nullptr;
PrenetTopoManager*    g_prenet_topomanager = nullptr;
PrenetPredictor*      g_prenet_predictor   = nullptr;
PrenetArbiter*        g_prenet_arbiter     = nullptr;
PrenetVariantPool*    g_prenet_variants    = nullptr;
FatTreeTopology*      g_prenet_ecs_underlay = nullptr;   // prenet 独立持有的 fattree

// prenet 配置
struct PrenetConfig {
  int    variant_pool_k       = 8;
  std::array<int,4> tage_history_lengths = {4,8,16,32};
  double probe_ratio          = 0.05;
  int    confidence_init      = 1;
  int    confidence_max       = 3;
  std::array<uint64_t,3> msg_size_buckets_bytes = {64*1024, 1024*1024, 16*1024*1024};
  simtime_picosec arbiter_window = timeFromUs(2);   // 与 conf/topo/prenet.json 一致
  uint64_t predictor_log_every = 1000;
};
PrenetConfig g_prenet_cfg;

// 统计
uint64_t g_prenet_predictions_total      = 0;
uint64_t g_prenet_predictions_correct    = 0;
uint64_t g_prenet_predictions_wrong      = 0;
uint64_t g_prenet_probes_emitted         = 0;
uint64_t g_prenet_action_stay_ecs        = 0;
uint64_t g_prenet_action_use_ocs_asis    = 0;
uint64_t g_prenet_action_reconfig_ocs    = 0;
uint64_t g_prenet_arbiter_wins           = 0;
uint64_t g_prenet_arbiter_losses         = 0;

// flow 元数据(predict 与 update 之间挂)
struct PrenetFlowMeta {
  uint64_t        decision_id;
  PrenetAction    action;
  uint64_t        bytes_main;
  uint64_t        bytes_probe;
  simtime_picosec start_time;
  double          rho_at_decision;
  TcpSrc*         main_tcp;
  TcpSrc*         probe_tcp;       // null 若没有 probe
  AstraSim::ncclFlowTag flow_tag;  // 给 finish 回调用
  int src, dst;
};
// key = decision_id
std::unordered_map<uint64_t, PrenetFlowMeta*> g_prenet_pending_flows;
```

### 7.2 `SendFlowPrenet` 流程

```cpp
void SendFlowPrenet(int src, int dst, uint64_t bytes,
                    void (*msg_handler)(void*), void* fun_arg,
                    int tag, AstraSim::sim_request* request) {
  // 1. NVLink 同机:直接复用 entry.h 现有 NVLink 路径
  int src_machine = src / g_gpus_per_server;
  int dst_machine = dst / g_gpus_per_server;
  if (src_machine == dst_machine) { /* 复制 entry.h:807-826 NVLink 块 */ return; }

  // 2. 决策
  AstraSim::ComType ct = (AstraSim::ComType)request->flowTag.com_type;
  int region_id        = src_machine / g_prenet_topo->region_size;

  PredictResult pr = g_prenet_predictor->predict(src, dst, bytes, ct, region_id);
  g_prenet_predictions_total++;
  switch (pr.action) {
    case STAY_ECS:        g_prenet_action_stay_ecs++; break;
    case USE_OCS_ASIS:    g_prenet_action_use_ocs_asis++; break;
    case RECONFIG_OCS:    g_prenet_action_reconfig_ocs++; break;
  }

  // 3. RECONFIG_OCS:走仲裁,defer 当前 flow
  if (pr.action == RECONFIG_OCS) {
    DeferredSendData dsd = {src, dst, bytes, msg_handler, fun_arg, tag, *request};
    auto cb = [dsd, pr](bool granted, simtime_picosec end_time) {
      if (granted) {
        g_prenet_arbiter_wins++;
        // reconfig 完成后再 SendFlow,此时 conn 已就位会走 OCS
        DeferredSendEvent* ev = new DeferredSendEvent(*g_eventlist, dsd);
        g_eventlist->sourceIsPending(*ev, end_time);
      } else {
        g_prenet_arbiter_losses++;
        // fallback ECS,直接发(action 改写为 STAY_ECS)
        send_via_ecs(dsd.src, dsd.dst, dsd.count, dsd.msg_handler, dsd.fun_arg, dsd.tag, dsd.request, /*decision_id=*/0);
      }
    };
    g_prenet_topomanager->request_reconfig(region_id,
        *g_prenet_variants->get(region_id, pr.variant_id),
        pr.confidence, bytes, cb);
    return;
  }

  // 4. STAY_ECS
  if (pr.action == STAY_ECS) {
    send_via_ecs(src, dst, bytes, msg_handler, fun_arg, tag, request, pr.decision_id);
    if (pr.emit_probe) {
      // 注:ECS 时 probe 应该走 OCS — 但这里通常 confidence 已高才走 ECS,
      //     默认不发 probe,只在配置打开 verify-stay-ecs 时才发。
    }
    return;
  }

  // 5. USE_OCS_ASIS
  uint64_t probe_bytes = (pr.emit_probe ? (uint64_t)(bytes * pr.probe_ratio) : 0);
  uint64_t main_bytes  = bytes - probe_bytes;

  // probe 走 ECS
  TcpSrc* probe_tcp = nullptr;
  if (probe_bytes > 0) {
    probe_tcp = send_via_ecs_subflow(src, dst, probe_bytes, /*decision_id=*/pr.decision_id);
    g_prenet_probes_emitted++;
  }

  // 主 flow 走 OCS;路径可能在仲裁丢失时被 reroute
  TcpSrc* main_tcp = send_via_ocs(src, dst, main_bytes, msg_handler, fun_arg, tag, request, pr.decision_id);

  // 登记 meta
  PrenetFlowMeta* meta = new PrenetFlowMeta{
    pr.decision_id, pr.action, main_bytes, probe_bytes,
    g_eventlist->now(), g_prenet_predictor->probe_rho_snapshot(),
    main_tcp, probe_tcp, request->flowTag, src, dst
  };
  g_prenet_pending_flows[pr.decision_id] = meta;
}
```

### 7.3 helper `send_via_ecs` / `send_via_ocs` / `send_via_ecs_subflow`

#### 7.3.1 新结构体 `PrenetFlowContext`(扩展 `HtsimFlowContext`)
```cpp
struct PrenetFlowContext {
  HtsimFlowContext base;       // 复用 entry.h:285 现有结构 — 兼容已有回调机制
  uint64_t decision_id;        // 关联 _pending 的 key
  bool     is_probe;           // false=主流, true=probe 副本流
  // 注意:base.flowTag 中的 current_flow_id 必须与同一 decision 的另一条流共用
};
```
- 完成回调用 `htsim_flow_finish_prenet(void*)`,签名与 `htsim_flow_finish` 相同;
  内部 cast 成 `PrenetFlowContext*`,先做 `base` 部分(SimAI 的 chunk 累加 / receiver
  fire),再按 `is_probe` 填 `meta->end_time_main` 或 `meta->end_time_probe`,
  两者都到齐时调用 `predictor.update`。

#### 7.3.2 关键约束:probe 与 main 必须共享 `flowTag.current_flow_id`
- SimAI 的 `received_chunksize[(flow_id,src,dst)] += flow_size`(`entry.h:307`)
  按 `current_flow_id` 累加;若 probe 用新 flow_id,接收端永远凑不齐 `t2.count`,
  handler 不 fire,**workload 卡死**。
- 因此 `send_via_ecs_subflow` **直接复用主流的 `flowTag`**,只把 bytes 拆开
  (`main_bytes + probe_bytes == total_bytes`),让累加自然完成。
- `waiting_to_sent_callback[(flow_id,src,dst)] += 1` 和
  `waiting_to_notify_receiver[(flow_id,src,dst)] += 1` 在 `SendFlowPrenet` 里
  **必须 ++两次**(主 + probe);两条子流各自 finish 时 `--`,归零时才真正
  调用 receiver/sender notify。

#### 7.3.3 helper 实现骨架
```cpp
TcpSrc* send_via_ocs(int src,int dst,uint64_t bytes, ..., uint64_t decision_id);
TcpSrc* send_via_ecs(int src,int dst,uint64_t bytes, ..., uint64_t decision_id);   // 主流 ECS 版本
TcpSrc* send_via_ecs_subflow(int src,int dst,uint64_t bytes,
                             AstraSim::ncclFlowTag shared_tag, uint64_t decision_id); // probe 专用
```
共同骨架(参考 `entry.h::SendFlow` line 918~1023):
- `new DCTCPSrc / new TcpSink`
- `set_flowsize(bytes)` / `set_ssthresh` / `_rto`
- `is_elec = (走 ECS)`,`is_all2all = (com_type==AllToAll)`
- `g_prenet_topo->get_paths` 或 `get_eps_paths` 拿路径
- `connect(routeout, routein, sink, now)`
- 回调 ctx 是 `PrenetFlowContext`,完成回调用 `htsim_flow_finish_prenet`
- **不**调用 `g_moe_reconfig_mgr.on_a2a_flow_start`(那是 mixnet 的)

### 7.4 `htsim_flow_finish_prenet` 流程
```text
1. PrenetFlowContext* pctx = (PrenetFlowContext*)ctx_ptr;
2. // —— 完成 SimAI 的累加和 receiver 通知(与 entry.h::htsim_flow_finish 完全等价) ——
   sid = pctx->base.src; did = pctx->base.dst;
   flow_size = pctx->base.flow_size; flowTag = pctx->base.flowTag;
   received_chunksize[(flow_id,sid,did)] += flow_size;
   if (is_receive_finished(sid,did,flowTag)) {  // 内部 -- waiting_to_notify_receiver
     uint64_t notify_size = received_chunksize[(flow_id,sid,did)];
     received_chunksize.erase(...);
     notify_receiver_receive_data(sid, did, notify_size, flowTag);
   }
   sent_chunksize[(flow_id,sid,did)] += flow_size;
   if (is_sending_finished(sid,did,flowTag)) {
     uint64_t all_sent = sent_chunksize[(flow_id,sid,did)];
     sent_chunksize.erase(...);
     notify_sender_sending_finished(sid, did, all_sent, flowTag);
   }
3. // —— prenet 自己的 update 闭环 ——
   meta = g_prenet_pending_flows[pctx->decision_id];
   if (pctx->is_probe) meta->end_time_probe = now; else meta->end_time_main = now;
   bool main_done  = meta->end_time_main  != 0;
   bool probe_done = meta->bytes_probe == 0 || meta->end_time_probe != 0;
   if (main_done && probe_done) {
     UpdateInput in = {...};   // 同原文
     g_prenet_predictor->update(in);
     g_prenet_pending_flows.erase(pctx->decision_id);
     delete meta;
   }
4. delete pctx;
```
- **关键:meta 必须等"主+probe 都到"才删。** 否则后到的那条流找不到 meta。
- meta 的 `end_time_main = 0` 用作 sentinel,正常时间不会精确等于 0
  (htsim 启动 sim time 已经 > 0)。如担心冲突可加 `bool main_done` 字段。

### 7.5 `on_rank_pass_end_hook` 注册
- `AstraSimNetwork.cc` 的拓扑构造分支末尾,**条件**注册:
  ```cpp
  if (g_topo_type == TOPO_PRENET) {
    on_rank_pass_end_hook = [](int rank, int pass) {
      if (g_prenet_predictor) g_prenet_predictor->on_pass_end(rank, pass);
    };
  } else if (g_topo_type == TOPO_MIXNET) {
    on_rank_pass_end_hook = [](int rank, int pass) {
      if (g_mixnet_topo == nullptr) return;
      g_moe_reconfig_mgr.on_rank_pass_end(rank, pass, g_total_gpus, g_mixnet_topo->region_size);
    };
  }
  ```
  这是对 `AstraSimNetwork.cc` 唯一**改动**到 mixnet 周围代码的地方
  (现在那段 lambda 是无条件赋值)。改完之后 mixnet 还是同一份 lambda,
  prenet 是另一份,互不干扰。

---

## 8. `entry.h` 的最小修改

只两处修改,不动其他逻辑。

### 8.1 enum 新增

```cpp
// entry.h 第 67 行附近
enum TopoType {
  TOPO_MIXNET,
  TOPO_FATTREE,
  TOPO_OS_FATTREE,
  TOPO_AGG_OS_FATTREE,
  TOPO_FC,
  TOPO_FLAT,
#ifdef PRENET_ENABLED
  TOPO_PRENET,
#endif
};
```

### 8.2 SendFlow 顶部 dispatcher

```cpp
// entry.h SendFlow 函数体最开头(line 783 之后)
void SendFlow(int src, int dst, uint64_t maxPacketCount,
              void (*msg_handler)(void*), void* fun_arg,
              int tag, AstraSim::sim_request* request) {
#ifdef PRENET_ENABLED
  if (g_topo_type == TOPO_PRENET) {
    SendFlowPrenet(src, dst, maxPacketCount, msg_handler, fun_arg, tag, request);
    return;
  }
#endif
  // ... 原有 mixnet/fattree 逻辑保持不变
  ...
}
```

### 8.3 include
```cpp
// entry.h 顶部 include 区
#include "entry_prenet.h"   // 内部用 #ifdef PRENET_ENABLED 自闭环
```

---

## 9. `AstraSimNetwork.cc` 的修改

### 9.1 解析 topo
```cpp
// 第 357 行附近,新增一行
else if (topo_name == "prenet")     g_topo_type = TOPO_PRENET;
```

### 9.2 解析 prenet 专属 CLI
- `user_param` 增加字段(全部 `prenet_` 前缀):
  ```
  int    prenet_variant_k = 8;
  double prenet_probe_ratio = 0.05;
  int    prenet_arbiter_window_us = 2;
  int    prenet_confidence_init = 1;
  int    prenet_confidence_max = 3;
  std::string prenet_history_lengths = "4,8,16,32";   // 解析为 array
  uint64_t prenet_predictor_log_every = 1000;
  ```
- `long_options` 追加对应 long-option;`while (getopt_long(...))` switch 加 case。
- `--help` 输出新参数。

### 9.3 创建 prenet 拓扑分支
```cpp
// 在第 519 行 TOPO_FC 分支之后追加
} else if (g_topo_type == TOPO_PRENET) {
#ifdef PRENET_ENABLED
  uint32_t ecs_link_speed = params.speed * (params.gpus_per_server - params.alpha);

  g_prenet_ecs_underlay = new FatTreeTopology(
      fattree_node, memFromPkt(params.queuesize_pkts),
      NULL, &eventlist, NULL, LOSSLESS_INPUT_ECN, ecs_link_speed);

  g_prenet_topo = new Prenet(
      gpu_num, memFromPkt(params.queuesize_pkts), eventlist, ECN,
      timeFromUs((double)params.reconf_delay_us),
      g_prenet_ecs_underlay,
      params.alpha, params.dp_degree, params.tp_degree,
      params.pp_degree, params.ep_degree, params.gpus_per_server,
      params.prenet_variant_k);
  g_topology = g_prenet_topo;

  // 配置注入
  g_prenet_cfg.variant_pool_k     = params.prenet_variant_k;
  g_prenet_cfg.probe_ratio        = params.prenet_probe_ratio;
  g_prenet_cfg.arbiter_window     = timeFromUs((double)params.prenet_arbiter_window_us);
  g_prenet_cfg.confidence_init    = params.prenet_confidence_init;
  g_prenet_cfg.confidence_max     = params.prenet_confidence_max;
  g_prenet_cfg.predictor_log_every= params.prenet_predictor_log_every;
  parse_history_lengths_csv(params.prenet_history_lengths, g_prenet_cfg.tage_history_lengths);

  // 子组件
  g_prenet_variants    = new PrenetVariantPool(
      g_prenet_topo->region_size, g_prenet_topo->region_num, params.alpha,
      params.prenet_variant_k);
  g_prenet_topo->variant_pool = g_prenet_variants;

  g_prenet_arbiter     = new PrenetArbiter(eventlist, g_prenet_cfg.arbiter_window);

  g_prenet_topomanager = new PrenetTopoManager(g_prenet_topo, g_prenet_arbiter,
      timeFromUs((double)params.reconf_delay_us), eventlist);
  g_prenet_topo->topomanager = g_prenet_topomanager;

  g_prenet_predictor   = new PrenetPredictor(g_prenet_cfg, g_prenet_topo, g_prenet_ecs_underlay);

  // dead-route reroute hook 绑 prenet 拓扑
  // 注意:reroute_flow_if_dead_prenet 第一行守卫:
  //   if (g_topo_type != TOPO_PRENET) return false;
  // 防止误装在非 prenet 模式下被触发(虽然单次运行不会走到,但守卫提供二保险)
  TcpSrc::on_rtx_stuck = &reroute_flow_if_dead_prenet;

  cout << "[PRENET] topology created: region_size=" << g_prenet_topo->region_size
       << " region_num=" << g_prenet_topo->region_num
       << " variant_k=" << params.prenet_variant_k << endl;
#else
  cerr << "Error: prenet built with PRENET=0" << endl;
  return 1;
#endif
}
```

### 9.4 hook 注册分裂(见 §7.5)

### 9.5 `stats.txt` 输出
- 在 mixnet 段类似位置追加 `if (g_topo_type == TOPO_PRENET)` 段,
  打印 prediction accuracy、各 action 计数、arbiter 胜负数。

### 9.6 cleanup
```cpp
delete g_prenet_predictor;
delete g_prenet_topomanager;
delete g_prenet_arbiter;
delete g_prenet_variants;
delete g_prenet_topo;
delete g_prenet_ecs_underlay;
```

---

## 10. `Makefile.htsim` 修改

```makefile
PRENET ?= 1

ifeq ($(PRENET),1)
  HTSIM_DC_SRCS += $(HTSIM_DIR)/datacenter/prenet.cpp
  HTSIM_SRCS    += $(HTSIM_DIR)/prenet_topomanager.cpp \
                   $(HTSIM_DIR)/prenet_predictor.cpp \
                   $(HTSIM_DIR)/prenet_arbiter.cpp \
                   $(HTSIM_DIR)/prenet_variant_pool.cpp
  CXXFLAGS += -DPRENET_ENABLED
endif
```

不删除、不重排现有条目;`PRENET=0` 时所有 prenet 相关代码完全跳过编译。

---

## 11. `conf/topo/prenet.json`

```json
{
  "topology": {
    "type": "prenet",
    "speed": 100000,
    "queuesize": 8,
    "alpha": 6,
    "reconf_delay": 10,
    "ecs_only": false,
    "prenet": {
      "variant_pool_k": 8,
      "tage_history_lengths": "4,8,16,32",
      "probe_ratio": 0.05,
      "confidence_init": 1,
      "confidence_max": 3,
      "msg_size_buckets_kb": [64, 1024, 16384],
      "arbiter_window_us": 2,
      "predictor_log_every": 1000
    }
  },
  "simulation": { "iterations": 5, "rto_ms": 1 }
}
```

`run.sh` 已经按 `.topology.type` 分支,只需 case 里加:
```sh
prenet)
  CMD+=(--alpha "$ALPHA")
  CMD+=(--reconf_delay "$RECONF_DELAY")
  CMD+=(--prenet_variant_k "$(jq -r '.topology.prenet.variant_pool_k // 8' "$topo_cfg")")
  CMD+=(--prenet_probe_ratio "$(jq -r '.topology.prenet.probe_ratio // 0.05' "$topo_cfg")")
  CMD+=(--prenet_arbiter_window_us "$(jq -r '.topology.prenet.arbiter_window_us // 2' "$topo_cfg")")
  CMD+=(--prenet_confidence_init "$(jq -r '.topology.prenet.confidence_init // 1' "$topo_cfg")")
  CMD+=(--prenet_confidence_max "$(jq -r '.topology.prenet.confidence_max // 3' "$topo_cfg")")
  CMD+=(--prenet_history_lengths "$(jq -r '.topology.prenet.tage_history_lengths // "4,8,16,32"' "$topo_cfg")")
  CMD+=(--prenet_predictor_log_every "$(jq -r '.topology.prenet.predictor_log_every // 1000' "$topo_cfg")")
  if [[ "$ECS_ONLY" == "true" ]]; then CMD+=(--ecs_only); fi
  ;;
```
(纯追加 case,不影响其他 topo 分支)

---

## 12. 测试与验收

### 12.1 编译/链接
1. `make -f Makefile.htsim PRENET=0 -j8` → 不含 prenet 的二进制;
   旧 mixnet/fattree 测试全部通过,产物逐字节等同合入 prenet 之前。
2. `make -f Makefile.htsim PRENET=1 -j8` → 含 prenet;
   `./run.sh conf/topo/mixnet.json ...` 仍然产出与 PRENET=0 相同的 stats.txt(允许 timestamp 等非语义字段不同)。

### 12.2 功能冒烟
- `./run.sh conf/topo/prenet.json conf/workload/<small>.json`
  - stdout 含 `[PRENET]` 行
  - `stats.txt` 含 prenet 段
  - 程序正常退出,不 abort
- `./run.sh conf/topo/prenet.json conf/workload/<small>.json --prenet_ecs_only`
  - 强制全 ECS;`g_prenet_action_stay_ecs == g_prenet_predictions_total`
  - 性能与 `--topo fattree` 接近(误差由 prenet 自身 fattree 实例随机种子产生)

### 12.3 预测精度回归
- 跑 5 个 iteration,observe `g_prenet_predictions_correct / total`
  - 第 1 iter 预测器冷启动,允许 < 50%
  - 第 5 iter 应当显著上升(如果 workload 有规律)
- 与 mixnet 同 workload 对比 final time;
  期望 prenet 在 MoE 之外的 workload 上**也**比 mixnet 快(因为 mixnet 不动)
  在 MoE workload 上不显著劣于 mixnet。

### 12.4 异常注入
- 故意把 `variant_pool_k=0`:`PrenetVariantPool` 拒绝构造,程序在 init 阶段
  WARN + fallback all-ECS,不 abort。
- 故意写错的 prenet.json 字段(类型不对) → CLI 解析失败 → exit 1,不影响其他配置。

---

## 13. 不在本计划范围内(后续工作)

1. **Runahead reconfig**:`plan.md §6` 提到的"利用 compute 窗口提前重构"。
   prenet v0 不做,因为需要预测下一次 collective 起始点,而 htsim 的
   collective 起始点是 Workload 的 sim_send 调度,prenet 还拿不到。
2. **多租户仲裁**:本计划假设单 job。
3. **Variant pool 自适应增删 K**:见 §6.3。
4. **更精细的排队论模型**:M/D/1 替换为基于 token bucket 实测 + 神经网络拟合。
5. **GPU 端聚合**:把多个小 send 合并成一个 prenet 决策,减少 predictor 调用频次。

---

## 14. 与 `plan.md` 的对应关系

| `plan.md` 节 | 在本计划的位置 |
|------|---------|
| §2 TCT 表 | §4.1 `TctKey` + §4.2 TAGE 表 |
| §3 验证(5% probe + 排队论) | §4.4 `predict()` emit_probe + §4.5 update + §4.6 extrapolate + §7.2 SendFlowPrenet probe 分支 |
| §4 仲裁 | §5 `PrenetArbiter` |
| §5 Variant pool | §6 `PrenetVariantPool` |
| §6 Runahead reconfig | §13 后续 |
| §7 Fallback 保底 | `prenet_principles.md §5` + §7.2 cb 失败分支 |
| §8 NCCL Patch 点 | 不适用(本工程是 SimAI/htsim,非真实 NCCL) |
| §9 FPGA 职责 | 不适用(本工程纯软件仿真) |
| §10 差异化 | §0.2 表格 |
| §11 Open Questions | §13 后续 + 各模块的 v0/v1 标注 |

---

## 15. 实施顺序(建议提交粒度)

1. **PR-1**:`prenet_principles.md` + `prenet_plan.md`(本两份文档)。**只评审,不动代码。**
2. **PR-2**:`Makefile.htsim` 加 `PRENET ?= 1` 段;`entry.h` 加 enum + dispatcher;
   空壳 `entry_prenet.h`(`SendFlowPrenet` = abort + WARN);
   空壳 `prenet.h/.cpp`(只有空 `Prenet::get_paths` 返回空 vec);
   `AstraSimNetwork.cc` 加 `topo == "prenet"` 解析 + 空构造分支。
   ✅ 验收:`PRENET=0` / `PRENET=1` 都能编;`mixnet` workload 行为不变。
3. **PR-3**:`PrenetVariantPool` v0 + `Prenet::apply_variant` + 启动时 apply default。
   ✅ 验收:`prenet` 启动后 conn 矩阵打印与 mixnet random_connect 同分布。
4. **PR-4**:`PrenetTopoManager` + `PrenetArbiter` + `SendFlowPrenet` 骨架版。
   骨架版语义:**直接转发到 ECS,不调用 predictor、不登记 pending map、不创建 meta**。
   `PrenetTopoManager` 只实现 reconfig 状态机,但本 PR 不会被触发。
   ✅ 验收:`prenet` 跑 workload 与 fattree 性能接近(仅作用 ECS underlay);
   `g_prenet_pending_flows.size() == 0` 全程恒为零。
5. **PR-5**:`PrenetPredictor` v0(只用 base_table,无 TAGE) + 简单
   "size > 1MB → USE_OCS_ASIS" 决策;接 update 闭环(无 probe,反事实计算 truth)。
   ✅ 验收:大消息走 OCS,小消息走 ECS;预测精度统计可见。
6. **PR-6**:probe 拆分(`send_via_ecs_subflow`)+ 排队论外推 + 真正 update。
   ✅ 验收:`g_prenet_probes_emitted > 0`;预测精度比 v0 提高。
7. **PR-7**:TAGE 多表 + global_history + tag/index hash + usefulness 衰减。
   ✅ 验收:同 workload 第 5 iter 比 PR-6 准。
8. **PR-8**:`RECONFIG_OCS` 完整路径(arbiter + topomanager 实际 reconfig)。
   ✅ 验收:`g_prenet_action_reconfig_ocs > 0`;arbiter 胜负计数合理。
9. **PR-9**:大型回归(8+ 个 workload × {mixnet, fattree, prenet} 三组)。
