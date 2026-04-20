# Prenet 设计准则（隔离 / 低耦合）

本文件**只规定准则**,具体怎么改文件参考 `prenet_plan.md`。
准则的总目标:即使 prenet 出现严重 bug、写错算法、甚至崩溃,**也不能影响**
`mixnet`、`fattree`、`os_fattree`、`agg_os_fattree`、`fc`、`flat` 任何一种已有拓扑的运行结果。

---

## 1. 命名 / 文件物理隔离

1. 所有 prenet 专属源码文件以 `prenet` 为前缀,放在与 mixnet 平行的目录:

   - `mixnet-sim/mixnet-htsim/src/clos/datacenter/prenet.h`
   - `mixnet-sim/mixnet-htsim/src/clos/datacenter/prenet.cpp`
   - `mixnet-sim/mixnet-htsim/src/clos/prenet_topomanager.h`
   - `mixnet-sim/mixnet-htsim/src/clos/prenet_topomanager.cpp`
   - `mixnet-sim/mixnet-htsim/src/clos/prenet_predictor.h`
   - `mixnet-sim/mixnet-htsim/src/clos/prenet_predictor.cpp`
   - `mixnet-sim/mixnet-htsim/src/clos/prenet_arbiter.h`
   - `mixnet-sim/mixnet-htsim/src/clos/prenet_arbiter.cpp`
   - `mixnet-sim/mixnet-htsim/src/clos/prenet_variant_pool.h`
   - `mixnet-sim/mixnet-htsim/src/clos/prenet_variant_pool.cpp`
   - `SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/htsim/entry_prenet.h`
   - `conf/topo/prenet.json`

2. **不允许**在 `mixnet.h/.cpp`、`mixnet_topomanager.h/.cpp`、
   `fat_tree_topology.*`、`flat_topology.*` 等非 prenet 源文件里写一行 prenet 逻辑。
   prenet 需要的辅助接口(如 query 队列利用率)只能用现有 public 接口,
   不能为了 prenet 去改 mixnet/fattree 的 public/private 成员。

3. prenet 的所有全局变量、单例、状态都放在 `entry_prenet.h` 里以 `g_prenet_*`
   前缀命名,**不复用** `g_mixnet_topo`、`g_topomanager`、`g_demand_recorder`、
   `g_moe_reconfig_mgr` 等任何 mixnet 全局对象。

---

## 2. 运行期分支隔离

1. `g_topo_type` 增加新枚举 `TOPO_PRENET`。所有 prenet 行为都必须在
   `if (g_topo_type == TOPO_PRENET) { ... }` 分支内执行。

2. **入口分发**:`entry.h::SendFlow` 在最开头(NVLink 同机分支之后,
   reconfig defer 之前)插入**唯一一处** prenet 短路:

   ```cpp
   if (g_topo_type == TOPO_PRENET) {
     SendFlowPrenet(src, dst, maxPacketCount, msg_handler, fun_arg, tag, request);
     return;   // 不再走原有 mixnet/fattree 路径
   }
   ```

   - 这是 entry.h 唯一允许新增的 prenet 相关代码。
   - 其余 mixnet/fattree 分支保持不动。

3. `AstraSimNetwork.cc` 对 prenet 的所有创建逻辑都放进
   `else if (g_topo_type == TOPO_PRENET) { ... }` 新分支,**不修改**
   现有 `if (g_topo_type == TOPO_MIXNET) { ... }` 等分支的任何一行。

4. `on_rank_pass_end_hook` 在 `AstraSimNetwork.cc` 中按 `g_topo_type` **注册不同 lambda**:
   `if (g_topo_type == TOPO_MIXNET)` 注册 mixnet 的;
   `else if (g_topo_type == TOPO_PRENET)` 注册 prenet 的;
   其它拓扑保持原本的 nullptr 或 mixnet lambda 行为(因 mixnet lambda 内部已自带
   `if (g_mixnet_topo == nullptr) return` 守卫,空跑即可)。
   **不在同一份 lambda 内混杂分支**——避免 mixnet 行为被 prenet 改动间接影响。

---

## 3. 编译/链接隔离

1. `Makefile.htsim` 只**追加** prenet 源文件到 `HTSIM_SRCS` / `HTSIM_DC_SRCS`,
   不删除、不重排现有条目。

2. prenet 文件之间可以互相 include,但**禁止** prenet 的 `.h`
   被 `mixnet*.h`、`fat_tree_topology.h` 等非 prenet 头文件 include。
   反向是允许的:`prenet.h` 可以 include `topology.h`、`fat_tree_topology.h`
   (因为 prenet 也复用 fattree 作为 ECS underlay)。

3. prenet 不引入新的第三方库依赖。所有需要的容器、随机数、数学函数用
   STL + `<cmath>` + `<random>`,与 mixnet 一致。

---

## 4. 状态隔离

1. **不读写** mixnet 的状态(`g_mixnet_topo->conn`、`g_demand_recorder->traffic_matrix`、
   `g_moe_reconfig_mgr.*` 等)。
   - 例外:**只读** `g_topology` (作为通用 fallback 指针)。

2. prenet 自己的 conn 矩阵、history register、TAGE table、variant pool、
   confidence counter、deferred flow queue 全部是 prenet 拥有的成员变量,
   生命周期跟随 `g_prenet_topo` / `g_prenet_topomanager` / `g_prenet_predictor`。

3. **资源对接 fattree**:prenet 复用现有 `FatTreeTopology` 作为 ECS 底座
   (跟 mixnet 一样)。但 prenet 不与 mixnet 共享同一个 fattree 实例 ——
   构造时**新建**一个 `FatTreeTopology`,只在 prenet 这条路径使用。

4. TCP src 上原本就有的 `is_elec`、`is_all2all` 字段是与 mixnet 共用的。
   prenet 复用 `is_elec`(语义"该 flow 走 ECS 而非 OCS")保持不变,
   `is_all2all` 字段对 prenet 不再有 OCS-only 含义,prenet 自己用一个新字段
   `tcp->prenet_action`(枚举 `STAY_ECS / USE_OCS_ASIS / RECONFIG_OCS / PROBE`)
   来标记预测决策,挂在 `TcpSrc` 的扩展字段或者用一个 prenet 内部 map
   `(TcpSrc*) -> PrenetFlowMeta` 维护。优先用外部 map,**避免改 tcp.h**。

---

## 5. 失败保底准则

1. **prenet 任何环节失败必须 fallback ECS,而不是 abort**。包括:
   - 预测器表查不到 → fallback STAY_ECS
   - variant_id 在 pool 里找不到 → fallback STAY_ECS
   - 仲裁失败(被别的请求抢走 OCS) → fallback STAY_ECS
   - probe 触发失败(分裂出来的 ECS sub-flow 没法创建) → 不分裂,主流走预测决策,只是这次没法验证
   - reconfig 期间到达的新 flow → defer(同 mixnet)或直接 fallback ECS

2. 严格区分两类异常:
   - **本次预测/操作失败 → fallback ECS,继续跑**(通过 NcclLog WARN 打印,不 assert)
   - **不变量破坏 → assert 立即崩**(例如 prenet 内部 K 值 vs variant pool 大小不一致,
     这是代码 bug,不该 fallback 掩盖)

3. **不改动** mixnet 的 fallback 路径(`get_eps_paths` 已经存在),
   prenet 通过自己 new 出来的 fattree 调用 `get_paths` / `get_eps_paths`。

---

## 6. 测试隔离准则

1. 引入 prenet **不能让任何现有测试结果变化**。验证方式:
   ```
   ./run.sh conf/topo/mixnet.json conf/workload/<x>.json
   ./run.sh conf/topo/fattree.json conf/workload/<x>.json
   ```
   在合入 prenet 之前后,对相同 seed/工作负载,FCT、OCS flow 数、ECS flow 数、
   `stats.txt` 中的 reconfigs_triggered 等关键指标必须**逐字节相等**
   (因为只追加分支、未改原代码)。

2. 引入 prenet 的新测试:
   ```
   ./run.sh conf/topo/prenet.json conf/workload/<x>.json
   ```
   stdout 必须出现 `[PRENET]` 前缀的预测/验证日志,与 `[MOE_*]`/`[RECONF_*]` 互不重叠。

3. 设计 build-only 编译开关 `-DPRENET_DISABLED`(或 `#ifndef PRENET_ENABLED`),
   使得即使 prenet 源码出现编译错误,也能通过 `make -f Makefile.htsim PRENET=0`
   完整编译出不含 prenet 的二进制(应急用)。

   - 实现:`Makefile.htsim` 顶部增加:
     ```
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
   - `entry_prenet.h` 内部用 `#ifdef PRENET_ENABLED` 包裹声明;
     `entry.h` 中那唯一一行 dispatcher 也用 `#ifdef PRENET_ENABLED` 包裹。
   - `AstraSimNetwork.cc` 中 prenet 分支用 `#ifdef PRENET_ENABLED` 包裹,
     `else if (topo_name == "prenet")` 在 `#ifndef PRENET_ENABLED` 下
     变成 "unknown topology" 错误。

---

## 7. 配置/CLI 隔离

1. `conf/topo/prenet.json` 是 prenet 专属配置文件,字段不与 `mixnet.json` 共用
   key 名称(防止用户把 prenet 配置错误传给 mixnet):
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
         "tage_history_lengths": [4, 8, 16, 32],
         "probe_ratio": 0.05,
         "confidence_init": 1,
         "confidence_max": 3,
         "msg_size_buckets_kb": [64, 1024, 16384],
         "arbiter_window_us": 5,
         "predictor_log_every": 1000
       }
     },
     "simulation": { "iterations": 5, "rto_ms": 1 }
   }
   ```

2. CLI 新增的所有参数都以 `--prenet_` 前缀(`--prenet_probe_ratio`、
   `--prenet_variant_k` 等)。`AstraSimNetwork.cc` 的 `user_param` 结构里
   新增字段也都用 `prenet_` 前缀。**不复用** `--alpha`、`--reconf_top_n` 等
   mixnet 字段的语义改写。

3. 当 `--topo prenet` 与某个 mixnet-only 参数(如 `--reconf_top_n`)同时出现,
   prenet 路径**忽略**该参数(打印 WARN),不改解析逻辑。

---

## 8. 日志/统计隔离

1. prenet 相关日志全部带 `[PRENET]` 前缀,统计字段以 `prenet_` 开头写入
   `stats.txt`。`stats.txt` 中 mixnet 相关字段在 prenet 模式下不输出。

2. 现有的 `g_flow_count_ocs` / `g_flow_count_ecs` / `g_flow_bytes_*`
   计数器**复用**(它们语义就是 OCS vs ECS,与拓扑无关),
   但 prenet 额外新增:
   ```
   g_prenet_predictions_total
   g_prenet_predictions_correct
   g_prenet_predictions_wrong
   g_prenet_probes_emitted
   g_prenet_action_stay_ecs
   g_prenet_action_use_ocs_asis
   g_prenet_action_reconfig_ocs
   g_prenet_arbiter_wins
   g_prenet_arbiter_losses
   g_prenet_confidence_avg_at_decision
   ```
   都在 `entry_prenet.h` 静态全局,**不污染** entry.h。

---

## 9. 设计原则的冲突仲裁

如果以下两条冲突,以"隔离"优先:
- (A) 复用代码:prenet 想直接 include `mixnet_topomanager.h` 拿 `RegionalTopoManager` 的实现细节
- (B) 隔离:prenet 不依赖 mixnet 任何头文件

→ 选 (B):宁可在 `prenet_topomanager.cpp` 里复制粘贴 + 改写
`RegionalTopoManager` 的代码,也不要 include/继承 mixnet 的类。
**理由**:类继承会让 mixnet 父类的任何修改间接影响 prenet,违反隔离目标。
代码重复带来的维护成本是可控的,且 prenet 演化方向跟 mixnet 不同
(prenet 要支持非 A2A、要做 verification、要做 arbitration),
未来这些代码会越走越远,共享父类反而拖累。

---

## 10. 该准则的有效范围

- 适用于 **prenet 第一版** 直至它通过基本回归测试。
- 通过测试后,如果发现某个低耦合做法显著增加复杂度而维护成本不划算,
  可以再讨论是否把 prenet 与 mixnet 的某些通用基础设施(例如
  "regional topology manager 的 reconfig 机制框架")抽取为公共基类。
  这种重构必须**单独立项**,不在 prenet 自己的 PR 里做。
