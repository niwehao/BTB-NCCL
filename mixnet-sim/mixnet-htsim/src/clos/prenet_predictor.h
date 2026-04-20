// prenet_predictor.h — 3-class action predictor.
//
// **v0 SCOPE NOTE**: The plan (predict-plan/prenet_plan.md §4.2) describes a
// full TAGE-style design with base_table + 4 tagged tables (history lengths
// 4/8/16/32). v0 implements only the `base_table` (64 entries, 2-bit signed
// counter) plus a single 1-bit history fold into the index. Tagged tables and
// usefulness counters are intentionally omitted to keep v0 small and verifiable;
// they will be added in PR-7 per the plan's incremental schedule.
//
// `tage_history_lengths` config field is parsed and stored but currently unused.
#ifndef PRENET_PREDICTOR_H
#define PRENET_PREDICTOR_H
#ifdef PRENET_ENABLED

#include "eventlist.h"
#include "prenet.h"
#include "prenet_variant_pool.h"
#include "fat_tree_topology.h"
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <ostream>

enum class PrenetAction : uint8_t {
  STAY_ECS      = 0,
  USE_OCS_ASIS  = 1,
  RECONFIG_OCS  = 2,
};

const char* prenet_action_name(PrenetAction a);

struct TctKey {
  uint8_t  msg_size_bucket;  // 0..3
  uint8_t  coll_type;        // AstraSim::ComType cast
  uint8_t  reconfig_needed;  // 0/1
  uint16_t src_node;
  uint16_t dst_node;
  uint8_t  region_id;
};

struct PredictResult {
  PrenetAction action = PrenetAction::STAY_ECS;
  int          variant_id = -1;
  int          confidence = 0;
  bool         emit_probe = false;
  double       probe_ratio = 0.0;
  uint64_t     decision_id = 0;
};

struct UpdateInput {
  uint64_t        decision_id;
  PrenetAction    taken_action;
  uint64_t        bytes_main;
  uint64_t        bytes_probe;         // 0 if no probe
  simtime_picosec start_time;
  simtime_picosec end_time_main;
  simtime_picosec end_time_probe;      // 0 if no probe
  double          fattree_rho_at_finish;
};

struct PrenetConfig {
  int variant_pool_k = 8;
  std::array<int, 4> tage_history_lengths = {4, 8, 16, 32};
  double probe_ratio = 0.05;
  int confidence_init = 1;
  int confidence_max = 3;
  // Bucket boundaries in bytes: <64KB bucket 0, <1MB bucket 1, <16MB bucket 2, else 3.
  std::array<uint64_t, 3> msg_size_buckets_bytes = {
      64ULL * 1024,
      1024ULL * 1024,
      16ULL * 1024 * 1024};
  simtime_picosec arbiter_window = 2ULL * 1000 * 1000;  // 2 us in ps
  uint64_t predictor_log_every = 1000;
  uint32_t link_speed_mbps = 100000;   // injected from main; used for counterfactual
  int alpha = 6;
};

class PrenetPredictor {
public:
  PrenetPredictor(const PrenetConfig& cfg, Prenet* topo, FatTreeTopology* ecs);

  // Decision on every cross-machine flow start.
  PredictResult predict(int src_gpu, int dst_gpu, uint64_t bytes,
                        int coll_type, int region_id);

  // Close the loop on every flow completion (after both main and probe done).
  void update(const UpdateInput& in);

  // Drop a pending decision without updating the counters (e.g. arbiter loser
  // — the predicted action was never executed, so there's nothing to learn).
  // No-op if decision_id is unknown.
  void erase_pending(uint64_t decision_id);

  // Pass end: usefulness decay + logging snapshot.
  void on_pass_end(int rank, int pass);

  void dump_stats(std::ostream& out) const;

  // Debug/telemetry snapshot of current aggregate fattree rho (path-max).
  double probe_rho_snapshot() const;

  // Accessors for stats.
  uint64_t total_predictions() const { return _total_predictions; }
  uint64_t correct_predictions() const { return _correct_predictions; }
  uint64_t wrong_predictions() const { return _wrong_predictions; }

private:
  // ---- key / hash ----
  uint8_t bucket_of(uint64_t bytes) const;
  TctKey  build_key(int src_gpu, int dst_gpu, uint64_t bytes,
                    int coll_type, int region_id) const;
  uint32_t base_index(const TctKey& k) const;

  // ---- base table ----
  // Signed counter in [-4, 3]: >= 0 means "OCS preferred", < 0 means "ECS preferred".
  static constexpr int CTR_MIN = -4;
  static constexpr int CTR_MAX = 3;
  std::vector<int8_t> _base_table;     // size 64

  // ---- global history (push bit is action_chose_ocs, see plan §4.5) ----
  uint32_t _global_history = 0;
  uint8_t  _history_bits = 32;

  // ---- pending ctx (decision_id -> key/action) ----
  struct DecisionCtx {
    TctKey key;
    PrenetAction action;
    bool action_chose_ocs;
    uint64_t bytes;
    int coll_type;
    simtime_picosec start_time;
    double rho_at_decision;
    uint64_t main_bytes;
    uint64_t probe_bytes;
  };
  std::unordered_map<uint64_t, DecisionCtx> _pending;
  uint64_t _next_decision_id = 1;

  // ---- ECS path rho estimation cache ----
  // pair_to_hop_queues[src_machine][dst_machine] -> pointers to queues on that ECS path.
  std::vector<std::vector<std::vector<Queue*>>> _pair_hop_queues;
  void build_hop_cache();
  double max_rho_on_path(int src_machine, int dst_machine) const;

  // ---- extrapolation ----
  simtime_picosec extrapolate_full_ecs(uint64_t main_bytes, uint64_t probe_bytes,
                                       simtime_picosec probe_actual,
                                       double rho_now, double link_bps) const;

  // ---- config / refs ----
  PrenetConfig _cfg;
  Prenet* _topo;
  FatTreeTopology* _ecs;

  // ---- stats ----
  uint64_t _total_predictions = 0;
  uint64_t _correct_predictions = 0;
  uint64_t _wrong_predictions = 0;
  std::array<uint64_t, 3> _action_counts = {0, 0, 0};
};

#endif // PRENET_ENABLED
#endif // PRENET_PREDICTOR_H
