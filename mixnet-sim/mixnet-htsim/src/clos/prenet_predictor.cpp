// prenet_predictor.cpp
#ifdef PRENET_ENABLED
#include "prenet_predictor.h"
#include "queue.h"
#include "route.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// Must match AstraSim::ComType: 0 None, 1 Reduce_Scatter, 2 All_Gather,
// 3 All_Reduce, 4 All_to_All, 5 All_Reduce_All_to_All, 6 All_Reduce_NVLS
static constexpr int COMTYPE_ALL_TO_ALL = 4;
static constexpr int COMTYPE_ALL_REDUCE_A2A = 5;

const char* prenet_action_name(PrenetAction a) {
  switch (a) {
    case PrenetAction::STAY_ECS:     return "STAY_ECS";
    case PrenetAction::USE_OCS_ASIS: return "USE_OCS_ASIS";
    case PrenetAction::RECONFIG_OCS: return "RECONFIG_OCS";
  }
  return "?";
}

PrenetPredictor::PrenetPredictor(const PrenetConfig& cfg, Prenet* topo,
                                 FatTreeTopology* ecs)
  : _cfg(cfg), _topo(topo), _ecs(ecs) {
  _base_table.assign(64, 0);
  build_hop_cache();
  std::cerr << "[PRENET] predictor init: base_table=64 buckets, alpha=" << cfg.alpha
            << " probe_ratio=" << cfg.probe_ratio << std::endl;
}

uint8_t PrenetPredictor::bucket_of(uint64_t bytes) const {
  if (bytes < _cfg.msg_size_buckets_bytes[0]) return 0;
  if (bytes < _cfg.msg_size_buckets_bytes[1]) return 1;
  if (bytes < _cfg.msg_size_buckets_bytes[2]) return 2;
  return 3;
}

TctKey PrenetPredictor::build_key(int src_gpu, int dst_gpu, uint64_t bytes,
                                  int coll_type, int region_id) const {
  TctKey k;
  k.msg_size_bucket = bucket_of(bytes);
  k.coll_type = (uint8_t)(coll_type & 0x7);
  int src_node = src_gpu / _topo->gpus_per_node;
  int dst_node = dst_gpu / _topo->gpus_per_node;
  k.reconfig_needed = _topo->circuit_exists(src_node, dst_node) ? 0 : 1;
  k.src_node = (uint16_t)src_node;
  k.dst_node = (uint16_t)dst_node;
  k.region_id = (uint8_t)(region_id & 0xFF);
  return k;
}

uint32_t PrenetPredictor::base_index(const TctKey& k) const {
  uint32_t h = (uint32_t)k.msg_size_bucket * 32
             + (uint32_t)k.coll_type * 4
             + (uint32_t)k.reconfig_needed * 2;
  // Fold in 1 bit of history for mild context.
  h ^= (_global_history & 0x1);
  return h & 0x3F;  // 64 entries
}

void PrenetPredictor::build_hop_cache() {
  if (_ecs == nullptr) return;
  int N = _ecs->no_of_nodes();
  _pair_hop_queues.assign(N, std::vector<std::vector<Queue*>>(N));
  for (int s = 0; s < N; s++) {
    for (int d = 0; d < N; d++) {
      if (s == d) continue;
      // Pull paths lazily on demand; building full matrix up-front may be heavy.
      // Leave empty here — max_rho_on_path will resolve on first use.
    }
  }
}

double PrenetPredictor::max_rho_on_path(int src_machine, int dst_machine) const {
  if (_ecs == nullptr) return 0.0;
  // Lazy lookup: call ecs->get_paths and inspect queue occupancy along the
  // first path. CRITICAL: get_paths() returns a freshly-allocated vector +
  // Routes; we must free them, otherwise predict()/update() leak per call.
  auto* paths = const_cast<FatTreeTopology*>(_ecs)->get_paths(src_machine, dst_machine);
  if (paths == nullptr) return 0.0;
  double maxrho = 0.0;
  if (!paths->empty()) {
    const Route* r = paths->at(0);
    for (auto it = r->begin(); it != r->end(); ++it) {
      Queue* q = dynamic_cast<Queue*>(*it);
      if (q == nullptr) continue;
      if (q->_maxsize <= 0) continue;
      double rho = (double)q->queuesize() / (double)q->_maxsize;
      if (rho < 0) rho = 0;
      if (rho > 0.99) rho = 0.99;
      if (rho > maxrho) maxrho = rho;
    }
  }
  for (const Route* r : *paths) delete r;
  delete paths;
  return maxrho;
}

double PrenetPredictor::probe_rho_snapshot() const {
  // Summary rho: average of a small sample of pairs. Cheap heuristic only.
  if (_ecs == nullptr) return 0.0;
  int N = _ecs->no_of_nodes();
  if (N < 2) return 0.0;
  double acc = 0.0;
  int samples = 0;
  int step = std::max(1, N / 8);
  for (int s = 0; s < N; s += step) {
    int d = (s + 1) % N;
    if (d == s) d = (d + 1) % N;
    acc += max_rho_on_path(s, d);
    samples++;
  }
  return samples > 0 ? acc / samples : 0.0;
}

PredictResult PrenetPredictor::predict(int src_gpu, int dst_gpu, uint64_t bytes,
                                       int coll_type, int region_id) {
  PredictResult pr;
  TctKey k = build_key(src_gpu, dst_gpu, bytes, coll_type, region_id);

  uint32_t idx = base_index(k);
  int8_t ctr = _base_table[idx];
  int confidence = std::min(std::abs((int)ctr), _cfg.confidence_max);
  bool ocs_preferred = (ctr >= 0);

  // v0 safety: RECONFIG_OCS limited to AllToAll / AllReduce_All_to_All.
  bool a2a_like = (coll_type == COMTYPE_ALL_TO_ALL ||
                   coll_type == COMTYPE_ALL_REDUCE_A2A);

  PrenetAction action = PrenetAction::STAY_ECS;

  if (ocs_preferred) {
    if (k.reconfig_needed == 0) {
      // Circuit already exists — use OCS as-is (any collective).
      action = PrenetAction::USE_OCS_ASIS;
    } else if (a2a_like) {
      // Need reconfig; only A2A is allowed to reconfig in v0.
      if (k.msg_size_bucket >= 1) {
        action = PrenetAction::RECONFIG_OCS;
      } else {
        action = PrenetAction::STAY_ECS;
      }
    } else {
      // Non-A2A cannot reconfig in v0.
      action = PrenetAction::STAY_ECS;
    }
  } else {
    action = PrenetAction::STAY_ECS;
  }

  // If action is RECONFIG_OCS, pick a variant that covers this src->dst.
  int variant_id = -1;
  if (action == PrenetAction::RECONFIG_OCS && _topo->variant_pool != nullptr) {
    int src_node = k.src_node;
    int dst_node = k.dst_node;
    int rs = _topo->region_size;
    int src_local = src_node % rs;
    int dst_local = dst_node % rs;
    variant_id = _topo->variant_pool->pick_covering_variant(region_id, src_local, dst_local);
    if (variant_id < 0) {
      // No covering variant — degrade to STAY_ECS; still register so update fires.
      action = PrenetAction::STAY_ECS;
    }
  }

  bool action_chose_ocs = (action != PrenetAction::STAY_ECS);
  // Probe split is only safe for PURE AllToAll (each pair independent); see
  // prenet_plan.md §4.5 / entry_prenet.h's probe_safe_coll. Setting emit_probe
  // only when coll_type==AllToAll avoids the "pr.emit_probe=true but
  // probe_bytes=0" state the send path would otherwise have to filter for.
  bool emit_probe = action_chose_ocs && (confidence < _cfg.confidence_max)
                    && (coll_type == COMTYPE_ALL_TO_ALL);

  uint64_t did = _next_decision_id++;
  DecisionCtx ctx;
  ctx.key = k;
  ctx.action = action;
  ctx.action_chose_ocs = action_chose_ocs;
  ctx.bytes = bytes;
  ctx.coll_type = coll_type;
  // Stamp start_time at predict() — flow goes out at this sim time (or 1ps
  // later); precise enough for FCT comparison and removes the "0 sentinel" risk.
  ctx.start_time = _topo ? _topo->eventlist.now() : 0;
  ctx.rho_at_decision = max_rho_on_path(k.src_node, k.dst_node);
  ctx.main_bytes = bytes;
  ctx.probe_bytes = 0;
  _pending[did] = ctx;

  pr.action = action;
  pr.variant_id = variant_id;
  pr.confidence = confidence;
  pr.emit_probe = emit_probe;
  pr.probe_ratio = emit_probe ? _cfg.probe_ratio : 0.0;
  pr.decision_id = did;

  _total_predictions++;
  int ai = (int)action;
  if (ai >= 0 && ai < 3) _action_counts[ai]++;

  if (_cfg.predictor_log_every > 0 &&
      (_total_predictions % _cfg.predictor_log_every == 0)) {
    std::cerr << "[PRENET] predicted " << _total_predictions
              << " stay_ecs=" << _action_counts[0]
              << " ocs_asis=" << _action_counts[1]
              << " reconfig=" << _action_counts[2]
              << " correct=" << _correct_predictions
              << " wrong=" << _wrong_predictions << std::endl;
  }
  return pr;
}

simtime_picosec PrenetPredictor::extrapolate_full_ecs(uint64_t main_bytes,
                                                      uint64_t probe_bytes,
                                                      simtime_picosec probe_actual,
                                                      double rho_now,
                                                      double link_bps) const {
  if (link_bps <= 0) return 0;
  double service_ps = (double)main_bytes * 8.0 / link_bps * 1e12;
  double rho = std::min(0.99, std::max(0.0, rho_now));
  // Empirical bump: assume sending the hypothetical full payload over ECS
  // would drive rho slightly higher than the current observation. 0.05 is a
  // v0 rule-of-thumb; principled approach would use a closed-form ρ_new based
  // on arrival rate, but that requires traffic model beyond what we track.
  double rho_full = std::min(0.99, rho + 0.05);
  double wait_factor = rho_full / (2.0 * (1.0 - rho_full));
  double T_full = service_ps * (1.0 + wait_factor);

  if (probe_bytes > 0 && probe_actual > 0) {
    double probe_service_ps = (double)probe_bytes * 8.0 / link_bps * 1e12;
    double expected_probe = probe_service_ps / std::max(0.01, 1.0 - rho);
    if (expected_probe > 0) {
      double cal = (double)probe_actual / expected_probe;
      cal = std::min(4.0, std::max(0.25, cal));  // clamp calibration
      T_full *= cal;
    }
  }
  return (simtime_picosec)T_full;
}

void PrenetPredictor::update(const UpdateInput& in) {
  auto it = _pending.find(in.decision_id);
  if (it == _pending.end()) return;
  DecisionCtx ctx = it->second;
  _pending.erase(it);

  simtime_picosec now = in.end_time_main > 0 ? in.end_time_main : in.end_time_probe;
  // ctx.start_time is stamped at predict(); for RECONFIG_OCS winner the
  // replay happens AFTER reconf_delay so meta->start_time > ctx.start_time,
  // but for the predictor's "did this decision pay off" math we want the
  // ORIGINAL decision timestamp (includes the reconf cost).
  simtime_picosec start = ctx.start_time;
  if (now <= start) return;
  simtime_picosec actual_main = now - start;
  simtime_picosec actual_probe = (in.end_time_probe > start) ? (in.end_time_probe - start) : 0;

  // Link bw estimate (ECS link speed in this v0 is the fattree link speed).
  double link_bps = (double)_cfg.link_speed_mbps * 1e6;

  bool was_correct = false;
  // Branch on the ACTUALLY EXECUTED action (in.taken_action), not the
  // predicted one (ctx.action). They can differ when OCS silently degraded to
  // ECS (e.g. circuit torn down between predict() and spawn).
  bool executed_went_ocs = (in.taken_action == PrenetAction::USE_OCS_ASIS ||
                            in.taken_action == PrenetAction::RECONFIG_OCS);
  if (!executed_went_ocs) {
    // Prenet variant_pool generates 0/1 conn matrices (one circuit per pair),
    // so a single OCS link's bw is `link_bps`, NOT `alpha * link_bps`.
    // (The alpha cap is per-node degree, not per-link aggregation.)
    double ocs_bw_bps = link_bps;
    double counterfactual_ocs_ps = (double)ctx.bytes * 8.0 / std::max(1e6, ocs_bw_bps) * 1e12;
    if (ctx.key.reconfig_needed) counterfactual_ocs_ps += _topo->reconf_delay;
    was_correct = ((double)actual_main <= counterfactual_ocs_ps);
  } else {
    // OCS chosen: compare actual OCS time (for the main_bytes) vs the
    // counterfactual "if the FULL payload had gone through ECS".
    // Full payload = main_bytes + probe_bytes (we transmitted both halves;
    // probe_bytes via ECS, main_bytes via OCS). Always apply the M/D/1 model
    // so high-rho ECS paths don't get credited zero wait — that used to make
    // OCS decisions look bad when they were actually good (ECS was saturated).
    uint64_t total_bytes = in.bytes_main + in.bytes_probe;
    simtime_picosec extrap = extrapolate_full_ecs(
        total_bytes, in.bytes_probe, actual_probe,
        in.fattree_rho_at_finish, link_bps);
    was_correct = (actual_main <= extrap);
  }

  // Update base_table ctr based on what ACTUALLY happened:
  // executed=OCS & fast  → OCS preferred (ctr+)
  // executed=OCS & slow  → ECS preferred (ctr-)
  // executed=ECS & fast  → ECS preferred (ctr-)
  // executed=ECS & slow  → OCS preferred (ctr+)
  uint32_t idx = base_index(ctx.key);
  int8_t ctr = _base_table[idx];
  bool ocs_should_be_preferred = executed_went_ocs ? was_correct : !was_correct;
  if (ocs_should_be_preferred) {
    if (ctr < CTR_MAX) ctr++;
  } else {
    if (ctr > CTR_MIN) ctr--;
  }
  _base_table[idx] = ctr;

  // Push history bit: what path was ACTUALLY tested (1=OCS, 0=ECS), not what
  // was predicted. If prediction and execution diverged (degrade), the history
  // reflects reality.
  _global_history = (_global_history << 1) | (executed_went_ocs ? 1u : 0u);

  if (was_correct) _correct_predictions++;
  else _wrong_predictions++;
}

void PrenetPredictor::erase_pending(uint64_t decision_id) {
  _pending.erase(decision_id);
}

void PrenetPredictor::on_pass_end(int rank, int pass) {
  // v0: no usefulness decay (base_table only). Print aggregate stats from rank 0.
  if (rank != 0) return;
  std::cerr << "[PRENET] pass_end pass=" << pass
            << " total=" << _total_predictions
            << " correct=" << _correct_predictions
            << " wrong=" << _wrong_predictions
            << " stay_ecs=" << _action_counts[0]
            << " ocs_asis=" << _action_counts[1]
            << " reconfig=" << _action_counts[2]
            << " pending=" << _pending.size() << std::endl;
}

void PrenetPredictor::dump_stats(std::ostream& out) const {
  out << "prenet_predictions_total: " << _total_predictions << "\n";
  out << "prenet_predictions_correct: " << _correct_predictions << "\n";
  out << "prenet_predictions_wrong: " << _wrong_predictions << "\n";
  out << "prenet_action_stay_ecs: " << _action_counts[0] << "\n";
  out << "prenet_action_use_ocs_asis: " << _action_counts[1] << "\n";
  out << "prenet_action_reconfig_ocs: " << _action_counts[2] << "\n";
}

#endif // PRENET_ENABLED
