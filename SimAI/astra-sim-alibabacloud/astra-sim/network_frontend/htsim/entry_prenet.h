// entry_prenet.h — prenet-owned SendFlow / finish path.
// Gated entirely behind PRENET_ENABLED so that PRENET=0 builds stay
// byte-for-byte identical to before.
#ifndef __ENTRY_PRENET_H__
#define __ENTRY_PRENET_H__
#ifdef PRENET_ENABLED

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/MockNcclLog.h"

#include "eventlist.h"
#include "tcp.h"
#include "dctcp.h"
#include "prenet.h"
#include "prenet_topomanager.h"
#include "prenet_predictor.h"
#include "prenet_arbiter.h"
#include "prenet_variant_pool.h"
#include "fat_tree_topology.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>

// Forward decls from entry.h — defined in the same translation unit at main.
struct HtsimFlowContext;
void schedule_callback(simtime_picosec delay_ps, void (*fun_ptr)(void*), void* fun_arg);
extern void notify_receiver_receive_data(int sender_node, int receiver_node,
                                         uint64_t message_size, AstraSim::ncclFlowTag flowTag);
extern void notify_sender_sending_finished(int sender_node, int receiver_node,
                                           uint64_t message_size, AstraSim::ncclFlowTag flowTag);
extern bool is_sending_finished(int src, int dst, AstraSim::ncclFlowTag flowTag);
extern bool is_receive_finished(int src, int dst, AstraSim::ncclFlowTag flowTag);

// Maps/state declared in entry.h (same translation unit after include).
extern std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_sent_callback;
extern std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_notify_receiver;
extern std::map<std::pair<int, std::pair<int, int>>, uint64_t> received_chunksize;
extern std::map<std::pair<int, std::pair<int, int>>, uint64_t> sent_chunksize;
extern std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint16_t>> portNumber;
extern std::map<std::pair<int, std::pair<int, int>>, AstraSim::ncclFlowTag> sender_src_port_map;

// Global: event list and basic cfg — shared with entry.h.
extern EventList* g_eventlist;
extern int g_gpus_per_server;
extern int g_rto_ms;

// ---- Prenet globals (owned by this TU) ----
inline Prenet*              g_prenet_topo           = nullptr;
inline PrenetTopoManager*   g_prenet_topomanager    = nullptr;
inline PrenetPredictor*     g_prenet_predictor      = nullptr;
inline PrenetArbiter*       g_prenet_arbiter        = nullptr;
inline PrenetVariantPool*   g_prenet_variants       = nullptr;
inline FatTreeTopology*     g_prenet_ecs_underlay   = nullptr;
inline PrenetConfig         g_prenet_cfg;

inline uint64_t g_prenet_predictions_total      = 0;
inline uint64_t g_prenet_predictions_correct    = 0;
inline uint64_t g_prenet_predictions_wrong      = 0;
inline uint64_t g_prenet_probes_emitted         = 0;
inline uint64_t g_prenet_action_stay_ecs        = 0;
inline uint64_t g_prenet_action_use_ocs_asis    = 0;
inline uint64_t g_prenet_action_reconfig_ocs    = 0;
inline uint64_t g_prenet_arbiter_wins           = 0;
inline uint64_t g_prenet_arbiter_losses         = 0;

// Per-flow meta — lives from SendFlowPrenet until both main+probe finish.
struct PrenetFlowMeta {
  uint64_t decision_id = 0;
  PrenetAction action = PrenetAction::STAY_ECS;
  uint64_t bytes_main = 0;
  uint64_t bytes_probe = 0;
  simtime_picosec start_time = 0;
  double rho_at_decision = 0.0;
  simtime_picosec end_time_main = 0;
  simtime_picosec end_time_probe = 0;
  bool main_done = false;
  bool probe_done = false;
  AstraSim::ncclFlowTag flow_tag;
  int src = 0, dst = 0;
};
inline std::unordered_map<uint64_t, PrenetFlowMeta*> g_prenet_pending_flows;

// Context passed to each TCP flow's finish callback.
struct PrenetFlowContext {
  int src;
  int dst;
  uint64_t flow_size;
  AstraSim::ncclFlowTag flow_tag;
  uint64_t decision_id;
  bool is_probe;
};

// Forward decl.
extern TopoType g_topo_type;

// ---- Helper: dead-OCS-route reroute to ECS (prenet-flavoured) ----
inline bool reroute_flow_if_dead_prenet(TcpSrc* tcp) {
  if (g_topo_type != TOPO_PRENET) return false;  // safety guard (principles §2.1)
  if (tcp == nullptr || tcp->_finished) return false;
  if (tcp->is_elec) return false;
  if (g_prenet_topo == nullptr) return false;
  if (tcp->_route == nullptr) return false;

  bool dead = false;
  for (Route::const_iterator it = tcp->_route->begin(); it != tcp->_route->end(); ++it) {
    Queue* q = dynamic_cast<Queue*>(*it);
    if (q != nullptr && q->_bitrate == 0) { dead = true; break; }
  }
  if (!dead) return false;

  auto* fwd = g_prenet_topo->get_eps_paths(tcp->_flow_src, tcp->_flow_dst);
  auto* bck = g_prenet_topo->get_eps_paths(tcp->_flow_dst, tcp->_flow_src);
  if (fwd == nullptr || fwd->empty() || bck == nullptr || bck->empty()) {
    if (fwd) { for (const Route* r : *fwd) delete r; delete fwd; }
    if (bck) { for (const Route* r : *bck) delete r; delete bck; }
    return false;
  }

  Route* newfwd = new Route(*(fwd->at(rand() % fwd->size())));
  Route* newback = new Route(*(bck->at(rand() % bck->size())));
  newfwd->push_back(tcp->_sink);
  newback->push_back(tcp);
  newfwd->set_reverse(newback);
  newback->set_reverse(newfwd);
  tcp->reroute_to(newfwd, newback);
  tcp->is_elec = true;
  // Free the path containers (we kept copies via `new Route(...)`).
  for (const Route* r : *fwd) delete r; delete fwd;
  for (const Route* r : *bck) delete r; delete bck;
  std::cerr << "[PRENET][REROUTE_ECS] src=" << tcp->_flow_src << " dst=" << tcp->_flow_dst
            << std::endl;
  return true;
}

// ---- Finish callback: prenet-specific ----
inline void htsim_flow_finish_prenet(void* ctx_ptr) {
  PrenetFlowContext* pctx = static_cast<PrenetFlowContext*>(ctx_ptr);
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,
      "[prenet] flow finish src %d dst %d size %llu probe=%d",
      pctx->src, pctx->dst, (unsigned long long)pctx->flow_size, (int)pctx->is_probe);

  int sid = pctx->src;
  int did = pctx->dst;
  uint64_t flow_size = pctx->flow_size;
  AstraSim::ncclFlowTag flowTag = pctx->flow_tag;
  uint64_t decision_id = pctx->decision_id;
  bool is_probe = pctx->is_probe;
  delete pctx;

  // ---- 1. SimAI chunk-completion plumbing (identical to entry.h's version) ----
  int flow_id = flowTag.current_flow_id;
  received_chunksize[std::make_pair(flow_id, std::make_pair(sid, did))] += flow_size;
  if (is_receive_finished(sid, did, flowTag)) {
    uint64_t notify_size = received_chunksize[std::make_pair(flow_id, std::make_pair(sid, did))];
    received_chunksize.erase(std::make_pair(flow_id, std::make_pair(sid, did)));
    notify_receiver_receive_data(sid, did, notify_size, flowTag);
  }
  sent_chunksize[std::make_pair(flow_id, std::make_pair(sid, did))] += flow_size;
  if (is_sending_finished(sid, did, flowTag)) {
    uint64_t all_sent = sent_chunksize[std::make_pair(flow_id, std::make_pair(sid, did))];
    sent_chunksize.erase(std::make_pair(flow_id, std::make_pair(sid, did)));
    notify_sender_sending_finished(sid, did, all_sent, flowTag);
  }

  // ---- 2. Prenet update closed loop ----
  if (decision_id == 0) return;  // skeleton-mode flow, no meta
  auto mit = g_prenet_pending_flows.find(decision_id);
  if (mit == g_prenet_pending_flows.end()) return;
  PrenetFlowMeta* meta = mit->second;
  simtime_picosec now = g_eventlist->now();
  if (is_probe) {
    meta->end_time_probe = now;
    meta->probe_done = true;
  } else {
    meta->end_time_main = now;
    meta->main_done = true;
  }
  bool probe_ok = (meta->bytes_probe == 0) || meta->probe_done;
  if (meta->main_done && probe_ok && g_prenet_predictor != nullptr) {
    UpdateInput in;
    in.decision_id = meta->decision_id;
    in.taken_action = meta->action;
    in.bytes_main = meta->bytes_main;
    in.bytes_probe = meta->bytes_probe;
    in.start_time = meta->start_time;
    in.end_time_main = meta->end_time_main;
    in.end_time_probe = meta->end_time_probe;
    in.fattree_rho_at_finish = g_prenet_predictor->probe_rho_snapshot();
    g_prenet_predictor->update(in);

    g_prenet_predictions_total = g_prenet_predictor->total_predictions();
    g_prenet_predictions_correct = g_prenet_predictor->correct_predictions();
    g_prenet_predictions_wrong = g_prenet_predictor->wrong_predictions();

    g_prenet_pending_flows.erase(mit);
    delete meta;
  }
}

// ---- Helper: create a TCP src+sink on a given route, register with scanner ----
extern std::ofstream g_fct_output;
extern TcpRtxTimerScanner* g_tcp_scanner;
extern uint64_t g_total_tcp_flows_created;

// Free a path container previously returned by get_paths/get_eps_paths.
inline void free_path_container(std::vector<const Route*>* p) {
  if (!p) return;
  for (const Route* r : *p) delete r;
  delete p;
}

// Returns flowSrc on success. On failure (no usable path even via ECS), returns
// nullptr — caller MUST NOT increment waiting counters in that case (otherwise
// SimAI will hang waiting for a notify that never comes).
//
// `out_used_ocs` (optional): on success, set to true iff the flow was actually
// placed on an OCS route. Lets callers detect silent OCS→ECS degradation so
// they can correct meta->action / skip unnecessary probes.
inline TcpSrc* prenet_spawn_flow(int src, int dst, uint64_t bytes,
                                 bool use_ocs, bool is_probe,
                                 const AstraSim::ncclFlowTag& flow_tag,
                                 uint64_t decision_id, int com_type,
                                 bool* out_used_ocs = nullptr) {
  std::vector<const Route*>* srcpaths = nullptr;
  std::vector<const Route*>* dstpaths = nullptr;
  if (use_ocs) {
    srcpaths = g_prenet_topo->get_paths(src, dst);
    dstpaths = g_prenet_topo->get_paths(dst, src);
    if (srcpaths == nullptr || srcpaths->empty() ||
        dstpaths == nullptr || dstpaths->empty()) {
      // OCS path disappeared — degrade to ECS. Free the empty containers first.
      free_path_container(srcpaths);
      free_path_container(dstpaths);
      use_ocs = false;
      srcpaths = g_prenet_topo->get_eps_paths(src, dst);
      dstpaths = g_prenet_topo->get_eps_paths(dst, src);
    }
  } else {
    srcpaths = g_prenet_topo->get_eps_paths(src, dst);
    dstpaths = g_prenet_topo->get_eps_paths(dst, src);
  }
  if (srcpaths == nullptr || srcpaths->empty() ||
      dstpaths == nullptr || dstpaths->empty()) {
    std::cerr << "[PRENET][DROP] no path for src " << src << " dst " << dst
              << " use_ocs=" << use_ocs << std::endl;
    free_path_container(srcpaths);
    free_path_container(dstpaths);
    if (out_used_ocs) *out_used_ocs = false;
    return nullptr;
  }

  PrenetFlowContext* ctx = new PrenetFlowContext{src, dst, bytes, flow_tag, decision_id, is_probe};

  g_total_tcp_flows_created++;
  DCTCPSrc* flowSrc = new DCTCPSrc(nullptr, nullptr, &g_fct_output,
                                   *g_eventlist, src, dst,
                                   htsim_flow_finish_prenet, (void*)ctx);
  TcpSink* flowSnk = new TcpSink();
  flowSrc->set_flowsize(bytes);
  flowSrc->set_ssthresh(65535 * Packet::data_packet_size());
  flowSrc->_rto = timeFromMs(g_rto_ms);
  flowSrc->is_elec = !use_ocs;
  flowSrc->is_all2all = (com_type == 4);

  if (g_tcp_scanner) g_tcp_scanner->registerTcp(*flowSrc);

  int choice = rand() % (int)srcpaths->size();
  Route* routeout = new Route(*(srcpaths->at(choice)));
  routeout->push_back(flowSnk);

  choice = rand() % (int)dstpaths->size();
  Route* routein = new Route(*(dstpaths->at(choice)));
  routein->push_back(flowSrc);

  flowSrc->connect(*routeout, *routein, *flowSnk, g_eventlist->now());
  // Free the path containers (we kept deep copies above).
  free_path_container(srcpaths);
  free_path_container(dstpaths);
  if (out_used_ocs) *out_used_ocs = use_ocs;
  return flowSrc;
}

// ---- Actually send: ECS-only convenience ----
// ECS (fattree underlay) should ALWAYS have a path — failure here means an
// invariant is broken (principles §5 second class) and we abort loudly rather
// than let SimAI hang waiting for a notify that will never come.
inline void prenet_send_ecs_only(int src, int dst, uint64_t bytes,
                                 const AstraSim::ncclFlowTag& flow_tag,
                                 uint64_t decision_id, int com_type) {
  TcpSrc* s = prenet_spawn_flow(src, dst, bytes, /*use_ocs=*/false, /*is_probe=*/false,
                                flow_tag, decision_id, com_type);
  if (s == nullptr) {
    std::cerr << "[PRENET][FATAL] ECS spawn failed src=" << src << " dst=" << dst
              << " bytes=" << bytes << " — fattree underlay missing path?" << std::endl;
    assert(0 && "prenet_send_ecs_only: ECS fattree should always have a path");
  }
  waiting_to_sent_callback[std::make_pair(flow_tag.current_flow_id, std::make_pair(src, dst))]++;
  waiting_to_notify_receiver[std::make_pair(flow_tag.current_flow_id, std::make_pair(src, dst))]++;
}

// ---- SendFlowPrenet ----
void SendFlowPrenet(int src, int dst, uint64_t maxPacketCount,
                    void (*msg_handler)(void*), void* fun_arg,
                    int tag, AstraSim::sim_request* request);

// Deferred send (prenet version). Two flavours:
//   (a) Region-busy defer: a new flow arrived while the target region was in
//       the middle of a reconfig. Replay by re-calling SendFlowPrenet — the
//       predictor runs fresh (new rho, new conn state).
//   (b) Reconfig-winner defer: this exact flow's predictor call won the
//       arbitration and caused the reconfig. After reconf_delay, we must
//       spawn an OCS flow with the ORIGINAL decision_id so the predictor
//       closes the update loop. We do NOT re-run predict() — the decision is
//       already locked in.
// The `is_reconfig_winner` flag + `decision_id` lets one event class handle
// both cases without the old `g_prenet_replaying_deferred` global.
struct PrenetDeferredData {
  int src, dst;
  uint64_t count;
  // Lifetime assumption: msg_handler function ptr is static code (never moved),
  // and fun_arg points into SimAI's Sys instance / sentHash entries that outlive
  // any reconf_delay + arbiter_window + TCP flow completion. If this assumption
  // ever breaks (e.g., Workload cleanup mid-iteration), these fields become
  // dangling — but then entry.h's mixnet DeferredSendEvent has the same issue.
  void (*msg_handler)(void*);
  void* fun_arg;
  int tag;
  AstraSim::sim_request request;  // POD: ncclFlowTag + scalars, safe to copy
  uint64_t decision_id;       // 0 for case (a); real did for (b)
  bool is_reconfig_winner;    // true for (b)
};

class PrenetDeferredSendEvent : public EventSource {
public:
  PrenetDeferredData _data;
  PrenetDeferredSendEvent(EventList& el, PrenetDeferredData data)
    : EventSource(el, "prenet_deferred_send"), _data(data) {}
  void doNextEvent() override;
};

// ---- SendFlowPrenet (definition) ----
inline void SendFlowPrenet(int src, int dst, uint64_t maxPacketCount,
                           void (*msg_handler)(void*), void* fun_arg,
                           int tag, AstraSim::sim_request* request) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (maxPacketCount == 0) maxPacketCount = 1;

  uint32_t port = portNumber[src][dst]++;
  sender_src_port_map[std::make_pair(port, std::make_pair(src, dst))] = request->flowTag;

  int flow_id = request->flowTag.current_flow_id;
  int com_type = request->flowTag.com_type;

  int src_machine = src / g_gpus_per_server;
  int dst_machine = dst / g_gpus_per_server;

  // ---- NVLink same-machine: mirror entry.h's NVLink block exactly ----
  if (src_machine == dst_machine) {
    double nvlink_bw_bps = 900e9;
    simtime_picosec transfer_time = (simtime_picosec)(maxPacketCount * 8.0 / nvlink_bw_bps * 1e12);
    if (transfer_time < 1000) transfer_time = 1000;

    // NVLink: no TCP — schedule the finish callback directly.
    // (schedule_callback is declared at file scope earlier; entry.h defines it.)
    PrenetFlowContext* ctx = new PrenetFlowContext{src, dst, maxPacketCount,
                                                   request->flowTag, 0, false};
    schedule_callback(transfer_time, htsim_flow_finish_prenet, (void*)ctx);

    waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    return;
  }

  // ---- Cross-machine: decide via predictor ----
  if (g_prenet_predictor == nullptr || g_prenet_topo == nullptr) {
    // Skeleton mode (PR-2/PR-3/PR-4): just do ECS. ECS path existence is an
    // invariant of prenet's fattree underlay — failure aborts (see
    // prenet_send_ecs_only comment).
    prenet_send_ecs_only(src, dst, maxPacketCount, request->flowTag, 0, com_type);
    return;
  }

  // If EITHER endpoint's region is currently reconfiguring, defer until both
  // settle. Looking only at src_region (the previous behavior) could send a
  // flow into a partially-rewired dst region and race with apply_variant.
  int region_id_src = src_machine / g_prenet_topo->region_size;
  int region_id_dst = dst_machine / g_prenet_topo->region_size;
  if (g_prenet_topomanager) {
    simtime_picosec end_src = g_prenet_topomanager->region_reconfig_end(region_id_src);
    simtime_picosec end_dst = g_prenet_topomanager->region_reconfig_end(region_id_dst);
    simtime_picosec end = (end_src > end_dst) ? end_src : end_dst;
    if (end > g_eventlist->now()) {
      PrenetDeferredData d{src, dst, maxPacketCount, msg_handler, fun_arg, tag,
                           *request, /*decision_id=*/0,
                           /*is_reconfig_winner=*/false};
      PrenetDeferredSendEvent* ev = new PrenetDeferredSendEvent(*g_eventlist, d);
      g_eventlist->sourceIsPending(*ev, end);
      return;
    }
  }
  int region_id = region_id_src;  // keep variable name below

  // --- Predict ---
  PredictResult pr = g_prenet_predictor->predict(src, dst, maxPacketCount,
                                                 com_type, region_id);
  switch (pr.action) {
    case PrenetAction::STAY_ECS:     g_prenet_action_stay_ecs++; break;
    case PrenetAction::USE_OCS_ASIS: g_prenet_action_use_ocs_asis++; break;
    case PrenetAction::RECONFIG_OCS: g_prenet_action_reconfig_ocs++; break;
  }

  // --- RECONFIG_OCS: go through arbiter ---
  if (pr.action == PrenetAction::RECONFIG_OCS) {
    PrenetDeferredData d_winner{src, dst, maxPacketCount, msg_handler, fun_arg, tag,
                                *request, pr.decision_id,
                                /*is_reconfig_winner=*/true};
    int variant_id = pr.variant_id;
    double confidence = (double)pr.confidence;
    uint64_t msg_size = maxPacketCount;
    uint64_t decision_id = pr.decision_id;
    auto cb = [d_winner, decision_id](bool granted, simtime_picosec end_time) mutable {
      if (granted) {
        g_prenet_arbiter_wins++;
        // Carry the same decision_id through the defer so the OCS flow can
        // close the predictor's update loop when it finishes.
        PrenetDeferredSendEvent* ev = new PrenetDeferredSendEvent(*g_eventlist, d_winner);
        g_eventlist->sourceIsPending(*ev, end_time);
      } else {
        g_prenet_arbiter_losses++;
        // Arbiter rejected us — the RECONFIG_OCS action was NOT executed.
        // Drop the pending ctx so the predictor doesn't leak entries, and
        // don't inject a stale update.
        if (g_prenet_predictor) g_prenet_predictor->erase_pending(decision_id);
        // Fallback: send via ECS now. decision_id=0 keeps this flow out of the
        // update loop (its outcome doesn't belong to the predicted action).
        prenet_send_ecs_only(d_winner.src, d_winner.dst, d_winner.count,
                             d_winner.request.flowTag,
                             0, d_winner.request.flowTag.com_type);
      }
    };
    g_prenet_topomanager->request_reconfig(region_id, variant_id, confidence, msg_size, cb);
    return;
  }

  // --- STAY_ECS ---
  if (pr.action == PrenetAction::STAY_ECS) {
    uint64_t did = pr.decision_id;
    TcpSrc* s = prenet_spawn_flow(src, dst, maxPacketCount, /*use_ocs=*/false,
                                  /*is_probe=*/false,
                                  request->flowTag, did, com_type,
                                  /*out_used_ocs=*/nullptr);
    if (s == nullptr) {
      // ECS always-reachable invariant (principles §5 second class).
      std::cerr << "[PRENET][FATAL] STAY_ECS spawn failed src=" << src
                << " dst=" << dst << std::endl;
      if (g_prenet_predictor) g_prenet_predictor->erase_pending(did);
      assert(0 && "STAY_ECS spawn failed — fattree path missing?");
      return;
    }
    PrenetFlowMeta* meta = new PrenetFlowMeta();
    meta->decision_id = did;
    meta->action = pr.action;  // STAY_ECS; used_ocs must be false for this path
    meta->bytes_main = maxPacketCount;
    meta->bytes_probe = 0;
    meta->start_time = g_eventlist->now();
    meta->rho_at_decision = 0.0;
    meta->flow_tag = request->flowTag;
    meta->src = src; meta->dst = dst;
    g_prenet_pending_flows[did] = meta;
    waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    return;
  }

  // --- USE_OCS_ASIS (with optional probe) ---
  // v0 safety: probe-split is only safe for PURE AllToAll (each pair
  // independent). Predictor guarantees emit_probe=true ⇒ coll_type==AllToAll,
  // but we double-check here to be defensive.
  bool probe_safe_coll = (com_type == 4 /*AllToAll*/);
  uint64_t decision_id = pr.decision_id;

  // ---- Pre-check path availability ----
  // Split is only safe if BOTH OCS and ECS are reachable — otherwise we'd end
  // up with a half-spawned flow that SimAI waits on forever. We check before
  // any spawn so failure means "don't split" rather than "rollback in flight".
  bool want_split = pr.emit_probe && probe_safe_coll;
  bool ocs_reachable = false;
  {
    auto* f = g_prenet_topo->get_paths(src, dst);
    auto* b = g_prenet_topo->get_paths(dst, src);
    ocs_reachable = f && !f->empty() && b && !b->empty();
    free_path_container(f);
    free_path_container(b);
  }
  bool ecs_reachable = false;
  {
    auto* f = g_prenet_topo->get_eps_paths(src, dst);
    auto* b = g_prenet_topo->get_eps_paths(dst, src);
    ecs_reachable = f && !f->empty() && b && !b->empty();
    free_path_container(f);
    free_path_container(b);
  }
  if (!ecs_reachable) {
    // Fattree underlay should always connect — abort loudly (principles §5).
    std::cerr << "[PRENET][FATAL] ECS unreachable src=" << src << " dst=" << dst
              << std::endl;
    assert(0 && "fattree ECS path missing — prenet invariant broken");
  }
  if (want_split && !ocs_reachable) {
    // OCS gone between predict() and spawn; no split, flow will degrade.
    want_split = false;
  }

  uint64_t probe_bytes = 0;
  uint64_t main_bytes = maxPacketCount;
  if (want_split) {
    probe_bytes = (uint64_t)(maxPacketCount * pr.probe_ratio);
    if (probe_bytes == 0) probe_bytes = 1;
    if (probe_bytes >= maxPacketCount) probe_bytes = maxPacketCount / 20 + 1;
    main_bytes = maxPacketCount - probe_bytes;
  }

  // Spawn main. `want_split=true` ⇒ OCS reachable ⇒ this should land on OCS.
  // `want_split=false` ⇒ either coll disallows probe, emit_probe was false,
  // or OCS unreachable; spawn asks for OCS but degrades cleanly to ECS.
  bool main_used_ocs = false;
  TcpSrc* main_src = prenet_spawn_flow(src, dst, main_bytes, /*use_ocs=*/true,
                                       /*is_probe=*/false,
                                       request->flowTag, decision_id, com_type,
                                       &main_used_ocs);
  if (main_src == nullptr) {
    // Shouldn't happen: ecs_reachable was true. Defensive cleanup only.
    std::cerr << "[PRENET][FATAL] USE_OCS_ASIS main spawn failed src=" << src
              << " dst=" << dst << std::endl;
    if (g_prenet_predictor) g_prenet_predictor->erase_pending(decision_id);
    assert(0 && "main spawn failed despite ecs_reachable pre-check");
    return;
  }
  PrenetFlowMeta* meta = new PrenetFlowMeta();
  meta->decision_id = decision_id;
  // meta->action reflects what actually happened, not what was predicted.
  meta->action = main_used_ocs ? pr.action : PrenetAction::STAY_ECS;
  meta->bytes_main = main_bytes;
  meta->bytes_probe = 0;  // may grow below
  meta->start_time = g_eventlist->now();
  meta->rho_at_decision = 0.0;
  meta->flow_tag = request->flowTag;
  meta->src = src; meta->dst = dst;
  g_prenet_pending_flows[decision_id] = meta;
  waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))]++;
  waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]++;

  if (want_split) {
    // Pre-check guaranteed ECS reachable. Spawn probe on ECS.
    TcpSrc* probe_src = prenet_spawn_flow(src, dst, probe_bytes, /*use_ocs=*/false,
                                          /*is_probe=*/true,
                                          request->flowTag, decision_id, com_type,
                                          /*out_used_ocs=*/nullptr);
    if (probe_src == nullptr) {
      // Should be impossible given the pre-check (single-threaded htsim: no
      // topology mutation between pre-check and here). Treat as broken
      // invariant rather than orphan half a flow.
      std::cerr << "[PRENET][FATAL] probe spawn failed after pre-check src=" << src
                << " dst=" << dst << std::endl;
      assert(0 && "probe spawn failed despite ecs_reachable pre-check");
    }
    meta->bytes_probe = probe_bytes;
    waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    g_prenet_probes_emitted++;
  }
  // If !want_split, main carries the full maxPacketCount (main_bytes was
  // initialized to maxPacketCount and not adjusted). No tail needed. This
  // covers the OCS-unreachable-pre-check case too: main degraded to ECS,
  // carries full payload, chunk accumulator will reach maxPacketCount from
  // the single flow's finish.
}

inline void PrenetDeferredSendEvent::doNextEvent() {
  if (_data.is_reconfig_winner && _data.decision_id != 0) {
    // Reconfig-winner path: the original SendFlowPrenet already paid the
    // predict() bookkeeping cost and registered ctx in _pending. Do NOT
    // re-enter the predictor — just execute the OCS flow and register a meta
    // so the finish path closes update().
    int src = _data.src, dst = _data.dst;
    uint64_t bytes = _data.count;
    int com_type = _data.request.flowTag.com_type;
    int flow_id = _data.request.flowTag.current_flow_id;
    uint64_t decision_id = _data.decision_id;

    bool used_ocs = false;
    TcpSrc* s = prenet_spawn_flow(src, dst, bytes, /*use_ocs=*/true,
                                  /*is_probe=*/false,
                                  _data.request.flowTag, decision_id, com_type,
                                  &used_ocs);
    if (s == nullptr) {
      // spawn_flow internally degrades OCS→ECS; ECS always-reachable invariant.
      std::cerr << "[PRENET][FATAL] reconfig-winner spawn failed src=" << src
                << " dst=" << dst << std::endl;
      if (g_prenet_predictor) g_prenet_predictor->erase_pending(decision_id);
      assert(0 && "reconfig-winner spawn failed — fattree path missing?");
      delete this;
      return;
    }
    PrenetFlowMeta* meta = new PrenetFlowMeta();
    meta->decision_id = decision_id;
    // If OCS actually held (it should — we just reconfigured), action is
    // RECONFIG_OCS; if spawn degraded to ECS (another region's later reconfig
    // tore it down in the window), record STAY_ECS for accurate attribution.
    meta->action = used_ocs ? PrenetAction::RECONFIG_OCS : PrenetAction::STAY_ECS;
    meta->bytes_main = bytes;
    meta->bytes_probe = 0;     // no probe on replay — decision already locked
    meta->start_time = g_eventlist->now();
    meta->rho_at_decision = 0.0;
    meta->flow_tag = _data.request.flowTag;
    meta->src = src; meta->dst = dst;
    g_prenet_pending_flows[decision_id] = meta;
    waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    delete this;
    return;
  }

  // Region-busy defer (a new flow arrived mid-reconfig): re-enter SendFlowPrenet
  // so the predictor sees fresh conn/rho state.
  SendFlowPrenet(_data.src, _data.dst, _data.count,
                 _data.msg_handler, _data.fun_arg, _data.tag, &_data.request);
  delete this;
}

#endif // PRENET_ENABLED
#endif // __ENTRY_PRENET_H__
