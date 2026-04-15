/*
 * htsim network backend entry.h
 * Replaces ns3/entry.h for the htsim-based OCS-ECS hybrid network backend.
 * Reuses the same callback mechanism (sentHash, expeRecvHash, recvHash, notify_*)
 * but replaces NS-3 RDMA flows with htsim DCTCP TCP flows.
 */

#ifndef __ENTRY_HTSIM_H__
#define __ENTRY_HTSIM_H__

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"

// htsim headers
#include "eventlist.h"
#include "tcp.h"
#include "dctcp.h"
#include "config.h"
#include "topology.h"
#include "mixnet.h"
#include "fat_tree_topology.h"
#include "fc_topology.h"
#include "flat_topology.h"
#include "os_fattree.h"
#include "agg_os_fattree.h"
#include "mixnet_topomanager.h"

#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

// ======== Data structures (same as ns3/entry.h) ========
std::map<std::pair<std::pair<int, int>, int>, AstraSim::ncclFlowTag> receiver_pending_queue;
std::map<std::pair<int, std::pair<int, int>>, AstraSim::ncclFlowTag> sender_src_port_map;

struct task1 {
  int src;
  int dest;
  int type;
  uint64_t count;
  void *fun_arg;
  void (*msg_handler)(void *fun_arg);
  double schTime;
};

map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;
map<std::pair<int, std::pair<int, int>>, uint64_t> recvHash;
map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;
map<std::pair<int, int>, int64_t> nodeHash;
map<std::pair<int, std::pair<int, int>>, int> waiting_to_sent_callback;
map<std::pair<int, std::pair<int, int>>, int> waiting_to_notify_receiver;
map<std::pair<int, std::pair<int, int>>, uint64_t> received_chunksize;
map<std::pair<int, std::pair<int, int>>, uint64_t> sent_chunksize;

// Port number tracking (same as ns3/common.h)
std::unordered_map<uint32_t, unordered_map<uint32_t, uint16_t>> portNumber;

// ======== Topology type enum ========
enum TopoType {
  TOPO_MIXNET,       // OCS-ECS hybrid (Mixnet wrapping FatTree)
  TOPO_FATTREE,      // Pure fat-tree (full bandwidth)
  TOPO_OS_FATTREE,   // Oversubscribed fat-tree
  TOPO_AGG_OS_FATTREE, // Aggregated oversubscribed fat-tree
  TOPO_FC,           // Full circuit (all-to-all direct)
  TOPO_FLAT,         // Flat topology
};

// ======== htsim global objects ========
EventList* g_eventlist = nullptr;
Topology* g_topology = nullptr;       // Generic topology pointer (all topos)
Mixnet* g_mixnet_topo = nullptr;      // Only set for TOPO_MIXNET
FatTreeTopology* g_fattree_topo = nullptr;
MixnetTopoManager* g_topomanager = nullptr;
All2AllTrafficRecorder* g_demand_recorder = nullptr;
TcpRtxTimerScanner* g_tcp_scanner = nullptr;
TopoType g_topo_type = TOPO_MIXNET;

// OCS/ECS config
int g_gpus_per_server = 8;
bool g_force_ecs_only = false;  // When true, all cross-machine traffic uses ECS
int g_rto_ms = 1;  // TCP retransmission timeout in ms
int g_total_gpus = 0;  // Total GPU count, for FCT sampling

// Expert hotspot parameters (MoE non-uniform all-to-all traffic)
int g_expert_topk = 0;       // top-k routing (0=uniform, no hotspot)
double g_expert_skew = 1.0;  // Zipf distribution skew parameter
int g_expert_seed = 42;      // random seed for expert routing
int g_moe_volatility = 1;    // MoE hotspot bucket size: every N layers share one distribution (1=per-layer)

// Reconfig skip: skip OCS reconfig if top-N traffic pairs are already connected
int g_reconf_top_n = 0;      // 0=always reconfig, >0=skip if top-N pairs covered


// Global retransmission counters (accumulated as flows finish, not at end)
uint64_t g_total_packets_sent = 0;
uint64_t g_total_retransmissions = 0;
uint64_t g_total_tcp_flows = 0;       // finished TCP flows
uint64_t g_total_tcp_flows_created = 0; // total TCP flows created

// Per-ComType traffic breakdown. Index == ComType int value.
// 0=None,1=Reduce_Scatter,2=All_Gather,3=All_Reduce,4=All_to_All,
// 5=All_Reduce_All_to_All,6=All_Reduce_NVLS
static const int G_COMTYPE_N = 8;
uint64_t g_bytes_by_comtype[G_COMTYPE_N] = {0};
uint64_t g_flows_by_comtype[G_COMTYPE_N] = {0};
inline const char* comtype_name(int t) {
  switch (t) {
    case 0: return "None";
    case 1: return "ReduceScatter";
    case 2: return "AllGather";
    case 3: return "AllReduce";
    case 4: return "AllToAll";
    case 5: return "AllReduce+AllToAll";
    case 6: return "AllReduceNVLS";
    default: return "Other";
  }
}

// FCT output stream for tcp.cpp (required by TcpSrc::receivePacket at flow completion)
std::ofstream g_fct_output;

void init_fct_output(const std::string& filename = "fct_output.txt") {
  g_fct_output.open(filename, std::ios::out);
  if (!g_fct_output.is_open()) {
    std::cerr << "[WARN] Could not open FCT output file: " << filename << std::endl;
  }
}

// ======== CallbackEvent: bridges astra-sim void(*)(void*) to htsim EventSource ========
class CallbackEvent : public EventSource {
public:
  void (*_fun_ptr)(void*);
  void* _fun_arg;

  CallbackEvent(EventList& el, void (*fun_ptr)(void*), void* fun_arg)
      : EventSource(el, "callback"), _fun_ptr(fun_ptr), _fun_arg(fun_arg) {}

  void doNextEvent() override {
    _fun_ptr(_fun_arg);
    delete this;  // one-shot event
  }
};

void schedule_callback(simtime_picosec delay_ps, void (*fun_ptr)(void*), void* fun_arg) {
  CallbackEvent* ev = new CallbackEvent(*g_eventlist, fun_ptr, fun_arg);
  g_eventlist->sourceIsPendingRel(*ev, delay_ps);
}

// ======== Flow completion tracking (same logic as ns3/entry.h) ========
bool is_sending_finished(int src, int dst, AstraSim::ncclFlowTag flowTag) {
  int tag_id = flowTag.current_flow_id;
  if (waiting_to_sent_callback.count(
          std::make_pair(tag_id, std::make_pair(src, dst)))) {
    if (--waiting_to_sent_callback[std::make_pair(
            tag_id, std::make_pair(src, dst))] == 0) {
      waiting_to_sent_callback.erase(
          std::make_pair(tag_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}

bool is_receive_finished(int src, int dst, AstraSim::ncclFlowTag flowTag) {
  int tag_id = flowTag.current_flow_id;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (waiting_to_notify_receiver.count(
          std::make_pair(tag_id, std::make_pair(src, dst)))) {
    NcclLog->writeLog(NcclLogLevel::DEBUG,
        " is_receive_finished waiting_to_notify_receiver tag_id %d src %d dst %d count %d",
        tag_id, src, dst,
        waiting_to_notify_receiver[std::make_pair(tag_id, std::make_pair(src, dst))]);
    if (--waiting_to_notify_receiver[std::make_pair(
            tag_id, std::make_pair(src, dst))] == 0) {
      waiting_to_notify_receiver.erase(
          std::make_pair(tag_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}

// ======== notify callbacks (copied from ns3/entry.h, NS-3 independent) ========
void notify_receiver_receive_data(int sender_node, int receiver_node,
                                  uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG,
        " %d notify receiver: %d message size: %llu", sender_node, receiver_node, message_size);
    int tag = flowTag.tag_id;
    if (expeRecvHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) != expeRecvHash.end()) {
      task1 t2 = expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))];
      NcclLog->writeLog(NcclLogLevel::DEBUG,
          " %d notify receiver: %d message size: %llu t2.count: %llu channel id: %d",
          sender_node, receiver_node, message_size, t2.count, flowTag.channel_id);
      bool is_pp_simple_recv = (tag >= 2000000); // PP point-to-point recv
      if (message_size == t2.count) {
        expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        if (!is_pp_simple_recv) {
          AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*)t2.fun_arg;
          assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
          ehd->flowTag = flowTag;
        }
        t2.msg_handler(t2.fun_arg);
        goto receiver_end_1st_section;
      } else if (message_size > t2.count) {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] =
            message_size - t2.count;
        expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        if (!is_pp_simple_recv) {
          AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*)t2.fun_arg;
          assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
          ehd->flowTag = flowTag;
        }
        t2.msg_handler(t2.fun_arg);
        goto receiver_end_1st_section;
      } else {
        t2.count -= message_size;
        expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))] = t2;
      }
    } else {
      receiver_pending_queue[std::make_pair(std::make_pair(receiver_node, sender_node), tag)] = flowTag;
      if (recvHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) == recvHash.end()) {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] = message_size;
      } else {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] += message_size;
      }
    }
  receiver_end_1st_section:
    if (nodeHash.find(make_pair(receiver_node, 1)) == nodeHash.end()) {
      nodeHash[make_pair(receiver_node, 1)] = message_size;
    } else {
      nodeHash[make_pair(receiver_node, 1)] += message_size;
    }
  }
}

void notify_sender_sending_finished(int sender_node, int receiver_node,
                                    uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int tag = flowTag.tag_id;
    if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) != sentHash.end()) {
      task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
      bool is_pp_simple_send = (tag >= 2000000); // PP point-to-point send
      if (!is_pp_simple_send) {
        AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*)t2.fun_arg;
        ehd->flowTag = flowTag;
      }
      if (t2.count == message_size) {
        sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
          nodeHash[make_pair(sender_node, 0)] = message_size;
        } else {
          nodeHash[make_pair(sender_node, 0)] += message_size;
        }
        if (t2.msg_handler != nullptr) {
          t2.msg_handler(t2.fun_arg);
        }
        goto sender_end_1st_section;
      } else {
        NcclLog->writeLog(NcclLogLevel::ERROR,
            "sentHash msg size != sender_node %d receiver_node %d message_size %lu",
            sender_node, receiver_node, message_size);
      }
    } else {
      NcclLog->writeLog(NcclLogLevel::ERROR,
          "sentHash can't find sender_node %d receiver_node %d message_size %lu",
          sender_node, receiver_node, message_size);
    }
  }
sender_end_1st_section:
  return;
}

// ======== htsim flow context for completion callback ========
struct HtsimFlowContext {
  int src;
  int dst;
  uint64_t flow_size;
  AstraSim::ncclFlowTag flowTag;
};

// Called when a htsim TCP flow completes (via application_callback)
void htsim_flow_finish(void* ctx_ptr) {
  HtsimFlowContext* ctx = static_cast<HtsimFlowContext*>(ctx_ptr);
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,
      "[htsim] flow finish src %d dst %d size %llu",
      ctx->src, ctx->dst, ctx->flow_size);

  int sid = ctx->src;
  int did = ctx->dst;
  uint64_t flow_size = ctx->flow_size;
  AstraSim::ncclFlowTag flowTag = ctx->flowTag;
  delete ctx;

  // Accumulate received chunk size and check completion
  received_chunksize[std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did))] += flow_size;
  if (!is_receive_finished(sid, did, flowTag)) {
    return;
  }
  uint64_t notify_size = received_chunksize[std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did))];
  received_chunksize.erase(std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did)));

  // Notify receiver
  notify_receiver_receive_data(sid, did, notify_size, flowTag);

  // Accumulate sent chunk size and check completion
  sent_chunksize[std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did))] += flow_size;
  if (!is_sending_finished(sid, did, flowTag)) {
    return;
  }
  uint64_t all_sent = sent_chunksize[std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did))];
  sent_chunksize.erase(std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did)));

  // Notify sender
  notify_sender_sending_finished(sid, did, all_sent, flowTag);

  // NOTE: Reactive reconfig trigger removed. Proactive reconfig is now handled
  // in SendFlow() via MoEReconfigManager::on_a2a_flow_start().
}

// ======== Deferred send (for flows during reconfiguration) ========
struct DeferredSendData {
  int src, dst;
  uint64_t count;
  void (*msg_handler)(void*);
  void* fun_arg;
  int tag;
  AstraSim::sim_request request;
};

// When a flow was deferred by the proactive-reconfig path, its initial SendFlow
// call already accumulated the flow into MoEReconfigManager's layer_tm. Replay
// through DeferredSendEvent::doNextEvent must NOT re-enter the manager, or the
// bytes are double-counted. htsim is a single-threaded event-driven simulator,
// so a plain static bool is sufficient as a re-entrancy guard.
static bool g_replaying_deferred_flow = false;

class DeferredSendEvent : public EventSource {
public:
  DeferredSendData _data;
  DeferredSendEvent(EventList& el, DeferredSendData data)
      : EventSource(el, "deferred_send"), _data(data) {}
  void doNextEvent() override;  // forward declaration, defined after SendFlow
};

// ======== Proactive MoE OCS Reconfiguration Manager ========
// Replaces the reactive LayerDemandTracker with a proactive algorithm:
//   - fwd dispatch: reconfig using prediction/history BEFORE flows start
//   - fwd combine:  reuse fwd dispatch config (no reconfig), record real TM
//   - bwd combine:  reconfig using fwd real TM BEFORE flows start
//   - bwd dispatch: reuse bwd combine config (no reconfig)

struct MoEBlockInfo {
  int dispatch_layer;  // workload layer_num of moe_route (dispatch a2a)
  int combine_layer;   // workload layer_num of moe_expert (combine a2a)
};

// ============================================================================
// MoEReconfigManager (v2)
//
// Design goals:
//   - Single global singleton shared by all 128+ ranks, but event-order safe.
//   - Never infer pass/direction from layer_num ordering — ranks run concurrently
//     under event-driven simulation, so layer_num monotonicity is unreliable.
//   - Use (pass_counter, loop_state, layer_num) triples stamped on every flow
//     as the authoritative identity. pass_counter is the workload's own
//     iteration index; loop_state (0=Fwd, 1=WeightGrad, 2=InputGrad, ...) tells
//     us the training phase.
//   - Block structure (which a2a layers form a dispatch/combine pair) is
//     discovered during pass 0 forward by observing layer_nums in the order
//     they appear, then frozen once we see pass_counter >= 1 or loop_state
//     changes. Subsequent passes reuse the cached block→layer map.
//   - Traffic accumulation is keyed by (pass_counter, layer_num, region).
//     There is no rolling "current layer" window that mis-attributes flows
//     across concurrent ranks.
//   - Reconfig is triggered at most once per (pass_counter, dispatch_layer).
//     The FIRST flow observed for a given (pass, dispatch layer) triggers the
//     reconfig decision; every subsequent flow for that same (pass, layer)
//     just accumulates, regardless of which rank it came from.
//   - Prediction for pass P's block B = traffic matrix of pass (P-1)'s block B
//     once that pass finished. If the previous pass's matrix isn't finalised
//     yet when we need it, we fall back to the most recent completed copy.
// ============================================================================

struct MoEReconfigManager {
  // ---- Block structure (discovered in pass 0 forward, then frozen) ----
  // Ordered list of a2a layer_nums as they appear in a single forward pass.
  // Two consecutive a2a layers = one MoE block (dispatch + combine).
  std::vector<int> fwd_a2a_layer_order;          // sorted-as-seen, first pass only
  std::set<int> fwd_a2a_layer_set;               // dedup helper
  std::map<int,int> layer_to_block;              // layer_num → block_idx
  std::map<int,int> layer_pair_pos;              // layer_num → 0 (dispatch) / 1 (combine)
  bool structure_frozen = false;                 // true once we've seen pass >= 1

  // ---- Traffic matrices ----
  // Active accumulator: keyed by (pass_counter, layer_num, region_id)
  std::map<std::tuple<int,int,int>, Matrix2D<double>*> layer_tm;
  // Finalised block TMs: keyed by (pass_counter, block_idx, region_id)
  std::map<std::tuple<int,int,int>, Matrix2D<double>*> block_tm;
  // Prediction source: latest completed block_tm per (block_idx, region_id)
  // (overwritten whenever a newer pass's block is closed)
  std::map<std::pair<int,int>, Matrix2D<double>*> last_block_tm;
  std::map<std::pair<int,int>, int> last_block_tm_pass;  // which pass produced it

  // Idempotency guards — one entry per (pass, layer) or (pass, block).
  std::set<std::pair<int,int>> reconf_done;      // dispatch reconfig already triggered
  std::set<std::pair<int,int>> block_closed;     // block already rolled into last_block_tm

  // ---- Per-rank pass-end bookkeeping (drives hook-based drain) ----
  // `pass_end_ranks[pass]` = set of ranks that have reported completion of
  // `pass`. Once its size reaches total_ranks, pass P is fully drained and
  // we close every block of pass P.
  std::map<int, std::set<int>> pass_end_ranks;
  std::set<int> pass_fully_drained;              // passes already processed by the hook

  // ---- Reconfig statistics ----
  uint64_t reconfig_triggered = 0;
  uint64_t reconfig_skipped   = 0;

  struct ReconfigResult {
    bool should_defer;
    simtime_picosec reconfig_end_time;
  };

  // Freeze block structure from pass 0 observations — called from the pass-end
  // hook after every rank has completed pass 0. Pairs consecutive a2a layers
  // into blocks by sorted layer_num.
  void freeze_structure() {
    if (structure_frozen) return;
    if (fwd_a2a_layer_order.empty()) return;
    std::sort(fwd_a2a_layer_order.begin(), fwd_a2a_layer_order.end());
    for (int i = 0; i < (int)fwd_a2a_layer_order.size(); i++) {
      int ln = fwd_a2a_layer_order[i];
      layer_to_block[ln] = i / 2;
      layer_pair_pos[ln] = i % 2;
    }
    structure_frozen = true;
    std::cout << "[MOE_STRUCTURE] frozen: " << fwd_a2a_layer_order.size()
              << " a2a layers → " << ((fwd_a2a_layer_order.size() + 1) / 2)
              << " blocks" << std::endl;
  }

  // Record a forward-pass a2a layer during pass 0 (structure discovery).
  void record_fwd_layer(int layer_num) {
    if (structure_frozen) return;
    if (fwd_a2a_layer_set.insert(layer_num).second) {
      fwd_a2a_layer_order.push_back(layer_num);
    }
  }

  // Called by the per-rank pass-end hook (from every rank, via Workload.cc).
  // When all `total_ranks` have reported completion of pass P, pass P is
  // safe to close: every flow of pass P has already been accumulated into
  // layer_tm (because each rank issues all its a2a flows before its own
  // pass_counter++).
  void on_rank_pass_end(int rank, int pass, int total_ranks, int region_size) {
    if (pass < 0 || total_ranks <= 0) return;
    if (pass_fully_drained.count(pass)) return;   // already drained
    pass_end_ranks[pass].insert(rank);
    if ((int)pass_end_ranks[pass].size() < total_ranks) return;

    pass_fully_drained.insert(pass);
    std::cout << "[PASS_DRAINED] pass=" << pass
              << " ranks_reported=" << total_ranks << std::endl;

    // Freeze structure once pass 0 is fully observed.
    if (pass == 0 && !structure_frozen) freeze_structure();

    // Close every block of this pass. If structure isn't frozen yet (e.g. the
    // very first call with pass>=1 arrived before pass 0 drained — shouldn't
    // happen given pass_counter is per-rank monotonic, but guard anyway),
    // skip and let it close later.
    if (!structure_frozen) return;
    int num_blocks = (int)(fwd_a2a_layer_order.size() + 1) / 2;
    for (int blk = 0; blk < num_blocks; blk++) {
      close_block(pass, blk, region_size);
    }
    // Free keeping-alive entries we no longer need.
    pass_end_ranks.erase(pass);
  }

  // Close a (pass, block): sum the 2 constituent layer_tm entries into
  // block_tm and promote to last_block_tm as prediction for future passes.
  // Idempotent.
  void close_block(int pass, int block, int region_size) {
    if (pass < 0 || block < 0) return;
    auto marker = std::make_pair(pass, block);
    if (block_closed.count(marker)) return;
    block_closed.insert(marker);

    // Find the two layer_nums that belong to this block.
    std::vector<int> layers_in_block;
    for (const auto& [ln, b] : layer_to_block) {
      if (b == block) layers_in_block.push_back(ln);
    }
    if (layers_in_block.empty()) return;

    int num_regions = (g_topomanager != nullptr) ?
        (int)g_topomanager->regional_topo_managers.size() : 1;
    for (int rid = 0; rid < num_regions; rid++) {
      Matrix2D<double>* sum = nullptr;
      for (int ln : layers_in_block) {
        auto lkey = std::make_tuple(pass, ln, rid);
        auto it = layer_tm.find(lkey);
        if (it == layer_tm.end()) continue;
        if (sum == nullptr) {
          sum = new Matrix2D<double>(region_size, region_size);
          sum->copy_from(*it->second);
        } else {
          for (int i = 0; i < region_size; i++)
            for (int j = 0; j < region_size; j++)
              sum->add_elem_by(i, j, it->second->get_elem(i, j));
        }
        delete it->second;
        layer_tm.erase(it);
      }
      if (sum == nullptr) continue;
      auto bkey = std::make_tuple(pass, block, rid);
      if (block_tm.count(bkey)) { delete block_tm[bkey]; }
      block_tm[bkey] = sum;

      // Promote to last_block_tm for cross-pass prediction lookups.
      auto lb_key = std::make_pair(block, rid);
      auto lb_it = last_block_tm_pass.find(lb_key);
      if (lb_it == last_block_tm_pass.end() || lb_it->second < pass) {
        if (last_block_tm.count(lb_key)) delete last_block_tm[lb_key];
        Matrix2D<double>* copy = new Matrix2D<double>(region_size, region_size);
        copy->copy_from(*sum);
        last_block_tm[lb_key] = copy;
        last_block_tm_pass[lb_key] = pass;
      }
    }
  }

  // Prediction matrix for (pass, block, region): the most recent completed
  // traffic matrix we have for this block, from any earlier pass.
  // Returns nullptr if we have no history yet (pass 0 block 0 typically).
  Matrix2D<double>* get_prediction_matrix(int pass, int block, int region_id) {
    // Prefer immediate predecessor pass (P-1, block, region)
    for (int p = pass - 1; p >= 0; p--) {
      auto bkey = std::make_tuple(p, block, region_id);
      auto it = block_tm.find(bkey);
      if (it != block_tm.end()) return it->second;
    }
    // Fallback: last_block_tm if present
    auto lb_key = std::make_pair(block, region_id);
    auto it = last_block_tm.find(lb_key);
    if (it != last_block_tm.end()) return it->second;
    return nullptr;
  }

  // Skip-reconfig check: if the top-N hottest pairs from the prediction
  // matrix are already present in the current OCS conn, no reconfig needed.
  bool should_skip_reconfig(int pass, int block, int region_size) {
    if (g_reconf_top_n <= 0 || g_mixnet_topo == nullptr || g_topomanager == nullptr) return false;
    if (pass <= 0) return false;  // pass 0 has no prediction, must reconfig (cold)

    int num_regions = (int)g_topomanager->regional_topo_managers.size();
    bool decision_skip = true;
    int deciding_region = -1;
    int missing_i = -1, missing_j = -1;

    for (int rid = 0; rid < num_regions; rid++) {
      Matrix2D<double>* pred = get_prediction_matrix(pass, block, rid);
      if (pred == nullptr) continue;

      int rs = region_size;
      int start_node = rid * rs;

      std::vector<std::pair<double, std::pair<int,int>>> pairs;
      for (int i = 0; i < rs; i++) {
        for (int j = 0; j < rs; j++) {
          if (i == j) continue;
          double val = pred->get_elem(i, j);
          if (val > 0) pairs.push_back({val, {i, j}});
        }
      }
      std::sort(pairs.begin(), pairs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

      if (decision_skip) {
        int check_count = std::min(g_reconf_top_n, (int)pairs.size());
        for (int k = 0; k < check_count; k++) {
          int i = pairs[k].second.first;
          int j = pairs[k].second.second;
          if (g_mixnet_topo->conn[i + start_node][j + start_node] <= 0) {
            decision_skip = false;
            deciding_region = rid;
            missing_i = i; missing_j = j;
            break;
          }
        }
      }
    }

    if (decision_skip) {
      std::cout << "[RECONF_CHECK] pass=" << pass << " block=" << block
                << " decision=SKIP (all top-N already connected)" << std::endl;
    } else {
      std::cout << "[RECONF_CHECK] pass=" << pass << " block=" << block
                << " decision=RECONFIG (region=" << deciding_region
                << " missing " << missing_i << "->" << missing_j << ")" << std::endl;
    }
    return decision_skip;
  }

  // Submit current prediction to demand_recorder and schedule the regional
  // reconfig events. Returns the latest scheduled reconfig_end_time so the
  // triggering flow can be deferred until reconfig completes.
  simtime_picosec trigger_proactive_reconfig(int pass, int block, int layer_num) {
    if (g_demand_recorder == nullptr || g_topomanager == nullptr) return 0;

    int num_regions = (int)g_topomanager->regional_topo_managers.size();
    int slot = block % g_demand_recorder->layer_num;
    simtime_picosec max_end = 0;

    for (int rid = 0; rid < num_regions; rid++) {
      Matrix2D<double>* pred = get_prediction_matrix(pass, block, rid);
      if (pred == nullptr) continue;

      g_demand_recorder->append_traffic_matrix(slot, rid, *pred);

      RegionalTopoManager* rtm = g_topomanager->regional_topo_managers[rid];
      rtm->current_layer_id = slot;
      rtm->reconfig_end_time = g_eventlist->now() + rtm->reconf_delay + 1;
      g_eventlist->sourceIsPending(*rtm, g_eventlist->now());
      if (rtm->reconfig_end_time > max_end) max_end = rtm->reconfig_end_time;
    }
    return max_end;
  }

  // Main entry — called once per cross-machine a2a flow start in SendFlow().
  //   pass_counter: flow's workload pass index (>=0)
  //   loop_state:   Workload::LoopState int (0=Forward, 1=WeightGrad, 2=InputGrad)
  ReconfigResult on_a2a_flow_start(int pass_counter, int loop_state,
                                    int layer_num, int src_machine, int dst_machine,
                                    uint64_t flow_size, int region_size) {
    ReconfigResult result = {false, 0};
    if (src_machine == dst_machine) return result;
    if (pass_counter < 0) return result;  // unstamped; can't attribute

    // Structure discovery: record every a2a layer we see during pass 0.
    // We don't gate on loop_state — Forward_In_BackPass recompute a2as share
    // layer_num with their fwd counterparts (already recorded), and any other
    // phase in pass 0 that does a2a still belongs to the same block structure.
    // Freeze is triggered by on_rank_pass_end once all ranks finish pass 0.
    if (!structure_frozen && pass_counter == 0) {
      record_fwd_layer(layer_num);
    }
    (void)loop_state;  // currently unused post-redesign

    // Always accumulate traffic, keyed by (pass, layer, region). Close of
    // older-pass blocks is handled by on_rank_pass_end, not by per-flow race.
    int region_id = src_machine / region_size;
    int local_src = src_machine % region_size;
    int local_dst = dst_machine % region_size;
    auto lkey = std::make_tuple(pass_counter, layer_num, region_id);
    auto it = layer_tm.find(lkey);
    if (it == layer_tm.end()) {
      layer_tm[lkey] = new Matrix2D<double>(region_size, region_size);
      it = layer_tm.find(lkey);
    }
    it->second->add_elem_by(local_src, local_dst, (double)flow_size);
    std::cout << "[ACC] pass=" << pass_counter
              << " layer=" << layer_num
              << " region=" << region_id
              << " src_m=" << src_machine << " (local " << local_src << ")"
              << " dst_m=" << dst_machine << " (local " << local_dst << ")"
              << " size=" << flow_size << std::endl;

    // Reconfig decision: only at the dispatch layer (first-in-pair), once per
    // (pass, layer). We gate on structure_frozen so the cold first pass
    // doesn't attempt to reconfig with no history.
    if (!structure_frozen) return result;
    if (!layer_to_block.count(layer_num)) return result;
    if (layer_pair_pos[layer_num] != 0) return result;  // combine layer — no decision here

    auto reconf_key = std::make_pair(pass_counter, layer_num);
    if (reconf_done.count(reconf_key)) return result;
    reconf_done.insert(reconf_key);

    int block_idx = layer_to_block[layer_num];

    // Check if we have any prediction at all.
    int num_regions = (g_topomanager != nullptr) ?
        (int)g_topomanager->regional_topo_managers.size() : 0;
    bool has_pred = false;
    for (int rid = 0; rid < num_regions; rid++) {
      if (get_prediction_matrix(pass_counter, block_idx, rid) != nullptr) {
        has_pred = true;
        break;
      }
    }
    if (!has_pred) return result;  // nothing to base reconfig on, keep current conn

    if (should_skip_reconfig(pass_counter, block_idx, region_size)) {
      reconfig_skipped++;
      int slot = block_idx % g_demand_recorder->layer_num;
      for (int rid = 0; rid < num_regions; rid++) {
        Matrix2D<double>* pred = get_prediction_matrix(pass_counter, block_idx, rid);
        if (pred != nullptr) g_demand_recorder->append_traffic_matrix(slot, rid, *pred);
      }
      return result;
    }

    reconfig_triggered++;
    simtime_picosec max_end = trigger_proactive_reconfig(pass_counter, block_idx, layer_num);
    result.should_defer = true;
    result.reconfig_end_time = max_end;
    static int sync_log = 0;
    if (sync_log < 20) {
      std::cout << "[RECONFIG_SYNC] pass=" << pass_counter
                << " block=" << block_idx << " layer=" << layer_num
                << " delay=" << ((max_end - g_eventlist->now()) / 1000) << "ns" << std::endl;
      sync_log++;
    }
    return result;
  }
};

static MoEReconfigManager g_moe_reconfig_mgr;

// ======== Dead-route self-heal: invoked from TcpSrc::rtx_timer_hook on every RTO ========
// Checks whether this flow's Route crosses any queue with _bitrate==0 (i.e. its
// OCS circuit was torn down by a subsequent reconfig). If so, rebuild an ECS
// Route (copy topology path + append Sink/Src endpoints) and swap via reroute_to.
// Returns true if we rerouted. Registered as TcpSrc::on_rtx_stuck.
inline bool reroute_flow_if_dead(TcpSrc* tcp) {
  if (tcp == nullptr || tcp->_finished) return false;
  if (tcp->is_elec) return false;                // already on ECS
  if (g_mixnet_topo == nullptr) return false;
  if (tcp->_route == nullptr) return false;

  // Detect dead queue on current route
  bool dead = false;
  for (Route::const_iterator it = tcp->_route->begin(); it != tcp->_route->end(); ++it) {
    Queue *q = dynamic_cast<Queue*>(*it);
    if (q != nullptr && q->_bitrate == 0) { dead = true; break; }
  }
  if (!dead) return false;

  std::vector<const Route*>* fwd = g_mixnet_topo->get_eps_paths(tcp->_flow_src, tcp->_flow_dst);
  std::vector<const Route*>* bck = g_mixnet_topo->get_eps_paths(tcp->_flow_dst, tcp->_flow_src);
  if (fwd == nullptr || fwd->empty() || bck == nullptr || bck->empty()) return false;

  Route *newfwd  = new Route(*(fwd->at(rand() % fwd->size())));
  Route *newback = new Route(*(bck->at(rand() % bck->size())));
  newfwd->push_back(tcp->_sink);
  newback->push_back(tcp);
  newfwd->set_reverse(newback);
  newback->set_reverse(newfwd);

  tcp->reroute_to(newfwd, newback);
  tcp->is_elec = true;
  std::cout << "[REROUTE_ECS_RTX] src=" << tcp->_flow_src << " dst=" << tcp->_flow_dst
            << " dead OCS -> ECS (triggered by RTO)" << std::endl;
  return true;
}

// ======== SendFlow: core function replacing NS-3 RDMA with htsim TCP ========
// Global counters for OCS/ECS tracking
static uint64_t g_flow_count_ocs = 0;
static uint64_t g_flow_count_ecs = 0;
static uint64_t g_flow_count_nvlink = 0;
static uint64_t g_flow_bytes_ocs = 0;
static uint64_t g_flow_bytes_ecs = 0;
static uint64_t g_flow_bytes_nvlink = 0;
static uint64_t g_flow_count_deferred = 0;

void SendFlow(int src, int dst, uint64_t maxPacketCount,
              void (*msg_handler)(void *fun_arg), void *fun_arg,
              int tag, AstraSim::sim_request *request) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();

  if (maxPacketCount == 0) maxPacketCount = 1;

  // Track port number (same as ns3)
  uint32_t port = portNumber[src][dst]++;
  sender_src_port_map[make_pair(port, make_pair(src, dst))] = request->flowTag;

  int flow_id = request->flowTag.current_flow_id;
  int com_type = request->flowTag.com_type;

  // Per-ComType byte/flow breakdown (covers NVLink, OCS, ECS)
  {
    int idx = (com_type >= 0 && com_type < G_COMTYPE_N) ? com_type : (G_COMTYPE_N - 1);
    g_bytes_by_comtype[idx] += maxPacketCount;
    g_flows_by_comtype[idx] += 1;
  }

  // Determine machine IDs
  int src_machine = src / g_gpus_per_server;
  int dst_machine = dst / g_gpus_per_server;

  // ---- NVLink: same-machine communication ----
  if (src_machine == dst_machine) {
    // Model NVLink as fixed-bandwidth transfer
    double nvlink_bw_bps = 900e9;  // 900 Gbps NVLink (H100)
    simtime_picosec transfer_time = (simtime_picosec)(maxPacketCount * 8.0 / nvlink_bw_bps * 1e12);
    if (transfer_time < 1000) transfer_time = 1000;  // minimum 1ns

    g_flow_count_nvlink++;
    g_flow_bytes_nvlink += maxPacketCount;
    if (g_flow_count_nvlink <= 5) {
      cout << "[SendFlow] NVLink: src=" << src << " dst=" << dst
           << " size=" << maxPacketCount << " com_type=" << com_type << endl;
    }

    HtsimFlowContext* ctx = new HtsimFlowContext{src, dst, maxPacketCount, request->flowTag};
    CallbackEvent* ev = new CallbackEvent(*g_eventlist, htsim_flow_finish, (void*)ctx);
    g_eventlist->sourceIsPendingRel(*ev, transfer_time);

    waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    return;
  }

  // ---- Proactive reconfig check (BEFORE routing decision, a2a only) ----
  // Skip on deferred replay: the flow was already accumulated and decided on
  // its first entry through SendFlow; we must not re-enter the manager.
  if (g_topo_type == TOPO_MIXNET && com_type == 4 && g_mixnet_topo != nullptr &&
      !g_force_ecs_only && !g_replaying_deferred_flow) {
    int layer_num = request->flowTag.layer_num;
    int pass_counter = request->flowTag.pass_counter;
    int loop_state = request->flowTag.loop_state;
    auto reconf_result = g_moe_reconfig_mgr.on_a2a_flow_start(
        pass_counter, loop_state, layer_num,
        src_machine, dst_machine, maxPacketCount, g_mixnet_topo->region_size);

    if (reconf_result.should_defer) {
      // Proactive reconfig triggered — defer this flow
      g_flow_count_deferred++;
      DeferredSendData dsd;
      dsd.src = src; dsd.dst = dst;
      dsd.count = maxPacketCount;
      dsd.msg_handler = msg_handler;
      dsd.fun_arg = fun_arg;
      dsd.tag = tag;
      dsd.request = *request;
      DeferredSendEvent* ev = new DeferredSendEvent(*g_eventlist, dsd);
      g_eventlist->sourceIsPending(*ev, reconf_result.reconfig_end_time);
      return;
    }
  }

  // ---- Ongoing reconfig defer check (all a2a flows, not just OCS) ----
  if (g_topo_type == TOPO_MIXNET && com_type == 4 && g_topomanager != nullptr && g_mixnet_topo != nullptr) {
    int region_id = src_machine / g_mixnet_topo->region_size;
    if (region_id < (int)g_topomanager->regional_topo_managers.size()) {
      RegionalTopoManager* rtm = g_topomanager->regional_topo_managers[region_id];
      if (rtm->reconfig_end_time > 0 && g_eventlist->now() < rtm->reconfig_end_time) {
        g_flow_count_deferred++;
        DeferredSendData dsd;
        dsd.src = src; dsd.dst = dst;
        dsd.count = maxPacketCount;
        dsd.msg_handler = msg_handler;
        dsd.fun_arg = fun_arg;
        dsd.tag = tag;
        dsd.request = *request;
        DeferredSendEvent* ev = new DeferredSendEvent(*g_eventlist, dsd);
        g_eventlist->sourceIsPending(*ev, rtm->reconfig_end_time);
        return;
      }
    }
  }

  // ---- Routing decision ----
  bool use_ocs = false;

  if (g_topo_type == TOPO_MIXNET) {
    // Mixnet mode: OCS/ECS selection logic
    static int ocs_dbg_count = 0;
    if (com_type == 4 && g_mixnet_topo != nullptr) {
      if (src_machine < (int)g_mixnet_topo->conn.size() &&
          dst_machine < (int)g_mixnet_topo->conn[src_machine].size()) {
        int cv = g_mixnet_topo->conn[src_machine][dst_machine];
        if (ocs_dbg_count < 10) {
          cerr << "[OCS_DBG] src=" << src << " m" << src_machine << " dst=" << dst << " m" << dst_machine
               << " conn=" << cv << endl;
          ocs_dbg_count++;
        }
        if (cv > 0) use_ocs = true;
      }
    }
    if (g_force_ecs_only) use_ocs = false;
  }

  // Log flow decision
  if (use_ocs) {
    g_flow_count_ocs++;
    g_flow_bytes_ocs += maxPacketCount;
    if (g_flow_count_ocs <= 10) {
      cout << "[SendFlow] OCS: src=" << src << "(m" << src_machine << ") dst=" << dst
           << "(m" << dst_machine << ") size=" << maxPacketCount
           << " com_type=" << com_type << endl;
    }
  } else {
    g_flow_count_ecs++;
    g_flow_bytes_ecs += maxPacketCount;
    if (g_flow_count_ecs <= 10) {
      cout << "[SendFlow] ECS: src=" << src << "(m" << src_machine << ") dst=" << dst
           << "(m" << dst_machine << ") size=" << maxPacketCount
           << " com_type=" << com_type << endl;
    }
  }

  // ---- Create htsim TCP flow ----
  g_total_tcp_flows_created++;
  HtsimFlowContext* ctx = new HtsimFlowContext{src, dst, maxPacketCount, request->flowTag};

  DCTCPSrc* flowSrc = new DCTCPSrc(NULL, NULL, &g_fct_output,
                                    *g_eventlist, src, dst,
                                    htsim_flow_finish, (void*)ctx);
  TcpSink* flowSnk = new TcpSink();

  flowSrc->set_flowsize(maxPacketCount);
  flowSrc->set_ssthresh(65535 * Packet::data_packet_size());
  flowSrc->_rto = timeFromMs(g_rto_ms);
  flowSrc->is_elec = !use_ocs;
  flowSrc->is_all2all = (com_type == 4);

  g_tcp_scanner->registerTcp(*flowSrc);

  // ---- Get routes ----
  vector<const Route*>* srcpaths = nullptr;
  vector<const Route*>* dstpaths = nullptr;

  static int get_path_count = 0;

  if (g_topo_type == TOPO_MIXNET) {
    // Mixnet: OCS uses get_paths(), ECS uses get_eps_paths()
    if (use_ocs) {
      int conn_val = g_mixnet_topo->conn[src_machine][dst_machine];
      if (conn_val > 0) {
        srcpaths = g_mixnet_topo->get_paths(src, dst);
        int conn_rev = g_mixnet_topo->conn[dst_machine][src_machine];
        if (conn_rev > 0) {
          dstpaths = g_mixnet_topo->get_paths(dst, src);
        } else {
          use_ocs = false;
          srcpaths = g_mixnet_topo->get_eps_paths(src, dst);
          dstpaths = g_mixnet_topo->get_eps_paths(dst, src);
          flowSrc->is_elec = true;
        }
      } else {
        use_ocs = false;
        srcpaths = g_mixnet_topo->get_eps_paths(src, dst);
        dstpaths = g_mixnet_topo->get_eps_paths(dst, src);
        flowSrc->is_elec = true;
      }
    } else {
      srcpaths = g_mixnet_topo->get_eps_paths(src, dst);
      dstpaths = g_mixnet_topo->get_eps_paths(dst, src);
    }
    // Fallback if paths still empty
    if (srcpaths == nullptr || srcpaths->empty() || dstpaths == nullptr || dstpaths->empty()) {
      if (use_ocs) {
        srcpaths = g_mixnet_topo->get_eps_paths(src, dst);
        dstpaths = g_mixnet_topo->get_eps_paths(dst, src);
        use_ocs = false;
      }
    }
  } else {
    // Generic topology: use get_paths() with machine-level addressing
    srcpaths = g_topology->get_paths(src_machine, dst_machine);
    dstpaths = g_topology->get_paths(dst_machine, src_machine);
  }

  if (get_path_count < 5) {
    cerr << "[SendFlow] src=" << src << " dst=" << dst
         << " src_m=" << src_machine << " dst_m=" << dst_machine
         << " ocs=" << use_ocs
         << " srcpaths=" << (srcpaths ? (int)srcpaths->size() : -1)
         << " dstpaths=" << (dstpaths ? (int)dstpaths->size() : -1) << endl;
  }
  get_path_count++;

  if (srcpaths == nullptr || srcpaths->empty() || dstpaths == nullptr || dstpaths->empty()) {
    NcclLog->writeLog(NcclLogLevel::ERROR,
        "[htsim] No path found for src %d dst %d (m%d->m%d)", src, dst, src_machine, dst_machine);
    cout << "[htsim] FATAL: No path for src " << src << " dst " << dst
         << " (m" << src_machine << "->m" << dst_machine << ")" << endl;
    delete flowSrc;
    delete flowSnk;
    delete ctx;
    return;
  }

  int choice = rand() % srcpaths->size();
  Route* routeout = new Route(*(srcpaths->at(choice)));
  routeout->push_back(flowSnk);

  choice = rand() % dstpaths->size();
  Route* routein = new Route(*(dstpaths->at(choice)));
  routein->push_back(flowSrc);

  // Debug: print route lengths for first few flows
  static int route_debug_count = 0;
  if (route_debug_count < 5) {
    cout << "[SendFlow] Route debug: src=" << src << " dst=" << dst
         << " ocs=" << use_ocs
         << " routeout_size=" << routeout->size()
         << " routein_size=" << routein->size() << endl;
    route_debug_count++;
  }

  // Connect and start flow
  flowSrc->connect(*routeout, *routein, *flowSnk, g_eventlist->now());

  // Track completion
  waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))]++;
  waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]++;
}

// Deferred send implementation
void DeferredSendEvent::doNextEvent() {
  // Mark as replay so SendFlow skips the MoEReconfigManager accumulator/decision
  // path (already done on the initial SendFlow call when this flow was deferred).
  bool prev = g_replaying_deferred_flow;
  g_replaying_deferred_flow = true;
  SendFlow(_data.src, _data.dst, _data.count,
           _data.msg_handler, _data.fun_arg, _data.tag, &_data.request);
  g_replaying_deferred_flow = prev;
  delete this;
}

#endif // __ENTRY_HTSIM_H__
