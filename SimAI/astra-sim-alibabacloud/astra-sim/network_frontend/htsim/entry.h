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

// Reconfig skip: skip OCS reconfig if top-N traffic pairs are already connected
int g_reconf_top_n = 0;      // 0=always reconfig, >0=skip if top-N pairs covered

// Global retransmission counters (accumulated as flows finish, not at end)
uint64_t g_total_packets_sent = 0;
uint64_t g_total_retransmissions = 0;
uint64_t g_total_tcp_flows = 0;       // finished TCP flows
uint64_t g_total_tcp_flows_created = 0; // total TCP flows created

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

struct MoEReconfigManager {
  // MoE block structure (discovered during first forward pass)
  std::vector<MoEBlockInfo> moe_blocks;
  bool structure_discovered = false;

  // Reconfig statistics
  uint64_t reconfig_triggered = 0;
  uint64_t reconfig_skipped = 0;
  uint64_t reconfig_proactive = 0;   // proactive reconfigs triggered at second-in-pair
  uint64_t reconfig_hidden = 0;      // times reconfig was fully hidden (0 delay at dispatch)

  // Pending proactive reconfig (triggered at previous block's last a2a)
  int pending_reconfig_for_block = -1;
  simtime_picosec pending_reconfig_end_time = 0;

  // Direction tracking
  int last_a2a_layer_num = -1;
  bool is_forward = true;
  int a2a_count_in_direction = 0;  // 1-based count of a2a layers in current direction

  // Iteration tracking
  int current_iteration = 0;

  // Traffic matrices: key = (block_idx, region_id)
  std::map<std::pair<int,int>, Matrix2D<double>*> real_tm;     // recorded during fwd phase
  std::map<std::pair<int,int>, Matrix2D<double>*> history_tm;  // real_tm from previous iteration

  // Current layer demand accumulation: key = (layer_num, region_id)
  std::map<std::pair<int,int>, Matrix2D<double>*> current_layer_tm;

  // Prevent duplicate triggers
  std::set<int> reconf_triggered_layers;

  struct ReconfigResult {
    bool should_defer;
    simtime_picosec reconfig_end_time;
  };

  // Detect direction change and handle iteration boundary
  // ONLY call this for NEW layer_nums (not repeated flows of same layer)
  void detect_direction(int layer_num, int region_size) {
    if (last_a2a_layer_num < 0) {
      is_forward = true;
      return;
    }
    if (layer_num == last_a2a_layer_num) return;  // Same layer, no direction change
    bool new_forward = (layer_num > last_a2a_layer_num);
    if (new_forward != is_forward) {
      // Direction changed — finalize last layer's TM BEFORE resetting counter
      finalize_prev_layer(region_size);
      if (new_forward && !is_forward) {
        // bwd → fwd: new iteration
        rotate_iteration();
      }
      is_forward = new_forward;
      a2a_count_in_direction = 0;
      reconf_triggered_layers.clear();
      pending_reconfig_for_block = -1;
      pending_reconfig_end_time = 0;
    }
  }

  // Rotate iteration: real_tm → history_tm
  void rotate_iteration() {
    for (auto& [key, tm] : history_tm) {
      if (tm) delete tm;
    }
    history_tm.clear();
    // Move real_tm to history_tm
    for (auto& [key, tm] : real_tm) {
      history_tm[key] = tm;  // transfer ownership
    }
    real_tm.clear();
    // Clear per-layer accumulation
    for (auto& [key, tm] : current_layer_tm) {
      if (tm) delete tm;
    }
    current_layer_tm.clear();
    reconf_triggered_layers.clear();
    pending_reconfig_for_block = -1;
    pending_reconfig_end_time = 0;
    current_iteration++;
  }

  // Finalize previous layer's TM into block's real_tm
  void finalize_prev_layer(int region_size) {
    if (last_a2a_layer_num < 0 || a2a_count_in_direction <= 0) return;

    int prev_pair_pos = a2a_count_in_direction;  // position of the layer being finalized
    int block_idx_in_dir = (prev_pair_pos - 1) / 2;

    // For backward pass, reverse the block index
    int actual_block_idx = block_idx_in_dir;
    if (!is_forward && structure_discovered) {
      actual_block_idx = (int)moe_blocks.size() - 1 - block_idx_in_dir;
    }
    if (actual_block_idx < 0) return;

    int region_num = (g_topomanager != nullptr) ? (int)g_topomanager->regional_topo_managers.size() : 1;
    for (int rid = 0; rid < region_num; rid++) {
      auto layer_key = std::make_pair(last_a2a_layer_num, rid);
      if (current_layer_tm.find(layer_key) == current_layer_tm.end()) continue;

      Matrix2D<double>* tm = current_layer_tm[layer_key];
      auto block_key = std::make_pair(actual_block_idx, rid);

      if (real_tm.find(block_key) == real_tm.end()) {
        // First layer of this block: copy
        real_tm[block_key] = new Matrix2D<double>(region_size, region_size);
        real_tm[block_key]->copy_from(*tm);
      } else {
        // Second layer (combine): accumulate into existing real_tm
        for (int i = 0; i < region_size; i++)
          for (int j = 0; j < region_size; j++)
            real_tm[block_key]->add_elem_by(i, j, tm->get_elem(i, j));
      }
    }
  }

  // Get prediction matrix for a given block and region
  // Returns nullptr if no matrix available (use default topo)
  Matrix2D<double>* get_prediction_matrix(int block_idx, int region_id) {
    if (is_forward) {
      if (current_iteration == 0 && block_idx == 0) {
        return nullptr;  // Block 0, first iteration: default topo
      }
      if (current_iteration == 0) {
        // First iteration, block K>0: use previous block's real_tm
        auto prev_key = std::make_pair(block_idx - 1, region_id);
        if (real_tm.find(prev_key) != real_tm.end()) return real_tm[prev_key];
        return nullptr;
      }
      // Iteration 1+: use own history
      auto key = std::make_pair(block_idx, region_id);
      if (history_tm.find(key) != history_tm.end()) return history_tm[key];
      return nullptr;
    } else {
      // Backward: use real_tm from forward phase
      // block_idx_in_dir → actual block (reversed)
      int actual_block = (structure_discovered) ?
          (int)moe_blocks.size() - 1 - block_idx : block_idx;
      auto key = std::make_pair(actual_block, region_id);
      if (real_tm.find(key) != real_tm.end()) return real_tm[key];
      return nullptr;
    }
  }

  // Check if reconfig can be skipped: top-N traffic pairs already have OCS connections
  bool should_skip_reconfig(int block_idx, int region_size) {
    if (g_reconf_top_n <= 0 || g_mixnet_topo == nullptr || g_topomanager == nullptr) return false;

    // Never skip on first iteration — let greedy algorithm optimize initial random conn
    if (current_iteration == 0) return false;

    int num_regions = (int)g_topomanager->regional_topo_managers.size();
    for (int rid = 0; rid < num_regions; rid++) {
      Matrix2D<double>* pred = get_prediction_matrix(block_idx, rid);
      if (pred == nullptr) continue;

      int rs = region_size;
      int start_node = rid * rs;

      // Collect all (i,j) pairs with their traffic values
      std::vector<std::pair<double, std::pair<int,int>>> pairs;
      for (int i = 0; i < rs; i++) {
        for (int j = 0; j < rs; j++) {
          if (i == j) continue;
          double val = pred->get_elem(i, j);
          if (val > 0) {
            pairs.push_back({val, {i, j}});
          }
        }
      }

      // Sort descending by traffic volume
      std::sort(pairs.begin(), pairs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

      // Check if top-N pairs all have OCS connections
      int check_count = std::min(g_reconf_top_n, (int)pairs.size());
      for (int k = 0; k < check_count; k++) {
        int i = pairs[k].second.first;
        int j = pairs[k].second.second;
        if (g_mixnet_topo->conn[i + start_node][j + start_node] <= 0) {
          return false;  // At least one top-N pair not connected → need reconfig
        }
      }
    }
    return true;  // All top-N pairs already connected in all regions → skip
  }

  // Trigger proactive reconfig for all regions
  void trigger_proactive_reconfig(int block_idx, int layer_num) {
    if (g_demand_recorder == nullptr || g_topomanager == nullptr) return;

    int num_regions = (int)g_topomanager->regional_topo_managers.size();
    int slot = block_idx % g_demand_recorder->layer_num;

    for (int rid = 0; rid < num_regions; rid++) {
      Matrix2D<double>* pred = get_prediction_matrix(block_idx, rid);
      if (pred == nullptr) continue;

      g_demand_recorder->append_traffic_matrix(slot, rid, *pred);

      RegionalTopoManager* rtm = g_topomanager->regional_topo_managers[rid];
      rtm->current_layer_id = slot;
      // Pre-set reconfig_end_time so deferred check works immediately
      rtm->reconfig_end_time = g_eventlist->now() + rtm->reconf_delay + 1;
      // Schedule the actual reconfig event
      g_eventlist->sourceIsPending(*rtm, g_eventlist->now());
    }
  }

  // Main entry point: called from SendFlow() for each a2a flow
  ReconfigResult on_a2a_flow_start(int layer_num, int src_machine, int dst_machine,
                                    uint64_t flow_size, int region_size) {
    ReconfigResult result = {false, 0};

    // Only process cross-machine flows
    if (src_machine == dst_machine) return result;

    // Detect direction change
    detect_direction(layer_num, region_size);

    // New layer transition?
    bool is_new_layer = (layer_num != last_a2a_layer_num);
    if (is_new_layer) {
      // Finalize previous layer's TM
      finalize_prev_layer(region_size);

      // Increment counter
      a2a_count_in_direction++;
      last_a2a_layer_num = layer_num;

      // MoE block structure discovery (during first forward pass)
      if (is_forward && !structure_discovered) {
        int pair_pos = a2a_count_in_direction;
        if (pair_pos % 2 == 1) {
          // 1st in pair: dispatch
          MoEBlockInfo info;
          info.dispatch_layer = layer_num;
          info.combine_layer = -1;
          moe_blocks.push_back(info);
        } else {
          // 2nd in pair: combine
          int block_idx = (pair_pos - 1) / 2;
          if (block_idx < (int)moe_blocks.size()) {
            moe_blocks[block_idx].combine_layer = layer_num;
          }
        }
      }
      // Mark structure as discovered when backward starts
      if (!is_forward && !structure_discovered && !moe_blocks.empty()) {
        structure_discovered = true;
      }

      // Determine pair position and block index
      int pair_pos = a2a_count_in_direction;
      bool is_first_in_pair = (pair_pos % 2 == 1);
      int block_idx_in_dir = (pair_pos - 1) / 2;

      // ---- FIRST-IN-PAIR: block's first a2a (fwd dispatch / bwd combine) ----
      if (is_first_in_pair && !reconf_triggered_layers.count(layer_num)) {
        reconf_triggered_layers.insert(layer_num);

        if (pending_reconfig_for_block == block_idx_in_dir) {
          // Proactive reconfig was triggered at previous block's last a2a
          if (g_eventlist->now() >= pending_reconfig_end_time) {
            // Already done — flows proceed immediately with zero delay
            reconfig_hidden++;
            static int hidden_log = 0;
            if (hidden_log < 20) {
              cout << "[RECONFIG_HIDDEN] layer=" << layer_num
                   << " block=" << block_idx_in_dir
                   << " reconfig completed before dispatch, 0 delay" << endl;
              hidden_log++;
            }
          } else {
            // Still in progress — defer for remaining time only
            result.should_defer = true;
            result.reconfig_end_time = pending_reconfig_end_time;
            static int partial_log = 0;
            if (partial_log < 20) {
              simtime_picosec remaining = pending_reconfig_end_time - g_eventlist->now();
              cout << "[RECONFIG_PARTIAL] layer=" << layer_num
                   << " block=" << block_idx_in_dir
                   << " remaining=" << (remaining / 1000) << "ns" << endl;
              partial_log++;
            }
          }
          pending_reconfig_for_block = -1;
          pending_reconfig_end_time = 0;
        } else {
          // No pending reconfig — cold start (iter0 block0 or edge case)
          // Fall back to immediate reconfig + full defer
          bool has_matrix = false;
          int num_regions = (g_topomanager != nullptr) ?
              (int)g_topomanager->regional_topo_managers.size() : 0;
          for (int rid = 0; rid < num_regions; rid++) {
            if (get_prediction_matrix(block_idx_in_dir, rid) != nullptr) {
              has_matrix = true;
              break;
            }
          }
          if (has_matrix) {
            if (should_skip_reconfig(block_idx_in_dir, region_size)) {
              reconfig_skipped++;
              int slot = block_idx_in_dir % g_demand_recorder->layer_num;
              for (int rid = 0; rid < num_regions; rid++) {
                Matrix2D<double>* pred = get_prediction_matrix(block_idx_in_dir, rid);
                if (pred != nullptr) {
                  g_demand_recorder->append_traffic_matrix(slot, rid, *pred);
                }
              }
            } else {
              reconfig_triggered++;
              trigger_proactive_reconfig(block_idx_in_dir, layer_num);
              simtime_picosec max_end = 0;
              for (int rid = 0; rid < num_regions; rid++) {
                RegionalTopoManager* rtm = g_topomanager->regional_topo_managers[rid];
                if (rtm->reconfig_end_time > max_end) max_end = rtm->reconfig_end_time;
              }
              result.should_defer = true;
              result.reconfig_end_time = max_end;
            }
          }
        }
      }

      // ---- SECOND-IN-PAIR: block's last a2a (fwd combine / bwd dispatch) ----
      // Proactively trigger reconfig for the NEXT block, overlapping with current a2a + computation
      if (!is_first_in_pair && !reconf_triggered_layers.count(layer_num)) {
        reconf_triggered_layers.insert(layer_num);

        int next_block_idx = block_idx_in_dir + 1;

        // Check if next block has a prediction matrix
        bool has_next_matrix = false;
        int num_regions = (g_topomanager != nullptr) ?
            (int)g_topomanager->regional_topo_managers.size() : 0;
        for (int rid = 0; rid < num_regions; rid++) {
          if (get_prediction_matrix(next_block_idx, rid) != nullptr) {
            has_next_matrix = true;
            break;
          }
        }

        if (has_next_matrix) {
          if (should_skip_reconfig(next_block_idx, region_size)) {
            reconfig_skipped++;
            int slot = next_block_idx % g_demand_recorder->layer_num;
            for (int rid = 0; rid < num_regions; rid++) {
              Matrix2D<double>* pred = get_prediction_matrix(next_block_idx, rid);
              if (pred != nullptr) {
                g_demand_recorder->append_traffic_matrix(slot, rid, *pred);
              }
            }
          } else {
            // Trigger proactive reconfig for next block — do NOT defer current flows
            reconfig_proactive++;
            // Set proactive mode so TCP flows won't be paused
            for (int rid = 0; rid < num_regions; rid++) {
              RegionalTopoManager* rtm = g_topomanager->regional_topo_managers[rid];
              rtm->proactive_mode = true;
            }
            trigger_proactive_reconfig(next_block_idx, layer_num);
            simtime_picosec max_end = 0;
            for (int rid = 0; rid < num_regions; rid++) {
              RegionalTopoManager* rtm = g_topomanager->regional_topo_managers[rid];
              if (rtm->reconfig_end_time > max_end) max_end = rtm->reconfig_end_time;
              rtm->proactive_mode = false;
            }
            pending_reconfig_for_block = next_block_idx;
            pending_reconfig_end_time = max_end;
            static int proactive_log = 0;
            if (proactive_log < 20) {
              cout << "[RECONFIG_PROACTIVE] at layer=" << layer_num
                   << " block=" << block_idx_in_dir
                   << " for next_block=" << next_block_idx << endl;
              proactive_log++;
            }
            // result.should_defer remains false — current combine/dispatch flows proceed
          }
        }
      }
    }

    // Always accumulate real demand (even for deferred flows when they re-enter)
    int region_id = src_machine / region_size;
    auto layer_key = std::make_pair(layer_num, region_id);
    if (current_layer_tm.find(layer_key) == current_layer_tm.end()) {
      current_layer_tm[layer_key] = new Matrix2D<double>(region_size, region_size);
    }
    int local_src = src_machine % region_size;
    int local_dst = dst_machine % region_size;
    current_layer_tm[layer_key]->add_elem_by(local_src, local_dst, (double)flow_size);

    return result;
  }
};

static MoEReconfigManager g_moe_reconfig_mgr;

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
  if (g_topo_type == TOPO_MIXNET && com_type == 4 && g_mixnet_topo != nullptr && !g_force_ecs_only) {
    int layer_num = request->flowTag.layer_num;
    auto reconf_result = g_moe_reconfig_mgr.on_a2a_flow_start(
        layer_num, src_machine, dst_machine, maxPacketCount, g_mixnet_topo->region_size);

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
  SendFlow(_data.src, _data.dst, _data.count,
           _data.msg_handler, _data.fun_arg, _data.tag, &_data.request);
  delete this;
}

#endif // __ENTRY_HTSIM_H__
