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

  // Trigger reconfiguration check for AllToAll flows (skip in ECS-only mode)
  if (flowTag.com_type == 4 && g_mixnet_topo != nullptr && !g_force_ecs_only) {
    int src_machine = sid / g_gpus_per_server;
    int dst_machine = did / g_gpus_per_server;
    if (src_machine != dst_machine) {
      // This is a cross-machine AllToAll flow that completed
      // Defer the reconfiguration check to after LayerDemandTracker is defined
      // We use a static function pointer set up after definition
      extern void check_reconf_trigger(int src_machine, int dst_machine, int layer_tag);
      check_reconf_trigger(src_machine, dst_machine, flowTag.tag_id);
    }
  }
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

// ======== Demand recording and reconfiguration trigger ========
// Mirrors ffapp.cpp's FFAlltoAll::updatetrafficmatrix() + FFTask::cleanup() logic
// Tracks per-layer traffic demand matrices and triggers OCS reconfiguration

struct LayerDemandTracker {
  // Per-region traffic matrices keyed by (layer_tag, region_id)
  std::map<std::pair<int,int>, Matrix2D<double>*> region_traffic_matrices;
  // Track cross-machine a2a flows CREATED and COMPLETED per (layer_tag, region_id)
  std::map<std::pair<int,int>, int> a2a_flows_created;
  std::map<std::pair<int,int>, int> a2a_flows_completed;
  // Track which (layer_tag, region_id) already triggered reconfiguration
  std::set<std::pair<int,int>> reconf_triggered;
  // Track which layer_tags we've seen "all created" for
  std::set<std::pair<int,int>> creation_complete;

  // Track the last (layer_tag, region_id) seen, to detect layer transitions
  std::set<std::pair<int,int>> active_layers;

  void record_demand(int src_machine, int dst_machine, uint64_t flow_size,
                     int layer_tag, int region_size) {
    int region_id = src_machine / region_size;
    auto key = std::make_pair(layer_tag, region_id);

    // Detect new layer: if this is a new (layer_tag, region_id),
    // mark all previously active layers as creation-complete
    if (!active_layers.count(key)) {
      for (auto& prev_key : active_layers) {
        if (prev_key.second == region_id && !creation_complete.count(prev_key)) {
          mark_creation_complete(prev_key.first, prev_key.second);
        }
      }
      active_layers.insert(key);
    }

    if (region_traffic_matrices.find(key) == region_traffic_matrices.end()) {
      region_traffic_matrices[key] = new Matrix2D<double>(region_size, region_size);
    }
    int local_src = src_machine % region_size;
    int local_dst = dst_machine % region_size;
    region_traffic_matrices[key]->add_elem_by(local_src, local_dst, (double)flow_size);
    a2a_flows_created[key]++;
  }

  // Called when we know all flows for a layer_tag have been created
  // (detected when a new layer starts or from heuristic)
  void mark_creation_complete(int layer_tag, int region_id) {
    auto key = std::make_pair(layer_tag, region_id);
    creation_complete.insert(key);
    maybe_trigger_reconf(key);
  }

  void flow_completed(int src_machine, int dst_machine, int layer_tag, int region_size) {
    int region_id = src_machine / region_size;
    auto key = std::make_pair(layer_tag, region_id);
    a2a_flows_completed[key]++;
    maybe_trigger_reconf(key);
  }

  void maybe_trigger_reconf(std::pair<int,int> key) {
    // Only trigger when ALL created flows have completed AND we know creation is done
    if (reconf_triggered.count(key)) return;
    if (!creation_complete.count(key)) return;
    if (a2a_flows_completed[key] < a2a_flows_created[key]) return;

    // All flows for this (layer_tag, region_id) are done
    reconf_triggered.insert(key);
    int layer_tag = key.first;
    int region_id = key.second;

    if (g_demand_recorder != nullptr && g_topomanager != nullptr &&
        region_traffic_matrices.count(key)) {
      Matrix2D<double>* tm = region_traffic_matrices[key];

      g_demand_recorder->append_traffic_matrix(layer_tag % g_demand_recorder->layer_num,
                                                region_id, *tm);

      RegionalTopoManager* rtm = g_topomanager->regional_topo_managers[region_id];
      rtm->current_layer_id = layer_tag % g_demand_recorder->layer_num;

      cout << "[RECONF] Triggering reconfiguration for region " << region_id
           << " layer_tag=" << layer_tag
           << " created=" << a2a_flows_created[key]
           << " completed=" << a2a_flows_completed[key]
           << " at time=" << timeAsMs(g_eventlist->now()) << "ms" << endl;

      g_eventlist->sourceIsPending(*rtm, g_eventlist->now());
    }
  }
};

static LayerDemandTracker g_demand_tracker;

// Called from htsim_flow_finish when a cross-machine AllToAll flow completes
void check_reconf_trigger(int src_machine, int dst_machine, int layer_tag) {
  if (g_mixnet_topo == nullptr) return;
  g_demand_tracker.flow_completed(src_machine, dst_machine, layer_tag,
                                   g_mixnet_topo->region_size);
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

  // Record traffic demand for reconfiguration (mixnet only)
  if (g_topo_type == TOPO_MIXNET && com_type == 4 && g_mixnet_topo != nullptr && !g_force_ecs_only) {
    int layer_tag = request->flowTag.tag_id;
    g_demand_tracker.record_demand(src_machine, dst_machine, maxPacketCount,
                                    layer_tag, g_mixnet_topo->region_size);
  }

  // Check if OCS region is under reconfiguration (mixnet only)
  if (use_ocs && g_topomanager != nullptr) {
    int region_id = src_machine / g_mixnet_topo->region_size;
    if (region_id < (int)g_topomanager->regional_topo_managers.size()) {
      RegionalTopoManager* rtm = g_topomanager->regional_topo_managers[region_id];
      if (rtm->status == RegionalTopoManager::TopoStatus::TOPO_RECONF) {
        simtime_picosec resume_time = rtm->reconfig_end_time;
        if (g_eventlist->now() < resume_time) {
          g_flow_count_deferred++;
          if (g_flow_count_deferred <= 5) {
            cout << "[SendFlow] DEFERRED (reconf): src=" << src << " dst=" << dst
                 << " region=" << region_id
                 << " resume_at=" << timeAsMs(resume_time) << "ms" << endl;
          }
          DeferredSendData dsd;
          dsd.src = src; dsd.dst = dst;
          dsd.count = maxPacketCount;
          dsd.msg_handler = msg_handler;
          dsd.fun_arg = fun_arg;
          dsd.tag = tag;
          dsd.request = *request;
          DeferredSendEvent* ev = new DeferredSendEvent(*g_eventlist, dsd);
          g_eventlist->sourceIsPending(*ev, resume_time);
          return;
        }
      }
    }
  }

  // ---- Create htsim TCP flow ----
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
