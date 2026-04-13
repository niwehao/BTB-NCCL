/*
 * htsim network backend for SimAI
 * Replaces NS-3 with htsim's OCS-ECS hybrid topology (Mixnet + FatTree).
 * Implements AstraNetworkAPI interface using htsim EventList and DCTCP flows.
 */

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "entry.h"

#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <getopt.h>
#include <sys/stat.h>
#include <ctime>

#define RESULT_PATH "./ncclFlowModel_"

using namespace std;

// Global variables required by htsim library
uint32_t SPEED = 100000; // default 100Gbps
uint32_t RTT = 1000;     // per-link delay in ns (for OCS Mixnet topology)
uint32_t RTT_rack = 500; // intra-rack RTT in ns (for FatTree topology)
uint32_t RTT_net = 500;  // inter-rack RTT in ns (for FatTree topology)

// ntoa/itoa utilities required by htsim topology code
#include <sstream>
string ntoa(double n) {
  stringstream s;
  s << n;
  return s.str();
}
string itoa(uint64_t n) {
  stringstream s;
  s << n;
  return s.str();
}

extern std::map<std::pair<std::pair<int, int>, int>, AstraSim::ncclFlowTag> receiver_pending_queue;

// Termination flag: set by sim_finish, checked by event loop
static bool g_simulation_done = false;
static int g_finished_count = 0;
static int g_total_nodes = 0;

// ======== ASTRASimNetwork class (same interface as ns3 version) ========
class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
private:
  int npu_offset;

public:
  ASTRASimNetwork(int rank, int npu_offset) : AstraNetworkAPI(rank) {
    this->npu_offset = npu_offset;
  }
  ~ASTRASimNetwork() {}

  int sim_comm_size(AstraSim::sim_comm comm, int *size) { return 0; }

  int sim_finish() {
    cout << "[htsim] sim_finish called by node " << rank << endl;
    for (auto it = nodeHash.begin(); it != nodeHash.end(); it++) {
      pair<int, int> p = it->first;
      if (p.second == 0) {
        cout << "All data sent from node " << p.first << " is " << it->second << "\n";
      } else {
        cout << "All data received by node " << p.first << " is " << it->second << "\n";
      }
    }
    g_simulation_done = true;
    return 0;
  }

  double sim_time_resolution() { return 0; }
  int sim_init(AstraSim::AstraMemoryAPI *MEM) { return 0; }

  AstraSim::timespec_t sim_get_time() {
    AstraSim::timespec_t timeSpec;
    // htsim time is picoseconds, convert to nanoseconds for astra-sim
    timeSpec.time_val = (double)(g_eventlist->now()) / 1000.0;
    return timeSpec;
  }

  virtual void sim_schedule(AstraSim::timespec_t delta,
                            void (*fun_ptr)(void *fun_arg), void *fun_arg) {
    // Convert nanoseconds to picoseconds
    simtime_picosec delay_ps = (simtime_picosec)(delta.time_val * 1000.0);
    if (delay_ps == 0) delay_ps = 1;  // minimum delay
    schedule_callback(delay_ps, fun_ptr, fun_arg);
    return;
  }

  virtual int sim_send(void *buffer, uint64_t count, int type, int dst, int tag,
                       AstraSim::sim_request *request,
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    dst += npu_offset;
    task1 t;
    t.src = rank;
    t.dest = dst;
    t.count = count;
    t.type = 0;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
    SendFlow(rank, dst, count, msg_handler, fun_arg, tag, request);
    return 0;
  }

  virtual int sim_recv(void *buffer, uint64_t count, int type, int src, int tag,
                       AstraSim::sim_request *request,
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    AstraSim::ncclFlowTag flowTag = request->flowTag;
    src += npu_offset;
    task1 t;
    t.src = src;
    t.dest = rank;
    t.count = count;
    t.type = 1;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    bool is_pp_simple = (tag >= 2000000);
    if (!is_pp_simple) {
      AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*)t.fun_arg;
      AstraSim::EventType event = ehd->event;
      tag = ehd->flowTag.tag_id;
      NcclLog->writeLog(NcclLogLevel::DEBUG,
          "[htsim recv] src %d sim_recv on rank %d tag_id %d channel id %d",
          src, rank, tag, ehd->flowTag.channel_id);
    } else {
      NcclLog->writeLog(NcclLogLevel::DEBUG,
          "[htsim recv] PP recv src %d on rank %d tag %d",
          src, rank, tag);
    }

    if (recvHash.find(make_pair(tag, make_pair(t.src, t.dest))) != recvHash.end()) {
      uint64_t count = recvHash[make_pair(tag, make_pair(t.src, t.dest))];
      if (count == t.count) {
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
        if (!is_pp_simple) {
          AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*)t.fun_arg;
          assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
          if (receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src), tag)) != 0) {
            AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src), tag)];
            receiver_pending_queue.erase(std::make_pair(std::make_pair(rank, src), tag));
            ehd->flowTag = pending_tag;
          }
        }
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else if (count > t.count) {
        recvHash[make_pair(tag, make_pair(t.src, t.dest))] = count - t.count;
        if (!is_pp_simple) {
          AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*)t.fun_arg;
          assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
          if (receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src), tag)) != 0) {
            AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src), tag)];
            receiver_pending_queue.erase(std::make_pair(std::make_pair(rank, src), tag));
            ehd->flowTag = pending_tag;
          }
        }
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else {
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
        t.count -= count;
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
      }
    } else {
      if (expeRecvHash.find(make_pair(tag, make_pair(t.src, t.dest))) == expeRecvHash.end()) {
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
        NcclLog->writeLog(NcclLogLevel::DEBUG,
            " [Packet arrived late, registering] src %d dest %d t.count: %llu tag_id %d",
            t.src, t.dest, t.count, tag);
      } else {
        NcclLog->writeLog(NcclLogLevel::DEBUG,
            " [Packet arrived late, re-registering] src %d dest %d tag_id %d",
            t.src, t.dest, tag);
      }
    }

  sim_recv_end_section:
    return 0;
  }
};

// ======== Command-line parameters ========
struct user_param {
  string workload;
  string topo;          // topology type
  int nodes;           // total GPU count
  int alpha;           // max OCS circuits per machine
  uint32_t speed;      // link speed in Mbps
  int reconf_delay_us; // reconfiguration delay in microseconds
  int dp_degree;
  int tp_degree;
  int pp_degree;
  int ep_degree;
  int gpus_per_server;
  int queuesize_pkts;  // queue size in packets
  int iterations;      // number of forward+backward passes
  bool ecs_only;       // force all traffic through ECS (no OCS)
  int os_ratio;        // oversubscription ratio for os_fattree/agg_os_fattree (default: 2)
  int rto_ms;          // TCP retransmission timeout in milliseconds (default: 1)

  user_param() {
    workload = "";
    topo = "mixnet";
    nodes = 8;
    alpha = 4;
    speed = 100000;       // 100Gbps
    reconf_delay_us = 10; // 10us
    dp_degree = 1;
    tp_degree = 1;
    pp_degree = 1;
    ep_degree = 8;
    gpus_per_server = 8;
    queuesize_pkts = 8;
    iterations = 1;
    ecs_only = false;
    os_ratio = 2;
    rto_ms = 1;
  }
};

static void print_usage() {
  cout << "Usage: simai_htsim [options]" << endl;
  cout << "  -w, --workload FILE     Workload file path (required)" << endl;
  cout << "  --topo TYPE             Topology: mixnet|fattree|os_fattree|agg_os_fattree|fc|flat (default: mixnet)" << endl;
  cout << "  --nodes N               Total GPU count (default: 8)" << endl;
  cout << "  --alpha N               Max OCS circuits per machine (default: 4, mixnet only)" << endl;
  cout << "  --speed N               Link speed in Mbps (default: 100000)" << endl;
  cout << "  --reconf_delay N        Reconf delay in us (default: 10, mixnet only)" << endl;
  cout << "  --dp_degree N           Data parallel degree (default: 1)" << endl;
  cout << "  --tp_degree N           Tensor parallel degree (default: 1)" << endl;
  cout << "  --pp_degree N           Pipeline parallel degree (default: 1)" << endl;
  cout << "  --ep_degree N           Expert parallel degree (default: 8)" << endl;
  cout << "  --gpus_per_server N     GPUs per server (default: 8)" << endl;
  cout << "  --queuesize N           Queue size in packets (default: 8)" << endl;
  cout << "  --iterations N          Number of passes (default: 1)" << endl;
  cout << "  --ecs_only              Force all traffic through ECS (no OCS, mixnet only)" << endl;
  cout << "  --os_ratio N            Oversubscription ratio (default: 2, os_fattree/agg_os_fattree only)" << endl;
  cout << "  --rto N                 TCP retransmission timeout in ms (default: 1)" << endl;
}

static int parse_params(int argc, char* argv[], struct user_param* params) {
  static struct option long_options[] = {
    {"workload",      required_argument, 0, 'w'},
    {"topo",          required_argument, 0, 'T'},
    {"nodes",         required_argument, 0, 'N'},
    {"alpha",         required_argument, 0, 'a'},
    {"speed",         required_argument, 0, 's'},
    {"reconf_delay",  required_argument, 0, 'r'},
    {"dp_degree",     required_argument, 0, 'd'},
    {"tp_degree",     required_argument, 0, 't'},
    {"pp_degree",     required_argument, 0, 'p'},
    {"ep_degree",     required_argument, 0, 'e'},
    {"gpus_per_server", required_argument, 0, 'g'},
    {"queuesize",     required_argument, 0, 'q'},
    {"iterations",    required_argument, 0, 'i'},
    {"ecs_only",      no_argument,       0, 'E'},
    {"os_ratio",      required_argument, 0, 'O'},
    {"rto",           required_argument, 0, 'R'},
    {"help",          no_argument,       0, 'h'},
    {0, 0, 0, 0}
  };

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "w:h", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'w': params->workload = optarg; break;
      case 'T': params->topo = optarg; break;
      case 'N': params->nodes = stoi(optarg); break;
      case 'a': params->alpha = stoi(optarg); break;
      case 's': params->speed = stoi(optarg); break;
      case 'r': params->reconf_delay_us = stoi(optarg); break;
      case 'd': params->dp_degree = stoi(optarg); break;
      case 't': params->tp_degree = stoi(optarg); break;
      case 'p': params->pp_degree = stoi(optarg); break;
      case 'e': params->ep_degree = stoi(optarg); break;
      case 'g': params->gpus_per_server = stoi(optarg); break;
      case 'q': params->queuesize_pkts = stoi(optarg); break;
      case 'i': params->iterations = stoi(optarg); break;
      case 'E': params->ecs_only = true; break;
      case 'O': params->os_ratio = stoi(optarg); break;
      case 'R': params->rto_ms = stoi(optarg); break;
      case 'h': print_usage(); return 1;
      default:  print_usage(); return 1;
    }
  }

  if (params->workload.empty()) {
    cerr << "Error: workload file is required (-w)" << endl;
    print_usage();
    return 1;
  }
  return 0;
}

// ======== main ========
int main(int argc, char *argv[]) {
  struct user_param params;

  MockNcclLog::set_log_name("SimAI_htsim.log");
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::INFO, " init SimAI_htsim.log ");

  if (parse_params(argc, argv, &params)) {
    return 0;
  }

  // Set global SPEED
  SPEED = params.speed;
  g_gpus_per_server = params.gpus_per_server;

  int gpu_num = params.nodes;
  int nodes_num = gpu_num;  // htsim uses GPU-level nodes only

  // Parse topology type
  string topo_name = params.topo;
  if (topo_name == "mixnet")          g_topo_type = TOPO_MIXNET;
  else if (topo_name == "fattree")    g_topo_type = TOPO_FATTREE;
  else if (topo_name == "os_fattree") g_topo_type = TOPO_OS_FATTREE;
  else if (topo_name == "agg_os_fattree") g_topo_type = TOPO_AGG_OS_FATTREE;
  else if (topo_name == "fc")         g_topo_type = TOPO_FC;
  else if (topo_name == "flat")       g_topo_type = TOPO_FLAT;
  else {
    cerr << "Error: unknown topology '" << topo_name << "'" << endl;
    print_usage();
    return 1;
  }

  // ---- Create log directory ----
  string exe_path = argv[0];
  string project_root = exe_path.substr(0, exe_path.find_last_of('/'));
  if (project_root.empty()) project_root = ".";
  string log_base = project_root + "/log";
  mkdir(log_base.c_str(), 0755);

  // Build run dir name: {topo}_{timestamp}
  string net_mode;
  if (g_topo_type == TOPO_MIXNET) {
    net_mode = params.ecs_only ? "mixnet_ECS" : "mixnet_OCS";
  } else {
    net_mode = topo_name;
  }
  string run_dir_name = net_mode;

  // Add timestamp to avoid overwrite
  time_t now_t = time(nullptr);
  struct tm* tm_info = localtime(&now_t);
  char ts_buf[32];
  strftime(ts_buf, sizeof(ts_buf), "%m%d_%H%M%S", tm_info);
  string log_dir = log_base + "/" + run_dir_name + "_" + ts_buf;
  mkdir(log_dir.c_str(), 0755);

  // Redirect stdout to trace.log, keep stderr for errors
  string trace_path = log_dir + "/trace.log";
  freopen(trace_path.c_str(), "w", stdout);

  // FCT output goes into log dir
  string fct_path = log_dir + "/fct_output.txt";

  cout << "=== SimAI htsim Backend ===" << endl;
  cout << "Topology: " << topo_name << endl;
  cout << "GPUs: " << gpu_num << endl;
  cout << "Link speed: " << params.speed << " Mbps" << endl;
  if (g_topo_type == TOPO_MIXNET) {
    cout << "Alpha (OCS circuits): " << params.alpha << endl;
    cout << "Reconf delay: " << params.reconf_delay_us << " us" << endl;
    cout << "ECS only: " << (params.ecs_only ? "YES" : "NO") << endl;
  }
  cout << "DP/TP/PP/EP: " << params.dp_degree << "/" << params.tp_degree
       << "/" << params.pp_degree << "/" << params.ep_degree << endl;
  cout << "GPUs per server: " << params.gpus_per_server << endl;
  cout << "Iterations: " << params.iterations << endl;
  cout << "Workload: " << params.workload << endl;
  cout << "==========================" << endl;

  // Set ECS-only mode and RTO in entry.h
  g_force_ecs_only = params.ecs_only;
  g_rto_ms = params.rto_ms;

  // 1. Create htsim EventList
  EventList eventlist;
  eventlist.setEndtime(timeFromSec(2000000));
  g_eventlist = &eventlist;

  // 2. Compute machine count and FatTree K parameter
  int num_machines = gpu_num / params.gpus_per_server;
  int fattree_node = num_machines;
  int fattree_k = 0;
  {
    int k = 0;
    while (k * k * k / 4 < num_machines) {
      k += 2;
    }
    fattree_k = k;
    fattree_node = k * k * k / 4;
  }

  // 3. Create topology based on --topo
  Mixnet* mixnet = nullptr;
  FatTreeTopology* fattree = nullptr;
  cout << "[htsim] Machines: " << num_machines << " FatTree K=" << fattree_k
       << " FatTree nodes: " << fattree_node << endl;

  if (g_topo_type == TOPO_MIXNET) {
    // OCS-ECS hybrid: FatTree (partial BW) + Mixnet OCS overlay
    uint32_t ecs_link_speed = params.speed * (8 - params.alpha);
    cout << "[htsim] ECS link speed: " << ecs_link_speed << " Mbps" << endl;

    fattree = new FatTreeTopology(
        fattree_node, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, LOSSLESS_INPUT_ECN, ecs_link_speed);
    g_fattree_topo = fattree;

    mixnet = new Mixnet(
        gpu_num, memFromPkt(params.queuesize_pkts),
        NULL, eventlist, NULL, ECN,
        timeFromUs((double)params.reconf_delay_us),
        fattree, params.alpha,
        params.dp_degree, params.tp_degree, params.pp_degree, params.ep_degree);
    g_mixnet_topo = mixnet;
    g_topology = mixnet;

    cout << "[htsim] Mixnet topology created: region_size=" << mixnet->region_size
         << " region_num=" << mixnet->region_num << endl;

  } else if (g_topo_type == TOPO_FATTREE) {
    // Full-bandwidth fat-tree (all 8 ports)
    uint32_t full_speed = params.speed * 8;
    fattree = new FatTreeTopology(
        fattree_node, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, LOSSLESS_INPUT_ECN, full_speed);
    g_fattree_topo = fattree;
    g_topology = fattree;
    cout << "[htsim] FatTree topology created (full BW: " << full_speed << " Mbps)" << endl;

  } else if (g_topo_type == TOPO_OS_FATTREE) {
    int racksz = params.os_ratio;  // hosts per rack / uplinks per rack
    auto* top = new OverSubscribedFatTree(
        fattree_k, racksz, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, LOSSLESS_INPUT_ECN);
    g_topology = top;
    cout << "[htsim] OverSubscribedFatTree created (K=" << fattree_k << " racksz=" << racksz << ")" << endl;

  } else if (g_topo_type == TOPO_AGG_OS_FATTREE) {
    int racksz = params.os_ratio;
    auto* top = new AggOverSubscribedFatTree(
        fattree_k, racksz, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, LOSSLESS_INPUT_ECN);
    g_topology = top;
    cout << "[htsim] AggOverSubscribedFatTree created (K=" << fattree_k << " racksz=" << racksz << ")" << endl;

  } else if (g_topo_type == TOPO_FC) {
    auto* top = new FCTopology(
        num_machines, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, ECN);
    g_topology = top;
    cout << "[htsim] FCTopology created (" << num_machines << " nodes, ECN queuesize=" << params.queuesize_pkts << "pkts)" << endl;

  } else if (g_topo_type == TOPO_FLAT) {
    auto* top = new FlatTopology(
        num_machines, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, ECN);
    g_topology = top;
    cout << "[htsim] FlatTopology created (" << num_machines << " nodes, ECN queuesize=" << params.queuesize_pkts << "pkts)" << endl;
  }

  // 4. Create TCP scanner and FCT output
  TcpRtxTimerScanner tcpRtxScanner(timeFromMs(1), eventlist);
  g_tcp_scanner = &tcpRtxScanner;
  init_fct_output(fct_path);

  // 5. Mixnet-specific: traffic recorder and topo manager
  // Use unique_ptr-like pattern with raw pointers for stack lifetime
  All2AllTrafficRecorder* _demand_recorder_storage = nullptr;
  MixnetTopoManager* _topomanager_storage = nullptr;

  if (g_topo_type == TOPO_MIXNET && mixnet != nullptr) {
    int layer_num = 200;
    _demand_recorder_storage = new All2AllTrafficRecorder(
        layer_num, mixnet->region_num, mixnet->region_size, &tcpRtxScanner);
    g_demand_recorder = _demand_recorder_storage;

    _topomanager_storage = new MixnetTopoManager(
        mixnet, _demand_recorder_storage,
        timeFromUs((double)params.reconf_delay_us), eventlist);
    g_topomanager = _topomanager_storage;
    cout << "[htsim] MixnetTopoManager created" << endl;
  }

  // 6. Initialize port numbers
  for (int i = 0; i < nodes_num; i++) {
    for (int j = 0; j < nodes_num; j++) {
      portNumber[i][j] = 10000;
    }
  }

  // 7. Set global node count for termination
  g_total_nodes = nodes_num;

  // 8. Create astra-sim systems (same pattern as ns3 version)
  std::vector<ASTRASimNetwork*> networks(nodes_num, nullptr);
  std::vector<AstraSim::Sys*> systems(nodes_num, nullptr);

  // NVSwitch virtual nodes (same pattern as NS-3 version)
  // Each physical machine has one NVSwitch; NVSwitch IDs start after GPU IDs
  int nvswitch_num = num_machines;
  std::vector<int> NVswitchs;
  std::map<int, int> node2nvswitch;
  for (int i = 0; i < gpu_num; ++i) {
    node2nvswitch[i] = gpu_num + i / params.gpus_per_server;
  }
  for (int i = gpu_num; i < gpu_num + nvswitch_num; ++i) {
    node2nvswitch[i] = i;
    NVswitchs.push_back(i);
  }
  cout << "[htsim] NVSwitch nodes: " << NVswitchs.size() << " (IDs: "
       << gpu_num << " to " << gpu_num + nvswitch_num - 1 << ")" << endl;

  for (int j = 0; j < nodes_num; j++) {
    networks[j] = new ASTRASimNetwork(j, 0);
    systems[j] = new AstraSim::Sys(
        networks[j],          // NI
        nullptr,              // MEM
        j,                    // id
        0,                    // npu_offset
        params.iterations,    // num_passes
        {nodes_num},          // physical_dims
        {1},                  // queues_per_dim
        "",                   // my_sys
        params.workload,      // my_workload
        1,                    // comm_scale
        1,                    // compute_scale
        1,                    // injection_scale
        1,                    // total_stat_rows
        0,                    // stat_row
        RESULT_PATH,          // path
        "test1",              // run_name
        true,                 // seprate_log
        false,                // rendezvous_enabled
        GPUType::H100,        // gpu_type
        {gpu_num},            // all_gpus
        NVswitchs,            // NVSwitchs
        params.gpus_per_server // ngpus_per_node
    );
    systems[j]->nvswitch_id = node2nvswitch[j];
    systems[j]->num_gpus = nodes_num;
  }
  cout << "[htsim] Created " << nodes_num << " Sys instances" << endl;

  // 8. Fire workloads
  for (int i = 0; i < nodes_num; i++) {
    systems[i]->workload->fire();
  }
  cout << "[htsim] Workloads fired" << endl;

  // 9. Run htsim event loop (replaces NS-3's Simulator::Run())
  cout << "Running htsim simulation..." << endl;
  uint64_t event_count = 0;
  while (!g_simulation_done && eventlist.doNextEvent()) {
    event_count++;
    if (event_count % 10000000 == 0) {
      cout << "[htsim] Events processed: " << event_count
           << " Time: " << timeAsMs(eventlist.now()) << " ms" << endl;
    }
  }
  cout << "Simulation complete. Total events: " << event_count << endl;
  cout << "Final time: " << timeAsMs(eventlist.now()) << " ms ("
       << timeAsSec(eventlist.now()) << " s)" << endl;

  // 10. Print flow statistics
  cout << endl;
  cout << "======== Flow Statistics ========" << endl;
  if (g_topo_type == TOPO_MIXNET) {
    cout << "OCS flows:    " << g_flow_count_ocs << " (" << g_flow_bytes_ocs << " bytes)" << endl;
  }
  cout << "Network flows: " << g_flow_count_ecs << " (" << g_flow_bytes_ecs << " bytes)" << endl;
  cout << "NVLink flows:  " << g_flow_count_nvlink << " (" << g_flow_bytes_nvlink << " bytes)" << endl;
  if (g_topo_type == TOPO_MIXNET) {
    cout << "Deferred flows (reconf): " << g_flow_count_deferred << endl;
  }
  uint64_t total_flows = g_flow_count_ocs + g_flow_count_ecs + g_flow_count_nvlink;

  // Collect retransmission statistics from all TCP sources
  uint64_t total_packets_sent = 0;
  uint64_t total_retransmissions = 0;
  for (auto it = g_tcp_scanner->_tcps.begin(); it != g_tcp_scanner->_tcps.end(); ++it) {
    total_packets_sent += (*it)->_packets_sent;
    total_retransmissions += (*it)->_drops;
  }
  double retx_rate = (total_packets_sent > 0) ? (100.0 * total_retransmissions / total_packets_sent) : 0.0;

  cout << endl;
  cout << "======== Retransmission Statistics ========" << endl;
  cout << "RTO: " << params.rto_ms << " ms" << endl;
  cout << "Total packets sent: " << total_packets_sent << endl;
  cout << "Total retransmissions: " << total_retransmissions << endl;
  cout << "Retransmission rate: " << retx_rate << "%" << endl;
  cout << "TCP flows tracked: " << g_tcp_scanner->_tcps.size() << endl;
  cout << "============================================" << endl;
  cout << "========================================" << endl;

  // Write stats.txt summary
  {
    ofstream stats(log_dir + "/stats.txt");
    stats << "======== Run Configuration ========" << endl;
    stats << "Topology: " << topo_name << endl;
    stats << "GPUs: " << gpu_num << endl;
    stats << "TP: " << params.tp_degree << "  PP: " << params.pp_degree
          << "  EP: " << params.ep_degree << "  DP: " << params.dp_degree << endl;
    stats << "Iterations: " << params.iterations << endl;
    stats << "Link speed: " << params.speed << " Mbps" << endl;
    stats << "Machines: " << num_machines << endl;
    if (g_topo_type == TOPO_MIXNET) {
      stats << "Network mode: " << (params.ecs_only ? "ECS only" : "OCS+ECS mixnet") << endl;
      stats << "Alpha (OCS circuits): " << params.alpha << endl;
      uint32_t ecs_link_speed = params.speed * (8 - params.alpha);
      stats << "ECS link speed: " << ecs_link_speed << " Mbps" << endl;
      stats << "Queue type (ECS): LOSSLESS_INPUT_ECN" << endl;
      stats << "Queue type (OCS): ECN" << endl;
    } else if (g_topo_type == TOPO_FATTREE) {
      stats << "FatTree link speed: " << (params.speed * 8) << " Mbps" << endl;
      stats << "FatTree nodes: " << fattree_node << "  K=" << fattree_k << endl;
      stats << "Queue type: LOSSLESS_INPUT_ECN" << endl;
    } else if (g_topo_type == TOPO_OS_FATTREE || g_topo_type == TOPO_AGG_OS_FATTREE) {
      stats << "K=" << fattree_k << "  OS ratio=" << params.os_ratio << endl;
      stats << "Queue type: LOSSLESS_INPUT_ECN" << endl;
    } else {
      stats << "Nodes: " << num_machines << endl;
    }
    stats << endl;

    stats << "======== Flow Statistics ========" << endl;
    if (g_topo_type == TOPO_MIXNET) {
      stats << "OCS flows:    " << g_flow_count_ocs << " (" << g_flow_bytes_ocs << " bytes, "
            << (g_flow_bytes_ocs / 1048576.0) << " MB)" << endl;
    }
    stats << "Network flows: " << g_flow_count_ecs << " (" << g_flow_bytes_ecs << " bytes, "
          << (g_flow_bytes_ecs / 1048576.0) << " MB)" << endl;
    stats << "NVLink flows:  " << g_flow_count_nvlink << " (" << g_flow_bytes_nvlink << " bytes, "
          << (g_flow_bytes_nvlink / 1048576.0) << " MB)" << endl;
    if (g_topo_type == TOPO_MIXNET) {
      stats << "Deferred flows (reconf): " << g_flow_count_deferred << endl;
    }
    if (total_flows > 0) {
      stats << endl;
      stats << "Flow count ratio:" << endl;
      if (g_topo_type == TOPO_MIXNET) {
        stats << "  OCS:     " << (100.0 * g_flow_count_ocs / total_flows) << "%" << endl;
      }
      stats << "  Network: " << (100.0 * g_flow_count_ecs / total_flows) << "%" << endl;
      stats << "  NVLink:  " << (100.0 * g_flow_count_nvlink / total_flows) << "%" << endl;
      uint64_t total_bytes = g_flow_bytes_ocs + g_flow_bytes_ecs + g_flow_bytes_nvlink;
      if (total_bytes > 0) {
        stats << endl;
        stats << "Bytes ratio:" << endl;
        if (g_topo_type == TOPO_MIXNET) {
          stats << "  OCS:     " << (100.0 * g_flow_bytes_ocs / total_bytes) << "%" << endl;
        }
        stats << "  Network: " << (100.0 * g_flow_bytes_ecs / total_bytes) << "%" << endl;
        stats << "  NVLink:  " << (100.0 * g_flow_bytes_nvlink / total_bytes) << "%" << endl;
        if (g_topo_type == TOPO_MIXNET) {
          uint64_t cross_machine_bytes = g_flow_bytes_ocs + g_flow_bytes_ecs;
          if (cross_machine_bytes > 0) {
            stats << endl;
            stats << "Cross-machine bytes ratio (OCS vs ECS):" << endl;
            stats << "  OCS: " << (100.0 * g_flow_bytes_ocs / cross_machine_bytes) << "%" << endl;
            stats << "  ECS: " << (100.0 * g_flow_bytes_ecs / cross_machine_bytes) << "%" << endl;
          }
        }
      }
    }

    stats << endl;
    stats << "======== Retransmission Statistics ========" << endl;
    stats << "RTO: " << params.rto_ms << " ms" << endl;
    stats << "Total packets sent: " << total_packets_sent << endl;
    stats << "Total retransmissions: " << total_retransmissions << endl;
    stats << "Retransmission rate: " << retx_rate << "%" << endl;
    stats << "TCP flows tracked: " << g_tcp_scanner->_tcps.size() << endl;

    stats << endl;
    stats << "======== Timing ========" << endl;
    stats << "Total events: " << event_count << endl;
    stats << "Final sim time: " << timeAsMs(eventlist.now()) << " ms" << endl;
    stats << endl;
    stats << "Log directory: " << log_dir << endl;
    stats.close();
  }

  // Print log directory to stderr so user can find it
  cerr << "[LOG] Output directory: " << log_dir << endl;

  // 11. Cleanup
  for (int j = 0; j < nodes_num; j++) {
    if (networks[j]) networks[j]->sim_finish();
  }

  for (int j = 0; j < nodes_num; j++) {
    delete networks[j];
  }
  delete _topomanager_storage;
  delete _demand_recorder_storage;
  delete mixnet;
  delete fattree;
  // Note: non-mixnet topologies are cleaned up via g_topology if needed

  return 0;
}
