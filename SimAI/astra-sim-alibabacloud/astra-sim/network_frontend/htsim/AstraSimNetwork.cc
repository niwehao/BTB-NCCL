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
#include <set>
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

// Per-pass timing: filled via on_pass_end_hook, read in stats output.
static std::vector<double> g_pass_end_ms;
void (*on_pass_end_hook)(int) = nullptr;
// Per-rank pass-end hook: called by Workload.cc from every rank once that rank
// finishes a pass. MoEReconfigManager uses this to close pass P's traffic
// matrix only after all ranks have reported pass P done.
void (*on_rank_pass_end_hook)(int rank, int pass) = nullptr;

// Global trace verbosity (declared extern in MockNcclLog.h, read by TRACE1/TRACE2).
// CLI: --trace_level N (0=silent, 1=important, 2=full debug). Default: 1.
int g_trace_level = 1;

// ======== ASTRASimNetwork class (same interface as ns3 version) ========
class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
  //ASTRASimNetwork 是 astra-sim 核心和 htsim
  // 仿真器之间的适配器(adapter),每个 GPU 一个实例,实现了
  // AstraNetworkAPI 抽象接口。
private:
  int npu_offset;

public:
  ASTRASimNetwork(int rank, int npu_offset) : AstraNetworkAPI(rank) {
    this->npu_offset = npu_offset;
  }
  ~ASTRASimNetwork() {}

  int sim_comm_size(AstraSim::sim_comm comm, int *size) { return 0; }

  int sim_finish() {//当一个 rank 跑完所有 pass, 退出 sim loop 时通知网络后端。
    cout << "[htsim] sim_finish called by node " << rank << endl;
    // Per-node "All data sent/received" dump disabled by user request.
    g_simulation_done = true;
    return 0;
  }

  double sim_time_resolution() { return 0; }//返回这个后端最小的时间步长(可能用于 astra-sim决定调度精度)。
  int sim_init(AstraSim::AstraMemoryAPI *MEM) { return 0; }
  // 让网络后端完成它自己的启动准备(分配资源、绑定内存接 
  // 口、连接外部仿真器等)。参数 MEM 是 astra-sim
  // 的内存子系统接口

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
    sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;// 记"我发过这条消息"(供后续 recv 匹配
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
  //   匹配逻辑(简化):
  // if (recvHash 已有 tag 的数据)
  //     if (已到量 == 预期量) → 立刻回调 msg_handler,删条目
  //     if (已到量  > 预期量) → 分一部分走,剩的留在 recvHash
  //     if (已到量  < 预期量) → 把已到的消掉,剩的挂到
  // expeRecvHash 等
  // else
  //     把 recv 请求挂到 expeRecvHash,等数据到

    if (recvHash.find(make_pair(tag, make_pair(t.src, t.dest))) != recvHash.end()) {
      uint64_t count = recvHash[make_pair(tag, make_pair(t.src, t.dest))];
      if (count == t.count) {
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));// 已经到达但还没有匹配 recv
  //的数据量(从 entry.h 那边的 flow 完成回调写进来)
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
        //expeRecvHash[tag, src, dst] — 已经 post 但数据还没到的recv
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
  int expert_topk;     // MoE top-k routing (0=uniform, no hotspot)
  double expert_skew;  // Zipf skew (1.0=moderate, 2.0=strong)
  int expert_seed;     // random seed for expert routing
  int moe_volatility;  // MoE hotspot bucket size: every N layers share one distribution (default 1)
  int reconf_top_n;    // skip reconfig if top-N pairs already connected (0=always reconfig)
  int trace_level;     // trace.log verbosity: 0=silent, 1=important, 2=full debug (default 1)
  queue_type ecs_qt;   // queue type for ECS / standalone fattree-family / fc / flat
  queue_type ocs_qt;   // queue type for mixnet / prenet OCS overlay

  // --- prenet-only (ignored unless --topo prenet) ---
  int      prenet_variant_k;
  double   prenet_probe_ratio;
  int      prenet_arbiter_window_us;
  int      prenet_confidence_init;
  int      prenet_confidence_max;
  uint64_t prenet_predictor_log_every;

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
    expert_topk = 0;
    expert_skew = 1.0;
    expert_seed = 42;
    moe_volatility = 1;
    reconf_top_n = 0;
    trace_level = 1;
    prenet_variant_k = 8;
    prenet_probe_ratio = 0.05;
    prenet_arbiter_window_us = 2;
    prenet_confidence_init = 1;
    prenet_confidence_max = 3;
    prenet_predictor_log_every = 1000;
    ecs_qt = LOSSLESS_INPUT_ECN;
    ocs_qt = ECN;
  }
};

// Map queue_type enum <-> JSON/CLI string name.
static bool parse_qt_name(const std::string& v, queue_type* out) {
  if      (v == "random")             *out = RANDOM;
  else if (v == "composite")          *out = COMPOSITE;
  else if (v == "ctrl_prio")          *out = CTRL_PRIO;
  else if (v == "ecn")                *out = ECN;
  else if (v == "lossless")           *out = LOSSLESS;
  else if (v == "lossless_input")     *out = LOSSLESS_INPUT;
  else if (v == "lossless_input_ecn") *out = LOSSLESS_INPUT_ECN;
  else return false;
  return true;
}
static const char* qt_to_string(queue_type qt) {
  switch (qt) {
    case RANDOM:             return "RANDOM";
    case COMPOSITE:          return "COMPOSITE";
    case CTRL_PRIO:          return "CTRL_PRIO";
    case ECN:                return "ECN";
    case LOSSLESS:           return "LOSSLESS";
    case LOSSLESS_INPUT:     return "LOSSLESS_INPUT";
    case LOSSLESS_INPUT_ECN: return "LOSSLESS_INPUT_ECN";
  }
  return "UNKNOWN";
}

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
  cout << "  --expert_topk N         MoE top-k routing, 0=uniform (default: 0)" << endl;
  cout << "  --expert_skew F         Zipf skew for expert hotspot (default: 1.0)" << endl;
  cout << "  --expert_seed N         Random seed for expert routing (default: 42)" << endl;
  cout << "  --moe_volatility N      MoE hotspot bucket size — every N layers share one distribution (default: 1)" << endl;
  cout << "  --reconf_top_n N        Skip reconfig if top-N pairs already connected (default: 0=always reconfig, mixnet only)" << endl;
  cout << "  --trace_level N         trace.log verbosity: 0=silent, 1=important (default), 2=full debug" << endl;
  cout << "  --ecs_qt NAME           ECS queue type: ecn|lossless|lossless_input|lossless_input_ecn|composite|ctrl_prio|random (default: lossless_input_ecn)" << endl;
  cout << "  --ocs_qt NAME           OCS overlay queue type (mixnet/prenet only), same values (default: ecn)" << endl;
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
    {"expert_topk",   required_argument, 0, 'K'},
    {"expert_skew",   required_argument, 0, 'S'},
    {"expert_seed",   required_argument, 0, 'D'},
    {"moe_volatility",required_argument, 0, 'V'},
    {"reconf_top_n",  required_argument, 0, 'n'},
    {"trace_level",   required_argument, 0, 'L'},
    {"prenet_variant_k",           required_argument, 0, 300},
    {"prenet_probe_ratio",         required_argument, 0, 301},
    {"prenet_arbiter_window_us",   required_argument, 0, 302},
    {"prenet_confidence_init",     required_argument, 0, 303},
    {"prenet_confidence_max",      required_argument, 0, 304},
    {"prenet_predictor_log_every", required_argument, 0, 305},
    {"ecs_qt",        required_argument, 0, 400},
    {"ocs_qt",        required_argument, 0, 401},
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
      case 'K': params->expert_topk = stoi(optarg); break;
      case 'S': params->expert_skew = stod(optarg); break;
      case 'D': params->expert_seed = stoi(optarg); break;
      case 'V': params->moe_volatility = stoi(optarg); break;
      case 'n': params->reconf_top_n = stoi(optarg); break;
      case 'L': params->trace_level = stoi(optarg); break;
      case 300: params->prenet_variant_k = stoi(optarg); break;
      case 301: params->prenet_probe_ratio = stod(optarg); break;
      case 302: params->prenet_arbiter_window_us = stoi(optarg); break;
      case 303: params->prenet_confidence_init = stoi(optarg); break;
      case 304: params->prenet_confidence_max = stoi(optarg); break;
      case 305: params->prenet_predictor_log_every = (uint64_t)stoll(optarg); break;
      case 400: {
        queue_type q;
        if (!parse_qt_name(optarg, &q)) { cerr << "Error: unknown --ecs_qt '" << optarg << "'" << endl; return 1; }
        params->ecs_qt = q; break;
      }
      case 401: {
        queue_type q;
        if (!parse_qt_name(optarg, &q)) { cerr << "Error: unknown --ocs_qt '" << optarg << "'" << endl; return 1; }
        params->ocs_qt = q; break;
      }
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
#ifdef PRENET_ENABLED
  else if (topo_name == "prenet")     g_topo_type = TOPO_PRENET;
#endif
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

  // Redirect stdout:
  //   trace_level == 0 → /dev/null (stats.txt still written separately)
  //   trace_level >= 1 → trace.log; TRACE1/TRACE2 macros gate content inside.
  string trace_path = log_dir + "/trace.log";
  if (params.trace_level <= 0) {
    freopen("/dev/null", "w", stdout);
  } else {
    freopen(trace_path.c_str(), "w", stdout);
  }

  // FCT output goes into log dir
  string fct_path = log_dir + "/fct_output.txt";

  cout << "=== SimAI htsim Backend ===" << endl;
  cout << "Topology: " << topo_name << endl;
  cout << "GPUs: " << gpu_num << endl;
  cout << "Link speed: " << params.speed << " Mbps" << endl;
  if (g_topo_type == TOPO_MIXNET) {
    cout << "Alpha (OCS circuits): " << params.alpha << endl;
    cout << "Reconf delay: " << params.reconf_delay_us << " us" << endl;
    cout << "Reconf top-N: " << params.reconf_top_n << " (0=always reconfig)" << endl;
    cout << "ECS only: " << (params.ecs_only ? "YES" : "NO") << endl;
  }
  cout << "DP/TP/PP/EP: " << params.dp_degree << "/" << params.tp_degree
       << "/" << params.pp_degree << "/" << params.ep_degree << endl;
  cout << "GPUs per server: " << params.gpus_per_server << endl;
  cout << "Iterations: " << params.iterations << endl;
  cout << "Workload: " << params.workload << endl;
  if (params.expert_topk > 0) {
    cout << "Expert hotspot: topk=" << params.expert_topk
         << " skew=" << params.expert_skew
         << " seed=" << params.expert_seed
         << " volatility=" << params.moe_volatility << endl;
  }
  cout << "==========================" << endl;

  // Set ECS-only mode, RTO, and GPU count in entry.h
  g_force_ecs_only = params.ecs_only;
  g_rto_ms = params.rto_ms;
  g_total_gpus = gpu_num;
  g_expert_topk = params.expert_topk;
  g_expert_skew = params.expert_skew;
  g_expert_seed = params.expert_seed;
  g_moe_volatility = params.moe_volatility;
  g_trace_level    = params.trace_level;
  g_reconf_top_n = params.reconf_top_n;

  // 1. Create htsim EventList
  EventList eventlist;
  eventlist.setEndtime(timeFromSec(2000000));
  g_eventlist = &eventlist;

  // Register per-pass timing hook (called by Workload.cc at each pass_counter++)
  on_pass_end_hook = [](int pass_idx) {//pass(iteration)结束时的仿真时间,后面写进统计报告。  
    if (g_eventlist) g_pass_end_ms.push_back(timeAsMs(g_eventlist->now()));
  };
  // Register per-rank pass-end hook. The manager counts how many ranks have
  // reported completion of pass P; once all of them do, pass P's block_tm is
  // closed and promoted to last_block_tm for the next pass's prediction.
  //
  // prenet/mixnet use DIFFERENT lambdas (principles §2.4) — one lambda per topo
  // type so neither's code path interferes with the other's.
#ifdef PRENET_ENABLED//假函数，预留接口，暂时不实现
  if (g_topo_type == TOPO_PRENET) {
    on_rank_pass_end_hook = [](int rank, int pass) {
      if (g_prenet_predictor) g_prenet_predictor->on_pass_end(rank, pass);
    };
  } else
#endif
  {
    on_rank_pass_end_hook = [](int rank, int pass) {
      if (g_mixnet_topo == nullptr) return;  // only needed for mixnet topology
      int region_size = g_mixnet_topo->region_size;
      g_moe_reconfig_mgr.on_rank_pass_end(rank, pass, g_total_gpus, region_size);
    };
  }

  // 2. Compute machine count and FatTree K parameter
  //按照数量去构造fat-tree的形状，num_machines 决定了 fat-tree 的规模和 K 参数。作者选了 "把 machine 当叶节点 + per-machine 聚合链路" 的抽象,用带宽缩减代替端口缩减。你的直觉是对的,只是代码没做那个物理级的细化 —— 要细化就得改成 per-GPU fat-tree(fattree_node = gpu_num),然后根据 α 实际去掉某些叶端口。
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
  //   代码里 fat-tree 的拓扑形状只由 num_machines 决定,alpha 不改 K、不改 switch 数、不改端口数 ——            
  // 它只改每条链路的带宽(通过 ecs_link_speed = speed × (gpus_per_server − alpha))。也就是作者选了 "把
  // machine 当叶节点 + per-machine 聚合链路"                                                                
  // 的抽象,用带宽缩减代替端口缩减。你的直觉是对的,只是代码没做那个物理级的细化 —— 要细化就得改成 per-GPU
  // fat-tree(fattree_node = gpu_num),然后根据 α 实际去掉某些叶端口。
  //1:1 无收敛 fat-tree",等速是自洽的 因为链路数量不一样
    uint32_t ecs_link_speed = params.speed * (params.gpus_per_server - params.alpha);
    cout << "[htsim] ECS link speed: " << ecs_link_speed << " Mbps" << endl;

    fattree = new FatTreeTopology(
        fattree_node, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, params.ecs_qt, ecs_link_speed);
    g_fattree_topo = fattree;

    mixnet = new Mixnet(
        gpu_num, memFromPkt(params.queuesize_pkts),
        NULL, eventlist, NULL, params.ocs_qt,
        timeFromUs((double)params.reconf_delay_us),
        fattree, params.alpha,
        params.dp_degree, params.tp_degree, params.pp_degree, params.ep_degree,
        params.gpus_per_server);
    g_mixnet_topo = mixnet;
    g_topology = mixnet;

    cout << "[htsim] Mixnet topology created: region_size=" << mixnet->region_size
         << " region_num=" << mixnet->region_num << endl;

  } else if (g_topo_type == TOPO_FATTREE) {
    // Full-bandwidth fat-tree (all 8 ports)
    uint32_t full_speed = params.speed * params.gpus_per_server;
    fattree = new FatTreeTopology(
        fattree_node, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, params.ecs_qt, full_speed);
    g_fattree_topo = fattree;
    g_topology = fattree;
    cout << "[htsim] FatTree topology created (full BW: " << full_speed << " Mbps)" << endl;

  } else if (g_topo_type == TOPO_OS_FATTREE) {
    int racksz = params.os_ratio;  // hosts per rack / uplinks per rack
    auto* top = new OverSubscribedFatTree(
        fattree_k, racksz, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, params.ecs_qt);
    g_topology = top;
    cout << "[htsim] OverSubscribedFatTree created (K=" << fattree_k << " racksz=" << racksz << ")" << endl;

  } else if (g_topo_type == TOPO_AGG_OS_FATTREE) {
    int racksz = params.os_ratio;
    auto* top = new AggOverSubscribedFatTree(
        fattree_k, racksz, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, params.ecs_qt);
    g_topology = top;
    cout << "[htsim] AggOverSubscribedFatTree created (K=" << fattree_k << " racksz=" << racksz << ")" << endl;

  } else if (g_topo_type == TOPO_FC) {
    auto* top = new FCTopology(
        num_machines, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, params.ecs_qt);
    g_topology = top;
    cout << "[htsim] FCTopology created (" << num_machines << " nodes, qt=" << qt_to_string(params.ecs_qt)
         << " queuesize=" << params.queuesize_pkts << "pkts)" << endl;

  } else if (g_topo_type == TOPO_FLAT) {
    auto* top = new FlatTopology(
        num_machines, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, params.ecs_qt);
    g_topology = top;
    cout << "[htsim] FlatTopology created (" << num_machines << " nodes, qt=" << qt_to_string(params.ecs_qt)
         << " queuesize=" << params.queuesize_pkts << "pkts)" << endl;
  }
#ifdef PRENET_ENABLED //生成prenetwork的拓扑，预留接口，暂时不实现
  else if (g_topo_type == TOPO_PRENET) {
    // ECS underlay: independent FatTree instance (isolated from mixnet).
    uint32_t ecs_link_speed = params.speed * (params.gpus_per_server - params.alpha);
    g_prenet_ecs_underlay = new FatTreeTopology(
        fattree_node, memFromPkt(params.queuesize_pkts),
        NULL, &eventlist, NULL, params.ecs_qt, ecs_link_speed);
    cout << "[PRENET] ECS underlay fattree built (k=" << fattree_k
         << " link_speed=" << ecs_link_speed << " Mbps)" << endl;

    g_prenet_topo = new Prenet(
        gpu_num, memFromPkt(params.queuesize_pkts), eventlist, params.ocs_qt,
        timeFromUs((double)params.reconf_delay_us),
        g_prenet_ecs_underlay,
        params.alpha, params.dp_degree, params.tp_degree,
        params.pp_degree, params.ep_degree, params.gpus_per_server);
    g_topology = g_prenet_topo;

    // Config injection.
    g_prenet_cfg.variant_pool_k        = params.prenet_variant_k;
    g_prenet_cfg.probe_ratio           = params.prenet_probe_ratio;
    g_prenet_cfg.arbiter_window        = timeFromUs((double)params.prenet_arbiter_window_us);
    g_prenet_cfg.confidence_init       = params.prenet_confidence_init;
    g_prenet_cfg.confidence_max        = params.prenet_confidence_max;
    g_prenet_cfg.predictor_log_every   = params.prenet_predictor_log_every;
    g_prenet_cfg.link_speed_mbps       = params.speed;
    g_prenet_cfg.alpha                 = params.alpha;

    g_prenet_variants = new PrenetVariantPool(
        g_prenet_topo->region_size, g_prenet_topo->region_num,
        params.alpha, params.prenet_variant_k, /*seed=*/42);
    g_prenet_topo->variant_pool = g_prenet_variants;

    // Apply default variant (index 0) to every region.
    for (int r = 0; r < g_prenet_topo->region_num; r++) {
      g_prenet_topo->apply_variant(r, g_prenet_variants->default_variant(r));
    }

    g_prenet_arbiter = new PrenetArbiter(eventlist, g_prenet_cfg.arbiter_window);

    g_prenet_topomanager = new PrenetTopoManager(
        g_prenet_topo, g_prenet_arbiter,
        timeFromUs((double)params.reconf_delay_us), eventlist);
    g_prenet_topo->topomanager = g_prenet_topomanager;

    g_prenet_predictor = new PrenetPredictor(g_prenet_cfg, g_prenet_topo, g_prenet_ecs_underlay);

    // Dead-OCS-route self-heal (prenet version — guarded inside).
    TcpSrc::on_rtx_stuck = &reroute_flow_if_dead_prenet;

    cout << "[PRENET] topology created: region_size=" << g_prenet_topo->region_size
         << " region_num=" << g_prenet_topo->region_num
         << " variant_k=" << params.prenet_variant_k << endl;
  }
#endif

  // 4. Create TCP scanner and FCT output
  TcpRtxTimerScanner tcpRtxScanner(timeFromMs(1), eventlist);//作用:创建全局 TCP 重传定时器扫描器,每 1ms 遍历一次所有注册过的 TcpSrc,检查它们是否需要重传。
  //见mixnet-sim/mixnet-htsim/src/clos/tcp.h:612:
  g_tcp_scanner = &tcpRtxScanner;
  init_fct_output(fct_path);
  // 作用:打开一个全局输出文件流 g_fct_output,让每条 TCP 流完成时把自己的 FCT(Flow
  // Completion Time)记录追加进去。

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
    // Enable dead-OCS-route self-heal: on every RTO, check if this flow's route
    // crosses a dead queue and reroute to ECS. Backstop for finish_reconf-based
    // scanning when timing windows miss flows.
    TcpSrc::on_rtx_stuck = &reroute_flow_if_dead;
  }

  // 6. Initialize port numbers
  for (int i = 0; i < nodes_num; i++) {
    for (int j = 0; j < nodes_num; j++) {
      portNumber[i][j] = 10000;//tcp从10000端口开始分配，portNumber[i][j]表示从i到j的流使用的端口号，初始值为10000，之后每创建一个流就自增1，确保每条流的五元组唯一。
    }//代表机器对之前的通信,但是i代表GPU,只是同一个机器复用
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
  // 给每个 GPU 分配它所属的                        
  // NVSwitch(同机内的高带宽交换芯片)的 ID,并把 NVSwitch 
  // 当成"虚拟节点"加入编号空间 
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
        1,                    // comm_scale,集合通信的字节数缩放或者增加
        1,                    // compute_scale,把从 txt 读进来的三种 compute_time 全部乘一次
        1,                    // injection_scale per-event  的端点开销,用来近似 NIC / PCIe / 驱动栈的处理时间
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
      // Show network drain progress on stderr
      if (g_total_tcp_flows_created > 0) {
        int pct = (int)(100.0 * g_total_tcp_flows / g_total_tcp_flows_created);
        int bar_width = 40;
        int filled = bar_width * pct / 100;
        string bar(filled, '#');
        bar += string(bar_width - filled, '-');
        fprintf(stderr, "\r[%s] %3d%% flows: %lu/%lu  time: %.1f ms  events: %luM  ",
               bar.c_str(), pct,
               (unsigned long)g_total_tcp_flows, (unsigned long)g_total_tcp_flows_created,
               timeAsMs(eventlist.now()), (unsigned long)(event_count / 1000000));
        fflush(stderr);
      }
    }
  }
  fprintf(stderr, "\n");
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
    cout << "Reconfigs triggered: " << g_moe_reconfig_mgr.reconfig_triggered << endl;
    cout << "Reconfigs skipped:   " << g_moe_reconfig_mgr.reconfig_skipped << endl;
  }
  uint64_t total_flows = g_flow_count_ocs + g_flow_count_ecs + g_flow_count_nvlink;

  // Also count any flows still in the scanner (not yet finished)
  for (auto it = g_tcp_scanner->_tcps.begin(); it != g_tcp_scanner->_tcps.end(); ++it) {
    g_total_packets_sent += (*it)->_packets_sent;
    g_total_retransmissions += (*it)->_drops;
    g_total_tcp_flows++;
  }
  double retx_rate = (g_total_packets_sent > 0) ? (100.0 * g_total_retransmissions / g_total_packets_sent) : 0.0;

  cout << endl;
  cout << "======== Retransmission Statistics ========" << endl;
  cout << "RTO: " << params.rto_ms << " ms" << endl;
  cout << "Total packets sent: " << g_total_packets_sent << endl;
  cout << "Total retransmissions: " << g_total_retransmissions << endl;
  cout << "Retransmission rate: " << retx_rate << "%" << endl;
  cout << "TCP flows tracked: " << g_total_tcp_flows << endl;
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
      uint32_t ecs_link_speed = params.speed * (params.gpus_per_server - params.alpha);
      stats << "ECS link speed: " << ecs_link_speed << " Mbps" << endl;
      stats << "Reconf top-N: " << params.reconf_top_n << endl;
      stats << "Queue type (ECS): " << qt_to_string(params.ecs_qt) << endl;
      stats << "Queue type (OCS): " << qt_to_string(params.ocs_qt) << endl;
    } else if (g_topo_type == TOPO_FATTREE) {
      stats << "FatTree link speed: " << (params.speed * params.gpus_per_server) << " Mbps" << endl;
      stats << "FatTree nodes: " << fattree_node << "  K=" << fattree_k << endl;
      stats << "Queue type: " << qt_to_string(params.ecs_qt) << endl;
    } else if (g_topo_type == TOPO_OS_FATTREE || g_topo_type == TOPO_AGG_OS_FATTREE) {
      stats << "K=" << fattree_k << "  OS ratio=" << params.os_ratio << endl;
      stats << "Queue type: " << qt_to_string(params.ecs_qt) << endl;
    } else {
      stats << "Nodes: " << num_machines << endl;
      stats << "Queue type: " << qt_to_string(params.ecs_qt) << endl;
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
      stats << "Reconfigs triggered: " << g_moe_reconfig_mgr.reconfig_triggered << endl;
      stats << "Reconfigs skipped:   " << g_moe_reconfig_mgr.reconfig_skipped << endl;
    }
#ifdef PRENET_ENABLED
    if (g_topo_type == TOPO_PRENET) {
      stats << endl;
      stats << "======== Prenet Stats ========" << endl;
      stats << "prenet_predictions_total: " << g_prenet_predictions_total << endl;
      stats << "prenet_predictions_correct: " << g_prenet_predictions_correct << endl;
      stats << "prenet_predictions_wrong: " << g_prenet_predictions_wrong << endl;
      stats << "prenet_action_stay_ecs: " << g_prenet_action_stay_ecs << endl;
      stats << "prenet_action_use_ocs_asis: " << g_prenet_action_use_ocs_asis << endl;
      stats << "prenet_action_reconfig_ocs: " << g_prenet_action_reconfig_ocs << endl;
      stats << "prenet_probes_emitted: " << g_prenet_probes_emitted << endl;
      stats << "prenet_arbiter_wins: " << g_prenet_arbiter_wins << endl;
      stats << "prenet_arbiter_losses: " << g_prenet_arbiter_losses << endl;
      uint64_t total = g_prenet_predictions_correct + g_prenet_predictions_wrong;
      if (total > 0) {
        stats << "prenet_accuracy: " << (100.0 * g_prenet_predictions_correct / total) << "%" << endl;
      }
    }
#endif
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

    // Per-pass a2a OCS hit stats (mixnet only). OCS share == "a2a traffic
    // covered by the prediction-driven OCS configuration". pass=0 is cold
    // (no prediction available), so the "excluding pass 0" line is the
    // steady-state hit rate.
    if (g_topo_type == TOPO_MIXNET) {
      std::set<int> all_passes;
      for (const auto& kv : g_a2a_flow_count_ocs_by_pass) all_passes.insert(kv.first);
      for (const auto& kv : g_a2a_flow_count_ecs_by_pass) all_passes.insert(kv.first);
      for (const auto& kv : g_a2a_bytes_ocs_by_pass)      all_passes.insert(kv.first);
      for (const auto& kv : g_a2a_bytes_ecs_by_pass)      all_passes.insert(kv.first);

      if (!all_passes.empty()) {
        auto getv = [](const std::map<int,uint64_t>& m, int k)->uint64_t {
          auto it = m.find(k); return it == m.end() ? 0 : it->second;
        };
        stats << endl;
        stats << "======== A2A Prediction Hit (OCS) - Per Pass ========" << endl;
        stats << "  OCS share = a2a bytes routed over OCS / all cross-machine a2a bytes" << endl;
        stats << "  pass=0 is cold (no prediction available); pass>=1 reflects prediction hit rate" << endl;

        uint64_t tot_f_ocs = 0, tot_f_ecs = 0, tot_b_ocs = 0, tot_b_ecs = 0;
        uint64_t tot_f_ocs_np0 = 0, tot_f_ecs_np0 = 0, tot_b_ocs_np0 = 0, tot_b_ecs_np0 = 0;

        for (int p : all_passes) {
          uint64_t f_ocs = getv(g_a2a_flow_count_ocs_by_pass, p);
          uint64_t f_ecs = getv(g_a2a_flow_count_ecs_by_pass, p);
          uint64_t b_ocs = getv(g_a2a_bytes_ocs_by_pass, p);
          uint64_t b_ecs = getv(g_a2a_bytes_ecs_by_pass, p);
          uint64_t f_tot = f_ocs + f_ecs;
          uint64_t b_tot = b_ocs + b_ecs;
          double f_share = (f_tot > 0) ? (100.0 * f_ocs / f_tot) : 0.0;
          double b_share = (b_tot > 0) ? (100.0 * b_ocs / b_tot) : 0.0;

          std::string label = (p < 0) ? "unstamped" : ("pass=" + std::to_string(p));
          std::string note  = (p == 0) ? "  (cold: no prediction)" : "";
          stats << "  " << label
                << ": a2a_flows=" << f_tot
                << " ocs=" << f_ocs << " ecs=" << f_ecs
                << " | bytes=" << (b_tot / 1048576.0) << "MB"
                << " ocs=" << (b_ocs / 1048576.0) << "MB"
                << " ecs=" << (b_ecs / 1048576.0) << "MB"
                << " | ocs_flow_share=" << f_share << "%"
                << " ocs_bytes_share=" << b_share << "%"
                << note << endl;

          tot_f_ocs += f_ocs; tot_f_ecs += f_ecs;
          tot_b_ocs += b_ocs; tot_b_ecs += b_ecs;
          if (p >= 1) {
            tot_f_ocs_np0 += f_ocs; tot_f_ecs_np0 += f_ecs;
            tot_b_ocs_np0 += b_ocs; tot_b_ecs_np0 += b_ecs;
          }
        }

        uint64_t f_tot = tot_f_ocs + tot_f_ecs;
        uint64_t b_tot = tot_b_ocs + tot_b_ecs;
        double f_share = (f_tot > 0) ? (100.0 * tot_f_ocs / f_tot) : 0.0;
        double b_share = (b_tot > 0) ? (100.0 * tot_b_ocs / b_tot) : 0.0;
        stats << "  ----------------------------------------------------" << endl;
        stats << "  Total: a2a_flows=" << f_tot
              << " ocs=" << tot_f_ocs << " ecs=" << tot_f_ecs
              << " | bytes=" << (b_tot / 1048576.0) << "MB"
              << " ocs=" << (tot_b_ocs / 1048576.0) << "MB"
              << " ecs=" << (tot_b_ecs / 1048576.0) << "MB"
              << " | ocs_flow_share=" << f_share << "%"
              << " ocs_bytes_share=" << b_share << "%" << endl;

        uint64_t f_tot_np0 = tot_f_ocs_np0 + tot_f_ecs_np0;
        uint64_t b_tot_np0 = tot_b_ocs_np0 + tot_b_ecs_np0;
        if (f_tot_np0 > 0 || b_tot_np0 > 0) {
          double f_share_np0 = (f_tot_np0 > 0) ? (100.0 * tot_f_ocs_np0 / f_tot_np0) : 0.0;
          double b_share_np0 = (b_tot_np0 > 0) ? (100.0 * tot_b_ocs_np0 / b_tot_np0) : 0.0;
          stats << "  Total (excl. pass 0): a2a_flows=" << f_tot_np0
                << " | bytes=" << (b_tot_np0 / 1048576.0) << "MB"
                << " | ocs_flow_share=" << f_share_np0 << "%"
                << " ocs_bytes_share=" << b_share_np0 << "%" << endl;
        }
      }
    }

    // Per-ComType traffic breakdown
    {
      uint64_t total_ct_bytes = 0, total_ct_flows = 0;
      for (int i = 0; i < G_COMTYPE_N; i++) {
        total_ct_bytes += g_bytes_by_comtype[i];
        total_ct_flows += g_flows_by_comtype[i];
      }
      stats << endl;
      stats << "======== Traffic by Collective Type ========" << endl;
      for (int i = 0; i < G_COMTYPE_N; i++) {
        if (g_flows_by_comtype[i] == 0 && g_bytes_by_comtype[i] == 0) continue;
        double mb = g_bytes_by_comtype[i] / 1048576.0;
        double pct = (total_ct_bytes > 0) ? (100.0 * g_bytes_by_comtype[i] / total_ct_bytes) : 0.0;
        stats << "  " << comtype_name(i)
              << ": flows=" << g_flows_by_comtype[i]
              << " bytes=" << g_bytes_by_comtype[i]
              << " (" << mb << " MB, " << pct << "%)" << endl;
      }
      cout << endl;
      cout << "======== Traffic by Collective Type ========" << endl;
      for (int i = 0; i < G_COMTYPE_N; i++) {
        if (g_flows_by_comtype[i] == 0 && g_bytes_by_comtype[i] == 0) continue;
        double mb = g_bytes_by_comtype[i] / 1048576.0;
        double pct = (total_ct_bytes > 0) ? (100.0 * g_bytes_by_comtype[i] / total_ct_bytes) : 0.0;
        cout << "  " << comtype_name(i)
             << ": flows=" << g_flows_by_comtype[i]
             << " bytes=" << g_bytes_by_comtype[i]
             << " (" << mb << " MB, " << pct << "%)" << endl;
      }
    }

    stats << endl;
    stats << "======== Retransmission Statistics ========" << endl;
    stats << "RTO: " << params.rto_ms << " ms" << endl;
    stats << "Total packets sent: " << g_total_packets_sent << endl;
    stats << "Total retransmissions: " << g_total_retransmissions << endl;
    stats << "Retransmission rate: " << retx_rate << "%" << endl;
    stats << "TCP flows tracked: " << g_total_tcp_flows << endl;

    stats << endl;
    stats << "======== Timing ========" << endl;
    stats << "Total events: " << event_count << endl;
    stats << "Final sim time: " << timeAsMs(eventlist.now()) << " ms" << endl;
    if (!g_pass_end_ms.empty()) {
      stats << endl;
      stats << "======== Per-Pass Timing ========" << endl;
      double prev = 0.0;
      double sum = 0.0;
      double minv = 1e18, maxv = -1e18;
      for (size_t i = 0; i < g_pass_end_ms.size(); i++) {
        double dur = g_pass_end_ms[i] - prev;
        stats << "Pass " << (i + 1) << ": " << dur << " ms (end "
              << g_pass_end_ms[i] << " ms)" << endl;
        sum += dur;
        if (dur < minv) minv = dur;
        if (dur > maxv) maxv = dur;
        prev = g_pass_end_ms[i];
      }
      double avg = sum / g_pass_end_ms.size();
      stats << "Avg per pass: " << avg << " ms  (min " << minv
            << ", max " << maxv << ")" << endl;
    }
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

#ifdef PRENET_ENABLED
  delete g_prenet_predictor;    g_prenet_predictor    = nullptr;
  delete g_prenet_topomanager;  g_prenet_topomanager  = nullptr;
  delete g_prenet_arbiter;      g_prenet_arbiter      = nullptr;
  delete g_prenet_variants;     g_prenet_variants     = nullptr;
  delete g_prenet_topo;         g_prenet_topo         = nullptr;
  delete g_prenet_ecs_underlay; g_prenet_ecs_underlay = nullptr;
#endif

  return 0;
}
