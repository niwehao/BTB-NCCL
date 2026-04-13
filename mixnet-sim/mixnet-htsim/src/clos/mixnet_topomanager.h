#ifndef MIXNET_TOPO_MANAGER_H
#define MIXNET_TOPO_MANAGER_H
// #ifndef FLAT_TOPO
// #define FLAT_TOPO
#include <iostream>
#include <unordered_map>
#include "main.h"
#include "randomqueue.h"
#include <queue>
#include "pipe.h"
#include "config.h"
#include "loggers.h"
#include "network.h"
#include "firstfit.h"
#include "topology.h"
#include "logfile.h"
#include "eventlist.h"
#include "switch.h"
#include "mixnet.h"
#include <algorithm>
#include <random>
#include <cstdint>
#include "dyn_net_sch.h"
#include <numeric> 
#include "ecnqueue.h"
#include "queue_lossless.h"
#include "tcp.h"
#ifndef QT
#define QT
typedef enum {RANDOM, ECN, COMPOSITE, CTRL_PRIO, LOSSLESS, LOSSLESS_INPUT, LOSSLESS_INPUT_ECN} queue_type;
#endif
using namespace std;


class TcpRtxTimerScanner;

struct All2AllTrafficRecorder {
  All2AllTrafficRecorder(int layer_num, int region_num, int region_size, TcpRtxTimerScanner * rtx_scanner);
  void append_traffic_matrix(int layer_id, int region_id, Matrix2D<double> & tm);
  bool check_traffic_matrix(int layer_id, int region_id);
  TcpRtxTimerScanner * rtx_scanner;
  std::vector<std::vector<Matrix2D<double>* >> traffic_matrix; // [layer, region, matrix]
  int layer_num;
  int region_num;
  int region_size;
};


class RegionalTopoManager : public EventSource {
  // regional topology reconfiguration
public:
  enum TopoStatus {
    TOPO_INIT,
    TOPO_LIVE,
    TOPO_RECONF
  };
  RegionalTopoManager(Mixnet* topo, All2AllTrafficRecorder* demandrecorder, simtime_picosec refonc_delay, EventList & eventlist, int region_id);
  virtual void doNextEvent();
  void start_reconf();
  void finish_reconf();
  void do_reconf();
  void _do_reconf();
  // for regional reconf
  void update_regional_queue_bandwidth();
  void set_regional_queues_pause_recved();
  void set_regional_tcp_pause();
  void update_regional_route();
  void resume_regional_tcp_flows();
  void normalize_tm(Matrix2D<double> & normal_tm);
  bool is_in_region(TcpSrc* tcpsrc) const;
  std::vector<std::vector<int>> regional_topo_reconfig();
  int nnodes;
  int region_size;
  int region_num;
  int alpha;
  int start_node;
  int end_node;
  Mixnet* topo;
  int current_layer_id=-1;
  int current_micro_batch_id=-1;
  int region_id;
  simtime_picosec reconf_delay;
  simtime_picosec reconfig_end_time;
  TopoStatus status;
  EventList & eventlist;
  All2AllTrafficRecorder * demandrecorder;
  uint64_t non_empty_queues;
  std::vector<std::vector<int>> reconfig_conn_matrix;  // Store the conn matrix computed in start_reconf
};

class MixnetTopoManager {
public:
  MixnetTopoManager(Mixnet* topo, All2AllTrafficRecorder* demandrecorder, simtime_picosec refonc_delay, EventList & eventlist);
  vector<RegionalTopoManager*> regional_topo_managers;
  int nnodes;
  int region_size;
  int region_num;
  int alpha;
  Mixnet* topo;
  simtime_picosec reconf_delay;
  All2AllTrafficRecorder * demandrecorder;
};

#endif // MIXNET_TOPO_MANAGER_H