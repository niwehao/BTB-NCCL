// prenet.cpp — prenet topology implementation (isolated from mixnet)
#ifdef PRENET_ENABLED
#include "prenet.h"
#include "prenet_variant_pool.h"
#include "compositequeue.h"
#include "ecnqueue.h"
#include "queue_lossless.h"
#include "queue_lossless_input.h"
#include "queue_lossless_output.h"
#include "randomqueue.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cassert>

extern uint32_t RTT;
extern uint32_t SPEED;

using std::vector;
using std::endl;
using std::cerr;

// Declared elsewhere in htsim
extern std::string ntoa(double);

static void check_non_null_ocs_prenet(Route *rt) {
  for (unsigned int i = 1; i < rt->size() - 1; i += 2) {
    if (rt->at(i) == NULL) {
      std::cerr << "[PRENET][FATAL] null queue in OCS route" << std::endl;
      assert(0);
    }
  }
}

void Prenet::set_params(int no_of_gpus) {
  _no_of_nodes = no_of_gpus / gpus_per_node;
  switchs.resize(_no_of_nodes, nullptr);
  pipes.resize(_no_of_nodes, vector<Pipe*>(_no_of_nodes, nullptr));
  queues.resize(_no_of_nodes, vector<Queue*>(_no_of_nodes, nullptr));
  region_size = ep_degree * tp_degree / gpus_per_node;
  if (region_size <= 0) region_size = 1;
  region_num = _no_of_nodes / region_size;
  if (region_num <= 0) region_num = 1;
}

Prenet::Prenet(int no_of_gpus, mem_b queuesize, EventList& ev, queue_type qt,
               simtime_picosec reconf_delay_, FatTreeTopology* elec_topology_,
               int alpha_, int dp, int tp, int pp, int ep, int gpus_per_node_)
  : elec_topology(elec_topology_), alpha(alpha_),
    dp_degree(dp), tp_degree(tp), pp_degree(pp), ep_degree(ep),
    gpus_per_node(gpus_per_node_), reconf_delay(reconf_delay_),
    eventlist(ev), _queuesize(queuesize), _qt(qt) {
  set_params(no_of_gpus);
  conn = std::vector<std::vector<int>>(_no_of_nodes, std::vector<int>(_no_of_nodes, 0));
  init_network();
  cerr << "[PRENET] topology built: nodes=" << _no_of_nodes
       << " region_size=" << region_size << " region_num=" << region_num
       << " alpha=" << alpha << endl;
}

Prenet::~Prenet() {
  // Release the pre-populated 1-hop path templates.
  for (auto& kv : _routes) {
    for (std::vector<size_t>* v : kv.second) delete v;
  }
  _routes.clear();
  // Queues/Pipes/Switches are owned by the event system and freed elsewhere;
  // mirrors Mixnet's non-ownership convention.
}

Queue* Prenet::alloc_queue(uint64_t speed_mbps, mem_b queuesize) {
  // Prenet only supports queue types that don't require external switch wiring
  // (ECN/COMPOSITE/LOSSLESS_INPUT*). LOSSLESS needs Switch* + setRemoteEndpoint
  // pairing which mixnet does in init_network; prenet doesn't replicate that
  // and would deadlock if requested.
  if (_qt == ECN) {
    return new ECNQueue(speedFromMbps(speed_mbps), memFromPkt(queuesize),
                        eventlist, nullptr, memFromPkt(50));
  } else if (_qt == COMPOSITE) {
    return new CompositeQueue(speedFromMbps(speed_mbps), queuesize, eventlist, nullptr);
  } else if (_qt == LOSSLESS_INPUT || _qt == LOSSLESS_INPUT_ECN) {
    return new LosslessOutputQueue(speedFromMbps(speed_mbps), memFromPkt(200), eventlist, nullptr);
  } else if (_qt == RANDOM) {
    return new RandomQueue(speedFromMbps(speed_mbps),
                           memFromPkt(SWITCH_BUFFER + RANDOM_BUFFER),
                           eventlist, nullptr, memFromPkt(RANDOM_BUFFER));
  }
  // Reject LOSSLESS (and any other unsupported qt) loudly rather than build a
  // half-wired queue that will hang.
  std::cerr << "[PRENET][FATAL] alloc_queue: queue type " << (int)_qt
            << " not supported by prenet" << std::endl;
  assert(0 && "prenet does not support LOSSLESS queue type");
  return nullptr;
}

void Prenet::init_network() {
  // Start all links with zero bitrate; apply_variant() will populate.
  for (int j = 0; j < _no_of_nodes; j++) {
    for (int k = 0; k <= j; k++) {
      if (j == k) continue;
      queues[j][k] = alloc_queue(SPEED, _queuesize);
      queues[k][j] = alloc_queue(SPEED, _queuesize);
      queues[j][k]->setName("PreL" + ntoa(j) + "->DST" + ntoa(k));
      queues[k][j]->setName("PreL" + ntoa(k) + "->DST" + ntoa(j));
      pipes[j][k] = new Pipe(timeFromNs(RTT), eventlist);
      pipes[k][j] = new Pipe(timeFromNs(RTT), eventlist);
      pipes[j][k]->setName("PrePipe-LS" + ntoa(j) + "->DST" + ntoa(k));
      pipes[k][j]->setName("PrePipe-LS" + ntoa(k) + "->DST" + ntoa(j));
      // Start all links with bitrate=0 (no OCS circuit yet).
      queues[j][k]->_bitrate = 0;
      queues[k][j]->_bitrate = 0;
      queues[j][k]->_ps_per_byte = 0;
      queues[k][j]->_ps_per_byte = 0;
    }
  }
  // Populate _routes with direct 1-hop paths for every (i,j), i != j.
  renew_routes(0, _no_of_nodes);
}

void Prenet::renew_routes(int start_node, int end_node) {
  for (int i = start_node; i < end_node; i++) {
    for (int j = start_node; j < end_node; j++) {
      if (i == j) continue;
      uint64_t idx = (uint64_t)i * _no_of_nodes + j;
      vector<size_t>* pv = new vector<size_t>();
      pv->push_back(i);
      pv->push_back(j);
      auto it = _routes.find(idx);
      if (it == _routes.end()) {
        _routes[idx] = vector<vector<size_t>*>();
      } else {
        for (vector<size_t>* r : it->second) delete r;
        it->second.clear();
      }
      _routes[idx].push_back(pv);
    }
  }
}

std::vector<const Route*>* Prenet::get_paths(int src_gpu_idx, int dest_gpu_idx) {
  int src = src_gpu_idx / gpus_per_node;
  int dest = dest_gpu_idx / gpus_per_node;
  if (src == dest) return new vector<const Route*>();
  if (conn[src][dest] <= 0) {
    // No OCS circuit; caller must fall back to ECS.
    return new vector<const Route*>();
  }
  auto it = _routes.find((uint64_t)src * _no_of_nodes + dest);
  if (it == _routes.end()) return new vector<const Route*>();

  auto* paths = new vector<const Route*>();
  for (const vector<size_t>* r : it->second) {
    const vector<size_t>& route = *r;
    Route* routeout = new Route();
    Route* routeback = new Route();
    bool dead = false;
    for (size_t i = 0; i < route.size() - 1; i++) {
      Queue* q = queues[route[i]][route[i+1]];
      if (q == nullptr || q->_bitrate == 0) { dead = true; break; }
      routeout->push_back(q);
      routeout->push_back(pipes[route[i]][route[i+1]]);
    }
    if (dead) { delete routeout; delete routeback; continue; }
    for (size_t i = route.size() - 1; i > 0; i--) {
      Queue* q = queues[route[i]][route[i-1]];
      if (q == nullptr || q->_bitrate == 0) { dead = true; break; }
      routeback->push_back(q);
      routeback->push_back(pipes[route[i]][route[i-1]]);
    }
    if (dead) { delete routeout; delete routeback; continue; }
    routeout->set_reverse(routeback);
    routeback->set_reverse(routeout);
    check_non_null_ocs_prenet(routeout);
    paths->push_back(routeout);
  }
  return paths;
}

std::vector<const Route*>* Prenet::get_eps_paths(int src_gpu_idx, int dest_gpu_idx) {
  int src = src_gpu_idx / gpus_per_node;
  int dest = dest_gpu_idx / gpus_per_node;
  if (elec_topology == nullptr) return new vector<const Route*>();
  // FatTreeTopology::get_paths may return nullptr for disconnected pairs; our
  // callers check !empty() but would NPE on nullptr. Wrap to always-vector.
  auto* p = elec_topology->get_paths(src, dest);
  return p != nullptr ? p : new vector<const Route*>();
}

void Prenet::apply_variant(int region_id, const ConnVariant& v) {
  int rs = region_size;
  int start = region_id * rs;
  // Write conn for this region.
  for (int i = 0; i < rs; i++) {
    for (int j = 0; j < rs; j++) {
      if (i == j) continue;
      int val = (int)v.conn_local[i][j];
      // v0 invariant: variants are 0/1 matrices (a single direct OCS circuit
      // per pair). predictor.cpp's counterfactual math (ocs_bw_bps = link_bps)
      // relies on this. If we ever move to alpha-accumulating variants,
      // update prenet_predictor.cpp accordingly.
      assert(val == 0 || val == 1);
      conn[start + i][start + j] = val;
    }
  }
  // Update queue bitrates accordingly.
  for (int i = 0; i < rs; i++) {
    for (int j = 0; j < rs; j++) {
      if (i == j) continue;
      int c = conn[start + i][start + j];
      uint64_t new_bitrate = (c > 0) ? speedFromMbps((uint64_t)SPEED * (uint64_t)c) : 0;
      Queue* q = queues[start + i][start + j];
      if (q == nullptr) continue;
      q->_bitrate = new_bitrate;
      q->_ps_per_byte = (new_bitrate > 0)
                          ? (simtime_picosec)((std::pow(10.0, 12.0) * 8) / new_bitrate)
                          : 0;
    }
  }
  // 1-hop direct routes don't need recomputation beyond renew_routes, already
  // populated in init_network. Keep them.
}

bool Prenet::circuit_exists(int src_node, int dst_node) const {
  if (src_node < 0 || dst_node < 0 || src_node >= _no_of_nodes || dst_node >= _no_of_nodes)
    return false;
  return conn[src_node][dst_node] > 0;
}

uint64_t Prenet::get_link_bitrate(int src_node, int dst_node) const {
  if (src_node < 0 || dst_node < 0 || src_node >= _no_of_nodes || dst_node >= _no_of_nodes)
    return 0;
  Queue* q = queues[src_node][dst_node];
  return q ? q->_bitrate : 0;
}

#endif // PRENET_ENABLED
