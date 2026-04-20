// prenet.h — prenet topology (isolated from mixnet)
// See predict-plan/prenet_principles.md and prenet_plan.md
#ifndef PRENET_H
#define PRENET_H
#ifdef PRENET_ENABLED

#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include "main.h"
#include "pipe.h"
#include "config.h"
#include "network.h"
#include "eventlist.h"
#include "topology.h"
#include "switch.h"
#include "fat_tree_topology.h"

#ifndef QT
#define QT
typedef enum {RANDOM, ECN, COMPOSITE, CTRL_PRIO, LOSSLESS, LOSSLESS_INPUT, LOSSLESS_INPUT_ECN} queue_type;
#endif

class PrenetVariantPool;   // fwd
class PrenetTopoManager;   // fwd
struct ConnVariant;        // fwd

// Prenet topology: OCS direct links (conn matrix) + an ECS underlay fattree.
// Structurally parallel to Mixnet but NOT a subclass / not sharing types.
class Prenet : public Topology {
public:
  Prenet(int no_of_gpus, mem_b queuesize, EventList& ev, queue_type qt,
         simtime_picosec reconf_delay, FatTreeTopology* elec_topology,
         int alpha, int dp, int tp, int pp, int ep, int gpus_per_node);
  ~Prenet();

  // Topology interface
  std::vector<const Route*>* get_paths(int src_gpu_idx, int dest_gpu_idx) override;
  std::vector<const Route*>* get_eps_paths(int src_gpu_idx, int dest_gpu_idx) override;
  std::vector<int>* get_neighbours(int src) override { return nullptr; }
  int no_of_nodes() const override { return _no_of_nodes; }

  // Prenet-specific
  void apply_variant(int region_id, const ConnVariant& v);
  bool circuit_exists(int src_node, int dst_node) const;
  uint64_t get_link_bitrate(int src_node, int dst_node) const;

  // Data
  int _no_of_nodes;
  std::vector<std::vector<Pipe*>> pipes;
  std::vector<std::vector<Queue*>> queues;
  std::vector<Switch*> switchs;
  FatTreeTopology* elec_topology;   // ECS underlay (owned outside)
  int alpha, dp_degree, tp_degree, pp_degree, ep_degree, gpus_per_node;
  int region_size, region_num;
  simtime_picosec reconf_delay;
  EventList& eventlist;

  PrenetVariantPool* variant_pool = nullptr;  // weak ref
  PrenetTopoManager* topomanager  = nullptr;  // weak ref

private:
  void set_params(int no_of_gpus);
  void init_network();
  void renew_routes(int start_node, int end_node);
  Queue* alloc_queue(uint64_t speed_mbps, mem_b queuesize);

  std::unordered_map<uint64_t, std::vector<std::vector<size_t>*>> _routes;
  mem_b _queuesize;
  queue_type _qt;
};

#endif // PRENET_ENABLED
#endif // PRENET_H
