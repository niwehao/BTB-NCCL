// prenet_topomanager.h — prenet-owned reconfig state machine per region.
// Intentionally DOES NOT include or derive from mixnet_topomanager.h (principles §9).
#ifndef PRENET_TOPOMANAGER_H
#define PRENET_TOPOMANAGER_H
#ifdef PRENET_ENABLED

#include "eventlist.h"
#include "prenet.h"
#include "prenet_variant_pool.h"
#include "prenet_arbiter.h"
#include <vector>
#include <functional>
#include <cstdint>

class PrenetRegionalManager : public EventSource {
public:
  enum Status { LIVE, RECONF };

  PrenetRegionalManager(Prenet* topo, simtime_picosec reconf_delay,
                        EventList& eventlist, int region_id);

  // Kick off a reconfig targeting `variant`. Returns reconfig_end_time (abs).
  // Returns 0 if region is busy.
  simtime_picosec start_reconf(const ConnVariant& target);

  // EventSource: fires at reconfig_end_time, finalises.
  void doNextEvent() override;

  int region_id() const { return _region_id; }
  Status status() const { return _status; }
  simtime_picosec reconfig_end_time() const { return _reconfig_end_time; }

private:
  void finish_reconf();

  Prenet* _topo;
  simtime_picosec _reconf_delay;
  int _region_id;
  Status _status = LIVE;
  simtime_picosec _reconfig_end_time = 0;
  ConnVariant _pending_target;
};

class PrenetTopoManager {
public:
  PrenetTopoManager(Prenet* topo, PrenetArbiter* arbiter,
                    simtime_picosec reconf_delay, EventList& eventlist);

  // Called by predictor path to request a reconfig via the arbiter.
  void request_reconfig(int region_id, int variant_id,
                        double confidence, uint64_t msg_size,
                        PrenetArbiter::Callback cb);

  int region_count() const { return (int)_regional_managers.size(); }
  PrenetRegionalManager* region(int rid) { return _regional_managers[rid]; }

  // Query: is region `rid` currently in reconfig window? Returns 0 if LIVE.
  simtime_picosec region_reconfig_end(int region_id) const;

private:
  Prenet* _topo;
  PrenetArbiter* _arbiter;
  simtime_picosec _reconf_delay;
  EventList& _eventlist;
  std::vector<PrenetRegionalManager*> _regional_managers;
};

#endif // PRENET_ENABLED
#endif // PRENET_TOPOMANAGER_H
