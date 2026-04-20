// prenet_topomanager.cpp
#ifdef PRENET_ENABLED
#include "prenet_topomanager.h"
#include <iostream>
#include <cassert>

PrenetRegionalManager::PrenetRegionalManager(Prenet* topo, simtime_picosec reconf_delay,
                                             EventList& eventlist, int region_id)
  : EventSource(eventlist, "PrenetRegionalManager"),
    _topo(topo), _reconf_delay(reconf_delay), _region_id(region_id) {}

simtime_picosec PrenetRegionalManager::start_reconf(const ConnVariant& target) {
  if (_status == RECONF) {
    // Already in progress — can't accept another.
    return 0;
  }
  _pending_target = target;
  // Apply the new conn immediately (pause semantics are skipped — mimics
  // mixnet's immediate-reconfig mode where running flows stay on old routes
  // until they drain, new flows see the new conn at SendFlowPrenet time which
  // is gated behind reconfig_end_time defer).
  _topo->apply_variant(_region_id, target);
  _status = RECONF;
  _reconfig_end_time = eventlist().now() + _reconf_delay + 1;
  eventlist().sourceIsPendingRel(*this, _reconf_delay);
  return _reconfig_end_time;
}

void PrenetRegionalManager::doNextEvent() {
  finish_reconf();
}

void PrenetRegionalManager::finish_reconf() {
  _status = LIVE;
  _reconfig_end_time = 0;
  // Queues with _bitrate > 0 are READY; those with 0 remain unusable (get_paths
  // returns empty → caller falls back to ECS).
}

PrenetTopoManager::PrenetTopoManager(Prenet* topo, PrenetArbiter* arbiter,
                                     simtime_picosec reconf_delay,
                                     EventList& eventlist)
  : _topo(topo), _arbiter(arbiter), _reconf_delay(reconf_delay),
    _eventlist(eventlist) {
  int rc = topo->region_num;
  _regional_managers.reserve(rc);
  for (int i = 0; i < rc; i++) {
    _regional_managers.push_back(new PrenetRegionalManager(topo, reconf_delay, eventlist, i));
  }
  if (_arbiter) {
    _arbiter->set_grant_fn([this](int region_id, int variant_id) -> simtime_picosec {
      if (region_id < 0 || region_id >= (int)_regional_managers.size()) return 0;
      const ConnVariant* v = _topo->variant_pool
          ? _topo->variant_pool->get(region_id, variant_id)
          : nullptr;
      if (!v) return 0;
      PrenetRegionalManager* rm = _regional_managers[region_id];
      if (rm->status() == PrenetRegionalManager::RECONF) return 0;
      return rm->start_reconf(*v);
    });
  }
}

void PrenetTopoManager::request_reconfig(int region_id, int variant_id,
                                         double confidence, uint64_t msg_size,
                                         PrenetArbiter::Callback cb) {
  if (_arbiter == nullptr) {
    if (cb) cb(false, 0);
    return;
  }
  // If already reconfiguring, reject early to avoid stacking.
  if (region_id >= 0 && region_id < (int)_regional_managers.size()) {
    PrenetRegionalManager* rm = _regional_managers[region_id];
    if (rm->status() == PrenetRegionalManager::RECONF) {
      if (cb) cb(false, 0);
      return;
    }
  }
  _arbiter->submit(region_id, variant_id, confidence, msg_size, std::move(cb));
}

simtime_picosec PrenetTopoManager::region_reconfig_end(int region_id) const {
  if (region_id < 0 || region_id >= (int)_regional_managers.size()) return 0;
  return _regional_managers[region_id]->reconfig_end_time();
}

#endif // PRENET_ENABLED
