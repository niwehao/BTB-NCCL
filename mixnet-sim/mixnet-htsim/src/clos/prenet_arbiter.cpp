// prenet_arbiter.cpp
#ifdef PRENET_ENABLED
#include "prenet_arbiter.h"
#include <algorithm>
#include <unordered_map>
#include <iostream>

PrenetArbiter::PrenetArbiter(EventList& ev, simtime_picosec window)
  : EventSource(ev, "PrenetArbiter"), _window(window) {}

void PrenetArbiter::submit(int region_id, int variant_id,
                           double confidence, uint64_t msg_size,
                           Callback cb) {
  _queue.push_back({region_id, variant_id, confidence, msg_size, std::move(cb)});
  if (!_scheduled) {
    _scheduled = true;
    eventlist().sourceIsPendingRel(*this, _window);
  }
}

void PrenetArbiter::doNextEvent() {
  _scheduled = false;
  if (_queue.empty()) return;

  // Move queue aside so newly submitted during callbacks go to next window.
  std::vector<Pending> batch;
  batch.swap(_queue);

  // Bucket by region, pick the best (confidence, msg_size) per region.
  std::unordered_map<int, int> winner_idx;  // region_id -> idx in batch
  for (int i = 0; i < (int)batch.size(); i++) {
    const Pending& p = batch[i];
    auto it = winner_idx.find(p.region_id);
    if (it == winner_idx.end()) { winner_idx[p.region_id] = i; continue; }
    const Pending& cur = batch[it->second];
    // Higher confidence wins; tie-break by msg_size.
    if (p.confidence > cur.confidence ||
        (p.confidence == cur.confidence && p.msg_size > cur.msg_size)) {
      it->second = i;
    }
  }

  // Grant winners, reject losers.
  for (int i = 0; i < (int)batch.size(); i++) {
    Pending& p = batch[i];
    auto it = winner_idx.find(p.region_id);
    bool is_winner = (it != winner_idx.end() && it->second == i);
    if (!is_winner) {
      if (p.cb) p.cb(false, 0);
      continue;
    }
    if (!_grant_fn) {
      // Wired wrong — safest is to reject.
      std::cerr << "[PRENET_ARB] WARN: no grant_fn wired; rejecting winner" << std::endl;
      if (p.cb) p.cb(false, 0);
      continue;
    }
    simtime_picosec end_time = _grant_fn(p.region_id, p.variant_id);
    if (end_time == 0) {
      if (p.cb) p.cb(false, 0);
    } else {
      if (p.cb) p.cb(true, end_time);
    }
  }
}

#endif // PRENET_ENABLED
