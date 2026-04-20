// prenet_arbiter.h — window-based reconfig arbiter.
#ifndef PRENET_ARBITER_H
#define PRENET_ARBITER_H
#ifdef PRENET_ENABLED

#include "eventlist.h"
#include "prenet_variant_pool.h"
#include <functional>
#include <vector>
#include <cstdint>

class PrenetArbiter : public EventSource {
public:
  using Callback = std::function<void(bool granted, simtime_picosec end_time)>;

  PrenetArbiter(EventList& ev, simtime_picosec window);

  // Submit reconfig request. After `window` picoseconds elapsed since the
  // first pending submit, arbitration runs: for each region, the highest
  // (confidence, msg_size) wins; others receive cb(false, 0).
  // `grant_fn(region_id, variant_id)` is called on the winner to actually
  // kick off the reconfig and returns the scheduled reconfig_end_time.
  void submit(int region_id, int variant_id,
              double confidence, uint64_t msg_size,
              Callback cb);

  // grant_fn provided once at wire time by PrenetTopoManager.
  void set_grant_fn(std::function<simtime_picosec(int region_id, int variant_id)> fn) {
    _grant_fn = std::move(fn);
  }

  void doNextEvent() override;

private:
  struct Pending {
    int region_id;
    int variant_id;
    double confidence;
    uint64_t msg_size;
    Callback cb;
  };
  std::vector<Pending> _queue;
  bool _scheduled = false;
  simtime_picosec _window;
  std::function<simtime_picosec(int,int)> _grant_fn;
};

#endif // PRENET_ENABLED
#endif // PRENET_ARBITER_H
