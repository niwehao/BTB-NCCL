// prenet_variant_pool.h — precomputed OCS conn variants per region.
#ifndef PRENET_VARIANT_POOL_H
#define PRENET_VARIANT_POOL_H
#ifdef PRENET_ENABLED

#include <vector>
#include <random>
#include <cstdint>

struct ConnVariant {
  int variant_id = -1;
  int region_id = -1;
  // conn_local[i][j] = number of direct OCS circuits between local node i and j
  // (0/1 in the simplest case).
  std::vector<std::vector<int>> conn_local;
};

class PrenetVariantPool {
public:
  // Build K variants per region by random_connect-style generation.
  // Each node has <= alpha outgoing local edges; matrix symmetric.
  PrenetVariantPool(int region_size, int region_num, int alpha, int K, int seed);

  const ConnVariant& default_variant(int region_id) const;
  const ConnVariant* get(int region_id, int variant_id) const;
  int size_per_region() const { return _K; }
  int region_size() const { return _region_size; }

  // Select any variant whose conn_local[src_local][dst_local] > 0.
  // Returns -1 if no such variant exists (caller should STAY_ECS).
  int pick_covering_variant(int region_id, int src_local, int dst_local) const;

  // Debug: print a variant's conn matrix to stderr.
  void debug_print(int region_id, int variant_id) const;

private:
  int _region_size;
  int _region_num;
  int _alpha;
  int _K;
  std::vector<std::vector<ConnVariant>> _variants;  // [region_id][variant_id]

  ConnVariant build_random_variant(int region_id, int variant_id,
                                   std::mt19937& rng) const;
};

#endif // PRENET_ENABLED
#endif // PRENET_VARIANT_POOL_H
