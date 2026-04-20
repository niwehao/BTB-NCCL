// prenet_variant_pool.cpp
#ifdef PRENET_ENABLED
#include "prenet_variant_pool.h"
#include <iostream>
#include <cassert>

PrenetVariantPool::PrenetVariantPool(int region_size, int region_num, int alpha,
                                     int K, int seed)
  : _region_size(region_size), _region_num(region_num), _alpha(alpha), _K(K) {
  assert(region_size > 0);
  assert(region_num > 0);
  assert(K > 0);
  _variants.assign(_region_num, {});
  std::mt19937 rng((uint32_t)seed);
  for (int r = 0; r < _region_num; r++) {
    _variants[r].reserve(K);
    for (int k = 0; k < K; k++) {
      _variants[r].push_back(build_random_variant(r, k, rng));
    }
  }
  std::cerr << "[PRENET] variant pool: " << region_num << " regions x " << K
            << " variants each (region_size=" << region_size
            << " alpha=" << alpha << ")" << std::endl;
}

ConnVariant PrenetVariantPool::build_random_variant(int region_id, int variant_id,
                                                    std::mt19937& rng) const {
  ConnVariant v;
  v.variant_id = variant_id;
  v.region_id = region_id;
  v.conn_local.assign(_region_size, std::vector<int>(_region_size, 0));

  // For each node, add up to alpha edges randomly to distinct partners,
  // respecting alpha cap on both endpoints.
  std::vector<int> deg(_region_size, 0);
  std::uniform_int_distribution<int> dist(0, _region_size - 1);

  for (int i = 0; i < _region_size; i++) {
    int attempts = 0;
    const int MAX_ATTEMPTS = _region_size * 10;
    while (deg[i] < _alpha && attempts < MAX_ATTEMPTS) {
      int j = dist(rng);
      if (j == i || v.conn_local[i][j] == 1 || deg[j] >= _alpha) {
        attempts++;
        continue;
      }
      v.conn_local[i][j] = 1;
      v.conn_local[j][i] = 1;
      deg[i]++;
      deg[j]++;
      attempts = 0;
    }
  }
  return v;
}

const ConnVariant& PrenetVariantPool::default_variant(int region_id) const {
  assert(region_id >= 0 && region_id < _region_num);
  return _variants[region_id][0];
}

const ConnVariant* PrenetVariantPool::get(int region_id, int variant_id) const {
  if (region_id < 0 || region_id >= _region_num) return nullptr;
  if (variant_id < 0 || variant_id >= _K) return nullptr;
  return &_variants[region_id][variant_id];
}

int PrenetVariantPool::pick_covering_variant(int region_id, int src_local,
                                             int dst_local) const {
  if (region_id < 0 || region_id >= _region_num) return -1;
  if (src_local < 0 || src_local >= _region_size) return -1;
  if (dst_local < 0 || dst_local >= _region_size) return -1;
  for (int k = 0; k < _K; k++) {
    if (_variants[region_id][k].conn_local[src_local][dst_local] > 0) return k;
  }
  return -1;
}

void PrenetVariantPool::debug_print(int region_id, int variant_id) const {
  const ConnVariant* v = get(region_id, variant_id);
  if (!v) {
    std::cerr << "[PRENET] variant oob region=" << region_id
              << " variant=" << variant_id << std::endl;
    return;
  }
  std::cerr << "[PRENET] variant r=" << region_id << " v=" << variant_id << ":" << std::endl;
  for (int i = 0; i < _region_size; i++) {
    std::cerr << "  " << i << ": ";
    int deg = 0;
    for (int j = 0; j < _region_size; j++) {
      std::cerr << v->conn_local[i][j] << " ";
      if (v->conn_local[i][j] > 0) deg++;
    }
    std::cerr << "(deg=" << deg << ")" << std::endl;
  }
}

#endif // PRENET_ENABLED
