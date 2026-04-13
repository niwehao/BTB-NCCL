/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/ops/attention.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

bool MultiHeadAttentionParams::is_valid(
    std::tuple<ParallelTensorShape,
               ParallelTensorShape,
               ParallelTensorShape> const &input) const {
  bool is_valid = true;
  is_valid &= std::get<0>(input).is_valid();
  is_valid &= std::get<1>(input).is_valid();
  is_valid &= std::get<2>(input).is_valid();
  return is_valid;
}

Tensor FFModel::multihead_attention(const Tensor query,
                                    const Tensor key,
                                    const Tensor value,
                                    int embed_dim,
                                    int num_heads,
                                    int kdim,
                                    int vdim,
                                    float dropout,
                                    bool bias,
                                    bool add_bias_kv,
                                    bool add_zero_attn,
                                    Initializer *kernel_initializer,
                                    char const *name,
                                    bool is_moe,
                                    int batch_size,
                                    int dp,
                                    int tp,
                                    int pp,
                                    int ep) {
  Layer *li = new Layer(this,
                        OP_MULTIHEAD_ATTENTION,
                        DT_FLOAT,
                        name,
                        3 /*inputs*/,
                        1 /*weights*/,
                        1 /*outputs*/,
                        query,
                        key,
                        value);
  {
    int numdims = query->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = query->dims[i];
    }
    dims[0] = embed_dim;
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, DT_FLOAT, li, 0, true /*create_grad*/);
  }
  {
    // Compute weight size
    int qProjSize = kdim, kProjSize = kdim, vProjSize = kdim,
        oProjSize = embed_dim;
    int qSize = query->dims[0], kSize = key->dims[0], vSize = value->dims[0];
    int qParas = qProjSize * qSize;
    int kParas = kProjSize * kSize;
    int vParas = vProjSize * vSize;
    int oParas = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
    int dims[2] = {qParas + kParas + vParas + oParas, num_heads};

    li->weights[0] = create_weight_legion_ordering(2,
                                                   dims,
                                                   DT_FLOAT,
                                                   li,
                                                   true /*create_grad*/,
                                                   kernel_initializer,
                                                   CHOSEN_SYNC_TYPE);
    // fprintf(stderr,
    //         "attention dim:qParas + kParas + vParas + oParas=%d, num_head=%d, %f \n", dims[0],dims[1],li->weights.get_total_num_parts());
  }
  li->data_type = DT_FLOAT;
  li->add_int_property("embed_dim", embed_dim);
  li->add_int_property("num_heads", num_heads);
  li->add_int_property("kdim", kdim);
  li->add_int_property("vdim", vdim);
  li->add_int_property("bias", bias);
  li->add_int_property("add_bias_kv", add_bias_kv);
  li->add_int_property("add_zero_attn", add_zero_attn);
  li->add_int_property("is_moe", is_moe);
  li->add_int_property("batch_size", batch_size);
  li->add_float_property("dropout", dropout);
  li->add_int_property("dp",dp);
  li->add_int_property("tp",tp);
  li->add_int_property("pp",pp);
  li->add_int_property("ep",ep);
  layers.push_back(li);
  return li->outputs[0];
}

Op *MultiHeadAttention::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("embed_dim", value);
  int embed_dim = value;
  layer->get_int_property("num_heads", value);
  int num_heads = value;
  layer->get_int_property("kdim", value);
  int kdim = value;
  layer->get_int_property("vdim", value);
  int vdim = value;
  float dropout;
  layer->get_float_property("dropout", dropout);
  layer->get_int_property("bias", value);
  bool bias = (bool)value;
  layer->get_int_property("add_bias_kv", value);
  bool add_bias_kv = (bool)value;
  layer->get_int_property("add_zero_attn", value);
  bool add_zero_attn = (bool)value;
  layer->get_int_property("is_moe", value);
  bool is_moe = (bool)value;
  layer->get_int_property("batch_size", value);
  int batch_size = value;
  fprintf(stderr,"attention batch size:%d",batch_size);
  layer->get_int_property("dp", value);
  int dp=value;

  layer->get_int_property("tp", value);
  int tp=value;

  layer->get_int_property("pp", value);
  int pp=value;

  layer->get_int_property("ep", value);
  int ep=value;

  return new MultiHeadAttention(model,
                                layer->layer_guid,
                                inputs[0],
                                inputs[1],
                                inputs[2],
                                embed_dim,
                                num_heads,
                                kdim,
                                vdim,
                                dropout,
                                bias,
                                add_bias_kv,
                                add_zero_attn,
                                false /*allocate_weights*/,
                                layer->name,
                                is_moe,
                                batch_size,
                                dp,
                                tp,
                                pp,
                                ep
                                
                                );
}

MultiHeadAttention::MultiHeadAttention(FFModel &model,
                                       LayerID const &_layer_guid,
                                       const ParallelTensor _query,
                                       const ParallelTensor _key,
                                       const ParallelTensor _value,
                                       int _embed_dim,
                                       int _num_heads,
                                       int _kdim,
                                       int _vdim,
                                       float _dropout,
                                       bool _bias,
                                       bool _add_bias_kv,
                                       bool _add_zero_attn,
                                       bool allocate_weights,
                                       char const *name,
                                       bool _is_moe,
                                       int _batch_size,
                                       int dp,
                                       int tp,
                                       int pp,
                                       int ep
                                       )
    // Initializer* _bias_initializer)
    : Op(model,
         OP_MULTIHEAD_ATTENTION,
         DT_FLOAT,
         name,
         3 /*inputs*/,
         1 /*weights*/,
         1 /*outputs*/,
         _query,
         _key,
         _value),
      num_heads(_num_heads), dropout(_dropout), bias(_bias),
      add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
      qSize(_query->dims[0].size), kSize(_key->dims[0].size),
      vSize(_value->dims[0].size), qProjSize(_kdim/tp), kProjSize(_kdim/tp),
      vProjSize(_vdim/tp), oProjSize(_embed_dim/tp),
      qoSeqLength(_query->dims[1].size/_batch_size), kvSeqLength(_key->dims[1].size/_batch_size),
      is_moe(_is_moe),
      batch_size(_batch_size),
      dp(dp), tp(tp), pp(pp), ep(ep)
// bias_initializer(_bias_initializer)
{
  // overwrite layer_guid
  layer_guid = _layer_guid;

  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  numOutputs = 1;
  int numdim = _query->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _query->dims[i];
  }
  dims[0].size = _embed_dim;
  // Currently require no parallelism along this dim
  assert(dims[0].degree == 1);
  if (allocate_weights) {
    // Create weight tensor
    int num_dims = inputs[0]->num_dims;
    // Compute weight size
    int qParas = this->qProjSize * this->qSize;
    int kParas = this->kProjSize * this->kSize;
    int vParas = this->vProjSize * this->vSize;
    int oParas =
        this->oProjSize * (this->vProjSize > 0 ? this->vProjSize : this->vSize);
    ParallelDim dims[3];
    dims[0] = inputs[0]->dims[num_dims - 2];
    dims[0].size = dims[0].degree;
    dims[1] = inputs[0]->dims[num_dims - 1];
    dims[1].size = this->num_heads;
    dims[2].size = qParas + kParas + vParas + oParas;
    dims[2].degree = 1;
    dims[2].parallel_idx = -1;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
#ifdef USE_NCCL
    ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
    ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
    weights[0] = model.create_parallel_weight<3>(dims,
                                                 DT_FLOAT,
                                                 NULL /*owner_op*/,
                                                 true /*create_grad*/,
                                                 initializer,
                                                 comm_type);
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _query->num_dims, dims, DT_FLOAT, this);
  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* // Check correctness */
  /* assert(check_output_input_weight_parallel_dims()); */
  //! TOCHECK
  // numWeights = 0;
}

MultiHeadAttention::MultiHeadAttention(FFModel &model,
                                       const ParallelTensor _query,
                                       const ParallelTensor _key,
                                       const ParallelTensor _value,
                                       const ParallelTensor _weight,
                                       int _embed_dim,
                                       int _num_heads,
                                       int _kdim,
                                       int _vdim,
                                       float _dropout,
                                       bool _bias,
                                       bool _add_bias_kv,
                                       bool _add_zero_attn,
                                       bool allocate_weights,
                                       char const *name,
                                       bool _is_moe,
                                       int _batch_size)
    // Initializer* _bias_initializer)
    : Op(model,
         OP_MULTIHEAD_ATTENTION,
         DT_FLOAT,
         name,
         3 /*inputs*/,
         1 /*weights*/,
         1 /*outputs*/,
         _query,
         _key,
         _value,
         _weight),
      num_heads(_num_heads), dropout(_dropout), bias(_bias),
      add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
      qSize(_query->dims[0].size), kSize(_key->dims[0].size),
      vSize(_value->dims[0].size), qProjSize(_kdim), kProjSize(_kdim),
      vProjSize(_vdim), oProjSize(_embed_dim),
      qoSeqLength(_query->dims[1].size/_batch_size), kvSeqLength(_key->dims[1].size/_batch_size),
      is_moe(_is_moe),
      batch_size(_batch_size)
// bias_initializer(_bias_initializer)
{
  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  numOutputs = 1;
  int numdim = _query->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _query->dims[i];
  }
  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  dims[0].size = _embed_dim;
  // Currently require no parallelism along this dim
  assert(dims[0].degree == 1);
  if (allocate_weights) {
    // Create weight tensor
    int num_dims = inputs[0]->num_dims;
    // Compute weight size
    int qParas = this->qProjSize * this->qSize;
    int kParas = this->kProjSize * this->kSize;
    int vParas = this->vProjSize * this->vSize;
    int oParas =
        this->oProjSize * (this->vProjSize > 0 ? this->vProjSize : this->vSize);
    ParallelDim dims[3];
    dims[0] = inputs[0]->dims[num_dims - 2];
    dims[0].size = dims[0].degree;
    dims[1] = inputs[0]->dims[num_dims - 1];
    dims[1].size = this->num_heads;
    dims[2].size = qParas + kParas + vParas + oParas;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
#ifdef USE_NCCL
    ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
    ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
    weights[0] = model.create_parallel_weight<3>(dims,
                                                 DT_FLOAT,
                                                 NULL /*owner_op*/,
                                                 true /*create_grad*/,
                                                 initializer,
                                                 comm_type);
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _query->num_dims, dims, DT_FLOAT, this);
  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* register_output_weight_parallel_dims(outputs[0], numdim-1, _weight, 1); */
  /* register_output_weight_parallel_dims(outputs[0], numdim-2, _weight, 2); */
  // Check correctness
  /* assert(check_output_input_weight_parallel_dims()); */
  //! TOCHECK
  // numWeights = 0;
}

MultiHeadAttention::MultiHeadAttention(FFModel &model,
                                       MultiHeadAttention const &other,
                                       const ParallelTensor query,
                                       const ParallelTensor key,
                                       const ParallelTensor value,
                                       bool allocate_weights)
    : MultiHeadAttention(model,
                         other.layer_guid,
                         query,
                         key,
                         value,
                         other.oProjSize,
                         other.num_heads,
                         other.qProjSize,
                         other.vProjSize,
                         other.dropout,
                         other.bias,
                         other.add_bias_kv,
                         other.add_zero_attn,
                         allocate_weights,
                         other.name,
                         other.is_moe,
                         other.batch_size,
                         other.dp,
                         other.tp,
                         other.pp,
                         other.ep) {}

MultiHeadAttention::MultiHeadAttention(
    FFModel &model,
    MultiHeadAttentionParams const &params,
    std::tuple<ParallelTensor, ParallelTensor, ParallelTensor> const &inputs,
    bool allocate_weights,
    char const *name)
    : MultiHeadAttention(model,
                         params.layer_guid,
                         std::get<0>(inputs),
                         std::get<1>(inputs),
                         std::get<2>(inputs),
                         params.embed_dim,
                         params.num_heads,
                         params.kdim,
                         params.vdim,
                         params.dropout,
                         params.bias,
                         params.add_bias_kv,
                         params.add_zero_attn,
                         allocate_weights,
                         name,
                         params.is_moe,
                         params.batch_size,
                         params.dp,
                         params.tp,
                         params.pp,
                         params.ep) {}

void MultiHeadAttention::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ATTENTION_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(MultiHeadAttention)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(4, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*
  regions[0](I): query
  regions[1](I): key
  regions[2](I): value
  regions[3](I): weight
  regions[4](O): output
*/
OpMeta *
    MultiHeadAttention::init_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  MultiHeadAttention const *attn = (MultiHeadAttention *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  TensorAccessorR<float, 4> acc_query(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 4> acc_key(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 4> acc_value(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_weight(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output(regions[4],
                                       task->regions[4],
                                       FID_DATA,
                                       ctx,
                                       runtime,
                                       false /*readOutput*/);
  int num_samples = acc_query.rect.hi[2] - acc_query.rect.lo[2] + 1;
  assert(attn->qoSeqLength == acc_query.rect.hi[1] - acc_query.rect.lo[1] + 1);
  assert(attn->qSize == acc_query.rect.hi[0] - acc_query.rect.lo[0] + 1);
  assert(num_samples == acc_key.rect.hi[2] - acc_key.rect.lo[2] + 1);
  assert(attn->kvSeqLength == acc_key.rect.hi[1] - acc_key.rect.lo[1] + 1);
  assert(attn->kSize == acc_key.rect.hi[0] - acc_key.rect.lo[0] + 1);
  assert(num_samples == acc_value.rect.hi[2] - acc_value.rect.lo[2] + 1);
  assert(attn->kvSeqLength == acc_value.rect.hi[1] - acc_value.rect.lo[1] + 1);
  assert(attn->vSize == acc_value.rect.hi[0] - acc_value.rect.lo[0] + 1);
  int num_heads = acc_weight.rect.hi[1] - acc_weight.rect.lo[1] + 1;
  assert(num_samples == acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1);
  assert(attn->qoSeqLength ==
         acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1);
  assert(attn->oProjSize == acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1);

  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MultiHeadAttentionMeta *m =
      new MultiHeadAttentionMeta(handle, attn, gpu_mem, num_samples, num_heads);
  m->profiling = attn->profiling;
  assert(acc_weight.rect.volume() * sizeof(float) == m->weightSize);
  return m;
}

void MultiHeadAttention::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  int idx = 0;
  IndexLauncher launcher(ATTENTION_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(idx++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(idx++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[2]->region));
  launcher.add_field(idx++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(idx++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(4, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): query
  regions[1](I): key
  regions[2](I): value
  regions[3](I): weight
  regions[4](O): output
*/
void MultiHeadAttention::forward_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 5);
  assert(task->regions.size() == regions.size());
  // const MultiHeadAttention* attn = (MultiHeadAttention*) task->args;
  MultiHeadAttentionMeta const *m =
      *((MultiHeadAttentionMeta **)task->local_args);
  TensorAccessorR<float, 4> acc_query(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 4> acc_key(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 4> acc_value(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_weight(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_output(regions[4],
                                       task->regions[4],
                                       FID_DATA,
                                       ctx,
                                       runtime,
                                       false /*readOutput*/);

  MultiHeadAttention::forward_kernel_wrapper(m,
                                             acc_query.ptr,
                                             acc_key.ptr,
                                             acc_value.ptr,
                                             acc_weight.ptr,
                                             acc_output.ptr);
}

void MultiHeadAttention::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(ATTENTION_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(4, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[0]->region_grad));
  launcher.add_field(5, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(6, FID_DATA);
  int num_regions = 7;
  if (inputs[1]->region != inputs[0]->region) {
    // when key != query
    launcher.add_region_requirement(RegionRequirement(inputs[1]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      inputs[1]->region_grad));
    launcher.add_field(num_regions++, FID_DATA);
  }
  if ((inputs[2]->region != inputs[0]->region) &&
      (inputs[2]->region != inputs[1]->region)) {
    // when value != key and value != query
    launcher.add_region_requirement(RegionRequirement(inputs[2]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      inputs[2]->region_grad));
    launcher.add_field(num_regions++, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): query
  regions[1](I): key
  regions[2](I): value
  regions[3](I): weight
  regions[4](I): output_grad
  regions[5](I/O): weight_grad
  regions[6](I/O): query_grad
  regions[7](I/O) (optional): key_grad
  regions[8](I/O) (optional): value_grad
*/
void MultiHeadAttention::backward_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() >= 7);
  assert(task->regions.size() == regions.size());
  // MultiHeadAttention* attn = (MultiHeadAttention*) task->args;
  MultiHeadAttentionMeta const *m =
      *((MultiHeadAttentionMeta **)task->local_args);
  TensorAccessorR<float, 4> acc_query(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 4> acc_key(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 4> acc_value(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> acc_weight(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 4> acc_output_grad(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 3> acc_weight_grad(regions[5],
                                            task->regions[5],
                                            FID_DATA,
                                            ctx,
                                            runtime,
                                            true /*readOutput*/);
  TensorAccessorW<float, 4> acc_query_grad(regions[6],
                                           task->regions[6],
                                           FID_DATA,
                                           ctx,
                                           runtime,
                                           true /*readOutput*/);
  float *key_grad_ptr, *value_grad_ptr;
  assert(acc_query_grad.rect == acc_query.rect);
  assert(acc_weight_grad.rect.volume() == acc_weight.rect.volume());
  if (regions.size() == 7) {
    // assert query == key and query == value
    assert(regions[0].get_logical_region() == regions[1].get_logical_region());
    assert(regions[0].get_logical_region() == regions[2].get_logical_region());
    key_grad_ptr = acc_query_grad.ptr;
    value_grad_ptr = acc_query_grad.ptr;
  } else if (regions.size() == 8) {
    // assert query == key
    assert(regions[0].get_logical_region() == regions[1].get_logical_region());
    TensorAccessorW<float, 4> acc_value_grad(regions[7],
                                             task->regions[7],
                                             FID_DATA,
                                             ctx,
                                             runtime,
                                             true /*readOutput*/);
    assert(acc_value_grad.rect == acc_value.rect);
    key_grad_ptr = acc_query_grad.ptr;
    value_grad_ptr = acc_value_grad.ptr;
  } else {
    assert(regions.size() == 10);
    TensorAccessorW<float, 4> acc_key_grad(regions[7],
                                           task->regions[7],
                                           FID_DATA,
                                           ctx,
                                           runtime,
                                           true /*readOutput*/);
    TensorAccessorW<float, 4> acc_value_grad(regions[8],
                                             task->regions[8],
                                             FID_DATA,
                                             ctx,
                                             runtime,
                                             true /*readOutput*/);
    assert(acc_key.rect == acc_key_grad.rect);
    assert(acc_value.rect == acc_value_grad.rect);
    value_grad_ptr = acc_value_grad.ptr;
    key_grad_ptr = acc_key_grad.ptr;
  }

  MultiHeadAttention::backward_kernel_wrapper(m,
                                              acc_query.ptr,
                                              acc_query_grad.ptr,
                                              acc_key.ptr,
                                              key_grad_ptr,
                                              acc_value.ptr,
                                              value_grad_ptr,
                                              acc_weight.ptr,
                                              acc_weight_grad.ptr,
                                              acc_output_grad.ptr);
}

bool MultiHeadAttention::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
    case PM_NUM_HEADS:
      *value = num_heads;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool MultiHeadAttention::measure_operator_cost(
    Simulator *sim, MachineView const &mv, CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_query, sub_key, sub_value;

  if (true) {
    // Transform inputs from 3 to 4
    // emb, seq, batch, replica?
    sub_query.num_dims = inputs[0]->num_dims + 1;
    sub_key.num_dims = inputs[1]->num_dims + 1;
    sub_value.num_dims = inputs[2]->num_dims + 1;

    sub_output.num_dims = outputs[0]->num_dims + 1;

    for (int i = 0; i < sub_query.num_dims; i++) {

      switch (i) {
      case 0: { //emb
        sub_query.dims[i] = inputs[0]->dims[i] ;
        sub_key.dims[i] = inputs[1]->dims[i] ;
        sub_value.dims[i] = inputs[2]->dims[i] ;
        sub_output.dims[i] = outputs[0]->dims[i];
        //degree seems useless here
        if(mv.ndims == 2) { // tp is the last dim
          // sub_query.dims[i].size /= mv.dim[mv.ndims-1];
          // sub_query.dims[i].degree = -1;

          // sub_key.dims[i].size /= mv.dim[mv.ndims-1];
          // sub_key.dims[i].degree = -1;

          // sub_value.dims[i].size /= mv.dim[mv.ndims-1];
          // sub_value.dims[i].degree = -1;

          sub_output.dims[i].size /= mv.dim[i];
          sub_output.dims[i].degree = -1;
        }
        
        break;
      }
        
      case 1: { //seq
        sub_query.dims[i] = inputs[0]->dims[i];
        sub_key.dims[i] = inputs[1]->dims[i];
        sub_value.dims[i] = inputs[2]->dims[i];
        sub_output.dims[i] = outputs[0]->dims[i];
        
        sub_query.dims[i].size /= batch_size;
        sub_query.dims[i].degree = -1;

        sub_key.dims[i].size /= batch_size;
        sub_key.dims[i].degree = -1;

        sub_value.dims[i].size /= batch_size;
        sub_value.dims[i].degree = -1;

        sub_output.dims[i].size /= batch_size;
        sub_output.dims[i].degree = -1;
        break;
      }
        
      //maybe it is not right here, it should be batchsize/node number
      case 2: { //batch
        sub_query.dims[i] = inputs[0]->dims[i-1];
        sub_key.dims[i] = inputs[1]->dims[i-1];
        sub_value.dims[i] = inputs[2]->dims[i-1];
        sub_output.dims[i] = outputs[0]->dims[i-1];

        // int q_idx = inputs[0]->dims[i-1].parallel_idx;
        // int k_idx = inputs[1]->dims[i-1].parallel_idx;
        // int v_idx = inputs[2]->dims[i-1].parallel_idx;
        // int o_idx = outputs[0]->dims[i-1].parallel_idx;
    
        sub_query.dims[i].size = batch_size / (mv.dim[0]*mv.dim[2]);
        //sub_query.dims[i].degree =0;

        sub_key.dims[i].size = batch_size / (mv.dim[0]*mv.dim[2]);
        //sub_key.dims[i].degree /= mv.dim[q_idx];

        sub_value.dims[i].size = batch_size / (mv.dim[0]*mv.dim[2]);
        //sub_value.dims[i].degree /= mv.dim[q_idx];

        sub_output.dims[i].size = batch_size / (mv.dim[0]*mv.dim[2]);
        //sub_output.dims[i].degree /= mv.dim[o_idx];
        break;
      }
        
      case 3: { //replica, seems useless
        sub_query.dims[i] = inputs[0]->dims[i-1];
        sub_key.dims[i] = inputs[1]->dims[i-1];
        sub_value.dims[i] = inputs[2]->dims[i-1];
        sub_output.dims[i] = outputs[0]->dims[i-1];
        break;
      }
        
      default:
        break;
      }
    }


  }
  else {
    if (!inputs[0]->get_sub_tensor(mv, sub_query)) {
      return false;
    }
    if (!inputs[1]->get_sub_tensor(mv, sub_key)) {
      return false;
    }
    if (!inputs[2]->get_sub_tensor(mv, sub_value)) {
      return false;
    }
    if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
      return false;
    }
  }
  // Currently assume only data parallel
  size_t num_weights = 0;
  {
    // Compute weight size
    int qSize = sub_query.dims[0].size;
    int kSize = sub_key.dims[0].size;
    int vSize = sub_value.dims[0].size;
    int qParas = qProjSize * qSize;
    int kParas = kProjSize * kSize;
    int vParas = vProjSize * vSize;
    int oParas = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
    num_weights = num_heads * (qParas + kParas + vParas + oParas);
  }
  assert(sub_query.num_dims == 4);
  int num_samples = sub_query.dims[2].size; //batch size
  //re-adjust the AttMeta
  // dropout(_dropout), bias(_bias),
  // add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),

  //qSize=sub_query.dims[0].size;
  // kSize=sub_key.dims[0].size;
  // vSize=sub_value.dims[0].size;
  
  // qProjSize(_kdim), kProjSize(_kdim),
  // vProjSize(_vdim), oProjSize(_embed_dim),
  // qoSeqLength(_query->dims[1].size/_batch_size), kvSeqLength(_key->dims[1].size/_batch_size),
  // is_moe(_is_moe),
  // batch_size(_batch_size)


  printf("query(%d %d %d %d)\n",sub_query.dims[0].size,sub_query.dims[1].size,sub_query.dims[2].size,sub_query.dims[3].size);//1024 512, 4,1
  MultiHeadAttentionMeta *m = new MultiHeadAttentionMeta(
      sim->handler, this, sim->memory, num_samples, num_heads);

  // allocate tensors in simulator
  sim->free_all();
  float const *query_ptr =
      (float const *)sim->allocate(sub_query.get_volume(), DT_FLOAT);
  float const *key_ptr =
      (float const *)sim->allocate(sub_key.get_volume(), DT_FLOAT);
  float const *value_ptr =
      (float const *)sim->allocate(sub_value.get_volume(), DT_FLOAT);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float const *weight_ptr = (float const *)sim->allocate(num_weights, DT_FLOAT);
  cost_metrics.weights_memory += cost_metrics.total_mem_diff_from(sim->offset);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(
        m, query_ptr, key_ptr, value_ptr, weight_ptr, output_ptr);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *query_grad_ptr =
        (float *)sim->allocate(sub_query.get_volume(), DT_FLOAT);
    float *key_grad_ptr =
        (float *)sim->allocate(sub_key.get_volume(), DT_FLOAT);
    float *value_grad_ptr =
        (float *)sim->allocate(sub_value.get_volume(), DT_FLOAT);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *weight_grad_ptr = (float *)sim->allocate(num_weights, DT_FLOAT);
    cost_metrics.weights_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [=] {
      backward_kernel_wrapper(m,
                              query_ptr,
                              query_grad_ptr,
                              key_ptr,
                              key_grad_ptr,
                              value_ptr,
                              value_grad_ptr,
                              weight_ptr,
                              weight_grad_ptr,
                              output_grad_ptr);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure MultiHeadAttention] query(%d %d %d) key(%d %d %d) "
           "value(%d %d %d) output(%d %d %d)"
           "forward_time(%.4lf) backward_time(%.4lf)\n",
           sub_query.dims[2].size,//4
           sub_query.dims[1].size,//512
           sub_query.dims[0].size,//1024
           sub_key.dims[2].size,
           sub_key.dims[1].size,
           sub_key.dims[0].size,
           sub_value.dims[2].size,
           sub_value.dims[1].size,
           sub_value.dims[0].size,
           sub_output.dims[2].size,
           sub_output.dims[1].size,
           sub_output.dims[0].size,
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure MultiHeadAttention] query(%d %d %d) key(%d %d %d) "
           "value(%d %d %d) output(%d %d %d)"
           "forward_time(%.4lf)\n",
           sub_query.dims[2].size,
           sub_query.dims[1].size,
           sub_query.dims[0].size,
           sub_key.dims[2].size,
           sub_key.dims[1].size,
           sub_key.dims[0].size,
           sub_value.dims[2].size,
           sub_value.dims[1].size,
           sub_value.dims[0].size,
           sub_output.dims[2].size,
           sub_output.dims[1].size,
           sub_output.dims[0].size,
           cost_metrics.forward_time);
  }
  
  // Free multiheadattentionmeta
  delete m;
  return true;
}

using PCG::Node;

bool operator==(MultiHeadAttentionParams const &lhs,
                MultiHeadAttentionParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.embed_dim == rhs.embed_dim &&
         lhs.num_heads == rhs.num_heads && lhs.kdim == rhs.kdim &&
         lhs.vdim == rhs.vdim && lhs.dropout == rhs.dropout &&
         lhs.bias == rhs.bias && lhs.add_bias_kv == rhs.add_bias_kv &&
         lhs.add_zero_attn == rhs.add_zero_attn && lhs.is_moe == rhs.is_moe && lhs.batch_size == rhs.batch_size;
}

MultiHeadAttentionParams MultiHeadAttention::get_params() const {
  MultiHeadAttentionParams params;
  params.layer_guid = this->layer_guid;
  params.embed_dim = this->oProjSize;
  params.num_heads = this->num_heads;
  params.kdim = this->kProjSize;
  params.vdim = this->vProjSize;
  params.dropout = this->dropout;
  params.bias = this->bias;
  params.add_bias_kv = this->add_bias_kv;
  params.add_zero_attn = this->add_zero_attn;
  params.is_moe = this->is_moe;
  params.batch_size = this->batch_size;
  return params;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::MultiHeadAttentionParams>::operator()(
    FlexFlow::MultiHeadAttentionParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.embed_dim);
  hash_combine(key, params.num_heads);
  hash_combine(key, params.kdim);
  hash_combine(key, params.vdim);
  hash_combine(key, params.dropout);
  hash_combine(key, params.bias);
  hash_combine(key, params.add_bias_kv);
  hash_combine(key, params.add_zero_attn);
  hash_combine(key, params.is_moe);
  hash_combine(key, params.batch_size);
  return key;
}
}; // namespace std
