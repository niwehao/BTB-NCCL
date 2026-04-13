/* Copyright 2023 CMU, Facebook, Stanford
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

#include "flexflow/model.h"

using namespace FlexFlow;

Tensor FFModel::moe(const Tensor input,
                    int num_exp,
                    int num_select,
                    int expert_hidden_size,
                    int output_dim_size,
                    float alpha,
                    float lambda) {
  // MoE model
  // std::cout << "moe input dim " << input->num_dims<<" " << input->dims[0]<<" " <<input->dims[1] << std::endl;
  Tensor gate_preds = dense(input, num_exp, AC_MODE_RELU);
  // std::cout << "gate pred " << gate_preds->num_dims<<" " << gate_preds->dims[0]<<" " <<gate_preds->dims[1] << std::endl;

  Tensor topK_output[2];
  top_k(gate_preds, topK_output, num_select, false);
  // std::cout << "topK_output " << topK_output[1]->num_dims<<" " << topK_output[1]->dims[0]<<" " <<topK_output[1]->dims[1] << std::endl;
  Tensor exp_tensors[num_exp];
  group_by(input, topK_output[1], exp_tensors, num_exp, alpha);
  Tensor agg_inputs[num_exp + 4];
  agg_inputs[0] = softmax(topK_output[0]); // gate preds
  agg_inputs[1] = topK_output[1];          // gate assign
  agg_inputs[2] = topK_output[1];          // gate assign TopK (for cache)
  agg_inputs[3] = gate_preds;              // full gate preds
  for (int i = 0; i < num_exp; i++) {
    // std::cout <<  "****************" <<std::endl;
    // std::cout <<" exp_tensors-"<<i <<" " << exp_tensors[i]->num_dims<<" " << exp_tensors[i]->dims[0]<<" " <<exp_tensors[i]->dims[1] << std::endl;
    Tensor exp_pred_tmp = dense(exp_tensors[i], expert_hidden_size, AC_MODE_RELU);
    // std::cout <<" exp_pred-"<<i <<" " << exp_pred->num_dims<<" " << exp_pred->dims[0]<<" " <<exp_pred->dims[1] << std::endl;
    Tensor exp_pred = dense(exp_pred_tmp, output_dim_size, AC_MODE_RELU);
    agg_inputs[i + 4] = softmax(exp_pred);
    // std::cout <<" agg input-"<<i <<" " << agg_inputs[i + 4]->num_dims<<" " << agg_inputs[i + 4]->dims[0]<<" " <<agg_inputs[i + 4]->dims[1] << std::endl;
    // std::cout <<  "****************" <<std::endl;
  }
  Tensor coop_output = aggregate(agg_inputs, num_exp, lambda);
  // std::cout << "coop_output " << coop_output->num_dims<<" " << coop_output->dims[0]<<" " <<coop_output->dims[1] << std::endl;
  // get_metrics();
  return coop_output;
}
