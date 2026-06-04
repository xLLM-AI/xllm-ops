/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

void select_unshared_kv_impl_npu(
    const at::Tensor& beam_index,
    std::vector<at::Tensor> x_key_block,
    std::vector<at::Tensor> x_value_block,
    const at::Tensor& block_table,
    const at::Tensor& group_token_num,
    int64_t decode_step,
    int64_t beam_size,
    int64_t layer_num) {
  at::TensorList x_key_block_list = at::TensorList(x_key_block);
  at::TensorList x_value_block_list = at::TensorList(x_value_block);
  EXEC_NPU_CMD(aclnnSelectUnsharedKV,
               beam_index,
               block_table,
               x_key_block_list,
               x_value_block_list,
               group_token_num,
               decode_step,
               beam_size,
               layer_num,
               x_key_block_list,
               x_value_block_list);
  return ;
}

std::tuple<at::Tensor&, at::Tensor&> cache_unshared_kv_impl_npu(
    at::Tensor& x_key_block,
    at::Tensor& x_value_block,
    const at::Tensor& curr_key,
    const at::Tensor& curr_value,
    const at::Tensor& block_table,
    const at::Tensor& decode_step) {
  EXEC_NPU_CMD(aclnnCacheUnsharedKv,
               x_key_block,
               x_value_block,
               curr_key,
               curr_value,
               block_table,
               decode_step,
               x_key_block,
               x_value_block);
  return std::tuple<at::Tensor&, at::Tensor&>(x_key_block, x_value_block);
}

at::Tensor x_attention_impl_npu(const at::Tensor& query,
                                const at::Tensor& key_cache,
                                const at::Tensor& value_cache,
                                const at::Tensor& unshared_key,
                                const at::Tensor& unshared_value,
                                const c10::optional<at::Tensor>& shared_block_tables,
                                const c10::optional<at::Tensor>& unshared_block_tables,
                                const at::Tensor& actual_shared_kvlen,
                                const at::Tensor& decode_step) {
  at::Tensor attnOut = at::empty_like(query);
  
  EXEC_NPU_CMD(aclnnXAttention,
               query,
               key_cache,
               value_cache,
               unshared_key,
               unshared_value,
               unshared_block_tables,
               actual_shared_kvlen,
               decode_step,
               shared_block_tables,
               attnOut);

  return attnOut;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> beam_search_group_impl_npu(
                                const at::Tensor& log_probs,
                                const at::Tensor& top_tokens,
                                const at::Tensor& top_probs,
                                const at::Tensor& sequence,
                                int64_t current_step,
                                int64_t top_k
                              ) {
  auto output_shape = log_probs.sizes();
  // Allocate outputs on the same device with appropriate dtypes
  auto beam_width = top_tokens.size(1);
  auto request_num = output_shape[0] / beam_width;
  at::Tensor out_token_ids = at::zeros({request_num, top_k}, top_tokens.options());
  at::Tensor out_token_index = at::zeros({request_num, top_k}, top_tokens.options());
  at::Tensor out_log_probs = at::zeros({request_num, top_k}, log_probs.options());
  at::Tensor out_beam_count_prefix_sums = at::zeros({request_num, beam_width}, top_tokens.options());
  at::Tensor out_sequence = at::zeros({request_num, top_k, current_step + 1}, top_tokens.options());
  EXEC_NPU_CMD(aclnnBeamSearchGroup,
               log_probs, top_tokens, top_probs, sequence, current_step, top_k,
               out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, out_sequence);
  return std::make_tuple(out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, out_sequence);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
beam_search_rec_final_select_impl_npu(const at::Tensor& log_probs,
                                      const at::Tensor& top_tokens,
                                      const at::Tensor& top_probs,
                                      const at::Tensor& sequence,
                                      int64_t current_step,
                                      int64_t result_width) {
  auto output_shape = log_probs.sizes().vec();
  output_shape[0] = sequence.size(0) * result_width;
  at::Tensor out_token_ids = at::zeros(output_shape, top_tokens.options());
  at::Tensor out_token_index = at::zeros(output_shape, top_tokens.options());
  at::Tensor out_log_probs = at::zeros(output_shape, log_probs.options());
  at::Tensor out_sequence =
      at::zeros({sequence.size(0), result_width, sequence.size(2)},
                top_tokens.options());
  EXEC_NPU_CMD(aclnnOnerecFinalBeamSelect,
               log_probs,
               top_tokens,
               top_probs,
               sequence,
               current_step,
               out_token_ids,
               out_token_index,
               out_log_probs,
               out_sequence);
  return std::make_tuple(
      out_token_ids, out_token_index, out_log_probs, out_sequence);
}

at::Tensor causal_conv1d(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,    
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt,
    at::IntArrayRef num_accepted_tokens_opt,
    int64_t  activation_mode,
    int64_t  pad_slot_id,
    int64_t  run_mode)
{
    at::Tensor output = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnCausalConv1d,
                    x,                 
                    weight,
                    bias_opt,
                    conv_state,
                    query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                    num_accepted_tokens_opt,
                    activation_mode,
                    pad_slot_id,
                    run_mode,
                    output		    
                ); 

    return output;
}

at::Tensor recurrent_gated_delta_rule(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    at::Tensor &state,
    const c10::optional<at::Tensor> &beta,
    const c10::optional<double> scale,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &ssm_state_indices,
    const c10::optional<at::Tensor> &num_accepted_tokens,
    const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &gk)
{
    at::Tensor outResult = at::empty_like(value);
    float scale_real = static_cast<float>(scale.value());

    EXEC_NPU_CMD(aclnnRecurrentGatedDeltaRule, query, key, value, beta, state, actual_seq_lengths, ssm_state_indices,
                 g, gk, num_accepted_tokens, scale_real, outResult);

    return outResult;
}

std::tuple<at::Tensor, at::Tensor> rec_constrained_topk_impl_npu(
    const at::Tensor& logits,
    const at::Tensor& sequence_group,
    const at::Tensor& first_token_ids,
    const at::Tensor& prefix1_offsets,
    const at::Tensor& prefix1_values,
    const at::Tensor& prefix1_pair_keys,
    const at::Tensor& prefix2_value_offsets,
    const at::Tensor& prefix2_values,
    const at::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k,
    int64_t max_prefix1_degree,
    int64_t max_prefix2_degree) {
  at::Tensor out_tokens =
      at::empty({logits.size(0), top_k},
                logits.options().dtype(at::ScalarType::Int));
  at::Tensor out_logprobs =
      at::empty({logits.size(0), top_k},
                logits.options().dtype(at::ScalarType::Float));
  EXEC_NPU_CMD(aclnnRecConstrainedTopK,
               logits,
               sequence_group,
               first_token_ids,
               prefix1_offsets,
               prefix1_values,
               prefix1_pair_keys,
               prefix2_value_offsets,
               prefix2_values,
               temperatures,
               current_step,
               top_k,
               max_prefix1_degree,
               max_prefix2_degree,
               out_tokens,
               out_logprobs);
  return std::make_tuple(out_tokens, out_logprobs);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mega_chunk_gdn(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &g,
    const at::Tensor &beta,
    const at::Tensor &mask_lower,
    const at::Tensor &mask_full,
    const at::Tensor &minus_identity,
    const at::Tensor &cu_seqlens,
    const at::Tensor &initial_state,
    int64_t num_matrices,
    bool has_initial_state)
{
    constexpr int64_t kChunkSize = 128;

    TORCH_CHECK(q.dim() == 4, "q must have shape [1, total_tokens, num_key_heads, head_dim]");
    TORCH_CHECK(k.dim() == 4, "k must have shape [1, total_tokens, num_key_heads, head_dim]");
    TORCH_CHECK(v.dim() == 4, "v must have shape [1, total_tokens, num_value_heads, head_dim]");
    TORCH_CHECK(g.dim() == 3, "g must have shape [1, total_tokens, num_value_heads]");
    TORCH_CHECK(beta.dim() == 3, "beta must have shape [1, total_tokens, num_value_heads]");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2, "cu_seqlens must be a 1-D tensor");
    TORCH_CHECK(num_matrices > 0, "num_matrices must be positive");

    const int64_t total_tokens = q.size(1);
    const int64_t num_heads = v.size(2);
    const int64_t head_dim = q.size(3);
    const int64_t num_sequences = cu_seqlens.numel() - 1;

    at::Tensor out = at::empty_like(v);
    at::Tensor g_sum = at::empty(g.sizes(), g.options().dtype(at::kFloat));
    at::Tensor g_t = at::empty({num_heads, total_tokens}, g.options().dtype(at::kFloat));
    at::Tensor beta_t = at::empty({num_heads, total_tokens}, beta.options());
    at::Tensor a = at::zeros({1, total_tokens, num_heads, kChunkSize}, beta.options());
    at::Tensor a_inv_f32 = at::zeros({1, total_tokens, num_heads, kChunkSize}, g.options().dtype(at::kFloat));
    at::Tensor a_inv = at::zeros({1, total_tokens, num_heads, kChunkSize}, beta.options());
    at::Tensor w = at::empty_like(v);
    at::Tensor u = at::empty_like(v);
    at::Tensor h = at::zeros({num_matrices, head_dim, head_dim}, v.options());
    at::Tensor v_new = at::empty_like(v);
    at::Tensor final_state = at::zeros({num_sequences * num_heads, head_dim, head_dim}, v.options());

    EXEC_NPU_CMD(aclnnMegaChunkGdn,
                 q,
                 k,
                 v,
                 g,
                 beta,
                 mask_lower,
                 mask_full,
                 minus_identity,
                 cu_seqlens,
                 initial_state,
                 num_matrices,
                 has_initial_state,
                 out,
                 g_sum,
                 g_t,
                 beta_t,
                 a,
                 a_inv_f32,
                 a_inv,
                 w,
                 u,
                 h,
                 v_new,
                 final_state);

    return std::make_tuple(out, g_sum, g_t, beta_t, a, a_inv_f32, a_inv, w, u, h, v_new, final_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select_unshared_kv", &select_unshared_kv_impl_npu, "select_unshared_kv");
  m.def("cache_unshared_kv", &cache_unshared_kv_impl_npu, "cache_unshared_kv");
  m.def("x_attention", &x_attention_impl_npu, "x_attention");
  m.def("beam_search_group", &beam_search_group_impl_npu, "beam_search_group");
  m.def("beam_search_rec_final_select",
        &beam_search_rec_final_select_impl_npu,
        "beam_search_rec_final_select");
  m.def("causal_conv1d", &causal_conv1d, "causal_conv1d");
  m.def("recurrent_gated_delta_rule", &recurrent_gated_delta_rule, "recurrent_gated_delta_rule");
  m.def("rec_constrained_topk", &rec_constrained_topk_impl_npu, "rec_constrained_topk");
  m.def("mega_chunk_gdn", &mega_chunk_gdn, "mega_chunk_gdn");
}
