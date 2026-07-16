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
                                const at::Tensor& decode_step,
                                double scale_value = 0.0) {
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
               scale_value,
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

at::Tensor causal_conv1d_packed_qkv_general_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt,
    int64_t q_dim,
    int64_t k_dim,
    int64_t v_dim,
    int64_t head_dim,
    at::ScalarType output_dtype)
{
    constexpr int64_t kPackedQkvActivationMode = 2;
    constexpr int64_t kPadSlotId = -1;
    constexpr int64_t kForwardRunMode = 0;
    at::Tensor output = at::empty(x.sizes(), x.options().dtype(output_dtype));
    c10::optional<at::Tensor> bias_opt = c10::nullopt;
    at::IntArrayRef num_accepted_tokens_opt;
    EXEC_NPU_CMD(aclnnCausalConv1dQkv,
                 x,
                 weight,
                 bias_opt,
                 conv_state,
                 query_start_loc_opt,
                 cache_indices_opt,
                 initial_state_mode_opt,
                 num_accepted_tokens_opt,
                 kPackedQkvActivationMode,
                 kPadSlotId,
                 kForwardRunMode,
                 q_dim,
                 k_dim,
                 v_dim,
                 head_dim,
                 output);
    return output;
}

at::Tensor causal_conv1d_packed_qkv_general(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt,
    int64_t q_dim,
    int64_t k_dim,
    int64_t v_dim,
    int64_t head_dim)
{
    return causal_conv1d_packed_qkv_general_impl(x,
                                                 weight,
                                                 conv_state,
                                                 query_start_loc_opt,
                                                 cache_indices_opt,
                                                 initial_state_mode_opt,
                                                 q_dim,
                                                 k_dim,
                                                 v_dim,
                                                 head_dim,
                                                 at::kHalf);
}

at::Tensor causal_conv1d_packed_qkv_general_bf16(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt,
    int64_t q_dim,
    int64_t k_dim,
    int64_t v_dim,
    int64_t head_dim)
{
    return causal_conv1d_packed_qkv_general_impl(x,
                                                 weight,
                                                 conv_state,
                                                 query_start_loc_opt,
                                                 cache_indices_opt,
                                                 initial_state_mode_opt,
                                                 q_dim,
                                                 k_dim,
                                                 v_dim,
                                                 head_dim,
                                                 at::kBFloat16);
}

at::Tensor causal_conv1d_packed_qkv(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt)
{
    return causal_conv1d_packed_qkv_general(x,
                                            weight,
                                            conv_state,
                                            query_start_loc_opt,
                                            cache_indices_opt,
                                            initial_state_mode_opt,
                                            1024,
                                            1024,
                                            3072,
                                            128);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_fwd_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& z,
    double eps,
    int64_t group_size,
    bool norm_before_gate,
    bool is_rms_norm) {
  auto sizes = x.sizes().vec();
  const int64_t dim_num = sizes.size();
  int64_t full_n = sizes[dim_num - 1];
  int64_t m = 1;
  for (int64_t i = 0; i < dim_num - 1; ++i) {
    m *= sizes[i];
  }
  int64_t gs = group_size <= 0 ? full_n : group_size;
  int64_t ngroups = gs > 0 ? full_n / gs : 1;

  at::Tensor y = at::empty(sizes, x.options());
  at::Tensor mean = at::zeros({is_rms_norm ? 0 : ngroups * m},
                              x.options().dtype(at::ScalarType::Float));
  at::Tensor rstd = at::zeros({ngroups * m},
                              x.options().dtype(at::ScalarType::Float));

  EXEC_NPU_CMD(aclnnLayerNormFwd,
               x,
               weight,
               bias,
               z,
               eps,
               group_size,
               norm_before_gate,
               is_rms_norm,
               y,
               mean,
               rstd);
  return std::make_tuple(y, mean, rstd);
}

std::tuple<at::Tensor, at::Tensor> moe_fused_add_topk_impl_npu(
    const at::Tensor& x,
    const at::Tensor& add_num,
    const c10::optional<at::Tensor>& mapping_num,
    const c10::optional<at::Tensor>& mapping_table,
    int64_t group_num,
    int64_t group_topk,
    int64_t top_n,
    int64_t top_k,
    int64_t activate_type,
    bool is_norm,
    double scale,
    bool enable_expert_mapping) {
  const int64_t batch = x.size(0);

  at::Tensor y = at::empty({batch, top_k},
                           x.options().dtype(at::ScalarType::Float));
  at::Tensor indices = at::empty({batch, top_k},
                                 x.options().dtype(at::ScalarType::Int));

  EXEC_NPU_CMD(aclnnMoeFusedAddTopk,
               x,
               add_num,
               mapping_num,
               mapping_table,
               group_num,
               group_topk,
               top_n,
               top_k,
               activate_type,
               is_norm,
               scale,
               enable_expert_mapping,
               y,
               indices);
  return std::make_tuple(y, indices);
}

at::Tensor moe_fused_reducesum_div_impl_npu(const at::Tensor& input) {
  at::Tensor output = at::empty(input.sizes(), input.options());
  EXEC_NPU_CMD(aclnnMoeFusedReducesumDiv, input, output);
  return output;
}

at::Tensor replace_token_impl_npu(const at::Tensor& forked_token_ids,
                                  const at::Tensor& last_step_output_token_ids) {
  at::Tensor output = at::empty(forked_token_ids.sizes(), forked_token_ids.options());
  EXEC_NPU_CMD(aclnnReplaceToken, forked_token_ids, last_step_output_token_ids, output);
  return output;
}

std::tuple<at::Tensor&, at::Tensor&> convert_kv_cache_format_impl_npu(
    at::Tensor& k_cache_ptr,
    at::Tensor& v_cache_ptr,
    const at::Tensor& kv_cache_offset,
    const at::Tensor& kv_seq_len,
    bool is_prefill,
    int64_t num_kv_heads,
    int64_t head_size_k,
    int64_t head_size_v) {
  // In-place ND2NZ rewrite: k_cache_ptr / v_cache_ptr are modified in place;
  // the aclnn interface takes only the 4 inputs + 4 attrs (no extra outputs).
  EXEC_NPU_CMD(aclnnConvertKvCacheFormat,
               k_cache_ptr,
               v_cache_ptr,
               kv_cache_offset,
               kv_seq_len,
               is_prefill,
               num_kv_heads,
               head_size_k,
               head_size_v);
  return std::tuple<at::Tensor&, at::Tensor&>(k_cache_ptr, v_cache_ptr);
}

at::Tensor& scatter_nd_update_v2_impl_npu(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& updates,
    at::IntArrayRef strides) {
  // In-place scatter: var is modified in place; the aclnn interface takes
  // (varRef, indices, updates, strides) with no extra output tensor.
  EXEC_NPU_CMD(aclnnScatterNdUpdateV2, var, indices, updates, strides);
  return var;
}

at::Tensor hc_post_impl_npu(const at::Tensor& x,
                            const at::Tensor& residual,
                            const at::Tensor& post,
                            const at::Tensor& comb) {
  // y has the same shape / dtype as residual ([bs, hc, d]).
  at::Tensor y = at::empty(residual.sizes(), residual.options());
  EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, y);
  return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm_bias_impl_npu(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& beta,
    double epsilon) {
  // x = x1 + x2; rstd = 1/sqrt(mean(x^2, last dim) + eps); y = x*rstd*gamma+beta.
  // rstd keeps the leading dims of x1 with the last (gamma) dims collapsed to 1.
  auto sizes = x1.sizes().vec();
  const int64_t dim_x = static_cast<int64_t>(sizes.size());
  const int64_t dim_gamma = gamma.dim();
  std::vector<int64_t> rstd_shape(sizes.begin(), sizes.end());
  for (int64_t i = dim_x - dim_gamma; i < dim_x && i >= 0; ++i) {
    rstd_shape[i] = 1;
  }

  at::Tensor y = at::empty(sizes, x1.options());
  at::Tensor x = at::empty(sizes, x1.options());
  at::Tensor rstd = at::empty(rstd_shape,
                              x1.options().dtype(at::ScalarType::Float));

  EXEC_NPU_CMD(aclnnAddRmsNormBias, x1, x2, gamma, beta, epsilon, y, rstd, x);
  return std::make_tuple(y, rstd, x);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
moe_init_routing_custom_impl_npu(const at::Tensor& x,
                                 const at::Tensor& expert_idx,
                                 int64_t active_num,
                                 int64_t expert_num,
                                 int64_t drop_pad_mode,
                                 int64_t expert_tokens_num_type,
                                 bool expert_tokens_num_flag,
                                 int64_t quant_mode,
                                 int64_t row_idx_type) {
  // Minimal-config wrapper: unquant + dropless + GATHER + COUNT is the target.
  // active_expert_range defaults to [0, expert_num]; scale/offset are unused.
  constexpr int64_t QUANT_MODE_UNQUANT = -1;
  constexpr int64_t CUMSUM = 0;
  constexpr int64_t COUNT = 1;
  constexpr int64_t KEY_VALUE = 2;
  constexpr int64_t expert_capacity = -1;

  c10::optional<at::Tensor> scale = c10::nullopt;
  c10::optional<at::Tensor> offset = c10::nullopt;

  auto x_size = x.sizes();
  auto expert_idx_size = expert_idx.sizes();
  const int64_t bs = x_size[0];
  const int64_t h = x_size[1];
  const int64_t k = expert_idx_size[1];

  std::vector<int64_t> active_expert_range_vec = {0, expert_num};
  at::IntArrayRef active_expert_range(active_expert_range_vec);
  const int64_t expert_length = active_expert_range_vec[1] - active_expert_range_vec[0];

  int64_t expanded_scale_len = 0;
  at::Tensor expanded_x;
  if (drop_pad_mode == 1) {
    if (quant_mode == QUANT_MODE_UNQUANT) {
      expanded_x = at::empty({expert_num, expert_capacity, h}, x.options());
    } else {
      expanded_x =
          at::empty({expert_num, expert_capacity, h}, x.options().dtype(at::kChar));
    }
    expanded_scale_len = expert_num * expert_capacity;
  } else {
    if (active_num > 0) {
      int64_t num_out_tokens = std::min(bs * k, active_num);
      if (quant_mode == QUANT_MODE_UNQUANT) {
        expanded_x = at::empty({num_out_tokens, h}, x.options());
      } else {
        expanded_x = at::empty({num_out_tokens, h}, x.options().dtype(at::kChar));
      }
      expanded_scale_len = num_out_tokens;
    } else {
      if (quant_mode == QUANT_MODE_UNQUANT) {
        expanded_x = at::empty({bs * k, h}, x.options());
      } else {
        expanded_x = at::empty({bs * k, h}, x.options().dtype(at::kChar));
      }
      expanded_scale_len = bs * k;
    }
  }

  at::Tensor expanded_row_idx = at::empty({bs * k}, expert_idx.options());
  at::Tensor expert_tokens_count_or_cumsum;
  if (expert_tokens_num_type >= CUMSUM && expert_tokens_num_type <= COUNT) {
    expert_tokens_count_or_cumsum =
        at::empty({expert_length}, x.options().dtype(at::kLong));
  } else if (expert_tokens_num_type == KEY_VALUE) {
    expert_tokens_count_or_cumsum =
        at::empty({expert_num, 2}, x.options().dtype(at::kLong));
  }
  at::Tensor expanded_scale =
      at::empty({expanded_scale_len}, x.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnMoeInitRoutingCustom,
               x,
               expert_idx,
               scale,
               offset,
               active_num,
               expert_capacity,
               expert_num,
               drop_pad_mode,
               expert_tokens_num_type,
               expert_tokens_num_flag,
               quant_mode,
               active_expert_range,
               row_idx_type,
               expanded_x,
               expanded_row_idx,
               expert_tokens_count_or_cumsum,
               expanded_scale);
  return std::make_tuple(expanded_x,
                         expanded_row_idx,
                         expert_tokens_count_or_cumsum,
                         expanded_scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
moe_init_routing_v3_impl_npu(const at::Tensor& x,
                             const at::Tensor& expert_idx,
                             int64_t active_num,
                             int64_t expert_num,
                             int64_t drop_pad_mode,
                             int64_t expert_tokens_num_type,
                             bool expert_tokens_num_flag,
                             int64_t quant_mode,
                             int64_t row_idx_type) {
  // Minimal-config wrapper mirroring moe_init_routing_custom: the target config
  // is unquant + dropless + GATHER + COUNT. active_expert_range defaults to
  // [0, expert_num]; scale/offset are unused. Only the underlying aclnn op name
  // differs (aclnnMoeInitRoutingV3 instead of aclnnMoeInitRoutingCustom).
  constexpr int64_t QUANT_MODE_UNQUANT = -1;
  constexpr int64_t CUMSUM = 0;
  constexpr int64_t COUNT = 1;
  constexpr int64_t KEY_VALUE = 2;
  constexpr int64_t expert_capacity = -1;

  c10::optional<at::Tensor> scale = c10::nullopt;
  c10::optional<at::Tensor> offset = c10::nullopt;

  auto x_size = x.sizes();
  auto expert_idx_size = expert_idx.sizes();
  const int64_t bs = x_size[0];
  const int64_t h = x_size[1];
  const int64_t k = expert_idx_size[1];

  std::vector<int64_t> active_expert_range_vec = {0, expert_num};
  at::IntArrayRef active_expert_range(active_expert_range_vec);
  const int64_t expert_length = active_expert_range_vec[1] - active_expert_range_vec[0];

  int64_t expanded_scale_len = 0;
  at::Tensor expanded_x;
  if (drop_pad_mode == 1) {
    if (quant_mode == QUANT_MODE_UNQUANT) {
      expanded_x = at::empty({expert_num, expert_capacity, h}, x.options());
    } else {
      expanded_x =
          at::empty({expert_num, expert_capacity, h}, x.options().dtype(at::kChar));
    }
    expanded_scale_len = expert_num * expert_capacity;
  } else {
    if (active_num > 0) {
      int64_t num_out_tokens = std::min(bs * k, active_num);
      if (quant_mode == QUANT_MODE_UNQUANT) {
        expanded_x = at::empty({num_out_tokens, h}, x.options());
      } else {
        expanded_x = at::empty({num_out_tokens, h}, x.options().dtype(at::kChar));
      }
      expanded_scale_len = num_out_tokens;
    } else {
      if (quant_mode == QUANT_MODE_UNQUANT) {
        expanded_x = at::empty({bs * k, h}, x.options());
      } else {
        expanded_x = at::empty({bs * k, h}, x.options().dtype(at::kChar));
      }
      expanded_scale_len = bs * k;
    }
  }

  at::Tensor expanded_row_idx = at::empty({bs * k}, expert_idx.options());
  at::Tensor expert_tokens_count_or_cumsum;
  if (expert_tokens_num_type >= CUMSUM && expert_tokens_num_type <= COUNT) {
    expert_tokens_count_or_cumsum =
        at::empty({expert_length}, x.options().dtype(at::kLong));
  } else if (expert_tokens_num_type == KEY_VALUE) {
    expert_tokens_count_or_cumsum =
        at::empty({expert_num, 2}, x.options().dtype(at::kLong));
  }
  at::Tensor expanded_scale =
      at::empty({expanded_scale_len}, x.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnMoeInitRoutingV3,
               x,
               expert_idx,
               scale,
               offset,
               active_num,
               expert_capacity,
               expert_num,
               drop_pad_mode,
               expert_tokens_num_type,
               expert_tokens_num_flag,
               quant_mode,
               active_expert_range,
               row_idx_type,
               expanded_x,
               expanded_row_idx,
               expert_tokens_count_or_cumsum,
               expanded_scale);
  return std::make_tuple(expanded_x,
                         expanded_row_idx,
                         expert_tokens_count_or_cumsum,
                         expanded_scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_sinkhorn_impl_npu(
    const at::Tensor& mixes,
    const at::Tensor& rsqrt,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const at::Tensor& x,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double hc_eps) {
  // y = sum_m(pre[b,m] * x[b,:]) -> (bs, d), same dtype as x (bf16).
  // post -> (bs, hc_mult) fp32; comb_frag -> (bs, hc_mult, hc_mult) fp32.
  auto x_sizes = x.sizes();
  const int64_t bs = x_sizes[0];
  const int64_t d = x_sizes[x_sizes.size() - 1];

  at::Tensor y = at::empty({bs, d}, x.options());
  at::Tensor post =
      at::empty({bs, hc_mult}, x.options().dtype(at::ScalarType::Float));
  at::Tensor comb_frag =
      at::empty({bs, hc_mult, hc_mult}, x.options().dtype(at::ScalarType::Float));

  EXEC_NPU_CMD(aclnnHcPreSinkhorn,
               mixes,
               rsqrt,
               hc_scale,
               hc_base,
               x,
               hc_mult,
               hc_sinkhorn_iters,
               hc_eps,
               y,
               post,
               comb_frag);
  return std::make_tuple(y, post, comb_frag);
}

at::Tensor hc_pre_inv_rms_impl_npu(const at::Tensor& x, double epsilon) {
  // y = 1/sqrt(mean(x^2 over last two dims) + eps), output always fp32.
  // x: (b, s, hc, d) -> y: (b, s, 1)   or   x: (t, hc, d) -> y: (t, 1)
  auto x_sizes = x.sizes();
  const auto ndim = x_sizes.size();
  std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
  y_sizes[ndim - 1] = 1;
  y_sizes[ndim - 2] = 1;
  // collapse last two dims into one dim of size 1: keep leading dims, last = 1.
  std::vector<int64_t> out_sizes;
  for (size_t i = 0; i + 2 < ndim; ++i) {
    out_sizes.push_back(x_sizes[i]);
  }
  out_sizes.push_back(1);

  at::Tensor y =
      at::empty(out_sizes, x.options().dtype(at::ScalarType::Float));

  EXEC_NPU_CMD(aclnnHcPreInvRms, x, epsilon, y);
  return y;
}

at::Tensor pp_matmul_opt_impl_npu(const at::Tensor& a, const at::Tensor& b) {
  // c = a @ b^T, a:(M,K), b:(N,K) -> c:(M,N), same dtype as a (bf16/fp16).
  auto a_sizes = a.sizes();
  auto b_sizes = b.sizes();
  const int64_t M = a_sizes[a_sizes.size() - 2];
  const int64_t N = b_sizes[b_sizes.size() - 2];
  at::Tensor c = at::empty({M, N}, a.options());
  EXEC_NPU_CMD(aclnnPpMatmulOpt, a, b, c);
  return c;
}

at::Tensor moe_grouped_matmul_impl_npu(const at::Tensor& x,
                                       const at::Tensor& weight,
                                       const at::Tensor& group_list,
                                       bool transpose_weight = false) {
  // x:(M,K) ND, weight:(G,K,N) ND (or (G,N,K) when transpose_weight),
  // group_list:(G,2) int32/int64 key-value [group_idx, count], y:(M,N).
  const int64_t M = x.sizes()[x.dim() - 2];
  const int64_t N = transpose_weight ? weight.sizes()[weight.dim() - 2]
                                     : weight.sizes()[weight.dim() - 1];
  at::Tensor y = at::empty({M, N}, x.options());
  std::vector<at::Tensor> x_vec{x};
  std::vector<at::Tensor> w_vec{weight};
  std::vector<at::Tensor> y_vec{y};
  at::TensorList x_list = at::TensorList(x_vec);
  at::TensorList w_list = at::TensorList(w_vec);
  at::TensorList y_list = at::TensorList(y_vec);
  EXEC_NPU_CMD(aclnnMoeGroupedMatmul,
               x_list,
               w_list,
               group_list,
               transpose_weight,
               y_list);
  return y;
}

at::Tensor index_group_matmul_impl_npu(const at::Tensor& a,
                                       const at::Tensor& b,
                                       const at::Tensor& scale,
                                       const at::Tensor& per_token_scale,
                                       const at::Tensor& group_list) {
  // int8 grouped matmul + dequant.
  // a:(M,K) int8 ND, b:(G,K,N) int8 (ND passed, framework -> NZ),
  // scale:(G,N) bf16, per_token_scale:(M,) float, group_list:(G,) int64
  // cumulative offset. c:(M,N) bf16 = dequant(a_g@b_g) * scale_g * pts[i].
  const int64_t M = a.sizes()[a.dim() - 2];
  const int64_t N = b.sizes()[b.dim() - 1];
  at::Tensor c = at::empty({M, N}, a.options().dtype(at::kBFloat16));
  std::vector<at::Tensor> a_vec{a};
  std::vector<at::Tensor> b_vec{b};
  std::vector<at::Tensor> scale_vec{scale};
  std::vector<at::Tensor> pts_vec{per_token_scale};
  std::vector<at::Tensor> c_vec{c};
  at::TensorList a_list = at::TensorList(a_vec);
  at::TensorList b_list = at::TensorList(b_vec);
  at::TensorList scale_list = at::TensorList(scale_vec);
  at::TensorList pts_list = at::TensorList(pts_vec);
  at::TensorList c_list = at::TensorList(c_vec);
  EXEC_NPU_CMD(aclnnIndexGroupMatmul,
               a_list,
               b_list,
               scale_list,
               pts_list,
               group_list,
               c_list);
  return c;
}

std::tuple<at::Tensor, at::Tensor> dequant_swiglu_quant_impl_npu(
    const at::Tensor& x,
    bool activate_left,
    std::string quant_mode) {
  // swiglu + dynamic per-token quant (minimal path: no bias/scale inputs).
  // x:(rows,H) bf16/fp16. y:(rows,H/2) int8, scale:(rows,) fp32.
  // aclnn signature: x, weightScale, activationScale, bias, quantScale,
  //   quantOffset, groupIndex, activateLeft(bool), quantMode(char*),
  //   yOut, scaleOut.
  auto sizes = x.sizes().vec();
  const int64_t dim_x = static_cast<int64_t>(sizes.size());
  const int64_t H = sizes[dim_x - 1];

  std::vector<int64_t> y_shape(sizes.begin(), sizes.end());
  y_shape[dim_x - 1] = H / 2;
  std::vector<int64_t> scale_shape(sizes.begin(), sizes.end() - 1);

  at::Tensor y = at::empty(y_shape, x.options().dtype(at::ScalarType::Char));
  at::Tensor scale = at::empty(scale_shape,
                               x.options().dtype(at::ScalarType::Float));

  c10::optional<at::Tensor> weight_scale = c10::nullopt;
  c10::optional<at::Tensor> activation_scale = c10::nullopt;
  c10::optional<at::Tensor> bias = c10::nullopt;
  c10::optional<at::Tensor> quant_scale = c10::nullopt;
  c10::optional<at::Tensor> quant_offset = c10::nullopt;
  c10::optional<at::Tensor> group_index = c10::nullopt;

  char* quant_mode_c = const_cast<char*>(quant_mode.c_str());

  EXEC_NPU_CMD(aclnnDequantSwigluQuant,
               x,
               weight_scale,
               activation_scale,
               bias,
               quant_scale,
               quant_offset,
               group_index,
               activate_left,
               quant_mode_c,
               y,
               scale);
  return std::make_tuple(y, scale);
}

std::tuple<at::Tensor, at::Tensor> moe_grouped_matmul_swiglu_quant_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& x_scale,
    const at::Tensor& group_list) {
  // int8 grouped matmul + dequant + swiglu + dynamic per-token int8 quant.
  // x:(M,K) int8 ND, weight:(G,K,2N) int8 (ND passed, framework -> NZ),
  // weight_scale:(G,2N) float, x_scale:(M,) float, group_list:(G,) int64
  // cumulative offset. aclnn signature: x, weight, weightScale, xScale,
  // groupList, output, outputScale. output:(M,N) int8, outputScale:(M,) fp32.
  const int64_t M = x.sizes()[x.dim() - 2];
  const int64_t W = weight.sizes()[weight.dim() - 1];  // 2N
  const int64_t N = W / 2;
  at::Tensor y = at::empty({M, N}, x.options().dtype(at::ScalarType::Char));
  at::Tensor y_scale = at::empty({M},
                                 x.options().dtype(at::ScalarType::Float));
  EXEC_NPU_CMD(aclnnMoeGroupedMatmulSwigluQuant,
               x,
               weight,
               weight_scale,
               x_scale,
               group_list,
               y,
               y_scale);
  return std::make_tuple(y, y_scale);
}

// grouped_matmul_swiglu_quant_v2: same int8 pertoken chain as v1 but weight and
// weight_scale are TensorLists (one tensor per expert) and group_list is a 1D
// int64 per-group token count (groupListType=0).
// Wrapper passes per-expert lists directly: weight[g]:(K,2N) int8 already
// carrying FRACTAL_NZ format (set on the Python side via npu_format_cast, which
// performs the physical repack); weight_scale[g]:(2N,) float. Splitting on the
// C++ side would drop the NZ format back to ND, so lists are received as-is.
// x:(M,K) int8, x_scale:(M,) float, group_list:(G,) int64. Output y:(M,N) int8,
// y_scale:(M,) fp32. Optional inputs (weightAssistMatrix/bias/smoothScale/
// tuningConfig) must be empty (CheckAttrs enforces nullptr in this version).
// aclnn signature: x, weight(List), weightScale(List), weightAssistMatrix(List),
//   bias, xScale, smoothScale, groupList, dequantMode(i64), dequantDtype(i64),
//   quantMode(i64), groupListType(i64), tuningConfig(aclIntArray), swigluLimit
//   (double), output, outputScale.
std::tuple<at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_v2_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& x_scale,
    const at::Tensor& group_list,
    int64_t group_list_type) {
  const int64_t M = x.sizes()[x.dim() - 2];
  // Pertoken SINGLE-tensor scenario (CheckInputOutDimsForPertoken):
  // weight is 3D (E, K, 2N) FRACTAL_NZ; weight_scale is 2D (E, 2N). Both wrapped
  // in one-element TensorLists. Derive N (= half of 2N) from weight_scale.
  const int64_t W = weight_scale.sizes()[weight_scale.dim() - 1];  // 2N
  const int64_t N = W / 2;

  (void)group_list_type;  // v1 WeightNZ uses cumulative group_list (type-fixed)

  // Empty optional inputs for the int8 A8W8 path.
  at::Tensor bias = at::Tensor();
  at::Tensor offset = at::Tensor();

  at::Tensor y = at::empty({M, N}, x.options().dtype(at::ScalarType::Char));
  at::Tensor y_scale = at::empty({M},
                                 x.options().dtype(at::ScalarType::Float));
  // v1 WeightNZ has an outputOffset output; unused for pertoken int8, pass empty.
  at::Tensor y_offset = at::Tensor();

  // aclnnGroupedMatmulSwigluQuantWeightNZ (v1) IS the int8 A8W8 path:
  // x=INT8, weight=INT8(NZ, single tensor), weightScale=FLOAT32, xScale=FLOAT32,
  // output=INT8, outputScale=FLOAT32. The v2 WeightNz entry only accepts
  // MXFP8/MXFP4 (E4M3/E5M2 + E8M0 scale), incompatible with our int8 golden.
  // Signature: x, weight, bias, offset, weightScale, xScale, groupList,
  //            output, outputScale, outputOffset.
  EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantWeightNZ,
               x,
               weight,
               bias,
               offset,
               weight_scale,
               x_scale,
               group_list,
               y,
               y_scale,
               y_offset);
  return std::make_tuple(y, y_scale);
}

// moe_gating_top_k: normalize (softmax/sigmoid/softplus) + topk + gather +
// optional renorm + scale. Mirrors moe_gating_top_k_torch_adpt.h signature.
// Outputs: y=(rows,k) x.dtype; expert_idx=(rows,k) int32; out=(rows,expert_num)
// float (valid when out_flag=true).
std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_impl_npu(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps) {
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  auto x_size = x.sizes();
  const int64_t rows = x_size[0];
  const int64_t expert_num = x_size[1];

  at::Tensor y = at::empty({rows, k}, x.options());
  at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
  at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnMoeGatingTopK,
               x,
               bias,
               k,
               k_group,
               group_count,
               group_select_mode,
               renorm,
               norm_type,
               out_flag,
               routed_scaling_factor,
               eps,
               y,
               expert_idx,
               out);
  return std::make_tuple(y, expert_idx, out);
}

// moe_gating_top_k_hash: superset of moe_gating_top_k. When both input_ids and
// tid2eid are provided, expert_idx is looked up from the hash table
// (tid2eid[input_ids[row]*k : +k]) instead of running topk; otherwise falls
// back to topk. input_ids/tid2eid use int32 to hit the without_group
// <int32_t,int32_t> tilingbranch.
std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash_impl_npu(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& input_ids_opt,
    const c10::optional<at::Tensor>& tid2eid_opt,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps) {
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  const at::Tensor& input_ids =
      c10::value_or_else(input_ids_opt,[] { return at::Tensor(); });
  const at::Tensor& tid2eid =
      c10::value_or_else(tid2eid_opt, [] { return at::Tensor(); });
  auto x_size = x.sizes();
  const int64_t rows = x_size[0];
  const int64_t expert_num = x_size[1];

  at::Tensor y = at::empty({rows, k}, x.options());
  at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
  at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnMoeGatingTopKHash,
               x,
               bias,
               input_ids,
               tid2eid,
               k,
               k_group,
               group_count,
               group_select_mode,
               renorm,
               norm_type,
               out_flag,
               routed_scaling_factor,
               eps,
               y,
               expert_idx,
               out);
  return std::make_tuple(y, expert_idx, out);
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
  m.def("causal_conv1d_packed_qkv",
        &causal_conv1d_packed_qkv,
        "causal_conv1d_packed_qkv");
  m.def("causal_conv1d_packed_qkv_general",
        &causal_conv1d_packed_qkv_general,
        "causal_conv1d_packed_qkv_general");
  m.def("causal_conv1d_packed_qkv_general_bf16",
        &causal_conv1d_packed_qkv_general_bf16,
        "causal_conv1d_packed_qkv_general_bf16");
  m.def("recurrent_gated_delta_rule", &recurrent_gated_delta_rule, "recurrent_gated_delta_rule");
  m.def("rec_constrained_topk", &rec_constrained_topk_impl_npu, "rec_constrained_topk");
  m.def("mega_chunk_gdn", &mega_chunk_gdn, "mega_chunk_gdn");
  m.def("layer_norm_fwd", &layer_norm_fwd_impl_npu, "layer_norm_fwd");
  m.def("moe_fused_add_topk", &moe_fused_add_topk_impl_npu, "moe_fused_add_topk");
  m.def("moe_fused_reducesum_div", &moe_fused_reducesum_div_impl_npu, "moe_fused_reducesum_div");
  m.def("replace_token", &replace_token_impl_npu, "replace_token");
  m.def("convert_kv_cache_format", &convert_kv_cache_format_impl_npu, "convert_kv_cache_format");
  m.def("scatter_nd_update_v2", &scatter_nd_update_v2_impl_npu, "scatter_nd_update_v2");
  m.def("hc_post", &hc_post_impl_npu, "hc_post");
  m.def("add_rms_norm_bias", &add_rms_norm_bias_impl_npu, "add_rms_norm_bias");
  m.def("moe_init_routing_custom", &moe_init_routing_custom_impl_npu, "moe_init_routing_custom");
  m.def("moe_init_routing_v3", &moe_init_routing_v3_impl_npu, "moe_init_routing_v3");
  m.def("hc_pre_sinkhorn", &hc_pre_sinkhorn_impl_npu, "hc_pre_sinkhorn");
  m.def("pp_matmul_opt", &pp_matmul_opt_impl_npu, "pp_matmul_opt");
  m.def("moe_grouped_matmul", &moe_grouped_matmul_impl_npu, "moe_grouped_matmul");
  m.def("index_group_matmul", &index_group_matmul_impl_npu, "index_group_matmul");
  m.def("dequant_swiglu_quant", &dequant_swiglu_quant_impl_npu, "dequant_swiglu_quant");
  m.def("moe_grouped_matmul_swiglu_quant", &moe_grouped_matmul_swiglu_quant_impl_npu, "moe_grouped_matmul_swiglu_quant");
  m.def("grouped_matmul_swiglu_quant_v2", &grouped_matmul_swiglu_quant_v2_impl_npu, "grouped_matmul_swiglu_quant_v2");
  m.def("moe_gating_top_k", &moe_gating_top_k_impl_npu, "moe_gating_top_k");
  m.def("moe_gating_top_k_hash", &moe_gating_top_k_hash_impl_npu, "moe_gating_top_k_hash");
  m.def("hc_pre_inv_rms", &hc_pre_inv_rms_impl_npu, "hc_pre_inv_rms");
}
