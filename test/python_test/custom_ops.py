import torch
import custom_ops_lib


CHUNK_SIZE = 128


# x_attention
def x_attention_npu(query, key_cache, value_cache, unshared_key, unshared_value, actual_shared_kvlen, decode_step, 
                    shared_block_tables = None, 
                    unshared_block_tables = None,
                    scale_value = None):
    if scale_value is None:
        scale_value = 0.0
    return custom_ops_lib.x_attention(query, key_cache, value_cache, unshared_key, unshared_value, 
                                      shared_block_tables, unshared_block_tables, actual_shared_kvlen, decode_step, scale_value)

# reshape cache kv
def cache_unshared_kv_npu(x_key_block, x_value_block, curr_key, curr_value, block_table, decode_step):
    return custom_ops_lib.cache_unshared_kv(x_key_block, x_value_block, curr_key, curr_value, block_table, decode_step)

# select unshared kv
def select_unshared_kv_npu(beam_index, x_key_block, x_value_block, block_table, group_token_num, decode_step, beam_size, layer_num):
    return custom_ops_lib.select_unshared_kv(beam_index, x_key_block, x_value_block, block_table, group_token_num, decode_step, beam_size, layer_num)

# beam search group
def beam_search_group_npu(log_probs, top_tokens, top_probs, sequence, current_step, top_k):
    return custom_ops_lib.beam_search_group(log_probs, top_tokens, top_probs, sequence, current_step, top_k)


def beam_search_rec_final_select_npu(log_probs: torch.Tensor,
                                     top_tokens: torch.Tensor,
                                     top_probs: torch.Tensor,
                                     sequence: torch.Tensor,
                                     current_step: int,
                                     result_width: int
                                     ) -> tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor, torch.Tensor]:
    return custom_ops_lib.beam_search_rec_final_select(
        log_probs, top_tokens, top_probs, sequence, current_step, result_width
    )

# causal_conv1d_fn
def causal_conv1d_npu( x, weight, conv_state, bias_opt, query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                    num_accepted_tokens_opt,
                    activation_mode,
                    pad_slot_id,
                    run_mode):
    return  custom_ops_lib.causal_conv1d( x, weight, conv_state, bias_opt, query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                    num_accepted_tokens_opt,
                    activation_mode,
                    pad_slot_id,
                    run_mode)

# recurrent_gated_delta_rule
def recurrent_gated_delta_rule_npu(query, key, value, state, beta,
                                scale,
                                actual_seq_lengths,
                                ssm_state_indices,
                                num_accepted_tokens,
                                g,
                                gk):
    return  custom_ops_lib.recurrent_gated_delta_rule(query, key, value, state, beta,
                                scale,
                                actual_seq_lengths,
                                ssm_state_indices,
                                num_accepted_tokens,
                                g,
                                gk)


def rec_constrained_topk_npu(logits, sequence_group, first_token_ids,
                             prefix1_offsets, prefix1_values,
                             prefix1_pair_keys, prefix2_value_offsets,
                             prefix2_values, temperatures, current_step,
                             top_k, max_prefix1_degree,
                             max_prefix2_degree):
    return custom_ops_lib.rec_constrained_topk(
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
    )


_MEGA_MASK_CACHE = {}
_MEGA_MINUS_IDENTITY_CACHE = {}


def _mega_device_key(device):
    index = 0 if device.index is None else device.index
    return device.type, index


def _mega_get_masks(device):
    key = _mega_device_key(device)
    if key not in _MEGA_MASK_CACHE:
        mask_lower = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device), diagonal=-1).float()
        mask_full = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device), diagonal=0).float()
        _MEGA_MASK_CACHE[key] = (mask_lower, mask_full)
    return _MEGA_MASK_CACHE[key]


def _mega_get_minus_identity(device):
    key = _mega_device_key(device)
    if key not in _MEGA_MINUS_IDENTITY_CACHE:
        minus_identity = torch.zeros(CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16)
        minus_identity.fill_diagonal_(-1)
        _MEGA_MINUS_IDENTITY_CACHE[key] = minus_identity
    return _MEGA_MINUS_IDENTITY_CACHE[key]


def _mega_total_chunks(cu_seqlens):
    cu = cu_seqlens.cpu().tolist()
    return sum((int(end) - int(start) + CHUNK_SIZE - 1) // CHUNK_SIZE for start, end in zip(cu, cu[1:]))


# layer_norm_fwd (gated group (rms)norm, Mamba2-style)
def layer_norm_fwd_npu(x, weight, bias=None, z=None, eps=1e-6, group_size=-1,
                       norm_before_gate=True, is_rms_norm=False):
    return custom_ops_lib.layer_norm_fwd(
        x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)


# moe_fused_add_topk (MoE fused sigmoid + group topk routing)
def moe_fused_add_topk_npu(x, add_num, group_num, group_topk, top_n, top_k,
                           mapping_num=None, mapping_table=None,
                           activate_type=0, is_norm=True, scale=1.0,
                           enable_expert_mapping=False):
    return custom_ops_lib.moe_fused_add_topk(
        x, add_num, mapping_num, mapping_table,
        group_num, group_topk, top_n, top_k,
        activate_type, is_norm, scale, enable_expert_mapping)


# moe_fused_reducesum_div (row-wise normalization: x / sum(x, dim=-1))
def moe_fused_reducesum_div_npu(input):
    return custom_ops_lib.moe_fused_reducesum_div(input)


# replace_token (replace negative token ids with last-step output tokens)
def replace_token_npu(forked_token_ids, last_step_output_token_ids):
    return custom_ops_lib.replace_token(forked_token_ids,
                                        last_step_output_token_ids)


# convert_kv_cache_format (in-place ND2NZ rewrite of k/v cache blocks)
def convert_kv_cache_format_npu(k_cache_ptr, v_cache_ptr, kv_cache_offset,
                                kv_seq_len, is_prefill, num_kv_heads,
                                head_size_k, head_size_v):
    return custom_ops_lib.convert_kv_cache_format(
        k_cache_ptr, v_cache_ptr, kv_cache_offset, kv_seq_len,
        is_prefill, num_kv_heads, head_size_k, head_size_v)


# scatter_nd_update_v2 (in-place scatter: var.flat[gmIdx*L : ...] = updates[row])
def scatter_nd_update_v2_npu(var, indices, updates, strides):
    return custom_ops_lib.scatter_nd_update_v2(var, indices, updates, strides)


# hc_post (per-token fused combine:
#   y[b,hc,d] = sum_j residual[b,j,d]*comb[b,j,hc] + x[b,d]*post[b,hc])
def hc_post_npu(x, residual, post, comb):
    return custom_ops_lib.hc_post(x, residual, post, comb)


def add_rms_norm_bias_npu(x1, x2, gamma, beta=None, eps=1e-6):
    return custom_ops_lib.add_rms_norm_bias(x1, x2, gamma, beta, eps)


# hc_pre_sinkhorn (per-token: pre/post gating + sinkhorn-normalized comb_frag)
# mixes last dim = 2*hc_mult + hc_mult^2 ([pre | post | comb]).
# returns (y[bs,d] bf16, post[bs,hc_mult] fp32, comb_frag[bs,hc_mult,hc_mult] fp32)
def hc_pre_sinkhorn_npu(mixes, rsqrt, hc_scale, hc_base, x,
                        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6):
    return custom_ops_lib.hc_pre_sinkhorn(
        mixes, rsqrt, hc_scale, hc_base, x,
        hc_mult, hc_sinkhorn_iters, hc_eps)


# moe_init_routing_custom
# row_idx_type: GATHER=0 / SCATTER=1
# expert_tokens_num_type: CUMSUM=0 / COUNT=1 / KEY_VALUE=2
# drop_pad_mode: DROPLESS=0 / DROP_PAD=1
# quant_mode: UNQUANT=-1 / DYNAMIC_QUANT=1
def moe_init_routing_custom_npu(x, expert_idx, active_num=-1, expert_num=8,
                                drop_pad_mode=0, expert_tokens_num_type=1,
                                expert_tokens_num_flag=True, quant_mode=-1,
                                row_idx_type=0):
    return custom_ops_lib.moe_init_routing_custom(
        x, expert_idx, active_num, expert_num, drop_pad_mode,
        expert_tokens_num_type, expert_tokens_num_flag, quant_mode, row_idx_type)


# mega_chunk_gdn
def mega_chunk_gdn_npu(q, k, v, g, beta, scale=None, initial_state=None,
                       output_final_state=False, cu_seqlens=None):
    if scale is None:
        scale = k.shape[-1] ** -0.5

    q_dtype, k_dtype, v_dtype = q.dtype, k.dtype, v.dtype
    q, k, v, beta = (t.half() for t in (q, k, v, beta))

    _, total_tokens, _, _ = q.shape
    num_value_heads = v.shape[-2]
    if cu_seqlens is None:
        cu32 = torch.tensor([0, total_tokens], dtype=torch.int32, device=q.device)
    else:
        cu32 = cu_seqlens.to(device=q.device, dtype=torch.int32)

    num_sequences = cu32.numel() - 1
    num_chunks = _mega_total_chunks(cu32)
    num_matrices = num_chunks * num_value_heads
    has_initial_state = initial_state is not None
    if has_initial_state:
        initial_state_arg = initial_state.half()
    else:
        initial_state_arg = torch.zeros(
            num_sequences,
            num_value_heads,
            q.shape[-1],
            q.shape[-1],
            device=q.device,
            dtype=torch.float16,
        )

    mask_lower, mask_full = _mega_get_masks(q.device)
    minus_identity = _mega_get_minus_identity(q.device)

    (out, g_sum, _, _, _, _, a_inv, w, _, h, v_new, final_state) = custom_ops_lib.mega_chunk_gdn(
        q,
        k,
        v,
        g,
        beta,
        mask_lower,
        mask_full,
        minus_identity,
        cu32,
        initial_state_arg,
        num_matrices,
        has_initial_state,
    )

    h = h.view(1, num_chunks, num_value_heads, q.shape[-1], q.shape[-1])
    final_state_out = None
    if output_final_state:
        final_state_out = final_state.view(num_sequences, num_value_heads, q.shape[-1], q.shape[-1]).to(torch.float32)

    return (
        g_sum,
        (out * scale).to(q_dtype),
        a_inv.to(k_dtype),
        final_state_out,
        w.to(k_dtype),
        h.to(k_dtype),
        v_new.to(v_dtype),
    )


# pp_matmul_opt: c = a @ b^T
# a:(M,K), b:(N,K) -> c:(M,N), dtype bf16/fp16.
# constraints: K % 256 == 0, N % 256 == 0, M <= 24.
def pp_matmul_opt_npu(a, b):
    return custom_ops_lib.pp_matmul_opt(a, b)


# moe_grouped_matmul: grouped matmul, no quantization.
# x:(M,K) ND, weight:(G,K,N) ND ((G,N,K) when transpose_weight),
# group_list:(G,2) int32/int64 key-value [group_idx, count], y:(M,N).
def moe_grouped_matmul_npu(x, weight, group_list, transpose_weight=False):
    return custom_ops_lib.moe_grouped_matmul(x, weight, group_list, transpose_weight)


# index_group_matmul: int8 grouped matmul + dequant.
# a:(M,K) int8, b:(G,K,N) int8, scale:(G,N) bf16, per_token_scale:(M,) float,
# group_list:(G,) int64 cumulative offset. Output c:(M,N) bf16.
def index_group_matmul_npu(a, b, scale, per_token_scale, group_list):
    return custom_ops_lib.index_group_matmul(a, b, scale, per_token_scale, group_list)


# dequant_swiglu_quant: swiglu + dynamic per-token quant.
# x:(rows,H) bf16/fp16 (simplest path, no bias/scale). Output y:(rows,H/2) int8, scale:(rows,) fp32.
# when activate_left=False, gate=x[:,H/2:], up=x[:,:H/2].
def dequant_swiglu_quant_npu(x, activate_left=False, quant_mode="dynamic"):
    return custom_ops_lib.dequant_swiglu_quant(x, activate_left, quant_mode)


# moe_grouped_matmul_swiglu_quant: int8 grouped matmul + dequant + swiglu + dynamic quant.
# x:(M,K) int8, weight:(G,K,2N) int8 (passed as ND, framework converts to NZ), weight_scale:(G,2N) float,
# x_scale:(M,) float, group_list:(G,) int64 cumulative offset.
# Output y:(M,N) int8, y_scale:(M,) fp32. N=2N/2.
def moe_grouped_matmul_swiglu_quant_npu(x, weight, weight_scale, x_scale, group_list):
    return custom_ops_lib.moe_grouped_matmul_swiglu_quant(
        x, weight, weight_scale, x_scale, group_list)
