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
