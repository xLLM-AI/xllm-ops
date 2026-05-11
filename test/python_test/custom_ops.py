import torch
import custom_ops_lib


# x_attention
def x_attention_npu(query, key_cache, value_cache, unshared_key, unshared_value, actual_shared_kvlen, decode_step, 
                    shared_block_tables = None, 
                    unshared_block_tables = None):
    return custom_ops_lib.x_attention(query, key_cache, value_cache, unshared_key, unshared_value, 
                                      shared_block_tables, unshared_block_tables, actual_shared_kvlen, decode_step)

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
