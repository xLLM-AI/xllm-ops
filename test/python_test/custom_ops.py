import torch
import custom_ops_lib


# x_attention
def x_attention_npu(query, key_cache, value_cache, unshared_key, unshared_value, block_tables, actual_shared_kvlen, decode_step):
    return custom_ops_lib.x_attention(query, key_cache, value_cache, unshared_key, unshared_value, block_tables, actual_shared_kvlen, decode_step)

# reshape cache kv
def cache_unshared_kv_npu(x_key_block, x_value_block, curr_key, curr_value, block_table, decode_step):
    return custom_ops_lib.cache_unshared_kv(x_key_block, x_value_block, curr_key, curr_value, block_table, decode_step)

# select unshared kv
def select_unshared_kv_npu(beam_index, x_key_block, x_value_block, block_table, group_token_num, decode_step, beam_size):
    return custom_ops_lib.select_unshared_kv(beam_index, x_key_block, x_value_block, block_table, group_token_num, decode_step, beam_size)