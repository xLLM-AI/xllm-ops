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


def gamma_add_rms_norm_npu(x1, x2, gamma, eps=1e-6,
                           add_gamma_offset=False):
    return custom_ops_lib.gamma_add_rms_norm(
        x1, x2, gamma, eps, add_gamma_offset)


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


# moe_init_routing_v3
# Same aclnn signature as moe_init_routing_custom (only the backend op differs).
# row_idx_type: GATHER=0 / SCATTER=1
# expert_tokens_num_type: CUMSUM=0 / COUNT=1 / KEY_VALUE=2
# drop_pad_mode: DROPLESS=0 / DROP_PAD=1
# quant_mode: UNQUANT=-1 / DYNAMIC_QUANT=1
def moe_init_routing_v3_npu(x, expert_idx, active_num=-1, expert_num=8,
                            drop_pad_mode=0, expert_tokens_num_type=1,
                            expert_tokens_num_flag=True, quant_mode=-1,
                            row_idx_type=0):
    return custom_ops_lib.moe_init_routing_v3(
        x, expert_idx, active_num, expert_num, drop_pad_mode,
        expert_tokens_num_type, expert_tokens_num_flag, quant_mode, row_idx_type)


# moe_gating_top_k
# norm_type: SOFTMAX=0 / SIGMOID=1 / SOFTPLUS=2
# renorm: only 0 (RENORM_NO) supported
# out_flag=True to also output the full-expert xnorm (rows, expert_num) fp32.
# Returns (y, expert_idx, out): y=(rows,k), expert_idx=(rows,k) int32,
# out=(rows,expert_num) fp32.
def moe_gating_top_k_npu(x, k, bias=None, k_group=1, group_count=1,
                         group_select_mode=0, renorm=0, norm_type=0,
                         out_flag=False, routed_scaling_factor=1.0,
                         eps=1e-20):
    return custom_ops_lib.moe_gating_top_k(
        x, bias, k, k_group, group_count, group_select_mode, renorm,
        norm_type, out_flag, routed_scaling_factor, eps)


# moe_gating_top_k_hash
# Superset of moe_gating_top_k. When both input_ids and tid2eid are provided
# (hashFlag=1), expert_idx is looked up from tid2eid[input_ids[row]*k : +k]
# instead of topk. input_ids/tid2eid should be int32 to hit the without_group
# <int32,int32> branch.
def moe_gating_top_k_hash_npu(x, k, bias=None, input_ids=None, tid2eid=None,
                              k_group=1, group_count=1, group_select_mode=0,
                              renorm=0, norm_type=0, out_flag=False,
                              routed_scaling_factor=1.0, eps=1e-20):
    return custom_ops_lib.moe_gating_top_k_hash(
        x, bias, input_ids, tid2eid, k, k_group, group_count,
        group_select_mode, renorm, norm_type, out_flag,
        routed_scaling_factor, eps)


# hc_pre_inv_rms
# y = 1/sqrt(mean(x^2 over last two dims) + eps), output always fp32.
# x: (b,s,hc,d) -> y: (b,s,1)  or  x: (t,hc,d) -> y: (t,1).
# NOTE: large_d kernel path is only enabled when R = hc*d == 28672.
def hc_pre_inv_rms_npu(x, epsilon=1e-6):
    return custom_ops_lib.hc_pre_inv_rms(x, epsilon)


# laser_attention: standard flash attention (BNSD layout), used by wan2.2.
# query/key/value: [B, N, S, D] fp16/bf16. Returns (softmax_log_max_sum, attention_out) in fp32.
def laser_attention_npu(query, key, value, scale_value, head_num,
                        input_layout="BNSD", atten_mask=None, alibi_mask=None,
                        drop_mask=None, keep_prob=1.0, pre_tokens=2147483647,
                        next_tokens=1, is_high_precision=True):
    return custom_ops_lib.laser_attention(
        query, key, value, atten_mask, alibi_mask, drop_mask,
        scale_value, head_num, input_layout, keep_prob, pre_tokens,
        next_tokens, is_high_precision,
    )


# x_flash_attention_infer: paged-KV flash-decoding attention (TND layout, causal mask).
# extra_tiling encodes per-core KV-split task assignment (SplitKvExtraInfo).
# The host tiling does NOT fill this; we build a single-core layout: core 0 handles
# the whole (batch x kvHead) range without KV-split (isSplitKV=false -> writes attn_out
# directly), all other cores are skipped (startBIdx = 0xFFFFFFFF).
# Layout (matches op_kernel SplitKvExtraInfo, ALL sizes fixed by struct):
#   CoreNode  = 6*uint32 + 2*uint64 = 40 bytes = 10 uint32 slots
#   SplitNode = 6*uint32 + 2*uint64 = 40 bytes = 10 uint32 slots
#   SplitKvExtraInfo = CoreNode[25] + SplitNode[25] +uint32 totalSplitNodeNum
#                    = 25*40 + 25*40 + 4 = 2004 bytes = 501 int32 elements
# NOTE: arrays are ALWAYS 25 (fixed by struct). cube core num (24) only decides
#       how many entries beyond core 0 are marked as "skip" (startBIdx=UINT32_MAX).
_XFA_MAX_KV_STACK_LEN = 512    # op_kernel MAX_KV_STACK_LEN
_XFA_NODE_ARRAY_LEN = 25       # coreInfo[25] / splitInfo[25] (fixed by struct)
_XFA_NODE_U32 = 10             # 6 u32 + 2 u64 (=4 u32) per node


def _build_xfa_extra_tiling(batch, kv_heads, kv_seqlen, device):
    import numpy as np
    UINT32_MAX = 0xFFFFFFFF
    cur_ks_block_num = (kv_seqlen + _XFA_MAX_KV_STACK_LEN - 1) // _XFA_MAX_KV_STACK_LEN
    n = _XFA_NODE_ARRAY_LEN
    core_info = np.zeros((n, _XFA_NODE_U32), dtype=np.uint32)
    # core 0 = single worker covering the whole (batch x kvHead) range, no KV split.
    # CoreNode u32 slots: [startBIdx, startN1Idx, startS2Idx, endBIdx, endN1Idx,
    #                      endS2Idx, lseOff_lo, lseOff_hi, oOff_lo, oOff_hi]
    core_info[0, 3] = batch - 1            # endBIdx
    core_info[0, 4] = kv_heads - 1         # endN1Idx
    core_info[0, 5] = cur_ks_block_num     # endS2Idx (== curKSBlockNum -> isSplitKV false)
    # start* and u64 offsets already 0
    for c in range(1, n):
        core_info[c, 0] = UINT32_MAX       # startBIdx -> skip core -> SyncAll only
    split_info = np.zeros((n, _XFA_NODE_U32), dtype=np.uint32)  # unused (totalSplitNodeNum=0)
    total_split_node_num = np.zeros((1,), dtype=np.uint32)      # 0 -> no combineScale work
    buf = np.concatenate([core_info.reshape(-1), split_info.reshape(-1),
                          total_split_node_num])
    t = torch.from_numpy(buf.view(np.int32).copy()).to(device)
    return t


# query: [numTokens, qHead, headDim] fp16/bf16 (TND). key_cache/value_cache paged:
# [numBlocks, blockSize, kvHead, headDim]. Returns attn_out [numTokens, qHead, headDim].
def x_flash_attention_infer_npu(query, key_cache, value_cache, block_table,
                                actual_q_lens, actual_kv_lens, q_head, kv_head,
                                scale, batch, kv_seqlen, mask=None, layout="TND"):
    extra_tiling = _build_xfa_extra_tiling(batch, kv_head, kv_seqlen, query.device)
    return custom_ops_lib.x_flash_attention_infer(
        query, key_cache, value_cache, mask, block_table,
        actual_q_lens, actual_kv_lens, extra_tiling,
        layout, q_head, kv_head, scale,
    )


# multi_latent_attention: DeepSeek MLA split-cache paged decode.
# Q/KV split into a NoPE (latent, 512-dim) part and a RoPE (64-dim) part; the
# kernel hardcodes embedding=512, rope=64. K = [kvCache|kvCacheRope] (576),
# V = kvCache NoPE part (512). Host tiling is fully automatic (no extra_tiling).
#   query      [numTokens, q_head, 512] fp16       queryRope   [numTokens, q_head, 64]
#   kvCache    [numBlocks, blockSize, kv_head, 512] kvCacheRope [numBlocks, blockSize, kv_head, 64]
#   block_tables [batch, maxBlocks] int32           contextLens [batch] int32
# Returns attenOut [numTokens, q_head, 512]. type=0 SPLIT_CACHE, maskType=0 NONE,
# isRing=0 (no lse). Use q_head < 128 to avoid the MTP TP1 special path.
def multi_latent_attention_npu(query, query_rope, kv_cache, kv_cache_rope,
                               block_tables, context_lens, q_head, kv_head,
                               scale, kv_seqlen, q_seqlen=None, mask=None,
                               mask_type=0, is_ring=0):
    if q_seqlen is None:
        q_seqlen = [-1]
    return custom_ops_lib.multi_latent_attention(
        query, query_rope, kv_cache, kv_cache_rope, block_tables, context_lens,
        mask, None, None, None,
        0, q_head, scale, kv_head, mask_type,
        q_seqlen, kv_seqlen, is_ring,
    )


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


# grouped_matmul_swiglu_quant_v2: int8 pertoken chain identical to v1. The v2
# aclnn WeightNz entry only accepts MXFP8/MXFP4 (E4M3/E5M2 + E8M0 scale) on
# CANN 8.5.0, incompatible with our int8 golden. The C++ impl therefore routes
# to the v1 aclnnGroupedMatmulSwigluQuantWeightNZ (true int8 A8W8 path):
#   x=int8 (M,K) ND, weight=int8 (K,2N) NZ single tensor, weight_scale float32
#   (2N,), x_scale float32 (M,), group_list int64 CUMULATIVE offsets.
# Wrapper takes packed weight (G,K,2N) ND / weight_scale (G,2N) and a per-group
# COUNT group_list (groupListType=0); it casts weight to NZ and converts the
# count list to the cumulative form the v1 kernel expects.
def grouped_matmul_swiglu_quant_v2_npu(x, weight, weight_scale, x_scale,
                                    group_list, group_list_type=0):
    import torch
    import torch_npu
    torch_npu.npu.config.allow_internal_format = True
    E, K, N2 = weight.shape
    # v1 WeightNZ expects int8 weight in FRACTAL_NZ (single 3D tensor per the
    # packed (E,K,2N) layout). Cast the whole tensor once; op keeps NZ storage.
    weight_nz = torch_npu.npu_format_cast(weight.clone().contiguous(), 29)
    weight_scale = weight_scale.contiguous()  # (E, 2N) float32 ND
    # group_list_type 0 = per-group count -> convert to cumulative offsets that
    # the v1 kernel consumes.
    if group_list_type == 0:
        group_list = torch.cumsum(group_list.to(torch.int64), dim=0)
    return custom_ops_lib.grouped_matmul_swiglu_quant_v2(
        x, weight_nz, weight_scale, x_scale, group_list,
        group_list_type)
