// Qwen3.5 prefill Conv-to-gated-RMSNorm single-launch PTO kernel.

#include "gdn_prefill_arch.h"

struct MegaGdnPrefillOpKernelTilingData {
    uint32_t block_dim;
    uint32_t target_arch;
    uint32_t num_matrices;
    uint32_t batch_size;
    uint32_t num_heads;
    uint32_t num_key_heads;
    uint32_t token_block_size;
    uint32_t token_block_count;
    uint32_t base_dim;
    uint32_t base_dim_count;
    uint32_t conv_dim;
    uint32_t conv_state_slots;
    uint32_t conv_state_len;
    uint32_t ssm_state_slots;
    uint32_t checkpoint_stride;
    int64_t total_tokens;
    uint64_t ffts_addr;
};

#define GDN_PREFILL_COMPUTE_DTYPE half

#define GDN_COMPUTE_DTYPE GDN_PREFILL_COMPUTE_DTYPE
#define GDN_PUBLIC_DTYPE DTYPE_MIXED_QKV
#define MEGA_CHUNK_GDN_HELPERS_ONLY
#define MEGA_CHUNK_GDN_HELPER_NAMESPACE qwen35_e2e_pto
#define MEGA_GDN_BUILD_REV 2026082515
#if defined(GDN_PREFILL_ARCH_A2A3)
#define MEGA_CHUNK_GDN_PRECOMPUTED_SOLVE_AUX
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_KK
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#endif
// The included helpers keep the public BF16 boundary and cast into FP16 on
// A2/A3.
#include "../../mega_chunk_gdn/op_kernel/mega_chunk_gdn.cpp"
#if defined(GDN_PREFILL_ARCH_A2A3)
#undef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#undef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_KK
#undef MEGA_CHUNK_GDN_PRECOMPUTED_SOLVE_AUX
#endif
#undef MEGA_CHUNK_GDN_HELPER_NAMESPACE
#undef MEGA_CHUNK_GDN_HELPERS_ONLY
#undef GDN_PUBLIC_DTYPE
#undef GDN_COMPUTE_DTYPE

#define GDN_PREFILL_PACKED_QKV_DTYPE GDN_PREFILL_COMPUTE_DTYPE
#include "gdn_prefill_frontend.h"
#undef GDN_PREFILL_PACKED_QKV_DTYPE

namespace {
constexpr uint64_t kAlignBytes = 512;
constexpr uint64_t kDtypeBytes = 2;
constexpr uint64_t kFloatBytes = 4;
constexpr uint64_t kHeadDim = 128;
constexpr uint64_t kChunkSize = 128;

AICORE inline uint64_t AlignWorkspace(uint64_t bytes)
{
    return (bytes + kAlignBytes - 1) / kAlignBytes * kAlignBytes;
}
}  // namespace

extern "C" __global__ __aicore__ void GDN_KERNEL_NAME(
    GM_ADDR mixed_qkv_ptr, GM_ADDR b_ptr, GM_ADDR a_ptr, GM_ADDR z_ptr,
    GM_ADDR conv_weight_ptr, GM_ADDR conv_state_ptr, GM_ADDR a_log_ptr,
    GM_ADDR dt_bias_ptr, GM_ADDR conv_state_read_indices_ptr,
    GM_ADDR conv_state_write_indices_ptr, GM_ADDR ssm_state_read_indices_ptr,
    GM_ADDR ssm_state_write_indices_ptr, GM_ADDR ssm_cache_ptr,
    GM_ADDR mask_lower_ptr, GM_ADDR mask_full_ptr, GM_ADDR minus_identity_ptr,
    GM_ADDR cu_seqlens_ptr, GM_ADDR norm_weight_ptr, GM_ADDR norm_output_ptr,
    GM_ADDR conv_state_out_ptr, GM_ADDR ssm_cache_out_ptr,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaGdnPrefillOpKernelTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaGdnPrefillOpKernelTilingData,
                                tiling_data, tiling);

#ifdef __CCE_AICORE__
    if (tiling_data.target_arch != GDN_PREFILL_ARCH_ID) {
        return;
    }
    GM_ADDR user_workspace = AscendC::GetUserWorkspace(workspace);
    const uint32_t block_dim = tiling_data.block_dim;
    const uint32_t batch_size = tiling_data.batch_size;
    const uint32_t num_heads = tiling_data.num_heads;
    const uint32_t num_key_heads = tiling_data.num_key_heads;
    const int64_t total_tokens = tiling_data.total_tokens;
    const uint64_t tokens = static_cast<uint64_t>(total_tokens);
    const uint64_t heads = num_heads;
    const uint64_t key_heads = num_key_heads;
    const uint64_t matrices = tiling_data.num_matrices;
    uint64_t offset = 0;

    GM_ADDR compact_conv_state_snapshot_ptr = user_workspace + offset;
    offset += AlignWorkspace(
        static_cast<uint64_t>(batch_size) * tiling_data.conv_state_len *
        tiling_data.conv_dim * kDtypeBytes);
    GM_ADDR packed_qkv_compute_ptr = user_workspace + offset;
    offset += AlignWorkspace(
        tokens * tiling_data.conv_dim * kDtypeBytes);
    GM_ADDR g_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kFloatBytes);
    GM_ADDR beta_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kDtypeBytes);
    GM_ADDR beta_compute_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kDtypeBytes);
    GM_ADDR minus_identity_compute_ptr = user_workspace + offset;
    offset += AlignWorkspace(3 * kChunkSize * kChunkSize * kDtypeBytes);
    GM_ADDR g_sum_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kFloatBytes);
    GM_ADDR g_t_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kFloatBytes);
    GM_ADDR beta_t_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kDtypeBytes);
    GM_ADDR a_matrix_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kChunkSize * kDtypeBytes);
    GM_ADDR a_inv_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kChunkSize * kDtypeBytes);
    GM_ADDR w_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kHeadDim * kDtypeBytes);
    GM_ADDR u_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kHeadDim * kDtypeBytes);
    GM_ADDR s_ptr = user_workspace + offset;
    offset += AlignWorkspace(matrices * kHeadDim * kHeadDim * kDtypeBytes);
    GM_ADDR v_new_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kHeadDim * kDtypeBytes);

    const uint64_t tile_bytes = kChunkSize * kChunkSize * kDtypeBytes;
    GM_ADDR kkt_workspace_ptr = user_workspace + offset;
    offset += static_cast<uint64_t>(block_dim) * 2 * tile_bytes;
    GM_ADDR wy_workspace_a1_ptr = user_workspace + offset;
    offset += static_cast<uint64_t>(block_dim) * tile_bytes;
    GM_ADDR wy_workspace_a2_ptr = user_workspace + offset;
    offset += static_cast<uint64_t>(block_dim) * tile_bytes;
    GM_ADDR h_workspace_unaligned = user_workspace + offset;
    const uint64_t h_workspace_address =
        (reinterpret_cast<uint64_t>(h_workspace_unaligned) +
         GDN_H_WORKSPACE_ALIGNMENT_BYTES - 1) &
        ~(static_cast<uint64_t>(GDN_H_WORKSPACE_ALIGNMENT_BYTES) - 1);
    const uint64_t h_workspace_phased_address =
        h_workspace_address + GDN_H_WORKSPACE_PHASE_BYTES;
    GM_ADDR h_workspace_ptr =
        reinterpret_cast<GM_ADDR>(h_workspace_phased_address);
    GM_ADDR o_workspace_qk_ptr =
        h_workspace_ptr + static_cast<uint64_t>(block_dim) *
                              8 * (tile_bytes +
                                   GDN_H_WORKSPACE_PAD_BYTES);
    GM_ADDR o_workspace_qs_ptr =
        o_workspace_qk_ptr + static_cast<uint64_t>(block_dim) * tile_bytes;
    GM_ADDR o_workspace_gated_ptr =
        o_workspace_qs_ptr + static_cast<uint64_t>(block_dim) * tile_bytes;

    GdnPrefillFrontendTilingData frontend_tiling{};
    frontend_tiling.num_heads = num_heads;
    frontend_tiling.num_key_heads = num_key_heads;
    frontend_tiling.token_block_size = tiling_data.token_block_size;
    frontend_tiling.token_block_count = tiling_data.token_block_count;
    frontend_tiling.base_dim = tiling_data.base_dim;
    frontend_tiling.base_dim_count = tiling_data.base_dim_count;
    frontend_tiling.conv_dim = tiling_data.conv_dim;
    frontend_tiling.conv_state_slots = tiling_data.conv_state_slots;
    frontend_tiling.total_tokens = total_tokens;

    gdn_prefill_frontend::RunConv(
        mixed_qkv_ptr, conv_weight_ptr, conv_state_ptr, conv_state_out_ptr,
        conv_state_read_indices_ptr, conv_state_write_indices_ptr,
        cu_seqlens_ptr, packed_qkv_compute_ptr,
        compact_conv_state_snapshot_ptr, frontend_tiling,
        batch_size, tiling_data.conv_state_len);

#if defined(GDN_PREFILL_ARCH_A5)
    // RunConv owns a temporary TPipe and publishes both packed QKV and the
    // final convolution state through MTE3.  Drain those stores before the
    // same AIV initializes the gate stage and reuses pipe/event resources.
    // PR #41's intra-block flags order peer consumers, but they do not by
    // themselves protect this local TPipe lifetime boundary.
    pipe_barrier(PIPE_ALL);
#endif

#ifdef E2E_STOP_AFTER_CONV
    pipe_barrier(PIPE_ALL);
    return;
#endif

    const bool round_g_to_bf16 =
#if defined(GDN_PREFILL_ARCH_A5)
        true;
#else
        tiling_data.checkpoint_stride > 1;
#endif
    gdn_prefill_frontend::PrepareGate<GDN_PREFILL_COMPUTE_DTYPE>(
        a_ptr, b_ptr, a_log_ptr, dt_bias_ptr, g_ptr, beta_compute_ptr,
        total_tokens, static_cast<int32_t>(num_heads), round_g_to_bf16);
    qwen35_e2e_pto::mega_prepare_solve_constants<
        bfloat16_t, GDN_PREFILL_COMPUTE_DTYPE>(
        reinterpret_cast<__gm__ bfloat16_t *>(minus_identity_ptr),
        reinterpret_cast<__gm__ GDN_PREFILL_COMPUTE_DTYPE *>(
            minus_identity_compute_ptr),
        static_cast<int64_t>(kChunkSize * kChunkSize));

    qwen35_e2e_pto::SyncAllImpl<false>();

#ifdef E2E_STOP_AFTER_FRONTEND
    return;
#endif

#ifdef E2E_STOP_AFTER_CAST
    return;
#endif

    const uint64_t q_bytes = tokens * key_heads * kHeadDim * kDtypeBytes;
    GM_ADDR q_ptr = packed_qkv_compute_ptr;
    GM_ADDR k_ptr = q_ptr + q_bytes;
    GM_ADDR v_ptr = k_ptr + q_bytes;
#if defined(GDN_PREFILL_ARCH_A5)
    // Recycling one of four KKT-to-Solve slots while the fifth producer wave
    // is still live can deadlock at 16 chunks. A5 uses the existing full
    // stage rendezvous instead of the pipelined slot protocol.
#endif
    qwen35_e2e_pto::mega_kernel_impl<true, true, true, true, true, true>(
        q_ptr, k_ptr, v_ptr, g_ptr, beta_compute_ptr, mask_lower_ptr,
        mask_full_ptr, minus_identity_compute_ptr, cu_seqlens_ptr,
        norm_output_ptr, g_sum_ptr,
        g_t_ptr, beta_t_ptr, a_matrix_ptr, a_matrix_ptr, a_inv_ptr, w_ptr,
        u_ptr, s_ptr, v_new_ptr, s_ptr, s_ptr,
        1, kkt_workspace_ptr,
        wy_workspace_a1_ptr, wy_workspace_a2_ptr, h_workspace_ptr,
        o_workspace_qk_ptr, o_workspace_qs_ptr, o_workspace_gated_ptr,
        static_cast<int32_t>(heads), num_key_heads, batch_size, total_tokens,
        total_tokens, static_cast<uint32_t>(matrices), tiling_data.ffts_addr,
        z_ptr, norm_weight_ptr, ssm_cache_out_ptr,
        ssm_state_write_indices_ptr, 1, ssm_cache_ptr,
        ssm_state_read_indices_ptr,
        static_cast<int64_t>(tiling_data.ssm_state_slots));
#if defined(GDN_PREFILL_ARCH_A2A3) && defined(__DAV_C220_VEC__)
    // A PIPE_ALL barrier orders issued work but does not acknowledge the final
    // GM stores. Drain MTE3 once before kernel completion so output and state
    // cache writes are visible to the following op and to host-side checks.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

#undef GDN_PREFILL_COMPUTE_DTYPE
