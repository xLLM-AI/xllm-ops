/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kda_prefill_pipeline_a5.h"
#else
// Parse CANN definitions before PTO exports its compatibility names.
#include "kda_prefill_chunk_h.h"
#include "kda_prefill_chunk_o.h"
#include "kda_prefill_frontend.h"
#include "kda_prefill_inverse_fp32.h"
#include "kda_prefill_kkt_rows.h"
#include "kda_prefill_wy.h"
#include "lib/matmul_intf.h"
#endif

using MegaKdaPrefillTilingData = kda_prefill::TilingData;

extern "C" __global__ __aicore__ void mega_kda_prefill(
    GM_ADDR qkv,
    GM_ADDR gate,
    GM_ADDR beta,
    GM_ADDR a_log,
    GM_ADDR gate_bias,
    GM_ADDR conv_weight,
    GM_ADDR conv_bias,
    GM_ADDR conv_state_in,
    GM_ADDR ssm_state_in,
    GM_ADDR cu_seqlens,
    GM_ADDR conv_read_indices,
    GM_ADDR conv_write_indices,
    GM_ADDR ssm_read_indices,
    GM_ADDR ssm_write_indices,
    GM_ADDR output,
    GM_ADDR conv_state_out,
    GM_ADDR ssm_state_out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  REGISTER_TILING_DEFAULT(MegaKdaPrefillTilingData);
  GET_TILING_DATA_WITH_STRUCT(MegaKdaPrefillTilingData, t, tiling);
#ifdef __CCE_AICORE__
#if __CCE_AICORE__ == 310
  if constexpr (TILING_KEY_IS(1)) {
#else
  if constexpr (TILING_KEY_IS(1) || TILING_KEY_IS(2)) {
#endif
    using namespace kda_prefill;
    Args a{reinterpret_cast<__gm__ bfloat16_t*>(qkv),
           reinterpret_cast<__gm__ bfloat16_t*>(gate),
           reinterpret_cast<__gm__ float*>(beta),
           reinterpret_cast<__gm__ float*>(a_log),
           reinterpret_cast<__gm__ float*>(gate_bias),
           reinterpret_cast<__gm__ bfloat16_t*>(conv_weight),
           reinterpret_cast<__gm__ bfloat16_t*>(conv_bias),
           reinterpret_cast<__gm__ bfloat16_t*>(conv_state_in),
           reinterpret_cast<__gm__ float*>(ssm_state_in),
           reinterpret_cast<__gm__ int32_t*>(cu_seqlens),
           reinterpret_cast<__gm__ int32_t*>(conv_read_indices),
           reinterpret_cast<__gm__ int32_t*>(conv_write_indices),
           reinterpret_cast<__gm__ int32_t*>(ssm_read_indices),
           reinterpret_cast<__gm__ int32_t*>(ssm_write_indices),
           reinterpret_cast<__gm__ bfloat16_t*>(output),
           reinterpret_cast<__gm__ bfloat16_t*>(conv_state_out),
           reinterpret_cast<__gm__ float*>(ssm_state_out)};
#if __CCE_AICORE__ == 310
    run_pipeline_a5(a, t, workspace);
#else
    // Every core validates the same immutable metadata before any publication
    // or barrier.
    if (!valid_metadata(a, t)) {
      return;
    }
    set_ffts_base_addr(t.ffts_addr);
    auto* ws = AscendC::GetUserWorkspace(workspace);
    constexpr int chunk_size = kAscend910bSmallChunkSize;
    constexpr bool use_main_conv_rounding = TILING_KEY_IS(2);
    constexpr bool use_log2_gate = TILING_KEY_IS(2);
    const Workspace w(t, chunk_size);
    prepare<chunk_size, use_main_conv_rounding, use_log2_gate>(a, t, ws, w);
    pipe_barrier(PIPE_ALL);
    state_io<false>(a, t, ws, w);
    rendezvous();
#define KDA_F32(name) reinterpret_cast<__gm__ float*>(ws + w.name)
#define KDA_BF16(name) reinterpret_cast<__gm__ bfloat16_t*>(ws + w.name)
    mega_kda_prefill_kkt_rows::launch_kkt_rows<chunk_size, use_log2_gate>(KDA_BF16(k),
                                               KDA_F32(g),
                                               KDA_F32(beta),
                                               KDA_F32(lower),
                                               KDA_BF16(q),
                                               KDA_F32(aqk),
                                               a.cu,
                                               t.batch,
                                               t.tokens,
                                               t.tokens,
                                               t.heads,
                                               KDA_BF16(u),
                                               KDA_BF16(corrected));
    rendezvous();
    if constexpr (use_log2_gate) {
      // These two buffers are overwritten by W/U and ChunkH after the solve.
      mega_kda_prefill_kkt_rows::launch_full_akk(
          KDA_BF16(u), KDA_BF16(corrected), KDA_F32(lower), a.cu, t.batch,
          t.tokens, t.heads);
      rendezvous();
    }
    mega_kda_prefill_inverse_fp32::inverse_fp32<chunk_size, use_log2_gate>(
        KDA_F32(lower), KDA_F32(inverse), KDA_F32(eye), nullptr, a.cu,
        t.batch, t.tokens, t.heads, KDA_F32(beta));
    rendezvous();
    mega_kda_prefill_wy_kda::wy_kda_kernel<128, chunk_size, use_log2_gate>(KDA_BF16(k),
                                                     KDA_F32(value),
                                                     KDA_F32(beta),
                                                     KDA_F32(g),
                                                     KDA_F32(inverse),
                                                     KDA_F32(wy_a),
                                                     KDA_F32(wy_k),
                                                     KDA_F32(u),
                                                     KDA_F32(w),
                                                     a.cu,
                                                     t.batch,
                                                     t.tokens,
                                                     t.heads,
                                                     t.ffts_addr);
    rendezvous();
    if constexpr (use_log2_gate) {
      mega_kda_prefill_wy_kda::round_wu(KDA_F32(w), KDA_F32(u), t.tokens, t.heads);
      rendezvous();
    }
    mega_kda_prefill_chunk_h_kda::chunk_h_kda_kernel<128, chunk_size, use_log2_gate>(
        KDA_BF16(k),
        KDA_F32(w),
        KDA_F32(u),
        KDA_F32(g),
        KDA_F32(snapshots),
        KDA_F32(corrected),
        KDA_F32(h_ws),
        a.cu,
        t.batch,
        t.tokens,
        t.tokens,
        t.heads,
        t.ffts_addr,
        KDA_F32(initial),
        KDA_F32(final));
    rendezvous();
    mega_kda_prefill_chunk_o_exact::chunk_o_kda_kernel<128, chunk_size, use_log2_gate>(
        KDA_BF16(q),
        KDA_BF16(k),
        KDA_F32(corrected),
        KDA_F32(snapshots),
        KDA_F32(g),
        KDA_F32(aqk),
        KDA_F32(o_ws),
        a.output,
        a.cu,
        t.batch,
        t.tokens,
        t.tokens,
        t.heads,
        t.ffts_addr);
    rendezvous();
    state_io<true>(a, t, ws, w);
    pipe_barrier(PIPE_ALL);
    rendezvous();
#undef KDA_F32
#undef KDA_BF16
#endif
  }
#endif
}
