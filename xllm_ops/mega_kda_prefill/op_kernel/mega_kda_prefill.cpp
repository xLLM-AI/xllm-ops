/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "kda_prefill_frontend.h"
#include "kda_prefill_kkt_rows.h"
#include "kda_prefill_inverse_fp32.h"
#include "kda_prefill_wy.h"
#include "kda_prefill_chunk_h.h"
#include "kda_prefill_chunk_o.h"
#include "lib/matmul_intf.h"

using MegaKdaPrefillTilingData = kda_prefill::TilingData;

extern "C" __global__ __aicore__ void mega_kda_prefill(
    GM_ADDR qkv, GM_ADDR gate, GM_ADDR beta, GM_ADDR a_log, GM_ADDR gate_bias,
    GM_ADDR conv_weight, GM_ADDR conv_bias, GM_ADDR conv_state_in, GM_ADDR ssm_state_in,
    GM_ADDR cu_seqlens, GM_ADDR conv_read_indices, GM_ADDR conv_write_indices,
    GM_ADDR ssm_read_indices, GM_ADDR ssm_write_indices,
    GM_ADDR output, GM_ADDR conv_state_out, GM_ADDR ssm_state_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaKdaPrefillTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaKdaPrefillTilingData, t, tiling);
#ifdef __CCE_AICORE__
    if constexpr (TILING_KEY_IS(1)) {
    using namespace kda_prefill;
    Args a{
        reinterpret_cast<__gm__ bfloat16_t*>(qkv), reinterpret_cast<__gm__ bfloat16_t*>(gate),
        reinterpret_cast<__gm__ float*>(beta), reinterpret_cast<__gm__ float*>(a_log),
        reinterpret_cast<__gm__ float*>(gate_bias), reinterpret_cast<__gm__ bfloat16_t*>(conv_weight),
        reinterpret_cast<__gm__ bfloat16_t*>(conv_bias), reinterpret_cast<__gm__ bfloat16_t*>(conv_state_in),
        reinterpret_cast<__gm__ float*>(ssm_state_in), reinterpret_cast<__gm__ int32_t*>(cu_seqlens),
        reinterpret_cast<__gm__ int32_t*>(conv_read_indices), reinterpret_cast<__gm__ int32_t*>(conv_write_indices),
        reinterpret_cast<__gm__ int32_t*>(ssm_read_indices), reinterpret_cast<__gm__ int32_t*>(ssm_write_indices),
        reinterpret_cast<__gm__ bfloat16_t*>(output), reinterpret_cast<__gm__ bfloat16_t*>(conv_state_out),
        reinterpret_cast<__gm__ float*>(ssm_state_out)};
    // Every core validates the same immutable metadata before any publication or barrier.
    if (!ValidMetadata(a, t)) return;
    set_ffts_base_addr(t.ffts_addr);
    auto* ws = AscendC::GetUserWorkspace(workspace);
    const Workspace w(t);
    Prepare(a, t, ws, w);
    pipe_barrier(PIPE_ALL);
    StateIO<false>(a, t, ws, w);
    Rendezvous();
#define KDA_F32(name) reinterpret_cast<__gm__ float*>(ws + w.name)
#define KDA_BF16(name) reinterpret_cast<__gm__ bfloat16_t*>(ws + w.name)
    mega_kda_prefill_kkt_rows::launch_kkt_rows(KDA_BF16(k), KDA_F32(g), KDA_F32(beta),
        KDA_F32(masks), KDA_F32(lower), KDA_BF16(q), KDA_F32(aqk), a.cu,
        t.batch, t.tokens, t.tokens, t.heads);
    Rendezvous();
    mega_kda_prefill_inverse_fp32::inverse_fp32(KDA_F32(lower), KDA_F32(inverse),
        KDA_F32(eye), nullptr, a.cu, t.batch, t.tokens, t.heads);
    Rendezvous();
    mega_kda_prefill_wy_kda::wy_kda_kernel<128, 128>(KDA_BF16(k), KDA_F32(value),
        KDA_F32(beta), KDA_F32(g), KDA_F32(inverse), KDA_F32(wy_a), KDA_F32(wy_k),
        KDA_F32(u), KDA_F32(w), a.cu, t.batch, t.tokens, t.tokens, t.heads, t.ffts_addr);
    Rendezvous();
    mega_kda_prefill_chunk_h_kda::chunk_h_kda_kernel<128, 128>(KDA_BF16(k), KDA_F32(w),
        KDA_F32(u), KDA_F32(g), KDA_F32(snapshots), KDA_F32(corrected), KDA_F32(h_ws),
        a.cu, t.batch, t.tokens, t.tokens, t.heads, t.ffts_addr, KDA_F32(initial), KDA_F32(final));
    Rendezvous();
    mega_kda_prefill_chunk_o_exact::chunk_o_kda_kernel<128, 128>(KDA_BF16(q), KDA_BF16(k),
        KDA_F32(corrected), KDA_F32(snapshots), KDA_F32(g), KDA_F32(aqk), KDA_F32(o_ws),
        KDA_F32(output), a.cu, t.batch, t.tokens, t.tokens, t.heads, t.ffts_addr);
    Rendezvous();
    StateIO<true>(a, t, ws, w);
    pipe_barrier(PIPE_ALL);
    Output(a, t, ws, w);
    Rendezvous();
#undef KDA_F32
#undef KDA_BF16
    }
#endif
}
