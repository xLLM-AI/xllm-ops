/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include "kda_prefill_frontend_a5.h"
#include "kda_prefill_short_a5.h"
#include "kda_prefill_chunk_h_a5.h"
#include "kda_prefill_chunk_o_a5.h"
#include "kda_prefill_kkt_rows_a5.h"
#include "kda_prefill_wy_a5.h"
#include "kda_prefill_kkt_cube_a5.h"
#include "kda_prefill_inverse_block_a5.h"
#include "lib/matmul_intf.h"

namespace kda_prefill {
AICORE KDA_ARCH_INLINE void run_pipeline_a5(
    const Args& a, const TilingData& t, GM_ADDR workspace) {
    // Every core validates the same immutable metadata before any publication or barrier.
    if (!ValidMetadata(a, t)) return;
#if __CCE_AICORE__ == 310
    if (t.batch == 1 && t.tokens > 0 && t.tokens <= 4 && t.heads == 8) {
        RunShortA5(a, t);
        return;
    }
#endif
#if __CCE_AICORE__ == 220
    set_ffts_base_addr(t.ffts_addr);
#endif
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
#if __CCE_AICORE__ == 310
    mega_kda_prefill_kkt_cube_a5::Run(KDA_BF16(q), KDA_BF16(k), KDA_F32(g),
        KDA_F32(beta), KDA_F32(lower), KDA_F32(aqk), KDA_F32(h_ws), a.cu,
        t.batch, t.tokens, t.heads);
#endif
    Rendezvous();
#if __CCE_AICORE__ == 310
    mega_kda_prefill_inverse_block_a5::Run(KDA_F32(lower), KDA_F32(inverse),
        KDA_F32(eye), KDA_F32(h_ws), a.cu, t.batch, t.heads);
#else
    mega_kda_prefill_inverse_fp32::inverse_fp32(KDA_F32(lower), KDA_F32(inverse),
        KDA_F32(eye), nullptr, a.cu, t.batch, t.tokens, t.heads);
#endif
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
}  // namespace kda_prefill
