/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "mega_kda_decode_pto_kernel.h"

using MegaKdaDecodeTilingData = mega_kda::TilingData;

extern "C" __global__ __aicore__ void mega_kda_decode(
    GM_ADDR qkv, GM_ADDR gate, GM_ADDR beta, GM_ADDR a_log, GM_ADDR gate_bias,
    GM_ADDR conv_weight, GM_ADDR conv_bias, GM_ADDR conv_state_in, GM_ADDR ssm_state_in,
    GM_ADDR cu_seqlens, GM_ADDR conv_read_indices, GM_ADDR conv_write_indices,
    GM_ADDR ssm_read_indices, GM_ADDR ssm_write_indices, GM_ADDR num_accepted_tokens,
    GM_ADDR output, GM_ADDR conv_state_out, GM_ADDR ssm_state_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MegaKdaDecodeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaKdaDecodeTilingData, t, tiling);
    mega_kda::Args args{
        reinterpret_cast<__gm__ bfloat16_t*>(qkv),
        reinterpret_cast<__gm__ bfloat16_t*>(gate),
        reinterpret_cast<__gm__ bfloat16_t*>(beta),
        reinterpret_cast<__gm__ float*>(a_log), reinterpret_cast<__gm__ float*>(gate_bias),
        reinterpret_cast<__gm__ bfloat16_t*>(conv_weight), reinterpret_cast<__gm__ bfloat16_t*>(conv_bias),
        reinterpret_cast<__gm__ bfloat16_t*>(conv_state_in), reinterpret_cast<__gm__ float*>(ssm_state_in),
        reinterpret_cast<__gm__ int32_t*>(cu_seqlens), reinterpret_cast<__gm__ int32_t*>(conv_read_indices),
        reinterpret_cast<__gm__ int32_t*>(conv_write_indices), reinterpret_cast<__gm__ int32_t*>(ssm_read_indices),
        reinterpret_cast<__gm__ int32_t*>(ssm_write_indices), reinterpret_cast<__gm__ int32_t*>(num_accepted_tokens),
        reinterpret_cast<__gm__ bfloat16_t*>(output), reinterpret_cast<__gm__ bfloat16_t*>(conv_state_out),
        reinterpret_cast<__gm__ float*>(ssm_state_out)};
    if constexpr (TILING_KEY_IS(1)) {
        mega_kda::Run<false>(args, t);
    } else if constexpr (TILING_KEY_IS(2)) {
        mega_kda::Run<true>(args, t);
    }
}
