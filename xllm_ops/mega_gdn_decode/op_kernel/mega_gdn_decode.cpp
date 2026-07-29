/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "mega_gdn_decode_pto_kernel.h"

struct MegaGdnDecodeTilingData {
    int64_t batch_size;
    int64_t num_k_heads;
    int64_t num_v_heads;
};

template <bool IsBatchOne>
AICORE PTO_INLINE void RunMegaGdnDecode(
    GM_ADDR qkv, GM_ADDR z, GM_ADDR b, GM_ADDR a, GM_ADDR convWeight,
    GM_ADDR convState, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR ssmState,
    GM_ADDR stateIndices, GM_ADDR normWeight, GM_ADDR convOut,
    GM_ADDR convStateOut, GM_ADDR ssmStateOut, GM_ADDR out,
    int32_t numKHeads, int32_t numVHeads, int32_t batchSize)
{
    qwen35_decode_pto::Run<IsBatchOne>(
        reinterpret_cast<__gm__ bfloat16_t *>(qkv),
        reinterpret_cast<__gm__ bfloat16_t *>(z),
        reinterpret_cast<__gm__ bfloat16_t *>(b),
        reinterpret_cast<__gm__ bfloat16_t *>(a),
        reinterpret_cast<__gm__ bfloat16_t *>(convWeight),
        reinterpret_cast<__gm__ bfloat16_t *>(convState),
        reinterpret_cast<__gm__ float *>(aLog),
        reinterpret_cast<__gm__ float *>(dtBias),
        reinterpret_cast<__gm__ float *>(ssmState),
        reinterpret_cast<__gm__ int *>(stateIndices),
        reinterpret_cast<__gm__ bfloat16_t *>(normWeight),
        reinterpret_cast<__gm__ bfloat16_t *>(convOut),
        reinterpret_cast<__gm__ bfloat16_t *>(convStateOut),
        reinterpret_cast<__gm__ float *>(ssmStateOut),
        reinterpret_cast<__gm__ bfloat16_t *>(out),
        numKHeads, numVHeads, batchSize);
}

extern "C" __global__ __aicore__ void mega_gdn_decode(
    GM_ADDR qkv, GM_ADDR z, GM_ADDR b, GM_ADDR a, GM_ADDR convWeight,
    GM_ADDR convState, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR ssmState,
    GM_ADDR stateIndices, GM_ADDR normWeight, GM_ADDR convOut,
    GM_ADDR convStateOut, GM_ADDR ssmStateOut, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(PTO_NPU_ARCH_A5)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    REGISTER_TILING_DEFAULT(MegaGdnDecodeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaGdnDecodeTilingData,
                                tilingData, tiling);
    (void)workspace;

    if constexpr (TILING_KEY_IS(2)) {
        RunMegaGdnDecode<true>(
            qkv, z, b, a, convWeight, convState, aLog, dtBias, ssmState,
            stateIndices, normWeight, convOut, convStateOut, ssmStateOut,
            out, static_cast<int32_t>(tilingData.num_k_heads),
            static_cast<int32_t>(tilingData.num_v_heads), 1);
    } else if constexpr (TILING_KEY_IS(1)) {
        const int32_t batchSize = static_cast<int32_t>(tilingData.batch_size);
        RunMegaGdnDecode<false>(
            qkv, z, b, a, convWeight, convState, aLog, dtBias, ssmState,
            stateIndices, normWeight, convOut, convStateOut, ssmStateOut,
            out, static_cast<int32_t>(tilingData.num_k_heads),
            static_cast<int32_t>(tilingData.num_v_heads), batchSize);
    }
}

// The generated mixed-kernel wrapper calls matmul::clearWorkspace. Keep this
// include after PTO code to avoid the CANN DYNAMIC/pto::DYNAMIC name collision.
#include "lib/matmul_intf.h"
