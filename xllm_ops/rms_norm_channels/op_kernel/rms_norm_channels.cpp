/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

/*!
 * \file rms_norm_channels.cpp
 * \brief RmsNormChannels kernel entry: dispatches on the host tiling key.
 *
 * out[b, c, n] = value[b, c, n] * rsqrt(mean_c(value[b, :, n]^2) + eps) * gamma[c]
 * with n = T * H * W and the channel axis at stride n (no transpose).
 */

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/device_warp_functions.h"
#include "c_api/asc_simd.h"

#include "rms_norm_channels_common.h"
#include "rms_norm_channels_kernel_impl.h"

namespace {
constexpr int64_t SIMT_WARP_SIZE = 32;

// One warp handles one output position: the lanes split the C axis,
// asc_reduce_add joins the partial sums, then the warp normalizes its position.
template <typename T>
__simt_vf__ __launch_bounds__(32) inline void RmsNormChannelsWarpVF(
    __gm__ T *valueGm, __gm__ T *gammaGm, __gm__ T *outGm, int64_t channels, int64_t n,
    int64_t totalPositions, float eps)
{
    const int64_t pos = static_cast<int64_t>(blockIdx.x);
    if (pos >= totalPositions) {
        return;
    }
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    const int64_t batch = pos / n;
    const int64_t p = pos - batch * n;
    const int64_t base = batch * channels * n + p;

    float sum = 0.0f;
    for (int64_t c = lane; c < channels; c += SIMT_WARP_SIZE) {
        const float v = static_cast<float>(valueGm[base + c * n]);
        sum += v * v;
    }
    sum = asc_reduce_add(sum);

    const float invC = 1.0f / static_cast<float>(channels);
    const float rstd = rsqrtf(sum * invC + eps);
    for (int64_t c = lane; c < channels; c += SIMT_WARP_SIZE) {
        const int64_t off = base + c * n;
        const float v = static_cast<float>(valueGm[off]);
        const float g = static_cast<float>(gammaGm[c]);
        outGm[off] = static_cast<T>(v * rstd * g);
    }
}
} // namespace

extern "C" __global__ __aicore__ void rms_norm_channels(
    GM_ADDR value, GM_ADDR gamma, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    const int64_t b = static_cast<int64_t>(tilingData.b);
    const int64_t c = static_cast<int64_t>(tilingData.c);
    const int64_t n = static_cast<int64_t>(tilingData.n);
    if (b <= 0 || c <= 0 || n <= 0) {
        return;
    }
    const uint32_t tilingKey = tilingData.tilingKey;
    const float eps = tilingData.epsilon;
    const int64_t numBlocks = static_cast<int64_t>(tilingData.numBlocks);
    const int64_t cPerBlock = static_cast<int64_t>(tilingData.cPerBlock);
    const int64_t workspaceStride = static_cast<int64_t>(tilingData.workspaceStride);
    const uint32_t nTile = tilingData.nTile;
    const uint32_t cChunk = tilingData.cChunk;

    if (tilingKey == RMS_NORM_CHANNELS_TK_SIMT_WARP) {
        asc_init();
        asc_vf_call<RmsNormChannelsWarpVF<DTYPE_VALUE>>(dim3(32),
            reinterpret_cast<__gm__ DTYPE_VALUE *>(value),
            reinterpret_cast<__gm__ DTYPE_VALUE *>(gamma),
            reinterpret_cast<__gm__ DTYPE_VALUE *>(out), c, n,
            static_cast<int64_t>(tilingData.totalTiles), eps);
        return;
    }

    if (tilingKey == RMS_NORM_CHANNELS_TK_SPLIT_C) {
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        // Phase 1: every block reduces its own C range into the workspace.
        RmsNormChannelsKernel::KernelRmsNormChannelsSplitSum<DTYPE_VALUE> sumOp;
        sumOp.Init(value, userWorkspace, b, c, n, numBlocks, cPerBlock, cChunk, nTile,
            workspaceStride);
        sumOp.Process();
        // Phase 2 needs every partial row, so the whole grid must join here.
        AscendC::SyncAll();
        RmsNormChannelsKernel::KernelRmsNormChannelsSplitNorm<DTYPE_VALUE> normOp;
        normOp.Init(value, gamma, out, userWorkspace, b, c, n, numBlocks, cPerBlock, cChunk,
            nTile, workspaceStride, eps);
        normOp.Process();
        return;
    }

    const int64_t totalTiles = static_cast<int64_t>(tilingData.totalTiles);
    const int64_t tilesPerBatch = static_cast<int64_t>(tilingData.tilesPerBatch);
    if (tilingKey == RMS_NORM_CHANNELS_TK_SINGLE_CHANNEL) {
        RmsNormChannelsKernel::KernelRmsNormChannelsSingleChannel<DTYPE_VALUE> op;
        op.Init(value, gamma, out, b, c, n, totalTiles, tilesPerBatch, numBlocks, nTile, eps);
        op.Process();
        return;
    }
    if (tilingKey == RMS_NORM_CHANNELS_TK_SMALL_C) {
        RmsNormChannelsKernel::KernelRmsNormChannelsSmallC<DTYPE_VALUE> op;
        op.Init(value, gamma, out, b, c, n, totalTiles, tilesPerBatch, numBlocks, nTile, eps);
        op.Process();
        return;
    }
    RmsNormChannelsKernel::KernelRmsNormChannelsStream<DTYPE_VALUE> op;
    op.Init(value, gamma, out, b, c, n, totalTiles, tilesPerBatch, numBlocks, nTile, cChunk, eps);
    op.Process();
}
