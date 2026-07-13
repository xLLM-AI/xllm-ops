// Qwen3.5 TP2 prefill state super-kernel.
//
// Same pipeline as pto_mega_kernel, with value heads (H) and key heads (Hg)
// passed at runtime. H dispatches to finite compile-time specializations.
//
// Stages:
//   1. cumsum      (Vec)
//   2. transpose   (Vec)
//   3. kkt         (Cube+Vec)  — K has Hg heads; β,g,A use H value heads
//   4. solve_tril  (Cube)
//   5. wy_fast     (Vec+Cube)
//   6. chunk_h     (Cube+Vec)
//   7. chunk_o     (Cube+Vec)

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#ifndef GDN_KERNEL_NAME
#define GDN_KERNEL_NAME launch_qwen35_gdn_prefill_super_op
#endif
// Note the codegen parser does not support arguments of form "type *name", only "type* name"
// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format off

#include "acl/acl.h"
#include "kernel_operator.h"
#define CAUSAL_CONV1D_SKIP_TPL_REGISTRATION
#include "../../causal_conv1d/op_kernel/causal_conv1d_fn.h"
#undef CAUSAL_CONV1D_SKIP_TPL_REGISTRATION
#include <pto/pto-inst.hpp>
#include <type_traits>
using namespace pto;

namespace qwen35_mem {
template <typename T, uint32_t dstN, uint32_t dstM = 1>
__aicore__ inline void CopyGmToUb(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                  uint32_t realSrcN = 1, uint32_t maskShapeM = dstM,
                                  uint32_t maskShapeN = dstN, T padValue = T(0))
{
    if (maskShapeM != dstM || maskShapeN != dstN) {
        AscendC::Duplicate(dst, padValue, dstM * dstN);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const bool isPad = maskShapeN != dstN;
    AscendC::DataCopyExtParams params(maskShapeM, maskShapeN * sizeof(T),
                                      (realSrcN - maskShapeN) * sizeof(T),
                                      (dstN - maskShapeN) * sizeof(T) / 32, 0);
    AscendC::DataCopyPadExtParams<T> padParams(isPad, 0, isPad ? 1 : 0, padValue);
    AscendC::DataCopyPad(dst, src, params, padParams);
}

template <typename T, uint32_t srcN, uint32_t srcM = 1>
__aicore__ inline void CopyUbToGm(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src,
                                  uint32_t realDstN = 1, uint32_t maskShapeM = srcM,
                                  uint32_t maskShapeN = srcN)
{
    AscendC::DataCopyExtParams params(maskShapeM, maskShapeN * sizeof(T),
                                      (srcN - maskShapeN) * sizeof(T) / 32,
                                      (realDstN - maskShapeN) * sizeof(T), 0);
    AscendC::DataCopyPad(dst, src, params);
}

template <typename T, uint32_t length>
__aicore__ inline void CopyUbToUb(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src)
{
    AscendC::DataCopy(dst, src, length);
}

template <typename T, uint32_t M, uint32_t N>
__aicore__ inline void ReduceRows(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src,
                                  AscendC::LocalTensor<uint8_t> tmp)
{
    uint32_t shape[] = {M, N};
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
}

template <typename T, int32_t dim, int32_t axis>
__aicore__ inline void Broadcast(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src,
                                 AscendC::LocalTensor<uint8_t> &tmp,
                                 const uint32_t dstShape[dim], const uint32_t srcShape[dim])
{
    AscendC::Broadcast<T, dim, axis, false>(dst, src, dstShape, srcShape, tmp);
}
}  // namespace qwen35_mem

struct Qwen35GdnPrefillSuperOpTilingData {
    uint32_t block_dim;
    uint32_t num_matrices;
    uint32_t num_heads;
    uint32_t num_key_heads;
    uint32_t token_block_size;
    uint32_t token_block_count;
    int64_t conv_state_index;
    int64_t ssm_state_index;
    int64_t batch_size;
    int64_t seq_len;
    int64_t total_tokens;
};

// ===================================================================
// Device-only helpers (shared with standard mega-kernel)
// ===================================================================
#ifdef __CCE_AICORE__

constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllImpl()
{
    pipe_barrier(PIPE_ALL);
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
#if defined(__DAV_C220_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_C220_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
}

namespace NsCausalConv1d {
template <typename T, uint32_t widthKey, uint32_t fnPlanKey>
class Qwen35PrefillConv : public CausalConv1d<T, CAUSAL_CONV1D_TPL_RUN_MODE_FN, widthKey, fnPlanKey> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convState, GM_ADDR packedQkv,
                                const CausalConv1dTilingData *tilingData)
    {
        this->ResetRuntimeState(tilingData);
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        this->convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convState));
        this->packedQkvYGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(packedQkv));
        this->InitSharedBuffersAndEvents();
    }

    __aicore__ inline void Process(int32_t cacheIndex)
    {
        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        const int32_t tokenCount = static_cast<int32_t>(this->tilingData_->cuSeqlen);
        const auto task = ResolveFnPackedQkvBlockTask(
            blockIdx, static_cast<int32_t>(this->tilingData_->tokenBlockCnt),
            static_cast<int32_t>(this->tilingData_->tokenBlockSize), tokenCount,
            static_cast<int32_t>(this->tilingData_->baseDimCnt),
            static_cast<int32_t>(this->tilingData_->baseDim), static_cast<int32_t>(this->tilingData_->dim));
        if (task.valid) {
            this->ProcessFnChunk(cacheIndex, false, 0, tokenCount, task.tokenStart,
                                 task.tokenEnd - task.tokenStart, task.channelStart,
                                 task.baseDimSize, static_cast<int32_t>(this->tilingData_->dim));
        }
        this->ReleaseEvents();
    }
};
}  // namespace NsCausalConv1d

AICORE inline void RunQwen35PrefillConv(GM_ADDR mixedQkv, GM_ADDR convWeight, GM_ADDR convState,
                                        GM_ADDR packedQkv, int64_t totalTokens, uint32_t tokenBlockSize,
                                        uint32_t tokenBlockCount, int64_t cacheIndex)
{
#if defined(__DAV_C220_VEC__)
    CausalConv1dTilingData tiling{};
    tiling.dim = 5120;
    tiling.cuSeqlen = totalTokens;
    tiling.seqLen = totalTokens;
    tiling.inputMode = 0;
    tiling.width = 4;
    tiling.stateLen = 3;
    tiling.numCacheLines = cacheIndex + 1;
    tiling.batch = 1;
    tiling.activationMode = CAUSAL_CONV1D_ACTIVATION_SILU_PACKED_QKV;
    tiling.padSlotId = -1;
    tiling.hasBias = 0;
    tiling.baseDim = 3072;
    tiling.baseDimCnt = 2;
    tiling.hasNumAcceptedTokens = 0;
    tiling.hasCacheIndices = 0;
    tiling.hasInitialStateMode = 0;
    tiling.hasInitStateWorkspace = 0;
    tiling.tokenBlockSize = tokenBlockSize;
    tiling.tokenBlockCnt = tokenBlockCount;
    tiling.hasExplicitTokenSeqRanges = 0;

    NsCausalConv1d::Qwen35PrefillConv<bfloat16_t, CAUSAL_CONV1D_TPL_WIDTH_4,
                                     CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD>
        op;
    op.Init(mixedQkv, convWeight, convState, packedQkv, &tiling);
    op.Process(static_cast<int32_t>(cacheIndex));
#endif
}

AICORE inline int32_t QwenVectorTaskId()
{
#if defined(__DAV_C220_VEC__)
    return static_cast<int32_t>(AscendC::GetBlockIdx() / 2 * 2 + AscendC::GetSubBlockIdx());
#else
    return 0;
#endif
}

AICORE inline void PrepareQwen35Gate(GM_ADDR aPtr, GM_ADDR bPtr, GM_ADDR aLogPtr, GM_ADDR dtBiasPtr,
                                     GM_ADDR gPtr, GM_ADDR betaPtr, int64_t totalTokens)
{
#if defined(__DAV_C220_VEC__)
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
    pipe.InitBuffer(ub, 32768);

    auto aLog = ub.GetWithOffset<float>(64, 0);
    auto dtBias = ub.GetWithOffset<float>(64, 256);
    auto aBf16 = ub.GetWithOffset<bfloat16_t>(64, 512);
    auto bBf16 = ub.GetWithOffset<bfloat16_t>(64, 640);
    auto x = ub.GetWithOffset<float>(64, 768);
    auto betaX = ub.GetWithOffset<float>(64, 1024);
    auto absX = ub.GetWithOffset<float>(64, 1280);
    auto tmp = ub.GetWithOffset<float>(64, 1536);
    auto betaFp32 = ub.GetWithOffset<float>(64, 1792);
    auto betaRounded = ub.GetWithOffset<bfloat16_t>(64, 2048);
    auto betaHalf = ub.GetWithOffset<half>(64, 2176);
    auto cmpMask = ub.GetWithOffset<uint8_t>(64, 2304);
    auto sigmoidTmp = ub.GetWithOffset<uint8_t>(2048, 2368);

    AscendC::GlobalTensor<float> aLogGm;
    AscendC::GlobalTensor<float> dtBiasGm;
    AscendC::GlobalTensor<bfloat16_t> aGm;
    AscendC::GlobalTensor<bfloat16_t> bGm;
    AscendC::GlobalTensor<float> gGm;
    AscendC::GlobalTensor<half> betaGm;
    aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLogPtr));
    dtBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBiasPtr));
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aPtr));
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bPtr));
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gPtr));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(betaPtr));

    qwen35_mem::CopyGmToUb<float, 24, 1>(aLog[0], aLogGm[0], 24, 1, 24, 0.0f);
    qwen35_mem::CopyGmToUb<float, 24, 1>(dtBias[0], dtBiasGm[0], 24, 1, 24, 0.0f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Exp(aLog[0], aLog[0], 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(aLog[0], aLog[0], -1.0f, 64);

    const int32_t taskId = QwenVectorTaskId();
    constexpr int32_t kVectorTasks = 40;
    for (int64_t token = taskId; token < totalTokens; token += kVectorTasks) {
        AscendC::PipeBarrier<PIPE_ALL>();
        qwen35_mem::CopyGmToUb<bfloat16_t, 24, 1>(
            aBf16[0], aGm[token * 24], 24, 1, 24, bfloat16_t(0.0f));
        qwen35_mem::CopyGmToUb<bfloat16_t, 24, 1>(
            bBf16[0], bGm[token * 24], 24, 1, 24, bfloat16_t(0.0f));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::Cast(x[0], aBf16[0], AscendC::RoundMode::CAST_NONE, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(x[0], x[0], dtBias[0], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(absX[0], x[0], 64);
        AscendC::Muls(tmp[0], absX[0], -1.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(betaFp32[0], tmp[0], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(betaFp32[0], betaFp32[0], 1.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Ln(tmp[0], betaFp32[0], 64);
        AscendC::CompareScalar(cmpMask[0], x[0], 20.0f, AscendC::CMPMODE::GT, 64);
        AscendC::Add(betaX[0], x[0], absX[0], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(betaX[0], betaX[0], 0.5f, 64);
        AscendC::Add(betaX[0], betaX[0], tmp[0], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(betaX[0], cmpMask[0], x[0], betaX[0],
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
        AscendC::Mul(x[0], aLog[0], betaX[0], 64);

        AscendC::Cast(betaFp32[0], bBf16[0], AscendC::RoundMode::CAST_NONE, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sigmoid(betaFp32[0], betaFp32[0], sigmoidTmp[0], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(betaRounded[0], betaFp32[0], AscendC::RoundMode::CAST_RINT, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(betaFp32[0], betaRounded[0], AscendC::RoundMode::CAST_NONE, 64);
        AscendC::Cast(betaHalf[0], betaFp32[0], AscendC::RoundMode::CAST_RINT, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
        qwen35_mem::CopyUbToGm<float, 24, 1>(gGm[token * 24], x[0], 24, 1, 24);
        qwen35_mem::CopyUbToGm<half, 24, 1>(betaGm[token * 24], betaHalf[0], 24, 1, 24);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

AICORE inline void ZeroInitialState(GM_ADDR initialStatePtr)
{
#if defined(__DAV_C220_VEC__)
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
    pipe.InitBuffer(ub, 512);
    auto zero = ub.Get<half>();
    AscendC::Duplicate(zero, half(0.0f), 256);
    AscendC::GlobalTensor<half> state;
    state.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initialStatePtr));
    constexpr int64_t kElements = 24 * 128 * 128;
    constexpr int64_t kChunk = 256;
    const int32_t taskId = QwenVectorTaskId();
    for (int64_t offset = static_cast<int64_t>(taskId) * kChunk; offset < kElements;
         offset += 40 * kChunk) {
        const int64_t count = (offset + kChunk <= kElements) ? kChunk : (kElements - offset);
        AscendC::DataCopy(state[offset], zero, count);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

AICORE inline void ZeroMegaScratch(GM_ADDR megaAPtr, GM_ADDR aInvF32Ptr,
                                   GM_ADDR aInvPtr, GM_ADDR hPtr,
                                   GM_ADDR finalStatePtr,
                                   int64_t totalTokens,
                                   uint32_t numMatrices)
{
#if defined(__DAV_C220_VEC__)
    constexpr int64_t kChunk = 256;
    constexpr int64_t kTasks = 40;
    constexpr int64_t kHeads = 24;
    constexpr int64_t kHeadDim = 128;
    const int32_t taskId = QwenVectorTaskId();
    const int64_t matrixElements = totalTokens * kHeads * kHeadDim;
    const int64_t hElements = static_cast<int64_t>(numMatrices) * kHeadDim * kHeadDim;
    const int64_t finalStateElements = kHeads * kHeadDim * kHeadDim;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
    pipe.InitBuffer(ub, 1536);
    auto zeroHalf = ub.GetWithOffset<half>(kChunk, 0);
    auto zeroFloat = ub.GetWithOffset<float>(kChunk, 512);
    AscendC::Duplicate(zeroHalf, half(0.0f), kChunk);
    AscendC::Duplicate(zeroFloat, 0.0f, kChunk);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);

    AscendC::GlobalTensor<half> megaA;
    AscendC::GlobalTensor<float> aInvF32;
    AscendC::GlobalTensor<half> aInv;
    AscendC::GlobalTensor<half> h;
    AscendC::GlobalTensor<half> finalState;
    megaA.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(megaAPtr));
    aInvF32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aInvF32Ptr));
    aInv.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(aInvPtr));
    h.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(hPtr));
    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(finalStatePtr));

    for (int64_t offset = static_cast<int64_t>(taskId) * kChunk;
         offset < matrixElements; offset += kTasks * kChunk) {
        AscendC::DataCopy(megaA[offset], zeroHalf, kChunk);
        AscendC::DataCopy(aInvF32[offset], zeroFloat, kChunk);
        AscendC::DataCopy(aInv[offset], zeroHalf, kChunk);
    }
    for (int64_t offset = static_cast<int64_t>(taskId) * kChunk;
         offset < hElements; offset += kTasks * kChunk) {
        AscendC::DataCopy(h[offset], zeroHalf, kChunk);
    }
    for (int64_t offset = static_cast<int64_t>(taskId) * kChunk;
         offset < finalStateElements; offset += kTasks * kChunk) {
        AscendC::DataCopy(finalState[offset], zeroHalf, kChunk);
    }
    AscendC::PipeBarrier<PIPE_MTE3>();
#endif
}

AICORE inline void RunQwen35OutputNorm(GM_ADDR megaOutPtr, GM_ADDR zPtr, GM_ADDR normWeightPtr,
                                       GM_ADDR outPtr, int64_t totalTokens)
{
#if defined(__DAV_C220_VEC__)
    constexpr int32_t kRows = 8;
    constexpr int32_t kHeadDim = 128;
    constexpr int32_t kElements = kRows * kHeadDim;
    constexpr float kScale = 0.08838834764831845f;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
    pipe.InitBuffer(ub, 49152);
    auto weightBf16 = ub.GetWithOffset<bfloat16_t>(128, 0);
    auto weightBase = ub.GetWithOffset<float>(128, 256);
    auto weight = ub.GetWithOffset<float>(kElements, 768);
    auto xHalf = ub.GetWithOffset<half>(kElements, 4864);
    auto zBf16 = ub.GetWithOffset<bfloat16_t>(kElements, 6912);
    auto x = ub.GetWithOffset<float>(kElements, 8960);
    auto z = ub.GetWithOffset<float>(kElements, 13056);
    auto silu = ub.GetWithOffset<float>(kElements, 17152);
    auto square = ub.GetWithOffset<float>(kElements, 21248);
    auto rounded = ub.GetWithOffset<bfloat16_t>(kElements, 25344);
    auto rms = ub.GetWithOffset<float>(kRows, 27392);
    auto rms2d = ub.GetWithOffset<float>(kElements, 27424);
    auto tmp = ub.GetWithOffset<uint8_t>(4096, 31520);

    AscendC::GlobalTensor<half> megaOut;
    AscendC::GlobalTensor<bfloat16_t> gate;
    AscendC::GlobalTensor<bfloat16_t> normWeight;
    AscendC::GlobalTensor<bfloat16_t> out;
    megaOut.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(megaOutPtr));
    gate.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(zPtr));
    normWeight.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(normWeightPtr));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(outPtr));

    qwen35_mem::CopyGmToUb<bfloat16_t, 128, 1>(
        weightBf16[0], normWeight[0], 128, 1, 128, bfloat16_t(0.0f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
    AscendC::Cast(weightBase[0], weightBf16[0], AscendC::RoundMode::CAST_NONE, 128);
    for (int32_t row = 0; row < kRows; ++row) {
        AscendC::PipeBarrier<PIPE_V>();
        qwen35_mem::CopyUbToUb<float, 128>(weight[row * 128], weightBase[0]);
    }

    const int64_t numRows = totalTokens * 24;
    const int64_t numChunks = (numRows + kRows - 1) / kRows;
    const int32_t taskId = QwenVectorTaskId();
    for (int64_t chunk = taskId; chunk < numChunks; chunk += 40) {
        const int64_t rowStart = chunk * kRows;
        const int32_t validRows =
            static_cast<int32_t>((rowStart + kRows <= numRows) ? kRows : (numRows - rowStart));
        for (int32_t row = 0; row < validRows; ++row) {
            qwen35_mem::CopyGmToUb<half, 128, 1>(
                xHalf[row * 128], megaOut[(rowStart + row) * 128], 128, 1, 128, half(0.0f));
            qwen35_mem::CopyGmToUb<bfloat16_t, 128, 1>(
                zBf16[row * 128], gate[(rowStart + row) * 128], 128, 1, 128, bfloat16_t(0.0f));
        }
        for (int32_t row = validRows; row < kRows; ++row) {
            AscendC::Duplicate(xHalf[row * 128], half(0.0f), 128);
            AscendC::Duplicate(zBf16[row * 128], bfloat16_t(0.0f), 128);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::Cast(x[0], xHalf[0], AscendC::RoundMode::CAST_NONE, kElements);
        AscendC::Muls(x[0], x[0], kScale, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(rounded[0], x[0], AscendC::RoundMode::CAST_RINT, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x[0], rounded[0], AscendC::RoundMode::CAST_NONE, kElements);
        AscendC::Cast(z[0], zBf16[0], AscendC::RoundMode::CAST_NONE, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Silu(silu[0], z[0], kElements);
        AscendC::Mul(square[0], x[0], x[0], kElements);
        AscendC::PipeBarrier<PIPE_V>();
        qwen35_mem::ReduceRows<float, 8, 128>(rms[0], square[0], tmp[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(rms[0], rms[0], 1.0f / 128.0f, kRows);
        AscendC::Adds(rms[0], rms[0], 1.0e-6f, kRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(rms[0], rms[0], kRows);
        AscendC::PipeBarrier<PIPE_V>();
        qwen35_mem::Broadcast<float, 2, 1>(
            rms2d[0], rms[0], tmp, (uint32_t[]){8, 128}, (uint32_t[]){8, 1});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(x[0], x[0], rms2d[0], kElements);
        AscendC::Mul(x[0], x[0], weight[0], kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(x[0], x[0], silu[0], kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(rounded[0], x[0], AscendC::RoundMode::CAST_RINT, kElements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
        for (int32_t row = 0; row < validRows; ++row) {
            qwen35_mem::CopyUbToGm<bfloat16_t, 128, 1>(
                out[(rowStart + row) * 128], rounded[row * 128], 128, 1, 128);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif
}

template <typename T, int32_t H_val>
AICORE inline void mega_transpose_TH_to_HT(__gm__ T *src, __gm__ T *dst, int64_t T_len)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t BLOCK = 128;
    constexpr int32_t H = static_cast<int32_t>(H_val);
    constexpr int32_t ES = static_cast<int32_t>(sizeof(T));
    constexpr int32_t AlignBytes = 32;
    constexpr int32_t AlignRows = AlignBytes / ES;
    constexpr int32_t MinTransposeCols = 16;
    constexpr int32_t AlignElems = (AlignRows > MinTransposeCols) ? AlignRows : MinTransposeCols;
    constexpr int32_t HP = ((H + AlignElems - 1) / AlignElems) * AlignElems;
    constexpr int32_t SRC_UB = 0;
    constexpr int32_t DST_UB = SRC_UB + BLOCK * HP * ES;
    constexpr int32_t TMP_UB = DST_UB + HP * BLOCK * ES;

    using UBSrcFull =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBSrcDyn =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBDst = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, HP, BLOCK, SLayout::NoneBox, 512>;
    using UBDstDyn = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using UBTmp = Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512>;

    using UBRow = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, 1, BLOCK, SLayout::NoneBox, 512>;
    using UBRowDyn = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using UBHeadDyn = Tile<TileType::Vec, T, AlignRows, BLOCK, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox, 512>;

    using Gm2D = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmSrcS = Stride<1, 1, 1, H, 1>;
    using GmHeadS = Stride<1, 1, 1, 1, H>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    if constexpr (H < MinTransposeCols) {
        int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;
        for (int64_t bi = static_cast<int64_t>(cid); bi < num_tok_blocks; bi += static_cast<int64_t>(block_num)) {
            int64_t t0 = bi * BLOCK;
            int32_t valid = (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);

            for (int32_t h = 0; h < H; ++h) {
                Gm1D gs;
                gs.shape[4] = valid;
                UBHeadDyn row(valid);
                TASSIGN(row, SRC_UB);
                {
                    GlobalTensor<T, Gm1D, GmHeadS, Layout::DN> gm(src + t0 * H + h, gs);
                    TLOAD(row, gm);
                }
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                {
                    GlobalTensor<T, Gm1D, GmS1, Layout::DN> gm(dst + h * T_len + t0, gs);
                    TSTORE(gm, row);
                }
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }
        return;
    }

    UBSrcFull ub_src;
    TASSIGN(ub_src, SRC_UB);
    UBDst ub_dst;
    TASSIGN(ub_dst, DST_UB);
    UBTmp ub_tmp;
    TASSIGN(ub_tmp, TMP_UB);

    int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;

    for (int64_t bi = static_cast<int64_t>(cid); bi < num_tok_blocks; bi += static_cast<int64_t>(block_num)) {
        int64_t t0 = bi * BLOCK;
        int32_t valid = (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);

        {
            Gm2D gs;
            gs.shape[3] = valid;
            gs.shape[4] = H;
            GlobalTensor<T, Gm2D, GmSrcS> gm(src + t0 * H, gs);
            UBSrcDyn ld(valid, H);
            TASSIGN(ld, SRC_UB);
            TLOAD(ld, gm);
            if (valid != BLOCK || H != HP) TFILLPAD_INPLACE(ub_src, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TTRANS(ub_dst, ub_src, ub_tmp);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        for (int32_t h = 0; h < H; ++h) {
            Gm1D gs;
            gs.shape[4] = valid;
            GlobalTensor<T, Gm1D, GmS1> gm(dst + h * T_len + t0, gs);
            UBRowDyn st(1, valid);
            TASSIGN(st, DST_UB + h * BLOCK * ES);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
#endif
}

template <int32_t H, int32_t C>
AICORE inline void mega_cast_fp32_to_fp16_bsnd(__gm__ float *src, __gm__ half *dst, uint32_t num_matrices,
                                               int64_t total_tokens)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t F32_UB = 0;
    constexpr int32_t F16_UB = C * static_cast<int32_t>(sizeof(float));

    using SrcUB = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512, PadValue::Zero>;
    using DynSrcUB =
        Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstUB = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512>;
    using DynDstUB = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    SrcUB src_ub;
    TASSIGN(src_ub, F32_UB);
    DstUB dst_ub;
    TASSIGN(dst_ub, F16_UB);

    for (uint32_t m = cid; m < num_matrices; m += block_num) {
        uint32_t h = m % static_cast<uint32_t>(H);
        uint32_t chunk_idx = m / static_cast<uint32_t>(H);

        for (int64_t t = 0; t < total_tokens; ++t) {
            int64_t off = t * static_cast<int64_t>(H * C) + static_cast<int64_t>(h * C);

            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<float, Gm1D, GmS1> gm(src + off, gs);
                SrcUB ld;
                TASSIGN(ld, F32_UB);
                TLOAD(ld, gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(dst_ub, src_ub, RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<half, Gm1D, GmS1> gm(dst + off, gs);
                DstUB st;
                TASSIGN(st, F16_UB);
                TSTORE(gm, st);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
#endif
}

#endif  // __CCE_AICORE__

// ===================================================================
// Include original kernel implementations in separate namespaces.
// ===================================================================

namespace mk_cumsum {
#include "chunk_cumsum.cpp"
}

namespace mk_kkt {
#include "scaled_dot_kkt.cpp"
}

namespace mk_solve {
#include "tri_inverse_impl.cpp"
}

namespace mk_wy {
#include "wy_fast.cpp"
}

namespace mk_h {
#include "chunk_h.cpp"
}

namespace mk_o {
#include "chunk_o.cpp"
}

AICORE inline void mega_solve_tril(__gm__ half *out, __gm__ half *in, __gm__ half *minus_id, uint32_t matrix_size,
                                   uint32_t num_matrices, uint32_t num_bsnd_heads, __gm__ int32_t *cu_seqlens,
                                   uint32_t is_lower)
{
    if (num_matrices <= get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 1, true, half>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
    else if (num_matrices <= 2u * get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 2, true, half>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
    else
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 4, true, half>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
}

template <int32_t H>
AICORE inline void mega_kernel_impl(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr,
                                    GM_ADDR msk_lower_ptr, GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr,
                                    GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr,
                                    GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr,
                                    GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr,
                                    GM_ADDR h0_ptr, int64_t has_initial_state, GM_ADDR kkt_ws_ptr,
                                    GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr,
                                    GM_ADDR o_ws_qk_ptr, GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr,
                                    uint32_t num_key_heads, int64_t batch_size, int64_t seq_len,
                                    int64_t total_tokens, uint32_t num_matrices,
                                    GM_ADDR ssm_state_ptr, int64_t ssm_state_index)
{
    constexpr int32_t D = GDN_D;
    constexpr int32_t C = GDN_C;

    if (num_key_heads == 0 || (static_cast<uint32_t>(H) % num_key_heads) != 0) {
        return;
    }

    mk_cumsum::cumsum_kernel<H, C>(reinterpret_cast<__gm__ float *>(g_in_ptr),
                                   reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                   reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len);

#ifdef MEGA_STOP_AFTER_CUMSUM
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_SYNC1
    return;
#endif

    mega_transpose_TH_to_HT<float, H>(reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                      reinterpret_cast<__gm__ float *>(g_t_ptr), total_tokens);
    mega_transpose_TH_to_HT<half, H>(reinterpret_cast<__gm__ half *>(beta_ptr),
                                     reinterpret_cast<__gm__ half *>(beta_t_ptr), total_tokens);

#ifdef MEGA_STOP_AFTER_TRANSPOSE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_kkt::kkt_kernel<H, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(beta_t_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_lower_ptr),
        reinterpret_cast<__gm__ half *>(kkt_ws_ptr), reinterpret_cast<__gm__ half *>(A_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens, num_key_heads);

#if defined(__DAV_C220_CUBE__)
    pipe_barrier(PIPE_ALL);
    wait_flag_dev(2);
    wait_flag_dev(3);
#endif

#ifdef MEGA_STOP_AFTER_KKT
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mega_solve_tril(reinterpret_cast<__gm__ half *>(A_inv_ptr), reinterpret_cast<__gm__ half *>(A_ptr),
                    reinterpret_cast<__gm__ half *>(minus_id_ptr), C, num_matrices, H,
                    reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), 1);

#ifdef MEGA_STOP_AFTER_SOLVE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_CAST
    pipe_barrier(PIPE_ALL);
    return;
#endif

#ifdef MEGA_STOP_AFTER_SYNC_BEFORE_WY
    return;
#endif

    mk_wy::wy_fast_kernel<H, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(v_ptr),
        reinterpret_cast<__gm__ half *>(beta_t_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ half *>(A_inv_ptr), reinterpret_cast<__gm__ half *>(wy_ws_a1_ptr),
        reinterpret_cast<__gm__ half *>(wy_ws_a2_ptr), reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len,
        total_tokens, num_key_heads);

#if defined(__DAV_C220_VEC__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
        wait_flag_dev(4);
    }
#endif

#ifdef MEGA_STOP_AFTER_WY
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_h::chunk_h_kernel<H, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ half *>(s_ptr), reinterpret_cast<__gm__ half *>(v_new_ptr),
        reinterpret_cast<__gm__ half *>(fs_ptr), reinterpret_cast<__gm__ half *>(h0_ptr), has_initial_state,
        reinterpret_cast<__gm__ half *>(h_ws_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size,
        seq_len, total_tokens, num_key_heads, reinterpret_cast<__gm__ float *>(ssm_state_ptr), ssm_state_index);

#ifdef MEGA_STOP_AFTER_H
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_o::chunk_o_kernel<H, D, C>(
        reinterpret_cast<__gm__ half *>(q_ptr), reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ half *>(v_new_ptr), reinterpret_cast<__gm__ half *>(s_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_full_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_qk_ptr), reinterpret_cast<__gm__ half *>(o_ws_qs_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_gated_ptr), reinterpret_cast<__gm__ half *>(o_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens, num_key_heads);

#if defined(__DAV_C220_CUBE__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
    }
#endif
}

extern "C" __global__ __aicore__ void
GDN_KERNEL_NAME(GM_ADDR mixed_qkv_ptr, GM_ADDR z_ptr, GM_ADDR b_ptr, GM_ADDR a_input_ptr,
    GM_ADDR conv_weight_ptr, GM_ADDR conv_state_ptr, GM_ADDR a_log_ptr, GM_ADDR dt_bias_ptr,
    GM_ADDR ssm_state_ptr, GM_ADDR norm_weight_ptr, GM_ADDR msk_lower_ptr, GM_ADDR msk_full_ptr,
    GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR packed_qkv_ptr, GM_ADDR g_in_ptr,
    GM_ADDR beta_ptr, GM_ADDR initial_state_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr,
    GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr, GM_ADDR w_ptr,
    GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr, GM_ADDR conv_state_out_ptr,
    GM_ADDR ssm_state_out_ptr, GM_ADDR out_ptr, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(Qwen35GdnPrefillSuperOpTilingData);
    GET_TILING_DATA_WITH_STRUCT(Qwen35GdnPrefillSuperOpTilingData, tiling_data, tiling);
    (void)conv_state_out_ptr;

    RunQwen35PrefillConv(mixed_qkv_ptr, conv_weight_ptr, conv_state_ptr, packed_qkv_ptr,
                         tiling_data.total_tokens, tiling_data.token_block_size,
                         tiling_data.token_block_count, tiling_data.conv_state_index);
    PrepareQwen35Gate(a_input_ptr, b_ptr, a_log_ptr, dt_bias_ptr, g_in_ptr, beta_ptr,
                      tiling_data.total_tokens);
    ZeroInitialState(initial_state_ptr);
    ZeroMegaScratch(A_ptr, A_inv_f32_ptr, A_inv_ptr, s_ptr, fs_ptr,
                    tiling_data.total_tokens, tiling_data.num_matrices);
    SyncAllImpl<false>();

    GM_ADDR q_ptr = packed_qkv_ptr;
    GM_ADDR k_ptr = packed_qkv_ptr + static_cast<uint64_t>(tiling_data.total_tokens) * 1024 * sizeof(half);
    GM_ADDR v_ptr = packed_qkv_ptr + static_cast<uint64_t>(tiling_data.total_tokens) * 2048 * sizeof(half);
    GM_ADDR user_ws = AscendC::GetUserWorkspace(workspace);
    const uint64_t tile_bytes = static_cast<uint64_t>(GDN_C) * GDN_C * sizeof(half);

    GM_ADDR kkt_ws_ptr = user_ws;
    GM_ADDR wy_ws_a1_ptr = kkt_ws_ptr + static_cast<uint64_t>(tiling_data.block_dim) * 2 * tile_bytes;
    GM_ADDR wy_ws_a2_ptr = wy_ws_a1_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR h_ws_ptr = wy_ws_a2_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR o_ws_qk_ptr = h_ws_ptr + static_cast<uint64_t>(tiling_data.block_dim) * 4 * tile_bytes;
    GM_ADDR o_ws_qs_ptr = o_ws_qk_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR o_ws_gated_ptr = o_ws_qs_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;

    mega_kernel_impl<24>(
        q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr, msk_full_ptr, minus_id_ptr,
        cu_seqlens_ptr, o_ptr, g_sum_ptr, g_t_ptr, beta_t_ptr, A_ptr, A_inv_f32_ptr,
        A_inv_ptr, w_ptr, u_ptr, s_ptr, v_new_ptr, fs_ptr, initial_state_ptr, 0,
        kkt_ws_ptr, wy_ws_a1_ptr, wy_ws_a2_ptr, h_ws_ptr, o_ws_qk_ptr, o_ws_qs_ptr,
        o_ws_gated_ptr, tiling_data.num_key_heads, tiling_data.batch_size,
        tiling_data.seq_len, tiling_data.total_tokens, tiling_data.num_matrices,
        ssm_state_out_ptr, tiling_data.ssm_state_index);

    SyncAllImpl<false>();
    RunQwen35OutputNorm(o_ptr, z_ptr, norm_weight_ptr, out_ptr, tiling_data.total_tokens);
}

// The CANN wrapper generated for mixed AIC/AIV kernels calls matmul::clearWorkspace
// after including this source. Keep this include after PTO code so CANN's DYNAMIC
// enum does not collide with pto::DYNAMIC in the kernel templates above.
#include "lib/matmul_intf.h"
