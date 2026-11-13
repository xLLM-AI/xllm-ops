/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * CrossAttention kernel body: AIC (QK / PV) and AIV (softmax / normalisation).
 *
 * CV direct connect: S / P / O never go through GM staging.
 *   QK: the cube result stays in L0C and is fixpipe'd straight into each AIV sub's UB (S ring)
 *   softmax: AIV computes exp(s - m) in UB with RegBase (__VEC_SCOPE__ + RegTensor) and writes P
 *            from UB into the L1 A1 ring slot (CopyUbufToCbuf, row major to NZ fractal)
 *   PV: cube loads the A1 ring slot straight into L0A; O accumulates in L0C across beats and is
 *       fixpipe'd into both subs' UB once per task, then normalised and written back to GM
 * Beat pipeline, one KV tile per beat:
 *   AIC pre-issues QK one S ring deep, then runs PV beat by beat (driven by P_READY)
 *   AIV runs softmax beat by beat; the P ring is CA_P_RING deep and P_FREE recycles the L1 slots
 * Numerics: m is computed once per task from the beat 0 block maximum and reused, so no rescale is
 *   needed; l accumulates element-wise in UB and is reduced per row at the end of the task.
 */
#ifndef CROSS_ATTENTION_KERNEL_H
#define CROSS_ATTENTION_KERNEL_H

#include "kernel_operator.h"
#include "cross_attention_common.h"

namespace Xllm {


using AscendC::Add;
using AscendC::Cast;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopy;
using AscendC::DataCopyParams;
using AscendC::Duplicate;
using AscendC::Fixpipe;
using AscendC::FixpipeParamsArch3510;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::LoadData;
using AscendC::LoadData2DParams;
using AscendC::LoadData2DParamsV2;
using AscendC::LoadData2dTransposeParams;
using AscendC::LoadDataWithTranspose;
using AscendC::Mmad;
using AscendC::MmadParams;
using AscendC::Muls;
using AscendC::Nd2NzParams;
using AscendC::PipeBarrier;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;

// RegBase namespace: AscendC::MicroAPI is only an alias of AscendC::Reg and it exists only in the
// __NPU_ARCH__ == 3510 pass; using Reg directly also resolves in the host pass.
namespace Mapi = AscendC::Reg;

// L0C -> UB Fixpipe config: isFixPixelMerge (second argument) set to true selects the UB destination.
constexpr AscendC::FixpipeConfig CA_CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

// RegBase Cast traits for fp32 -> 16 bit. fp16 uses NO_SAT (the softmax output is inside [0, 1]) and
// bf16 uses SAT, matching the official arch35 operators.
constexpr Mapi::CastTrait CA_CAST_F32_TO_FP16 = {Mapi::RegLayout::ZERO, Mapi::SatMode::NO_SAT,
                                                 Mapi::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr Mapi::CastTrait CA_CAST_F32_TO_BF16 = {Mapi::RegLayout::ZERO, Mapi::SatMode::SAT,
                                                 Mapi::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// ---- Offsets inside the UB scalar area, relative to CA_UB_SCAL ----
constexpr uint32_t CA_SCAL_RED = 0;   // 64 floats: ReduceMax / Exp landing buffer
constexpr uint32_t CA_SCAL_TMP = 384; // 32 floats: merge denominator
constexpr uint32_t CA_SCAL_MRG = 512; // 64 floats: merge group maximum m*

// Task descriptor, parsed independently from tiling on AIC and AIV
struct CaTaskDesc {
    uint32_t batchIdx;
    uint32_t kvHeadIdx;
    uint32_t qRowTileIdx;
    uint32_t kvSegIdx;
    uint32_t nQ;      // q rows actually covered by this tile
    uint32_t rows;    // R = nQ * groupSize
    uint32_t kvLoops; // beat count
    uint32_t segStartBlock;
    uint32_t segValid;
    uint64_t qHeadBase; // element offset of q row 0, head kh*groupSize
};

__aicore__ inline void ParseTask(const __gm__ CrossAttentionTilingData* t, uint32_t taskIdx,
                                 const GlobalTensor<int64_t>& lensGm, CaTaskDesc& d)
{
    const uint32_t kSeg = t->kSeg;
    d.kvSegIdx = taskIdx % kSeg;
    const uint32_t base = taskIdx / kSeg;
    d.qRowTileIdx = base % t->qRowTileNum;
    d.kvHeadIdx = (base / t->qRowTileNum) % t->kvHeads;
    d.batchIdx = base / (t->qRowTileNum * t->kvHeads);

    const uint32_t qRowStart = d.qRowTileIdx * t->qRowsPerTile;
    d.nQ = t->qSeqlen - qRowStart;
    if (d.nQ > t->qRowsPerTile) {
        d.nQ = t->qRowsPerTile;
    }
    d.rows = d.nQ * t->groupSize;

    d.segStartBlock = d.kvSegIdx * t->mergeSegBlocks;
    uint32_t segBlocksThis = 0;
    if (d.segStartBlock < t->maxNumBlocksPerBatch) {
        segBlocksThis = t->maxNumBlocksPerBatch - d.segStartBlock;
        if (segBlocksThis > t->mergeSegBlocks) {
            segBlocksThis = t->mergeSegBlocks;
        }
    }
    const int64_t kvLen = lensGm.GetValue(static_cast<int64_t>(d.batchIdx));
    int64_t valid = kvLen - static_cast<int64_t>(d.segStartBlock) * static_cast<int64_t>(t->blockSize);
    if (valid < 0) {
        valid = 0;
    }
    const int64_t cap = static_cast<int64_t>(segBlocksThis) * static_cast<int64_t>(t->blockSize);
    if (valid > cap) {
        valid = cap;
    }
    d.segValid = static_cast<uint32_t>(valid);
    d.kvLoops = CaCeilDiv(d.segValid, t->kvTile);

    d.qHeadBase = ((static_cast<uint64_t>(d.batchIdx) * t->qSeqlen + qRowStart) * t->qHeads +
                   static_cast<uint64_t>(d.kvHeadIdx) * t->groupSize) *
                  t->headDim;
}

// Host derives kSeg from maxNumBlocksPerBatch (capacity only), so a small actual kvLen leaves empty
// segments behind. Every core recomputes the effective segment count from the same lens and falls
// back to the kSeg == 1 epilogue when the real block count does not fill one segment.
__aicore__ inline uint32_t CaEffSeg(const __gm__ CrossAttentionTilingData* t,
                                    const GlobalTensor<int64_t>& lensGm)
{
    const uint32_t kSeg = t->kSeg;
    if (kSeg <= 1u) {
        return 1u;
    }
    uint32_t maxBlocks = 0u;
    for (uint32_t b = 0; b < t->batch; ++b) {
        int64_t len = lensGm.GetValue(static_cast<int64_t>(b));
        if (len < 0) {
            len = 0;
        }
        const uint32_t blocks = CaCeilDiv(static_cast<uint32_t>(len), t->blockSize);
        if (blocks > maxBlocks) {
            maxBlocks = blocks;
        }
    }
    const uint32_t need = CaCeilDiv(maxBlocks, t->mergeSegBlocks);
    if (need <= 1u) {
        return 1u;
    }
    return need < kSeg ? need : kSeg;
}

// ---------------------------------------------------------------------------
// AIC: QK / PV
// ---------------------------------------------------------------------------
template <typename T>
class CrossAttentionAic {
public:
    __aicore__ inline void Init(const __gm__ CrossAttentionTilingData* t, __gm__ uint8_t* query, __gm__ uint8_t* key,
                                __gm__ uint8_t* value, __gm__ uint8_t* blockTable, __gm__ uint8_t* actualKvLens,
                                __gm__ uint8_t* workspace)
    {
        // The CV path needs no S / O staging workspace and P reaches L1 over the CV channel, so the
        // workspace only holds the merge slots.
        tiling_ = t;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value));
        btGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
        lensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(actualKvLens));
        coreIdx_ = GetBlockIdx();
        oPending_ = false;
        l0Slot_ = 0u;
        pvASlot_ = 0u;
        vHead_ = false;
        effSeg_ = CaEffSeg(t, lensGm_);
    }

    __aicore__ inline void Process()
    {
        PrimeBeatFlags();

        // Task chain: the front prefetch of task j is issued while task j-1 still runs PV, so the Q panel
        // and the first K window are already on MTE2 and the task boundary adds no GM latency in front of
        // the QK chain. The first task takes the same path: Q / K / V are issued together instead of
        // starting K only after Q has landed.
        CaTaskDesc cur;
        uint32_t taskIdx = coreIdx_;
        bool haveCur = false;
        while (taskIdx < tiling_->totalTaskNum) {
            ParseTask(tiling_, taskIdx, lensGm_, cur);
            if (cur.kvLoops != 0u) {
                haveCur = true;
                break;
            }
            taskIdx += tiling_->usedCoreNum;
        }
        if (haveCur) {
            // A cold start has nothing in flight, so the V head window can go with the front prefetch; across
            // tasks it cannot, because the V slots are still held by this task's PV.
            PrefetchFront(cur, true);
        }
        while (haveCur) {
            uint32_t nextIdx = taskIdx + tiling_->usedCoreNum;
            CaTaskDesc next;
            bool haveNext = false;
            while (nextIdx < tiling_->totalTaskNum) {
                ParseTask(tiling_, nextIdx, lensGm_, next);
                if (next.kvLoops != 0u) {
                    haveNext = true;
                    break;
                }
                nextIdx += tiling_->usedCoreNum;
            }
            RunTask(cur, haveNext ? &next : nullptr);
            cur = next;
            taskIdx = nextIdx;
            haveCur = haveNext;
        }

        // kSeg > 1: AIV needs every core's merge slot before merging. SyncAll<true> (AIV only) does not
        // guarantee cross-core visibility here, so the combined AIC + AIV barrier is used: AIV goes through
        // its own AIC and the AICs synchronise globally. When the real block count does not fill one segment
        // effSeg_ is already 1 and neither side enters this path.
        if (effSeg_ > 1u) {
            AscendC::SyncAll<false>();
        }
    }

private:
    __aicore__ inline void RunTask(const CaTaskDesc& d, const CaTaskDesc* next)
    {
        const uint32_t loops = d.kvLoops;
        loops_ = loops;
        // The Q panel was already prefetched into L1 by the previous task (or the cold start); only wait for
        // it to land and split it to L0A.
        PrepQ(d);
        // The V head window is filled only at the task head; PvBeat keeps CA_V_L1_SLOTS-1 beats of rolling
        // prefetch afterwards.
        if (!vHead_) {
            PrefetchVHead(d);
        }
        vHead_ = false;
        // Rolling schedule, one beat at a time:
        //   Prologue: pre-issue QK one S ring deep (AIV softmaxes concurrently) and, after each QK beat,
        //             refill the L1 slot just released by SplitB with beat +CA_S_RING.
        //   Steady state: run PV(i) first, then issue QK(i+CA_QK_LEAD). QK always leads PV by one S ring, so
        //             P(i+1) already exists before PV(i) and the P prefetch works across beats instead of
        //             exposing P_READY once per block head.
        //             PV stays contiguous (QK is appended to the same beat), so the CUBE PV chain is not
        //             interrupted.
        // L0B shares one slot pair between the K and V panels and l0Slot_ alternates it across QK and PV.
        const uint32_t pro = (loops < CA_QK_LEAD) ? loops : CA_QK_LEAD;
        for (uint32_t j = 0; j < pro; ++j) {
            if (j >= CA_S_RING) {
                // The previous owner of S slot j%CA_S_RING (beat j-CA_S_RING) has been read by AIV
                WaitSFree();
            }
            QkBeat(d, j);
            NotifySReady();
            // The K L1 slot is released by SplitB inside QkBeat, so refill it for beat +CA_S_RING right away
            if (j + CA_S_RING < loops) {
                PrefetchK(d, j + CA_S_RING);
            }
        }
        for (uint32_t i = 0; i < loops; ++i) {
            // Tail: push the next task's Q and first K window onto MTE2. This task's K slots have all been
            // returned and the remaining PV beats cover that GM time. V is left out, its slots are still held.
            if (next != nullptr && i == ((loops >= 2u) ? (loops - 2u) : 0u)) {
                PrefetchFront(*next, false);
            }
            PvBeat(d, i);
            if (i + pro < loops) {
                // The previous owner of ring slot (i+pro)%CA_S_RING (beat i+pro-CA_S_RING) has been read by AIV
                WaitSFree();
                QkBeat(d, i + pro);
                NotifySReady();
                if (i + pro + CA_S_RING < loops) {
                    PrefetchK(d, i + pro + CA_S_RING);
                }
            }
        }

        // Credit drain: one S_FREE is produced per beat and AIC only consumes the ones beyond the ring
        // depth, so the remainder is topped up at the end of the task to keep the count constant across
        // tasks. P_FREE goes the other way: AIC is the producer and never consumes it.
        const uint32_t drainS = loops < CA_S_RING ? loops : CA_S_RING;
        for (uint32_t k = 0; k < drainS; ++k) {
            WaitSFree();
        }

        // Deliver O from L0C into both subs' UB once per task, then wait for AIV to return the O buffer
        if (oPending_) {
            WaitOFree();
        }
        FinishO(d);
        oPending_ = true;
    }

    // ---- AIC side cross-core handshake (mode 4 intra block, one id per sub, see cross_attention_common.h) ----
    __aicore__ inline void NotifySReady()
    {
        CrossCoreSetFlag<4, PIPE_FIX>(CA_FLAG_S_READY);
        CrossCoreSetFlag<4, PIPE_FIX>(CA_FLAG_S_READY + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void NotifyOReady()
    {
        CrossCoreSetFlag<4, PIPE_FIX>(CA_FLAG_O_READY);
        CrossCoreSetFlag<4, PIPE_FIX>(CA_FLAG_O_READY + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void NotifyPFree()
    {
        CrossCoreSetFlag<4, PIPE_MTE1>(CA_FLAG_P_FREE);
        CrossCoreSetFlag<4, PIPE_MTE1>(CA_FLAG_P_FREE + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitPReady()
    {
        // Stays on PIPE_S: in the CV path these waits only order "P written" / "P read", and attaching them
        // to PIPE_MTE1 deadlocks on multi-beat shapes.
        CrossCoreWaitFlag<4>(CA_FLAG_P_READY);
        CrossCoreWaitFlag<4>(CA_FLAG_P_READY + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitPFree()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_P_FREE);
        CrossCoreWaitFlag<4>(CA_FLAG_P_FREE + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitOFree()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_O_FREE);
        CrossCoreWaitFlag<4>(CA_FLAG_O_FREE + CA_FLAG_SUB_STRIDE);
    }

    // AIV has finished reading the S ring slot (one credit per sub)
    __aicore__ inline void WaitSFree()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_S_FREE);
        CrossCoreWaitFlag<4>(CA_FLAG_S_FREE + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline uint64_t KVElemBase(const CaTaskDesc& d, uint32_t beat) const
    {
        const uint32_t logicalBlock = d.segStartBlock + beat;
        const int32_t physBlock =
            btGm_.GetValue(static_cast<int64_t>(d.batchIdx) * tiling_->maxNumBlocksPerBatch + logicalBlock);
        uint64_t off = static_cast<uint64_t>(physBlock) * tiling_->blockSize * tiling_->kvHeads * tiling_->headDim;
        off += static_cast<uint64_t>(d.kvHeadIdx) * tiling_->headDim;
        return off;
    }

    __aicore__ inline void BuildKV(const CaTaskDesc& d, uint32_t beat, LocalTensor<T>& b1, GlobalTensor<T>& src)
    {
        const uint64_t off = KVElemBase(d, beat);
        Nd2NzParams nd;
        nd.ndNum = 1;
        nd.nValue = CA_KV_TILE;
        nd.dValue = CA_HEAD_DIM;
        nd.srcNdMatrixStride = 0;
        nd.srcDValue = tiling_->kvHeads * tiling_->headDim;
        nd.dstNzC0Stride = CA_KV_TILE;
        nd.dstNzNStride = 1;
        nd.dstNzMatrixStride = 0;
        DataCopy(b1, src[off], nd);
    }

    // The A panel has two L1 layouts, selected by whether rows is a multiple of CA_BLOCK_CUBE:
    //   interleaved (rows%16==0): the whole panel is one big NZ matrix with fractal order
    //     k_block*nbCount+m_block. Nd2Nz uses dstNzC0Stride = rows (d direction block stride = rows
    //     units of 32 B) and a single descriptor copies the whole panel. MTE2 is limited by descriptor
    //     count, not by bytes, so this is the cheapest form.
    //   blocked (tail rows%16!=0): copy one 16-row block at a time, contiguous between blocks, with
    //     dstNzC0Stride = 16. When rows is not a multiple of 16 the d direction block stride would land
    //     in the middle of a fractal, so this is the only option.
    static constexpr bool CaInterleaved(uint32_t rows)
    {
        return (rows % CA_BLOCK_CUBE) == 0u;
    }

    __aicore__ inline void BuildQ(const CaTaskDesc& d, LocalTensor<T>& a1)
    {
        const uint32_t g = tiling_->groupSize;
        if (CaInterleaved(d.rows)) {
            // The g heads of one q row are contiguous in GM (row stride headDim) and q rows are qHeads*headDim
            // apart. On the interleaved L1 panel matrix p starts at row p*g and the matrices are g*C0 apart.
            Nd2NzParams nd;
            nd.ndNum = static_cast<uint16_t>(d.nQ);
            nd.nValue = static_cast<uint16_t>(g);
            nd.dValue = CA_HEAD_DIM;
            nd.srcNdMatrixStride = static_cast<uint64_t>(tiling_->qHeads) * tiling_->headDim;
            nd.srcDValue = CA_HEAD_DIM;
            nd.dstNzC0Stride = static_cast<uint16_t>(d.rows);
            nd.dstNzNStride = 1;
            nd.dstNzMatrixStride = g * CA_C0;
            DataCopy(a1, qGm_[d.qHeadBase], nd);
            return;
        }
        const uint32_t perBlock = (g <= CA_BLOCK_CUBE) ? (CA_BLOCK_CUBE / g) : 1u;
        for (uint32_t p = 0; p < d.nQ; ++p) {
            Nd2NzParams nd;
            nd.ndNum = 1;
            nd.nValue = static_cast<uint16_t>(g);
            nd.dValue = CA_HEAD_DIM;
            nd.srcNdMatrixStride = 0;
            nd.srcDValue = CA_HEAD_DIM;
            nd.dstNzC0Stride = CA_BLOCK_CUBE;
            nd.dstNzNStride = 1;
            nd.dstNzMatrixStride = 0;
            const uint32_t blk = p / perBlock;
            const uint32_t pos = p % perBlock;
            const uint32_t dst = blk * CA_BLOCK_CUBE * CA_HEAD_DIM + pos * g * CA_C0;
            const uint64_t off = d.qHeadBase + static_cast<uint64_t>(p) * tiling_->qHeads * tiling_->headDim;
            DataCopy(a1[dst], qGm_[off], nd);
        }
    }

    // Identity copy of a whole panel: the interleaved L1 panel (Nd2Nz dstNzC0Stride = rows) is byte
    // identical to the Zn layout of L0A, so one Mmad with M = rows needs a single descriptor:
    //   mStep covers all m blocks, kStep the 8 k fractals, and the k fractal stride is nbCount on both
    //   sides. The only difference from SplitA(interleaved=true) is the destination side: SplitA lays
    //   every 16-row block out flat for per-block Mmad.
    __aicore__ inline void SplitAWhole(LocalTensor<T>& a2, LocalTensor<T>& a1, uint32_t nbCount)
    {
        LoadData2DParamsV2 ld;
        ld.mStartPosition = 0;
        ld.kStartPosition = 0;
        ld.mStep = static_cast<uint16_t>(nbCount);
        ld.kStep = CA_HEAD_DIM / CA_C0;
        ld.srcStride = static_cast<int32_t>(nbCount);
        ld.dstStride = static_cast<uint16_t>(nbCount);
        ld.ifTranspose = false;
        LoadData(a2, a1, ld);
    }

    __aicore__ inline void SplitA(LocalTensor<T>& a2, LocalTensor<T>& a1, uint32_t nbCount, bool interleaved)
    {
        // One L1->L0A copy per 16-row block: the destination is contiguous in both layouts (dstStride=1);
        // on the source side the 8 k fractals of one m block are nbCount fractals apart when interleaved.
        LoadData2DParamsV2 ld;
        ld.mStartPosition = 0;
        ld.kStartPosition = 0;
        ld.mStep = 1;
        ld.kStep = CA_HEAD_DIM / CA_C0;
        ld.srcStride = static_cast<int32_t>(interleaved ? nbCount : 1u);
        ld.dstStride = 1;
        ld.ifTranspose = false;
        for (uint32_t nb = 0; nb < nbCount; ++nb) {
            const uint32_t src = interleaved ? (nb * CA_BLOCK_CUBE * CA_C0) : (nb * CA_BLOCK_CUBE * CA_HEAD_DIM);
            const uint32_t dst = nb * CA_BLOCK_CUBE * CA_HEAD_DIM;
            LoadData(a2[dst], a1[src], ld);
        }
    }

    // K stored row major as (kv, hd) in L1 is already the (K=hd, N=kv) Zn layout, copy it directly
    __aicore__ inline void SplitBForQk(LocalTensor<T>& b2, LocalTensor<T>& b1)
    {
        LoadData2DParams ld;
        ld.repeatTimes = (CA_KV_TILE / CA_BLOCK_CUBE) * (CA_HEAD_DIM / CA_C0);
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        LoadData(b2, b1, ld);
    }

    // V is row major (K=kv, N=hd) and needs an in-flight Nz -> Zn transpose
    __aicore__ inline void SplitBForPv(LocalTensor<T>& b2, LocalTensor<T>& b1)
    {
        constexpr uint32_t dstOffset = (CA_HEAD_DIM / CA_BLOCK_CUBE) * CA_BLOCK_CUBE * CA_C0;
        constexpr uint32_t srcOffset = CA_BLOCK_CUBE * CA_C0;
        LoadData2dTransposeParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = CA_HEAD_DIM / CA_C0;
        ld.srcStride = CA_HEAD_DIM / CA_BLOCK_CUBE;
        ld.dstGap = 0;
        ld.dstFracGap = 0;
        for (uint32_t i = 0; i < CA_KV_TILE / CA_BLOCK_CUBE; ++i) {
            LoadDataWithTranspose(b2[i * dstOffset], b1[i * srcOffset], ld);
        }
    }

    // ---- Reverse dependencies inside AIC (L1 / L0 reuse) ----
    // L0B shares one slot pair between the QK K panel and the PV V panel, alternating globally per beat
    // (l0Slot_), so the next beat can already copy L1->L0 into the other slot while the previous Mmad
    // still runs and MTE1 hides under Mmad. L0A uses two P slots as well; the Q panel is loaded once per
    // task, so no A dependency appears inside a beat.
    // The header only exports EVENT_ID0..3; 4..7 are used as integer literals (8 hardware event ids).
    static constexpr int32_t CA_L0B_S0 = EVENT_ID2;
    static constexpr int32_t CA_L0B_S1 = EVENT_ID3;
    static constexpr int32_t CA_PV_A_S0 = EVENT_ID0;
    static constexpr int32_t CA_PV_A_S1 = EVENT_ID1;
    static constexpr int32_t CA_L1_Q = 4;      // Q's L1 panel (issued once per task)

    // Two dependency families for the L1 ring slots, 8 ids each. MTE2_MTE1: Q 0 / V 1-2 / K 4-7.
    // MTE1_MTE2: KFree 0,5,6,7 / VFree 1,2 / Q 4.
    // P uses the A5 CV channel (AIV's MTE3 writes L1 directly), so AIC needs no MTE2 copy and id 3 is
    // free.
    //   MTE1_MTE2 (slot may be overwritten: produced by MTE1 after the read, consumed by MTE2 before load)
    static constexpr int32_t KFreeId(uint32_t slot)
    {
        return (slot == 0u) ? 5 : ((slot == 1u) ? 6 : ((slot == 2u) ? 7 : 0));
    }
    static constexpr int32_t VFreeId(uint32_t slot) { return 1 + static_cast<int32_t>(slot); }
    //   MTE2_MTE1 (this beat's copy has landed in L1; ED0 is reserved for the immediate Q barrier)
    static constexpr int32_t KCopyId(uint32_t slot) { return 4 + static_cast<int32_t>(slot); }
    static constexpr int32_t VCopyId(uint32_t slot) { return 1 + static_cast<int32_t>(slot); }

    static constexpr uint32_t KL1Off(uint32_t beat)
    {
        return CA_B1_ADDR_Q + (beat % CA_K_L1_SLOTS) * CA_K_SLOT_BYTES;
    }
    static constexpr uint32_t VL1Off(uint32_t beat)
    {
        return CA_B1_ADDR_P + (beat % CA_V_L1_SLOTS) * CA_V_SLOT_BYTES;
    }

    __aicore__ inline void PrimeBeatFlags()
    {
        SetFlag<HardEvent::MTE1_MTE2>(CA_L1_Q);
        // S L0C double-slot credit: every QkBeat waits for "previous fixpipe finished reading"; prime it.
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        // One "may overwrite" credit per K / V L1 slot, consumed by the first prefetch window of each task.
        // Every beat sets and consumes exactly one, so the credit count is conserved across tasks (see
        // RunTask / PvBeat).
        for (uint32_t s = 0; s < CA_K_L1_SLOTS; ++s) {
            SetFlag<HardEvent::MTE1_MTE2>(KFreeId(s));
        }
        for (uint32_t s = 0; s < CA_V_L1_SLOTS; ++s) {
            SetFlag<HardEvent::MTE1_MTE2>(VFreeId(s));
        }
        SetFlag<HardEvent::M_MTE1>(CA_L0B_S0);
        SetFlag<HardEvent::M_MTE1>(CA_L0B_S1);
        SetFlag<HardEvent::M_MTE1>(CA_PV_A_S0);
        SetFlag<HardEvent::M_MTE1>(CA_PV_A_S1);
    }

    static constexpr int32_t L0BSlotId(uint32_t slot) { return (slot == 0u) ? CA_L0B_S0 : CA_L0B_S1; }
    static constexpr int32_t PvASlotId(uint32_t slot) { return (slot == 0u) ? CA_PV_A_S0 : CA_PV_A_S1; }

    // S ring slot (both AIV subs use the same offsets on physically separate UB)
    __aicore__ inline uint32_t UbSlot(uint32_t ring, uint32_t slotBytes) const
    {
        return CA_UB_S + (ring % CA_S_RING) * slotBytes;
    }

    // ---- Front prefetch (issue only, never wait) ----
    // Shared by the cold start and the cross-task prefetch: issue Q into L1 and the first K window into
    // the L1 ring slots without any barrier. The former overlaps Q's GM latency with K's, the latter
    // moves the whole GM latency under the previous task's PV.
    // withV is true only on the cold start: across tasks the V slots are still held by this task's PV.
    __aicore__ inline void PrefetchFront(const CaTaskDesc& d, bool withV)
    {
        LocalTensor<T> a1(TPosition::A1, CA_A1_ADDR_Q, CA_A_ELEMS);
        WaitFlag<HardEvent::MTE1_MTE2>(CA_L1_Q);
        BuildQ(d, a1);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        const uint32_t hi = (d.kvLoops < CA_S_RING) ? d.kvLoops : CA_S_RING;
        for (uint32_t i = 0; i < hi; ++i) {
            const uint32_t ks = i % CA_K_L1_SLOTS;
            WaitFlag<HardEvent::MTE1_MTE2>(KFreeId(ks));
            LocalTensor<T> b1(TPosition::B1, KL1Off(i), CA_B_ELEMS);
            BuildKV(d, i, b1, kGm_);
            SetFlag<HardEvent::MTE2_MTE1>(KCopyId(ks));
        }
        if (withV) {
            PrefetchVHead(d);
        }
    }

    // Single-beat K prefetch: after every QK beat, refill beat +CA_S_RING.
    // The call site must follow QkBeat directly, where SplitB has just returned the MTE1_MTE2 credit.
    __aicore__ inline void PrefetchK(const CaTaskDesc& d, uint32_t beat)
    {
        const uint32_t ks = beat % CA_K_L1_SLOTS;
        WaitFlag<HardEvent::MTE1_MTE2>(KFreeId(ks));
        LocalTensor<T> b1(TPosition::B1, KL1Off(beat), CA_B_ELEMS);
        BuildKV(d, beat, b1, kGm_);
        SetFlag<HardEvent::MTE2_MTE1>(KCopyId(ks));
    }

    // V head window: issue CA_V_L1_SLOTS beats at once; PvBeat keeps the rolling prefetch afterwards
    __aicore__ inline void PrefetchVHead(const CaTaskDesc& d)
    {
        const uint32_t vn = (d.kvLoops < CA_V_L1_SLOTS) ? d.kvLoops : CA_V_L1_SLOTS;
        for (uint32_t i = 0; i < vn; ++i) {
            const uint32_t vs = i % CA_V_L1_SLOTS;
            WaitFlag<HardEvent::MTE1_MTE2>(VFreeId(vs));
            LocalTensor<T> b1(TPosition::B1, VL1Off(i), CA_B_ELEMS);
            BuildKV(d, i, b1, vGm_);
            SetFlag<HardEvent::MTE2_MTE1>(VCopyId(vs));
        }
        vHead_ = true;
    }

    // Task head: wait for Q to land in L1 (already issued by the front prefetch), split it to L0A, then
    // reuse the Q panel for every beat.
    __aicore__ inline void PrepQ(const CaTaskDesc& d)
    {
        LocalTensor<T> a1(TPosition::A1, CA_A1_ADDR_Q, CA_A_ELEMS);
        LocalTensor<T> a2(TPosition::A2, CA_A2_ADDR_Q, CA_A_ELEMS);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        const uint32_t nbCount = CaCeilDiv(d.rows, CA_BLOCK_CUBE);
        if ((d.rows % CA_BLOCK_CUBE == 0u) && (nbCount > 1u)) {
            // Pairs with the large-M path in QkBeat: copy the interleaved L1 panel into L0A unchanged
            SplitAWhole(a2, a1, nbCount);
        } else {
            SplitA(a2, a1, nbCount, CaInterleaved(d.rows));
        }
        SetFlag<HardEvent::MTE1_MTE2>(CA_L1_Q);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
    }

    __aicore__ inline void QkBeat(const CaTaskDesc& d, uint32_t beat)
    {
        const uint32_t slot = l0Slot_;
        const uint32_t ks = beat % CA_K_L1_SLOTS;
        LocalTensor<T> b1(TPosition::B1, KL1Off(beat), CA_B_ELEMS);
        LocalTensor<T> a2(TPosition::A2, CA_A2_ADDR_Q, CA_A_ELEMS);
        LocalTensor<T> b2(TPosition::B2, slot * CA_B2_SLOT_BYTES, CA_B_ELEMS);
        LocalTensor<float> co1(TPosition::CO1, CA_CO1_ADDR_Q + (beat & 1u) * CA_CO1_S_SLOT_BYTES,
                               CA_MAX_R * CA_HEAD_DIM);

        // This beat's K was issued by a prefetch window in RunTask; only wait for it to land in L1.
        WaitFlag<HardEvent::MTE2_MTE1>(KCopyId(ks));

        // The L0B slot may be overwritten once the previous Mmad has read it
        WaitFlag<HardEvent::M_MTE1>(L0BSlotId(slot));
        SplitBForQk(b2, b1);
        SetFlag<HardEvent::MTE1_MTE2>(KFreeId(ks));
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        // S uses two L0C slots alternating on beat parity: this beat's Mmad only waits for the fixpipe of
        // beat-2 to finish reading the same slot. With a single slot it would have to wait for its own
        // fixpipe to move all of S into UB, which costs a fixed bubble on every beat.
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        const uint32_t nbCount = CaCeilDiv(d.rows, CA_BLOCK_CUBE);
        // When rows is a multiple of 16 and spans more than one block, the whole A panel is one big NZ
        // matrix and a single Mmad computes it: eight m=16 Mmads per beat carry little math each while every
        // issue pays a fixed overhead, so the fragmented form wastes issue slots. The tail case (rows not a
        // multiple of 16) keeps the per-block form, where L0C holds parallel 16-row tiles.
        const bool bigM = (d.rows % CA_BLOCK_CUBE == 0u) && (nbCount > 1u);
        MmadParams mp;
        mp.m = static_cast<uint16_t>(bigM ? d.rows : CA_BLOCK_CUBE);
        mp.n = static_cast<uint16_t>(CA_KV_TILE);
        mp.k = static_cast<uint16_t>(CA_HEAD_DIM);
        if (bigM) {
            mp.cmatrixInitVal = true;
            Mmad(co1, a2, b2, mp);
        } else {
            for (uint32_t nb = 0; nb < nbCount; ++nb) {
                mp.cmatrixInitVal = true;
                Mmad(co1[nb * CA_BLOCK_CUBE * CA_HEAD_DIM], a2[nb * CA_BLOCK_CUBE * CA_HEAD_DIM], b2, mp);
            }
        }
        SetFlag<HardEvent::M_MTE1>(L0BSlotId(slot));
        l0Slot_ ^= 1u;
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

        // S goes from L0C straight into both AIV subs' UB: deliver per sub row range and let subBlockId pick
        // the target UB. The row ranges match CaSubBegin / CaSubRows on the AIV side and are 16-row aligned.
        const uint32_t ubOff = UbSlot(beat, CA_S_SLOT_BYTES);
        for (uint32_t sub = 0; sub < 2u; ++sub) {
            const uint32_t nRows = CaSubRows(d.rows, sub);
            if (nRows == 0u) {
                continue;
            }
            const uint32_t begin = CaSubBegin(d.rows, sub);
            LocalTensor<float> dst(TPosition::VECCALC, ubOff, CA_UB_ROWS * CA_HEAD_DIM);
            if (bigM) {
                // Large M: L0C is one rows x 128 NZ matrix with C0 stride = rows, so one descriptor per sub is enough
                FixpipeWide(dst, co1, begin, nRows, sub, CA_KV_TILE, d.rows);
            } else {
                FixpipeRows(dst, co1, begin, nRows, sub, CA_KV_TILE);
            }
        }
        // Return the credit without waiting: the next user of this slot is the Mmad of beat+2, which waits there
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void PvBeat(const CaTaskDesc& d, uint32_t beat)
    {
        const uint32_t a1Off = CA_A1_ADDR_P + (pSeq_ % CA_P_RING) * CA_P_SLOT_BYTES;
        const uint32_t slot = l0Slot_;
        const uint32_t aslot = pvASlot_;
        const uint32_t vs = beat % CA_V_L1_SLOTS;
        LocalTensor<T> a1(TPosition::A1, a1Off, CA_A_ELEMS);
        LocalTensor<T> b1(TPosition::B1, VL1Off(beat), CA_B_ELEMS);
        LocalTensor<T> a2(TPosition::A2, CA_A2_ADDR_P + aslot * CA_A2_P_SLOT_BYTES, CA_A_ELEMS);
        LocalTensor<T> b2(TPosition::B2, slot * CA_B2_SLOT_BYTES, CA_B_ELEMS);
        LocalTensor<float> co1(TPosition::CO1, CA_CO1_ADDR_P, CA_MAX_R * CA_HEAD_DIM);
        // AIV writes P into the L1 A1 ring slot over the A5 CV channel, each sub covering its own rows, so AIC
        // needs neither a GM read back nor an Nd2Nz: once P_READY arrives the slot already holds the L0A
        // fractal order. The L0A P slot being free means Mmad(beat-2) has finished and SplitA runs in order
        // on MTE1, hence SplitA(beat-4) is complete and the L1 P slot (beat%CA_P_RING) can be overwritten.
        WaitFlag<HardEvent::M_MTE1>(PvASlotId(aslot));

        // This beat's V was delivered by the prefetch of beat-CA_V_L1_SLOTS (or by the task head window)
        WaitFlag<HardEvent::MTE2_MTE1>(VCopyId(vs));

        WaitFlag<HardEvent::M_MTE1>(L0BSlotId(slot));
        SplitBForPv(b2, b1);
        // Return this slot's credit and spend it immediately on the prefetch of beat+CA_V_L1_SLOTS, which
        // needs exactly this slot; MTE1 has finished reading it, so the overwrite is safe.
        SetFlag<HardEvent::MTE1_MTE2>(VFreeId(vs));
        const uint32_t pf = beat + CA_V_L1_SLOTS;
        if (pf < loops_) {
            WaitFlag<HardEvent::MTE1_MTE2>(VFreeId(vs));
            BuildKV(d, pf, b1, vGm_);
            SetFlag<HardEvent::MTE2_MTE1>(VCopyId(vs));
        }
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);

        // This beat's P has been written into the L1 ring slot by AIV over the CV channel and both subs must
        // be done. The wait sits just before SplitA so AIV's latency hides under this beat's V copy / SplitB.
        WaitPReady();
        const uint32_t nbCount = CaCeilDiv(d.rows, CA_BLOCK_CUBE);
        // P's L1 layout has the same two cases as Q in QK:
        //   when rows is a multiple of 16 and there is more than one block, the interleaved panel is used
        //   (C0 stride = rows) with SplitAWhole + a single Mmad(m=rows), matching the QK large-M path;
        //   the tail case (rows%16 != 0) or a single block keeps the 16-row block layout (block base
        //   nb*16*128, C0 stride 16 inside a block), which matches CopyPToL1 on the AIV side and the
        //   byte-exact output of the previous GM based version.
        const bool wideP = (d.rows % CA_BLOCK_CUBE == 0u) && (nbCount > 1u);
        if (wideP) {
            SplitAWhole(a2, a1, nbCount);
        } else {
            SplitA(a2, a1, nbCount, false);
        }
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        // P's L1 slot has been read into L0A, return the credit so AIV may overwrite it (CA_P_RING deep)
        NotifyPFree();

        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        // One V panel serves all 16-row blocks; O accumulates in L0C across beats and is cleared on beat 0
        MmadParams mp;
        mp.m = static_cast<uint16_t>(wideP ? d.rows : CA_BLOCK_CUBE);
        mp.n = static_cast<uint16_t>(CA_KV_TILE);
        mp.k = static_cast<uint16_t>(CA_HEAD_DIM);
        if (wideP) {
            // Large M: one Mmad produces the whole rows x 128, so O is a complete NZ matrix in L0C (C0 stride = rows)
            mp.cmatrixInitVal = (beat == 0u);
            Mmad(co1, a2, b2, mp);
        } else {
            for (uint32_t nb = 0; nb < nbCount; ++nb) {
                mp.cmatrixInitVal = (beat == 0u);
                Mmad(co1[nb * CA_BLOCK_CUBE * CA_HEAD_DIM], a2[nb * CA_BLOCK_CUBE * CA_HEAD_DIM], b2, mp);
            }
        }
        SetFlag<HardEvent::M_MTE1>(L0BSlotId(slot));
        SetFlag<HardEvent::M_MTE1>(PvASlotId(aslot));
        l0Slot_ ^= 1u;
        pvASlot_ ^= 1u;
        ++pSeq_;
    }

    // Every 16-row block in L0C is the result of its own Mmad and they lie side by side as 16x128 tiles,
    // not as one NZ matrix. When reading mSize rows Fixpipe indexes by m1 inside the matrix, so mSize > 16
    // would read the neighbouring tile. Hence one Fixpipe per 16-row block; the large-M path uses
    // FixpipeWide below, where L0C is a complete NZ matrix with C0 stride = M.
    __aicore__ inline void FixpipeRows(LocalTensor<float> dst, LocalTensor<float> co1, uint32_t begin,
                                       uint32_t nRows, uint32_t sub, uint32_t nSize)
    {
        for (uint32_t off = 0; off < nRows; off += CA_BLOCK_CUBE) {
            const uint32_t rowsThis = (nRows - off < CA_BLOCK_CUBE) ? (nRows - off) : CA_BLOCK_CUBE;
            FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.mSize = static_cast<uint16_t>(rowsThis);
            fp.nSize = static_cast<uint16_t>(nSize);
            fp.srcStride = static_cast<uint16_t>(CA_BLOCK_CUBE);
            fp.dstStride = nSize;
            fp.subBlockId = (sub != 0u);
            Fixpipe<float, float, CA_CFG_ROW_MAJOR_UB>(dst[off * nSize], co1[(begin + off) * nSize], fp);
        }
    }

    // Large-M path: one Mmad produces a complete rows x nSize NZ matrix, so L0C's C0 stride is
    // mStride(=M). Each sub needs one Fixpipe starting at row begin inside the first n fractal, i.e.
    // begin*C0.
    __aicore__ inline void FixpipeWide(LocalTensor<float> dst, LocalTensor<float> co1, uint32_t begin,
                                       uint32_t nRows, uint32_t sub, uint32_t nSize, uint32_t mStride)
    {
        FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
        fp.mSize = static_cast<uint16_t>(nRows);
        fp.nSize = static_cast<uint16_t>(nSize);
        fp.srcStride = static_cast<uint16_t>(mStride);
        fp.dstStride = nSize;
        fp.subBlockId = (sub != 0u);
        Fixpipe<float, float, CA_CFG_ROW_MAJOR_UB>(dst, co1[begin * CA_C0], fp);
    }

    // Deliver the accumulated O from L0C into both subs' UB once per task
    __aicore__ inline void FinishO(const CaTaskDesc& d)
    {
        LocalTensor<float> co1(TPosition::CO1, CA_CO1_ADDR_P, CA_MAX_R * CA_HEAD_DIM);
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        for (uint32_t sub = 0; sub < 2u; ++sub) {
            const uint32_t nRows = CaSubRows(d.rows, sub);
            if (nRows == 0u) {
                continue;
            }
            const uint32_t begin = CaSubBegin(d.rows, sub);
            LocalTensor<float> dst(TPosition::VECCALC, CA_UB_O, CA_UB_ROWS * CA_HEAD_DIM);
            // Same condition as wideP in PvBeat: the large-M single Mmad leaves O as a complete NZ matrix in L0C
            if ((d.rows % CA_BLOCK_CUBE == 0u) && (d.rows > CA_BLOCK_CUBE)) {
                FixpipeWide(dst, co1, begin, nRows, sub, CA_HEAD_DIM, d.rows);
            } else {
                FixpipeRows(dst, co1, begin, nRows, sub, CA_HEAD_DIM);
            }
        }
        // O's fixpipe uses its own FIX_M credit (EVENT_ID1): EVENT_ID0 is already taken by the S double slot
        // in QK, and sharing one counter makes this wait return early, notifying AIV before O has landed.
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        NotifyOReady();
    }

    const __gm__ CrossAttentionTilingData* tiling_;
    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<int32_t> btGm_;
    GlobalTensor<int64_t> lensGm_;
    uint32_t coreIdx_;
    bool oPending_;
    uint32_t l0Slot_;  // shared L0B slot pointer (alternates across QK / PV)
    uint32_t pvASlot_; // L0A P panel double-slot pointer
    uint32_t loops_;   // KV beats of the current task (bound of the V rolling prefetch)
    bool vHead_;       // this task's V head window was issued by PrefetchFront
    uint32_t effSeg_;  // effective segment count from the actual kvLen (1 = no merge)
    // P's L1 slot index must use a beat counter that is continuous across tasks and paired with pUbSeq_
    // on the AIV side. With a per-task beat, a single-beat task would always use slot 0, the P_FREE wait
    // condition (beat >= CA_P_RING) would never fire across tasks, and AIV would overwrite P before AIC
    // had read the previous task's copy.
    uint32_t pSeq_ = 0u;
};

// ---------------------------------------------------------------------------
// AIV: softmax / normalisation / merge
// ---------------------------------------------------------------------------
template <typename T>
class CrossAttentionAiv {
public:
    __aicore__ inline void Init(const __gm__ CrossAttentionTilingData* t, __gm__ uint8_t* attnOut,
                                __gm__ uint8_t* actualKvLens, __gm__ uint8_t* workspace)
    {
        tiling_ = t;
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attnOut));
        lensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(actualKvLens));
        mergeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        coreIdx_ = GetBlockIdx() >> 1;
        subIdx_ = GetSubBlockIdx();
        // Scalar values are read into registers once in Init: feeding a GM value directly as the scalar
        // operand of a vector instruction is unreliable.
        scaleV_ = t->scaleValue;
        m_ = 0.0f;
        begin_ = 0u;
        nRows_ = 0u;
        pUbSeq_ = 0u;
        effSeg_ = CaEffSeg(t, lensGm_);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t taskIdx = coreIdx_; taskIdx < tiling_->totalTaskNum; taskIdx += tiling_->usedCoreNum) {
            CaTaskDesc d;
            ParseTask(tiling_, taskIdx, lensGm_, d);
            if (d.kvLoops == 0) {
                // The m/l area is partitioned per sub and both subs must write their own region
                if (effSeg_ > 1u) {
                    WriteSentinelSlot(taskIdx);
                }
                continue;
            }
            RunTask(d);
        }

        if (effSeg_ > 1u) {
            // Paired with the AIC side: AIC + AIV global barrier so every core's merge slots have landed
            AscendC::SyncAll<false>();
            for (uint32_t taskIdx = coreIdx_; taskIdx < tiling_->totalTaskNum; taskIdx += tiling_->usedCoreNum) {
                CaTaskDesc d;
                ParseTask(tiling_, taskIdx, lensGm_, d);
                if (d.kvSegIdx == 0) {
                    MergeTask(d);
                }
            }
        }
    }

private:
    // ---- UB views (all built from byte offsets, no TPipe) ----
    __aicore__ inline __local_mem__ float* UbF32(uint32_t off) const
    {
        LocalTensor<float> t(TPosition::VECCALC, off, 8);
        return (__local_mem__ float*)t.GetPhyAddr();
    }

    __aicore__ inline __local_mem__ T* UbT(uint32_t off) const
    {
        LocalTensor<T> t(TPosition::VECCALC, off, 8);
        return (__local_mem__ T*)t.GetPhyAddr();
    }

    __aicore__ inline LocalTensor<float> UbF32Tensor(uint32_t off, uint32_t elems) const
    {
        return LocalTensor<float>(TPosition::VECCALC, off, elems);
    }

    __aicore__ inline LocalTensor<T> UbTTensor(uint32_t off, uint32_t elems) const
    {
        return LocalTensor<T>(TPosition::VECCALC, off, elems);
    }

    // ---- AIV side cross-core handshake (mode 4 intra block, own sub id only) ----
    __aicore__ inline void WaitSReady()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_S_READY + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void NotifyPReady()
    {
        CrossCoreSetFlag<4, PIPE_MTE3>(CA_FLAG_P_READY + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    // The S slot has been read into UB and can be returned to AIC. The call site must sit after all
    // vector reads of S have landed.
    __aicore__ inline void NotifySFree()
    {
        CrossCoreSetFlag<4, PIPE_V>(CA_FLAG_S_FREE + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitPFree()
    {
        // Stays on PIPE_S (attaching it to PIPE_MTE3 deadlocks as well)
        CrossCoreWaitFlag<4>(CA_FLAG_P_FREE + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitOReady()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_O_READY + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }
    // O has been read, AIC may overwrite this sub's O buffer
    __aicore__ inline void NotifyOFree()
    {
        CrossCoreSetFlag<4, PIPE_V>(CA_FLAG_O_FREE + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    // ---- Scalar / vector helpers ----
    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> red = UbF32Tensor(CA_UB_SCAL + CA_SCAL_RED, 64);
        Duplicate(red, x, 64);
        PipeBarrier<PIPE_V>();
        AscendC::Exp(red, red, 64);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return red.GetValue(0);
    }

    __aicore__ inline void ZeroLAcc()
    {
        LocalTensor<float> lacc = UbF32Tensor(CA_UB_LACC, CA_UB_ROWS * CA_HEAD_DIM);
        Duplicate(lacc, 0.0f, CA_UB_ROWS * CA_HEAD_DIM);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunTask(const CaTaskDesc& d)
    {
        begin_ = CaSubBegin(d.rows, subIdx_);
        nRows_ = CaSubRows(d.rows, subIdx_);
        ZeroLAcc();
        const uint32_t loops = d.kvLoops;
        for (uint32_t i = 0; i < loops; ++i) {
            SoftmaxBeat(d, i);
        }
        // AIC produces one P_FREE per beat and this side only consumes the ones beyond the ring depth, so
        // top up the remainder at the end of the task
        const uint32_t drainP = loops < CA_P_RING ? loops : CA_P_RING;
        for (uint32_t k = 0; k < drainP; ++k) {
            WaitPFree();
        }

        WaitOReady();
        if (effSeg_ == 1u) {
            WriteOutput(d);
        } else {
            WriteMergeSlot(d);
        }
    }

    // Mask row: columns [0, nReal) get clampVal (upper clamp), the rest get CA_MINV so exp() returns 0
    __aicore__ inline void BuildMaskRow(uint32_t nReal, float clampVal)
    {
        __local_mem__ float* maskPtr = UbF32(CA_UB_MASK);
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vA, vB;
            uint32_t cFull = 64u;
            // UpdateMask takes the remaining count and post-decrements; beyond one vector width the result is
            // undefined, so clamp explicitly to <= 64.
            uint32_t c0 = (nReal < 64u) ? nReal : 64u;
            uint32_t c1 = (nReal > 64u) ? (nReal - 64u) : 0u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            Mapi::MaskReg m0 = Mapi::UpdateMask<float>(c0);
            Mapi::MaskReg m1 = Mapi::UpdateMask<float>(c1);
            // Note: a masked Duplicate defaults to ZEROING (unselected lanes cleared) and would wipe the CA_MINV
            // laid down above; MERGING is required to keep the previous dst values.
            Mapi::Duplicate(vA, CA_MINV, mFull);
            Mapi::Duplicate<float, Mapi::MaskMergeMode::MERGING>(vA, clampVal, m0);
            Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(maskPtr, vA, mFull);
            Mapi::Duplicate(vB, CA_MINV, mFull);
            Mapi::Duplicate<float, Mapi::MaskMergeMode::MERGING>(vB, clampVal, m1);
            Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(maskPtr + 64, vB, mFull);
        }
    }

    // Beat 0 block maximum over the whole tile, with scale and mask applied; the result lands in the
    // scalar area and is returned.
    __aicore__ inline float ComputeBlockMax(__local_mem__ float* sPtr)
    {
        __local_mem__ float* maskPtr = UbF32(CA_UB_MASK);
        __local_mem__ float* redPtr = UbF32(CA_UB_SCAL + CA_SCAL_RED);
        const uint16_t nRows = static_cast<uint16_t>(nRows_);
        const float scale = scaleV_;
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vS0, vS1, vN0, vN1, vAcc, vR;
            uint32_t cFull = 64u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vN0, maskPtr);
            Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vN1, maskPtr + 64);
            Mapi::Duplicate(vAcc, CA_MINV, mFull);
            for (uint16_t r = 0; r < nRows; ++r) {
                const uint32_t ro = static_cast<uint32_t>(r) * CA_HEAD_DIM;
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vS0, sPtr + ro);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vS1, sPtr + ro + 64);
                Mapi::Muls(vS0, vS0, scale, mFull);
                Mapi::Muls(vS1, vS1, scale, mFull);
                Mapi::Min(vS0, vS0, vN0, mFull);
                Mapi::Min(vS1, vS1, vN1, mFull);
                Mapi::Max(vAcc, vAcc, vS0, mFull);
                Mapi::Max(vAcc, vAcc, vS1, mFull);
            }
            Mapi::ReduceMax(vR, vAcc, mFull);
            Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(redPtr, vR, mFull);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        LocalTensor<float> red = UbF32Tensor(CA_UB_SCAL + CA_SCAL_RED, 64);
        const float mv = red.GetValue(0);
        return (mv > CA_MINV) ? mv : 0.0f;
    }

    // One softmax beat: exp(s*scale - m) accumulated into l and cast to T into the P UB buffer
    __aicore__ inline void SoftmaxRows(__local_mem__ float* sPtr, uint32_t pseq)
    {
        __local_mem__ float* laccPtr = UbF32(CA_UB_LACC);
        __local_mem__ float* maskPtr = UbF32(CA_UB_MASK);
        __local_mem__ T* pPtr = UbT(CA_UB_P + (pseq % CA_P_UB_SLOTS) * CA_UB_P_SLOT_BYTES);
        const uint16_t nRows = static_cast<uint16_t>(nRows_);
        const float mSub = m_;
        const float scale = scaleV_;
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vS0, vS1, vL0, vL1, vM, vN0, vN1;
            Mapi::RegTensor<T> vT0, vT1;
            uint32_t cFull = 64u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            Mapi::Duplicate(vM, mSub, mFull);
            Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vN0, maskPtr);
            Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vN1, maskPtr + 64);
            for (uint16_t r = 0; r < nRows; ++r) {
                const uint32_t ro = static_cast<uint32_t>(r) * CA_HEAD_DIM;
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vS0, sPtr + ro);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vS1, sPtr + ro + 64);
                Mapi::Muls(vS0, vS0, scale, mFull);
                Mapi::Muls(vS1, vS1, scale, mFull);
                // A single Min handles both the tail mask and the overflow clamp
                Mapi::Min(vS0, vS0, vN0, mFull);
                Mapi::Min(vS1, vS1, vN1, mFull);
                Mapi::Sub(vS0, vS0, vM, mFull);
                Mapi::Sub(vS1, vS1, vM, mFull);
                Mapi::Exp(vS0, vS0, mFull);
                Mapi::Exp(vS1, vS1, mFull);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vL0, laccPtr + ro);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vL1, laccPtr + ro + 64);
                Mapi::Add(vL0, vL0, vS0, mFull);
                Mapi::Add(vL1, vL1, vS1, mFull);
                Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(laccPtr + ro, vL0, mFull);
                Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(laccPtr + ro + 64, vL1, mFull);
                CaCastF32To(vT0, vS0, mFull);
                CaCastF32To(vT1, vS1, mFull);
                Mapi::DataCopy<T, Mapi::StoreDist::DIST_PACK_B32>(pPtr + ro, vT0, mFull);
                Mapi::DataCopy<T, Mapi::StoreDist::DIST_PACK_B32>(pPtr + ro + 64, vT1, mFull);
            }
        }
    }

    __aicore__ inline void CaCastF32To(Mapi::RegTensor<T>& dst, Mapi::RegTensor<float>& src, Mapi::MaskReg& mask)
    {
        if constexpr (AscendC::IsSameType<T, half>::value) {
            Mapi::Cast<half, float, CA_CAST_F32_TO_FP16>(dst, src, mask);
        } else {
            Mapi::Cast<bfloat16_t, float, CA_CAST_F32_TO_BF16>(dst, src, mask);
        }
    }

    // P: UB row major -> L1 A1 ring slot over the A5 CV channel (hardware copy_ubuf_to_cbuf).
    // No GM round trip and no TPipe: AIV's MTE3 writes the 32 B fragments straight into the L1 NZ fractal.
    // In UB each 16-column group is one contiguous 32 B run and adjacent rows are CA_HEAD_DIM*2 = 256 B
    // apart, see the per-column descriptor below.
    // The landing layout is byte identical to what AIC's Nd2Nz used to produce, so AIC can drop the GM
    // read back and the Nd2Nz and call SplitA directly.
    __aicore__ inline void CopyPToL1(const CaTaskDesc& d, uint32_t pseq)
    {
        // The slot index shares pSeq_ with AIC: continuous across tasks, otherwise single-beat tasks would
        // keep overwriting slot 0.
        const uint32_t slot = pseq % CA_P_RING;
        LocalTensor<T> pU(TPosition::VECCALC, CA_UB_P + (pseq % CA_P_UB_SLOTS) * CA_UB_P_SLOT_BYTES,
                          CA_UB_ROWS * CA_HEAD_DIM);
        LocalTensor<T> pL1(TPosition::A1, CA_A1_ADDR_P + slot * CA_P_SLOT_BYTES, CA_A_ELEMS);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // Issued per 16-column group: one descriptor covers one 16-column group of one 16-row block. On the
        // source side adjacent rows are 8 x 32 B apart (CA_HEAD_DIM*2 = 256 B); on the destination side the
        // 16 units of 32 B inside one fractal are contiguous (dstStride = 0) and the next column group moves
        // to the next fractal (+512 B = 32 units of 32 B).
        // Descriptor count = blocks x 8, half of the per-row form, while the MTE3 burst count is unchanged.
        // The landing layout is byte identical, so AIC still calls SplitA directly.
        // Large-M case (same condition as wideP in PvBeat): P's L1 layout becomes the interleaved panel, the
        // fractal base of column group g is g*rows*16 and this sub's rows are contiguous inside a fractal.
        // Each descriptor still stays within 16 rows (the hardware destination must be contiguous inside one
        // fractal); only the base is laid out per column group.
        const uint32_t nbCount = CaCeilDiv(d.rows, CA_BLOCK_CUBE);
        const bool wideP = (d.rows % CA_BLOCK_CUBE == 0u) && (nbCount > 1u);
        for (uint32_t c = 0; c < CaCeilDiv(nRows_, CA_BLOCK_CUBE); ++c) {
            const uint32_t left = nRows_ - c * CA_BLOCK_CUBE;
            const uint32_t cnt = left < CA_BLOCK_CUBE ? left : CA_BLOCK_CUBE;
            const uint32_t ubBase = c * CA_BLOCK_CUBE * CA_HEAD_DIM;
            const uint32_t rowBase = begin_ + c * CA_BLOCK_CUBE;
            const uint32_t blkBase = wideP ? 0u : (rowBase / CA_BLOCK_CUBE) * CA_BLOCK_CUBE * CA_HEAD_DIM;
            DataCopyParams cp;
            cp.blockCount = static_cast<uint16_t>(cnt);                              // rows of this block
            cp.blockLen = 1u;                                                        // 32 B = 16 T
            cp.srcStride = static_cast<uint16_t>(CA_HEAD_DIM / CA_C0 - 1u);           // 7 x 32 B between rows
            cp.dstStride = 0u;                                                       // contiguous in a fractal
            for (uint32_t g = 0; g < CA_HEAD_DIM / CA_C0; ++g) {
                const uint32_t dstOff = wideP ? (g * d.rows * CA_C0 + rowBase * CA_C0)
                                              : (blkBase + g * CA_BLOCK_CUBE * CA_C0);
                DataCopy(pL1[dstOff], pU[ubBase + g * CA_C0], cp);
            }
        }
        // P_READY is set only after all descriptors are issued; the flag rides on MTE3 and orders after them
        PipeBarrier<PIPE_MTE3>();
        NotifyPReady();
        // This UB slot is only rewritten CA_P_UB_SLOTS beats later, so set the credit here and leave the MTE3
        // copy under the next beat's vector work. The matching wait is in SoftmaxBeat.
        // Ids 2/3 stay clear of the OUTPUT buffer's MTE3_V (EVENT_ID0); with one id two Sets and one Wait
        // would merge.
        SetFlag<HardEvent::MTE3_V>((pseq % CA_P_UB_SLOTS) == 0u ? CA_PUB_S0 : CA_PUB_S1);
    }

    __aicore__ inline void SoftmaxBeat(const CaTaskDesc& d, uint32_t beat)
    {
        // P's UB slot index advances continuously across tasks: a single-beat task restarts beat at 0, so
        // deriving the slot from beat would misalign the SetFlag / WaitFlag credits at task boundaries.
        const uint32_t pseq = pUbSeq_;
        ++pUbSeq_;
        WaitSReady();
        __local_mem__ float* sPtr = UbF32(CA_UB_S + (beat % CA_S_RING) * CA_S_SLOT_BYTES);

        // Valid columns of this beat (the tail block needs a mask)
        const uint32_t nRealBase = beat * CA_KV_TILE;
        uint32_t nReal = (d.segValid > nRealBase) ? (d.segValid - nRealBase) : 0u;
        if (nReal > CA_KV_TILE) {
            nReal = CA_KV_TILE;
        }
        if (nReal == 0u) {
            nReal = 1u; // ParseTask keeps kvLoops consistent with segValid; this is only a guard
        }

        if (nRows_ != 0u) {
            if (beat == 0u) {
                // m is computed once per task on the first beat: block maximum plus mask, invalid columns excluded
                BuildMaskRow(nReal, CA_BIGV);
                m_ = ComputeBlockMax(sPtr);
            }
            // Valid columns are clamped to m+CA_SOFT_CLAMP, invalid ones are set to CA_MINV
            BuildMaskRow(nReal, m_ + CA_SOFT_CLAMP);
            // The MTE3 of the previous round must finish reading the P slot this beat writes (the credit was set
            // CA_P_UB_SLOTS beats earlier)
            if (pseq >= CA_P_UB_SLOTS) {
                WaitFlag<HardEvent::MTE3_V>((pseq % CA_P_UB_SLOTS) == 0u ? CA_PUB_S0 : CA_PUB_S1);
            }
            SoftmaxRows(sPtr, pseq);
        }
        // The S slot has been read (set on PIPE_V, so the vector read is complete) and AIC may overwrite it
        NotifySFree();

        // The ring credit threshold must follow the per-task beat, not the continuous pseq. Task boundaries
        // are already covered by the O_FREE barrier in FinishO: AIV answers O_FREE only after writing the
        // last P of the task and AIC enters the next task only then, so no extra P_FREE guard is needed
        // across tasks. Judging by pseq would create a cross-task wait that never completes, because a
        // mode 4 flag is just an event register and AIC's early sets do not accumulate.
        if (beat >= CA_P_RING) {
            WaitPFree();
        }
        if (nRows_ != 0u) {
            CopyPToL1(d, pseq);
        } else {
            NotifyPReady();
        }
    }

    __aicore__ inline void ReduceRows()
    {
        __local_mem__ float* laccPtr = UbF32(CA_UB_LACC);
        __local_mem__ float* lsumPtr = UbF32(CA_UB_LSUM);
        const uint16_t nRows = static_cast<uint16_t>(nRows_);
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vL0, vL1, vR;
            uint32_t cFull = 64u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            for (uint16_t r = 0; r < nRows; ++r) {
                const uint32_t ro = static_cast<uint32_t>(r) * CA_HEAD_DIM;
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vL0, laccPtr + ro);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vL1, laccPtr + ro + 64);
                Mapi::Add(vL0, vL0, vL1, mFull);
                Mapi::ReduceSum(vR, vL0, mFull);
                Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(lsumPtr + static_cast<uint32_t>(r) * 64, vR, mFull);
            }
        }
    }

    // out = O / l, cast to T into CA_UB_OUT.
    // 1/l is not computed per row as a scalar: l is broadcast per row and the division is done as a
    // vector op, which removes the per-row GetValue / SetValue pairs and the extra V_S / S_V syncs.
    __aicore__ inline void ScaleAndCast()
    {
        __local_mem__ float* oPtr = UbF32(CA_UB_O);
        __local_mem__ float* lsumPtr = UbF32(CA_UB_LSUM);
        __local_mem__ T* outPtr = UbT(CA_UB_OUT);
        const uint16_t nRows = static_cast<uint16_t>(nRows_);
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vO0, vO1, vI;
            Mapi::RegTensor<T> vT0, vT1;
            uint32_t cFull = 64u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            for (uint16_t r = 0; r < nRows; ++r) {
                const uint32_t ro = static_cast<uint32_t>(r) * CA_HEAD_DIM;
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_BRC_B32>(vI, lsumPtr + static_cast<uint32_t>(r) * 64);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vO0, oPtr + ro);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vO1, oPtr + ro + 64);
                Mapi::Div(vO0, vO0, vI, mFull);
                Mapi::Div(vO1, vO1, vI, mFull);
                CaCastF32To(vT0, vO0, mFull);
                CaCastF32To(vT1, vO1, mFull);
                Mapi::DataCopy<T, Mapi::StoreDist::DIST_PACK_B32>(outPtr + ro, vT0, mFull);
                Mapi::DataCopy<T, Mapi::StoreDist::DIST_PACK_B32>(outPtr + ro + 64, vT1, mFull);
            }
        }
    }

    __aicore__ inline uint64_t OutputElem(const CaTaskDesc& d, uint32_t r) const
    {
        const uint32_t qHeadIdx = d.kvHeadIdx * tiling_->groupSize + (r % tiling_->groupSize);
        const uint32_t qRowIdx = d.qRowTileIdx * tiling_->qRowsPerTile + (r / tiling_->groupSize);
        return ((static_cast<uint64_t>(d.batchIdx) * tiling_->qSeqlen + qRowIdx) * tiling_->qHeads + qHeadIdx) *
               tiling_->headDim;
    }

    // Write the output rows back to GM: the groupSize heads of one q row are adjacent in GM and adjacent
    // q rows are qHeads heads apart, so the whole range fits one multi-block descriptor instead of one
    // DataCopy per q row, each of which pays a fixed launch overhead.
    __aicore__ inline void CopyRowsToGm(const CaTaskDesc& d, const LocalTensor<T>& src, uint32_t begin,
                                        uint32_t rows)
    {
        const uint32_t g = tiling_->groupSize;
        const uint32_t headBytes = tiling_->headDim * sizeof(T);
        const uint32_t blkBytes = g * headBytes;
        const uint32_t gapBytes = (tiling_->qHeads - g) * headBytes;
        if (g != 0u && rows % g == 0u && begin % g == 0u && blkBytes % 32u == 0u && gapBytes % 32u == 0u) {
            DataCopyParams cp;
            cp.blockCount = static_cast<uint16_t>(rows / g);
            cp.blockLen = static_cast<uint16_t>(blkBytes / 32u);
            cp.srcStride = 0; // blocks are contiguous in UB
            cp.dstStride = static_cast<uint16_t>(gapBytes / 32u);
            DataCopy(outGm_[OutputElem(d, begin)], src, cp);
            return;
        }
        uint32_t i = 0;
        while (i < rows) {
            const uint64_t base = OutputElem(d, begin + i);
            uint32_t n = 1;
            while (i + n < rows &&
                   OutputElem(d, begin + i + n) == base + static_cast<uint64_t>(n) * CA_HEAD_DIM) {
                ++n;
            }
            DataCopy(outGm_[base], src[i * CA_HEAD_DIM], n * CA_HEAD_DIM);
            i += n;
        }
    }

    __aicore__ inline void WriteOutput(const CaTaskDesc& d)
    {
        if (nRows_ == 0u) {
            // This sub owns no rows (sub1 is empty when rows <= 16) but must still return the O buffer credit:
            // AIC's WaitOFree expects one credit per sub and would block forever on the next task otherwise.
            NotifyOFree();
            return;
        }
        ReduceRows();
        ScaleAndCast();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // The buffer can be returned as soon as the vector read of O is done: O_FREE is set on PIPE_V
        NotifyOFree();
        LocalTensor<T> outL = UbTTensor(CA_UB_OUT, CA_UB_ROWS * CA_HEAD_DIM);
        // The groupSize heads of one q row are adjacent in GM and adjacent q rows are qHeads heads apart, so
        // one multi-block descriptor covers the whole range (see CopyRowsToGm).
        CopyRowsToGm(d, outL, begin_, nRows_);
        // The OUT buffer is rewritten by the vector unit in the next task: wait for this MTE3 to finish
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline uint32_t TaskIdxOfSeg(const CaTaskDesc& d, uint32_t s) const
    {
        const uint32_t base = (d.batchIdx * tiling_->kvHeads + d.kvHeadIdx) * tiling_->qRowTileNum + d.qRowTileIdx;
        return base * tiling_->kSeg + s;
    }

    __aicore__ inline uint64_t MergeSlotElem(uint32_t taskIdx) const
    {
        return (static_cast<uint64_t>(taskIdx) * tiling_->mergeSlotBytes) / sizeof(float);
    }

    // m/l area: each AIV sub owns 2*CA_MAX_R floats so sub0 / sub1 on the same core cannot overwrite
    // each other
    __aicore__ inline uint64_t MLSlotElem(uint32_t taskIdx) const
    {
        const uint32_t rows = tiling_->rowsPerTask;
        return MergeSlotElem(taskIdx) + static_cast<uint64_t>(rows) * CA_HEAD_DIM +
               static_cast<uint64_t>(subIdx_) * 2ull * CA_MAX_R;
    }

    __aicore__ inline void StoreML(uint32_t taskIdx)
    {
        LocalTensor<float> mL = UbF32Tensor(CA_UB_ML, 2 * CA_MAX_R);
        LocalTensor<float> lsum = UbF32Tensor(CA_UB_LSUM, CA_UB_ROWS * 64);
        for (uint32_t r = 0; r < CA_MAX_R; ++r) {
            mL.SetValue(r, m_); // m
            float lv = 0.0f;
            if (r >= begin_ && (r - begin_) < nRows_) {
                lv = lsum.GetValue((r - begin_) * 64);
            }
            mL.SetValue(CA_MAX_R + r, lv);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(mergeGm_[MLSlotElem(taskIdx)], mL, 2 * CA_MAX_R);
        // mL must land before the cross-core SyncAll, otherwise the merging side reads l = 0 and zeroes the row
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void WriteSentinelSlot(uint32_t taskIdx)
    {
        LocalTensor<float> mL = UbF32Tensor(CA_UB_ML, 2 * CA_MAX_R);
        for (uint32_t r = 0; r < CA_MAX_R; ++r) {
            mL.SetValue(r, CA_MINV);
            mL.SetValue(CA_MAX_R + r, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(mergeGm_[MLSlotElem(taskIdx)], mL, 2 * CA_MAX_R);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void WriteMergeSlot(const CaTaskDesc& d)
    {
        ReduceRows();
        const uint64_t base = MergeSlotElem(TaskIdxOfSeg(d, d.kvSegIdx));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // The buffer can be returned as soon as the vector read of O is done
        NotifyOFree();
        LocalTensor<float> oL = UbF32Tensor(CA_UB_O, CA_UB_ROWS * CA_HEAD_DIM);
        // This sub's rows are contiguous inside the merge slot, one DataCopy covers them
        DataCopy(mergeGm_[base + static_cast<uint64_t>(begin_) * CA_HEAD_DIM], oL, nRows_ * CA_HEAD_DIM);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        StoreML(TaskIdxOfSeg(d, d.kvSegIdx));
    }

    // kSeg > 1: merge the O / l of all segments of one (batch, kvHead, qRowTile) after normalising with
    // the group maximum m*
    __aicore__ inline void MergeTask(const CaTaskDesc& d)
    {
        const uint32_t begin = CaSubBegin(d.rows, subIdx_);
        const uint32_t nRows = CaSubRows(d.rows, subIdx_);
        LocalTensor<float> mL = UbF32Tensor(CA_UB_ML, 2 * CA_MAX_R);
        // The merge accumulator is sized by this sub's nRows and indexed with sub-local row numbers
        LocalTensor<float> totO = UbF32Tensor(CA_UB_MRG, CA_UB_ROWS * CA_HEAD_DIM);
        LocalTensor<float> oL = UbF32Tensor(CA_UB_O, CA_UB_ROWS * CA_HEAD_DIM);
        LocalTensor<float> mStar = UbF32Tensor(CA_UB_SCAL + CA_SCAL_MRG, CA_MAX_R);
        LocalTensor<float> denL = UbF32Tensor(CA_UB_SCAL + CA_SCAL_TMP, 32);

        // 1) group maximum m*
        for (uint32_t i = 0; i < nRows; ++i) {
            mStar.SetValue(i, CA_MINV);
        }
        for (uint32_t s = 0; s < effSeg_; ++s) {
            const uint64_t mBase = MLSlotElem(TaskIdxOfSeg(d, s));
            DataCopy(mL, mergeGm_[mBase], CA_MAX_R);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t i = 0; i < nRows; ++i) {
                const float mv = mL.GetValue(begin + i);
                if (mv > mStar.GetValue(i)) {
                    mStar.SetValue(i, mv);
                }
            }
        }

        // 2) accumulate O and the denominator
        Duplicate(totO, 0.0f, CA_UB_ROWS * CA_HEAD_DIM);
        for (uint32_t i = 0; i < nRows; ++i) {
            denL.SetValue(i, 0.0f);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t s = 0; s < effSeg_; ++s) {
            const uint64_t sBase = MergeSlotElem(TaskIdxOfSeg(d, s));
            const uint64_t mBase = MLSlotElem(TaskIdxOfSeg(d, s));
            DataCopy(oL, mergeGm_[sBase + static_cast<uint64_t>(begin) * CA_HEAD_DIM], nRows * CA_HEAD_DIM);
            DataCopy(mL, mergeGm_[mBase], 2 * CA_MAX_R);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t i = 0; i < nRows; ++i) {
                const float lv = mL.GetValue(CA_MAX_R + begin + i);
                float w = 0.0f;
                if (lv > 0.0f) {
                    w = ExpScalar(mL.GetValue(begin + i) - mStar.GetValue(i));
                }
                denL.SetValue(i, denL.GetValue(i) + w * lv);
                if (w == 0.0f) {
                    Duplicate(oL[i * CA_HEAD_DIM], 0.0f, CA_HEAD_DIM);
                } else {
                    Muls(oL[i * CA_HEAD_DIM], oL[i * CA_HEAD_DIM], w, CA_HEAD_DIM);
                }
            }
            PipeBarrier<PIPE_V>();
            Add(totO, totO, oL, nRows * CA_HEAD_DIM);
            PipeBarrier<PIPE_V>();
            // oL is overwritten by the next DataCopy round: wait for this round's V read, otherwise data is lost
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        // 3) normalise and write back
        LocalTensor<T> outL = UbTTensor(CA_UB_OUT, CA_UB_ROWS * CA_HEAD_DIM);
        // Normalise and cast every row of this sub in UB first and synchronise with MTE3 once, instead of one
        // V_MTE3 / MTE3_V pair per row.
        for (uint32_t i = 0; i < nRows; ++i) {
            const float den = denL.GetValue(i);
            const float inv = den > 0.0f ? (1.0f / den) : 0.0f;
            Muls(totO[i * CA_HEAD_DIM], totO[i * CA_HEAD_DIM], inv, CA_HEAD_DIM);
            Cast(outL[i * CA_HEAD_DIM], totO[i * CA_HEAD_DIM], RoundMode::CAST_RINT, CA_HEAD_DIM);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // Adjacent q rows are a fixed distance apart in GM, one multi-block descriptor covers the range (see
        // CopyRowsToGm)
        CopyRowsToGm(d, outL, begin, nRows);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    const __gm__ CrossAttentionTilingData* tiling_;
    GlobalTensor<T> outGm_;
    GlobalTensor<int64_t> lensGm_;
    GlobalTensor<float> mergeGm_;
    uint32_t coreIdx_;
    uint32_t subIdx_;
    uint32_t begin_;
    uint32_t nRows_;
    float scaleV_;
    float m_;
    uint32_t effSeg_;  // effective segment count from the actual kvLen (1 = no merge)
    // P UB double-slot credit: event ids 2/3 keep it separate from the OUTPUT buffer's MTE3_V (EVENT_ID0).
    // Set and Wait on one id must alternate strictly, so the slot index uses the continuous pUbSeq_
    // rather than beat.
    static constexpr int32_t CA_PUB_S0 = 2;
    static constexpr int32_t CA_PUB_S1 = 3;
    uint32_t pUbSeq_ = 0u;  // P beat counter, continuous across tasks
};

} // namespace Xllm

#endif // CROSS_ATTENTION_KERNEL_H
