#ifndef X_ATTN_CATLASS_KERNEL_H
#define X_ATTN_CATLASS_KERNEL_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_common.hpp"
#include "kernel_operator.h"
using namespace Catlass;
 
template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueFAUnsharedSoftmax
    >
class UnsharedFAInferKernel {
  public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;
    using ElementO = typename BlockMmadPV::ElementC;
    using LayoutO = typename BlockMmadPV::LayoutC;

    CATLASS_DEVICE
    UnsharedFAInferKernel(XAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(XAttnKernelParams const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(XAttnKernelParams const &params) {
        static constexpr uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K
                                                   * sizeof(ElementQ)
                                               + BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K
                                                     * sizeof(ElementK) * 2;

        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);

        // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
        
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gUnsharedK;
        gUnsharedK.SetGlobalBuffer((__gm__ ElementK *)params.unshared_k);
        AscendC::GlobalTensor<ElementV> gUnsharedV;
        gUnsharedV.SetGlobalBuffer((__gm__ ElementV *)params.unshared_v);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementO> gUnsharedO;
        gUnsharedO.SetGlobalBuffer((__gm__ ElementO *)params.unshared_workspace);

        uint32_t maxDecodeStep = faTilingData->maxDecodeStep;
        uint32_t embeddingSize = faTilingData->embeddingSize;
        uint32_t groupSize = faTilingData->groupSize;
        uint32_t unsharedCoreNum = faTilingData->unsharedCoreNum;
        uint32_t unshareGroupCountPerLoop = faTilingData->unshareGroupCountPerLoop;
        uint32_t unshareGroupCountTailLoop = faTilingData->unshareGroupCountTailLoop;
        uint32_t unsharedTaskNumHead = faTilingData->unsharedTaskNumHead;
        uint32_t unsharedTaskNumTail = faTilingData->unsharedTaskNumTail;
        uint32_t unsharedFullCoreNum = faTilingData->unsharedFullCoreNum;
        uint32_t coreNumShared = faTilingData->sharedCoreNum;

        uint32_t relativeCoreIdx = AscendC::GetBlockIdx() - coreNumShared;
        uint32_t taskStartIdx = relativeCoreIdx <= unsharedFullCoreNum ? relativeCoreIdx * unsharedTaskNumHead:
                                (unsharedFullCoreNum * unsharedTaskNumHead + (relativeCoreIdx - unsharedFullCoreNum) * unsharedTaskNumTail) ;
        bool isTailCore = (relativeCoreIdx + 1) == unsharedCoreNum;
        uint32_t taskEndIdx = taskStartIdx + (relativeCoreIdx >= unsharedFullCoreNum ? unsharedTaskNumTail : unsharedTaskNumHead);
        //cce::printf("aic blockIdx:%d, sub block num:%d, loopCount:%d\n", coreIdx, AscendC::GetSubBlockNum(), taskEndIdx - taskStartIdx);
        
        int64_t gmQOffset = taskStartIdx * groupSize * unshareGroupCountPerLoop * embeddingSize;
        int64_t gmKVOffset = taskStartIdx * maxDecodeStep * unshareGroupCountPerLoop * embeddingSize;
        uint64_t gmSOffset = coreNumShared * WORKSPACE_BLOCK_SIZE_DB + relativeCoreIdx * UNSHARED_WORKSPACE_BLOCK_SIZE_DB;
        uint64_t gmPOffset = gmSOffset;

        LayoutQ layoutQTemp(unshareGroupCountPerLoop * groupSize, embeddingSize);
        LayoutK layoutKTemp(embeddingSize, unshareGroupCountPerLoop * maxDecodeStep);
        LayoutV layoutVTemp(unshareGroupCountPerLoop * maxDecodeStep, embeddingSize);
        LayoutO layoutOTemp(unshareGroupCountPerLoop * groupSize, embeddingSize);
        LayoutS layoutSTemp(unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep);
        LayoutP layoutPTemp(unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep);
        for (uint32_t taskIdx = taskStartIdx; taskIdx < taskEndIdx; ++taskIdx) {
            uint32_t actualGroupCount = (isTailCore && (taskIdx + 1 == taskEndIdx)) ? unshareGroupCountTailLoop : unshareGroupCountPerLoop;
            GemmCoord actualBlockShapeQK{actualGroupCount * groupSize, actualGroupCount * maxDecodeStep, embeddingSize};
            GemmCoord actualBlockShapePV{actualGroupCount * groupSize, embeddingSize, actualGroupCount * maxDecodeStep};
            blockMmadQK(
                gQ[gmQOffset], gUnsharedK[gmKVOffset], gS[gmSOffset], 
                layoutQTemp, layoutKTemp, layoutSTemp, actualBlockShapeQK);
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
            blockMmadPV(
                gP[gmPOffset], gUnsharedV[gmKVOffset], gUnsharedO[gmQOffset],
                layoutPTemp, layoutVTemp, layoutOTemp,
                actualBlockShapePV, softmaxReady);
            gmQOffset += groupSize * unshareGroupCountPerLoop * embeddingSize;
            gmKVOffset += maxDecodeStep * unshareGroupCountPerLoop * embeddingSize;
        }
    }
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(XAttnKernelParams const &params) {
        // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<uint32_t> gDecodeStep;
        gDecodeStep.SetGlobalBuffer((__gm__ uint32_t *)params.decodeStep);

        uint32_t batch = faTilingData->batch;
        uint32_t beamSize = faTilingData->beamSize;
        uint32_t numHeads = faTilingData->numHeads;
        uint32_t unsharedKvSeqLen = gDecodeStep.GetValue(0);
        uint32_t maxDecodeStep = faTilingData->maxDecodeStep;
        uint32_t embeddingSize = faTilingData->embeddingSize;
        uint32_t groupSize = faTilingData->groupSize;
        uint32_t unsharedCoreNum = faTilingData->unsharedCoreNum;
        uint32_t unshareGroupCountPerLoop = faTilingData->unshareGroupCountPerLoop;
        uint32_t unshareGroupCountTailLoop = faTilingData->unshareGroupCountTailLoop;
        uint32_t unsharedTaskNumHead = faTilingData->unsharedTaskNumHead;
        uint32_t unsharedTaskNumTail = faTilingData->unsharedTaskNumTail;
        uint32_t unsharedFullCoreNum = faTilingData->unsharedFullCoreNum;
        float scaleValue = faTilingData->scaleValue;
        uint32_t coreNumShared = faTilingData->sharedCoreNum;

        uint32_t gUnsharedOffset = batch * beamSize * numHeads * embeddingSize;
        AscendC::GlobalTensor<ElementO> gUnsharedGm;
        gUnsharedGm.SetGlobalBuffer(((__gm__ ElementO *)params.unshared_workspace) + gUnsharedOffset);
        gUnsharedOffset += batch * beamSize * numHeads;
        AscendC::GlobalTensor<ElementO> gUnsharedGl;
        gUnsharedGl.SetGlobalBuffer(((__gm__ ElementO *)params.unshared_workspace) + gUnsharedOffset);

        uint32_t relativeCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum() - coreNumShared;
        uint32_t taskStartIdx = relativeCoreIdx <= unsharedFullCoreNum ? relativeCoreIdx * unsharedTaskNumHead:
                                (unsharedFullCoreNum * unsharedTaskNumHead + (relativeCoreIdx - unsharedFullCoreNum) * unsharedTaskNumTail) ;
        bool isTailCore = (relativeCoreIdx + 1) == unsharedCoreNum;
        uint32_t taskEndIdx = taskStartIdx + (relativeCoreIdx >= unsharedFullCoreNum ? unsharedTaskNumTail : unsharedTaskNumHead);
        
        EpilogueFAUnsharedSoftmax epilogueFAUnsharedSoftmax(resource, scaleValue, unsharedKvSeqLen, maxDecodeStep, unshareGroupCountPerLoop,
                                                            groupSize);
   
        uint64_t gmSOffset = coreNumShared * WORKSPACE_BLOCK_SIZE_DB + relativeCoreIdx * UNSHARED_WORKSPACE_BLOCK_SIZE_DB;
        uint64_t gmPOffset = gmSOffset;
        uint64_t gmUnsharedGmGlOffset = taskStartIdx * groupSize * unshareGroupCountPerLoop;
        LayoutS layoutSTemp(unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep);
        LayoutP layoutPTemp(unshareGroupCountPerLoop * groupSize, unshareGroupCountPerLoop * maxDecodeStep);
        for (uint32_t taskIdx = taskStartIdx; taskIdx < taskEndIdx; ++taskIdx) {
            Arch::CrossCoreWaitFlag(qkReady);
            uint32_t actualGroupCount = (isTailCore && (taskIdx + 1 == taskEndIdx)) ? unshareGroupCountTailLoop : unshareGroupCountPerLoop;
            GemmCoord actualBlockShapeQK{actualGroupCount * groupSize, actualGroupCount * maxDecodeStep, embeddingSize};
            // FA unshared softmax
            epilogueFAUnsharedSoftmax(
                gP[gmPOffset], gS[gmSOffset], gUnsharedGm[gmUnsharedGmGlOffset], gUnsharedGl[gmUnsharedGmGlOffset], layoutPTemp, layoutSTemp,
                actualBlockShapeQK, actualGroupCount
            );
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
            gmUnsharedGmGlOffset += actualGroupCount * groupSize;
        }
    }
  private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    XAttentionTilingData* faTilingData;
};
 
 /*
     FASharedInferKernel
     Compute Stream
     1. BlockMmadQK
     2. OnlineSoftmax
     3. BlockMmadPV
     4. EpilogueRescaleO 最后rescaleO 不需要div rowsum
 */
 template <
     class BlockMmadQK,
     class BlockMmadPV,
     class BlockMmadQKTail,
     class BlockMmadPVTail,
     class EpilogueOnlineSoftmax,
     class EpilogueRescaleO,
     bool PAGED_CACHE_FLAG = true
 >
 class SharedFAInferKernel {
 public:
     using ArchTag = typename BlockMmadQK::ArchTag;
     using L1TileShape = typename BlockMmadQK::L1TileShape;
     using ElementQ = typename BlockMmadQK::ElementA;
     using LayoutQ = typename BlockMmadQK::LayoutA;
     using ElementK = typename BlockMmadQK::ElementB;
     using LayoutK = typename BlockMmadQK::LayoutB;
     using ElementS = typename BlockMmadQK::ElementC;
     using LayoutS = typename BlockMmadQK::LayoutC;
 
     using ElementP = typename BlockMmadPV::ElementA;
     using LayoutP = typename BlockMmadPV::LayoutA;
     using ElementV = typename BlockMmadPV::ElementB;
     using LayoutV = typename BlockMmadPV::LayoutB;
 
     using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
     using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;
 
     using ElementO = typename EpilogueRescaleO::ElementOutput;
     using LayoutO = typename EpilogueRescaleO::LayoutOutput;
 
     using ElementOTmp = typename EpilogueRescaleO::ElementInput;
     using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;
 
     // Methods
     CATLASS_DEVICE
     SharedFAInferKernel(XAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {
     }
 
     template <int32_t CORE_TYPE = g_coreType>
     CATLASS_DEVICE void operator()(XAttnKernelParams const &params);
 
     template <>
     CATLASS_DEVICE void operator()<AscendC::AIC>(XAttnKernelParams const &params) {
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
         AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
         static constexpr uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K
                                                    * sizeof(ElementQ)
                                                + BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K
                                                      * sizeof(ElementK) * 2;
         BlockMmadQK blockMmadQK(resource);
         BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
 
         BlockMmadQKTail blockMmadQKTail(resource);
         BlockMmadPVTail blockMmadPVTail(resource, L1_QK_SIZE);
         // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
         uint64_t mm1OutSize = faTilingData->mm1OutSize;
         uint64_t smOnlineOutSize = faTilingData->smOnlineOutSize;
         uint32_t batch = faTilingData->batch;  // requestNum 
         uint32_t beamSize = faTilingData->beamSize; 
         uint32_t qHeads = faTilingData->numHeads;
         uint32_t kvHeads = faTilingData->kvHeads;
         uint32_t embed = faTilingData->embeddingSize;
         uint32_t pagedBlockSize = faTilingData->blockSize;
         uint32_t sharedCoreNum = faTilingData->sharedCoreNum;
         uint32_t maxNumBlocksPerBatch = faTilingData->maxNumBlocksPerBatch;
         uint32_t curTotalTaskNum = faTilingData->firstSharedBatchTaskNum;
         uint32_t totalTaskNum = faTilingData->sharedTotalTaskNum;
         uint32_t blockSize = faTilingData->blockSize;
         uint32_t maskType = faTilingData->maskType;
         float scaleValue = faTilingData->scaleValue;
 
         AscendC::GlobalTensor<ElementQ> gQ;
         gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
         AscendC::GlobalTensor<ElementK> gK;
         gK.SetGlobalBuffer((__gm__ ElementK *)params.k_cache);
         AscendC::GlobalTensor<ElementK> gV;
         gV.SetGlobalBuffer((__gm__ ElementK *)params.v_cache);
         AscendC::GlobalTensor<int32_t> gBlockTable;
         gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
         AscendC::GlobalTensor<int64_t> gActualKvseqlen;
         gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
         AscendC::GlobalTensor<ElementS> gS;
         gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
         AscendC::GlobalTensor<ElementP> gP;
         gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
         AscendC::GlobalTensor<ElementOTmp> gOTmp;
         gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
 
         uint64_t strideQO = qHeads * embed;
         uint64_t strideKV = kvHeads * embed;
         uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
         uint32_t groupSize = qHeads / kvHeads;
 
         uint32_t coreIdx = AscendC::GetBlockIdx();
         uint32_t coreNum = sharedCoreNum; // TODO coreNum Need To be modified
         curTotalTaskNum = 0;
         uint32_t preTotalTaskNum = 0;
         uint32_t curBatch = 0;
         uint64_t qBOffset = 0;
         uint64_t kBOffset = 0;
         uint64_t vBOffset = 0;
         uint64_t blockBOffset = 0;
         int64_t qSeqlen = 0;
         int64_t kvSeqlen = 0;
         uint32_t curQNBlockTile;
         uint32_t qNBlockNumPerGroup;
         uint32_t curQNBlockNum;
         int64_t curQSBlockTile;
         uint32_t curQSBlockNum;
 
         preTotalTaskNum = curTotalTaskNum;
         qSeqlen = beamSize;
         kvSeqlen = reinterpret_cast<int64_t>(gActualKvseqlen.GetValue(curBatch));
         curQSBlockTile = GetQSBlockTile(kvSeqlen);
         curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
         qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
         curQNBlockNum = qNBlockNumPerGroup * kvHeads;
         curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
         curTotalTaskNum += curQNBlockNum * curQSBlockNum;
         for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
             while (taskIdx >= curTotalTaskNum) {
                 ++curBatch;
                 preTotalTaskNum = curTotalTaskNum;
                 qBOffset += qSeqlen * strideQO;
                 if constexpr (!PAGED_CACHE_FLAG) {
                     kBOffset += kvSeqlen * strideKV;
                     vBOffset += kvSeqlen * strideKV;
                 } else {
                     blockBOffset += maxNumBlocksPerBatch;
                 }
                 qSeqlen = beamSize;
                 kvSeqlen = reinterpret_cast<int64_t>(gActualKvseqlen.GetValue(curBatch));
                 curQSBlockTile = GetQSBlockTile(kvSeqlen);
                 curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                 qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                 curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                 curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                 curTotalTaskNum += curQNBlockNum * curQSBlockNum;
             }
             uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
             uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
             uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
             uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
             uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
             uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
             uint64_t gmQOffset = qBOffset + qSBlockIdx * curQSBlockTile * strideQO + qHeadIdx * embed;
             uint64_t gmKOffset = kBOffset + kvHeadIdx * embed;
             uint64_t gmVOffset = vBOffset + kvHeadIdx * embed;
             uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile)
                                                                        : curQSBlockTile;
             uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                        ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                        : curQNBlockTile;
             uint32_t rowNum = qSBlockSize * qNBlockSize;
             uint32_t rowNumRound = AlignUp(rowNum, BLOCK_SIZE);
             uint32_t noSkipKvS = kvSeqlen;
             uint32_t noMaskKvS = kvSeqlen;
             uint32_t noMaskTailS = 0;
             // if (maskType != 0) {
             //     uint32_t diffS = kvSeqlen - qSeqlen;
             //     noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
             //     noSkipKvS = Min((uint32_t)kvSeqlen, noSkipKvS);
             //     noMaskKvS = noSkipKvS - qSBlockSize;
             //     noMaskTailS = noMaskKvS % pagedBlockSize;
             // }
             uint32_t maskedKvS = qSBlockSize;
             uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, pagedBlockSize);
             uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
             uint32_t blockStackNum = 4;
             uint32_t stackSeqTile;
             uint32_t stackSeqTileRound = blockStackNum * 128;
             int32_t preLaunch = 2;
             int32_t totalStackSeqNum = (maskType != 0) ? (CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize) + 1)
                                                        : CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize);
             int32_t stackSeqCount = 0;
 
             LayoutQ layoutQTemp(rowNum, embed);
             LayoutK layoutKTemp(strideKV, blockStackNum * pagedBlockSize);
             LayoutV layoutVTemp(blockStackNum * pagedBlockSize, strideKV);
             blockMmadQK.loadQGM(gQ[gmQOffset], layoutQTemp, rowNum, qNBlockSize, qHeads);
             for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
                 if (kvSIdx < kvSLoopNumNoMask) {
                     if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     uint32_t SWorkSpacePingPongFlag = stackSeqCount % (preLaunch + 1);
                     uint64_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB /** (preLaunch + 1)*/
                                          + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                     if constexpr (!PAGED_CACHE_FLAG) {
                         blockMmadQK(
                             gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable, layoutQTemp, layoutKTemp,
                             actualBlockShapeQK, kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV
                         );
                     } else {
                         blockMmadQK(
                             gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable[blockBOffset], layoutQTemp,
                             layoutKTemp, actualBlockShapeQK, kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS,
                             strideKV
                         );
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                 }
                 if (kvSIdx >= preLaunch * blockStackNum) {
                     uint32_t nowkvSIdx = kvSIdx - preLaunch * blockStackNum;
                     if (nowkvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - nowkvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     uint32_t PVWorkSpacePingPongFlag = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint64_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB /** (preLaunch + 1)*/
                                          + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     uint64_t gmOTmpOffset = gmPOffset;
                     LayoutP layoutPTemp(rowNum, stackSeqTileRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     if constexpr (!PAGED_CACHE_FLAG) {
                         blockMmadPV(
                             gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable, layoutPTemp, layoutVTemp,
                             actualBlockShapePV, nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV,
                             softmaxReady
                         );
                     } else {
                         blockMmadPV(
                             gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable[blockBOffset], layoutPTemp,
                             layoutVTemp, actualBlockShapePV, nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS,
                             strideKV, softmaxReady
                         );
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                 }
                 stackSeqCount++;
             }
 
             /*
              * for the secondary loop
              * while masked, it deals the CV stage1(Qk^t/SMOnline) of the final base block(typical shape [128, 512]),
              * and the CV stage2(PV/rescaleO) of the last (prelaunch+1) base blocks while not masked, it deals only the
              * CV stage2(PV/rescaleO) of the last (prelaunch) base blocks
              */
 
             // deal secondary loop conditions
             uint32_t maskedStartIdx = (maskType != 0) ? ((noMaskTailS != 0) ? (kvSLoopNumNoMask - 1) : kvSLoopNumNoMask)
                                                       : AlignUp(kvSLoopNumNoMask, blockStackNum);
             uint32_t noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
             noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? noMaskTailInteStackNum
                                                                    : ((noMaskTailS != 0) ? 0 : blockStackNum);
             uint32_t preLaunchStackNum = (maskType != 0) ? ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum)
                                                          : (preLaunch * blockStackNum);
 
             // masked kvSeqlen loop 
    
             for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum;) {
                 if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
                     stackSeqTile = maskedKvS;
                     uint32_t SWorkSpacePingPongFlag = stackSeqCount % (preLaunch + 1);
                     uint64_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB /** (preLaunch + 1)*/
                                          + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                     if constexpr (!PAGED_CACHE_FLAG) {
                         blockMmadQKTail(
                             gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable, layoutQTemp, layoutKTemp,
                             actualBlockShapeQK, kvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS, strideKV,
                             noMaskTailS, 1
                         );
                     } else {
                         blockMmadQKTail(
                             gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable[blockBOffset], layoutQTemp,
                             layoutKTemp, actualBlockShapeQK, kvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS,
                             strideKV, noMaskTailS, 1
                         );
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                 }
 
                 if (kvSIdx >= preLaunchStackNum) {
                     uint32_t delayedKvSIdx = kvSIdx - preLaunchStackNum;
 
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && (maskType != 0)) {
                         stackSeqTile = maskedKvS;
                     } else if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - delayedKvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     uint32_t PVWorkSpacePingPongFlag = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint64_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB /** (preLaunch + 1)*/
                                          + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     uint64_t gmOTmpOffset = gmPOffset;
                     LayoutP layoutPTemp(rowNum, stackSeqTileRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
 
                     if ((stackSeqCount - preLaunch == totalStackSeqNum - 1) && (maskType != 0)) { // 加mask
                         if constexpr (!PAGED_CACHE_FLAG) {
                             blockMmadPVTail(
                                 gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable, layoutPTemp,
                                 layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumTotal, pagedBlockSize,
                                 noSkipKvS, strideKV, softmaxReady, noMaskTailS, 1
                             );
                         } else {
                             blockMmadPVTail(
                                 gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable[blockBOffset],
                                 layoutPTemp, layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumTotal,
                                 pagedBlockSize, noSkipKvS, strideKV, softmaxReady, noMaskTailS, 1
                             );
                         }
                     } else { // 不加mask
                         if constexpr (!PAGED_CACHE_FLAG) {
                             blockMmadPV(
                                 gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable, layoutPTemp,
                                 layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumNoMask, pagedBlockSize,
                                 noMaskKvS, strideKV, softmaxReady
                             );
                         } else {
                             blockMmadPV(
                                 gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable[blockBOffset],
                                 layoutPTemp, layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumNoMask,
                                 pagedBlockSize, noMaskKvS, strideKV, softmaxReady
                             );
                         }
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                 }
                 if ((maskType != 0) && (stackSeqCount - preLaunch == totalStackSeqNum - 2)) {
                     kvSIdx += noMaskTailInteStackNum;
                 } else {
                     kvSIdx += blockStackNum;
                 }
                 stackSeqCount++;
             }
         }
 
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
 
         AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
 
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
     }
 
 
     template <>
     CATLASS_DEVICE void operator()<AscendC::AIV>(XAttnKernelParams const &params) {
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
 
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
 
         // Get tiling parameters
         // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
         uint64_t mm1OutSize = faTilingData->mm1OutSize;
         uint64_t smOnlineOutSize = faTilingData->smOnlineOutSize;
         uint64_t mm2OutSize = faTilingData->mm2OutSize;
         uint32_t batch = faTilingData->batch;
         uint32_t beamSize = faTilingData->beamSize;
         uint32_t qHeads = faTilingData->numHeads;
         uint32_t kvHeads = faTilingData->kvHeads;
         uint32_t embed = faTilingData->embeddingSize;
         uint32_t sharedCoreNum = faTilingData->sharedCoreNum;
         uint32_t pagedBlockSize = faTilingData->blockSize;
         uint32_t maxNumBlocksPerBatch = faTilingData->maxNumBlocksPerBatch;
         uint32_t firstBatchTaskNum = faTilingData->firstSharedBatchTaskNum;
         uint32_t totalTaskNum = faTilingData->sharedTotalTaskNum;
         uint32_t maskType = faTilingData->maskType;
         float scaleValue = faTilingData->scaleValue;
         
         uint64_t gOffsetTempO = batch * beamSize * qHeads * embed;
         uint64_t gMaxOffset = batch * beamSize * qHeads;
         // Get the memory offset address of the input on Global Memory
         AscendC::GlobalTensor<int64_t> gActualKvseqlen;
         gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
         AscendC::GlobalTensor<ElementO> gO;
         gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
         AscendC::GlobalTensor<ElementS> gS;
         gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
         AscendC::GlobalTensor<ElementP> gP;
         gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
         AscendC::GlobalTensor<ElementOTmp> gOTmp;
         gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
         AscendC::GlobalTensor<ElementOTmp> gOUpdate;
         gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.oUpdate);
 
         // shared Gm and Gl output
         AscendC::GlobalTensor<float> gSharedO;
         AscendC::GlobalTensor<ElementS> gSharedSum;
         AscendC::GlobalTensor<ElementS> gSharedMax;
         gSharedO.SetGlobalBuffer((__gm__ float *)params.shared_workspace);
         gSharedMax.SetGlobalBuffer((__gm__ ElementS *)params.shared_workspace + gOffsetTempO);
         gSharedSum.SetGlobalBuffer((__gm__ ElementS *)params.shared_workspace + gOffsetTempO + gMaxOffset);
         

         uint32_t groupSize = qHeads / kvHeads;
         uint32_t embedRound = RoundUp(embed, BLOCK_SIZE);
 
         EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
         EpilogueRescaleO epilogueRescaleO(resource);
 
         // uint32_t curTotalTaskNum = 0;
         uint32_t preTotalTaskNum = 0;
         uint32_t curBatch = 0;
         uint32_t oBatchOffset = 0;
         uint32_t qSeqlen = static_cast<uint32_t>(beamSize);
         uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
         uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
         uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
         uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
         uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
         uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
         uint32_t curTotalTaskNum = firstBatchTaskNum;
 
         uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
         // coreNum Need to be changed
         uint32_t coreNum = sharedCoreNum;

         // Go through each task.
         for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
             // Get the offset of each core on the GM.
             while (taskIdx >= curTotalTaskNum) {
                 curBatch++;
                 oBatchOffset += qSeqlen * qHeads * embed;
                 preTotalTaskNum = curTotalTaskNum;
                 qSeqlen = static_cast<uint32_t>(beamSize);
                 kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                 curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                 qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                 curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                 curQSBlockTile = GetQSBlockTile(kvSeqlen);
                 curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                 curTotalTaskNum += curQNBlockNum * curQSBlockNum;
             }
             uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
             uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
             uint32_t qNBlockIdx = taskIdxCurBatch % curQNBlockNum;
             uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
 
             uint32_t oSOffset = qSBlockIdx * curQSBlockTile * qHeads * embed;
             uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
             uint32_t qStartNIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
             uint32_t oNOffset = qStartNIdx * embed;
             uint32_t gmOffsetO = oBatchOffset + oSOffset + oNOffset;

             // shared sum max workspace offset
             // Calculate shared workspace offset for this core's task
             // gsharedout size: [batch*beamSize, numHeads, 1]
             // Each core handles: [qSBlockSize, qNBlockSize] elements
             // Offset = curBatch * beamSize * qHeads + qSBlockIdx * curQSBlockTile * qHeads + qStartNIdx;
             uint32_t gSharedOffset = curBatch * qSeqlen * qHeads + 
             qSBlockIdx * curQSBlockTile * qHeads + qStartNIdx;
             // cce::printf("gSharedOffset:%d\n", gSharedOffset);
             uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile)
                                                                        : curQSBlockTile;
             uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                        ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                        : curQNBlockTile;
             uint32_t rowNum = qSBlockSize * qNBlockSize;
             uint32_t rowNumRound = RoundUp(rowNum, BLOCK_SIZE);
 
             uint32_t noSkipKvS = kvSeqlen;
             uint32_t noMaskKvS = kvSeqlen;
             uint32_t noMaskTailS = 0;
             if (maskType != 0) {
                 uint32_t diffS = kvSeqlen - qSeqlen;
                 noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
                 noSkipKvS = Min(kvSeqlen, noSkipKvS);
                 noMaskKvS = noSkipKvS - qSBlockSize;
                 noMaskTailS = noMaskKvS % pagedBlockSize;
             }
             uint32_t maskedKvS = qSBlockSize;
             uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
             uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, pagedBlockSize);
             uint32_t blockStackNum = 4;
             uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
             uint32_t stackSeqTile;
             int32_t preLaunch = 2;
             // totalStackSeqNum = 1
             int32_t totalStackSeqNum = (maskType != 0) ? (CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize) + 1)
                                                        : CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize);
             int32_t stackSeqCount = 0;
 
             // no mask kvSeqlen loop
             for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
 
                 if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                     stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                 } else {
                     stackSeqTile = pagedBlockSize * blockStackNum;
                 }
                 uint32_t isLastStackTile = (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) ? 1 : 0;
                 uint32_t stackSeqTileRound = RoundUp(stackSeqTile, BLOCK_SIZE);
                 LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
                 LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                 GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                 uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                 uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB /** (preLaunch + 1)*/ + // cube core offset
                                      curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;            // single cube core db offset
                 // vec core offset will be processed within epilogue block
                 uint32_t gmOffsetP = gmOffsetS;
                 // AscendC::printf("stackSeqCount:%d\n", stackSeqCount);
                 Arch::CrossCoreWaitFlag(qkReady);
                 
                 // online softmax
                 epilogueOnlineSoftmax(
                     gP[gmOffsetP], gS[gmOffsetS], gSharedMax[gSharedOffset], gSharedSum[gSharedOffset], layOutP, layOutS, actualBlockShapeQK, (stackSeqCount == 0),
                     isLastStackTile, qSBlockSize, qNBlockSize, curStackTileMod, qHeads
                 );
                 Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
 
                 if (kvSIdx >= preLaunch * blockStackNum) {
                     uint32_t delayedKvSIdx = kvSIdx - preLaunch * blockStackNum;
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     LayoutO layoutO(qSeqlen, embed * qHeads);
                     LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB /** (preLaunch + 1)*/
                                             + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                     Arch::CrossCoreWaitFlag(pvReady);
                     // rescale O
                     epilogueRescaleO(
                         gO[gmOffsetO], gOTmp[gmOffsetOTmp], gSharedO[gmOffsetO], layoutO, layoutOTmp, actualBlockShapePV, qSBlockSize,
                         qNBlockSize, (stackSeqCount - preLaunch == 0), 0, curStackTileMod
                     );
                 }
                 stackSeqCount++;
             }
             /*
              * for the secondary loop
              * while masked, it deals the CV stage1(Qk^t/SMOnline) of the final base block(typical shape [128, 512]),
              * and the CV stage2(PV/rescaleO) of the last (prelaunch+1) base blocks while unmasked, it deals the CV
              * stage1(Qk^t/SMOnline) of the last (prelaunch+1) base blocks
              */
             // deal secondary loop conditions
             uint32_t maskedStartIdx = (maskType != 0) ? ((noMaskTailS != 0) ? (kvSLoopNumNoMask - 1) : kvSLoopNumNoMask)
                                                       : AlignUp(kvSLoopNumNoMask, blockStackNum);
             uint32_t noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
             noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? noMaskTailInteStackNum
                                                                    : ((noMaskTailS != 0) ? 0 : blockStackNum);
             uint32_t preLaunchStackNum = (maskType != 0) ? ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum)
                                                          : (preLaunch * blockStackNum);
             // masked kvSeqlen loop
             // maskedStartIdx = AlignUp(kvSLoopNumNoMask, blockStackNum)
             // kvSLoopNumTotal = kvSLoopNumNoMask  so maskedStartIdx 即 kvSIdx >= kvSLoopNumTotal
             for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum;) {
                // nomask场景下 kvSIdx不会小于kvSLoopNumTotal 所以不会进入该if
                 if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
                    //  stackSeqTile = maskedKvS;
                    //  uint32_t stackSeqTileRound = RoundUp(stackSeqTile, BLOCK_SIZE);
                    //  LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
                    //  LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                    //  LayoutMask layOutMask(1024, 1024, 1024);
                    //  GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    //  uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                    //  uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + // cube core offset
                    //                       curStackTileMod * WORKSPACE_BLOCK_SIZE_DB; // single cube core db offset
                    //  // vec core offset will be processed within epilogue block
                    //  uint32_t gmOffsetP = gmOffsetS;
                    //  // online softmax
                    //  epilogueOnlineSoftmax(
                    //      gP[gmOffsetP], gS[gmOffsetS], gMask, layOutP, layOutS, layOutMask, actualBlockShapeQK,
                    //      (stackSeqCount == 0), qSBlockSize, qNBlockSize, curStackTileMod, qkReady
                    //  );
                     Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                 }
                 if (kvSIdx >= preLaunchStackNum) {
                     uint32_t delayedKvSIdx = kvSIdx - preLaunchStackNum;
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && (maskType != 0)) {
                         stackSeqTile = maskedKvS;
                     } else if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - delayedKvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     LayoutO layoutO(qSBlockSize, embed * qHeads);
                     LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB /** (preLaunch + 1)*/
                                             + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                     Arch::CrossCoreWaitFlag(pvReady);
                     // rescale O
                     epilogueRescaleO(
                         gO[gmOffsetO], gOTmp[gmOffsetOTmp], gSharedO[gmOffsetO], layoutO, layoutOTmp, actualBlockShapePV, qSBlockSize,
                         qNBlockSize, (stackSeqCount - preLaunch == 0),
                         (stackSeqCount - preLaunch == totalStackSeqNum - 1), curStackTileMod
                     );
                 }
                 if ((maskType != 0) && (stackSeqCount - preLaunch == totalStackSeqNum - 2)) {
                     kvSIdx += noMaskTailInteStackNum;
                 } else {
                     kvSIdx += blockStackNum;
                 }
                 stackSeqCount++;
             }
         }
 
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
 
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
     }
 
 private:
     Arch::Resource<ArchTag> resource;
     Arch::CrossCoreFlag qkReady{QK_READY_ID};
     Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
     Arch::CrossCoreFlag pvReady{PV_READY_ID};
     XAttentionTilingData* faTilingData;
 };
 

template <class EpilogueCombineScale>
class CombineScaleKernel {
public:
    using ArchTag = typename EpilogueCombineScale::ArchTag;
    using ElementOutput = typename EpilogueCombineScale::ElementOutput;
    using ElementInput = typename EpilogueCombineScale::ElementInput;

    CATLASS_DEVICE
    CombineScaleKernel(XAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(XAttnKernelParams const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(XAttnKernelParams const &params) {
        return;
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(XAttnKernelParams const &params) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        EpilogueCombineScale epilogueCombineScale(resource);

        // get tiling params
        // __gm__ XATilingData *faTilingData = reinterpret_cast<__gm__ XATilingData *>(params.tiling);
        uint32_t combineFormerCoreNum = faTilingData->combineFormerCoreNum;
        uint32_t combineFormerRowNum = faTilingData->combineFormerRowNum;
        uint32_t combineTailRowNum = faTilingData->combineTailRowNum;
        uint32_t numTokens = faTilingData->numTokens;
        uint32_t qHeads = faTilingData->numHeads;
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t currentCoreRowNum = 0;
        uint32_t embed = faTilingData->embeddingSize;
        uint32_t gmOffsetO = 0;
        uint32_t gmOffsetGl = 0;
        uint32_t gmOffsetGm = 0;
        uint32_t sumMaxGmOffset = numTokens * qHeads;
        uint32_t attnOutputOffset = numTokens * qHeads * embed;
        if (coreIdx >= faTilingData->combineCoreNum) {
            return;
        }

        if (coreIdx < combineFormerCoreNum) {
            currentCoreRowNum = combineFormerRowNum;
            gmOffsetO = coreIdx * combineFormerRowNum * embed;
            gmOffsetGl = coreIdx *combineFormerRowNum;
            gmOffsetGm = gmOffsetGl;
        } else {
            currentCoreRowNum = combineTailRowNum;
            gmOffsetO = combineFormerCoreNum * combineFormerRowNum * embed 
            + (coreIdx - combineFormerCoreNum) * combineTailRowNum * embed;
            gmOffsetGl = combineFormerCoreNum * combineFormerRowNum  
            + (coreIdx - combineFormerCoreNum) * combineTailRowNum;
            gmOffsetGm = gmOffsetGl;
        }
        // shared_workspace [attnout:[attnOut * 4Bytes], 
        // gm:[attnOutputOffset * 4Bytes], 
        // gl:[attnOutputOffset * 4Bytes]]
        AscendC::GlobalTensor<ElementInput> gSharedGm;
        gSharedGm.SetGlobalBuffer((__gm__ ElementInput *)params.shared_workspace + attnOutputOffset);
        AscendC::GlobalTensor<ElementInput> gSharedGl;
        gSharedGl.SetGlobalBuffer((__gm__ ElementInput *)params.shared_workspace + attnOutputOffset + sumMaxGmOffset); // TODO
        AscendC::GlobalTensor<ElementInput> gUnsharedGm;
        gUnsharedGm.SetGlobalBuffer((__gm__ ElementInput *)params.unshared_workspace + attnOutputOffset);
        AscendC::GlobalTensor<ElementInput> gUnsharedGl;
        gUnsharedGl.SetGlobalBuffer((__gm__ ElementInput *)params.unshared_workspace + attnOutputOffset + sumMaxGmOffset);
        AscendC::GlobalTensor<ElementInput> gSharedOut;
        gSharedOut.SetGlobalBuffer((__gm__ ElementInput *)params.shared_workspace);
        AscendC::GlobalTensor<ElementInput> gUnsharedOut;
        gUnsharedOut.SetGlobalBuffer((__gm__ ElementInput *)params.unshared_workspace);
        AscendC::GlobalTensor<ElementOutput> gFinalOut;
        gFinalOut.SetGlobalBuffer((__gm__ ElementOutput *)params.o);

        MatrixCoord actualBlockShape(currentCoreRowNum, embed);
        //cce::printf("coreIdx:%d, gmOffsetO:%d, currentCoreRowNum:%d, gmOffsetGl:%d, gmOffsetGm:%d\n", AscendC::GetBlockIdx(), 
        //gmOffsetO, currentCoreRowNum, gmOffsetGl, gmOffsetGm);
        epilogueCombineScale(
            gSharedGm[gmOffsetGm], gUnsharedGm[gmOffsetGl], 
            gSharedGl[gmOffsetGm], gUnsharedGl[gmOffsetGl], 
            gSharedOut[gmOffsetO], gUnsharedOut[gmOffsetO], 
            gFinalOut[gmOffsetO], actualBlockShape
        );
        
        // AscendC::PipeBarrier<PIPE_ALL>();
        
        
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    }
private:
    Arch::Resource<ArchTag> resource;
    XAttentionTilingData* faTilingData;
};

#endif