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

#ifndef CROSS_ATTN_CATLASS_KERNEL_H
#define CROSS_ATTN_CATLASS_KERNEL_H

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
#include "cross_attention_common.h"
#include "kernel_operator.h"
using namespace Catlass;


template <
    class BlockMmadQK,
    class BlockMmadPV,
    class BlockMmadQKTail,
    class BlockMmadPVTail,
    class EpilogueOnlineSoftmax,
    class EpilogueRescaleO,
    typename TABLE_T,
    typename KVLEN_T>
class CrossAttnInferKernel {
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
    CrossAttnInferKernel(CrossAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(CrossAttnKernelParams const &params);

    CATLASS_DEVICE void Init() {
        batch = faTilingData->batch;
        qHeads = faTilingData->qHeads;
        kvHeads = faTilingData->kvHeads;
        blockSize = faTilingData->blockSize;
        groupSize = faTilingData->groupSize;
        headDim = faTilingData->headDim;
        qSeqlen = faTilingData->qSeqlen;
        maxNumBlocksPerBatch = faTilingData->maxNumBlocksPerBatch;
        qNBlockTile = faTilingData->qNBlockTile;
        qSBlockTile = faTilingData->qSBlockTile;
        qNBlockNumPerGroup = faTilingData->qNBlockNumPerGroup;
        qNBlockNum = faTilingData->qNBlockNum;
        qSBlockNum = faTilingData->qSBlockNum;
        perBatchTaskNum = faTilingData->perBatchTaskNum;
        usedCoreNum = faTilingData->usedCoreNum;
        scaleValue = faTilingData->scaleValue;
        totalTaskNum = faTilingData->totalTaskNum;
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(CrossAttnKernelParams const &params) {
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
        Init();
        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);

        BlockMmadQKTail blockMmadQKTail(resource);
        BlockMmadPVTail blockMmadPVTail(resource, L1_QK_SIZE);

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.kCache);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.vCache);
        AscendC::GlobalTensor<TABLE_T> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ TABLE_T *)(params.blockTable));
        AscendC::GlobalTensor<KVLEN_T> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ KVLEN_T *)params.actualKvLens);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);

        uint64_t strideQO = qHeads * headDim;
        uint64_t strideKV = kvHeads * headDim;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t maskType = 0;

        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
            uint32_t batchIdx = taskIdx / perBatchTaskNum;
            uint32_t taskIdxCurBatch = taskIdx - batchIdx * perBatchTaskNum;
            uint32_t qSBlockIdx = taskIdxCurBatch / qNBlockNum;
            uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * qNBlockNum;
            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
            uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * qNBlockTile;
            uint64_t qBOffset = batchIdx * qSeqlen * strideQO;
            uint64_t kvBOffset = 0;
            uint64_t blockBOffset = batchIdx * maxNumBlocksPerBatch;
            uint64_t gmQOffset = qBOffset + qSBlockIdx * qSBlockTile * strideQO + qHeadIdx * headDim;
            uint64_t gmKOffset = kvBOffset + kvHeadIdx * headDim;
            uint64_t gmVOffset = kvBOffset + kvHeadIdx * headDim;
            uint32_t qSBlockSize = (qSBlockIdx == (qSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * qSBlockTile)
                                                                       : qSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                       ? (groupSize - qNBlockIdxCurGroup * qNBlockTile)
                                       : qNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t rowNumRound = AlignUp(rowNum, BLOCK_SIZE);
            int64_t kvSeqlen = gActualKvseqlen.GetValue(batchIdx);
            uint32_t noSkipKvS = kvSeqlen;
            uint32_t noMaskKvS = kvSeqlen;
            uint32_t noMaskTailS = 0;

            uint32_t maskedKvS = qSBlockSize;
            uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, blockSize);
            uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, blockSize);
            uint32_t blockStackNum = 4;
            uint32_t stackSeqTile;
            uint32_t stackSeqTileRound = blockStackNum * 128;
            int32_t preLaunch = 2;
            int32_t totalStackSeqNum = (maskType != 0) ? (CeilDiv(noMaskKvS, blockStackNum * blockSize) + 1)
                                                       : CeilDiv(noMaskKvS, blockStackNum * blockSize);
            int32_t stackSeqCount = 0;

            LayoutQ layoutQTemp(rowNum, headDim);
            LayoutK layoutKTemp(strideKV, blockStackNum * blockSize);
            LayoutV layoutVTemp(blockStackNum * blockSize, strideKV);
            blockMmadQK.loadQGM(gQ[gmQOffset], layoutQTemp, rowNum, qNBlockSize, qHeads);
            for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
                if (kvSIdx < kvSLoopNumNoMask) {
                    if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                        stackSeqTile = noMaskKvS - kvSIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * blockStackNum;
                    }
                    uint32_t SWorkSpacePingPongFlag = stackSeqCount % (preLaunch + 1);
                    uint64_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                         + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;

                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, headDim};

                    blockMmadQK(
                        gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable[blockBOffset], layoutQTemp,
                        layoutKTemp, actualBlockShapeQK, kvSIdx, kvSLoopNumNoMask, blockSize, noMaskKvS,
                        strideKV
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                }
                if (kvSIdx >= preLaunch * blockStackNum) {
                    uint32_t nowkvSIdx = kvSIdx - preLaunch * blockStackNum;
                    if (nowkvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                        stackSeqTile = noMaskKvS - nowkvSIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * blockStackNum;
                    }
                    uint32_t PVWorkSpacePingPongFlag = (stackSeqCount - preLaunch) % (preLaunch + 1);
                    uint64_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                         + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                    uint64_t gmOTmpOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                            + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                    LayoutP layoutPTemp(rowNum, stackSeqTileRound);
                    GemmCoord actualBlockShapePV{rowNum, headDim, stackSeqTile};
                    blockMmadPV(
                        gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable[blockBOffset], layoutPTemp,
                        layoutVTemp, actualBlockShapePV, nowkvSIdx, kvSLoopNumNoMask, blockSize, noMaskKvS,
                        strideKV, softmaxReady
                    );
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
            uint32_t maskedStartIdx = AlignUp(kvSLoopNumNoMask, blockStackNum);
            uint32_t noMaskTailInteStackNum = (noMaskKvS / blockSize) % blockStackNum;
            noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? noMaskTailInteStackNum
                                                                   : ((noMaskTailS != 0) ? 0 : blockStackNum);
            uint32_t preLaunchStackNum = preLaunch * blockStackNum;

            // masked kvSeqlen loop
            for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum;) {
                if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
                    stackSeqTile = maskedKvS;
                    uint32_t SWorkSpacePingPongFlag = stackSeqCount % (preLaunch + 1);
                    uint64_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                         + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, headDim};
                    blockMmadQKTail(
                        gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBlockTable[blockBOffset], layoutQTemp,
                        layoutKTemp, actualBlockShapeQK, kvSIdx, kvSLoopNumTotal, blockSize, noSkipKvS,
                        strideKV, noMaskTailS, 1
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                }

                if (kvSIdx >= preLaunchStackNum) {
                    uint32_t delayedKvSIdx = kvSIdx - preLaunchStackNum;

                    if (delayedKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && (maskType != 0)) {
                        stackSeqTile = maskedKvS;
                    } else if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                        stackSeqTile = noMaskKvS - delayedKvSIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * blockStackNum;
                    }
                    uint32_t PVWorkSpacePingPongFlag = (stackSeqCount - preLaunch) % (preLaunch + 1);
                    uint64_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                         + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                    uint64_t gmOTmpOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                            + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                    LayoutP layoutPTemp(rowNum, stackSeqTileRound);
                    GemmCoord actualBlockShapePV{rowNum, headDim, stackSeqTile};

                    blockMmadPV(
                        gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset], gBlockTable[blockBOffset],
                        layoutPTemp, layoutVTemp, actualBlockShapePV, delayedKvSIdx, kvSLoopNumNoMask,
                        blockSize, noMaskKvS, strideKV, softmaxReady
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                }
                kvSIdx += blockStackNum;
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
    CATLASS_DEVICE void operator()<AscendC::AIV>(CrossAttnKernelParams const &params) {
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
        Init();
        // Get tiling parameters
        // Get the memory offset address of the input on Global Memory
        AscendC::GlobalTensor<ElementMask> gMask;
        AscendC::GlobalTensor<KVLEN_T> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ KVLEN_T *)params.actualKvLens);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);

        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
        EpilogueRescaleO epilogueRescaleO(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t maskType = 0;
        // Go through each task.
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
            uint32_t batchIdx = taskIdx / perBatchTaskNum;
            uint32_t taskIdxCurBatch = taskIdx - batchIdx * perBatchTaskNum;
            uint32_t qSBlockIdx = taskIdxCurBatch / qNBlockNum;
            uint32_t qNBlockIdx = taskIdxCurBatch % qNBlockNum;
            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
            uint32_t oBatchOffset = batchIdx * qSeqlen * qHeads * headDim;
            uint32_t oSOffset = qSBlockIdx * qSBlockTile * qHeads * headDim;
            uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qStartNIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * qNBlockTile;
            uint32_t oNOffset = qStartNIdx * headDim;
            uint32_t gmOffsetO = oBatchOffset + oSOffset + oNOffset;
            uint32_t qSBlockSize = (qSBlockIdx == (qSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * qSBlockTile)
                                                                       : qSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                       ? (groupSize - qNBlockIdxCurGroup * qNBlockTile)
                                       : qNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t rowNumRound = AlignUp(rowNum, BLOCK_SIZE);
            int64_t kvSeqlen = gActualKvseqlen.GetValue(batchIdx);
            uint32_t noSkipKvS = kvSeqlen;
            uint32_t noMaskKvS = kvSeqlen;
            uint32_t noMaskTailS = 0;
            uint32_t maskedKvS = qSBlockSize;
            uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, blockSize);
            uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, blockSize);
            uint32_t blockStackNum = 4;
            uint32_t stackSeqTilePad = blockStackNum * blockSize;
            uint32_t stackSeqTile;
            int32_t preLaunch = 2;
            int32_t totalStackSeqNum = (maskType != 0) ? (CeilDiv(noMaskKvS, blockStackNum * blockSize) + 1)
                                                       : CeilDiv(noMaskKvS, blockStackNum * blockSize);
            int32_t stackSeqCount = 0;

            // no mask kvSeqlen loop
            for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {

                if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                    stackSeqTile = noMaskKvS - kvSIdx * blockSize;
                } else {
                    stackSeqTile = blockSize * blockStackNum;
                }
                uint32_t stackSeqTileRound = RoundUp(stackSeqTile, 8);
                LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
                LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, headDim};
                uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + // cube core offset
                                     curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;            // single cube core db offset
                // vec core offset will be processed within epilogue block
                uint32_t gmOffsetP = gmOffsetS;
                Arch::CrossCoreWaitFlag(qkReady);
                // online softmax

                epilogueOnlineSoftmax(
                    gP[gmOffsetP], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK, (stackSeqCount == 0),
                    qSBlockSize, qNBlockSize, curStackTileMod
                );
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);

                if (kvSIdx >= preLaunch * blockStackNum) {
                    uint32_t delayedKvSIdx = kvSIdx - preLaunch * blockStackNum;
                    if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                        stackSeqTile = noMaskKvS - kvSIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * blockStackNum;
                    }
                    LayoutO layoutO(qSeqlen, headDim * qHeads);
                    LayoutOTmp layoutOTmp(rowNum, headDim, headDim);
                    GemmCoord actualBlockShapePV{rowNum, headDim, stackSeqTile};
                    uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                    uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                            + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    Arch::CrossCoreWaitFlag(pvReady);
                    // rescale O
                    epilogueRescaleO(
                        gO[gmOffsetO], gOTmp[gmOffsetOTmp], layoutO, layoutOTmp, actualBlockShapePV, qSBlockSize,
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
            uint32_t noMaskTailInteStackNum = (noMaskKvS / blockSize) % blockStackNum;
            noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? noMaskTailInteStackNum
                                                                   : ((noMaskTailS != 0) ? 0 : blockStackNum);
            uint32_t preLaunchStackNum = (maskType != 0) ? ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum)
                                                         : (preLaunch * blockStackNum);
            // masked kvSeqlen loop
            for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum;) {
                if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
                    stackSeqTile = maskedKvS;
                    uint32_t stackSeqTileRound = RoundUp(stackSeqTile, BLOCK_SIZE);
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
                    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                    LayoutMask layOutMask(1024, 1024, 1024);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, headDim};
                    uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                    uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + // cube core offset
                                         curStackTileMod * WORKSPACE_BLOCK_SIZE_DB; // single cube core db offset
                    // vec core offset will be processed within epilogue block
                    uint32_t gmOffsetP = gmOffsetS;
                    // online softmax
                    epilogueOnlineSoftmax(
                        gP[gmOffsetP], gS[gmOffsetS], gMask, layOutP, layOutS, layOutMask, actualBlockShapeQK,
                        (stackSeqCount == 0), qSBlockSize, qNBlockSize, curStackTileMod, qkReady
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                }
                if (kvSIdx >= preLaunchStackNum) {
                    uint32_t delayedKvSIdx = kvSIdx - preLaunchStackNum;
                    if (delayedKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && (maskType != 0)) {
                        stackSeqTile = maskedKvS;
                    } else if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                        stackSeqTile = noMaskKvS - delayedKvSIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * blockStackNum;
                    }
                    LayoutO layoutO(qSBlockSize, headDim * qHeads);
                    LayoutOTmp layoutOTmp(rowNum, headDim, headDim);
                    GemmCoord actualBlockShapePV{rowNum, headDim, stackSeqTile};
                    uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                    uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1)
                                            + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    Arch::CrossCoreWaitFlag(pvReady);
                    // rescale O
                    epilogueRescaleO(
                        gO[gmOffsetO], gOTmp[gmOffsetOTmp], layoutO, layoutOTmp, actualBlockShapePV, qSBlockSize,
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
    Arch::CrossCoreFlag qkReady{SYNC_QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SYNC_SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{SYNC_PV_READY_ID};
    CrossAttentionTilingData* faTilingData;
    uint32_t batch;
    uint32_t qHeads;
    uint32_t kvHeads;
    uint32_t blockSize;
    uint32_t groupSize;
    uint32_t headDim;
    uint32_t qSeqlen;
    uint32_t maxNumBlocksPerBatch;
    uint32_t qNBlockTile;
    uint32_t qSBlockTile;
    uint32_t qNBlockNumPerGroup;
    uint32_t qNBlockNum;
    uint32_t qSBlockNum;
    uint32_t perBatchTaskNum;
    uint32_t totalTaskNum;
    uint32_t usedCoreNum;
    uint32_t wsBlockSizeDb;
    float scaleValue;
};

#endif