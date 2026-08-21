/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_KERNEL_H
#define XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_KERNEL_H

// Device(kernel) side: the catlass forwarding headers (copy_gm_to_l1.hpp / copy_l1_to_l0a.hpp
// etc.) dispatch to the ascend950 specialization ONLY when CATLASS_ARCH == 3510. The kernel
// compile command does not inject -DCATLASS_ARCH (that is only injected on the host side by
// op_host/CMakeLists.txt). On A5 the device arch macro __NPU_ARCH__ == 3510, so derive
// CATLASS_ARCH from it here, BEFORE including any catlass header, so the Ascend950 tile-copy
// templates (CopyGmToL1 / CopyL1ToL0A / CopyL1ToL0B) are actually visible.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && !defined(CATLASS_ARCH)
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "kernel_operator.h"

#include "a5_x_flash_attention_infer_kernel_utils.h"
#include "a5_x_flash_attention_infer_tiling_data_def.h"

using namespace Catlass;
using namespace tla;
using namespace AscendC;

template <
    class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO, bool PAGED_CACHE_FLAG>
class FAInferKernel {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutTagQ = typename BlockMmadQK::LayoutTagA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutTagK = typename BlockMmadQK::LayoutTagB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutTagS = typename BlockMmadQK::LayoutTagC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutTagP = typename BlockMmadPV::LayoutTagA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutTagV = typename BlockMmadPV::LayoutTagB;

    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
    using LayoutTagMask = typename EpilogueOnlineSoftmax::LayoutTagMask;

    using ElementOTmp = typename EpilogueRescaleO::ElementOTmp;
    using LayoutTagOTmp = typename EpilogueRescaleO::LayoutTagOTmp;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using LayoutTagO = typename EpilogueRescaleO::LayoutTagO;

    static constexpr uint32_t qSeqlenTemplateType = tla::get<0>(L1TileShape{});
    static constexpr uint32_t kvSeqlenTemplateType = tla::get<1>(L1TileShape{});
    static constexpr uint32_t embedTemplateType = tla::get<2>(L1TileShape{});

    static constexpr uint32_t MM2_LEFT_SIZE = qSeqlenTemplateType * kvSeqlenTemplateType * sizeof(ElementP);

    // Methods
    CATLASS_DEVICE
    FAInferKernel()
    {}

    CATLASS_DEVICE void Init(FAIKernelParams const& params)
    {
        // 获取当前aic idx 和sub blockidx
        if ASCEND_IS_AIC {
            this->blockIdx = AscendC::GetBlockIdx();
        } else {
            this->blockIdx = AscendC::GetBlockIdx() >> 1;
        }

        this->subBlockIdx = AscendC::GetSubBlockIdx();
        constInfo.subBlockIdx = this->subBlockIdx;

        // 调用Tiling接口
        auto faTilingStruct = (__gm__ FATilingData*)params.tiling;
        auto& inputParamsRegbase = faTilingStruct->inputParamsRegbase;
        this->constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
        this->constInfo.batch = inputParamsRegbase.batch;
        this->constInfo.qHeads = inputParamsRegbase.qHeads;
        this->constInfo.kvHeads = inputParamsRegbase.kvHeads;
        this->constInfo.groupSize = inputParamsRegbase.groupSize;
        this->constInfo.qSeqlen = inputParamsRegbase.qSeqlen;
        this->constInfo.kvSeqlen = inputParamsRegbase.kvSeqlen;
        this->constInfo.embed = inputParamsRegbase.embed;
        this->constInfo.attenMaskQSeqlen = inputParamsRegbase.attenMaskQSeqlen;
        this->constInfo.attenMaskKvSeqlen = inputParamsRegbase.attenMaskKvSeqlen;

        this->constInfo.headNumRatio = inputParamsRegbase.headNumRatio;
        this->constInfo.actualSeqLengthsSize = inputParamsRegbase.actualSeqLengthsSize;
        this->constInfo.actualSeqLengthsKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
        this->constInfo.isActualSeqLengthsNull = inputParamsRegbase.isActualSeqLengthsNull;
        this->constInfo.isActualSeqLengthsKVNull= inputParamsRegbase.isActualSeqLengthsKVNull;

        // pageAttention
        if constexpr (PAGED_CACHE_FLAG) {
            this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
            this->constInfo.blockSize = inputParamsRegbase.blockSize;
            this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
        }

        auto& multiCoreParamsRegbase = faTilingStruct->multiCoreParamsRegbase;
        this->constInfo.qSeqlenOuterSize = multiCoreParamsRegbase.qSeqlenOuterSize;
        this->constInfo.coreNum = multiCoreParamsRegbase.coreNum;
        /* 多核切分偏移计算 */
        this->constInfo.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[this->blockIdx];
        this->constInfo.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[this->blockIdx + 1];
        this->constInfo.bnAxisStartIdx = multiCoreParamsRegbase.bnAxisStartIdx[this->blockIdx];
        this->constInfo.bnAxisEndIdx = multiCoreParamsRegbase.bnAxisStartIdx[this->blockIdx + 1];

        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_RES_INTRA_EVENT[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_RES_INTRA_EVENT[1]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM1_RES_INTRA_EVENT[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM1_RES_INTRA_EVENT[1]);

        this->constInfo.qSeqlenBase = qSeqlenTemplateType;
        this->constInfo.kvSeqlenBase = kvSeqlenTemplateType;

        for (int i = 0; i < NUM2; i++) {
            bmm1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
            ubBufAddrStart += MM1_RESULT_SIZE;
            bmm2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ubBufAddrStart);
            ubBufAddrStart += MM2_RESULT_SIZE;
        }

        if ASCEND_IS_AIV {
            for (int i = 0; i < KERNEL_TASK_NUM; i++) {
                sumUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
                expUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
                maxUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
            }
        }

        // 初始化全局L1
        for (int i = 0; i < KERNEL_TASK_NUM; i++) {
            mm2AL1TensorList[i] = resource.l1Buf.template GetBufferByByte<ElementP>(l1BufAddrStart + i * MM2_LEFT_SIZE);
        }
        l1BufAddrStart += KERNEL_TASK_NUM * MM2_LEFT_SIZE;
    }

    CATLASS_DEVICE void operator()(FAIKernelParams const& params)
    {
        // Init
        Init(params); // 初始化ConstInfo，初始化L1/UB
        uint32_t l0CBufAddrStart = 0;
        BlockMmadQK blockMmadMmadQK(resource, l1BufAddrStart, l0CBufAddrStart);
        BlockMmadPV blockMmadMmadPV(resource, l1BufAddrStart, l0CBufAddrStart);
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, constInfo.scaleValue, ubBufAddrStart);
        EpilogueRescaleO epilogueRescaleO(resource, ubBufAddrStart);

        // Get blockIdx
        int32_t blockNum = this->constInfo.coreNum;
        if (this->blockIdx >= blockNum) {
            return;
        }

        int64_t batch = this->constInfo.batch;
        int64_t qHeads = this->constInfo.qHeads;
        int64_t qSeqlen = this->constInfo.qSeqlen;
        int64_t kvHeads = this->constInfo.kvHeads;
        int64_t kvSeqlen = this->constInfo.kvSeqlen;
        int64_t groupSize = this->constInfo.groupSize;
        int64_t embed = this->constInfo.embed;
        int64_t blockSize = this->constInfo.blockSize;
        // Read runtime actual kv length from the actualSeqLengthsKV tensor so that the
        // causal mask can shield the padding tail when actual_kv_len < paged-cache capacity.
        // The tiling-provided constInfo.kvSeqlen is the paged-cache capacity (maxBlockNumPerBatch
        // * blockSize), NOT the real kv length, so relying on it makes decode attend to padded kv
        // (aligns with the A3 path which reads gActualKvseqlen.GetValue(BIdx)).
        int64_t actualKvSeqlen = kvSeqlen;
        if (params.actualKvSeqlen != nullptr) {
            AscendC::GlobalTensor<int32_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualKvSeqlen);
            int64_t rtKv = static_cast<int64_t>(gActualKvseqlen.GetValue(0));
            if (rtKv > 0) {
                actualKvSeqlen = rtKv;
            }
        }
        int64_t kvSeqlenMask = actualKvSeqlen;
        // causal mask uses a shared [MASK_DIM, MASK_DIM] triu template (row = query
        // absolute logical position, col = kv absolute position), aligned with the A3
        // path which hardcodes LayoutMask(2048, 2048). diffS shifts single-token decode
        // query to the causal bottom so it can attend to all valid kv. diffS MUST use the
        // actual kv length (not the paged-cache capacity), otherwise the causal mask fails
        // to shield the padded tail (e.g. actual_kv=64 within a 128-slot block).
        constexpr int64_t MASK_DIM = 2048;
        int64_t diffS = actualKvSeqlen - qSeqlen;
        if constexpr (PAGED_CACHE_FLAG) {
            kvSeqlen = RoundUp(kvSeqlen, blockSize);
        }
        // Init Tensor
        AscendC::GlobalTensor<ElementQ> gmQ;
        gmQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        // Create TLA layouts for kernel usage
        auto layoutQ = MakeLayout<ElementQ, LayoutTagQ>(batch * qSeqlen, kvHeads * groupSize * embed);
        auto tensorQWithLayout = tla::MakeTensor(gmQ, layoutQ, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementK> gmK;
        gmK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        auto layoutK = MakeLayout<ElementK, LayoutTagK>(kvHeads * embed, batch * kvSeqlen);
        auto tensorKWithLayout = tla::MakeTensor(gmK, layoutK, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementV> gmV;
        gmV.SetGlobalBuffer((__gm__ ElementV*)params.v);
        auto layoutV = MakeLayout<ElementV, LayoutTagV>(batch * kvSeqlen, kvHeads * embed);
        auto tensorVWithLayout = tla::MakeTensor(gmV, layoutV, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementMask> gmMask;
        gmMask.SetGlobalBuffer((__gm__ ElementMask*)params.mask);
        auto layoutMask = MakeLayout<ElementMask, LayoutTagMask>(MASK_DIM, MASK_DIM);
        auto tensorMaskWithLayout = tla::MakeTensor(gmMask, layoutMask, Arch::PositionGM{});

        // BlockTable
        AscendC::GlobalTensor<int32_t> tensorTable;
        tensorTable.SetGlobalBuffer((__gm__ int32_t*)params.blockTables);

        AscendC::GlobalTensor<ElementO> attentionOutGm;
        AscendC::GlobalTensor<ElementP> workspaceGm;
        attentionOutGm.SetGlobalBuffer((__gm__ ElementO*)params.o);
        auto layoutO = MakeLayout<ElementO, LayoutTagO>(batch * qSeqlen, kvHeads * groupSize * embed);
        auto attentionOutGmWithLayout = tla::MakeTensor(attentionOutGm, layoutO, Arch::PositionGM{});

        uint32_t maxBlockNumPerBatch = this->constInfo.blockTableDim2;

        // Main process loop

        // 确定核内切分起点
        int64_t qSeqAxisStartIdx;
        uint32_t bnAxisStartIdx;
        uint32_t bnAxisEndIdx;
        int64_t kvSeqLoopLimit;
        int64_t nextQSeqAxisIdx = this->constInfo.multiCoreInnerLimit;
        bnAxisStartIdx = this->constInfo.bnAxisStartIdx;
        qSeqAxisStartIdx = this->constInfo.multiCoreInnerOffset;
        if (likely((this->constInfo.coreNum - 1) > this->blockIdx)) {
            bnAxisEndIdx = this->constInfo.bnAxisEndIdx;
            if (nextQSeqAxisIdx != 0) {
                bnAxisEndIdx++;
            }
        } else {
            bnAxisEndIdx = this->constInfo.batch * this->constInfo.kvHeads * this->constInfo.headNumRatio;
        }

        // 初始化CV流水状态信息
        int64_t taskId = 0;
        bool notLast = true;
        bool isLastBmm1 = false;
        int64_t multiCoreInnerIdx = 1;
        for (uint32_t bnIdx = bnAxisStartIdx; bnIdx < bnAxisEndIdx; ++bnIdx) {
            bool lastBN = (bnIdx == bnAxisEndIdx - 1);
            runParam.batchOuterIdx = bnIdx / (this->constInfo.kvHeads * this->constInfo.headNumRatio);
            runParam.kvHeadsOuterIdx =
                (bnIdx / this->constInfo.headNumRatio) % this->constInfo.kvHeads; // 切核逻辑，先N2G再B
            ComputeParamBatch(runParam, this->constInfo, this->attenMaskInfo);    // 计算runParam中参数值
            ComputeQseqLoopInfo<qSeqlenTemplateType>(runParam, this->constInfo, lastBN, nextQSeqAxisIdx);
            int64_t tempQSeqAxisEnd = lastBN ? (runParam.qSeqLoopTimes + 3) : runParam.qSeqLoopTimes;
            for (int64_t qSeqAxisIndex = qSeqAxisStartIdx; qSeqAxisIndex < tempQSeqAxisEnd; ++qSeqAxisIndex) {
                bool notLastThreeLoop = true;
                bool notLastTwoLoop = true;
                if (lastBN) {
                    int32_t extraQSeqAxis = qSeqAxisIndex - runParam.qSeqLoopTimes;
                    switch (extraQSeqAxis) {
                        case -1:
                            isLastBmm1 = true;
                            break;
                        case 0:
                            notLastThreeLoop = false;
                            break;
                        case 1:
                            notLastThreeLoop = false;
                            notLastTwoLoop = false;
                            break;
                        case 2:
                            notLast = false;
                            notLastThreeLoop = false;
                            notLastTwoLoop = false;
                            break;
                        default:
                            break;
                    }
                }
                if (notLastThreeLoop) {
                    runParam.groupIdx = bnIdx % this->constInfo.headNumRatio;
                    runParam.qSeqOuterAxisIdx = qSeqAxisIndex % this->constInfo.qSeqlenOuterSize;
                    ComputeParamQSeq<qSeqlenTemplateType>(runParam, this->constInfo, qSeqAxisIndex);
                    ComputeKvSeqLoopInfo<kvSeqlenTemplateType>(runParam, this->constInfo);
                    kvSeqLoopLimit = runParam.kvSeqLoopEndIdx - 1;
                } else {
                    runParam.kvSeqLoopStartIdx = 0;
                    kvSeqLoopLimit = 0;
                }
                for (int64_t kvSeqLoopCount = runParam.kvSeqLoopStartIdx; kvSeqLoopCount <= kvSeqLoopLimit;
                     ++kvSeqLoopCount) {
                    if (notLastThreeLoop) {
                        RunInfo& runInfo1 = runInfo[taskId & 3];
                        this->SetRunInfo(runInfo1, runParam, taskId, kvSeqLoopCount, kvSeqLoopLimit, multiCoreInnerIdx);
                      if ASCEND_IS_AIC {
                            CalcKvSeqCoord(runInfo1, this->constInfo);
                            CalcQSeqCoord(runInfo1, this->constInfo);
                            auto actualShape =
                                tla::MakeShape(runInfo1.qSeqRealSize, runInfo1.kvSeqRealSize, this->constInfo.embed);
                            auto layoutMM1O =
                              tla::MakeLayout<ElementS, LayoutTagS>(runInfo1.qSeqRealSize, kvSeqlenTemplateType);
                            auto tensorMM1OWithLayout =
                                tla::MakeTensor(bmm1TensorList[runInfo1.taskIdMod2], layoutMM1O, Arch::PositionUB{});

                            auto tensorInQ = GetTile(
                                tensorQWithLayout,
                                tla::MakeCoord(
                                    runInfo1.batchOuterIdx * qSeqlen + coordInfo[runInfo1.taskIdMod3].qSeqCoord,
                                    runInfo1.kvHeadsOuterIdx * groupSize * embed + runInfo1.groupIdx * embed),
                                tla::MakeShape(runInfo1.qSeqRealSize, this->constInfo.embed));
                            auto kCoord = runInfo1.kvHeadsOuterIdx * embed;
                            auto nCoord = 0;
                            auto nShape = runInfo1.kvSeqRealSize;
                            if constexpr (PAGED_CACHE_FLAG) {
                                uint32_t maxBlockNumPerBatch = this->constInfo.blockTableDim2;
                                uint64_t blockTableBaseOffset =
                                    runInfo1.batchOuterIdx * maxBlockNumPerBatch; // 块表的基偏移量
                                uint32_t curKvSeqAxisIdx = runInfo1.kvSeqLoopCount * this->constInfo.kvSeqlenBase;
                                uint64_t blockIdOffset =
                                    curKvSeqAxisIdx / this->constInfo.blockSize; // 获取block table上的索引
                                runInfo1.blockTableOffset = blockTableBaseOffset + blockIdOffset;
                                nShape = batch * kvSeqlen;
                            } else {
                                nCoord = coordInfo[runInfo1.taskIdMod3].curBIdx * kvSeqlen +
                                         coordInfo[runInfo1.taskIdMod3].kvSeqCoord;
                            }
                            auto tensorInK = GetTile(
                                tensorKWithLayout, tla::MakeCoord(kCoord, nCoord),
                                tla::MakeShape(this->constInfo.embed, nShape));

                            auto tensorInTable = tensorTable[runInfo1.blockTableOffset];

                            bool isFirstLoop = (runInfo1.kvSeqLoopCount == runInfo1.kvSeqLoopStartIdx) ? true : false;
                            bool isLastUpdate = (runInfo1.kvSeqLoopCount == runInfo1.kvSeqLoopLimit) ? true : false;

                            blockMmadMmadQK(
                                tensorInQ, tensorInK, tensorMM1OWithLayout, tensorInTable, actualShape,
                                runInfo1.taskIdMod2, this->constInfo.blockSize, isFirstLoop, isLastUpdate);

                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(
                                SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(
                                16 +
                                SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
                        }
                    }

                    if (taskId > 0 && notLastTwoLoop) {
                        if ASCEND_IS_AIV {
                            auto& runInfo3 = runInfo[(taskId + 3) & 3];
                            auto& taskIdMod2 = runInfo3.taskIdMod2;
                            auto& taskIdMod3 = runInfo3.taskIdMod3;
                            auto& multiCoreIdxMod3 = runInfo3.multiCoreIdxMod3;
                            bool isFirstLoop = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopStartIdx) ? true : false;
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(
                                SYNC_C1_V1_FLAG[taskIdMod2]); // 等待bmm1完成/等待SYNC_C1_V1_FLAG置位
                            auto bmm1Layout = tla::MakeLayout<ElementS, LayoutTagS>(
                                runInfo3.halfQSeqRealSize, runInfo3.kvSeqRealSize);
                            auto bmm1Tensor =
                                tla::MakeTensor(bmm1TensorList[taskIdMod2], bmm1Layout, Arch::PositionUB{});
                            auto l1Vf1OutLayout =
                                tla::MakeLayout<ElementP, LayoutTagP>(qSeqlenTemplateType, kvSeqlenTemplateType);
                            auto l1Vf1OutTensor =
                                tla::MakeTensor(mm2AL1TensorList[taskIdMod3], l1Vf1OutLayout, Arch::PositionL1{});

                            auto l1Vf1OutTile = GetTile(
                                l1Vf1OutTensor,
                                tla::MakeCoord(constInfo.subBlockIdx * runInfo3.firstHalfQSeqRealSize, 0),
                                tla::MakeShape(runInfo3.halfQSeqRealSize, kvSeqlenTemplateType));

                            // mask is a shared [MASK_DIM, MASK_DIM] triu template: row = query
                            // absolute logical position within the sequence (NOT batch-offset,
                            // the template is shared across batches like the A3 path). diffS aligns
                            // the query row to the causal bottom so decode (qSeqlen=1) attends to all kv.
                            int64_t qSeqOffset = runInfo3.qSeqOuterAxisIdx * qSeqlenTemplateType +
                                             runInfo3.firstHalfQSeqRealSize * constInfo.subBlockIdx;
                            int64_t kvSeqOffset = runInfo3.kvSeqLoopCount * kvSeqlenTemplateType;

                            auto gmMaskTile = GetTile(
                                tensorMaskWithLayout, tla::MakeCoord(diffS + qSeqOffset, kvSeqOffset),
                                tla::MakeShape(runInfo3.halfQSeqRealSize, runInfo3.kvSeqRealSize));

                            epilogueOnlineSoftmax(
                                l1Vf1OutTile, sumUb[multiCoreIdxMod3], maxUb[multiCoreIdxMod3], expUb[taskIdMod3],
                                bmm1Tensor, gmMaskTile, !isFirstLoop, taskIdMod2, taskIdMod3,
                             MM1_RES_INTRA_EVENT[taskIdMod2], SYNC_V1_C2_FLAG[taskIdMod3]);
                        }
                    }
                    if (taskId > 1 && notLast) {
                        if ASCEND_IS_AIC {
                            RunInfo& runInfo2 = runInfo[(taskId + 2) & 3];
                            auto& taskIdMod2 = runInfo2.taskIdMod2;
                            auto& taskIdMod3 = runInfo2.taskIdMod3;
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V1_C2_FLAG[taskIdMod3]);
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V1_C2_FLAG[taskIdMod3]);

                            auto layoutMM2O =
                                tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(runInfo2.qSeqRealSize, embedTemplateType);
                            auto mm2OutTensor =
                                tla::MakeTensor(bmm2TensorList[taskIdMod2], layoutMM2O, Arch::PositionUB{});

                            auto layoutVec1O =
                                tla::MakeLayout<ElementP, LayoutTagP>(qSeqlenTemplateType, kvSeqlenTemplateType);
                            auto mm2AL1Tensor =
                                tla::MakeTensor(mm2AL1TensorList[taskIdMod3], layoutVec1O, Arch::PositionL1{});
                            auto kCoord = 0;
                            auto nCoord = runInfo2.kvHeadsOuterIdx * embed;
                            auto kShape = runInfo2.kvSeqRealSize;
                            if constexpr (PAGED_CACHE_FLAG) {
                                kShape = batch * kvSeqlen;
                            } else {
                                kCoord = coordInfo[runInfo2.taskIdMod3].curBIdx * kvSeqlen +
                                         coordInfo[runInfo2.taskIdMod3].kvSeqCoord;
                            }
                            auto tensorInV = GetTile(
                                tensorVWithLayout, tla::MakeCoord(kCoord, nCoord),
                                tla::MakeShape(kShape, this->constInfo.embed));
                            auto actualShape =
                                tla::MakeShape(runInfo2.qSeqRealSize, embedTemplateType, runInfo2.kvSeqRealSize);
                            auto tensorInTableV = tensorTable[runInfo2.blockTableOffset];
                            blockMmadMmadPV(
                                mm2AL1Tensor, tensorInV, mm2OutTensor, tensorInTableV, actualShape, taskIdMod2,
                                this->constInfo.blockSize);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(
                                SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C2_V2_FLAG
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(
                                16 +
                                SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C2_V2_FLAG
                        }
                    }
                    if (taskId > 2) {
                        if ASCEND_IS_AIV {
                            RunInfo& runInfo3 = runInfo[(taskId + 1) & 3];
                            auto& taskIdMod2 = runInfo3.taskIdMod2;
                            auto& taskIdMod3 = runInfo3.taskIdMod3;
                            auto& multiCoreIdxMod3 = runInfo3.multiCoreIdxMod3;

                            bool isFirstLoop = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopStartIdx) ? true : false;
                            bool isLastUpdate = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopLimit) ? true : false;
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(
                                SYNC_C2_V2_FLAG[taskIdMod2]); // 等待bmm2完成/等待SYNC_C2_V2_FLAG置位
                            auto bmm2Layout =
                                MakeLayout<ElementOTmp, LayoutTagOTmp>(runInfo3.halfQSeqRealSize, embedTemplateType);
                            auto bmm2Tensor =
                                tla::MakeTensor(bmm2TensorList[taskIdMod2], bmm2Layout, Arch::PositionUB{});
                            int64_t bOffset = runInfo3.batchOuterIdx * qSeqlen;
                            int64_t qSeqOffset = runInfo3.qSeqOuterAxisIdx * qSeqlenTemplateType +
                                                 runInfo3.firstHalfQSeqRealSize * constInfo.subBlockIdx;
                            int64_t kvHeadsOffset = runInfo3.kvHeadsOuterIdx * groupSize * embed;
                            int64_t embedOffset = runInfo3.groupIdx * embed;

                            auto attenOutGmTile = GetTile(
                                attentionOutGmWithLayout,
                                tla::MakeCoord(
                                    bOffset + qSeqOffset,
                                    kvHeadsOffset + embedOffset), // batch * qSeqlen, kvHeads* groupSize * embed
                                tla::MakeShape(runInfo3.halfQSeqRealSize, embedTemplateType));
                            epilogueRescaleO(
                                attenOutGmTile, expUb[taskIdMod3], sumUb[multiCoreIdxMod3], bmm2Tensor, isFirstLoop,
                                isLastUpdate, MM2_RES_INTRA_EVENT[taskIdMod2]);
                        }
                    }
                    ++taskId;
                }
                ++multiCoreInnerIdx;
            }
            qSeqAxisStartIdx = 0;
        }
    }

private:
    static constexpr uint32_t embedTemplateAlign64 = Align64Func((uint16_t)embedTemplateType);
    static constexpr uint32_t MM1_RESULT_SIZE =
        qSeqlenTemplateType / CV_RATIO * kvSeqlenTemplateType * sizeof(ElementS);
    static constexpr uint32_t MM2_RESULT_SIZE =
        qSeqlenTemplateType / CV_RATIO * embedTemplateAlign64 * sizeof(ElementOTmp);
    static constexpr uint32_t SHARE_UB_SIZE = CeilDiv(qSeqlenTemplateType, NUM2) * sizeof(ElementS);

    AscendC::LocalTensor<ElementS> bmm1TensorList[NUM2];
    AscendC::LocalTensor<ElementP> mm2AL1TensorList[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementOTmp> bmm2TensorList[NUM2];
    AscendC::LocalTensor<ElementS> expUb[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementS> sumUb[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementS> maxUb[KERNEL_TASK_NUM];
    ConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    uint32_t blockIdx;
    uint32_t subBlockIdx;

    RunInfo runInfo[4];   // 最内层循环kvSeq参数
    RunParamStr runParam; // 外层参数
    uint32_t l1BufAddrStart = 0;
    uint32_t ubBufAddrStart = 0;

    Arch::Resource<ArchTag> resource;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];

    // =========================================== private functions ===========================================
    CATLASS_DEVICE void SetRunInfo(
        RunInfo& runInfo, RunParamStr& runParam, int64_t taskId, int64_t kvSeqLoopCount, int64_t kvSeqLoopLimit,
        int64_t multiCoreInnerIdx)
    {
        runInfo.kvSeqAxisStartIdx = runParam.kvSeqAxisLineStartIdx;
        runInfo.kvSeqLoopStartIdx = runParam.kvSeqLoopStartIdx;
        runInfo.kvSeqAxisEndIdx = runParam.kvSeqAxisLineEndIdx;
        runInfo.kvSeqLoopCount = kvSeqLoopCount;
        if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
            runInfo.qSeqOuterAxisIdx = runParam.qSeqOuterAxisIdx;
            runInfo.batchOuterIdx = runParam.batchOuterIdx;
            runInfo.kvHeadsOuterIdx = runParam.kvHeadsOuterIdx;
            runInfo.groupIdx = runParam.groupIdx;
            runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
            runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
            runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
        }

        runInfo.taskId = taskId;
        runInfo.taskIdMod2 = taskId & 1;
        runInfo.taskIdMod3 = taskId % 3;
        runInfo.kvSeqLoopLimit = kvSeqLoopLimit;

        runInfo.actualQSeqSize = runParam.actualQSeqSize;
        runInfo.actualKvSeqSize = runParam.actualKvSeqSize;
        this->ComputeBmm1Tail(runInfo, runParam);
        runInfo.batchOuterIdx = runParam.batchOuterIdx;
    }

    CATLASS_DEVICE void ComputeBmm1Tail(RunInfo& runInfo, RunParamStr& runParam)
    {
        // ------------------------qSeq Base Related---------------------------
        runInfo.qSeqRealSize = runParam.qSeqRealSize;
        runInfo.halfQSeqRealSize = runParam.halfQSeqRealSize;
        runInfo.firstHalfQSeqRealSize = runParam.firstHalfQSeqRealSize;

        // ------------------------kvSeq Base Related----------------------------
        runInfo.kvSeqRealSize = this->constInfo.kvSeqlenBase;
        if ((runInfo.kvSeqLoopCount + 1) * runInfo.kvSeqRealSize > runInfo.kvSeqAxisEndIdx) {
            runInfo.kvSeqRealSize = runInfo.kvSeqAxisEndIdx - runInfo.kvSeqLoopCount * runInfo.kvSeqRealSize;
        }
    }

    CATLASS_DEVICE void CalcQSeqCoord(RunInfo& runInfo, ConstInfo& constInfo)
    {
        // 计算qSeq方向偏移
        coordInfo[runInfo.taskIdMod3].qSeqCoord = runInfo.qSeqOuterAxisIdx * this->constInfo.qSeqlenBase;
    }

    CATLASS_DEVICE void CalcKvSeqCoord(RunInfo& runInfo, ConstInfo& constInfo)
    {
        coordInfo[runInfo.taskIdMod3].kvSeqCoord =
            runInfo.kvSeqAxisStartIdx +
            (runInfo.kvSeqLoopCount - runInfo.kvSeqLoopStartIdx) * this->constInfo.kvSeqlenBase;
        coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.batchOuterIdx;
    }
};

template <class Dtype, bool enableMaskFlag = false, bool enablePaFlag = false>
CATLASS_GLOBAL void FAInferTla(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR actualQSeqlen,
    GM_ADDR actualKvSeqlen, GM_ADDR tiling)
{
    using ArchTag = Arch::Ascend950;
    using ElementQ = Dtype;
    using LayoutTagQ = layout::RowMajor;
    using ElementK = Dtype;
    using LayoutTagK = layout::ColumnMajor;
    using ElementV = Dtype;
    using LayoutTagV = layout::RowMajor;
    using ElementS = float;
    using LayoutTagS = layout::RowMajor;
    using ElementP = Dtype;
    using LayoutTagP = layout::zN;
    using ElementO = Dtype;
    using LayoutTagO = layout::RowMajor;
    using ElementMask = uint8_t;
    using LayoutTagMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutTagOTmp = layout::RowMajor;
    // L1TileShape::K must be embdding
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
    // GEMM Block模块，实现Flash Attention Infer的Q * K^T
    using DispatchPolicyQK = Gemm::MmadFAIQK<ArchTag, enablePaFlag>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementS, LayoutTagS, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS, void, TileCopyQK, TileMmadQK>;

    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicySoftmax = Epilogue::EpilogueAscend950FASoftmax<enableMaskFlag>;
    using PType = Gemm::GemmType<ElementP, LayoutTagP>;
    using SType = Gemm::GemmType<ElementS, LayoutTagS>;
    using maskType = Gemm::GemmType<ElementMask, LayoutTagMask>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<DispatchPolicySoftmax, L1TileShape, PType, SType, maskType>;

    // GEMM Block模块，实现Flash Attention Infer的P * V
    using DispatchPolicyPV = Gemm::MmadFAIPV<ArchTag, enablePaFlag>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV, ElementOTmp, LayoutTagV, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShape, L0TileShape, ElementP, ElementV, ElementOTmp, void, TileCopyPV, TileMmadPV>;

    // Epilogue Block模块，实现Flash Attention Infer中当前O基块的更新
    using DispatchPolicyRescaleO = Epilogue::EpilogueAscend950FARescaleO;
    using OType = Gemm::GemmType<ElementO, LayoutTagO>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutTagOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, L1TileShape, OType, OTmpType>;

    using FAInferKernel =
        FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, enablePaFlag>;
    FAIKernelParams params{q, k, v, mask, blockTables, actualQSeqlen, actualKvSeqlen, o, tiling};
    // call kernel
    FAInferKernel flashAttnInfer;
    flashAttnInfer(params);
}

#endif // XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_KERNEL_H