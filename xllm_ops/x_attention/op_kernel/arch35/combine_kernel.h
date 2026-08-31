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
#include "x_attention_common.h"
#include "kernel_operator.h"

template <class EpilogueCombineScale>
class CombineScaleKernel {
public:
    using ArchTag = typename EpilogueCombineScale::ArchTag;
    using ElementOutput = typename EpilogueCombineScale::ElementOutput;
    using ElementInput = typename EpilogueCombineScale::ElementInput;

    CATLASS_DEVICE
    CombineScaleKernel(XAttentionTilingData* tilingDataPtr): faTilingData(tilingDataPtr) {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(XAttnKernelCommonParams const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(XAttnKernelCommonParams const &params) {
        return;
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(XAttnKernelCommonParams const &params) {
        uint32_t ubBufAddrStart = 0;
        uint32_t rowNumPerLoop = faTilingData->combineInfo.rowPerLoop;
        uint32_t headDim = faTilingData->baseInfo.headDim;
        EpilogueCombineScale epilogueCombineScale(resource, ubBufAddrStart, rowNumPerLoop, headDim);

        int32_t usedCoreNum = faTilingData->combineInfo.usedCoreNum;
        int32_t coreIdx = AscendC::GetBlockIdx();
        
        if (coreIdx >= usedCoreNum) {
            return;
        }

        int32_t formerCoreNum = faTilingData->combineInfo.formerCoreNum;
        int32_t tailCoreNum = usedCoreNum - formerCoreNum;
        int32_t totalTaskNum = faTilingData->combineInfo.totalTaskNum;
        int32_t formerTaskNum = faTilingData->combineInfo.formerTaskNum;
        int32_t tailTaskNum = faTilingData->combineInfo.tailTaskNum;
        int32_t rowNum = faTilingData->combineInfo.rowNum;
        int32_t coreTaskNum;
        int32_t mainTaskRowNum;
        int32_t tailTaskRowNum;
        int32_t attnOffsetPerCore;
        int32_t gmglOffsetPerCore;

        // AscendC::printf("rowNum %d rowNumPerLoop %d formerCoreNum %d formerTaskNum %d tailTaskNum %d usedCoreNum %d tailCoreNum %d\n", rowNum, 
        // rowNumPerLoop, formerCoreNum, formerTaskNum, tailTaskNum, usedCoreNum, tailCoreNum);

        if (coreIdx < formerCoreNum) {
            coreTaskNum = formerTaskNum;
            mainTaskRowNum = rowNumPerLoop;
            tailTaskRowNum = rowNumPerLoop;
            gmglOffsetPerCore = coreIdx * rowNumPerLoop * coreTaskNum;
            attnOffsetPerCore = gmglOffsetPerCore * headDim;
        } else {
            coreTaskNum = tailTaskNum;
            mainTaskRowNum = rowNumPerLoop;
            tailTaskRowNum = rowNum - formerCoreNum * formerTaskNum * rowNumPerLoop - (tailCoreNum - 1) * mainTaskRowNum;
            gmglOffsetPerCore = (formerCoreNum * formerTaskNum + (coreIdx - formerCoreNum) * tailTaskNum) * rowNumPerLoop;
            attnOffsetPerCore = gmglOffsetPerCore * headDim;
        }

        AscendC::GlobalTensor<ElementInput> gSharedGm;
        gSharedGm.SetGlobalBuffer((__gm__ ElementInput *)params.sharedMax + gmglOffsetPerCore);
        AscendC::GlobalTensor<ElementInput> gSharedGl;
        gSharedGl.SetGlobalBuffer((__gm__ ElementInput *)params.sharedSum + gmglOffsetPerCore);
        AscendC::GlobalTensor<ElementInput> gUnsharedGm;
        gUnsharedGm.SetGlobalBuffer((__gm__ ElementInput *)params.unsharedMax + gmglOffsetPerCore);
        AscendC::GlobalTensor<ElementInput> gUnsharedGl;
        gUnsharedGl.SetGlobalBuffer((__gm__ ElementInput *)params.unsharedSum + gmglOffsetPerCore);
        AscendC::GlobalTensor<ElementInput> gSharedOut;
        gSharedOut.SetGlobalBuffer((__gm__ ElementInput *)params.sharedO + attnOffsetPerCore);
        AscendC::GlobalTensor<ElementInput> gUnsharedOut;
        gUnsharedOut.SetGlobalBuffer((__gm__ ElementInput *)params.unsharedO + attnOffsetPerCore);
        AscendC::GlobalTensor<ElementOutput> gFinalOut;
        gFinalOut.SetGlobalBuffer((__gm__ ElementOutput *)params.o + attnOffsetPerCore);
        
        // if (coreIdx == 0) {
        //     for (int i = 68; i < 69; i++) {
        //         AscendC::printf("token %d sharedOut\n", i);
        //         AscendC::DumpTensor(gSharedOut[i * headDim], 1, 8);
        //         AscendC::printf("token %d sharedMax %f\n", i, gSharedGm.GetValue(i));
        //         AscendC::printf("token %d sharedSum %f\n", i, gSharedGl.GetValue(i));
        //         AscendC::printf("token %d unsharedOut\n", i);
        //         AscendC::DumpTensor(gUnsharedOut[i * headDim], 3, 8);
        //         AscendC::printf("token %d unsharedMax %f\n", i, gUnsharedGm.GetValue(i));
        //         AscendC::printf("token %d unsharedSum %f\n", i, gUnsharedGl.GetValue(i));
        //     }
        // }


        int8_t taskId = 0;
        for (int i = 0; i < coreTaskNum; i++) {
            // int32_t realRowNum = (i == coreTaskNum - 1) ? tailTaskRowNum : mainTaskRowNum;
            int64_t gmglTaskOffset = i * rowNumPerLoop;
            int64_t globalRowStart = gmglOffsetPerCore + gmglTaskOffset;
            int32_t remainingRows = rowNum - globalRowStart;
            int32_t realRowNum =
                remainingRows < static_cast<int32_t>(rowNumPerLoop)
                    ? remainingRows
                    : static_cast<int32_t>(rowNumPerLoop);

            if (realRowNum <= 0) {
                break;
            }
            int64_t attnTaskOffset = gmglTaskOffset * headDim;
            epilogueCombineScale(
                gSharedGm[gmglTaskOffset],
                gUnsharedGm[gmglTaskOffset],
                gSharedGl[gmglTaskOffset],
                gUnsharedGl[gmglTaskOffset],
                gSharedOut[attnTaskOffset],
                gUnsharedOut[attnTaskOffset],
                gFinalOut[attnTaskOffset],
                realRowNum,
                taskId
            );
        }
    }
private:
    Arch::Resource<ArchTag> resource;
    XAttentionTilingData* faTilingData;
};
