
#ifndef X_ATTN_UNSHARED_FA_INFER_CATLASS_KERNEL_H
#define X_ATTN_UNSHARED_FA_INFER_CATLASS_KERNEL_H

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

using namespace Catlass;

template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpiloueSoftmax,
    typename KVLEN_T,
    typename TABLE_T>
class UnSharedInferKernel {
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
        using LayoutTagPL1 = typename BlockMmadPV::TileCopy::LayoutTagL1A;
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutTagV = typename BlockMmadPV::LayoutTagB;
        using ElementOTmp = typename BlockMmadPV::ElementC;
        using LayoutTagOTmp = typename BlockMmadPV::LayoutTagC;

        static constexpr uint32_t qSeqlenTemplateType = tla::get<0>(L1TileShape{});
        static constexpr uint32_t kvSeqlenTemplateType = tla::get<1>(L1TileShape{});
        static constexpr uint32_t embedTemplateType = tla::get<2>(L1TileShape{});
        static constexpr uint32_t halfQSeqlenTemplateType = qSeqlenTemplateType / CV_RATIO;

        CATLASS_DEVICE
        UnSharedInferKernel(XAttentionTilingData *tilingData) {
            this->tilingData = tilingData;
        }

        template <int32_t CORE_TYPE = g_coreType>
        CATLASS_DEVICE void operator()(XAttnKernelCommonParams const &params);

        CATLASS_DEVICE void Init(XAttnKernelCommonParams const &params) {
            auto qkSize = halfQSeqlenTemplateType * kvSeqlenTemplateType * sizeof(ElementS);
            // AscendC::printf("qkSize %d ubBufAddrStart %d\n", qkSize, ubBufAddrStart);

            for (int i = 0; i < 2; i++) {
                qkTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += qkSize;
            }
            auto pL1Size = qSeqlenTemplateType * kvSeqlenTemplateType * sizeof(ElementP);
            for (int i = 0; i < 3; i++) {
                pL1TensorList[i] = resource.l1Buf.template GetBufferByByte<ElementP>(l1BufAddrStart);
                l1BufAddrStart += pL1Size;
            }

            decodeStepGm.SetGlobalBuffer((__gm__ KVLEN_T *)params.decodeStep);
            unsharedKvSeqLen = static_cast<int32_t>(decodeStepGm.GetValue(0));
            blockTableGm.SetGlobalBuffer((__gm__ TABLE_T *)params.unsharedBlockTable);

            batchSize = tilingData->baseInfo.batchSize;
            beamSize = tilingData->baseInfo.beamSize;
            qHeads = tilingData->baseInfo.qHeads;
            kvHeads = tilingData->baseInfo.kvHeads;
            groupSize = tilingData->baseInfo.groupSize;
            headDim = tilingData->baseInfo.headDim;
            scaleValue = tilingData->baseInfo.scaleValue;
            totalTokensQ = tilingData->baseInfo.totalTokensQ;
            maxDecodeStep = tilingData->baseInfo.maxDecodeStep;

            kvBatchStride = tilingData->unsharedInfo.kvBatchStride;
            groupCountPerLoop = tilingData->unsharedInfo.groupCountPerLoop;
            perBatchTaskNum = tilingData->unsharedInfo.perBatchTaskNum;
            perCoreTaskNum = tilingData->unsharedInfo.perCoreTaskNum;
            totalTaskNum = tilingData->unsharedInfo.totalTaskNum;
            coreNum = tilingData->unsharedInfo.usedCoreNum;

            coreIdx = AscendC::GetBlockIdx();
            subVecIdx = AscendC::GetSubBlockIdx();
            if ASCEND_IS_AIV {
                coreIdx = coreIdx / CV_RATIO;
                int32_t halfGroupCount = (groupCountPerLoop + CV_RATIO - 1) / CV_RATIO;
                halfVecGroupCount = (subVecIdx == 0) ? halfGroupCount : (groupCountPerLoop - halfGroupCount);
                halfVecGroupOffset = (subVecIdx == 0) ? 0 : halfGroupCount;
                halfVecRowCount = halfVecGroupCount * groupSize;
                halfVecRowOffset = halfVecGroupOffset * groupSize;
            }
            coreIdx = coreIdx - tilingData->sharedInfo.usedCoreNum;
            blockQLen = groupCountPerLoop * groupSize;
            blockKvLen = groupCountPerLoop * maxDecodeStep;
        }

        CATLASS_DEVICE void operator()(XAttnKernelCommonParams const &params) {
            uint32_t taskIdL0A = 0;
            uint32_t taskIdL0B = 0;
            uint32_t taskIdL0C = 0;

            Init(params);
            SetFlag();

            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            auto layoutQ = tla::MakeLayout<ElementQ, LayoutTagQ>(totalTokensQ * qHeads, headDim);
            auto tensorQ = tla::MakeTensor(gQ, layoutQ, Arch::PositionGM{});
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)params.unsharedK);
            AscendC::GlobalTensor<ElementV> gV;
            gV.SetGlobalBuffer((__gm__ ElementV *)params.unsharedV);

            AscendC::GlobalTensor<ElementOTmp> gUnSharedO;
            gUnSharedO.SetGlobalBuffer((__gm__ ElementOTmp *)params.unsharedO);
            auto layoutO = tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(totalTokensQ * qHeads, headDim);
            auto tensorO = tla::MakeTensor(gUnSharedO, layoutO, Arch::PositionGM{});
            
            AscendC::GlobalTensor<ElementOTmp> unsharedMaxGm;
            unsharedMaxGm.SetGlobalBuffer((__gm__ ElementOTmp *)params.unsharedMax);
            AscendC::GlobalTensor<ElementOTmp> unsharedSumGm;
            unsharedSumGm.SetGlobalBuffer((__gm__ ElementOTmp *)params.unsharedSum);

            BlockMmadQK blockMmadQK(resource, l1BufAddrStart, l0CBufAddrStart);
            BlockMmadPV blockMmadPV(resource, l1BufAddrStart, l0CBufAddrStart);
            EpiloueSoftmax epilogueSoftmax(resource, ubBufAddrStart, scaleValue, unsharedKvSeqLen, maxDecodeStep, groupCountPerLoop, groupSize);

            int32_t batchKvLen = beamSize * kvHeads * maxDecodeStep;
            int32_t taskId = 0;
            int32_t perCoreTaskNum = tilingData->unsharedInfo.perCoreTaskNum;
            int32_t totalTaskNum = tilingData->unsharedInfo.totalTaskNum;

            int32_t taskStartId = coreIdx * perCoreTaskNum;
            if (taskStartId >= totalTaskNum) {
                WaitFlag();
                return;
            }
            int32_t coreTaskNum = perCoreTaskNum;
            int32_t taskEndId = taskStartId + coreTaskNum;

            if (taskEndId > totalTaskNum) {
                taskEndId = totalTaskNum;
                coreTaskNum = taskEndId - taskStartId;
            }

            // AscendC::printf("coreIdx %d taskStartId %d taskEndId %d coreTaskNum %d\n", coreIdx, taskStartId, taskEndId, coreTaskNum);

            for (int32_t groupTaskId = taskStartId; groupTaskId < taskEndId + 2; groupTaskId++)
            {
                bool notLastTwoLoop = groupTaskId < taskEndId;
                bool notLast = groupTaskId < taskEndId + 1;

                if (notLastTwoLoop) {
                    UnSharedInfer::TaskArgs taskArgs;
                    GetTaskInfo(taskArgs, groupTaskId, taskId);
                    taskArgList[taskId % 3] = taskArgs;
                    if ASCEND_IS_AIC {
                        auto nowTaskId = taskId % 3;
                        UnSharedInfer::TaskArgs &taskArgsNow = taskArgList[nowTaskId];
                        auto actualShape = tla::MakeShape(blockQLen, blockKvLen, headDim);
                        auto tensorQTile = GetTile(
                            tensorQ,
                            tla::MakeCoord(taskArgsNow.qCoord, 0),
                            tla::MakeShape(blockQLen, headDim)
                        );

                        auto layoutK = tla::MakeLayout<ElementK, LayoutTagK>(headDim, batchKvLen);
                        auto tensorK = tla::MakeTensor(gK[taskArgsNow.cacheBlockId * kvBatchStride], layoutK, Arch::PositionGM{});
                        auto tensorKTile = GetTile(
                            tensorK,
                            tla::MakeCoord(0, taskArgsNow.kvCoord),
                            tla::MakeShape(headDim, blockKvLen)
                        );

                        auto layoutQKRes = tla::MakeLayout<ElementS, LayoutTagS>(blockQLen, kvSeqlenTemplateType);
                        auto tensorQKRes = tla::MakeTensor(qkTensorList[taskArgsNow.taskIdMod2], layoutQKRes, Arch::PositionUB{});
                        
                        blockMmadQK(
                            tensorQTile, tensorKTile, tensorQKRes, actualShape,
                            UnSharedInfer::QK_UB_RELEASE_FLAG[taskArgsNow.taskIdMod2],
                            taskIdL0A, taskIdL0B, taskIdL0C
                        );
                        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(UnSharedInfer::SYNC_QK_READY_FLAG[taskArgsNow.taskIdMod2]);
                        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + UnSharedInfer::SYNC_QK_READY_FLAG[taskArgsNow.taskIdMod2]);

                    }
                }

                if (taskId > 0 && notLast) {
                    if ASCEND_IS_AIV {
                        auto &taskArgsPre = taskArgList[(taskId - 1) % 3];
                        auto qkResLayout = tla::MakeLayout<ElementS, LayoutTagS>(halfVecRowCount, blockKvLen);
                        auto qkResTensor = tla::MakeTensor(qkTensorList[taskArgsPre.taskIdMod2], qkResLayout, Arch::PositionUB{});
                        auto pL1OutLayout = tla::MakeLayout<ElementP, LayoutTagPL1>(qSeqlenTemplateType, kvSeqlenTemplateType);
                        auto pL1OutTensor = tla::MakeTensor(pL1TensorList[taskArgsPre.taskIdMod3], pL1OutLayout, Arch::PositionL1{});
                        auto pL1OutTile = GetTile(
                            pL1OutTensor,
                            tla::MakeCoord(halfVecRowOffset, 0),
                            tla::MakeShape(halfVecRowCount, kvSeqlenTemplateType)
                        );
                        auto unsharedMaxTile = unsharedMaxGm[taskArgsPre.maxOutOffset];
                        auto unsharedSumTile = unsharedSumGm[taskArgsPre.maxOutOffset];

                        epilogueSoftmax(
                            pL1OutTile,
                            qkResTensor,
                            unsharedMaxTile,
                            unsharedSumTile,
                            UnSharedInfer::SYNC_QK_READY_FLAG[taskArgsPre.taskIdMod2],
                            UnSharedInfer::SYNC_SOFTMAX_READY_FLAG[taskArgsPre.taskIdMod3],
                            UnSharedInfer::QK_UB_RELEASE_FLAG[taskArgsPre.taskIdMod2],
                            taskArgsPre.taskIdMod2,
                            taskArgsPre.taskIdMod3
                        );

                    }
                }

                if (taskId > 1) {
                    if ASCEND_IS_AIC {
                        auto &taskArgsPre2 = taskArgList[(taskId - 2) % 3];
                        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(UnSharedInfer::SYNC_SOFTMAX_READY_FLAG[taskArgsPre2.taskIdMod3]);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + UnSharedInfer::SYNC_SOFTMAX_READY_FLAG[taskArgsPre2.taskIdMod3]);

                        auto tensorOTile = GetTile(
                            tensorO,
                            tla::MakeCoord(taskArgsPre2.qCoord, 0),
                            tla::MakeShape(blockQLen, headDim)
                        );

                        auto layoutPInL1 = tla::MakeLayout<ElementP, LayoutTagPL1>(qSeqlenTemplateType, kvSeqlenTemplateType);
                        auto tensorPInL1 = tla::MakeTensor(pL1TensorList[taskArgsPre2.taskIdMod3], layoutPInL1, Arch::PositionL1{});

                        auto layoutV = tla::MakeLayout<ElementV, LayoutTagV>(batchKvLen, headDim);
                        auto tensorV = tla::MakeTensor(gV[taskArgsPre2.cacheBlockId * kvBatchStride], layoutV, Arch::PositionGM{});
                        auto tensorVTile = GetTile(
                            tensorV,
                            tla::MakeCoord(taskArgsPre2.kvCoord, 0),
                            tla::MakeShape(blockKvLen, headDim)
                        );

                        auto actualShape = tla::MakeShape(blockQLen, headDim, blockKvLen);

                        blockMmadPV(
                            tensorPInL1, tensorVTile, tensorOTile,
                            actualShape, taskIdL0A, taskIdL0B, taskIdL0C
                        );
                    }
                }

                auto nextTaskId = (taskId + 1) % 3;
                auto currentTaskId = taskId % 3;
                taskArgList[nextTaskId] = taskArgList[currentTaskId];
                taskId++;
            }
            // if (coreIdx == 0) {
            //     AscendC::printf("qHeads %d kvHeads %d headDim %d\n", qHeads, kvHeads, headDim);
            //     for (int i = 64; i < 80; i++) {
            //         AscendC::printf("token %d unsharedO res\n", i);
            //         AscendC::DumpTensor(gUnSharedO[i * headDim], 1, 8);
            //     }
            //     AscendC::printf("unsharedMax res\n");
            //     AscendC::DumpTensor(unsharedMaxGm, 6, 8);
            //     AscendC::printf("unsharedSum res\n");
            //     AscendC::DumpTensor(unsharedSumGm, 8, 8);
            // }

            WaitFlag();
        }
    
    private:
        static constexpr uint8_t SYNC_MODE = 4;
        Arch::Resource<ArchTag> resource;
        AscendC::GlobalTensor<KVLEN_T> decodeStepGm;
        AscendC::LocalTensor<ElementS> qkTensorList[2];
        AscendC::LocalTensor<ElementP> pL1TensorList[3];
        AscendC::GlobalTensor<TABLE_T> blockTableGm;

        UnSharedInfer::TaskArgs taskArgList[3];
        XAttentionTilingData* tilingData;

        int32_t batchSize{0};
        int32_t beamSize{0};
        int32_t qHeads{0};
        int32_t kvHeads{0};
        int32_t groupSize{0};
        int32_t headDim{0};
        int32_t totalTokensQ{0};
        int32_t unsharedKvSeqLen{0};
        int32_t maxDecodeStep{0};
        int32_t groupCountPerLoop{0};
        int32_t kvBatchStride;
        int32_t perBatchTaskNum{0};
        int32_t perCoreTaskNum{0};
        int32_t totalTaskNum{0};
        int32_t halfVecGroupCount{0};
        int32_t halfVecGroupOffset{0};
        int32_t halfVecRowCount;
        int32_t halfVecRowOffset;
        int32_t blockQLen;
        int32_t blockKvLen;
        int32_t coreNum;
        int64_t coreIdx;
        int64_t subVecIdx{0};
        float scaleValue;

        uint32_t l1BufAddrStart = 0;
        uint32_t l0CBufAddrStart = 0;
        uint32_t ubBufAddrStart = 0;

    private:
        CATLASS_DEVICE void GetTaskInfo(UnSharedInfer::TaskArgs &taskArgs, int32_t groupTaskId, int32_t taskId) {
            taskArgs.taskId = taskId;
            taskArgs.taskIdMod2 = taskId % 2;
            taskArgs.taskIdMod3 = taskId % 3;

            int32_t batchId = groupTaskId / perBatchTaskNum;
            int32_t cacheBlockId = blockTableGm.GetValue(batchId);
            int32_t groupCountBlockId = groupTaskId % perBatchTaskNum;
            
            taskArgs.batchId = batchId;
            taskArgs.cacheBlockId = cacheBlockId;
            taskArgs.groupCountBlockId = groupCountBlockId;
            taskArgs.qCoord = batchId * beamSize * qHeads + groupCountBlockId * groupCountPerLoop * groupSize;
            taskArgs.kvCoord = groupCountBlockId * groupCountPerLoop * maxDecodeStep;

            if ASCEND_IS_AIV {
                taskArgs.maxOutOffset = taskArgs.qCoord + halfVecRowOffset;
            }

        }

        CATLASS_DEVICE void SetFlag() {
            if ASCEND_IS_AIC {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            } else {
                // AscendC::printf("coreIdx %d subVecIdx %d set qk_ub_flag %d %d \n", coreIdx, subVecIdx, UnSharedInfer::QK_UB_RELEASE_FLAG[0], UnSharedInfer::QK_UB_RELEASE_FLAG[1]);
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(UnSharedInfer::QK_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(UnSharedInfer::QK_UB_RELEASE_FLAG[1]);
            }
        }

        CATLASS_DEVICE void WaitFlag() {
            if ASCEND_IS_AIC {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(UnSharedInfer::QK_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(UnSharedInfer::QK_UB_RELEASE_FLAG[1]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + UnSharedInfer::QK_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + UnSharedInfer::QK_UB_RELEASE_FLAG[1]);
            }
        }
};

#endif