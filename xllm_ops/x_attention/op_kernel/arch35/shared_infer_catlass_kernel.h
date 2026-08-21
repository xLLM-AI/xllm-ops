
#ifndef X_ATTN_SHARED_FA_INFER_CATLASS_KERNEL_H
#define X_ATTN_SHARED_FA_INFER_CATLASS_KERNEL_H

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
    class EpilogueFASoftmax,
    class EpilogueFARescale,
    typename KVLEN_T>
class SharedFaInferKernel {
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
        SharedFaInferKernel(XAttentionTilingData *tilingData) {
            this->tilingData = tilingData;
        }

        template <int32_t CORE_TYPE = g_coreType>
        CATLASS_DEVICE void operator()(XAttnKernelCommonParams const &params);

        CATLASS_DEVICE void Init(XAttnKernelCommonParams const &params) {
            auto qkSize = halfQSeqlenTemplateType * kvSeqlenTemplateType * sizeof(ElementS);
            auto pvSize = halfQSeqlenTemplateType * embedTemplateType * sizeof(ElementOTmp);

            for (int i = 0; i < 2; i++) {
                qkTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += qkSize;
                pvTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ubBufAddrStart);
                ubBufAddrStart += pvSize;
            }

            auto reduceSize = halfQSeqlenTemplateType * sizeof(ElementS);

            if ASCEND_IS_AIV {
                for (int i = 0; i < 3; i++) {
                    expSumUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                    ubBufAddrStart += reduceSize;
                    expMaxUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                    ubBufAddrStart += reduceSize;
                    maxUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                    ubBufAddrStart += reduceSize;
                }
            }

            auto pL1Size = qSeqlenTemplateType * kvSeqlenTemplateType * sizeof(ElementP);
            for (int i = 0; i < 3; i++) {
                pL1TensorList[i] = resource.l1Buf.template GetBufferByByte<ElementP>(l1BufAddrStart);
                l1BufAddrStart += pL1Size;
            }

            sharedKvLensGm.SetGlobalBuffer((__gm__ KVLEN_T *)params.sharedKvLens);

            batchSize = tilingData->baseInfo.batchSize;
            beamSize = tilingData->baseInfo.beamSize;
            qHeads = tilingData->baseInfo.qHeads;
            kvHeads = tilingData->baseInfo.kvHeads;
            groupSize = tilingData->baseInfo.groupSize;
            headDim = tilingData->baseInfo.headDim;
            scaleValue = tilingData->baseInfo.scaleValue;
            totalTokensQ = tilingData->baseInfo.totalTokensQ;
            sharedKvTokens = tilingData->baseInfo.sharedKvTokens;

            coreNum = tilingData->sharedInfo.usedCoreNum;
            coreIdx = AscendC::GetBlockIdx();

            if ASCEND_IS_AIV {
                coreIdx = coreIdx / CV_RATIO;
                subVecIdx = AscendC::GetSubBlockIdx();
            }

            strideQO = qHeads * headDim;
            strideKV = kvHeads * headDim; 
        }

        CATLASS_DEVICE void operator()(XAttnKernelCommonParams const &params) {
            uint32_t taskIdL0A = 0;
            uint32_t taskIdL0B = 0;

            Init(params);
            SetFlag();

            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            auto layoutQ = tla::MakeLayout<ElementQ, LayoutTagQ>(totalTokensQ, qHeads * headDim);
            auto tensorQ = tla::MakeTensor(gQ, layoutQ, Arch::PositionGM{});
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)params.sharedK);
            auto layoutK = tla::MakeLayout<ElementK, LayoutTagK>(kvHeads * headDim, sharedKvTokens);
            auto tensorK = tla::MakeTensor(gK, layoutK, Arch::PositionGM{});
            AscendC::GlobalTensor<ElementV> gV;
            gV.SetGlobalBuffer((__gm__ ElementV *)params.sharedV);
            auto layoutV = tla::MakeLayout<ElementV, LayoutTagV>(sharedKvTokens, kvHeads * headDim);
            auto tensorV = tla::MakeTensor(gV, layoutV, Arch::PositionGM{});

            AscendC::GlobalTensor<ElementOTmp> gSharedO;
            gSharedO.SetGlobalBuffer((__gm__ ElementOTmp *)params.sharedO);
            auto layoutO = tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(totalTokensQ, qHeads * headDim);
            auto tensorO = tla::MakeTensor(gSharedO, layoutO, Arch::PositionGM{});
            AscendC::GlobalTensor<ElementOTmp> sharedMaxGm;
            sharedMaxGm.SetGlobalBuffer((__gm__ ElementOTmp *)params.sharedMax);
            AscendC::GlobalTensor<ElementOTmp> sharedSumGm;
            sharedSumGm.SetGlobalBuffer((__gm__ ElementOTmp *)params.sharedSum);

            BlockMmadQK blockMmadQK(resource, l1BufAddrStart, l0CBufAddrStart);
            BlockMmadPV blockMmadPV(resource, l1BufAddrStart, l0CBufAddrStart);
            EpilogueFASoftmax epilogueSoftmax(resource, ubBufAddrStart, scaleValue, qHeads);
            EpilogueFARescale epilogueRescale(resource, ubBufAddrStart);

            int32_t taskId = 0;
            int32_t perCoreTaskNum = tilingData->sharedInfo.perCoreTaskNum;
            int32_t totalTaskNum = tilingData->sharedInfo.totalTaskNum;

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

            // AscendC::printf("coreNum %d coreIdx %d coreTaskNum %d taskStartId %d taskEndId %d\n",
            // coreNum, coreIdx, coreTaskNum, taskStartId, taskEndId);

            for (int32_t qTaskId = taskStartId; qTaskId < taskEndId + 3; qTaskId++)
            {
                bool notLastThreeLoop = qTaskId < taskEndId;
                bool notLastTwoLoop = qTaskId < taskEndId + 1;
                bool notLast = qTaskId < taskEndId + 2;

                int32_t kvLen = 0;
                int32_t kvBlockNum = 1;

                if (notLastThreeLoop) {
                    SharedInfer::TaskArgs taskArgs;
                    GetQTaskInfo(taskArgs, qTaskId);
                    taskArgList[taskId % 4] = taskArgs;
                    kvLen = taskArgs.actualKvLen;
                    kvBlockNum = (kvLen + kvSeqlenTemplateType - 1) / kvSeqlenTemplateType;
                }

                for (int32_t kvBlockId = 0; kvBlockId < kvBlockNum; kvBlockId++) {
                    if (notLastThreeLoop) {
                        auto nowTaskId = taskId % 4;
                        SharedInfer::TaskArgs &taskArgsNow = taskArgList[nowTaskId];
                        GetKvTaskInfo(taskArgsNow, kvBlockId, kvBlockNum, taskId);

                        if ASCEND_IS_AIC {
                            auto actualShape = tla::MakeShape(taskArgsNow.blockQLen, taskArgsNow.blockKvLen, headDim);
                            auto tensorQTile = GetTile(
                                tensorQ,
                                tla::MakeCoord(taskArgsNow.qCoord, taskArgsNow.qNCoord),
                                tla::MakeShape(taskArgsNow.blockQLen, headDim)
                            );

                            auto tensorKTile = GetTile(
                                tensorK,
                                tla::MakeCoord(taskArgsNow.kvNCoord, taskArgsNow.kvCoord),
                                tla::MakeShape(headDim, taskArgsNow.blockKvLen)
                            );

                            auto layoutQKRes = tla::MakeLayout<ElementS, LayoutTagS>(taskArgsNow.blockQLen, kvSeqlenTemplateType);
                            auto tensorQKRes = tla::MakeTensor(qkTensorList[taskArgsNow.taskIdMod2], layoutQKRes, Arch::PositionUB{});
                            blockMmadQK(
                                tensorQTile, tensorKTile, tensorQKRes, actualShape,
                                SharedInfer::QK_UB_RELEASE_FLAG[taskArgsNow.taskIdMod2],
                                taskArgsNow.isFirstKv, taskArgsNow.isLastKv,
                                taskIdL0A, taskIdL0B
                            );

                            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SharedInfer::SYNC_QK_READY_FLAG[taskArgsNow.taskIdMod2]);
                            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SharedInfer::SYNC_QK_READY_FLAG[taskArgsNow.taskIdMod2]);
                        }
                    }

                    if (taskId > 0 && notLastTwoLoop) {
                        if ASCEND_IS_AIV {
                            auto &taskArgsPre = taskArgList[(taskId - 1) % 4];
                            auto qkResLayout = tla::MakeLayout<ElementS, LayoutTagS>(taskArgsPre.halfBlockQLen, taskArgsPre.blockKvLen);
                            auto qkResTensor = tla::MakeTensor(qkTensorList[taskArgsPre.taskIdMod2], qkResLayout, Arch::PositionUB{});
                            auto pL1OutLayout = tla::MakeLayout<ElementP, LayoutTagPL1>(qSeqlenTemplateType, kvSeqlenTemplateType);
                            auto pL1OutTensor = tla::MakeTensor(pL1TensorList[taskArgsPre.taskIdMod3], pL1OutLayout, Arch::PositionL1{});
                            auto pL1OutTile = GetTile(
                                pL1OutTensor,
                                tla::MakeCoord(taskArgsPre.halfBlockQOffset, 0),
                                tla::MakeShape(taskArgsPre.halfBlockQLen, kvSeqlenTemplateType)
                            );
                            auto sharedMaxTile = sharedMaxGm[taskArgsPre.maxOutOffset];
                            auto sharedSumTile = sharedSumGm[taskArgsPre.maxOutOffset];

                            epilogueSoftmax(
                                pL1OutTile,
                                qkResTensor,
                                expSumUb[(taskArgsPre.taskId - 1) % 3],
                                expSumUb[taskArgsPre.taskIdMod3],
                                expMaxUb[taskArgsPre.taskIdMod3],
                                maxUb[(taskArgsPre.taskId - 1) % 3],
                                maxUb[taskArgsPre.taskIdMod3],
                                sharedMaxTile,
                                sharedSumTile,
                                taskArgsPre.isUpdate,
                                taskArgsPre.isLastKv,
                                SharedInfer::SYNC_QK_READY_FLAG[taskArgsPre.taskIdMod2],
                                SharedInfer::SYNC_SOFTMAX_READY_FLAG[taskArgsPre.taskIdMod3],
                                SharedInfer::QK_UB_RELEASE_FLAG[taskArgsPre.taskIdMod2],
                                taskArgsPre.taskIdMod2,
                                taskArgsPre.taskIdMod3
                            );
                        }
                    }

                    if (taskId > 1 && notLast) {
                        if ASCEND_IS_AIC {
                            auto &taskArgsPre2 = taskArgList[(taskId - 2) % 4];
                            AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SharedInfer::SYNC_SOFTMAX_READY_FLAG[taskArgsPre2.taskIdMod3]);
                            AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SharedInfer::SYNC_SOFTMAX_READY_FLAG[taskArgsPre2.taskIdMod3]);

                            auto layoutPvRes = tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(taskArgsPre2.blockQLen, embedTemplateType);
                            auto tensorPvRes = tla::MakeTensor(pvTensorList[taskArgsPre2.taskIdMod2], layoutPvRes, Arch::PositionUB{});

                            auto layoutPInL1 = tla::MakeLayout<ElementP, LayoutTagPL1>(qSeqlenTemplateType, kvSeqlenTemplateType);
                            auto tensorPInL1 = tla::MakeTensor(pL1TensorList[taskArgsPre2.taskIdMod3], layoutPInL1, Arch::PositionL1{});

                            auto tensorInV = GetTile(
                                tensorV,
                                tla::MakeCoord(taskArgsPre2.kvCoord, taskArgsPre2.kvNCoord),
                                tla::MakeShape(taskArgsPre2.blockKvLen, headDim)
                            );

                            auto actualShape = tla::MakeShape(taskArgsPre2.blockQLen, headDim, taskArgsPre2.blockKvLen);

                            blockMmadPV(
                                tensorPInL1, tensorInV, tensorPvRes,
                                actualShape, taskIdL0A, taskIdL0B, SharedInfer::PV_UB_RELEASE_FLAG[taskArgsPre2.taskIdMod2]
                            );

                            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SharedInfer::SYNC_PV_READY_FLAG[taskArgsPre2.taskIdMod2]);
                            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SharedInfer::SYNC_PV_READY_FLAG[taskArgsPre2.taskIdMod2]);
                        }
                    }

                    if (taskId > 2) {
                        if ASCEND_IS_AIV {
                            auto &taskArgsPre3 = taskArgList[(taskId - 3) % 4];
                            AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SharedInfer::SYNC_PV_READY_FLAG[taskArgsPre3.taskIdMod2]);

                            auto layoutPvRes = tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(taskArgsPre3.halfBlockQLen, headDim);
                            auto tensorPvRes = tla::MakeTensor(pvTensorList[taskArgsPre3.taskIdMod2], layoutPvRes, Arch::PositionUB{});

                            auto sharedAttnOutGmTile = GetTile(
                                tensorO,
                                tla::MakeCoord(taskArgsPre3.qCoord + taskArgsPre3.halfBlockQOffset, taskArgsPre3.qNCoord),
                                tla::MakeShape(taskArgsPre3.halfBlockQLen, headDim)
                            );

                            epilogueRescale(
                                sharedAttnOutGmTile,
                                expMaxUb[taskArgsPre3.taskIdMod3],
                                tensorPvRes,
                                taskArgsPre3.isFirstKv,
                                taskArgsPre3.isLastKv,
                                SharedInfer::PV_UB_RELEASE_FLAG[taskArgsPre3.taskIdMod2]
                            );
                        }
                    }

                    auto nextTaskId = (taskId + 1) % 4;
                    auto currentTaskId = taskId % 4;
                    taskArgList[nextTaskId] = taskArgList[currentTaskId];
                    taskId++;
                }
            }

            WaitFlag();

            // dump sharedO
            // if (coreIdx == 0) {
            //     AscendC::printf("qHeads %d kvHeads %d headDim %d\n", qHeads, kvHeads, headDim);
            //     for (int i = 0; i < 8; i++) {
            //         AscendC::printf("token %d sharedO res\n", i);
            //         AscendC::DumpTensor(gSharedO[i * strideQO], 1, 8);
            //     }
            //     AscendC::printf("sharedMax res\n");
            //     AscendC::DumpTensor(sharedMaxGm, 6, 8);
            //     AscendC::printf("sharedSum res\n");
            //     AscendC::DumpTensor(sharedSumGm, 8, 8);
            // }
        }
    
    private:
        static constexpr uint8_t SYNC_MODE = 4;
        Arch::Resource<ArchTag> resource;
        AscendC::GlobalTensor<KVLEN_T> sharedKvLensGm;
        AscendC::LocalTensor<ElementS> qkTensorList[2];
        AscendC::LocalTensor<ElementP> pL1TensorList[3];
        AscendC::LocalTensor<ElementOTmp> pvTensorList[2];
        AscendC::LocalTensor<ElementS> expSumUb[3];
        AscendC::LocalTensor<ElementS> expMaxUb[3];
        AscendC::LocalTensor<ElementS> maxUb[3];

        SharedInfer::TaskArgs taskArgList[4];
        XAttentionTilingData* tilingData;

        int32_t batchSize{0};
        int32_t beamSize{0};
        int32_t qHeads{0};
        int32_t kvHeads{0};
        int32_t groupSize{0};
        int32_t headDim{0};
        int32_t totalTokensQ{0};
        int32_t sharedKvTokens{0};
        int64_t coreNum;
        int64_t coreIdx;
        int64_t subVecIdx{0};
        float scaleValue;

        uint64_t strideQO{0};
        uint64_t strideKV{0};
        uint32_t l1BufAddrStart = 0;
        uint32_t l0CBufAddrStart = 0;
        uint32_t ubBufAddrStart = 0;

    private:
        CATLASS_DEVICE void GetQTaskInfo(SharedInfer::TaskArgs &taskArgs, int32_t qTaskId) {
            int32_t perBatchHeadTaskNum = tilingData->sharedInfo.perBatchHeadTaskNum;
            int32_t qBlockId = qTaskId % perBatchHeadTaskNum;
            int32_t outerId = qTaskId / perBatchHeadTaskNum;
            int32_t qHeadId = outerId % qHeads;
            int32_t batchId = outerId / qHeads;
            taskArgs.batchId = batchId;
            taskArgs.qHeadId = qHeadId;
            taskArgs.kvHeadId = qHeadId / groupSize;
            taskArgs.qBlockId = qBlockId;
            taskArgs.blockQLen = qBlockId == (perBatchHeadTaskNum - 1) ? (beamSize - qBlockId * qSeqlenTemplateType) : qSeqlenTemplateType;
            taskArgs.actualKvLen = sharedKvLensGm.GetValue(batchId);
            taskArgs.qCoord = batchId * beamSize + qBlockId * qSeqlenTemplateType;
            taskArgs.qNCoord = qHeadId * headDim;
            taskArgs.kvNCoord = taskArgs.kvHeadId * headDim;

            if ASCEND_IS_AIV {
                int32_t halfQLen = (taskArgs.blockQLen + 1) / 2;
                taskArgs.halfBlockQLen = (subVecIdx == 0) ? halfQLen : (taskArgs.blockQLen - halfQLen);
                taskArgs.halfBlockQOffset = (subVecIdx == 0) ? 0 : halfQLen;
                taskArgs.maxOutOffset = (taskArgs.qCoord + taskArgs.halfBlockQOffset) * qHeads + qHeadId;
            }

            int32_t batchOffset = 0;
            for (int bId = 0; bId < batchId; bId++) {
                batchOffset += sharedKvLensGm.GetValue(bId);
            }

            taskArgs.kvBatchOffset = batchOffset;
        }

        CATLASS_DEVICE void GetKvTaskInfo(SharedInfer::TaskArgs &taskArgs, int32_t kvBlockId, int32_t kvBlockNum, int32_t taskId) {
            auto actualKvLen = taskArgs.actualKvLen;
            bool isFirstKv = kvBlockId == 0;
            bool isUpdate = kvBlockId > 0;
            bool isLastKv = kvBlockId == kvBlockNum - 1;
            taskArgs.taskId = taskId;
            taskArgs.kvBlockId = kvBlockId;
            taskArgs.blockKvLen = isLastKv ? (actualKvLen - kvBlockId * kvSeqlenTemplateType) : kvSeqlenTemplateType;
            taskArgs.kvCoord = taskArgs.kvBatchOffset + kvBlockId * kvSeqlenTemplateType;
            taskArgs.isFirstKv = isFirstKv;
            taskArgs.isUpdate = isUpdate;
            taskArgs.isLastKv = isLastKv;
            taskArgs.taskIdMod2 = taskId % 2;
            taskArgs.taskIdMod3 = taskId % 3;
        }

        CATLASS_DEVICE void SetFlag() {
            if ASCEND_IS_AIC {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            } else {
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SharedInfer::QK_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SharedInfer::QK_UB_RELEASE_FLAG[1]);
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SharedInfer::PV_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SharedInfer::PV_UB_RELEASE_FLAG[1]);
            }
        }

        CATLASS_DEVICE void WaitFlag() {
            if ASCEND_IS_AIC {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SharedInfer::QK_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SharedInfer::QK_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SharedInfer::QK_UB_RELEASE_FLAG[1]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SharedInfer::QK_UB_RELEASE_FLAG[1]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SharedInfer::PV_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SharedInfer::PV_UB_RELEASE_FLAG[0]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SharedInfer::PV_UB_RELEASE_FLAG[1]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SharedInfer::PV_UB_RELEASE_FLAG[1]);
            }
        }
};

#endif