/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_v2_scan.h
 * \brief Scatter Kernel (single-pass scan, no sort)
 */

#ifndef SCATTER_ND_UPDATE_V2_SCAN_H
#define SCATTER_ND_UPDATE_V2_SCAN_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdateV2 {

inline constexpr uint64_t MAX_DIM_NUM = 8;
inline constexpr uint64_t SMALL_EXACT_DEDUP_ROWS = 16;

/*!
 * \class ScatterNdUpdateV2ScanKernel
 */
template<typename T, typename IdxRawT>
class ScatterNdUpdateV2ScanKernel {
public:
    __aicore__ inline ScatterNdUpdateV2ScanKernel() = delete;
    __aicore__ inline ScatterNdUpdateV2ScanKernel(
        GM_ADDR indices, GM_ADDR updates, GM_ADDR output,
        const ScatterNdUpdateV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(indices, updates, output);
    }

    __aicore__ inline void InitParams(const ScatterNdUpdateV2TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        CalcBlockDistribution(blockIdx_, tiling.scatterTiling.frontNum, tiling.scatterTiling.frontRow,
                              tiling.scatterTiling.tailRow, computeRow_, start_);
        end_ = start_ + computeRow_;

        totalIndexRow_ = tiling.linearIndexTiling.blockNum * tiling.linearIndexTiling.blockLength
                         + tiling.linearIndexTiling.blockRemainLength;
        indexDim_ = tiling.linearIndexTiling.indexDim;
        for (uint64_t i = 0; i < indexDim_; ++i) {
            indicesMask_[i] = tiling.linearIndexTiling.indicesMask[i];
        }

        scatterLength_ = tiling.scatterTiling.scatterLength;
        scatterTileNum_ = tiling.scatterTiling.scatterTileNum;
        scatterTileLength_ = tiling.scatterTiling.scatterTileLength;
        scatterTileTail_ = tiling.scatterTiling.scatterTileTail;
        scatterTileAlignLength_ = tiling.scatterTiling.scatterTileAlignLength;
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        constexpr uint64_t kIdxElemBytes = sizeof(IdxRawT);
        rowBufElems_ = (totalIndexRow_ + ALIGN_NUM - 1) & ~(ALIGN_NUM - 1);
        if (rowBufElems_ == 0) {
            rowBufElems_ = 1;
        }
        uint64_t indexBufBytes = (totalIndexRow_ * indexDim_ * kIdxElemBytes + ALIGNED_SIZE - 1)
                                 & ~(ALIGNED_SIZE - 1);
        if (indexBufBytes == 0) {
            indexBufBytes = ALIGNED_SIZE;
        }
        uint64_t dstBufBytes = rowBufElems_ * sizeof(int);
        uint64_t tmpBufBytes = rowBufElems_ * sizeof(int);
        uint64_t rangeBufBytes = rowBufElems_ * sizeof(int);
        uint64_t updBufBytes = 2 * scatterTileAlignLength_ * sizeof(T);
        if (updBufBytes == 0) {
            updBufBytes = 2 * ALIGNED_SIZE;
        }

        pipe.InitBuffer(indexBuf_, indexBufBytes);
        pipe.InitBuffer(dstBuf_, dstBufBytes);
        pipe.InitBuffer(tmpBuf_, tmpBufBytes);
        pipe.InitBuffer(rangeBuf_, rangeBufBytes);
        pipe.InitBuffer(updBuf_, updBufBytes);

        evt2Id_[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        evt2Id_[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        evt3Id_[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        evt3Id_[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());

        indicesLocal_ = indexBuf_.Get<int>();
        if constexpr (std::is_same_v<IdxRawT, int64_t>) {
            indicesInt64Local_ = indexBuf_.Get<int64_t>();
        }
        dstLocal_ = dstBuf_.Get<int>();
        tmpLocal_ = tmpBuf_.Get<int>();
        rangeLocal_ = rangeBuf_.Get<int>();
        updLocal_ = updBuf_.Get<T>();
    }

    __aicore__ inline void SetGmAddr(GM_ADDR indices, GM_ADDR updates, GM_ADDR output)
    {
        if constexpr (std::is_same_v<IdxRawT, int64_t>) {
            indicesGmInt64_.SetGlobalBuffer((__gm__ int64_t*)indices);
        } else {
            indicesGm_.SetGlobalBuffer((__gm__ int*)indices);
        }
        updatesGm_.SetGlobalBuffer((__gm__ T*)updates);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
    }

    __aicore__ inline void Process()
    {
        CopyIndicesIn();
        if constexpr (std::is_same_v<IdxRawT, int64_t>) {
            CastToInt32();
        }
        ComputeAllDst();
        ScanAndUpdate();
        ReleaseEventIDs();
    }

private:
    __aicore__ inline void CopyIndicesIn()
    {
        uint64_t copyBytes = totalIndexRow_ * indexDim_ * sizeof(IdxRawT);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyBytes), 0, 0, 0};
        if constexpr (std::is_same_v<IdxRawT, int64_t>) {
            DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
            DataCopyPad(indicesInt64Local_, indicesGmInt64_[0], copyParams, padParams);
        } else {
            DataCopyPadExtParams<int> padParams{true, 0, 0, 0};
            DataCopyPad(indicesLocal_, indicesGm_[0], copyParams, padParams);
        }
        PipeMte2ToS();
    }

    __aicore__ inline void CastToInt32()
    {
        uint64_t totalElements = totalIndexRow_ * indexDim_;
        Cast(indicesLocal_, indicesInt64Local_, RoundMode::CAST_NONE, totalElements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeAllDst()
    {
        int32_t malValue = static_cast<int32_t>(indexDim_ * sizeof(int));
        Duplicate<int>(dstLocal_, 0, totalIndexRow_);
        CreateVecIndex(rangeLocal_, (int)0, totalIndexRow_);
        PipeBarrier<PIPE_V>();
        Muls(rangeLocal_, rangeLocal_, malValue, totalIndexRow_);
        PipeBarrier<PIPE_V>();
        for (uint64_t i = 0; i < indexDim_; ++i) {
            if (i != 0) {
                Adds(rangeLocal_, rangeLocal_, (int)(sizeof(int)), totalIndexRow_);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<uint32_t> rangeLocalCasted = rangeLocal_.ReinterpretCast<uint32_t>();
            Gather(tmpLocal_, indicesLocal_, rangeLocalCasted, (uint32_t)0, (uint32_t)totalIndexRow_);
            PipeBarrier<PIPE_V>();
            Muls(tmpLocal_, tmpLocal_, (int)indicesMask_[i], totalIndexRow_);
            PipeBarrier<PIPE_V>();
            Add(dstLocal_, dstLocal_, tmpLocal_, totalIndexRow_);
            PipeBarrier<PIPE_V>();
        }
        PipeVToS();
    }

    __aicore__ inline void PipeVToS()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }

    __aicore__ inline void ScanAndUpdate()
    {
        event_t evt2[2] = {evt2Id_[0], evt2Id_[1]};
        event_t evt3[2] = {evt3Id_[0], evt3Id_[1]};
        bool havePrev = false;
        uint64_t prevSlot = 0;
        uint64_t prevDst = 0;
        uint64_t prevTileIdx = 0;
        uint64_t prevTileLen = 0;
        uint64_t unitCount = 0;
        bool evt3Pending[2] = {false, false};

        const bool useHash = totalIndexRow_ > SMALL_EXACT_DEDUP_ROWS;
        uint64_t hashCapacity = 0;
        uint64_t hashMask = 0;
        if (useHash) {
            Duplicate<int>(tmpLocal_, -1, static_cast<uint32_t>(rowBufElems_));
            Duplicate<int>(rangeLocal_, -1, static_cast<uint32_t>(rowBufElems_));
            PipeVToS();
            hashCapacity = 1;
            const uint64_t hashStorage = 2 * rowBufElems_;
            while ((hashCapacity << 1) <= hashStorage) {
                hashCapacity <<= 1;
            }
            hashMask = hashCapacity - 1;
        }

        for (uint64_t rowEnd = totalIndexRow_; rowEnd > 0; --rowEnd) {
            const uint64_t r = rowEnd - 1;
            int64_t dstVal = static_cast<int64_t>(dstLocal_.GetValue(r));
            if (dstVal < (int64_t)start_ || dstVal >= (int64_t)end_) {
                continue;
            }

            bool isLastWrite = true;
            if (useHash) {
                uint64_t bucket =
                    (static_cast<uint64_t>(static_cast<uint32_t>(dstVal)) * 2654435761ULL) & hashMask;
                isLastWrite = false;
                for (uint64_t probe = 0; probe < hashCapacity; ++probe) {
                    int32_t key;
                    if (bucket < rowBufElems_) {
                        key = tmpLocal_.GetValue(bucket);
                    } else {
                        key = rangeLocal_.GetValue(bucket - rowBufElems_);
                    }
                    if (key == static_cast<int32_t>(dstVal)) {
                        break;
                    }
                    if (key == -1) {
                        if (bucket < rowBufElems_) {
                            tmpLocal_.SetValue(bucket, static_cast<int32_t>(dstVal));
                        } else {
                            rangeLocal_.SetValue(bucket - rowBufElems_, static_cast<int32_t>(dstVal));
                        }
                        isLastWrite = true;
                        break;
                    }
                    bucket = (bucket + 1) & hashMask;
                }
            } else {
                for (uint64_t later = r + 1; later < totalIndexRow_; ++later) {
                    if (dstLocal_.GetValue(later) == static_cast<int32_t>(dstVal)) {
                        isLastWrite = false;
                        break;
                    }
                }
            }
            if (!isLastWrite) {
                continue;
            }
            for (uint64_t tileIdx = 0; tileIdx < scatterTileNum_; ++tileIdx) {
                uint64_t tileLen = (tileIdx == scatterTileNum_ - 1) ? scatterTileTail_ : scatterTileLength_;
                uint64_t slot = unitCount & 1;
                ++unitCount;
                if (unitCount >= 3) {
                    WaitFlag<HardEvent::MTE3_MTE2>(evt3[slot]);
                    evt3Pending[slot] = false;
                }
                CopyUpdateIn(updLocal_, r, tileIdx, tileLen, slot, evt2[slot]);
                if (havePrev) {
                    WaitFlag<HardEvent::MTE2_MTE3>(evt2[prevSlot]);
                    CopyOut(updLocal_, prevDst, prevTileIdx, prevTileLen, prevSlot, evt3[prevSlot]);
                    evt3Pending[prevSlot] = true;
                }
                havePrev = true;
                prevSlot = slot;
                prevDst = static_cast<uint64_t>(dstVal);
                prevTileIdx = tileIdx;
                prevTileLen = tileLen;
            }
        }
        if (havePrev) {
            WaitFlag<HardEvent::MTE2_MTE3>(evt2[prevSlot]);
            CopyOut(updLocal_, prevDst, prevTileIdx, prevTileLen, prevSlot, evt3[prevSlot]);
            evt3Pending[prevSlot] = true;
            PipeMte3ToS();
        }
        bool needMte2Drain = false;
        for (uint64_t slot = 0; slot < 2; ++slot) {
            if (evt3Pending[slot]) {
                WaitFlag<HardEvent::MTE3_MTE2>(evt3[slot]);
                needMte2Drain = true;
            }
        }
        if (needMte2Drain) {
            PipeMte2ToS();
        }
    }

    __aicore__ inline void CopyUpdateIn(LocalTensor<T>& updBase, uint64_t rowIdx, uint64_t tileIdx,
                                        uint64_t tileLength, uint64_t slot, event_t& evt)
    {
        uint64_t gmOffset = rowIdx * scatterLength_ + tileIdx * scatterTileLength_;
        uint64_t ubOffset = slot * scatterTileAlignLength_;
        DataCopyExtParams updateCopyParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(updBase[ubOffset], updatesGm_[gmOffset], updateCopyParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(evt);
    }

    __aicore__ inline void CopyOut(LocalTensor<T>& updBase, uint64_t dstBase, uint64_t tileIdx,
                                   uint64_t tileLength, uint64_t slot, event_t& evt)
    {
        uint64_t ubOffset = slot * scatterTileAlignLength_;
        uint64_t outOffset = dstBase + tileIdx * scatterTileLength_;
        DataCopyExtParams outParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outOffset], updBase[ubOffset], outParams);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
    }

    __aicore__ inline void ReleaseEventIDs()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(evt2Id_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(evt2Id_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evt3Id_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evt3Id_[1]);
    }

private:
    GlobalTensor<int> indicesGm_;
    GlobalTensor<int64_t> indicesGmInt64_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> outputGm_;

    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> dstBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> rangeBuf_;
    TBuf<TPosition::VECCALC> updBuf_;

    LocalTensor<int> indicesLocal_;
    LocalTensor<int64_t> indicesInt64Local_;
    LocalTensor<int> dstLocal_;
    LocalTensor<int> tmpLocal_;
    LocalTensor<int> rangeLocal_;
    LocalTensor<T> updLocal_;

    event_t evt2Id_[2];
    event_t evt3Id_[2];

    uint64_t blockIdx_;
    uint64_t computeRow_;
    uint64_t start_;
    uint64_t end_;
    uint64_t totalIndexRow_;
    uint64_t rowBufElems_;
    uint64_t indexDim_;
    uint64_t indicesMask_[MAX_DIM_NUM];

    uint64_t scatterLength_;
    uint64_t scatterTileNum_;
    uint64_t scatterTileLength_;
    uint64_t scatterTileTail_;
    uint64_t scatterTileAlignLength_;
};

}  // namespace ScatterNdUpdateV2

#endif  // SCATTER_ND_UPDATE_V2_SCAN_H
