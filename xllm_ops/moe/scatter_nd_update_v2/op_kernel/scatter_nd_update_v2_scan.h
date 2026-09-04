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
 *
 * 单趟扫描核：不做 LinearIndex 预排序，因而免去 SyncAll、sortWorkspace 与
 * 二分查找。每核一次性载入全部 indices 算出 dst，再按行序递增扫描，只写落在
 * 本核区间 [start_, end_) 的行。
 *
 * 两个不变量：
 *   - 核区间互斥，故无需原子写；同一 dst 的重复行由"后写覆盖先写"保证语义，
 *     这要求扫描必须按行序递增，不能重排。
 *   - evt2/evt3 必须用 AllocEventID 为每个双缓冲槽各占一个独立事件 ID。
 *     TPipe::FetchEventID 在池空闲时恒返回 0 且不占用，逐次 Fetch 会让两个槽
 *     的事件 ID 塌缩为同一个，SetFlag/WaitFlag 跨槽错配 → 偶发死锁。
 *     GM→UB 的 MTE2 生产数据、UB→GM 的 MTE3 消费数据，RAW 使用 MTE2_MTE3；
 *     MTE3 读完 UB 后 MTE2 才能复用槽，WAR 使用 MTE3_MTE2。
 *   - 重复 dst 必须只写最后一行。扫描从最后一行向前；小 R 精确比较后缀，
 *     大 R 用 tmp/range 的空闲 UB 空间组成开放寻址表。每个 dst 首次命中即为
 *     golden 的最后写。这与原 sort 路径的去重语义一致，也避免对同一 GM
 *     尾块反复发 MTE3。
 */

#ifndef SCATTER_ND_UPDATE_V2_SCAN_H
#define SCATTER_ND_UPDATE_V2_SCAN_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdateV2 {

// 与 op_host tiling（op_host/scatter_nd_update_v2_tiling.h MAX_DIM_NUM=8）同值；
// kernel 编译单元内命名空间级定义，避免依赖框架生成头是否注入同名宏。
inline constexpr uint64_t MAX_DIM_NUM = 8;
// 小 R 的三角比较比两次 Vector Duplicate + V->S + hash 更轻；阈值覆盖性能集
// 中除 17-row C07 外的全部 case，且最坏只做 120 次精确 UB 标量比较。
inline constexpr uint64_t SMALL_EXACT_DEDUP_ROWS = 16;

/*!
 * \class ScatterNdUpdateV2ScanKernel
 * \brief REG-SCAN 单趟扫描核
 * \tparam T          var/updates 元素类型（half / float）
 * \tparam IdxRawT    indices 原始元素类型（int / int64_t）
 *
 * IdxRawT=int 时 indices 以 int32 直接载入；IdxRawT=int64_t 时先载入 int64 再
 * Cast 收窄为 int32（totalLength ≤ INT32_MAX，语义与基线 LinearIndex 一致）。
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
        // 分区复用 Tiling4Scatter frontNum/frontRow/tailRow（blockDim=min(coreNum, max(1,totalLength))）
        CalcBlockDistribution(blockIdx_, tiling.scatterTiling.frontNum, tiling.scatterTiling.frontRow,
                              tiling.scatterTiling.tailRow, computeRow_, start_);
        end_ = start_ + computeRow_;

        // R = 总索引行数（由线性索引 tiling 字段还原：blockNum*blockLength + blockRemainLength，
        // R≤2048 时 blockLength_ 恒 ≥ R，故总行数 == blockRemainLength）
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
        constexpr uint64_t kIdxElemBytes = sizeof(IdxRawT);   // 4 (int32) 或 8 (int64)
        rowBufElems_ = (totalIndexRow_ + ALIGN_NUM - 1) & ~(ALIGN_NUM - 1);   // 8 元素对齐
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
        // updBuf ×2（双缓冲）：每槽 scatterTileAlignLength_ 元素
        uint64_t updBufBytes = 2 * scatterTileAlignLength_ * sizeof(T);
        if (updBufBytes == 0) {
            updBufBytes = 2 * ALIGNED_SIZE;
        }

        pipe.InitBuffer(indexBuf_, indexBufBytes);
        pipe.InitBuffer(dstBuf_, dstBufBytes);
        pipe.InitBuffer(tmpBuf_, tmpBufBytes);
        pipe.InitBuffer(rangeBuf_, rangeBufBytes);
        pipe.InitBuffer(updBuf_, updBufBytes);

        // 每槽独立事件 ID（见文件头"事件方案"注释）：池初始全空闲 ⇒ AllocEventID 依次返回 0、1。
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
        // 事件 ID 释放（文档要求 AllocEventID/ReleaseEventID 成对）；ScanAndUpdate
        // 已等待最终 MTE3 写回，并精确消费每个未等待的 MTE3_MTE2 flag。
        ReleaseEventIDs();
    }

private:
    /*! 1) 一次性读入全部索引行（R×indexDim） */
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

    /*! int64 → int32 原位收窄（totalLength≤INT32_MAX；与基线 CastToInt32 同写法） */
    __aicore__ inline void CastToInt32()
    {
        uint64_t totalElements = totalIndexRow_ * indexDim_;
        Cast(indicesLocal_, indicesInt64Local_, RoundMode::CAST_NONE, totalElements);
        PipeBarrier<PIPE_V>();
    }

    /*! 2) 向量化对全部 R 行计算 dst（复用 Compute4LinearIndex 链，去 CastToInt32 后排序链） */
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
        // dst 由 V 管道写入，扫描阶段用 S 管道 GetValue 读取 → 显式 V→S 同步
        PipeVToS();
    }

    /*! V→S 显式同步（V 管完成退出后 S 才读 UB 中的 dst） */
    __aicore__ inline void PipeVToS()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }

    /*! 3)+4) 逆序扫描 + 最后写去重 + 分区写 */
    __aicore__ inline void ScanAndUpdate()
    {
        // updBuf ×2 双缓冲（lag-1 ping-pong）：
        //   - 命中行 r 先发 MTE2（GM→UB(slot)），不阻塞；
        //   - 等到下一命中行出现时，再对前一命中行发 MTE3（UB(prevSlot)→GM）；
        //   - slot 复用前等待该 slot 上次 MTE3 完成（WAR 保护）。
        // 事件 ID：InitBuffers 已按 slot 占用独立 ID（MTE2_MTE3×2 / MTE3_MTE2×2），
        // evt2[slot]/evt3[slot] 与 slot 一一对应，Raw/WAR 等待不会跨 slot 交叉匹配。
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
            // ComputeAllDst 后 tmp/range 不再有活跃数据，复用为 seen-dst hash table。
            // 容量取 <= 2*align8(R) 的最大 2 次幂，必定 >= R；-1 为合法 dst 外的空桶。
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

        // 逆序扫描：某 dst 第一次插入时对应顺序 golden 的最后一行。
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
                        break;  // 该 dst 已有更后行，当前行必须跳过
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
                // 小 R 精确检查当前行之后的所有 dst；无采样、Bloom 或碰撞误判。
                // 逆序首个命中仍是 max(row)，所以与 hash 路径语义完全相同。
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
                    // slot 即将被第 2 次复用：等待该 slot 上次 MTE3 完成（WAR）
                    WaitFlag<HardEvent::MTE3_MTE2>(evt3[slot]);
                    evt3Pending[slot] = false;
                }
                CopyUpdateIn(updLocal_, r, tileIdx, tileLen, slot, evt2[slot]);
                if (havePrev) {
                    // 前一命中单元的 MTE2 已就绪，发出 MTE3 写回
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
            // flush 最后一个命中单元
            WaitFlag<HardEvent::MTE2_MTE3>(evt2[prevSlot]);
            CopyOut(updLocal_, prevDst, prevTileIdx, prevTileLen, prevSlot, evt3[prevSlot]);
            evt3Pending[prevSlot] = true;
            PipeMte3ToS();
        }
        // PipeMte3ToS 只保证写回完成，不会消费每槽 MTE3_MTE2 的硬件 flag。
        // ReleaseEventID 前必须用相同 HardEvent+eventID 精确 drain 所有未消费 WAR。
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

    /*! 写路径-a：DataCopyPad GM→UB + SetFlag<MTE2_MTE3>（等待外置；
     *  evt 为 InitBuffers 按 slot 预占的独立事件 ID，不再 Fetch） */
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

    /*! 写路径-b：DataCopyPad UB→GM + SetFlag<MTE3_MTE2>（槽复用 WAR） */
    __aicore__ inline void CopyOut(LocalTensor<T>& updBase, uint64_t dstBase, uint64_t tileIdx,
                                   uint64_t tileLength, uint64_t slot, event_t& evt)
    {
        uint64_t ubOffset = slot * scatterTileAlignLength_;
        uint64_t outOffset = dstBase + tileIdx * scatterTileLength_;
        DataCopyExtParams outParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outOffset], updBase[ubOffset], outParams);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
    }

    /*! 释放 InitBuffers 占用的每槽事件 ID（ScanAndUpdate 结束后调用） */
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

    event_t evt2Id_[2];   // MTE2_MTE3 每槽独立事件 ID（InitBuffers 预占）
    event_t evt3Id_[2];   // MTE3_MTE2 每槽独立事件 ID（InitBuffers 预占）

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
