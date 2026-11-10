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

/*!
 * \file split_qkv_norm_rope.h
 * \brief SplitQKV + QK-RMSNorm + RoPE 融合算子的 kernel 实现（Ascend 950 / AIV）。
 */

#ifndef SPLIT_QKV_NORM_ROPE_KERNEL_H_
#define SPLIT_QKV_NORM_ROPE_KERNEL_H_

#include "kernel_operator.h"

namespace kernels {
using namespace AscendC;

constexpr uint32_t kHeadDim = 128;              // head_dim 固定 128
constexpr uint32_t kHalfDim = kHeadDim / 2;     // 64，RoPE half-split 分界
constexpr uint32_t kMaxHeads = 16;              // sumsq UB 上界：Nq+Nk <= kMaxHeads
constexpr uint32_t kTokenTile = 8;              // 每 tile 的 token 行数；8 行是可整块搬送的最小 32B 对齐周期（行宽 2572B=4×643，nTok 需为 8 的倍数），也是 UB 预算安全点；16 行经核算结构性放不下
// 每 tile 的 sumsq 个数 = kTokenTile × (Nq+Nk)，scale 链按 64 lane 向上取整分段

constexpr uint32_t kFlagGqaExpand = 1u << 0;    // 1：k/v 扩张到 Nq 头
constexpr uint32_t kFlagLayoutBNSD = 1u << 1;   // 1：[B,N,S,D]；0：[B,S,N*D]
constexpr uint32_t kFlagKvInterleaved = 1u << 2;  // 1：输入行内 KV 按 head 交错 [K_i,V_i]；0：[全部 K | 全部 V]

// ---------------------------------------------------------------------------
// SqnrNormRopeVfPack：逐行 QK-RMSNorm + 半分化 RoPE（只处理 q/k 头；v 由外层 UB→UB 直通拷贝）
//   outT      : T 输出 tile，[kTokenTile, numHeadTotal*128]，行距 outStride
//   xb        : T qkv tile，[kTokenTile, padStride]，行距 xStride
//   cosT/sinT : T cos/sin，[kTokenTile, ropeStride]，只用前 64 维（NeoX 半维）
//   gammaQ/K  : fp32 [128]
//   sumsq     : fp32 workspace，[kTokenTile, Nq+Nk]，行距 = Nq+Nk（容量按 kMaxHeads 上界分配）
// 固定处理 kTokenTile 行；末 tile 的脏行不会被拷出。
//
// 实现要点（当前实现的约束）：
//   1) q/k 与 cos/sin 全部用 LoadAlign<T, DIST_UNPACK_B16> 拆到 32 位 lane，紧跟一条寄存器内
//      Cast<float, T, layoutZMrgZ>；无整 tile 的 cast-in 预处理与 fp32 tile 副本（gamma 仍是 fp32）。
//   2) γ 折进 cos/sin（a=γ1·cos、c=γ1·sin、d=γ2·cos、b=-γ2·sin）；折进后的 4 个常量无法在
//      运行期选择（VF 内禁止运行期分支、寄存器不可运行期索引），故 q 组/k 组循环体各实例化一次，
//      仍在同一 __simd_vf__、同一寄存器预算内。
//   3) UB 行距必须 32B 对齐，否则 LoadAlign/StoreAlign 读错位（padStride 的由来）。
//   4) pass1 每 (t,h)：两半各一次取数+Cast，Mul 起手 + 一次 MulAddDst 累加 + ReduceSum + 单元素写回；
//      scale 链每 tile 一次（segs 段 64-lane，Muls/Adds/Sqrt/Div），per-head 只留 1 次 DIST_BRC 广播；
//      RoPE 每半 1 次 Mul + 1 次 MulAddDst。
//   5) 输出在寄存器内 Cast 成 T 后 StoreAlign DIST_PACK_B32 直写（pack-store），无整块 cast-out。
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void SqnrNormRopeVfPack(
    __ubuf__ T *outT, __ubuf__ T *xb, __ubuf__ T *cosT, __ubuf__ T *sinT,
    __ubuf__ float *gammaQ, __ubuf__ float *gammaK, __ubuf__ float *sumsq,
    uint16_t numHeadQ, uint16_t numHeadK, uint16_t xStride, uint16_t outStride, uint16_t ropeStride,
    uint16_t kvHeadStep, float eps, float recip)
{
    using namespace AscendC::Reg;
    RegTensor<float> y1;
    RegTensor<float> y2;
    RegTensor<float> t1;
    RegTensor<float> t2;
    RegTensor<float> cr;
    RegTensor<float> sr;
    RegTensor<float> sn;
    RegTensor<float> sg;
    RegTensor<float> one;
    RegTensor<float> acc;
    RegTensor<float> gq1;
    RegTensor<float> gq2;
    RegTensor<float> gk1;
    RegTensor<float> gk2;
    RegTensor<float> ca;
    RegTensor<float> cc;
    RegTensor<float> cd;
    RegTensor<float> cb;
    RegTensor<T> tv;
    RegTensor<T> pk;
    MaskReg mFull = CreateMask<float, MaskPattern::ALL>();  // 64 lanes = 256B / 4B
    MaskReg mOne = CreateMask<float, MaskPattern::VL1>();

    const uint16_t qkHeads = static_cast<uint16_t>(numHeadQ + numHeadK);
    // 输入行内 KV 区起点（k/v 区）；q 头从行首按 D 连续，k 头按 kvHeadStep 步进
    const uint32_t kvHeadBase = static_cast<uint32_t>(numHeadQ) * kHeadDim;
    // sumsq 段数：每段 64 lane，向上取整覆盖 kTokenTile*(Nq+Nk) 个有效元素；
    // 末段多出的 lane 只写回 sumsq，pass2 只按 (t,h) 单点广播读取，不回读这些 lane。
    constexpr uint16_t kSeg = 64;
    const uint16_t segs = static_cast<uint16_t>((kTokenTile * qkHeads + kSeg - 1) / kSeg);

    // γq / γk 每 tile 各装载 1 次
    LoadAlign(gq1, gammaQ);
    LoadAlign(gq2, gammaQ + kHalfDim);
    LoadAlign(gk1, gammaK);
    LoadAlign(gk2, gammaK + kHalfDim);

    // pass 1：每 (token, head) 的 sumsq（q/k 直接从 T 载入）
    for (uint16_t t = 0; t < static_cast<uint16_t>(kTokenTile); ++t) {
        __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
        for (uint16_t h = 0; h < numHeadQ; ++h) {
            __ubuf__ T *src = row + static_cast<uint32_t>(h) * kHeadDim;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
            Mul(acc, y1, y1, mFull);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + kHalfDim);
            Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
            MulAddDst(acc, y2, y2, mFull);
            ReduceSum(acc, acc, mFull);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                sumsq + static_cast<uint32_t>(t) * qkHeads + h, acc, mOne);
        }
        // k 头：块状布局 kvHeadStep = D（= 连续），KV 交错布局 kvHeadStep = 2D（跨过 v）
        for (uint16_t h = 0; h < numHeadK; ++h) {
            __ubuf__ T *src = row + kvHeadBase + static_cast<uint32_t>(h) * kvHeadStep;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
            Mul(acc, y1, y1, mFull);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + kHalfDim);
            Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
            MulAddDst(acc, y2, y2, mFull);
            ReduceSum(acc, acc, mFull);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                sumsq + static_cast<uint32_t>(t) * qkHeads + numHeadQ + h, acc, mOne);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // 每 tile 的 scale 链：segs 段 × 64 lane，原地写回
    Duplicate(one, 1.0f, mFull);
    for (uint16_t seg = 0; seg < segs; ++seg) {
        __ubuf__ float *sp = sumsq + static_cast<uint32_t>(seg) * kSeg;
        LoadAlign(acc, sp);
        Muls(acc, acc, recip, mFull);
        Adds(acc, acc, eps, mFull);
        Sqrt(acc, acc, mFull);
        Div(acc, one, acc, mFull);
        StoreAlign(sp, acc, mFull);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // pass 2：norm + rope，结果直接 pack 成 T 写进输出 tile
    for (uint16_t t = 0; t < static_cast<uint16_t>(kTokenTile); ++t) {
        __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
        __ubuf__ T *orow = outT + static_cast<uint32_t>(t) * outStride;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, cosT + static_cast<uint32_t>(t) * ropeStride);
        Cast<float, T, AscendC::layoutZMrgZ>(cr, tv, mFull);
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, sinT + static_cast<uint32_t>(t) * ropeStride);
        Cast<float, T, AscendC::layoutZMrgZ>(sr, tv, mFull);
        Muls(sn, sr, -1.0f, mFull);

        // q 组：γq 折进 cos/sin
        Mul(ca, gq1, cr, mFull);
        Mul(cc, gq1, sr, mFull);
        Mul(cd, gq2, cr, mFull);
        Mul(cb, gq2, sn, mFull);
        for (uint16_t h = 0; h < numHeadQ; ++h) {
            __ubuf__ T *src = row + static_cast<uint32_t>(h) * kHeadDim;
            __ubuf__ T *dst = orow + static_cast<uint32_t>(h) * kHeadDim;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumsq + static_cast<uint32_t>(t) * qkHeads + h);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
            Mul(y1, y1, sg, mFull);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + kHalfDim);
            Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
            Mul(y2, y2, sg, mFull);
            Mul(t1, y1, ca, mFull);
            MulAddDst(t1, y2, cb, mFull);
            Mul(t2, y2, cd, mFull);
            MulAddDst(t2, y1, cc, mFull);
            Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t1, mFull);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, pk, mFull);
            Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t2, mFull);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + kHalfDim, pk, mFull);
        }

        // k 组：γk 折进 cos/sin
        Mul(ca, gk1, cr, mFull);
        Mul(cc, gk1, sr, mFull);
        Mul(cd, gk2, cr, mFull);
        Mul(cb, gk2, sn, mFull);
        for (uint16_t h = 0; h < numHeadK; ++h) {
            uint32_t hh = static_cast<uint32_t>(numHeadQ) + h;
            __ubuf__ T *src = row + kvHeadBase + static_cast<uint32_t>(h) * kvHeadStep;
            __ubuf__ T *dst = orow + hh * kHeadDim;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumsq + static_cast<uint32_t>(t) * qkHeads + hh);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
            Mul(y1, y1, sg, mFull);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + kHalfDim);
            Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
            Mul(y2, y2, sg, mFull);
            Mul(t1, y1, ca, mFull);
            MulAddDst(t1, y2, cb, mFull);
            Mul(t2, y2, cd, mFull);
            MulAddDst(t2, y1, cc, mFull);
            Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t1, mFull);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, pk, mFull);
            Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t2, mFull);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + kHalfDim, pk, mFull);
        }
    }
}


template <typename T>
class KernelSplitQkvNormRope {
public:
    __aicore__ inline KernelSplitQkvNormRope() {}

    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR qGamma, GM_ADDR kGamma, GM_ADDR cosGm,
                                GM_ADDR sinGm, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR vOut,
                                TPipe *pipe, const SplitQkvNormRopeTilingData *tiling)
    {
        pipe_ = pipe;
        seqLen_ = tiling->seqLen;
        numHeadQ_ = tiling->numHeadQ;
        numHeadK_ = tiling->numHeadK;
        numHeadV_ = tiling->numHeadV;
        numHeadTotal_ = numHeadQ_ + numHeadK_ + numHeadV_;
        qkvStride_ = tiling->qkvStride;
        padStride_ = tiling->padStride;
        ropeStride_ = tiling->ropeStride;
        blockDim_ = tiling->blockDim;
        tokenPerCore_ = tiling->tokenPerCore;
        lastBlkTokens_ = tiling->lastBlockTokenNum;
        eps_ = tiling->eps;
        recip_ = tiling->reciprocal;

        gqaExpand_ = (tiling->flags & kFlagGqaExpand) != 0;
        layoutBNSD_ = (tiling->flags & kFlagLayoutBNSD) != 0;
        kvInterleaved_ = (tiling->flags & kFlagKvInterleaved) != 0;
        kvOutHeads_ = gqaExpand_ ? numHeadQ_ : numHeadK_;
        gqaGroup_ = (numHeadK_ == 0) ? 1 : (numHeadQ_ / numHeadK_);

        qkvGm_.SetGlobalBuffer((__gm__ T *)qkv, tiling->totalTokens * qkvStride_);
        cosGm_.SetGlobalBuffer((__gm__ T *)cosGm, tiling->totalTokens * ropeStride_);
        sinGm_.SetGlobalBuffer((__gm__ T *)sinGm, tiling->totalTokens * ropeStride_);
        gammaQGm_.SetGlobalBuffer((__gm__ T *)qGamma, kHeadDim);
        gammaKGm_.SetGlobalBuffer((__gm__ T *)kGamma, kHeadDim);
        qGm_.SetGlobalBuffer((__gm__ T *)qOut, tiling->totalTokens * numHeadQ_ * kHeadDim);
        kGm_.SetGlobalBuffer((__gm__ T *)kOut, tiling->totalTokens * kvOutHeads_ * kHeadDim);
        vGm_.SetGlobalBuffer((__gm__ T *)vOut, tiling->totalTokens * kvOutHeads_ * kHeadDim);

        // 深度 2：支持「下一个 tile 预取」+「上一个 tile 延后写出」
        pipe_->InitBuffer(inQkv_, 2, kTokenTile * padStride_ * sizeof(T));
        pipe_->InitBuffer(inCos_, 2, kTokenTile * ropeStride_ * sizeof(T));
        pipe_->InitBuffer(inSin_, 2, kTokenTile * ropeStride_ * sizeof(T));
        pipe_->InitBuffer(inGamma_, 2, kHeadDim * sizeof(T));
        pipe_->InitBuffer(outTile_, 2, kTokenTile * numHeadTotal_ * kHeadDim * sizeof(T));

        pipe_->InitBuffer(bufGammaQ_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(bufGammaK_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(bufSum_, kTokenTile * kMaxHeads * sizeof(float));

        // 本核 token 区间与 tile 数（供三级流水使用）
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        blkStart_ = blockIdx * tokenPerCore_;
        blkTokens_ = (blockIdx == blockDim_ - 1) ? lastBlkTokens_ : tokenPerCore_;
        tileNum_ = (blkTokens_ + kTokenTile - 1) / kTokenTile;

        // 只有一条路径：sumsq 段数按 ceil(kTokenTile*(Nq+Nk)/64) 运行期算，任意 Nq+Nk <= kMaxHeads 均可覆盖。
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();

        // 三级流水：先发下一个 tile 的 DMA，再算当前 tile，最后补上一个 tile 的写出
        CopyInTile(0);
        for (uint32_t k = 0; k < tileNum_; ++k) {
            if (k + 1 < tileNum_) {
                CopyInTile(k + 1);
            }
            ComputeTile(k);
            if (k > 0) {
                CopyOutTile(k - 1);
            }
        }
        CopyOutTile(tileNum_ - 1);
    }

private:
    __aicore__ inline uint32_t TileTokens(uint32_t k) const
    {
        const uint32_t left = blkTokens_ - k * kTokenTile;
        return left > kTokenTile ? kTokenTile : left;
    }

    __aicore__ inline void CopyInGamma()
    {
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        LocalTensor<T> gq = inGamma_.AllocTensor<T>();
        DataCopyPad(gq, gammaQGm_, DataCopyExtParams{1, static_cast<uint32_t>(kHeadDim * sizeof(T)), 0, 0, 0}, pad);
        inGamma_.EnQue(gq);
        LocalTensor<T> gk = inGamma_.AllocTensor<T>();
        DataCopyPad(gk, gammaKGm_, DataCopyExtParams{1, static_cast<uint32_t>(kHeadDim * sizeof(T)), 0, 0, 0}, pad);
        inGamma_.EnQue(gk);

        gq = inGamma_.DeQue<T>();
        gk = inGamma_.DeQue<T>();
        LocalTensor<float> gqf = bufGammaQ_.Get<float>();
        LocalTensor<float> gkf = bufGammaK_.Get<float>();
        Cast(gqf, gq, RoundMode::CAST_NONE, kHeadDim);
        Cast(gkf, gk, RoundMode::CAST_NONE, kHeadDim);
        inGamma_.FreeTensor(gq);
        inGamma_.FreeTensor(gk);
        PipeBarrier<PIPE_V>();
    }

    // 只发 DMA，不等：深度 2 的队列允许它和上一个 tile 的计算并行
    __aicore__ inline void CopyInTile(uint32_t k)
    {
        const uint32_t t0 = blkStart_ + k * kTokenTile;
        const uint32_t nTok = TileTokens(k);
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // 每 token 一行单独搬：GM 行距（qkvStride_*sizeof(T)）不保证 32B 对齐，UB 侧落到 32B 对齐的 padStride 行距
        LocalTensor<T> xb = inQkv_.AllocTensor<T>();
        for (uint32_t t = 0; t < nTok; ++t) {
            DataCopyPad(xb[t * padStride_], qkvGm_[(t0 + t) * qkvStride_],
                DataCopyExtParams{1, static_cast<uint32_t>(qkvStride_ * sizeof(T)), 0, 0, 0}, pad);
        }
        inQkv_.EnQue(xb);

        LocalTensor<T> cb = inCos_.AllocTensor<T>();
        DataCopyPad(cb, cosGm_[t0 * ropeStride_],
            DataCopyExtParams{1, static_cast<uint32_t>(nTok * ropeStride_ * sizeof(T)), 0, 0, 0}, pad);
        inCos_.EnQue(cb);

        LocalTensor<T> sb = inSin_.AllocTensor<T>();
        DataCopyPad(sb, sinGm_[t0 * ropeStride_],
            DataCopyExtParams{1, static_cast<uint32_t>(nTok * ropeStride_ * sizeof(T)), 0, 0, 0}, pad);
        inSin_.EnQue(sb);
    }

    __aicore__ inline void ComputeTile(uint32_t k)
    {
        LocalTensor<T> xb = inQkv_.DeQue<T>();
        LocalTensor<T> cb = inCos_.DeQue<T>();
        LocalTensor<T> sb = inSin_.DeQue<T>();

        // ---- 输出 tile 分配（pack-store 直写目标）----
        AscendC::LocalTensor<T> ob = outTile_.AllocTensor<T>();
        const uint32_t rowElems = numHeadTotal_ * kHeadDim;
        const uint32_t qkElems = (numHeadQ_ + numHeadK_) * kHeadDim;
        const uint32_t vElems = numHeadV_ * kHeadDim;

        // ---- v 直通：UB→UB 拷贝，原样 T，不经 fp32 ----
        //   块状布局一次多行搬完；KV 交错布局下 v 头间隔 2D，按 head 逐个搬
        if (!kvInterleaved_) {
            //   UB 侧 blockLen/srcGap/dstGap 单位均为 32B 块
            AscendC::DataCopyParams vParams{
                static_cast<uint16_t>(kTokenTile),
                static_cast<uint16_t>(vElems * sizeof(T) / 32),
                static_cast<uint16_t>((padStride_ - vElems) * sizeof(T) / 32),
                static_cast<uint16_t>((rowElems - vElems) * sizeof(T) / 32)};
            AscendC::DataCopy(ob[qkElems], xb[qkElems], vParams);
        } else {
            const uint16_t headBlocks = static_cast<uint16_t>(kHeadDim * sizeof(T) / 32);
            for (uint16_t h = 0; h < static_cast<uint16_t>(numHeadV_); ++h) {
                AscendC::DataCopyParams vParams{
                    static_cast<uint16_t>(kTokenTile),
                    headBlocks,
                    static_cast<uint16_t>((padStride_ - kHeadDim) * sizeof(T) / 32),
                    static_cast<uint16_t>((rowElems - kHeadDim) * sizeof(T) / 32)};
                AscendC::DataCopy(ob[qkElems + static_cast<uint32_t>(h) * kHeadDim],
                                  xb[(numHeadQ_ + 2 * h + 1) * kHeadDim], vParams);
            }
        }

        // ---- norm + rope (VF)：q/k 与 cos/sin 直接从 T 载入，无 cast-in ----
        AscendC::LocalTensor<float> sumL = bufSum_.Get<float>();
        AscendC::LocalTensor<float> gqf = bufGammaQ_.Get<float>();
        AscendC::LocalTensor<float> gkf = bufGammaK_.Get<float>();
        // v 已由上面的拷贝完成；VF 读 xb/cb/sb，故三个队列在 VF 之后才释放。
        SqnrNormRopeVfPack<T>((__ubuf__ T *)ob.GetPhyAddr(), (__ubuf__ T *)xb.GetPhyAddr(),
            (__ubuf__ T *)cb.GetPhyAddr(), (__ubuf__ T *)sb.GetPhyAddr(),
            (__ubuf__ float *)gqf.GetPhyAddr(), (__ubuf__ float *)gkf.GetPhyAddr(),
            (__ubuf__ float *)sumL.GetPhyAddr(), static_cast<uint16_t>(numHeadQ_),
            static_cast<uint16_t>(numHeadK_), static_cast<uint16_t>(padStride_),
            static_cast<uint16_t>(rowElems), static_cast<uint16_t>(ropeStride_),
            static_cast<uint16_t>(kvInterleaved_ ? 2 * kHeadDim : kHeadDim), eps_, recip_);
        AscendC::PipeBarrier<PIPE_V>();
        inQkv_.FreeTensor(xb);
        inCos_.FreeTensor(cb);
        inSin_.FreeTensor(sb);
        AscendC::PipeBarrier<PIPE_V>();
        outTile_.EnQue(ob);
    }

    __aicore__ inline void CopyOutTile(uint32_t k)
    {
        const uint32_t t0 = blkStart_ + k * kTokenTile;
        const uint32_t nTok = TileTokens(k);
        LocalTensor<T> ob = outTile_.DeQue<T>();

        if (layoutBNSD_) {
            CopyOutBnsd(qGm_, ob, t0, nTok, 0, numHeadQ_, numHeadQ_);
            CopyOutBnsd(kGm_, ob, t0, nTok, numHeadQ_, kvOutHeads_, numHeadK_);
            CopyOutBnsd(vGm_, ob, t0, nTok, numHeadQ_ + numHeadK_, kvOutHeads_, numHeadV_);
        } else {
            CopyOutBshd(qGm_, ob, t0, nTok, 0, numHeadQ_, numHeadQ_);
            CopyOutBshd(kGm_, ob, t0, nTok, numHeadQ_, kvOutHeads_, numHeadK_);
            CopyOutBshd(vGm_, ob, t0, nTok, numHeadQ_ + numHeadK_, kvOutHeads_, numHeadV_);
        }
        outTile_.FreeTensor(ob);
    }

    // [B,S,H*D]：q/k/v 行内 head 连续
    __aicore__ inline void CopyOutBshd(const GlobalTensor<T> &gm, const LocalTensor<T> &src,
                                      uint32_t t0, uint32_t nTok, uint32_t srcHeadBase, uint32_t dstHeads,
                                      uint32_t srcHeads)
    {
        const uint32_t rowElems = numHeadTotal_ * kHeadDim;
        const uint32_t elemBytes = sizeof(T);
        if (dstHeads == srcHeads) {
            uint32_t segBytes = dstHeads * kHeadDim * elemBytes;
            DataCopyPad(gm[t0 * dstHeads * kHeadDim], src[srcHeadBase * kHeadDim],
                DataCopyExtParams{static_cast<uint16_t>(nTok), segBytes,
                    (rowElems * elemBytes - segBytes) / 32, 0, 0});
        } else {
            // GQA 扩张：dest 每 token 行内 head 之间不连续（跨 dstHeads*D），dstStride 单位是字节
            uint32_t headBytes = kHeadDim * elemBytes;
            uint32_t srcStride = (rowElems - kHeadDim) * elemBytes / 32;
            uint32_t dstStride = (dstHeads - 1) * kHeadDim * elemBytes;
            for (uint32_t h = 0; h < dstHeads; ++h) {
                uint32_t sh = h / gqaGroup_;
                DataCopyPad(gm[(t0 * dstHeads + h) * kHeadDim], src[(srcHeadBase + sh) * kHeadDim],
                    DataCopyExtParams{static_cast<uint16_t>(nTok), headBytes, srcStride, dstStride, 0});
            }
        }
    }

    // [B,N,S,D]：同 head 内沿 S 连续，跨 batch 边界时拆段
    __aicore__ inline void CopyOutBnsd(const GlobalTensor<T> &gm, const LocalTensor<T> &src,
                                      uint32_t t0, uint32_t nTok, uint32_t srcHeadBase, uint32_t dstHeads,
                                      uint32_t srcHeads)
    {
        const uint32_t rowElems = numHeadTotal_ * kHeadDim;
        const uint32_t elemBytes = sizeof(T);
        const uint32_t headBytes = kHeadDim * elemBytes;
        const uint32_t srcStride = (rowElems - kHeadDim) * elemBytes / 32;
        const uint32_t dstStride = 0;  // BNSD 下同一 head 内 token 相邻，落盘连续

        uint32_t done = 0;
        while (done < nTok) {
            const uint32_t t = t0 + done;
            const uint32_t b = t / seqLen_;
            const uint32_t s = t - b * seqLen_;
            uint32_t rows = seqLen_ - s;
            if (rows > nTok - done) {
                rows = nTok - done;
            }
            for (uint32_t h = 0; h < dstHeads; ++h) {
                uint32_t sh = (dstHeads == srcHeads) ? h : (h / gqaGroup_);
                uint32_t dstOff = static_cast<uint32_t>(
                    (static_cast<uint64_t>(b) * dstHeads + h) * seqLen_ * kHeadDim +
                    static_cast<uint64_t>(s) * kHeadDim);
                DataCopyPad(gm[dstOff], src[done * rowElems + (srcHeadBase + sh) * kHeadDim],
                    DataCopyExtParams{static_cast<uint16_t>(rows), headBytes, srcStride, dstStride, 0});
            }
            done += rows;
        }
    }

private:
    TPipe *pipe_ = nullptr;

    GlobalTensor<T> qkvGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> gammaQGm_;
    GlobalTensor<T> gammaKGm_;
    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> vGm_;

    TQue<TPosition::VECIN, 2> inQkv_;
    TQue<TPosition::VECIN, 2> inCos_;
    TQue<TPosition::VECIN, 2> inSin_;
    TQue<TPosition::VECIN, 2> inGamma_;
    TQue<TPosition::VECOUT, 2> outTile_;

    TBuf<TPosition::VECCALC> bufGammaQ_;
    TBuf<TPosition::VECCALC> bufGammaK_;
    TBuf<TPosition::VECCALC> bufSum_;

    uint32_t seqLen_ = 0;
    uint32_t numHeadQ_ = 0;
    uint32_t numHeadK_ = 0;
    uint32_t numHeadV_ = 0;
    uint32_t numHeadTotal_ = 0;
    uint32_t qkvStride_ = 0;
    uint32_t padStride_ = 0;
    uint32_t ropeStride_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t tokenPerCore_ = 0;
    uint32_t lastBlkTokens_ = 0;
    uint32_t kvOutHeads_ = 0;
    uint32_t gqaGroup_ = 1;
    uint32_t blkStart_ = 0;
    uint32_t blkTokens_ = 0;
    uint32_t tileNum_ = 0;
    float eps_ = 1e-6f;
    float recip_ = 1.0f / 128.0f;
    bool gqaExpand_ = false;
    bool layoutBNSD_ = false;
    bool kvInterleaved_ = false;
};

}  // namespace kernels

#endif  // SPLIT_QKV_NORM_ROPE_KERNEL_H_
