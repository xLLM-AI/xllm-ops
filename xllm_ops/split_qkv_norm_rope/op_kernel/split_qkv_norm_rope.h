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

constexpr uint32_t kHeadDim = 128;              // head_dim 默认值；运行期实际 D 从 tiling 读入
constexpr uint32_t kHalfDim = kHeadDim / 2;     // 64，RoPE half-split 分界（仅作 VF 注释参照，不再硬引用）
constexpr uint32_t kMaxHeads = 64;              // sumsq UB 上界：Nq+Nk <= kMaxHeads
constexpr uint32_t kTokenTile = 8;              // 默认每 tile 行数；host 会按 UB 预算从 8/4/2/1 里选
// 每 tile 的 sumsq 个数 = kTokenTile × (Nq+Nk)，scale 链按 64 lane 向上取整分段

constexpr uint32_t kFlagGqaExpand = 1u << 0;    // 1：k/v 扩张到 Nq 头
constexpr uint32_t kFlagLayoutBNSD = 1u << 1;   // 1：[B,N,S,D]；0：[B,S,N*D]
constexpr uint32_t kFlagKvInterleaved = 1u << 2;  // 1：输入行内 KV 按 head 交错 [K_i,V_i]；0：[全部 K | 全部 V]

// ---------------------------------------------------------------------------
// SqnrNormRopeVfPack：逐行 QK-RMSNorm + 半分化 RoPE（只处理 q/k 头；v 由外层 UB→UB 直通拷贝）
//   outT      : T 输出 tile，[kTokenTile, numHeadTotal*headDim]，行距 outStride
//   xb        : T qkv tile，[kTokenTile, padStride]，行距 xStride
//   cosT/sinT : T cos/sin，[kTokenTile, ropeStride]，只用前 halfDim 维（NeoX 半维）
//   gammaQ/K  : fp32 [headDim]
//   sumsq     : fp32 workspace，[kTokenTile, Nq+Nk]，行距 = Nq+Nk（容量按 kMaxHeads 上界分配）
//   headDim   : 当前支持 {128, 256}；VF 按 64-lane chunk 处理：pass1 对 D/64 段累加 sumsq，
//               pass2 对 halfDim/64 个 (前半,后半) pair 做 NeoX 旋转，D=128 时各循环恰跑 1 次。
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
template <typename T, bool kHalfMasked, uint16_t kD, uint16_t kRD>
__simd_vf__ inline void SqnrNormRopeVfPack(
    __ubuf__ T *outT, __ubuf__ T *xb, __ubuf__ T *cosT, __ubuf__ T *sinT,
    __ubuf__ float *gammaQ, __ubuf__ float *gammaK, __ubuf__ float *sumsq,
    uint16_t numHeadQ, uint16_t numHeadK, uint16_t xStride, uint16_t outStride, uint16_t ropeStride,
    uint16_t tileRows, uint16_t kvHeadStep, float eps, float recip)
{
    using namespace AscendC::Reg;
    // kD/kRD 模板形参 → constexpr 局部：支持集闭环 {64,96,128,256}，VF 内所有循环
    // （pass1 的 c、pass2 的 p 与 passthrough）trip count 编译期已知而完全展开，
    // 消除运行期循环开销与寄存器地址依赖（D=128 全 rope 与 af38104 基线指令流等价）。
    constexpr uint16_t headDim = kD;
    constexpr uint16_t halfDim = static_cast<uint16_t>(kD / 2);
    constexpr uint16_t ropeDim = kRD;
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
    const uint32_t kvHeadBase = static_cast<uint32_t>(numHeadQ) * headDim;
    // sumsq 段数：每段 64 lane，向上取整覆盖 tileRows*(Nq+Nk) 个有效元素；
    // 末段多出的 lane 只写回 sumsq，pass2 只按 (t,h) 单点广播读取，不回读这些 lane。
    constexpr uint16_t kSeg = 64;
    const uint16_t segs = static_cast<uint16_t>((tileRows * qkHeads + kSeg - 1) / kSeg);

    if constexpr (kHalfMasked) {
    // ==== HalfMasked 路径：D ∈ {64, 96}，halfDim < 64 lane ====
    // 半区配对借 UpdateMask 计数掩码：前半/后半各自占一个 64-lane 寄存器的 [0,halfDim) lane，
    // NeoX 旋转在两寄存器间完成（官方 arch35 HalfAlignVF 范式），无需任何 lane 重排。
    // γ 布局（CopyInGamma 写入）：前半 [0,halfDim)，后半 [64,64+halfDim)；窗口外 lane 由掩码丢弃。
    // D=64/96 下 rope_dim 仅可为 0（全 rope，host %128 约束），故无 passthrough 分支。

    // ---- pass 1：sumsq（chunk0 全量 + 尾 chunk 计数掩码；D=96 时 c=1、tail=32）----
    const uint16_t nChunksC = static_cast<uint16_t>((headDim + kSeg - 1) / kSeg);
    uint32_t tailCnt = static_cast<uint32_t>(headDim - (nChunksC - 1) * kSeg);
    MaskReg mTail = UpdateMask<float>(tailCnt);
    for (uint16_t t = 0; t < tileRows; ++t) {
        __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
        for (uint16_t h = 0; h < numHeadQ; ++h) {
            __ubuf__ T *src = row + static_cast<uint32_t>(h) * headDim;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(acc, tv, mFull);
            Mul(acc, acc, acc, mFull);
            for (uint16_t c = 1; c < nChunksC; ++c) {
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(c) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mTail);
                // arch35 MulAddDst 仅支持 ZEROING（toolkit static_assert：mode ZEROING only），
                // 带 mTail 会清掉 acc inactive lane——那里已是 chunk0 的真实平方和。改走官方
                // layernorm_variance 同款 idiom：尾 chunk 掩码 Cast 落临时寄存器（layoutZMrgZ，
                // inactive lane=0），再用全量掩码 MulAddDst 累加（inactive += 0，chunk0 保留）。
                MulAddDst(acc, y2, y2, mFull);
            }
            ReduceSum(acc, acc, mFull);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                sumsq + static_cast<uint32_t>(t) * qkHeads + h, acc, mOne);
        }
        for (uint16_t h = 0; h < numHeadK; ++h) {
            __ubuf__ T *src = row + kvHeadBase + static_cast<uint32_t>(h) * kvHeadStep;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(acc, tv, mFull);
            Mul(acc, acc, acc, mFull);
            for (uint16_t c = 1; c < nChunksC; ++c) {
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(c) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mTail);
                // 同上：全量掩码 MulAddDst 累加尾 chunk（inactive lane=0，chunk0 保留）
                MulAddDst(acc, y2, y2, mFull);
            }
            ReduceSum(acc, acc, mFull);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                sumsq + static_cast<uint32_t>(t) * qkHeads + numHeadQ + h, acc, mOne);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // ---- 每 tile 的 scale 链（同全量路径）----
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

    // ---- pass 2：norm + rope，单 masked pair；前半 [0,halfDim) 配后半 [halfDim,D) ----
    uint32_t halfCnt = static_cast<uint32_t>(halfDim);  // 32 (D=64) / 48 (D=96)
    MaskReg pregHalf = UpdateMask<float>(halfCnt);
    // γ 每 VF 装载一次（布局：前半 [0,halfDim)、后半 [64,64+halfDim)，CopyInGamma 写入），
    // 避免 token 循环内重复 LoadAlign（与全量路径同款外提）
    LoadAlign(gq1, gammaQ);
    LoadAlign(gq2, gammaQ + 64);
    LoadAlign(gk1, gammaK);
    LoadAlign(gk2, gammaK + 64);
    for (uint16_t t = 0; t < tileRows; ++t) {
        __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
        __ubuf__ T *orow = outT + static_cast<uint32_t>(t) * outStride;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, cosT + static_cast<uint32_t>(t) * ropeStride);
        Cast<float, T, AscendC::layoutZMrgZ>(cr, tv, pregHalf);
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, sinT + static_cast<uint32_t>(t) * ropeStride);
        Cast<float, T, AscendC::layoutZMrgZ>(sr, tv, pregHalf);
        Muls(sn, sr, -1.0f, pregHalf);
        // q 组常量（γq 已 tile 外装载，折进 cos/sin）
        Mul(ca, gq1, cr, pregHalf);
        Mul(cc, gq1, sr, pregHalf);
        Mul(cd, gq2, cr, pregHalf);
        Mul(cb, gq2, sn, pregHalf);
        for (uint16_t h = 0; h < numHeadQ; ++h) {
            __ubuf__ T *src = row + static_cast<uint32_t>(h) * headDim;
            __ubuf__ T *dst = orow + static_cast<uint32_t>(h) * headDim;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumsq + static_cast<uint32_t>(t) * qkHeads + h);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, pregHalf);
            Mul(y1, y1, sg, pregHalf);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + halfDim);
            Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, pregHalf);
            Mul(y2, y2, sg, pregHalf);
            Mul(t1, y1, ca, pregHalf);
            MulAddDst(t1, y2, cb, pregHalf);
            Mul(t2, y2, cd, pregHalf);
            MulAddDst(t2, y1, cc, pregHalf);
            Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t1, pregHalf);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, pk, pregHalf);
            Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t2, pregHalf);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + halfDim, pk, pregHalf);
        }
        // k 组常量（γk 已 tile 外装载；cr/sr/sn 同 token 复用）
        Mul(ca, gk1, cr, pregHalf);
        Mul(cc, gk1, sr, pregHalf);
        Mul(cd, gk2, cr, pregHalf);
        Mul(cb, gk2, sn, pregHalf);
        for (uint16_t h = 0; h < numHeadK; ++h) {
            uint32_t hh = static_cast<uint32_t>(numHeadQ) + h;
            __ubuf__ T *src = row + kvHeadBase + static_cast<uint32_t>(h) * kvHeadStep;
            __ubuf__ T *dst = orow + hh * headDim;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumsq + static_cast<uint32_t>(t) * qkHeads + hh);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, pregHalf);
            Mul(y1, y1, sg, pregHalf);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + halfDim);
            Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, pregHalf);
            Mul(y2, y2, sg, pregHalf);
            Mul(t1, y1, ca, pregHalf);
            MulAddDst(t1, y2, cb, pregHalf);
            Mul(t2, y2, cd, pregHalf);
            MulAddDst(t2, y1, cc, pregHalf);
            Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t1, pregHalf);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, pk, pregHalf);
            Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t2, pregHalf);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + halfDim, pk, pregHalf);
        }
    }
    } else {
    // ==== 全量路径：D ∈ {128, 256}（与 Tier1/Tier3 实现逐行一致）====
    // D 按 64-lane chunk 处理：pass1 累加 D/64 段；pass2 把 half 区切成 halfDim/64 个 chunk，
    // 其中前 ropeDim/2/64 个 chunk 做 NeoX 旋转配对，其余 chunk 经 norm*gamma 直通（部分 rope）
    const uint16_t nChunks = static_cast<uint16_t>(headDim / kSeg);
    const uint16_t nPairs = static_cast<uint16_t>(halfDim / kSeg);
    const uint16_t nRopePairs = static_cast<uint16_t>(ropeDim / 2 / kSeg);

    // pass 1：每 (token, head) 的 sumsq（q/k 直接从 T 载入）
    for (uint16_t t = 0; t < tileRows; ++t) {
        __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
        for (uint16_t h = 0; h < numHeadQ; ++h) {
            __ubuf__ T *src = row + static_cast<uint32_t>(h) * headDim;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
            Mul(acc, y1, y1, mFull);
            // 余下 64-lane 段继续累加；D=128 时循环体恰跑 1 次（等价原 hardcode）
            for (uint16_t c = 1; c < nChunks; ++c) {
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(c) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
                MulAddDst(acc, y2, y2, mFull);
            }
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
            for (uint16_t c = 1; c < nChunks; ++c) {
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(c) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
                MulAddDst(acc, y2, y2, mFull);
            }
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
    // γ 每 VF 每对只装载一次（pair 外提，t 循环内仅重算依赖 token 的 cos/sin 与常量）——
    // D=128 全 rope 时与 af38104 基线指令流等价（γ Load 4 次/tile、cos/sin Load 2 次/token）。
    // q 组：γq 折进 cos/sin；逐 64-lane pair 处理（D=128 → nPairs=1，等价原 hardcode）
    for (uint16_t p = 0; p < nRopePairs; ++p) {
        LoadAlign(gq1, gammaQ + static_cast<uint32_t>(p) * kSeg);
        LoadAlign(gq2, gammaQ + halfDim + static_cast<uint32_t>(p) * kSeg);
        for (uint16_t t = 0; t < tileRows; ++t) {
            __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
            __ubuf__ T *orow = outT + static_cast<uint32_t>(t) * outStride;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, cosT + static_cast<uint32_t>(t) * ropeStride + static_cast<uint32_t>(p) * kSeg);
            Cast<float, T, AscendC::layoutZMrgZ>(cr, tv, mFull);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, sinT + static_cast<uint32_t>(t) * ropeStride + static_cast<uint32_t>(p) * kSeg);
            Cast<float, T, AscendC::layoutZMrgZ>(sr, tv, mFull);
            Muls(sn, sr, -1.0f, mFull);
            Mul(ca, gq1, cr, mFull);
            Mul(cc, gq1, sr, mFull);
            Mul(cd, gq2, cr, mFull);
            Mul(cb, gq2, sn, mFull);
            for (uint16_t h = 0; h < numHeadQ; ++h) {
                __ubuf__ T *src = row + static_cast<uint32_t>(h) * headDim;
                __ubuf__ T *dst = orow + static_cast<uint32_t>(h) * headDim;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumsq + static_cast<uint32_t>(t) * qkHeads + h);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(p) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
                Mul(y1, y1, sg, mFull);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + halfDim + static_cast<uint32_t>(p) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
                Mul(y2, y2, sg, mFull);
                Mul(t1, y1, ca, mFull);
                MulAddDst(t1, y2, cb, mFull);
                Mul(t2, y2, cd, mFull);
                MulAddDst(t2, y1, cc, mFull);
                Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t1, mFull);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + static_cast<uint32_t>(p) * kSeg, pk, mFull);
                Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t2, mFull);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + halfDim + static_cast<uint32_t>(p) * kSeg, pk, mFull);
            }
        }
    }

    // q 组 passthrough：未旋转的 half chunk 经 norm*γq 直通（部分 rope 时生效；
    // 全 rope 时 nRopePairs==nPairs，本循环不执行，路径与原实现完全一致）
    for (uint16_t p = nRopePairs; p < nPairs; ++p) {
        LoadAlign(gq1, gammaQ + static_cast<uint32_t>(p) * kSeg);
        LoadAlign(gq2, gammaQ + halfDim + static_cast<uint32_t>(p) * kSeg);
        for (uint16_t t = 0; t < tileRows; ++t) {
            __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
            __ubuf__ T *orow = outT + static_cast<uint32_t>(t) * outStride;
            for (uint16_t h = 0; h < numHeadQ; ++h) {
                __ubuf__ T *src = row + static_cast<uint32_t>(h) * headDim;
                __ubuf__ T *dst = orow + static_cast<uint32_t>(h) * headDim;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumsq + static_cast<uint32_t>(t) * qkHeads + h);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(p) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
                Mul(y1, y1, sg, mFull);
                Mul(y1, y1, gq1, mFull);
                Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, y1, mFull);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + static_cast<uint32_t>(p) * kSeg, pk, mFull);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + halfDim + static_cast<uint32_t>(p) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
                Mul(y2, y2, sg, mFull);
                Mul(y2, y2, gq2, mFull);
                Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, y2, mFull);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + halfDim + static_cast<uint32_t>(p) * kSeg, pk, mFull);
            }
        }
    }

    // k 组：γk 折进 cos/sin
    for (uint16_t p = 0; p < nRopePairs; ++p) {
        LoadAlign(gk1, gammaK + static_cast<uint32_t>(p) * kSeg);
        LoadAlign(gk2, gammaK + halfDim + static_cast<uint32_t>(p) * kSeg);
        for (uint16_t t = 0; t < tileRows; ++t) {
            __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
            __ubuf__ T *orow = outT + static_cast<uint32_t>(t) * outStride;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, cosT + static_cast<uint32_t>(t) * ropeStride + static_cast<uint32_t>(p) * kSeg);
            Cast<float, T, AscendC::layoutZMrgZ>(cr, tv, mFull);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, sinT + static_cast<uint32_t>(t) * ropeStride + static_cast<uint32_t>(p) * kSeg);
            Cast<float, T, AscendC::layoutZMrgZ>(sr, tv, mFull);
            Muls(sn, sr, -1.0f, mFull);
            Mul(ca, gk1, cr, mFull);
            Mul(cc, gk1, sr, mFull);
            Mul(cd, gk2, cr, mFull);
            Mul(cb, gk2, sn, mFull);
            for (uint16_t h = 0; h < numHeadK; ++h) {
                uint32_t hh = static_cast<uint32_t>(numHeadQ) + h;
                __ubuf__ T *src = row + kvHeadBase + static_cast<uint32_t>(h) * kvHeadStep;
                __ubuf__ T *dst = orow + hh * headDim;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumsq + static_cast<uint32_t>(t) * qkHeads + hh);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(p) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
                Mul(y1, y1, sg, mFull);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + halfDim + static_cast<uint32_t>(p) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
                Mul(y2, y2, sg, mFull);
                Mul(t1, y1, ca, mFull);
                MulAddDst(t1, y2, cb, mFull);
                Mul(t2, y2, cd, mFull);
                MulAddDst(t2, y1, cc, mFull);
                Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t1, mFull);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + static_cast<uint32_t>(p) * kSeg, pk, mFull);
                Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, t2, mFull);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + halfDim + static_cast<uint32_t>(p) * kSeg, pk, mFull);
            }
        }
    }

    // k 组 passthrough：未旋转的 half chunk 经 norm*γk 直通（部分 rope 时生效；
    // 全 rope 时 nRopePairs==nPairs，本循环不执行，路径与原实现完全一致）
    for (uint16_t p = nRopePairs; p < nPairs; ++p) {
        LoadAlign(gk1, gammaK + static_cast<uint32_t>(p) * kSeg);
        LoadAlign(gk2, gammaK + halfDim + static_cast<uint32_t>(p) * kSeg);
        for (uint16_t t = 0; t < tileRows; ++t) {
            __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
            __ubuf__ T *orow = outT + static_cast<uint32_t>(t) * outStride;
            for (uint16_t h = 0; h < numHeadK; ++h) {
                uint32_t hh = static_cast<uint32_t>(numHeadQ) + h;
                __ubuf__ T *src = row + kvHeadBase + static_cast<uint32_t>(h) * kvHeadStep;
                __ubuf__ T *dst = orow + hh * headDim;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumsq + static_cast<uint32_t>(t) * qkHeads + hh);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(p) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
                Mul(y1, y1, sg, mFull);
                Mul(y1, y1, gk1, mFull);
                Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, y1, mFull);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + static_cast<uint32_t>(p) * kSeg, pk, mFull);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + halfDim + static_cast<uint32_t>(p) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
                Mul(y2, y2, sg, mFull);
                Mul(y2, y2, gk2, mFull);
                Cast<T, float, AscendC::LayoutZMrgZRndRSatNS>(pk, y2, mFull);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + halfDim + static_cast<uint32_t>(p) * kSeg, pk, mFull);
            }
        }
    }
    }  // else：全量路径（D ∈ {128, 256}）
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
        headDim_ = tiling->headDim;
        halfDim_ = headDim_ / 2;
        qkvStride_ = tiling->qkvStride;
        padStride_ = tiling->padStride;
        ropeStride_ = tiling->ropeStride;
        blockDim_ = tiling->blockDim;
        tokenPerCore_ = tiling->tokenPerCore;
        lastBlkTokens_ = tiling->lastBlockTokenNum;
        tileRows_ = (tiling->tokenTile != 0) ? tiling->tokenTile : kTokenTile;
        eps_ = tiling->eps;
        recip_ = tiling->reciprocal;
        ropeDim_ = tiling->ropeDim;
        if (ropeDim_ == 0 || ropeDim_ > headDim_) {  // 缺省/异常回退：全 rope（与历史行为一致）
            ropeDim_ = headDim_;
        }

        gqaExpand_ = (tiling->flags & kFlagGqaExpand) != 0;
        layoutBNSD_ = (tiling->flags & kFlagLayoutBNSD) != 0;
        kvInterleaved_ = (tiling->flags & kFlagKvInterleaved) != 0;
        kvOutHeads_ = gqaExpand_ ? numHeadQ_ : numHeadK_;
        gqaGroup_ = (numHeadK_ == 0) ? 1 : (numHeadQ_ / numHeadK_);

        // 全局偏移按 u64 计算：B*S*stride 超 2^32 元素时 u32 乘法会回绕
        qkvGm_.SetGlobalBuffer((__gm__ T *)qkv, static_cast<uint64_t>(tiling->totalTokens) * qkvStride_);
        cosGm_.SetGlobalBuffer((__gm__ T *)cosGm, static_cast<uint64_t>(tiling->totalTokens) * ropeStride_);
        sinGm_.SetGlobalBuffer((__gm__ T *)sinGm, static_cast<uint64_t>(tiling->totalTokens) * ropeStride_);
        gammaQGm_.SetGlobalBuffer((__gm__ T *)qGamma, headDim_);
        gammaKGm_.SetGlobalBuffer((__gm__ T *)kGamma, headDim_);
        qGm_.SetGlobalBuffer(
            (__gm__ T *)qOut, static_cast<uint64_t>(tiling->totalTokens) * numHeadQ_ * headDim_);
        kGm_.SetGlobalBuffer(
            (__gm__ T *)kOut, static_cast<uint64_t>(tiling->totalTokens) * kvOutHeads_ * headDim_);
        vGm_.SetGlobalBuffer(
            (__gm__ T *)vOut, static_cast<uint64_t>(tiling->totalTokens) * kvOutHeads_ * headDim_);

        // 深度 2：支持「下一个 tile 预取」+「上一个 tile 延后写出」
        pipe_->InitBuffer(inQkv_, 2, tileRows_ * padStride_ * sizeof(T));
        pipe_->InitBuffer(inCos_, 2, tileRows_ * ropeStride_ * sizeof(T));
        pipe_->InitBuffer(inSin_, 2, tileRows_ * ropeStride_ * sizeof(T));
        pipe_->InitBuffer(inGamma_, 2, headDim_ * sizeof(T));
        pipe_->InitBuffer(outTile_, 2, tileRows_ * numHeadTotal_ * headDim_ * sizeof(T));

        // HalfMasked（D=64/96，halfDim<64）时 γ 拆成 front@0 / back@64 两个 64-lane 窗口，buffer 放宽到 128 float；
        // D=128/256 仍按 headDim 分配，与 Tier1 一致
        const uint32_t gammaElems = (halfDim_ < 64) ? 128u : headDim_;
        pipe_->InitBuffer(bufGammaQ_, gammaElems * sizeof(float));
        pipe_->InitBuffer(bufGammaK_, gammaElems * sizeof(float));
        pipe_->InitBuffer(bufSum_, tileRows_ * kMaxHeads * sizeof(float));

        // 本核 token 区间与 tile 数（供三级流水使用）
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        blkStart_ = blockIdx * tokenPerCore_;
        blkTokens_ = (blockIdx == blockDim_ - 1) ? lastBlkTokens_ : tokenPerCore_;
        tileNum_ = (blkTokens_ + tileRows_ - 1) / tileRows_;

        // 只有一条路径：sumsq 段数按 ceil(tileRows*(Nq+Nk)/64) 运行期算，任意 Nq+Nk <= kMaxHeads 均可覆盖。
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
        const uint32_t left = blkTokens_ - k * tileRows_;
        return left > tileRows_ ? tileRows_ : left;
    }

    __aicore__ inline void CopyInGamma()
    {
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        LocalTensor<T> gq = inGamma_.AllocTensor<T>();
        DataCopyPad(gq, gammaQGm_, DataCopyExtParams{1, static_cast<uint32_t>(headDim_ * sizeof(T)), 0, 0, 0}, pad);
        inGamma_.EnQue(gq);
        LocalTensor<T> gk = inGamma_.AllocTensor<T>();
        DataCopyPad(gk, gammaKGm_, DataCopyExtParams{1, static_cast<uint32_t>(headDim_ * sizeof(T)), 0, 0, 0}, pad);
        inGamma_.EnQue(gk);

        gq = inGamma_.DeQue<T>();
        gk = inGamma_.DeQue<T>();
        LocalTensor<float> gqf = bufGammaQ_.Get<float>();
        LocalTensor<float> gkf = bufGammaK_.Get<float>();
        if (halfDim_ < 64) {
            // HalfMasked：前半 γ 落 [0,halfDim)，后半 γ 落 [64,64+halfDim)；
            // 两 64-lane 窗口内 halfDim 之后的 lane 由 VF 的 pregHalf 丢弃，无需清零
            Cast(gqf, gq, RoundMode::CAST_NONE, halfDim_);
            Cast(gqf[64], gq[halfDim_], RoundMode::CAST_NONE, halfDim_);
            Cast(gkf, gk, RoundMode::CAST_NONE, halfDim_);
            Cast(gkf[64], gk[halfDim_], RoundMode::CAST_NONE, halfDim_);
        } else {
            Cast(gqf, gq, RoundMode::CAST_NONE, headDim_);
            Cast(gkf, gk, RoundMode::CAST_NONE, headDim_);
        }
        inGamma_.FreeTensor(gq);
        inGamma_.FreeTensor(gk);
        PipeBarrier<PIPE_V>();
    }

    // 只发 DMA，不等：深度 2 的队列允许它和上一个 tile 的计算并行
    __aicore__ inline void CopyInTile(uint32_t k)
    {
        const uint32_t t0 = blkStart_ + k * tileRows_;
        const uint32_t nTok = TileTokens(k);
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // 每 token 一行单独搬：GM 行距（qkvStride_*sizeof(T)）不保证 32B 对齐，UB 侧落到 32B 对齐的 padStride 行距
        LocalTensor<T> xb = inQkv_.AllocTensor<T>();
        for (uint32_t t = 0; t < nTok; ++t) {
            DataCopyPad(xb[t * padStride_], qkvGm_[static_cast<uint64_t>(t0 + t) * qkvStride_],
                DataCopyExtParams{1, static_cast<uint32_t>(qkvStride_ * sizeof(T)), 0, 0, 0}, pad);
        }
        inQkv_.EnQue(xb);

        LocalTensor<T> cb = inCos_.AllocTensor<T>();
        DataCopyPad(cb, cosGm_[static_cast<uint64_t>(t0) * ropeStride_],
            DataCopyExtParams{1, static_cast<uint32_t>(nTok * ropeStride_ * sizeof(T)), 0, 0, 0}, pad);
        inCos_.EnQue(cb);

        LocalTensor<T> sb = inSin_.AllocTensor<T>();
        DataCopyPad(sb, sinGm_[static_cast<uint64_t>(t0) * ropeStride_],
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
        const uint32_t rowElems = numHeadTotal_ * headDim_;
        const uint32_t qkElems = (numHeadQ_ + numHeadK_) * headDim_;
        const uint32_t vElems = numHeadV_ * headDim_;

        // ---- v 直通：UB→UB 拷贝，原样 T，不经 fp32 ----
        //   块状布局一次多行搬完；KV 交错布局下 v 头间隔 2D，按 head 逐个搬
        if (!kvInterleaved_) {
            //   UB 侧 blockLen/srcGap/dstGap 单位均为 32B 块
            AscendC::DataCopyParams vParams{
                static_cast<uint16_t>(tileRows_),
                static_cast<uint16_t>(vElems * sizeof(T) / 32),
                static_cast<uint16_t>((padStride_ - vElems) * sizeof(T) / 32),
                static_cast<uint16_t>((rowElems - vElems) * sizeof(T) / 32)};
            AscendC::DataCopy(ob[qkElems], xb[qkElems], vParams);
        } else {
            const uint16_t headBlocks = static_cast<uint16_t>(headDim_ * sizeof(T) / 32);
            for (uint16_t h = 0; h < static_cast<uint16_t>(numHeadV_); ++h) {
                AscendC::DataCopyParams vParams{
                    static_cast<uint16_t>(tileRows_),
                    headBlocks,
                    static_cast<uint16_t>((padStride_ - headDim_) * sizeof(T) / 32),
                    static_cast<uint16_t>((rowElems - headDim_) * sizeof(T) / 32)};
                AscendC::DataCopy(ob[qkElems + static_cast<uint32_t>(h) * headDim_],
                                  xb[(numHeadQ_ + 2 * h + 1) * headDim_], vParams);
            }
        }

        // ---- norm + rope (VF)：q/k 与 cos/sin 直接从 T 载入，无 cast-in ----
        AscendC::LocalTensor<float> sumL = bufSum_.Get<float>();
        AscendC::LocalTensor<float> gqf = bufGammaQ_.Get<float>();
        AscendC::LocalTensor<float> gkf = bufGammaK_.Get<float>();
        // v 已由上面的拷贝完成；VF 读 xb/cb/sb，故三个队列在 VF 之后才释放。
        // 支持集闭环 → 按 (headDim, rope_dim) 字面量实例化（host 保证取值集合），
        // VF 内所有循环编译期展开：D=64/96 HalfMasked；D=128 全 rope；
        // D=256 全 rope 或部分 rope（rope_dim ∈ {128, 256}）。
        const uint16_t hd = static_cast<uint16_t>(headDim_);
        const uint16_t rd = static_cast<uint16_t>(ropeDim_);
        const uint16_t kvStep = static_cast<uint16_t>(kvInterleaved_ ? 2 * headDim_ : headDim_);
#define SQNR_VF_ARGS \
    (__ubuf__ T *)ob.GetPhyAddr(), (__ubuf__ T *)xb.GetPhyAddr(), \
    (__ubuf__ T *)cb.GetPhyAddr(), (__ubuf__ T *)sb.GetPhyAddr(), \
    (__ubuf__ float *)gqf.GetPhyAddr(), (__ubuf__ float *)gkf.GetPhyAddr(), \
    (__ubuf__ float *)sumL.GetPhyAddr(), static_cast<uint16_t>(numHeadQ_), \
    static_cast<uint16_t>(numHeadK_), static_cast<uint16_t>(padStride_), \
    static_cast<uint16_t>(rowElems), static_cast<uint16_t>(ropeStride_), \
    static_cast<uint16_t>(tileRows_), kvStep, eps_, recip_
        if (hd == 64) {
            SqnrNormRopeVfPack<T, true, 64, 64>(SQNR_VF_ARGS);
        } else if (hd == 96) {
            SqnrNormRopeVfPack<T, true, 96, 96>(SQNR_VF_ARGS);
        } else if (hd == 128) {
            SqnrNormRopeVfPack<T, false, 128, 128>(SQNR_VF_ARGS);
        } else if (hd == 256 && rd == 128) {
            SqnrNormRopeVfPack<T, false, 256, 128>(SQNR_VF_ARGS);
        } else if (hd == 256) {
            SqnrNormRopeVfPack<T, false, 256, 256>(SQNR_VF_ARGS);
        } else {
            SqnrNormRopeVfPack<T, false, 128, 128>(SQNR_VF_ARGS);  // 不可达：host 已校验支持集
        }
#undef SQNR_VF_ARGS
        AscendC::PipeBarrier<PIPE_V>();
        inQkv_.FreeTensor(xb);
        inCos_.FreeTensor(cb);
        inSin_.FreeTensor(sb);
        AscendC::PipeBarrier<PIPE_V>();
        outTile_.EnQue(ob);
    }

    __aicore__ inline void CopyOutTile(uint32_t k)
    {
        const uint32_t t0 = blkStart_ + k * tileRows_;
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
        const uint32_t rowElems = numHeadTotal_ * headDim_;
        const uint32_t elemBytes = sizeof(T);
        if (dstHeads == srcHeads) {
            uint32_t segBytes = dstHeads * headDim_ * elemBytes;
            DataCopyPad(gm[static_cast<uint64_t>(t0) * dstHeads * headDim_], src[srcHeadBase * headDim_],
                DataCopyExtParams{static_cast<uint16_t>(nTok), segBytes,
                    (rowElems * elemBytes - segBytes) / 32, 0, 0});
        } else {
            // GQA 扩张：dest 每 token 行内 head 之间不连续（跨 dstHeads*D），dstStride 单位是字节
            uint32_t headBytes = headDim_ * elemBytes;
            uint32_t srcStride = (rowElems - headDim_) * elemBytes / 32;
            uint32_t dstStride = (dstHeads - 1) * headDim_ * elemBytes;
            for (uint32_t h = 0; h < dstHeads; ++h) {
                uint32_t sh = h / gqaGroup_;
                DataCopyPad(gm[(static_cast<uint64_t>(t0) * dstHeads + h) * headDim_],
                    src[(srcHeadBase + sh) * headDim_],
                    DataCopyExtParams{static_cast<uint16_t>(nTok), headBytes, srcStride, dstStride, 0});
            }
        }
    }

    // [B,N,S,D]：同 head 内沿 S 连续，跨 batch 边界时拆段
    __aicore__ inline void CopyOutBnsd(const GlobalTensor<T> &gm, const LocalTensor<T> &src,
                                      uint32_t t0, uint32_t nTok, uint32_t srcHeadBase, uint32_t dstHeads,
                                      uint32_t srcHeads)
    {
        const uint32_t rowElems = numHeadTotal_ * headDim_;
        const uint32_t elemBytes = sizeof(T);
        const uint32_t headBytes = headDim_ * elemBytes;
        const uint32_t srcStride = (rowElems - headDim_) * elemBytes / 32;
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
                const uint64_t dstOff =
                    (static_cast<uint64_t>(b) * dstHeads + h) * seqLen_ * headDim_ +
                    static_cast<uint64_t>(s) * headDim_;
                DataCopyPad(gm[dstOff], src[done * rowElems + (srcHeadBase + sh) * headDim_],
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
    uint32_t headDim_ = kHeadDim;
    uint32_t halfDim_ = kHalfDim;
    uint32_t ropeDim_ = kHeadDim;  //0/缺省 = 全 head_dim 旋转
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
    uint32_t tileRows_ = kTokenTile;
    float eps_ = 1e-6f;
    float recip_ = 1.0f / 128.0f;
    bool gqaExpand_ = false;
    bool layoutBNSD_ = false;
    bool kvInterleaved_ = false;
};

}  // namespace kernels

#endif  // SPLIT_QKV_NORM_ROPE_KERNEL_H_
