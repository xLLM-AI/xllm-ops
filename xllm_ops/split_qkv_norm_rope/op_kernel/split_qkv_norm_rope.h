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
 * \brief Fused SplitQKV + QK-RMSNorm + RoPE kernel (Ascend 950 / AIV).
 */

#ifndef SPLIT_QKV_NORM_ROPE_KERNEL_H_
#define SPLIT_QKV_NORM_ROPE_KERNEL_H_

#include "kernel_operator.h"

namespace kernels {
using namespace AscendC;

constexpr uint32_t kHeadDim = 128;           // default D; runtime D from tiling
constexpr uint32_t kHalfDim = kHeadDim / 2;
constexpr uint32_t kMaxHeads = 64;           // Nq+Nk upper bound (sumsq buffer)
constexpr uint32_t kTokenTile = 8;           // default tile rows; host picks 8/4/2/1 by UB budget

constexpr uint32_t kFlagGqaExpand = 1u << 0;       // expand k/v to Nq heads
constexpr uint32_t kFlagLayoutBNSD = 1u << 1;      // output [B,N,S,D]; else [B,S,N*D]
constexpr uint32_t kFlagKvInterleaved = 1u << 2;   // input KV per-head [K_i,V_i]; else [K|V]

template <typename T, bool kHalfMasked, uint16_t kD, uint16_t kRD>
__simd_vf__ inline void SqnrNormRopeVfPack(
    __ubuf__ T *outT, __ubuf__ T *xb, __ubuf__ T *cosT, __ubuf__ T *sinT,
    __ubuf__ float *gammaQ, __ubuf__ float *gammaK, __ubuf__ float *sumsq,
    uint16_t numHeadQ, uint16_t numHeadK, uint16_t xStride, uint16_t outStride, uint16_t ropeStride,
    uint16_t tileRows, uint16_t kvHeadStep, float eps, float recip)
{
    using namespace AscendC::Reg;
    // (kD, kRD) template params give compile-time loop trip counts; fully unrolled.
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
    // KV region base; q heads contiguous from row start, k heads stride by kvHeadStep
    const uint32_t kvHeadBase = static_cast<uint32_t>(numHeadQ) * headDim;
    // scale segments: ceil(tileRows*(Nq+Nk)/64)
    constexpr uint16_t kSeg = 64;
    const uint16_t segs = static_cast<uint16_t>((tileRows * qkHeads + kSeg - 1) / kSeg);

    if constexpr (kHalfMasked) {
    // ==== HalfMasked path: D in {64,96}, halfDim < 64 lanes ====
    // Rotate across two 64-lane registers via UpdateMask count mask (arch35 HalfAlignVF):
    // first half [0,halfDim), second half [64,64+halfDim) (gamma layout from CopyInGamma).
    // No passthrough: rope_dim is always 0 for D<128 (host-validated).

    // ---- pass 1: sumsq (tail chunk uses count mask; D=96 -> tail=32) ----
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
                // arch35 MulAddDst: ZEROING only. Masked Cast (layoutZMrgZ, inactive=0) then
                // full-mask MulAddDst keeps chunk0 sums in inactive lanes.
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
                // same: full-mask MulAddDst keeps chunk0 in inactive lanes
                MulAddDst(acc, y2, y2, mFull);
            }
            ReduceSum(acc, acc, mFull);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                sumsq + static_cast<uint32_t>(t) * qkHeads + numHeadQ + h, acc, mOne);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // ---- per-tile scale chain ----
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

    // ---- pass 2: norm + rope, one masked pair per half ----
    uint32_t halfCnt = static_cast<uint32_t>(halfDim);  // 32 (D=64) / 48 (D=96)
    MaskReg pregHalf = UpdateMask<float>(halfCnt);
    // gamma hoisted out of token loop (front [0,halfDim), back [64,64+halfDim))
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
        // q constants (gamma folded into cos/sin)
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
        // k constants (gamma folded; reuses cos/sin)
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
    // ==== Full path: D in {128,256} ====
    // 64-lane chunks: pass1 accumulates D/64 chunks; pass2 splits each half into
    // halfDim/64 pairs, first ropeDim/2/64 rotated, rest pass through.
    const uint16_t nChunks = static_cast<uint16_t>(headDim / kSeg);
    const uint16_t nPairs = static_cast<uint16_t>(halfDim / kSeg);
    const uint16_t nRopePairs = static_cast<uint16_t>(ropeDim / 2 / kSeg);

    // pass 1: sumsq per (token, head)
    for (uint16_t t = 0; t < tileRows; ++t) {
        __ubuf__ T *row = xb + static_cast<uint32_t>(t) * xStride;
        for (uint16_t h = 0; h < numHeadQ; ++h) {
            __ubuf__ T *src = row + static_cast<uint32_t>(h) * headDim;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src);
            Cast<float, T, AscendC::layoutZMrgZ>(y1, tv, mFull);
            Mul(acc, y1, y1, mFull);
            // D=128: loop runs once
            for (uint16_t c = 1; c < nChunks; ++c) {
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tv, src + static_cast<uint32_t>(c) * kSeg);
                Cast<float, T, AscendC::layoutZMrgZ>(y2, tv, mFull);
                MulAddDst(acc, y2, y2, mFull);
            }
            ReduceSum(acc, acc, mFull);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                sumsq + static_cast<uint32_t>(t) * qkHeads + h, acc, mOne);
        }
        // k heads: kvHeadStep = D (blocked) or 2D (interleaved, skip v)
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

    // per-tile scale chain, in-place
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

    // pass 2: norm + rope, pack-cast into T on store
    // gamma hoisted per pair (out of t loop); D=128 full-rope matches baseline stream.
    // q group: gamma folded into cos/sin; per 64-lane pair (D=128 -> nPairs=1)
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

    // q passthrough: unrotated chunks = norm*gamma (skipped when full rope)
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

    // k group: gamma folded into cos/sin
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

    // k passthrough: unrotated chunks = norm*gamma (skipped when full rope)
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
    }  // full path (D in {128,256})
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
        if (ropeDim_ == 0 || ropeDim_ > headDim_) {  // default/invalid: full rope
            ropeDim_ = headDim_;
        }

        gqaExpand_ = (tiling->flags & kFlagGqaExpand) != 0;
        layoutBNSD_ = (tiling->flags & kFlagLayoutBNSD) != 0;
        kvInterleaved_ = (tiling->flags & kFlagKvInterleaved) != 0;
        kvOutHeads_ = gqaExpand_ ? numHeadQ_ : numHeadK_;
        gqaGroup_ = (numHeadK_ == 0) ? 1 : (numHeadQ_ / numHeadK_);

        // u64 offsets: B*S*stride can exceed 2^32 elements
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

        // depth 2: prefetch next tile, defer writeout of previous
        pipe_->InitBuffer(inQkv_, 2, tileRows_ * padStride_ * sizeof(T));
        pipe_->InitBuffer(inCos_, 2, tileRows_ * ropeStride_ * sizeof(T));
        pipe_->InitBuffer(inSin_, 2, tileRows_ * ropeStride_ * sizeof(T));
        pipe_->InitBuffer(inGamma_, 2, headDim_ * sizeof(T));
        pipe_->InitBuffer(outTile_, 2, tileRows_ * numHeadTotal_ * headDim_ * sizeof(T));

        // D=64/96 (halfDim<64): gamma split front@0/back@64, buffer widened to 128 floats
        const uint32_t gammaElems = (halfDim_ < 64) ? 128u : headDim_;
        pipe_->InitBuffer(bufGammaQ_, gammaElems * sizeof(float));
        pipe_->InitBuffer(bufGammaK_, gammaElems * sizeof(float));
        pipe_->InitBuffer(bufSum_, tileRows_ * kMaxHeads * sizeof(float));

        // per-core token range
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        blkStart_ = blockIdx * tokenPerCore_;
        blkTokens_ = (blockIdx == blockDim_ - 1) ? lastBlkTokens_ : tokenPerCore_;
        tileNum_ = (blkTokens_ + tileRows_ - 1) / tileRows_;
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();

        // 3-stage pipeline: prefetch k+1, compute k, writeout k-1
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
            // D=64/96: front half [0,halfDim), back half [64,64+halfDim); excess lanes dropped by mask
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

    // issue DMA without waiting; depth-2 queues overlap with compute of previous tile
    __aicore__ inline void CopyInTile(uint32_t k)
    {
        const uint32_t t0 = blkStart_ + k * tileRows_;
        const uint32_t nTok = TileTokens(k);
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // per-token rows: GM row stride may be unaligned; UB lands on 32B-aligned padStride
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

        // output tile (pack-store target)
        AscendC::LocalTensor<T> ob = outTile_.AllocTensor<T>();
        const uint32_t rowElems = numHeadTotal_ * headDim_;
        const uint32_t qkElems = (numHeadQ_ + numHeadK_) * headDim_;
        const uint32_t vElems = numHeadV_ * headDim_;

        // v pass-through: UB->UB copy, no fp32 (blocked: one shot; interleaved: per head)
        if (!kvInterleaved_) {
            // UB-side blockLen/srcGap/dstGap in 32B blocks
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

        // norm + rope VF: load q/k/cos/sin directly from T, no cast-in
        AscendC::LocalTensor<float> sumL = bufSum_.Get<float>();
        AscendC::LocalTensor<float> gqf = bufGammaQ_.Get<float>();
        AscendC::LocalTensor<float> gkf = bufGammaK_.Get<float>();
        // v copy done; VF reads xb/cb/sb, so queues freed after VF.
        // Instantiate per (D, rope_dim) literal: D=64/96 HalfMasked, D=128 full rope,
        // D=256 full or partial rope (rope_dim in {128,256}).
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
            SqnrNormRopeVfPack<T, false, 128, 128>(SQNR_VF_ARGS);  // unreachable: host validates support set
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

    // [B,S,H*D]: heads contiguous per row
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
            // GQA expand: dest heads discontiguous per row; dstStride in bytes
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

    // [B,N,S,D]: S contiguous within a head; split at batch boundary
    __aicore__ inline void CopyOutBnsd(const GlobalTensor<T> &gm, const LocalTensor<T> &src,
                                      uint32_t t0, uint32_t nTok, uint32_t srcHeadBase, uint32_t dstHeads,
                                      uint32_t srcHeads)
    {
        const uint32_t rowElems = numHeadTotal_ * headDim_;
        const uint32_t elemBytes = sizeof(T);
        const uint32_t headBytes = headDim_ * elemBytes;
        const uint32_t srcStride = (rowElems - headDim_) * elemBytes / 32;
        const uint32_t dstStride = 0;  // tokens adjacent within a head

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
    uint32_t ropeDim_ = kHeadDim;  // 0/default = full head_dim rotation
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
