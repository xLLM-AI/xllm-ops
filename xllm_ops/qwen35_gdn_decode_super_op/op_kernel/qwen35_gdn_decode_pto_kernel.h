/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "kernel_operator.h"
#include <pto/pto-inst.hpp>

#include <cstdint>
#include <type_traits>

namespace qwen35_decode_pto {

using namespace pto;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::ColMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T1, typename T2, int32_t Shape1, int32_t Shape2,
          int32_t Shape3, int32_t Shape4, int32_t Shape5, int32_t Stride1,
          int32_t Stride2, int32_t Stride3, int32_t Stride4, int32_t Stride5,
          uint32_t UbRows, uint32_t UbCols,
          pto::PadValue PadVal = pto::PadValue::Null>
AICORE PTO_INLINE void CopyGmToUb(__gm__ T1 *handle, int32_t ubAddress,
                                  int32_t ubOffset, int32_t validRows,
                                  int32_t validCols)
{
    static_assert(std::is_same_v<T1, T2>);
    pto::Shape<Shape1, Shape2, Shape3, pto::DYNAMIC, pto::DYNAMIC> shape;
    shape.shape[3] = validRows;
    shape.shape[4] = validCols;
    pto::GlobalTensor<
        T1, pto::Shape<Shape1, Shape2, Shape3, pto::DYNAMIC, pto::DYNAMIC>,
        pto::Stride<Stride1, Stride2, Stride3, Stride4, Stride5>>
        tensor(handle, shape);

    TileUbDataND<T2, UbRows, UbCols, pto::DYNAMIC, pto::DYNAMIC, PadVal>
        tile(validRows, validCols);
    pto::TASSIGN(tile, ubAddress + ubOffset * sizeof(T2));
    pto::TLOAD(tile, tensor);

    if constexpr (PadVal != pto::PadValue::Null) {
        if (validRows != static_cast<int32_t>(UbRows) ||
            validCols != static_cast<int32_t>(UbCols)) {
            TileUbDataND<T2, UbRows, UbCols, UbRows, UbCols, PadVal> padded;
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            pto::TASSIGN(padded, ubAddress + ubOffset * sizeof(T2));
            pto::TFILLPAD_INPLACE(padded, tile);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename T1, typename T2, int32_t Shape1, int32_t Shape2,
          int32_t Shape3, int32_t Shape4, int32_t Shape5, int32_t Stride1,
          int32_t Stride2, int32_t Stride3, int32_t Stride4, int32_t Stride5,
          uint32_t UbRows, uint32_t UbCols>
AICORE PTO_INLINE void CopyUbToGm(__gm__ T1 *handle, int32_t ubAddress,
                                  int32_t ubOffset, int32_t validRows,
                                  int32_t validCols)
{
    static_assert(std::is_same_v<T1, T2>);
    pto::Shape<Shape1, Shape2, Shape3, pto::DYNAMIC, pto::DYNAMIC> shape;
    shape.shape[3] = validRows;
    shape.shape[4] = validCols;
    pto::GlobalTensor<
        T1, pto::Shape<Shape1, Shape2, Shape3, pto::DYNAMIC, pto::DYNAMIC>,
        pto::Stride<Stride1, Stride2, Stride3, Stride4, Stride5>>
        tensor(handle, shape);

    constexpr bool kUseNd = static_cast<uint64_t>(UbCols) * sizeof(T2) >= 32;
    if constexpr (kUseNd) {
        TileUbDataND<T2, UbRows, UbCols, pto::DYNAMIC, pto::DYNAMIC> tile(
            validRows, validCols);
        pto::TASSIGN(tile, ubAddress + ubOffset * sizeof(T2));
        pto::TSTORE(tensor, tile);
    } else {
        TileUbDataDN<T2, UbRows, UbCols, pto::DYNAMIC, pto::DYNAMIC> tile(
            validRows, validCols);
        pto::TASSIGN(tile, ubAddress + ubOffset * sizeof(T2));
        pto::TSTORE(tensor, tile);
    }
}

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void Sigmoid(TileUbDataND<T, Rows, Cols> &dst,
                               TileUbDataND<T, Rows, Cols> &src)
{
    TMULS(src, src, -1);
    pipe_barrier(PIPE_V);
    TEXP(src, src);
    pipe_barrier(PIPE_V);
    TADDS(src, src, 1);
    pipe_barrier(PIPE_V);
    TRECIP(dst, src);
}

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void Silu(TileUbDataND<T, Rows, Cols> &dst,
                            TileUbDataND<T, Rows, Cols> &src,
                            TileUbDataND<T, Rows, Cols> &tmp)
{
    TMOV(tmp, src);
    pipe_barrier(PIPE_V);
    Sigmoid(dst, src);
    pipe_barrier(PIPE_V);
    TMUL(dst, tmp, dst);
}

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void MulAddDst(TileUbDataND<T, Rows, Cols> &dst,
                                 TileUbDataND<T, Rows, Cols> &src0,
                                 TileUbDataND<T, Rows, Cols> &src1,
                                 TileUbDataND<T, Rows, Cols> &tmp)
{
    TMUL(tmp, src0, src1);
    pipe_barrier(PIPE_V);
    TADD(dst, dst, tmp);
}

template <typename TileDst, typename TileRow, typename TileCol>
__tf__ PTO_INTERNAL void OuterProduct128Impl(
    typename TileDst::TileDType __out__ dst,
    typename TileRow::TileDType __in__ row,
    typename TileCol::TileDType __in__ col)
{
    __ubuf__ float *dstPtr =
        reinterpret_cast<__ubuf__ float *>(__cce_get_tile_ptr(dst));
    __ubuf__ float *rowPtr =
        reinterpret_cast<__ubuf__ float *>(__cce_get_tile_ptr(row));
    __ubuf__ uint32_t *colPtr =
        reinterpret_cast<__ubuf__ uint32_t *>(__cce_get_tile_ptr(col));
    __ubuf__ uint32_t *broadcast =
        reinterpret_cast<__ubuf__ uint32_t *>(TMP_UB_OFFSET);
    __ubuf__ float *colValues = reinterpret_cast<__ubuf__ float *>(broadcast);

    vbrcb(broadcast, colPtr, 1, 8, 16);
    pipe_barrier(PIPE_V);
    vmul(dstPtr, rowPtr, colValues, 128, 1, 1, 0, 16, 0, 1);
    vmul(dstPtr + 64, rowPtr + 64, colValues, 128, 1, 1, 0, 16, 0, 1);
}

template <typename TileDst, typename TileRow, typename TileCol>
AICORE PTO_INLINE void OuterProduct128(TileDst &dst, TileRow &row, TileCol &col)
{
    OuterProduct128Impl<TileDst, TileRow, TileCol>(dst.data(), row.data(),
                                                   col.data());
}

template <typename TileDst, typename TileSrc>
__tf__ PTO_INTERNAL void ColSum128Impl(typename TileDst::TileDType __out__ dst,
                                      typename TileSrc::TileDType __in__ src)
{
    __ubuf__ float *dstPtr =
        reinterpret_cast<__ubuf__ float *>(__cce_get_tile_ptr(dst));
    __ubuf__ float *srcPtr =
        reinterpret_cast<__ubuf__ float *>(__cce_get_tile_ptr(src));

    set_mask_count();
    set_vector_mask(0, 128);
#define QWEN35_COLSUM_STAGE(rows)                                             \
    for (uint32_t i = 0; i < (rows) / 2; ++i) {                              \
        vadd(srcPtr + i * 128, srcPtr + 2 * i * 128,                         \
             srcPtr + (2 * i + 1) * 128, 0, 1, 1, 1, 8, 8, 8);              \
    }                                                                         \
    pipe_barrier(PIPE_V)
    QWEN35_COLSUM_STAGE(128);
    QWEN35_COLSUM_STAGE(64);
    QWEN35_COLSUM_STAGE(32);
    QWEN35_COLSUM_STAGE(16);
    QWEN35_COLSUM_STAGE(8);
    QWEN35_COLSUM_STAGE(4);
    QWEN35_COLSUM_STAGE(2);
#undef QWEN35_COLSUM_STAGE
    set_mask_norm();
    set_vector_mask(-1, -1);
    copy_ubuf_to_ubuf(dstPtr, srcPtr, 0, 1, 16, 0, 0);
    pipe_barrier(PIPE_V);
}

template <typename TileDst, typename TileSrc>
AICORE PTO_INLINE void ColSum128(TileDst &dst, TileSrc &src)
{
    ColSum128Impl<TileDst, TileSrc>(dst.data(), src.data());
}

constexpr uint16_t kSyncAivOnlyFlag = 14;
constexpr uint16_t kSyncModeShift = 4;
constexpr uint16_t kSyncFlagShift = 8;

AICORE PTO_INLINE uint16_t GetFftsMessage(uint16_t mode, uint16_t flag)
{
    return 0x1 + ((mode & 0x3) << kSyncModeShift) +
           ((flag & 0xf) << kSyncFlagShift);
}

AICORE PTO_INLINE void SyncAllAiv()
{
    pipe_barrier(PIPE_ALL);
    ffts_cross_core_sync(PIPE_MTE3, GetFftsMessage(0, kSyncAivOnlyFlag));
    wait_flag_dev(kSyncAivOnlyFlag);
}

// The generated PTO kernel body follows. The tile shape stays fixed while
// model-dependent tensor strides and loop bounds come from host tiling data.

template <bool IsBatchOne>
AICORE PTO_INLINE void Run(
    __gm__ bfloat16_t *qkv_handle, __gm__ bfloat16_t *z_handle,
    __gm__ bfloat16_t *b_handle, __gm__ bfloat16_t *a_handle,
    __gm__ bfloat16_t *conv_weight_handle,
    __gm__ bfloat16_t *conv_state_handle, __gm__ float *a_log_handle,
    __gm__ float *dt_bias_handle, __gm__ float *ssm_state_handle,
    __gm__ int *state_indices_handle, __gm__ bfloat16_t *norm_weight_handle,
    __gm__ bfloat16_t *conv_out_handle,
    __gm__ bfloat16_t *conv_state_out_handle,
    __gm__ float *ssm_state_out_handle, __gm__ bfloat16_t *out_handle,
    int32_t num_k_heads, int32_t num_v_heads, int32_t runtime_batch_size)
{
  constexpr int32_t kCompiledNumCacheSlots = 1024;
  constexpr int32_t kHeadDim = 128;
  constexpr int32_t kConvStateLen = 3;
  constexpr int32_t kSsmHeadElements = kHeadDim * kHeadDim;
  constexpr int32_t kMaxNumKHeads = 16;
  constexpr int32_t kMaxNumVHeads = 64;
  constexpr int32_t kMaxBatchSize = 32;
  constexpr int32_t kMaxConvDim =
      (2 * kMaxNumKHeads + kMaxNumVHeads) * kHeadDim;
  constexpr int32_t kMaxConvWeightElements = 4 * kMaxConvDim;
  constexpr int32_t kMaxConvStateElements =
      kCompiledNumCacheSlots * kConvStateLen * kMaxConvDim;
  constexpr int32_t kMaxSsmStateElements =
      kCompiledNumCacheSlots * kMaxNumVHeads * kSsmHeadElements;
  const int32_t batch_size = IsBatchOne ? 1 : runtime_batch_size;
  const int32_t conv_dim =
      (2 * num_k_heads + num_v_heads) * kHeadDim;
  const int32_t conv_tile_count = conv_dim / kHeadDim;
  const int32_t conv_state_stride = kConvStateLen * conv_dim;
  const int32_t v_heads_per_k = num_v_heads / num_k_heads;
  const int32_t v_width = num_v_heads * kHeadDim;
  const int32_t ssm_state_stride = num_v_heads * kSsmHeadElements;
  auto cid = get_block_idx();

  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> w_half0;
  TASSIGN(w_half0, 0);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> w_half1;
  TASSIGN(w_half1, 256);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> w_half2;
  TASSIGN(w_half2, 512);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> w_half3;
  TASSIGN(w_half3, 768);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> w0;
  TASSIGN(w0, 1024);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> w1;
  TASSIGN(w1, 1536);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> w2;
  TASSIGN(w2, 2048);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> w3;
  TASSIGN(w3, 2560);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half0;
  TASSIGN(hist_half0, 3072);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half1;
  TASSIGN(hist_half1, 3328);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> hist_half2;
  TASSIGN(hist_half2, 3584);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> x_half;
  TASSIGN(x_half, 3840);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> hist0;
  TASSIGN(hist0, 4096);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> hist1;
  TASSIGN(hist1, 4608);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> hist2;
  TASSIGN(hist2, 5120);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> x_fp32;
  TASSIGN(x_fp32, 5632);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> conv_acc;
  TASSIGN(conv_acc, 6144);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> conv_tmp;
  TASSIGN(conv_tmp, 6656);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> conv_y;
  TASSIGN(conv_y, 7168);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> y_half;
  TASSIGN(y_half, 7680);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half0;
  TASSIGN(save_half0, 7936);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half1;
  TASSIGN(save_half1, 8192);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> save_half2;
  TASSIGN(save_half2, 8448);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> q_half;
  TASSIGN(q_half, 8704);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> k_half;
  TASSIGN(k_half, 8960);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 16, 1, 1> a_half;
  TASSIGN(a_half, 9216);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 16, 1, 1> b_half;
  TASSIGN(b_half, 9248);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> q_fp32;
  TASSIGN(q_fp32, 9280);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> k_fp32;
  TASSIGN(k_fp32, 9792);
  qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar;
  TASSIGN(scalar, 10304);
  qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar2;
  TASSIGN(scalar2, 10336);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> norm_sq;
  TASSIGN(norm_sq, 10368);
  qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> norm_val;
  TASSIGN(norm_val, 10880);
  qwen35_decode_pto::TileUbDataND<uint8_t, 128, 64, 128, 64> tmp_ub;
  TASSIGN(tmp_ub, 10912);
  qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp;
  TASSIGN(scalar_tmp, 19104);
  qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> exp_a_buf;
  TASSIGN(exp_a_buf, 19136);
  qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work;
  TASSIGN(scalar_work, 19168);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> norm_half;
  TASSIGN(norm_half, 152320);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> z_half;
  TASSIGN(z_half, 152576);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> weight_half;
  TASSIGN(weight_half, 152832);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> norm_fp32;
  TASSIGN(norm_fp32, 153088);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> z_fp32;
  TASSIGN(z_fp32, 153600);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> weight_fp32;
  TASSIGN(weight_fp32, 154112);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> square_fp32;
  TASSIGN(square_fp32, 154624);
  qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> rms;
  TASSIGN(rms, 155136);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> gate_fp32;
  TASSIGN(gate_fp32, 155168);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> final_half;
  TASSIGN(final_half, 155680);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> v_half;
  TASSIGN(v_half, 19200);
  qwen35_decode_pto::TileUbDataND<float, 128, 128, 128, 128> h_vec;
  TASSIGN(h_vec, 19456);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> v_fp32;
  TASSIGN(v_fp32, 84992);
  qwen35_decode_pto::TileUbDataND<float, 128, 128, 128, 128> compute_buf;
  TASSIGN(compute_buf, 85504);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> pred;
  TASSIGN(pred, 151040);
  qwen35_decode_pto::TileUbDataND<float, 1, 128, 1, 128> delta;
  TASSIGN(delta, 151552);
  qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 128, 1, 128> out_half;
  TASSIGN(out_half, 152064);
  auto vid = get_subblockid();
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
  const int32_t vector_core_idx = cid * get_subblockdim() + vid;
  const int32_t vector_core_count = get_block_num() * get_subblockdim();
  const int32_t batch_one_state_idx =
      IsBatchOne ? *state_indices_handle : 0;

  for (int32_t conv_tile = vector_core_idx; conv_tile < conv_tile_count;
       conv_tile += vector_core_count) {
    const int32_t channel_offset = conv_tile * kHeadDim;
    qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvWeightElements, 1, 1, 128, pto::PadValue::Zero>(conv_weight_handle + channel_offset, 0, 0, 1, 128);
    qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvWeightElements, 1, 1, 128, pto::PadValue::Zero>(conv_weight_handle + channel_offset + conv_dim, 256, 0, 1, 128);
    qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvWeightElements, 1, 1, 128, pto::PadValue::Zero>(conv_weight_handle + channel_offset + 2 * conv_dim, 512, 0, 1, 128);
    qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvWeightElements, 1, 1, 128, pto::PadValue::Zero>(conv_weight_handle + channel_offset + 3 * conv_dim, 768, 0, 1, 128);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCVT(w0, w_half0, RoundMode::CAST_NONE);
    TCVT(w1, w_half1, RoundMode::CAST_NONE);
    TCVT(w2, w_half2, RoundMode::CAST_NONE);
    TCVT(w3, w_half3, RoundMode::CAST_NONE);

  for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      if constexpr (!IsBatchOne) {
        pipe_barrier(PIPE_ALL);
        pipe_barrier(PIPE_ALL);
      }
        const int32_t state_idx = IsBatchOne
            ? batch_one_state_idx
            : *(state_indices_handle + batch_idx);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvStateElements, 1, 1, 128, pto::PadValue::Zero>(conv_state_handle + state_idx * conv_state_stride + channel_offset, 3072, 0, 1, 128);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvStateElements, 1, 1, 128, pto::PadValue::Zero>(conv_state_handle + state_idx * conv_state_stride + channel_offset + conv_dim, 3328, 0, 1, 128);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvStateElements, 1, 1, 128, pto::PadValue::Zero>(conv_state_handle + state_idx * conv_state_stride + channel_offset + 2 * conv_dim, 3584, 0, 1, 128);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxBatchSize * kMaxConvDim, 1, 1, 128, pto::PadValue::Zero>(qkv_handle + batch_idx * conv_dim + channel_offset, 3840, 0, 1, 128);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TCVT(hist0, hist_half0, RoundMode::CAST_NONE);
        TCVT(hist1, hist_half1, RoundMode::CAST_NONE);
        TCVT(hist2, hist_half2, RoundMode::CAST_NONE);
        TCVT(x_fp32, x_half, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TMUL(conv_acc, w0, hist0);
        TMUL(conv_tmp, w1, hist1);
        pipe_barrier(PIPE_V);
        TADD(conv_acc, conv_acc, conv_tmp);
        pipe_barrier(PIPE_V);
        TMUL(conv_tmp, w2, hist2);
        pipe_barrier(PIPE_V);
        TADD(conv_acc, conv_acc, conv_tmp);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 128> conv_acc_temp_0_muladddst_tmp;
        TASSIGN(conv_acc_temp_0_muladddst_tmp, 155936);
        qwen35_decode_pto::MulAddDst<float, 1, 128>(conv_acc, x_fp32, w3, conv_acc_temp_0_muladddst_tmp);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 128> conv_y_temp_0_silu_tmp;
        TASSIGN(conv_y_temp_0_silu_tmp, 155936);
        qwen35_decode_pto::Silu<float, 1, 128>(conv_y, conv_acc, conv_y_temp_0_silu_tmp);
        pipe_barrier(PIPE_V);
        TCVT(y_half, conv_y, RoundMode::CAST_RINT);
        TCVT(save_half0, hist1, RoundMode::CAST_RINT);
        TCVT(save_half1, hist2, RoundMode::CAST_RINT);
        TCVT(save_half2, x_fp32, RoundMode::CAST_RINT);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        qwen35_decode_pto::CopyUbToGm<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxBatchSize * kMaxConvDim, 1, 1, 128>(conv_out_handle + batch_idx * conv_dim + channel_offset, 7680, 0, 1, 128);
        qwen35_decode_pto::CopyUbToGm<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvStateElements, 1, 1, 128>(conv_state_out_handle + state_idx * conv_state_stride + channel_offset, 7936, 0, 1, 128);
        pipe_barrier(PIPE_MTE3);
        qwen35_decode_pto::CopyUbToGm<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvStateElements, 1, 1, 128>(conv_state_out_handle + state_idx * conv_state_stride + channel_offset + conv_dim, 8192, 0, 1, 128);
        pipe_barrier(PIPE_MTE3);
        qwen35_decode_pto::CopyUbToGm<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxConvStateElements, 1, 1, 128>(conv_state_out_handle + state_idx * conv_state_stride + channel_offset + 2 * conv_dim, 8448, 0, 1, 128);
      if constexpr (!IsBatchOne) {
        pipe_barrier(PIPE_ALL);
        pipe_barrier(PIPE_ALL);
      }
    }
    if constexpr (IsBatchOne) {
      if (conv_tile + vector_core_count < conv_tile_count) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      }
    }
  }
    qwen35_decode_pto::SyncAllAiv();

  for (int32_t head_index = vector_core_idx;
       head_index < batch_size * num_v_heads;
       head_index += vector_core_count) {
      if constexpr (!IsBatchOne) {
        pipe_barrier(PIPE_ALL);
        pipe_barrier(PIPE_ALL);
      }
        const int32_t batch_idx =
            IsBatchOne ? 0 : head_index / num_v_heads;
        const int32_t head_idx =
            IsBatchOne ? head_index : head_index % num_v_heads;
        const int32_t qk_head_idx = head_idx / v_heads_per_k;
        const int32_t state_idx_1 = IsBatchOne
            ? batch_one_state_idx
            : *(state_indices_handle + batch_idx);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxBatchSize * kMaxConvDim, 1, 1, 128, pto::PadValue::Zero>(conv_out_handle + batch_idx * conv_dim + qk_head_idx * kHeadDim, 8704, 0, 1, 128);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxBatchSize * kMaxConvDim, 1, 1, 128, pto::PadValue::Zero>(conv_out_handle + batch_idx * conv_dim + num_k_heads * kHeadDim + qk_head_idx * kHeadDim, 8960, 0, 1, 128);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 1, 1, 1, 1, kMaxBatchSize * kMaxNumVHeads, 1, 1, 16, pto::PadValue::Zero>(a_handle + head_index, 9216, 0, 1, 1);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 1, 1, 1, 1, kMaxBatchSize * kMaxNumVHeads, 1, 1, 16, pto::PadValue::Zero>(b_handle + head_index, 9248, 0, 1, 1);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        TCVT(q_fp32, q_half, RoundMode::CAST_NONE);
        TCVT(k_fp32, k_half, RoundMode::CAST_NONE);
        qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 16, 1, 1> a_half_temp_0;
        TASSIGN(a_half_temp_0, 9216 + 0 * 2);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_temp_0;
        TASSIGN(scalar_temp_0, 10304 + 0 * 4);
        TCVT(scalar_temp_0, a_half_temp_0, RoundMode::CAST_NONE);
        qwen35_decode_pto::TileUbDataND<bfloat16_t, 1, 16, 1, 1> b_half_temp_0;
        TASSIGN(b_half_temp_0, 9248 + 0 * 2);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar2_temp_0;
        TASSIGN(scalar2_temp_0, 10336 + 0 * 4);
        TCVT(scalar2_temp_0, b_half_temp_0, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TMUL(norm_sq, q_fp32, q_fp32);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataDN<float, 8, 1, 1, 1> norm_val_temp_0;
        TASSIGN(norm_val_temp_0, 10880 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 64, 1, 64> tmp_ub_temp_0;
        TASSIGN(tmp_ub_temp_0, 10912 + 0 * 4);
        TROWSUM(norm_val_temp_0, norm_sq, tmp_ub_temp_0);
        pipe_barrier(PIPE_V);
        TADDS(norm_val, norm_val, 1.000000e-06f);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> norm_val_temp_1;
        TASSIGN(norm_val_temp_1, 10880 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_0;
        TASSIGN(scalar_tmp_temp_0, 19104 + 0 * 4);
        TSQRT(scalar_tmp_temp_0, norm_val_temp_1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float q_norm_scalar = scalar_tmp.GetValue(0);
        TMULS(q_fp32, q_fp32, 1.0f / q_norm_scalar);
        pipe_barrier(PIPE_V);
        TMULS(q_fp32, q_fp32, 8.838835e-02f);
        TMUL(norm_sq, k_fp32, k_fp32);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataDN<float, 8, 1, 1, 1> norm_val_temp_2;
        TASSIGN(norm_val_temp_2, 10880 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 64, 1, 64> tmp_ub_temp_1;
        TASSIGN(tmp_ub_temp_1, 10912 + 0 * 4);
        TROWSUM(norm_val_temp_2, norm_sq, tmp_ub_temp_1);
        pipe_barrier(PIPE_V);
        TADDS(norm_val, norm_val, 1.000000e-06f);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> norm_val_temp_3;
        TASSIGN(norm_val_temp_3, 10880 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_1;
        TASSIGN(scalar_tmp_temp_1, 19104 + 0 * 4);
        TSQRT(scalar_tmp_temp_1, norm_val_temp_3);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float k_norm_scalar = scalar_tmp.GetValue(0);
        TMULS(k_fp32, k_fp32, 1.0f / k_norm_scalar);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        qwen35_decode_pto::CopyGmToUb<float, float, 1, 1, 1, 1, 1, 1, 1, 1, kMaxNumVHeads, 1, 1, 8, pto::PadValue::Zero>(a_log_handle + head_idx, 19104, 0, 1, 1);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_2;
        TASSIGN(scalar_tmp_temp_2, 19104 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> exp_a_buf_temp_0;
        TASSIGN(exp_a_buf_temp_0, 19136 + 0 * 4);
        TEXP(exp_a_buf_temp_0, scalar_tmp_temp_2);
        qwen35_decode_pto::CopyGmToUb<float, float, 1, 1, 1, 1, 1, 1, 1, 1, kMaxNumVHeads, 1, 1, 8, pto::PadValue::Zero>(dt_bias_handle + head_idx, 10880, 0, 1, 1);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_temp_1;
        TASSIGN(scalar_temp_1, 10304 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> norm_val_temp_4;
        TASSIGN(norm_val_temp_4, 10880 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_3;
        TASSIGN(scalar_tmp_temp_3, 19104 + 0 * 4);
        TADD(scalar_tmp_temp_3, scalar_temp_1, norm_val_temp_4);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float x_gate = scalar_tmp.GetValue(0);
        if (2.000000e+01f < x_gate) {
          TADDS(scalar_work, scalar_tmp, 0.000000e+00f);
        } else {
          qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_4;
          TASSIGN(scalar_tmp_temp_4, 19104 + 0 * 4);
          qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_0;
          TASSIGN(scalar_work_temp_0, 19168 + 0 * 4);
          TEXP(scalar_work_temp_0, scalar_tmp_temp_4);
          pipe_barrier(PIPE_V);
          TADDS(norm_val, scalar_work, 1.000000e+00f);
          pipe_barrier(PIPE_V);
          qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> norm_val_temp_5;
          TASSIGN(norm_val_temp_5, 10880 + 0 * 4);
          qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_1;
          TASSIGN(scalar_work_temp_1, 19168 + 0 * 4);
          TLOG(scalar_work_temp_1, norm_val_temp_5);
        }
        pipe_barrier(PIPE_ALL);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> exp_a_buf_temp_1;
        TASSIGN(exp_a_buf_temp_1, 19136 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_2;
        TASSIGN(scalar_work_temp_2, 19168 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_5;
        TASSIGN(scalar_tmp_temp_5, 19104 + 0 * 4);
        TMUL(scalar_tmp_temp_5, exp_a_buf_temp_1, scalar_work_temp_2);
        pipe_barrier(PIPE_V);
        TMULS(scalar_tmp, scalar_tmp, -1.000000e+00f);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_6;
        TASSIGN(scalar_tmp_temp_6, 19104 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_3;
        TASSIGN(scalar_work_temp_3, 19168 + 0 * 4);
        TEXP(scalar_work_temp_3, scalar_tmp_temp_6);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float decay = scalar_work.GetValue(0);
        pipe_barrier(PIPE_V);
        TMULS(scalar_tmp, scalar2, -1.000000e+00f);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_7;
        TASSIGN(scalar_tmp_temp_7, 19104 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_4;
        TASSIGN(scalar_work_temp_4, 19168 + 0 * 4);
        TEXP(scalar_work_temp_4, scalar_tmp_temp_7);
        pipe_barrier(PIPE_V);
        TADDS(scalar_tmp, scalar_work, 1.000000e+00f);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_8;
        TASSIGN(scalar_tmp_temp_8, 19104 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_work_temp_5;
        TASSIGN(scalar_work_temp_5, 19168 + 0 * 4);
        TRECIP(scalar_work_temp_5, scalar_tmp_temp_8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float beta_gate = scalar_work.GetValue(0);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxBatchSize * kMaxConvDim, 1, 1, 128, pto::PadValue::Zero>(conv_out_handle + batch_idx * conv_dim + 2 * num_k_heads * kHeadDim + head_idx * kHeadDim, 19200, 0, 1, 128);
        qwen35_decode_pto::CopyGmToUb<float, float, 1, 1, 1, 128, 128, kMaxSsmStateElements, kMaxNumVHeads * kSsmHeadElements, kSsmHeadElements, 128, 1, 128, 128, pto::PadValue::Zero>(ssm_state_handle + state_idx_1 * ssm_state_stride + head_idx * kSsmHeadElements, 19456, 0, 128, 128);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TCVT(v_fp32, v_half, RoundMode::CAST_NONE);
        TMULS(h_vec, h_vec, decay);
        pipe_barrier(PIPE_V);
        {
          qwen35_decode_pto::TileUbDataDN<float, 128, 1, 128, 1> k_row;
          TASSIGN(k_row, reinterpret_cast<std::uintptr_t>(k_fp32.data()));
          TROWEXPANDMUL(compute_buf, h_vec, k_row);
        }
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::ColSum128(
            pred, compute_buf);
        TSUB(delta, v_fp32, pred);
        pipe_barrier(PIPE_V);
        TMULS(delta, delta, beta_gate);
        pipe_barrier(PIPE_V);
        {
          qwen35_decode_pto::TileUbDataDN<float, 128, 1, 128, 1> k_row;
          TASSIGN(k_row, reinterpret_cast<std::uintptr_t>(k_fp32.data()));
          qwen35_decode_pto::OuterProduct128(
              compute_buf, delta, k_row);
        }
        pipe_barrier(PIPE_V);
        TADD(h_vec, h_vec, compute_buf);
        pipe_barrier(PIPE_V);
        {
          qwen35_decode_pto::TileUbDataDN<float, 128, 1, 128, 1> q_row;
          TASSIGN(q_row, reinterpret_cast<std::uintptr_t>(q_fp32.data()));
          TROWEXPANDMUL(compute_buf, h_vec, q_row);
        }
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::ColSum128(
            pred, compute_buf);
        TCVT(out_half, pred, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        TMOV(norm_half, out_half);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        qwen35_decode_pto::CopyUbToGm<float, float, 1, 1, 1, 128, 128, kMaxSsmStateElements, kMaxNumVHeads * kSsmHeadElements, kSsmHeadElements, 128, 1, 128, 128>(ssm_state_out_handle + state_idx_1 * ssm_state_stride + head_idx * kSsmHeadElements, 19456, 0, 128, 128);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxBatchSize * kMaxNumVHeads * kHeadDim, 1, 1, 128, pto::PadValue::Zero>(z_handle + batch_idx * v_width + head_idx * kHeadDim, 152576, 0, 1, 128);
        qwen35_decode_pto::CopyGmToUb<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, 128, 1, 1, 128, pto::PadValue::Zero>(norm_weight_handle + 0, 152832, 0, 1, 128);
        pipe_barrier(PIPE_V);
        TCVT(norm_fp32, norm_half, RoundMode::CAST_NONE);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        TCVT(z_fp32, z_half, RoundMode::CAST_NONE);
        TCVT(weight_fp32, weight_half, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TMUL(square_fp32, norm_fp32, norm_fp32);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataDN<float, 8, 1, 1, 1> rms_temp_0;
        TASSIGN(rms_temp_0, 155136 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 64, 1, 64> tmp_ub_temp_4;
        TASSIGN(tmp_ub_temp_4, 10912 + 0 * 4);
        TROWSUM(rms_temp_0, square_fp32, tmp_ub_temp_4);
        pipe_barrier(PIPE_V);
        TMULS(rms, rms, 1.0f / 1.280000e+02f);
        pipe_barrier(PIPE_V);
        TADDS(rms, rms, 1.000000e-06f);
        pipe_barrier(PIPE_V);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> rms_temp_1;
        TASSIGN(rms_temp_1, 155136 + 0 * 4);
        qwen35_decode_pto::TileUbDataND<float, 1, 8, 1, 1> scalar_tmp_temp_9;
        TASSIGN(scalar_tmp_temp_9, 19104 + 0 * 4);
        TSQRT(scalar_tmp_temp_9, rms_temp_1);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto scalar_tmp_scalar_temp_0 = scalar_tmp.GetValue(0);
        TMULS(norm_fp32, norm_fp32, 1.0f / scalar_tmp_scalar_temp_0);
        pipe_barrier(PIPE_V);
        TMUL(norm_fp32, norm_fp32, weight_fp32);
        qwen35_decode_pto::TileUbDataND<float, 1, 128> gate_fp32_temp_0_silu_tmp;
        TASSIGN(gate_fp32_temp_0_silu_tmp, 155936);
        qwen35_decode_pto::Silu<float, 1, 128>(gate_fp32, z_fp32, gate_fp32_temp_0_silu_tmp);
        pipe_barrier(PIPE_V);
        TMUL(norm_fp32, norm_fp32, gate_fp32);
        pipe_barrier(PIPE_V);
        TCVT(final_half, norm_fp32, RoundMode::CAST_RINT);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        qwen35_decode_pto::CopyUbToGm<bfloat16_t, bfloat16_t, 1, 1, 1, 1, 128, 1, 1, 1, kMaxBatchSize * kMaxNumVHeads * kHeadDim, 1, 1, 128>(out_handle + batch_idx * v_width + head_idx * kHeadDim, 155680, 0, 1, 128);
      pipe_barrier(PIPE_ALL);
      if constexpr (!IsBatchOne) {
        pipe_barrier(PIPE_ALL);
      }
    }
#endif
}

}  // namespace qwen35_decode_pto
