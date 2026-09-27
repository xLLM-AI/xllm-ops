/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#pragma once

#include "catlass/arch/resource.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"

// Derived from Catlass 1483c7d245b200c62571a4130e24453bc81c31ed.
// This local pair (WindowOnlineSoftmax / WindowRescaleO) owns its UB layout
// and event protocol. Do not mix either half with a Catlass epilogue: upstream
// private buffer offsets may change. QK/PV still use Catlass public interfaces.

namespace xllm_ops::xfia {

template <class OutputType,
          class InputType,
          class MaskType,
          Catlass::Epilogue::LseMode LseOutputMode>
class WindowOnlineSoftmax final {
 public:
  using ArchTag = Catlass::Arch::AtlasA2;
  using ElementOutput = typename OutputType::Element;
  using ElementInput = typename InputType::Element;
  using ElementMask = typename MaskType::Element;

  using LayoutOutput = typename OutputType::Layout;
  using LayoutInput = typename InputType::Layout;
  using LayoutMask = typename MaskType::Layout;

  static constexpr Catlass::Epilogue::LseMode LSE_MODE = LseOutputMode;

  static constexpr uint32_t kFloatBlockSize = 8;
  static constexpr uint32_t kFloatVectorSize = 64;
  static constexpr uint32_t kBlockSize = 16;
  static constexpr uint32_t kUbUint8VectorSize = 1024;
  static constexpr uint32_t kUbUint8BlockSize = 16384;
  static constexpr uint32_t kVectorSize = 128;
  static constexpr uint32_t kMaxUbSElemNum = 8192;

  static constexpr uint32_t kReduceUbSize = 1024;
  static constexpr uint32_t kRowOpsSpecMask32 = 32;
  static constexpr uint32_t kRowOpsSpecMask4 = 4;
  static constexpr uint32_t kMaxRowNumSubCore = 256;

  __aicore__ inline WindowOnlineSoftmax(
      Catlass::Arch::Resource<ArchTag>& resource,
      float scale_value) {
    // Allocate UB space
    constexpr uint32_t kLsUbTensorOffset = 0;
    constexpr uint32_t kLpUbTensorOffset = 4 * kUbUint8BlockSize;

    constexpr uint32_t kTvUbTensorOffset = 10 * kUbUint8BlockSize;
    constexpr uint32_t kLmUbTensorOffset =
        10 * kUbUint8BlockSize + 8 * kUbUint8VectorSize;

    constexpr uint32_t kHmUbTensorOffset =
        10 * kUbUint8BlockSize + 9 * kUbUint8VectorSize;
    constexpr uint32_t kGmUbTensorOffset =
        10 * kUbUint8BlockSize + 10 * kUbUint8VectorSize;
    constexpr uint32_t kLlUbTensorOffset =
        10 * kUbUint8BlockSize + 11 * kUbUint8VectorSize;
    constexpr uint32_t kGlUbTensorOffset =
        10 * kUbUint8BlockSize + 12 * kUbUint8VectorSize;
    constexpr uint32_t kDmUbTensorOffset =
        10 * kUbUint8BlockSize + 13 * kUbUint8VectorSize;

    scale_value_ = scale_value;
    ls_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kLsUbTensorOffset);
    lp_ub_tensor_ = resource.ubBuf.template GetBufferByByte<ElementOutput>(
        kLpUbTensorOffset);
    lm_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kLmUbTensorOffset);
    hm_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kHmUbTensorOffset);
    gm_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kGmUbTensorOffset);
    dm_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kDmUbTensorOffset);
    ll_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kLlUbTensorOffset);
    tv_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kTvUbTensorOffset);
    gl_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kGlUbTensorOffset);
  }

 private:
  __aicore__ inline void set_vec_mask(int32_t len) {
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % kFloatVectorSize;
    for (int64_t i = 0; i < temp; i++) {
      mask |= one << i;
    }

    if (len == kVectorSize || len == 0) {
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
    } else if (len >= kFloatVectorSize) {
      AscendC::SetVectorMask<int8_t>(mask, static_cast<uint64_t>(-1));
    } else {
      AscendC::SetVectorMask<int8_t>(0x0, mask);
    }
  }

  __aicore__ inline void set_block_reduce_mask(int32_t len) {
    if (len > 8 || len < 1) {
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
      return;
    }
    uint64_t sub_mask = (static_cast<uint64_t>(1) << len) - 1;
    uint64_t mask_value = (sub_mask << 48) + (sub_mask << 32) +
                          (sub_mask << 16) + sub_mask + (sub_mask << 56) +
                          (sub_mask << 40) + (sub_mask << 24) + (sub_mask << 8);
    AscendC::SetVectorMask<int8_t>(mask_value, mask_value);
  }

  __aicore__ inline void rowsum_spectile512(
      const AscendC::LocalTensor<float>& src_ub,
      const AscendC::LocalTensor<float>& rowsum_ub,
      const AscendC::LocalTensor<float>& tv_ub_tensor_,
      uint32_t num_rows_round,
      uint32_t num_elems,
      uint32_t num_elems_aligned) {
    AscendC::BlockReduceSum<float, false>(
        tv_ub_tensor_,
        src_ub,
        num_rows_round * num_elems_aligned / kFloatVectorSize,
        0,
        1,
        1,
        8);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::BlockReduceSum<float, false>(
        tv_ub_tensor_[kReduceUbSize],
        tv_ub_tensor_,
        num_rows_round * num_elems_aligned / kFloatBlockSize / kFloatVectorSize,
        0,
        1,
        1,
        8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum<float, false>(rowsum_ub,
                                          tv_ub_tensor_[kReduceUbSize],
                                          num_rows_round * num_elems_aligned /
                                              kFloatVectorSize /
                                              kFloatVectorSize,
                                          0,
                                          1,
                                          1,
                                          8);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void rowsum_spectile256(
      const AscendC::LocalTensor<float>& src_ub,
      const AscendC::LocalTensor<float>& rowsum_ub,
      const AscendC::LocalTensor<float>& tv_ub_tensor_,
      uint32_t num_rows_round,
      uint32_t num_elems,
      uint32_t num_elems_aligned) {
    AscendC::BlockReduceSum<float, false>(
        tv_ub_tensor_,
        src_ub,
        num_rows_round * num_elems_aligned / kFloatVectorSize,
        0,
        1,
        1,
        8);
    AscendC::PipeBarrier<PIPE_V>();
    set_vec_mask(kRowOpsSpecMask32);
    AscendC::BlockReduceSum<float, false>(tv_ub_tensor_[kReduceUbSize],
                                          tv_ub_tensor_,
                                          num_rows_round,
                                          0,
                                          1,
                                          1,
                                          4);
    AscendC::PipeBarrier<PIPE_V>();
    set_block_reduce_mask(kRowOpsSpecMask4);
    AscendC::BlockReduceSum<float, false>(
        rowsum_ub,
        tv_ub_tensor_[kReduceUbSize],
        ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
        0,
        1,
        1,
        8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                   static_cast<uint64_t>(-1));
  }

  __aicore__ inline void rowsum_tailtile(
      const AscendC::LocalTensor<float>& src_ub,
      const AscendC::LocalTensor<float>& rowsum_ub,
      const AscendC::LocalTensor<float>& tv_ub_tensor_,
      uint32_t num_rows_round,
      uint32_t num_elems,
      uint32_t num_elems_aligned) {
    if (num_elems >= kFloatVectorSize) {
      AscendC::BlockReduceSum<float, false>(
          tv_ub_tensor_,
          src_ub,
          num_rows_round,
          0,
          1,
          1,
          num_elems_aligned / kFloatBlockSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::BlockReduceSum<float, false>(
          rowsum_ub,
          tv_ub_tensor_,
          ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
          0,
          1,
          1,
          8);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint64_t row_sum_idx = 1;
           row_sum_idx < static_cast<uint64_t>(num_elems) / kFloatVectorSize;
           ++row_sum_idx) {
        AscendC::BlockReduceSum<float, false>(
            tv_ub_tensor_,
            src_ub[row_sum_idx * kFloatVectorSize],
            num_rows_round,
            0,
            1,
            1,
            num_elems_aligned / kFloatBlockSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(
            tv_ub_tensor_[kReduceUbSize],
            tv_ub_tensor_,
            ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
            0,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
        set_vec_mask(num_rows_round);
        AscendC::Add<float, false>(
            rowsum_ub,
            rowsum_ub,
            tv_ub_tensor_[kReduceUbSize],
            static_cast<uint64_t>(0),
            1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                       static_cast<uint64_t>(-1));
      }
    }
    if (num_elems % kFloatVectorSize > 0) {
      set_vec_mask(num_elems % kFloatVectorSize);
      AscendC::BlockReduceSum<float, false>(
          tv_ub_tensor_,
          src_ub[num_elems / kFloatVectorSize * kFloatVectorSize],
          num_rows_round,
          0,
          1,
          1,
          num_elems_aligned / kFloatBlockSize);
      AscendC::PipeBarrier<PIPE_V>();
      set_block_reduce_mask(
          ::CeilDiv(num_elems % kFloatVectorSize, kFloatBlockSize));
      if (num_elems < kFloatVectorSize) {
        AscendC::BlockReduceSum<float, false>(
            rowsum_ub,
            tv_ub_tensor_,
            ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
            0,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
      } else {
        AscendC::BlockReduceSum<float, false>(
            tv_ub_tensor_[kReduceUbSize],
            tv_ub_tensor_,
            ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
            0,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
        set_vec_mask(num_rows_round);
        AscendC::Add<float, false>(
            rowsum_ub,
            rowsum_ub,
            tv_ub_tensor_[kReduceUbSize],
            static_cast<uint64_t>(0),
            1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
    }
  }

  __aicore__ inline void rowmax_spectile512(
      const AscendC::LocalTensor<float>& src_ub,
      const AscendC::LocalTensor<float>& rowmax_ub,
      const AscendC::LocalTensor<float>& tv_ub_tensor_,
      uint32_t num_rows_round,
      uint32_t num_elems,
      uint32_t num_elems_aligned) {
    AscendC::BlockReduceMax<float, false>(
        tv_ub_tensor_,
        src_ub,
        num_rows_round * num_elems_aligned / kFloatVectorSize,
        0,
        1,
        1,
        8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(
        tv_ub_tensor_[kReduceUbSize],
        tv_ub_tensor_,
        num_rows_round * num_elems_aligned / kFloatBlockSize / kFloatVectorSize,
        0,
        1,
        1,
        8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(rowmax_ub,
                                          tv_ub_tensor_[kReduceUbSize],
                                          num_rows_round * num_elems_aligned /
                                              kFloatVectorSize /
                                              kFloatVectorSize,
                                          0,
                                          1,
                                          1,
                                          8);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void rowmax_spectile256(
      const AscendC::LocalTensor<float>& src_ub,
      const AscendC::LocalTensor<float>& rowmax_ub,
      const AscendC::LocalTensor<float>& tv_ub_tensor_,
      uint32_t num_rows_round,
      uint32_t num_elems,
      uint32_t num_elems_aligned) {
    AscendC::BlockReduceMax<float, false>(
        tv_ub_tensor_,
        src_ub,
        num_rows_round * num_elems_aligned / kFloatVectorSize,
        0,
        1,
        1,
        8);
    AscendC::PipeBarrier<PIPE_V>();
    set_vec_mask(kRowOpsSpecMask32);
    AscendC::BlockReduceMax<float, false>(tv_ub_tensor_[kReduceUbSize],
                                          tv_ub_tensor_,
                                          num_rows_round,
                                          0,
                                          1,
                                          1,
                                          4);
    AscendC::PipeBarrier<PIPE_V>();
    set_block_reduce_mask(kRowOpsSpecMask4);
    AscendC::BlockReduceMax<float, false>(
        rowmax_ub,
        tv_ub_tensor_[kReduceUbSize],
        ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
        0,
        1,
        1,
        8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                   static_cast<uint64_t>(-1));
  }

  __aicore__ inline void rowmax_tailtile(
      const AscendC::LocalTensor<float>& src_ub,
      const AscendC::LocalTensor<float>& rowmax_ub,
      const AscendC::LocalTensor<float>& tv_ub_tensor_,
      uint32_t num_rows_round,
      uint32_t num_elems,
      uint32_t num_elems_aligned) {
    if (num_elems >= kFloatVectorSize) {
      AscendC::BlockReduceMax<float, false>(
          tv_ub_tensor_,
          src_ub,
          num_rows_round,
          0,
          1,
          1,
          num_elems_aligned / kFloatBlockSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::BlockReduceMax<float, false>(
          rowmax_ub,
          tv_ub_tensor_,
          ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
          0,
          1,
          1,
          8);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint64_t rowmax_idx = 1;
           rowmax_idx < static_cast<uint64_t>(num_elems) / kFloatVectorSize;
           ++rowmax_idx) {
        AscendC::BlockReduceMax<float, false>(
            tv_ub_tensor_,
            src_ub[rowmax_idx * kFloatVectorSize],
            num_rows_round,
            0,
            1,
            1,
            num_elems_aligned / kFloatBlockSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            tv_ub_tensor_[kReduceUbSize],
            tv_ub_tensor_,
            ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
            0,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
        set_vec_mask(num_rows_round);
        AscendC::Max<float, false>(
            rowmax_ub,
            rowmax_ub,
            tv_ub_tensor_[kReduceUbSize],
            static_cast<uint64_t>(0),
            1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                       static_cast<uint64_t>(-1));
      }
    }
    if (num_elems % kFloatVectorSize > 0) {
      set_vec_mask(num_elems % kFloatVectorSize);
      AscendC::BlockReduceMax<float, false>(
          tv_ub_tensor_,
          src_ub[num_elems / kFloatVectorSize * kFloatVectorSize],
          num_rows_round,
          0,
          1,
          1,
          num_elems_aligned / kFloatBlockSize);
      AscendC::PipeBarrier<PIPE_V>();
      set_block_reduce_mask(
          ::CeilDiv(num_elems % kFloatVectorSize, kFloatBlockSize));
      if (num_elems < kFloatVectorSize) {
        AscendC::BlockReduceMax<float, false>(
            rowmax_ub,
            tv_ub_tensor_,
            ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
            0,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
      } else {
        AscendC::BlockReduceMax<float, false>(
            tv_ub_tensor_[kReduceUbSize],
            tv_ub_tensor_,
            ::CeilDiv(num_rows_round * kFloatBlockSize, kFloatVectorSize),
            0,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
        set_vec_mask(num_rows_round);
        AscendC::Max<float, false>(
            rowmax_ub,
            rowmax_ub,
            tv_ub_tensor_[kReduceUbSize],
            static_cast<uint64_t>(0),
            1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
    }
  }

  __aicore__ inline void copy_s_gm_to_ub(
      AscendC::GlobalTensor<ElementInput> g_input,
      uint32_t s_ub_offset,
      uint32_t row_num_cur_loop,
      uint32_t column_num_round,
      uint32_t column_num_pad) {
    AscendC::DataCopy(ls_ub_tensor_[s_ub_offset],
                      g_input,
                      AscendC::DataCopyParams(
                          row_num_cur_loop,
                          column_num_round / kFloatBlockSize,
                          (column_num_pad - column_num_round) / kFloatBlockSize,
                          0));
  }

  __aicore__ inline void scale_s(uint32_t s_ub_offset,
                                 uint32_t row_num_cur_loop,
                                 uint32_t column_num_round) {
    AscendC::Muls<float, false>(
        ls_ub_tensor_[s_ub_offset],
        ls_ub_tensor_[s_ub_offset],
        scale_value_,
        static_cast<uint64_t>(0),
        ::CeilDiv(row_num_cur_loop * column_num_round, kFloatVectorSize),
        AscendC::UnaryRepeatParams(1, 1, 8, 8));

    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void calc_local_row_max(uint32_t s_ub_offset,
                                            uint32_t row_num_cur_loop_round,
                                            uint32_t column_num,
                                            uint32_t column_num_round,
                                            uint32_t row_offset) {
    if (column_num == 512) {
      rowmax_spectile512(ls_ub_tensor_[s_ub_offset],
                         lm_ub_tensor_[row_offset],
                         tv_ub_tensor_,
                         row_num_cur_loop_round,
                         column_num,
                         column_num_round);
    } else if (column_num == 256) {
      rowmax_spectile256(ls_ub_tensor_[s_ub_offset],
                         lm_ub_tensor_[row_offset],
                         tv_ub_tensor_,
                         row_num_cur_loop_round,
                         column_num,
                         column_num_round);
    } else {
      rowmax_tailtile(ls_ub_tensor_[s_ub_offset],
                      lm_ub_tensor_[row_offset],
                      tv_ub_tensor_,
                      row_num_cur_loop_round,
                      column_num,
                      column_num_round);
    }
  }

  __aicore__ inline void update_global_row_max(uint32_t row_num_cur_loop,
                                               uint32_t row_num_cur_loop_round,
                                               uint32_t column_num,
                                               uint32_t column_num_round,
                                               uint32_t dm_ub_offset_cur_cycle,
                                               uint32_t row_offset,
                                               uint32_t is_first_stack_tile) {
    if (is_first_stack_tile) {
      AscendC::DataCopy(hm_ub_tensor_[row_offset],
                        lm_ub_tensor_[row_offset],
                        AscendC::DataCopyParams(
                            1, row_num_cur_loop_round / kFloatBlockSize, 0, 0));
      AscendC::PipeBarrier<PIPE_V>();
    } else {
      set_vec_mask(row_num_cur_loop);
      // *** hm = vmax(lm, gm)
      AscendC::Max<float, false>(hm_ub_tensor_[row_offset],
                                 lm_ub_tensor_[row_offset],
                                 gm_ub_tensor_[row_offset],
                                 static_cast<uint64_t>(0),
                                 1,
                                 AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
      AscendC::PipeBarrier<PIPE_V>();
      // *** dm = gm - hm
      AscendC::Sub<float, false>(dm_ub_tensor_[dm_ub_offset_cur_cycle],
                                 gm_ub_tensor_[row_offset],
                                 hm_ub_tensor_[row_offset],
                                 static_cast<uint64_t>(0),
                                 1,
                                 AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
      AscendC::PipeBarrier<PIPE_V>();
      // *** dm = exp(dm)
      AscendC::Exp<float, false>(dm_ub_tensor_[dm_ub_offset_cur_cycle],
                                 dm_ub_tensor_[dm_ub_offset_cur_cycle],
                                 static_cast<uint64_t>(0),
                                 1,
                                 AscendC::UnaryRepeatParams(1, 1, 8, 8));
    }
    AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                   static_cast<uint64_t>(-1));
    AscendC::PipeBarrier<PIPE_V>();
    // *** gm = hm
    AscendC::DataCopy(gm_ub_tensor_[row_offset],
                      hm_ub_tensor_[row_offset],
                      AscendC::DataCopyParams(
                          1, row_num_cur_loop_round / kFloatBlockSize, 0, 0));
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void calc_exp(uint32_t s_ub_offset,
                                  uint32_t row_num_cur_loop,
                                  uint32_t row_num_cur_loop_round,
                                  uint32_t column_num,
                                  uint32_t column_num_round,
                                  uint32_t row_offset) {
    // *** hm_block = expand_to_block(hm), 存放于 tv
    AscendC::Brcb(
        tv_ub_tensor_.template ReinterpretCast<uint32_t>(),
        hm_ub_tensor_[row_offset].template ReinterpretCast<uint32_t>(),
        row_num_cur_loop_round / kFloatBlockSize,
        AscendC::BrcbRepeatParams(1, 8));
    AscendC::PipeBarrier<PIPE_V>();
    // *** ls = ls - hm_block
    for (uint32_t sub_idx = 0; sub_idx < column_num / kFloatVectorSize;
         ++sub_idx) {
      AscendC::Sub<float, false>(
          ls_ub_tensor_[s_ub_offset][sub_idx * kFloatVectorSize],
          ls_ub_tensor_[s_ub_offset][sub_idx * kFloatVectorSize],
          tv_ub_tensor_,
          static_cast<uint64_t>(0),
          row_num_cur_loop,
          AscendC::BinaryRepeatParams(1,
                                      1,
                                      0,
                                      column_num_round / kFloatBlockSize,
                                      column_num_round / kFloatBlockSize,
                                      1));
    }
    if (column_num % kFloatVectorSize > 0) {
      set_vec_mask(column_num % kFloatVectorSize);
      AscendC::Sub<float, false>(
          ls_ub_tensor_[s_ub_offset]
                       [column_num / kFloatVectorSize * kFloatVectorSize],
          ls_ub_tensor_[s_ub_offset]
                       [column_num / kFloatVectorSize * kFloatVectorSize],
          tv_ub_tensor_,
          static_cast<uint64_t>(0),
          row_num_cur_loop,
          AscendC::BinaryRepeatParams(1,
                                      1,
                                      0,
                                      column_num_round / kFloatBlockSize,
                                      column_num_round / kFloatBlockSize,
                                      1));
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
    }
    AscendC::PipeBarrier<PIPE_V>();
    // *** ls = exp(ls)
    AscendC::Exp<float, false>(
        ls_ub_tensor_[s_ub_offset],
        ls_ub_tensor_[s_ub_offset],
        static_cast<uint64_t>(0),
        ::CeilDiv(row_num_cur_loop * column_num_round, kFloatVectorSize),
        AscendC::UnaryRepeatParams(1, 1, 8, 8));
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void calc_local_row_sum(uint32_t s_ub_offset,
                                            uint32_t row_num_cur_loop_round,
                                            uint32_t column_num,
                                            uint32_t column_num_round,
                                            uint32_t row_offset) {
    // *** ll = rowsum(ls32)
    if (column_num == 512) {
      rowsum_spectile512(ls_ub_tensor_[s_ub_offset],
                         ll_ub_tensor_[row_offset],
                         tv_ub_tensor_,
                         row_num_cur_loop_round,
                         column_num,
                         column_num_round);
    } else if (column_num == 256) {
      rowsum_spectile256(ls_ub_tensor_[s_ub_offset],
                         ll_ub_tensor_[row_offset],
                         tv_ub_tensor_,
                         row_num_cur_loop_round,
                         column_num,
                         column_num_round);
    } else {
      rowsum_tailtile(ls_ub_tensor_[s_ub_offset],
                      ll_ub_tensor_[row_offset],
                      tv_ub_tensor_,
                      row_num_cur_loop_round,
                      column_num,
                      column_num_round);
    }
  }

  __aicore__ inline void update_global_row_sum(uint32_t s_ub_offset,
                                               uint32_t row_num_cur_loop,
                                               uint32_t row_num_cur_loop_round,
                                               uint32_t dm_ub_offset_cur_cycle,
                                               uint32_t row_offset,
                                               uint32_t is_first_stack_tile) {
    if (is_first_stack_tile) {
      // *** gl = ll
      AscendC::DataCopy(gl_ub_tensor_[row_offset],
                        ll_ub_tensor_[row_offset],
                        AscendC::DataCopyParams(
                            1, row_num_cur_loop_round / kFloatBlockSize, 0, 0));
      AscendC::PipeBarrier<PIPE_V>();
    } else {
      set_vec_mask(row_num_cur_loop);
      // *** gl = dm * gl
      AscendC::Mul<float, false>(gl_ub_tensor_[row_offset],
                                 dm_ub_tensor_[dm_ub_offset_cur_cycle],
                                 gl_ub_tensor_[row_offset],
                                 static_cast<uint64_t>(0),
                                 1,
                                 AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
      AscendC::PipeBarrier<PIPE_V>();
      // *** gl = ll + gl
      AscendC::Add<float, false>(gl_ub_tensor_[row_offset],
                                 gl_ub_tensor_[row_offset],
                                 ll_ub_tensor_[row_offset],
                                 static_cast<uint64_t>(0),
                                 1,
                                 AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
    }
  }

  __aicore__ inline void down_cast_p(uint32_t s_ub_offset,
                                     uint32_t row_num_cur_loop,
                                     uint32_t column_num_round) {
    // *** lp = castfp32to16(ls)
    if (std::is_same<ElementOutput, bfloat16_t>::value) {
      AscendC::Cast<ElementOutput, float, false>(
          lp_ub_tensor_[s_ub_offset],
          ls_ub_tensor_[s_ub_offset],
          AscendC::RoundMode::CAST_RINT,
          static_cast<uint64_t>(0),
          ::CeilDiv(row_num_cur_loop * column_num_round, kFloatVectorSize),
          AscendC::UnaryRepeatParams(1, 1, 4, 8));
    } else {
      AscendC::Cast<ElementOutput, float, false>(
          lp_ub_tensor_[s_ub_offset],
          ls_ub_tensor_[s_ub_offset],
          AscendC::RoundMode::CAST_NONE,
          static_cast<uint64_t>(0),
          ::CeilDiv(row_num_cur_loop * column_num_round, kFloatVectorSize),
          AscendC::UnaryRepeatParams(1, 1, 4, 8));
    }
  }

  __aicore__ inline void copy_p_ub_to_gm(
      AscendC::GlobalTensor<ElementOutput> g_output,
      uint32_t s_ub_offset,
      uint32_t row_num_cur_loop,
      uint32_t column_num_round,
      uint32_t column_num_pad) {
    AscendC::DataCopy(g_output,
                      lp_ub_tensor_[s_ub_offset],
                      AscendC::DataCopyParams(
                          row_num_cur_loop,
                          column_num_round / kBlockSize,
                          0,
                          (column_num_pad - column_num_round) / kBlockSize));
  }

  __aicore__ inline void sub_core_compute(
      AscendC::GlobalTensor<ElementOutput> g_output,
      const LayoutOutput& layout_output,
      uint32_t row_offset,
      uint32_t is_first_stack_tile,
      uint32_t is_last_no_mask_stack_tile,
      uint32_t is_first_row_loop,
      uint32_t is_last_row_loop,
      uint32_t column_num_round,
      uint32_t pingpong_flag,
      uint32_t cur_stack_tile_mod) {
    uint32_t row_num_cur_loop = layout_output.shape(0);
    uint32_t row_num_cur_loop_round =
        ::RoundUp(row_num_cur_loop, kFloatBlockSize);
    uint32_t column_num = layout_output.shape(1);
    uint32_t column_num_pad = layout_output.stride(0);
    uint32_t s_ub_offset = pingpong_flag * kMaxUbSElemNum;
    uint32_t dm_ub_offset_cur_cycle =
        cur_stack_tile_mod * kMaxRowNumSubCore + row_offset;

    if constexpr (LseOutputMode == Catlass::Epilogue::LseMode::LSE_OUT) {
      // In lse out-only mode, tv is used in the last stack tile to transport
      // lse
      if (is_first_stack_tile && is_first_row_loop) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
      }
    }
    calc_local_row_max(s_ub_offset,
                       row_num_cur_loop_round,
                       column_num,
                       column_num_round,
                       row_offset);
    update_global_row_max(row_num_cur_loop,
                          row_num_cur_loop_round,
                          column_num,
                          column_num_round,
                          dm_ub_offset_cur_cycle,
                          row_offset,
                          is_first_stack_tile);

    calc_exp(s_ub_offset,
             row_num_cur_loop,
             row_num_cur_loop_round,
             column_num,
             column_num_round,
             row_offset);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpong_flag);

    down_cast_p(s_ub_offset, row_num_cur_loop, column_num_round);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpong_flag);

    calc_local_row_sum(s_ub_offset,
                       row_num_cur_loop_round,
                       column_num,
                       column_num_round,
                       row_offset);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpong_flag);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpong_flag);
    copy_p_ub_to_gm(g_output,
                    s_ub_offset,
                    row_num_cur_loop,
                    column_num_round,
                    column_num_pad);
    {
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpong_flag);
      if (is_last_no_mask_stack_tile && is_last_row_loop) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
    update_global_row_sum(s_ub_offset,
                          row_num_cur_loop,
                          row_num_cur_loop_round,
                          dm_ub_offset_cur_cycle,
                          row_offset,
                          is_first_stack_tile);
  }

 public:
  __aicore__ inline void operator()(
      AscendC::GlobalTensor<ElementOutput> g_output,
      AscendC::GlobalTensor<ElementInput> g_input,
      const LayoutOutput& layout_output,
      const LayoutInput& layout_input,
      Catlass::GemmCoord actual_block_shape,
      uint32_t is_first_stack_tile,
      uint32_t is_last_no_mask_stack_tile,
      uint32_t q_s_block_size,
      uint32_t q_n_block_size,
      uint32_t cur_stack_tile_mod,
      uint32_t masked_prefix = 0) {
    uint32_t row_num = actual_block_shape.m();
    uint32_t column_num = actual_block_shape.n();
    uint32_t column_num_round = ::RoundUp(column_num, kBlockSize);
    uint32_t column_num_pad = layout_input.stride(0);

    uint32_t sub_block_idx = AscendC::GetSubBlockIdx();
    uint32_t sub_block_num = AscendC::GetSubBlockNum();

    uint32_t q_n_split_sub_block = q_n_block_size / sub_block_num;
    uint32_t row_split_sub_block = (q_n_block_size == 1)
                                       ? (q_s_block_size / 2)
                                       : (q_s_block_size * q_n_split_sub_block);
    uint32_t row_actual_this_sub_block = (sub_block_idx == 1)
                                             ? (row_num - row_split_sub_block)
                                             : row_split_sub_block;
    uint32_t row_offset_this_sub_block = sub_block_idx * row_split_sub_block;
    uint32_t max_row_num_per_loop = kMaxUbSElemNum / column_num_round;
    uint32_t row_num_tile = ::RoundDown(max_row_num_per_loop, kFloatBlockSize);
    row_num_tile = AscendC::Std::min(row_num_tile, kFloatVectorSize);
    uint32_t row_loop_num = ::CeilDiv(row_actual_this_sub_block, row_num_tile);
    uint32_t pre_load = 1;

    for (uint32_t row_loop_idx = 0; row_loop_idx < row_loop_num + pre_load;
         row_loop_idx++) {
      if (row_loop_idx < row_loop_num) {
        uint32_t pingpong_flag = row_loop_idx % 2;
        uint32_t row_offset_cur_loop = row_loop_idx * row_num_tile;
        uint32_t row_offset_io_gm =
            row_offset_cur_loop + row_offset_this_sub_block;
        uint32_t row_num_cur_loop =
            (row_loop_idx == row_loop_num - 1)
                ? (row_actual_this_sub_block - row_offset_cur_loop)
                : row_num_tile;

        int64_t offset_input =
            layout_input.GetOffset(Catlass::MatrixCoord(row_offset_io_gm, 0));
        auto g_input_cur_loop = g_input[offset_input];

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpong_flag);
        copy_s_gm_to_ub(g_input_cur_loop,
                        (pingpong_flag * kMaxUbSElemNum),
                        row_num_cur_loop,
                        column_num_round,
                        column_num_pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpong_flag);
      }
      if (row_loop_idx >= pre_load) {
        uint32_t delayed_row_loop_idx = row_loop_idx - pre_load;
        uint32_t pingpong_flag = delayed_row_loop_idx % 2;
        uint32_t row_offset_cur_loop = delayed_row_loop_idx * row_num_tile;
        uint32_t row_offset_io_gm =
            row_offset_cur_loop + row_offset_this_sub_block;
        uint32_t row_num_cur_loop =
            (delayed_row_loop_idx == row_loop_num - 1)
                ? (row_actual_this_sub_block - row_offset_cur_loop)
                : row_num_tile;

        int64_t offset_output =
            layout_output.GetOffset(Catlass::MatrixCoord(row_offset_io_gm, 0));
        auto g_output_cur_loop = g_output[offset_output];
        auto layout_output_cur_loop = layout_output.GetTileLayout(
            Catlass::MatrixCoord(row_num_cur_loop, column_num));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpong_flag);
        scale_s((pingpong_flag * kMaxUbSElemNum),
                row_num_cur_loop,
                column_num_round);
        // A decode window may begin inside its first physical KV page.
        // Exclude that prefix before the row maximum and softmax sum.
        if (masked_prefix != 0) {
          for (uint32_t row = 0; row < row_num_cur_loop; ++row) {
            AscendC::Duplicate(ls_ub_tensor_[pingpong_flag * kMaxUbSElemNum +
                                             row * column_num_round],
                               -3.402823466e+38F,
                               masked_prefix);
          }
          AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                         static_cast<uint64_t>(-1));
          AscendC::PipeBarrier<PIPE_V>();
        }
        sub_core_compute(g_output_cur_loop,
                         layout_output_cur_loop,
                         row_offset_cur_loop,
                         is_first_stack_tile,
                         is_last_no_mask_stack_tile,
                         delayed_row_loop_idx == 0,
                         delayed_row_loop_idx == row_loop_num - 1,
                         column_num_round,
                         pingpong_flag,
                         cur_stack_tile_mod);
      }
    }
  }

 private:
  float scale_value_;
  AscendC::LocalTensor<float> ls_ub_tensor_;
  AscendC::LocalTensor<ElementOutput> lp_ub_tensor_;
  AscendC::LocalTensor<float> lm_ub_tensor_;
  AscendC::LocalTensor<float> hm_ub_tensor_;
  AscendC::LocalTensor<float> gm_ub_tensor_;
  AscendC::LocalTensor<float> dm_ub_tensor_;
  AscendC::LocalTensor<float> ll_ub_tensor_;
  AscendC::LocalTensor<float> tv_ub_tensor_;
  AscendC::LocalTensor<float> gl_ub_tensor_;
};
}  // namespace xllm_ops::xfia
