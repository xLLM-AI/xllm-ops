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
          class UpdateType,
          class LseType,
          Catlass::Epilogue::LseMode LseOutputMode>
class WindowRescaleO final {
 public:
  // Type aliases
  using ArchTag = Catlass::Arch::AtlasA2;

  using ElementOutput = typename OutputType::Element;
  using ElementInput = typename InputType::Element;
  using ElementUpdate = typename UpdateType::Element;
  using ElementLse = typename LseType::Element;

  using LayoutOutput = typename OutputType::Layout;
  using LayoutInput = typename InputType::Layout;
  using LayoutUpdate = typename UpdateType::Layout;
  using LayoutLse = typename LseType::Layout;

  static constexpr Catlass::Epilogue::LseMode LSE_MODE = LseOutputMode;

  static constexpr uint32_t kFloatBlockSize = 8;
  static constexpr uint32_t kFloatVectorSize = 64;
  static constexpr uint32_t kUbUint8VectorSize = 1024;
  static constexpr uint32_t kUbUint8BlockSize = 16384;
  static constexpr uint32_t kVectorSize = 128;
  static constexpr uint32_t kMaxUbOElemNum = 8192;
  static constexpr uint32_t kMaxRowNumSubCore = 256;

  __aicore__ inline explicit WindowRescaleO(
      Catlass::Arch::Resource<ArchTag>& resource) {
    // Allocate UB space
    constexpr uint32_t kLoUbTensorOffset = 6 * kUbUint8BlockSize;
    constexpr uint32_t kGoUbTensorOffset = 8 * kUbUint8BlockSize;
    constexpr uint32_t kTvUbTensorOffset = 10 * kUbUint8BlockSize;

    constexpr uint32_t kHmUbTensorOffset =
        10 * kUbUint8BlockSize + 9 * kUbUint8VectorSize;
    constexpr uint32_t kGmUbTensorOffset =
        10 * kUbUint8BlockSize + 10 * kUbUint8VectorSize;
    constexpr uint32_t kGlUbTensorOffset =
        10 * kUbUint8BlockSize + 12 * kUbUint8VectorSize;
    constexpr uint32_t kLseUbTensorOffset =
        10 * kUbUint8BlockSize + 12 * kUbUint8VectorSize;
    constexpr uint32_t kDmUbTensorOffset =
        10 * kUbUint8BlockSize + 13 * kUbUint8VectorSize;

    lo_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kLoUbTensorOffset);
    dm_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kDmUbTensorOffset);
    gl_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kGlUbTensorOffset);
    tv_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kTvUbTensorOffset);
    go_ub_tensor16_ = resource.ubBuf.template GetBufferByByte<ElementOutput>(
        kGoUbTensorOffset);
    go_ub_tensor32_ =
        resource.ubBuf.template GetBufferByByte<float>(kGoUbTensorOffset);
    hm_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kHmUbTensorOffset);
    gm_ub_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kGmUbTensorOffset);
    lse32_ubuf_tensor_ =
        resource.ubBuf.template GetBufferByByte<float>(kLseUbTensorOffset);
  }

 private:
  __aicore__ inline void set_mask(int32_t len) {
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % kFloatVectorSize;
    for (int64_t i = 0; i < temp; i++) {
      mask |= one << i;
    }

    if (len == kVectorSize) {
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
    } else if (len >= kFloatVectorSize) {
      AscendC::SetVectorMask<int8_t>(mask, static_cast<uint64_t>(-1));
    } else {
      AscendC::SetVectorMask<int8_t>(0x0, mask);
    }
  }

  __aicore__ inline void copy_o_to_gm(
      AscendC::GlobalTensor<ElementOutput> g_output,
      uint32_t pro_token_idx,
      uint32_t pro_token_num,
      uint32_t epi_token_num,
      uint32_t integral_head_num,
      uint32_t q_s_this_sub_block,
      uint32_t embed,
      uint32_t o_hidden_size) {
    uint32_t inner_o_gm_offset = 0;
    uint32_t inner_go_ub_offset = 0;
    if (pro_token_num != 0) {
      AscendC::DataCopyPad(
          g_output[inner_o_gm_offset + pro_token_idx * o_hidden_size],
          go_ub_tensor16_[inner_go_ub_offset],
          AscendC::DataCopyExtParams(
              pro_token_num, embed * 2, 0, (o_hidden_size - embed) * 2, 0));
      inner_o_gm_offset += embed;
      inner_go_ub_offset += pro_token_num * embed;
    }
    for (uint32_t q_n_idx = 0; q_n_idx < integral_head_num; q_n_idx++) {
      AscendC::DataCopyPad(
          g_output[inner_o_gm_offset],
          go_ub_tensor16_[inner_go_ub_offset],
          AscendC::DataCopyExtParams(q_s_this_sub_block,
                                     embed * 2,
                                     0,
                                     (o_hidden_size - embed) * 2,
                                     0));
      inner_o_gm_offset += embed;
      inner_go_ub_offset += q_s_this_sub_block * embed;
    }
    if (epi_token_num != 0) {
      AscendC::DataCopyPad(
          g_output[inner_o_gm_offset],
          go_ub_tensor16_[inner_go_ub_offset],
          AscendC::DataCopyExtParams(
              epi_token_num, embed * 2, 0, (o_hidden_size - embed) * 2, 0));
    }
  }

  __aicore__ inline void sub_core_compute(
      AscendC::GlobalTensor<ElementOutput> g_output,
      AscendC::GlobalTensor<ElementInput> g_input,
      AscendC::GlobalTensor<ElementUpdate> g_update,
      AscendC::GlobalTensor<ElementLse> g_lse,
      const LayoutOutput& layout_output,
      const LayoutInput& layout_input,
      const LayoutUpdate& layout_update,
      const LayoutLse& layout_lse,
      uint32_t q_n_this_sub_block,
      uint32_t is_first_stack_tile,
      uint32_t is_last_stack_tile,
      uint32_t cur_stack_tile_mod,
      uint32_t need_row_loop,
      uint32_t is_last_row_loop,
      uint32_t row_offset_loop,
      uint32_t q_s_this_sub_block,
      uint32_t pro_token_idx,
      uint32_t pro_token_num,
      uint32_t epi_token_num,
      uint32_t integral_head_num) {
    uint32_t cur_row_num = layout_input.shape(0);
    uint32_t embed = layout_input.shape(1);
    uint32_t embed_round = layout_input.stride(0);
    uint32_t cur_row_num_round = ::RoundUp(cur_row_num, kFloatBlockSize);
    uint32_t q_s_block_size = layout_output.shape(0);
    uint32_t o_hidden_size = layout_output.shape(1);
    uint32_t q_heads = layout_lse.shape(1);
    uint32_t dm_ub_offset_cur_stack_tile =
        cur_stack_tile_mod * kMaxRowNumSubCore + row_offset_loop;

    if (!is_first_stack_tile) {
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
      AscendC::DataCopy(
          lo_ub_tensor_,
          g_input,
          AscendC::DataCopyParams(
              1, cur_row_num * embed_round / kFloatBlockSize, 0, 0));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    if (!is_first_stack_tile) {
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
      AscendC::Brcb(tv_ub_tensor_.ReinterpretCast<uint32_t>(),
                    dm_ub_tensor_[dm_ub_offset_cur_stack_tile]
                        .ReinterpretCast<uint32_t>(),
                    cur_row_num_round / kFloatBlockSize,
                    AscendC::BrcbRepeatParams(1, 8));
      AscendC::PipeBarrier<PIPE_V>();
      if (need_row_loop) {
        AscendC::DataCopy(
            go_ub_tensor32_,
            g_update,
            AscendC::DataCopyParams(
                1, cur_row_num * embed_round / kFloatBlockSize, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
      }
      // *** go = go * dm_block
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
      for (uint32_t vmul_idx = 0; vmul_idx < embed / kFloatVectorSize;
           ++vmul_idx) {
        AscendC::Mul<float, false>(
            go_ub_tensor32_[vmul_idx * kFloatVectorSize],
            go_ub_tensor32_[vmul_idx * kFloatVectorSize],
            tv_ub_tensor_,
            static_cast<uint64_t>(0),
            cur_row_num,
            AscendC::BinaryRepeatParams(1,
                                        1,
                                        0,
                                        embed_round / kFloatBlockSize,
                                        embed_round / kFloatBlockSize,
                                        1));
      }
      if (embed % kFloatVectorSize > 0) {
        set_mask(embed % kFloatVectorSize);
        AscendC::Mul<float, false>(
            go_ub_tensor32_[embed / kFloatVectorSize * kFloatVectorSize],
            go_ub_tensor32_[embed / kFloatVectorSize * kFloatVectorSize],
            tv_ub_tensor_,
            static_cast<uint64_t>(0),
            cur_row_num,
            AscendC::BinaryRepeatParams(1,
                                        1,
                                        0,
                                        embed_round / kFloatBlockSize,
                                        embed_round / kFloatBlockSize,
                                        1));
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                       static_cast<uint64_t>(-1));
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      // *** go = lo + go
      AscendC::Add<float, false>(
          go_ub_tensor32_,
          go_ub_tensor32_,
          lo_ub_tensor_,
          static_cast<uint64_t>(0),
          (cur_row_num * embed_round + kFloatVectorSize - 1) / kFloatVectorSize,
          AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    } else {
      // *** go = lo
      AscendC::DataCopy(
          go_ub_tensor32_,
          g_input,
          AscendC::DataCopyParams(
              1, cur_row_num * embed_round / kFloatBlockSize, 0, 0));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }

    if (is_last_stack_tile) {
      // *** gl_block = expand_to_block(gl), 存放于 tv
      AscendC::Brcb(tv_ub_tensor_.ReinterpretCast<uint32_t>(),
                    gl_ub_tensor_.ReinterpretCast<uint32_t>()[row_offset_loop],
                    cur_row_num_round / kFloatBlockSize,
                    AscendC::BrcbRepeatParams(1, 8));
      AscendC::PipeBarrier<PIPE_V>();
      // *** go = go / gl_block
      AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                     static_cast<uint64_t>(-1));
      for (uint32_t vdiv_idx = 0; vdiv_idx < embed / kFloatVectorSize;
           ++vdiv_idx) {
        AscendC::Div<float, false>(
            go_ub_tensor32_[vdiv_idx * kFloatVectorSize],
            go_ub_tensor32_[vdiv_idx * kFloatVectorSize],
            tv_ub_tensor_,
            static_cast<uint64_t>(0),
            cur_row_num,
            AscendC::BinaryRepeatParams(1,
                                        1,
                                        0,
                                        embed_round / kFloatBlockSize,
                                        embed_round / kFloatBlockSize,
                                        1));
      }
      if (embed % kFloatVectorSize > 0) {
        set_mask(embed % kFloatVectorSize);
        AscendC::Div<float, false>(
            go_ub_tensor32_[embed / kFloatVectorSize * kFloatVectorSize],
            go_ub_tensor32_[embed / kFloatVectorSize * kFloatVectorSize],
            tv_ub_tensor_,
            static_cast<uint64_t>(0),
            cur_row_num,
            AscendC::BinaryRepeatParams(1,
                                        1,
                                        0,
                                        embed_round / kFloatBlockSize,
                                        embed_round / kFloatBlockSize,
                                        1));
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1),
                                       static_cast<uint64_t>(-1));
      }
      AscendC::PipeBarrier<PIPE_V>();

      // *** go = castfp32to16(go)
      if (std::is_same<ElementOutput, bfloat16_t>::value) {
        AscendC::Cast<ElementOutput, float, false>(
            go_ub_tensor16_,
            go_ub_tensor32_,
            AscendC::RoundMode::CAST_RINT,
            static_cast<uint64_t>(0),
            (cur_row_num * embed_round + kFloatVectorSize - 1) /
                kFloatVectorSize,
            AscendC::UnaryRepeatParams(1, 1, 4, 8));
      } else {
        AscendC::Cast<ElementOutput, float, false>(
            go_ub_tensor16_,
            go_ub_tensor32_,
            AscendC::RoundMode::CAST_NONE,
            static_cast<uint64_t>(0),
            (cur_row_num * embed_round + kFloatVectorSize - 1) /
                kFloatVectorSize,
            AscendC::UnaryRepeatParams(1, 1, 4, 8));
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

      // ***move O to GM
      copy_o_to_gm(g_output,
                   pro_token_idx,
                   pro_token_num,
                   epi_token_num,
                   integral_head_num,
                   q_s_this_sub_block,
                   embed,
                   o_hidden_size);
      if constexpr (LseOutputMode == Catlass::Epilogue::LseMode::LSE_OUT) {
        if (is_last_row_loop) {
          AscendC::PipeBarrier<PIPE_V>();
          uint32_t len_brust = sizeof(float);
          AscendC::Ln<float, false>(lse32_ubuf_tensor_,
                                    gl_ub_tensor_,
                                    static_cast<uint64_t>(0),
                                    1,
                                    AscendC::UnaryRepeatParams(1, 1, 8, 8));

          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add<float, false>(
              lse32_ubuf_tensor_,
              lse32_ubuf_tensor_,
              gm_ub_tensor_,
              static_cast<uint64_t>(0),
              1,
              AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
          AscendC::PipeBarrier<PIPE_V>();

          // *** lse_block = expand_to_block(lse), 存放于 tv
          AscendC::Brcb(tv_ub_tensor_.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor_.ReinterpretCast<uint32_t>(),
                        cur_row_num_round / kFloatBlockSize,
                        AscendC::BrcbRepeatParams(1, 8));
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);

          if (q_n_this_sub_block == 0) {
            AscendC::DataCopyPad(
                g_lse,
                tv_ub_tensor_,
                AscendC::DataCopyExtParams(
                    cur_row_num, len_brust, 0, (q_heads - 1) * len_brust, 0));
          } else {
            for (uint32_t q_n_idx = 0; q_n_idx < q_n_this_sub_block;
                 q_n_idx++) {
              AscendC::DataCopyPad(
                  g_lse[q_n_idx],
                  tv_ub_tensor_[q_n_idx * q_s_block_size * kFloatBlockSize],
                  AscendC::DataCopyExtParams(q_s_block_size,
                                             len_brust,
                                             0,
                                             (q_heads - 1) * len_brust,
                                             0));
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        }
      }
    } else if (need_row_loop) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
      AscendC::DataCopy(
          g_update,
          go_ub_tensor32_,
          AscendC::DataCopyParams(
              1, cur_row_num * embed_round / kFloatBlockSize, 0, 0));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
  }

 public:
  __aicore__ inline void operator()(
      AscendC::GlobalTensor<ElementOutput> g_output,
      AscendC::GlobalTensor<ElementInput> g_input,
      AscendC::GlobalTensor<ElementUpdate> g_update,
      AscendC::GlobalTensor<ElementLse> g_lse,
      const LayoutOutput& layout_output,
      const LayoutInput& layout_input,
      const LayoutUpdate& layout_update,
      const LayoutLse& layout_lse,
      Catlass::GemmCoord actual_block_shape,
      uint32_t q_s_block_size,
      uint32_t q_n_block_size,
      uint32_t is_first_stack_tile,
      uint32_t is_last_stack_tile,
      uint32_t cur_stack_tile_mod) {
    uint32_t row_num = actual_block_shape.m();
    uint32_t embed = actual_block_shape.n();
    uint32_t row_tile = kMaxUbOElemNum / embed;

    uint32_t sub_block_idx = AscendC::GetSubBlockIdx();
    uint32_t sub_block_num = AscendC::GetSubBlockNum();

    uint32_t q_n_split_sub_block = q_n_block_size / sub_block_num;
    uint32_t q_n_this_sub_block = (q_n_block_size == 1) ? 0
                                  : (sub_block_idx == 1)
                                      ? (q_n_block_size - q_n_split_sub_block)
                                      : q_n_split_sub_block;
    uint32_t in_row_split_sub_block =
        (q_n_block_size == 1) ? (q_s_block_size / sub_block_num)
                              : (q_s_block_size * q_n_split_sub_block);
    uint32_t in_row_actual_this_sub_block =
        (sub_block_idx == 1) ? (row_num - in_row_split_sub_block)
                             : in_row_split_sub_block;
    uint32_t in_row_offset_this_sub_block =
        sub_block_idx * in_row_split_sub_block;
    uint32_t out_row_offset_this_sub_block =
        (q_n_block_size == 1) ? in_row_offset_this_sub_block : 0;
    uint32_t out_col_offset_this_sub_block =
        (q_n_block_size == 1) ? 0 : sub_block_idx * q_n_split_sub_block * embed;
    uint32_t q_s_this_sub_block =
        (q_n_block_size == 1) ? in_row_actual_this_sub_block : q_s_block_size;
    int64_t out_offset_sub_block = layout_output.GetOffset(Catlass::MatrixCoord(
        out_row_offset_this_sub_block, out_col_offset_this_sub_block));

    uint32_t out_lse_row_offset_this_sub_block =
        (q_n_block_size == 1) ? in_row_offset_this_sub_block : 0;
    uint32_t out_lse_col_offset_this_sub_block =
        (q_n_block_size == 1) ? 0 : sub_block_idx * q_n_split_sub_block;
    int64_t offset_lse = layout_lse.GetOffset(Catlass::MatrixCoord(
        out_lse_row_offset_this_sub_block, out_lse_col_offset_this_sub_block));
    auto g_lse_this_sub_block = g_lse[offset_lse];
    auto layout_out_lse_this_sub_block = layout_lse;

    if (in_row_actual_this_sub_block > 0) {
      uint32_t row_loop = ::CeilDiv(in_row_actual_this_sub_block, row_tile);
      uint32_t need_row_loop = (row_loop > 1) ? 1 : 0;

      // The rows of each cycle consist of multiple heads with several tokens.
      // There are several integral heads, one prologue head, one epilogue head.
      uint32_t pro_token_idx =
          0;  // the token idx of the start token of the prologue part
      uint32_t pro_token_num = 0;  // the token num of the prologue part
      uint32_t epi_token_num = 0;  // the token num of the epilogue part
      uint32_t integral_head_num =
          0;  // the number of integral heads within a cycle
      for (uint32_t row_loop_idx = 0; row_loop_idx < row_loop; row_loop_idx++) {
        uint32_t row_offset_loop = row_loop_idx * row_tile;
        uint32_t row_offset_cur_loop =
            in_row_offset_this_sub_block + row_offset_loop;
        uint32_t row_actual_cur_loop =
            (row_loop_idx == (row_loop - 1))
                ? in_row_actual_this_sub_block - row_loop_idx * row_tile
                : row_tile;

        int64_t offset_output =
            row_loop_idx * row_tile / q_s_this_sub_block * embed +
            out_offset_sub_block;
        auto g_output_cur_loop = g_output[offset_output];
        auto layout_output_cur_loop = layout_output;
        int64_t offset_input = layout_input.GetOffset(
            Catlass::MatrixCoord(row_offset_cur_loop, 0));
        auto g_input_cur_loop = g_input[offset_input];
        auto layout_input_cur_loop = layout_input.GetTileLayout(
            Catlass::MatrixCoord(row_actual_cur_loop, embed));

        int64_t offset_update = layout_update.GetOffset(
            Catlass::MatrixCoord(row_offset_cur_loop, 0));
        auto g_update_cur_loop = g_update[offset_update];
        auto layout_update_cur_loop = layout_update.GetTileLayout(
            Catlass::MatrixCoord(row_actual_cur_loop, embed));

        pro_token_idx = row_offset_loop % q_s_this_sub_block;
        pro_token_num =
            AscendC::Std::min(row_actual_cur_loop,
                              (q_s_this_sub_block - pro_token_idx)) %
            q_s_this_sub_block;
        integral_head_num =
            (row_actual_cur_loop - pro_token_num) / q_s_this_sub_block;
        epi_token_num = row_actual_cur_loop - pro_token_num -
                        integral_head_num * q_s_this_sub_block;

        sub_core_compute(g_output_cur_loop,
                         g_input_cur_loop,
                         g_update_cur_loop,
                         g_lse_this_sub_block,
                         layout_output_cur_loop,
                         layout_input_cur_loop,
                         layout_update_cur_loop,
                         layout_out_lse_this_sub_block,
                         q_n_this_sub_block,
                         is_first_stack_tile,
                         is_last_stack_tile,
                         cur_stack_tile_mod,
                         need_row_loop,
                         (row_loop_idx == row_loop - 1),
                         row_offset_loop,
                         q_s_this_sub_block,
                         pro_token_idx,
                         pro_token_num,
                         epi_token_num,
                         integral_head_num);
      }
    }
  }

 private:
  AscendC::LocalTensor<float> lo_ub_tensor_;
  AscendC::LocalTensor<float> dm_ub_tensor_;
  AscendC::LocalTensor<float> hm_ub_tensor_;
  AscendC::LocalTensor<float> gl_ub_tensor_;
  AscendC::LocalTensor<float> tv_ub_tensor_;
  AscendC::LocalTensor<ElementOutput> go_ub_tensor16_;
  AscendC::LocalTensor<float> go_ub_tensor32_;
  AscendC::LocalTensor<float> gm_ub_tensor_;
  AscendC::LocalTensor<float> lse32_ubuf_tensor_;
};
}  // namespace xllm_ops::xfia
