/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "activation/sigmoid.h"
#include "kda_replayssm_verify_record_tiling_data.h"
#include "kernel_operator.h"
#include "reduce/reduce.h"

namespace {

using namespace kda_replayssm;

constexpr int32_t kStateElements = kValueDimension * kKeyDimension;
constexpr int32_t kVectorLanes = 64;
constexpr uint8_t kStateRepeatStride = 16;

template <typename ElementType>
__aicore__ inline AscendC::GlobalTensor<ElementType> global_tensor(
    GM_ADDR address) {
  AscendC::GlobalTensor<ElementType> tensor;
  tensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementType*>(address));
  return tensor;
}

template <AscendC::HardEvent Event>
__aicore__ inline void synchronize() {
  AscendC::SetFlag<Event>(0);
  AscendC::WaitFlag<Event>(0);
}

template <typename ElementType, uint32_t Columns, uint32_t Rows>
__aicore__ inline void load_rows(AscendC::LocalTensor<ElementType> destination,
                                 AscendC::GlobalTensor<ElementType> source,
                                 int64_t offset,
                                 uint32_t stride,
                                 uint32_t rows = Rows,
                                 uint32_t columns = Columns) {
  const AscendC::DataCopyExtParams parameters(
      rows,
      columns * sizeof(ElementType),
      (stride - columns) * sizeof(ElementType),
      0,
      0);
  const AscendC::DataCopyPadExtParams<ElementType> padding(
      false, 0, 0, static_cast<ElementType>(0.0f));
  AscendC::DataCopyPad(destination, source[offset], parameters, padding);
}

template <typename ElementType, uint32_t Columns, uint32_t Rows>
__aicore__ inline void store_rows(
    AscendC::GlobalTensor<ElementType> destination,
    AscendC::LocalTensor<ElementType> source,
    int64_t offset,
    uint32_t stride,
    uint32_t columns = Columns) {
  const AscendC::DataCopyExtParams parameters(
      Rows,
      columns * sizeof(ElementType),
      0,
      (stride - columns) * sizeof(ElementType),
      0);
  AscendC::DataCopyPad(destination[offset], source, parameters);
}

__aicore__ inline void multiply_row(AscendC::LocalTensor<float> destination,
                                    AscendC::LocalTensor<float> source,
                                    AscendC::LocalTensor<float> row,
                                    int32_t value_rows) {
  for (int32_t column = 0; column < kKeyDimension; column += kVectorLanes) {
    AscendC::Mul(destination[column],
                 source[column],
                 row[column],
                 static_cast<uint64_t>(kVectorLanes),
                 static_cast<uint8_t>(value_rows),
                 {1, 1, 1, kStateRepeatStride, kStateRepeatStride, 0});
  }
}

__aicore__ inline void apply_correction(AscendC::LocalTensor<float> state,
                                        AscendC::LocalTensor<float> key,
                                        AscendC::LocalTensor<float> broadcast,
                                        int32_t value_rows) {
  for (int32_t column = 0; column < kKeyDimension; column += kVectorLanes) {
    AscendC::MulAddDst(state[column],
                       broadcast,
                       key[column],
                       static_cast<uint64_t>(kVectorLanes),
                       static_cast<uint8_t>(value_rows),
                       {1, 0, 1, kStateRepeatStride, 1, 0});
  }
}

__aicore__ inline void reduce_state_rows(
    AscendC::LocalTensor<float> destination,
    AscendC::LocalTensor<float> source,
    AscendC::LocalTensor<uint8_t> temporary,
    int32_t value_rows) {
  auto reduced_blocks = temporary.ReinterpretCast<float>();
  AscendC::Add(
      source,
      source,
      source[kVectorLanes],
      static_cast<uint64_t>(kVectorLanes),
      static_cast<uint8_t>(value_rows),
      {1, 1, 1, kStateRepeatStride, kStateRepeatStride, kStateRepeatStride});
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::BlockReduceSum(reduced_blocks,
                          source,
                          value_rows,
                          kVectorLanes,
                          1,
                          1,
                          kStateRepeatStride);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::BlockReduceSum(destination,
                          reduced_blocks,
                          value_rows / 8,
                          kVectorLanes,
                          1,
                          1,
                          kVectorLanes / 8);
}

__aicore__ inline void normalize_rows(AscendC::LocalTensor<float> input,
                                      AscendC::LocalTensor<float> scratch,
                                      AscendC::LocalTensor<float> norm,
                                      AscendC::LocalTensor<uint8_t> temporary) {
  AscendC::Mul(scratch, input, input, kWindow * kKeyDimension);
  AscendC::PipeBarrier<PIPE_V>();
  uint32_t shape[] = {kWindow, kKeyDimension};
  AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(
      norm, scratch, temporary, shape, /*isReuseSource=*/true);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Sqrt(norm, norm, kWindow);
  synchronize<AscendC::HardEvent::V_S>();
  for (int32_t token = 0; token < kWindow; ++token) {
    const float magnitude = norm.GetValue(token);
    if (magnitude > 0.0f) {
      const float inverse = 1.0f / magnitude;
      AscendC::Muls(input[token * kKeyDimension],
                    input[token * kKeyDimension],
                    inverse,
                    kKeyDimension);
    }
  }
  AscendC::PipeBarrier<PIPE_V>();
}

}  // namespace

extern "C" __global__ __aicore__ void kda_replayssm_verify_record(
    GM_ADDR query_handle,
    GM_ADDR key_handle,
    GM_ADDR value_handle,
    GM_ADDR gate_handle,
    GM_ADDR beta_handle,
    GM_ADDR state_handle,
    GM_ADDR a_log_handle,
    GM_ADDR dt_bias_handle,
    GM_ADDR state_indices_handle,
    GM_ADDR normalized_keys_handle,
    GM_ADDR corrections_handle,
    GM_ADDR decays_handle,
    GM_ADDR output_handle,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(KdaReplayssmVerifyRecordTilingData);
  GET_TILING_DATA_WITH_STRUCT(KdaReplayssmVerifyRecordTilingData, data, tiling);
  (void)workspace;
  if ASCEND_IS_AIV {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> memory;
    pipe.InitBuffer(memory, kUbBytes);
    const int32_t worker = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t total_work = data.sequences * kHeads;
    const int32_t workers = data.workers;
    const int32_t full_work = total_work / workers * workers;
    const int32_t remaining_work = total_work - full_work;
    const bool split_tail = remaining_work > 0 && remaining_work <= workers / 2;
    const int32_t tail_parts = data.sequences == 1 ? 4 : 2;
    const int32_t total_tasks =
        full_work + (split_tail ? remaining_work * tail_parts : remaining_work);
    auto query = global_tensor<bfloat16_t>(query_handle);
    auto key = global_tensor<bfloat16_t>(key_handle);
    auto value = global_tensor<bfloat16_t>(value_handle);
    auto gate = global_tensor<bfloat16_t>(gate_handle);
    auto beta = global_tensor<bfloat16_t>(beta_handle);
    auto state = global_tensor<float>(state_handle);
    auto a_log = global_tensor<float>(a_log_handle);
    auto dt_bias = global_tensor<float>(dt_bias_handle);
    auto indices = global_tensor<int32_t>(state_indices_handle);
    auto normalized_keys = global_tensor<float>(normalized_keys_handle);
    auto corrections = global_tensor<float>(corrections_handle);
    auto decays = global_tensor<float>(decays_handle);
    auto output = global_tensor<bfloat16_t>(output_handle);
    auto state_tile = memory.GetWithOffset<float>(kStateElements, 0);
    auto scratch = memory.GetWithOffset<float>(kStateElements, 65536);
    auto temporary = memory.GetWithOffset<uint8_t>(32768, 98304);
    auto query_bf16 = memory.GetWithOffset<bfloat16_t>(1024, 131072);
    auto key_bf16 = memory.GetWithOffset<bfloat16_t>(1024, 133120);
    auto value_bf16 = memory.GetWithOffset<bfloat16_t>(1024, 135168);
    auto gate_bf16 = memory.GetWithOffset<bfloat16_t>(1024, 137216);
    auto beta_bf16 = memory.GetWithOffset<bfloat16_t>(64, 139264);
    auto output_bf16 = memory.GetWithOffset<bfloat16_t>(1024, 139392);
    auto query_fp32 = memory.GetWithOffset<float>(1024, 141440);
    auto key_fp32 = memory.GetWithOffset<float>(1024, 145536);
    auto value_fp32 = memory.GetWithOffset<float>(1024, 149632);
    auto decay_fp32 = memory.GetWithOffset<float>(1024, 153728);
    auto gate_scratch = memory.GetWithOffset<float>(1024, 157824);
    auto beta_exp = memory.GetWithOffset<float>(64, 161920);
    auto correction_fp32 = memory.GetWithOffset<float>(1024, 162176);
    auto bias = memory.GetWithOffset<float>(128, 166272);
    auto norm = memory.GetWithOffset<float>(8, 166784);
    auto a_exp = memory.GetWithOffset<float>(1, 166816);
    auto predicted = memory.GetWithOffset<float>(128, 166848);
    auto broadcast = memory.GetWithOffset<float>(1024, 167360);
    auto reduction_temporary = memory.GetWithOffset<uint8_t>(4096, 131072);
    for (int32_t task = worker; task < total_tasks; task += workers) {
      const bool split_value_tail = split_tail && task >= full_work;
      const int32_t tail_task = task - full_work;
      const int32_t value_rows =
          split_value_tail ? kValueDimension / tail_parts : kValueDimension;
      const int32_t work =
          split_value_tail ? full_work + tail_task / tail_parts : task;
      const int32_t value_row =
          split_value_tail ? (tail_task % tail_parts) * value_rows : 0;
      const int32_t sequence = work / kHeads;
      const int32_t head = work % kHeads;
      const int32_t slot = indices.GetValue(sequence);
      const bool valid = slot >= 0 && slot < data.slots;
      const int64_t output_offset =
          (static_cast<int64_t>(sequence) * kWindow * kHeads + head) *
              kValueDimension +
          value_row;
      if (valid) {
        const int64_t input_offset =
            static_cast<int64_t>(sequence) * kWindow * data.token_stride +
            head * kKeyDimension;
        const int64_t gate_offset =
            (static_cast<int64_t>(sequence) * kWindow * kHeads + head) *
            kKeyDimension;
        const int64_t state_offset =
            (static_cast<int64_t>(slot) * kHeads + head) * kValueDimension *
                kKeyDimension +
            value_row * kKeyDimension;
        load_rows<bfloat16_t, kKeyDimension, kWindow>(
            query_bf16, query, input_offset, data.token_stride);
        load_rows<bfloat16_t, kKeyDimension, kWindow>(
            key_bf16, key, input_offset, data.token_stride);
        load_rows<bfloat16_t, kValueDimension, kWindow>(
            value_bf16,
            value,
            input_offset + value_row,
            data.token_stride,
            kWindow,
            value_rows);
        load_rows<float, kStateElements, 1>(state_tile,
                                            state,
                                            state_offset,
                                            kStateElements,
                                            1,
                                            value_rows * kKeyDimension);
        load_rows<bfloat16_t, kKeyDimension, kWindow>(
            gate_bf16, gate, gate_offset, kFeatures);
        load_rows<bfloat16_t, kWindow * kHeads, 1>(
            beta_bf16, beta, sequence * kWindow * kHeads, kWindow * kHeads);
        load_rows<float, 1, 1>(a_exp, a_log, head, 1);
        load_rows<float, kKeyDimension, 1>(
            bias, dt_bias, head * kKeyDimension, kKeyDimension);
        synchronize<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(
            query_fp32, query_bf16, AscendC::RoundMode::CAST_NONE, 1024);
        AscendC::Cast(key_fp32, key_bf16, AscendC::RoundMode::CAST_NONE, 1024);
        AscendC::Cast(value_fp32,
                      value_bf16,
                      AscendC::RoundMode::CAST_NONE,
                      value_rows * kWindow);
        AscendC::Cast(
            gate_scratch, gate_bf16, AscendC::RoundMode::CAST_NONE, 1024);
        AscendC::Cast(beta_exp, beta_bf16, AscendC::RoundMode::CAST_NONE, 64);
        AscendC::Exp(a_exp, a_exp, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(beta_exp, beta_exp, -1.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(beta_exp, beta_exp, 64);
        normalize_rows(query_fp32, scratch, norm, temporary);
        AscendC::Muls(query_fp32, query_fp32, data.scale, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        normalize_rows(key_fp32, scratch, norm, temporary);
        const float a_exp_value = a_exp.GetValue(0);
        float beta_values[kWindow];
        for (int32_t token = 0; token < kWindow; ++token) {
          const float denominator =
              1.0f + beta_exp.GetValue(token * kHeads + head);
          beta_values[token] = 1.0f / denominator;
        }
        for (int32_t column = 0; column < kKeyDimension;
             column += kVectorLanes) {
          AscendC::Add(gate_scratch[column],
                       gate_scratch[column],
                       bias[column],
                       static_cast<uint64_t>(kVectorLanes),
                       static_cast<uint8_t>(kWindow),
                       {1, 1, 1, kStateRepeatStride, kStateRepeatStride, 0});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(gate_scratch, gate_scratch, a_exp_value, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sigmoid(decay_fp32, gate_scratch, temporary, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(decay_fp32, decay_fp32, -5.0f, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(decay_fp32, decay_fp32, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t token = 0; token < kWindow; ++token) {
          const int32_t key_offset = token * kKeyDimension;
          const int32_t value_offset = token * value_rows;
          multiply_row(
              state_tile, state_tile, decay_fp32[key_offset], value_rows);
          AscendC::PipeBarrier<PIPE_V>();
          multiply_row(scratch, state_tile, key_fp32[key_offset], value_rows);
          AscendC::PipeBarrier<PIPE_V>();
          reduce_state_rows(
              predicted, scratch, reduction_temporary, value_rows);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Sub(correction_fp32[value_offset],
                       value_fp32[value_offset],
                       predicted,
                       value_rows);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Muls(correction_fp32[value_offset],
                        correction_fp32[value_offset],
                        beta_values[token],
                        value_rows);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Brcb(broadcast,
                        correction_fp32[value_offset],
                        static_cast<uint8_t>(value_rows / 8),
                        {1, 8});
          AscendC::PipeBarrier<PIPE_V>();
          apply_correction(
              state_tile, key_fp32[key_offset], broadcast, value_rows);
          AscendC::PipeBarrier<PIPE_V>();
          multiply_row(scratch, state_tile, query_fp32[key_offset], value_rows);
          AscendC::PipeBarrier<PIPE_V>();
          reduce_state_rows(value_fp32[value_offset],
                            scratch,
                            reduction_temporary,
                            value_rows);
          AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(output_bf16,
                      value_fp32,
                      AscendC::RoundMode::CAST_RINT,
                      value_rows * kWindow);
        AscendC::PipeBarrier<PIPE_V>();
      } else {
        AscendC::Duplicate(
            output_bf16, static_cast<bfloat16_t>(0.0f), value_rows * kWindow);
      }
      synchronize<AscendC::HardEvent::V_MTE3>();
      store_rows<bfloat16_t, kValueDimension, kWindow>(
          output, output_bf16, output_offset, kFeatures, value_rows);
      if (valid) {
        const int64_t record_offset =
            (static_cast<int64_t>(slot) * kWindow * kHeads + head) *
            kKeyDimension;
        if (value_row == 0) {
          store_rows<float, kKeyDimension, kWindow>(
              normalized_keys, key_fp32, record_offset, kFeatures);
          store_rows<float, kKeyDimension, kWindow>(
              decays, decay_fp32, record_offset, kFeatures);
        }
        store_rows<float, kValueDimension, kWindow>(corrections,
                                                    correction_fp32,
                                                    record_offset + value_row,
                                                    kFeatures,
                                                    value_rows);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}
