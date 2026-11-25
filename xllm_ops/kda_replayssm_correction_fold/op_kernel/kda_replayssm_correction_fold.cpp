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

#include "kda_replayssm_correction_fold_layout.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace {

using namespace kda_replayssm;

template <typename ElementType>
__aicore__ inline AscendC::GlobalTensor<ElementType> global_tensor(
    __gm__ ElementType* address) {
  AscendC::GlobalTensor<ElementType> tensor;
  tensor.SetGlobalBuffer(address);
  return tensor;
}

__aicore__ inline void load_state(AscendC::LocalTensor<float> target,
                                  __gm__ float* state) {
  AscendC::DataCopyPad(target,
                       global_tensor(state),
                       AscendC::DataCopyExtParams(1, kStateBytes, 0, 0, 0),
                       AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0.0f));
}

__aicore__ inline void load_records(AscendC::LocalTensor<float> target,
                                    __gm__ float* record,
                                    uint32_t count) {
  AscendC::DataCopyPad(
      target,
      global_tensor(record),
      AscendC::DataCopyExtParams(count,
                                 kKeyDimension * sizeof(float),
                                 (kFeatures - kKeyDimension) * sizeof(float),
                                 0,
                                 0),
      AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0.0f));
}

__aicore__ inline void store_state(__gm__ float* state,
                                   AscendC::LocalTensor<float> source) {
  AscendC::DataCopyPad(global_tensor(state),
                       source,
                       AscendC::DataCopyExtParams(1, kStateBytes, 0, 0, 0));
}

__aicore__ inline void fold_token(AscendC::LocalTensor<float> state,
                                  AscendC::LocalTensor<float> key,
                                  AscendC::LocalTensor<float> decay,
                                  AscendC::LocalTensor<float> correction,
                                  AscendC::LocalTensor<float> broadcast) {
  AscendC::Brcb(
      broadcast, correction, static_cast<uint8_t>(kValueDimension / 8), {1, 8});
  for (int32_t column = 0; column < kKeyDimension; column += kVectorLanes) {
    AscendC::Mul(state[column],
                 state[column],
                 decay[column],
                 static_cast<uint64_t>(kVectorLanes),
                 static_cast<uint8_t>(kValueDimension),
                 {1, 1, 1, kStateRepeatStride, kStateRepeatStride, 0});
  }
  AscendC::PipeBarrier<PIPE_V>();
  for (int32_t column = 0; column < kKeyDimension; column += kVectorLanes) {
    AscendC::MulAddDst(state[column],
                       broadcast,
                       key[column],
                       static_cast<uint64_t>(kVectorLanes),
                       static_cast<uint8_t>(kValueDimension),
                       {1, 0, 1, kStateRepeatStride, 1, 0});
  }
  AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void fold_layers(GM_ADDR states,
                                   GM_ADDR keys,
                                   GM_ADDR corrections,
                                   GM_ADDR decays,
                                   GM_ADDR state_indices,
                                   GM_ADDR accepted_lengths,
                                   GM_ADDR states_out,
                                   const KdaReplayssmFoldTilingData& tiling) {
  AscendC::ListTensorDesc state_layers;
  AscendC::ListTensorDesc key_layers;
  AscendC::ListTensorDesc correction_layers;
  AscendC::ListTensorDesc decay_layers;
  AscendC::ListTensorDesc output_layers;
  state_layers.Init(states);
  key_layers.Init(keys);
  correction_layers.Init(corrections);
  decay_layers.Init(decays);
  output_layers.Init(states_out);
  const auto indices =
      global_tensor(reinterpret_cast<__gm__ int32_t*>(state_indices));
  const auto accepted =
      global_tensor(reinterpret_cast<__gm__ int32_t*>(accepted_lengths));

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> memory;
  pipe.InitBuffer(memory, kUbBytes);
  auto broadcast =
      memory.GetWithOffset<float>(kRecordElements, 2 * kBufferBytes);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
  const int32_t sequences = tiling.sequences;
  const int32_t workers = tiling.workers;
  const int64_t total_work =
      static_cast<int64_t>(tiling.layers) * sequences * kHeads;
  int32_t previous_accepted = 0;
  __gm__ float* previous_output = nullptr;
  uint8_t buffer = 0;
  for (int64_t work = AscendC::GetBlockIdx(); work < total_work + workers;
       work += workers) {
    int32_t current_accepted = 0;
    __gm__ float* current_output = nullptr;
    if (work < total_work) {
      const int32_t head = static_cast<int32_t>(work % kHeads);
      const int32_t sequence =
          static_cast<int32_t>((work / kHeads) % sequences);
      const int32_t layer = static_cast<int32_t>(work / (kHeads * sequences));
      const int32_t slot = indices.GetValue(sequence);
      const int32_t num_accepted = accepted.GetValue(sequence);
      if (slot >= 0 && num_accepted > 0) {
        uint64_t state_shape[4];
        AscendC::TensorDesc<float> state_desc;
        state_desc.SetShapeAddr(state_shape);
        state_layers.GetDesc(state_desc, layer);
        if (static_cast<uint64_t>(slot) < state_desc.GetShape(0)) {
          current_accepted = num_accepted < kWindow ? num_accepted : kWindow;
          const int64_t state_offset =
              (static_cast<int64_t>(slot) * kHeads + head) * kStateElements;
          const int64_t record_offset =
              (static_cast<int64_t>(slot) * kWindow * kHeads + head) *
              kKeyDimension;
          current_output =
              output_layers.GetDataPtr<float>(layer) + state_offset;
          const int32_t base = buffer * kBufferBytes;
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(buffer);
          load_state(memory.GetWithOffset<float>(kStateElements, base),
                     state_desc.GetDataPtr() + state_offset);
          load_records(
              memory.GetWithOffset<float>(kRecordElements, base + kStateBytes),
              key_layers.GetDataPtr<float>(layer) + record_offset,
              current_accepted);
          load_records(memory.GetWithOffset<float>(
                           kRecordElements, base + kStateBytes + kRecordBytes),
                       decay_layers.GetDataPtr<float>(layer) + record_offset,
                       current_accepted);
          load_records(
              memory.GetWithOffset<float>(
                  kRecordElements, base + kStateBytes + 2 * kRecordBytes),
              correction_layers.GetDataPtr<float>(layer) + record_offset,
              current_accepted);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(buffer);
        }
      }
    }
    if (previous_accepted > 0) {
      const uint8_t previous_buffer = 1 - buffer;
      const int32_t base = previous_buffer * kBufferBytes;
      auto local_state = memory.GetWithOffset<float>(kStateElements, base);
      auto local_keys =
          memory.GetWithOffset<float>(kRecordElements, base + kStateBytes);
      auto local_decays = memory.GetWithOffset<float>(
          kRecordElements, base + kStateBytes + kRecordBytes);
      auto local_corrections = memory.GetWithOffset<float>(
          kRecordElements, base + kStateBytes + 2 * kRecordBytes);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(previous_buffer);
      for (int32_t token = 0; token < previous_accepted; ++token) {
        fold_token(local_state,
                   local_keys[token * kKeyDimension],
                   local_decays[token * kKeyDimension],
                   local_corrections[token * kValueDimension],
                   broadcast);
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(previous_buffer);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(previous_buffer);
      store_state(previous_output, local_state);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(previous_buffer);
    }
    previous_accepted = current_accepted;
    previous_output = current_output;
    buffer = 1 - buffer;
  }
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
}

}  // namespace

extern "C" __global__ __aicore__ void kda_replayssm_correction_fold(
    GM_ADDR states,
    GM_ADDR normalized_keys,
    GM_ADDR corrections,
    GM_ADDR decays,
    GM_ADDR state_indices,
    GM_ADDR accepted_lengths,
    GM_ADDR states_out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(KdaReplayssmFoldTilingData);
  GET_TILING_DATA_WITH_STRUCT(KdaReplayssmFoldTilingData, data, tiling);
  (void)workspace;
  if ASCEND_IS_AIV {
    fold_layers(states,
                normalized_keys,
                corrections,
                decays,
                state_indices,
                accepted_lengths,
                states_out,
                data);
  }
}
