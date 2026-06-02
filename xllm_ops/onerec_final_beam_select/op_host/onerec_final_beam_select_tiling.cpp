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

#include "onerec_final_beam_select_tiling.h"

#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")
#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")

constexpr uint32_t MAX_SUPPORT_REQUEST_NUM = 48;
constexpr uint32_t MAX_SUPPORT_TOPK_INNER = 2048;

namespace optiling {
class TilingOnerecFinalBeamSelectFunc {
 public:
  explicit TilingOnerecFinalBeamSelectFunc(gert::TilingContext* tiling_context)
      : tiling_context_(tiling_context) {}

  ge::graphStatus Init();
  ge::graphStatus RunKernelTiling();

 private:
  void SetTilingKey();
  void FillTilingData();

  OnerecFinalBeamSelectTilingData tiling_data_;
  gert::TilingContext* tiling_context_ = nullptr;

  uint32_t num_sequences_ = 0;
  uint32_t active_beam_width_ = 0;
  uint32_t candidate_top_k_ = 0;
  uint32_t result_width_ = 0;
  uint32_t request_num_ = 0;
  uint32_t current_step_ = 0;
  uint32_t max_decode_step_ = 0;
  uint32_t core_num_ = 0;
  uint32_t step_size_ = 0;
  uint32_t min_size_ = 0;
  size_t sync_workspace_size_ = 0;
};

ge::graphStatus TilingOnerecFinalBeamSelectFunc::Init() {
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  core_num_ = platform_info.GetCoreNumAiv();

  auto logprobs_shape = tiling_context_->GetInputShape(0)->GetOriginShape();
  auto top_tokens_shape = tiling_context_->GetInputShape(1)->GetOriginShape();
  auto sequence_shape = tiling_context_->GetInputShape(3)->GetOriginShape();
  auto out_token_ids_shape =
      tiling_context_->GetOutputShape(0)->GetOriginShape();
  auto out_sequence_shape = tiling_context_->GetOutputShape(3)->GetOriginShape();

  current_step_ =
      static_cast<uint32_t>(*(tiling_context_->GetAttrs()->GetAttrPointer<int>(
          0)));
  num_sequences_ = logprobs_shape.GetDim(logprobs_shape.GetDimNum() - 2);
  active_beam_width_ =
      sequence_shape.GetDim(sequence_shape.GetDimNum() - 2);
  max_decode_step_ = sequence_shape.GetDim(sequence_shape.GetDimNum() - 1);
  candidate_top_k_ =
      top_tokens_shape.GetDim(top_tokens_shape.GetDimNum() - 1);

  if (active_beam_width_ == 0 || candidate_top_k_ == 0) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "active_beam_width and candidate_top_k must be positive");
    return ge::GRAPH_FAILED;
  }
  if (num_sequences_ % active_beam_width_ != 0) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "logprobs rows must be divisible by active_beam_width");
    return ge::GRAPH_FAILED;
  }
  request_num_ = num_sequences_ / active_beam_width_;
  if (request_num_ > MAX_SUPPORT_REQUEST_NUM) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "request_num must be less than or equal to %u",
            MAX_SUPPORT_REQUEST_NUM);
    return ge::GRAPH_FAILED;
  }
  uint32_t output_rows =
      out_token_ids_shape.GetDim(out_token_ids_shape.GetDimNum() - 2);
  if (output_rows == 0 || output_rows % request_num_ != 0) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "out_token_ids rows must be positive and divisible by request_num");
    return ge::GRAPH_FAILED;
  }
  result_width_ = output_rows / request_num_;
  if (result_width_ == 0) {
    OP_LOGE(tiling_context_->GetNodeName(), "result_width must be positive");
    return ge::GRAPH_FAILED;
  }
  if (out_sequence_shape.GetDim(out_sequence_shape.GetDimNum() - 3) !=
          request_num_ ||
      out_sequence_shape.GetDim(out_sequence_shape.GetDimNum() - 2) !=
          result_width_ ||
      out_sequence_shape.GetDim(out_sequence_shape.GetDimNum() - 1) !=
          max_decode_step_) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "out_sequence shape must be [request_num, result_width, "
            "max_decode_step]");
    return ge::GRAPH_FAILED;
  }
  if (result_width_ > active_beam_width_ * candidate_top_k_) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "result_width must not exceed active_beam_width * candidate_top_k");
    return ge::GRAPH_FAILED;
  }
  if (result_width_ % 32 != 0 || candidate_top_k_ % 8 != 0) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "result_width must be a multiple of 32 and candidate_top_k must be "
            "a multiple of 8");
    return ge::GRAPH_FAILED;
  }

  const uint32_t align_candidate_top_k =
      (candidate_top_k_ + 7) / 8 * 8;
  if (align_candidate_top_k > MAX_SUPPORT_TOPK_INNER) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "candidate_top_k align size must be less than or equal to %u",
            MAX_SUPPORT_TOPK_INNER);
    return ge::GRAPH_FAILED;
  }
  if (result_width_ * 2 > MAX_SUPPORT_TOPK_INNER) {
    OP_LOGE(tiling_context_->GetNodeName(),
            "result_width merge size must be less than or equal to %u",
            MAX_SUPPORT_TOPK_INNER);
    return ge::GRAPH_FAILED;
  }
  step_size_ =
      std::min<uint32_t>(8, MAX_SUPPORT_TOPK_INNER / align_candidate_top_k);
  const uint32_t align_result_width =
      (result_width_ + 31) / 32 * 32;
  const uint32_t block_size =
      (step_size_ * align_candidate_top_k + 31) / 32 * 32;
  const uint32_t merge_block_size = (result_width_ * 2 + 31) / 32 * 32;
  const uint32_t dtype_size = sizeof(float);
  uint32_t max_size = 0;
  uint32_t min_size0 = 0;
  uint32_t min_size1 = 0;
  AscendC::TopKTilingFunc(platform_info,
                          block_size,
                          1,
                          align_candidate_top_k,
                          dtype_size,
                          false,
                          AscendC::TopKMode::TOPK_NORMAL,
                          true,
                          tiling_data_.firstTopkTilingData);
  AscendC::GetTopKMaxMinTmpSize(platform_info,
                                block_size,
                                1,
                                false,
                                false,
                                AscendC::TopKMode::TOPK_NORMAL,
                                true,
                                dtype_size,
                                max_size,
                                min_size0);
  AscendC::TopKTilingFunc(platform_info,
                          merge_block_size,
                          1,
                          align_result_width,
                          dtype_size,
                          true,
                          AscendC::TopKMode::TOPK_NORMAL,
                          true,
                          tiling_data_.mergeTopkTilingData);
  AscendC::GetTopKMaxMinTmpSize(platform_info,
                                merge_block_size,
                                1,
                                false,
                                true,
                                AscendC::TopKMode::TOPK_NORMAL,
                                true,
                                dtype_size,
                                max_size,
                                min_size1);
  min_size_ = std::max(min_size0, min_size1);

  sync_workspace_size_ =
      static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
  return ge::GRAPH_SUCCESS;
}

void TilingOnerecFinalBeamSelectFunc::SetTilingKey() {
  tiling_context_->SetTilingKey(0);
}

void TilingOnerecFinalBeamSelectFunc::FillTilingData() {
  tiling_data_.set_num_sequences(num_sequences_);
  tiling_data_.set_active_beam_width(active_beam_width_);
  tiling_data_.set_candidate_top_k(candidate_top_k_);
  tiling_data_.set_result_width(result_width_);
  tiling_data_.set_request_num(request_num_);
  tiling_data_.set_current_step(current_step_);
  tiling_data_.set_max_decode_step(max_decode_step_);
  tiling_data_.set_core_num(core_num_);
  tiling_data_.set_step_size(step_size_);
  tiling_data_.set_min_size(min_size_);
}

ge::graphStatus TilingOnerecFinalBeamSelectFunc::RunKernelTiling() {
  SetTilingKey();
  FillTilingData();
  size_t user_workspace_size = 0;
  size_t* current_workspace = tiling_context_->GetWorkspaceSizes(1);
  current_workspace[0] = user_workspace_size + sync_workspace_size_;
  tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                            tiling_context_->GetRawTilingData()->GetCapacity());
  tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
  tiling_context_->SetBlockDim(core_num_);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForOnerecFinalBeamSelectFunc(
    gert::TilingContext* context) {
  TilingOnerecFinalBeamSelectFunc tiling_object(context);
  auto ret = tiling_object.Init();
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(context->GetNodeName(), "tiling Init failed.");
    return ge::GRAPH_FAILED;
  }
  ret = tiling_object.RunKernelTiling();
  return ret;
}

// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(OnerecFinalBeamSelect)
    .Tiling(TilingForOnerecFinalBeamSelectFunc);
}  // namespace optiling