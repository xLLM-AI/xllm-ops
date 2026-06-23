/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "beam_search_group_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")
constexpr uint32_t MAX_SUPPORT_REQUEST_NUM = 48;
namespace optiling {


ge::graphStatus TilingBeamSearchGroupFunc::Init() {
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
     uint32_t aiv_num = platform_info.GetCoreNumAiv();
  // uint32_t aiv_num = 1;

  // check input shape
  auto log_probs_shape = tiling_context_->GetInputShape(0)->GetOriginShape(); // [request_num * beam_width, 1]
  auto top_probs_shape = tiling_context_->GetInputShape(2)->GetOriginShape(); // [request_num * beam_width, beam_width]
  auto sequence_shape = tiling_context_->GetInputShape(3)->GetOriginShape();  // [request_num, beam_width, current_step + 1]
  current_step_ = static_cast<uint32_t>(*(tiling_context_->GetAttrs()->GetAttrPointer<int>(0)));
  int top_k_attr = *(tiling_context_->GetAttrs()->GetAttrPointer<int>(1));
  // shape_num = sequence_shape.GetDim(sequence_shape.GetDimNum() - 2);
  max_decode_step_ = sequence_shape.GetDim(sequence_shape.GetDimNum() - 1);
  num_sequences_ = log_probs_shape.GetDim(log_probs_shape.GetDimNum() - 2);
  sequence_length_ = log_probs_shape.GetDim(log_probs_shape.GetDimNum() - 1);
  top_k_ = (top_k_attr > 0) ? static_cast<uint32_t>(top_k_attr)
                             : top_probs_shape.GetDim(top_probs_shape.GetDimNum() - 1);
  if (current_step_ == 0) {
      top_k_ = 1;
  }
  beam_width_ = sequence_shape.GetDim(sequence_shape.GetDimNum() - 2);
  request_num_ = num_sequences_ / beam_width_;
  if (top_k_ > beam_width_ * beam_width_) {
    top_k_ = beam_width_ * beam_width_;
  }

  if(request_num_ > MAX_SUPPORT_REQUEST_NUM) {
    OP_LOGE(context->GetNodeName(), "request_num must be less than %u", MAX_SUPPORT_REQUEST_NUM);
    return ge::GRAPH_FAILED;
  }
  core_num_ = aiv_num;
  int32_t block_size1 = 32/sizeof(float);
  int32_t align_top_k = (top_k_+block_size1-1)/block_size1*block_size1;
  // int32_t step_size = beam_width_ /48 == 0 ? beam_width_%48 : 48;
  int32_t align_beam_width1 = (beam_width_+31)/32*32;
  int32_t align_beam_width2 = (beam_width_+7)/8*8;
  // step_size_ =1024/align_beam_width;
  constexpr int MAX_INNER_SIZE = 4096;
  step_size_ = 8;
  if (step_size_ * align_beam_width2 > MAX_INNER_SIZE) {
    step_size_ = MAX_INNER_SIZE / align_beam_width2;
  }
  int32_t block_size = (step_size_ * align_beam_width2 + 31)/32*32; // device侧在[beam_width, beam_width]的probs空间中，分块计算[step_size_, beam_width] padding to [step_size_, align_beam_width2]
  uint32_t max_size = 0;
  uint32_t min_size0 = 0;
  uint32_t min_size1 = 0;
  uint32_t dtype_size = sizeof(float);
  AscendC::TopKTilingFunc(platform_info,
                          block_size,
                          1,
                          align_top_k,
                          dtype_size,
                          false,
                          AscendC::TopKMode::TOPK_NORMAL,
                          true,
                          tiling_data_.topKTilingData);
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
  int32_t block_tail = ((beam_width_ % step_size_) * align_beam_width2 + 31)/32*32;
  AscendC::TopKTilingFunc(platform_info,
                        block_tail,
                        1,
                        align_top_k,
                        dtype_size,
                        false,
                        AscendC::TopKMode::TOPK_NORMAL,
                        true,
                        tiling_data_.topKTilingDataTail);
  AscendC::GetTopKMaxMinTmpSize(platform_info,
                              block_tail,
                              1,
                              false,
                              false,
                              AscendC::TopKMode::TOPK_NORMAL,
                              true,
                              dtype_size,
                              max_size,
                              min_size0);
//   tiling.set_tmpsize(minsize);
int32_t block_size2 = (2 * align_top_k + 31)/32*32;
  AscendC::TopKTilingFunc(platform_info,
                          block_size2, // inner
                          1, // outter
                          align_top_k, // k
                          dtype_size,// dtype
                          true,
                          AscendC::TopKMode::TOPK_NORMAL,
                          true,
                          tiling_data_.topKTilingData1);
  AscendC::GetTopKMaxMinTmpSize(platform_info,
                                block_size2,
                                1,
                                false,
                                true,
                                AscendC::TopKMode::TOPK_NORMAL,
                                true,
                                dtype_size,
                                max_size,
                                min_size1);
  min_size_ = static_cast<int32_t>(std::max(min_size0, min_size1));
  sync_workspace_size_ =
      static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
  return ge::GRAPH_SUCCESS;
}

void TilingBeamSearchGroupFunc::SetTilingKey() { tiling_context_->SetTilingKey(0); }

void TilingBeamSearchGroupFunc::FillTilingData() {
  tiling_data_.set_num_sequences(num_sequences_);
  tiling_data_.set_sequence_length(sequence_length_);
  tiling_data_.set_beam_width(beam_width_);
  tiling_data_.set_top_k(top_k_);
  tiling_data_.set_request_num(request_num_);
  tiling_data_.set_core_num(core_num_);
  tiling_data_.set_min_size(min_size_);
  tiling_data_.set_step_size(step_size_);
  tiling_data_.set_current_step(current_step_);
  tiling_data_.set_max_decode_step(max_decode_step_);
}

ge::graphStatus TilingBeamSearchGroupFunc::RunKernelTiling() {
  SetTilingKey();
  FillTilingData();
  size_t userWorkspaceSize = 0;
  size_t* currentWorkspace = tiling_context_->GetWorkspaceSizes(1);
  currentWorkspace[0] = userWorkspaceSize + sync_workspace_size_;
  tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                            tiling_context_->GetRawTilingData()->GetCapacity());
  tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
  tiling_context_->SetBlockDim(core_num_);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForBeamSearchGroupFunc(gert::TilingContext* context) {
  TilingBeamSearchGroupFunc tilingObject(context);
  auto ret = tilingObject.Init();
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(context->GetNodeName(), "tiling Init failed.");
    return ge::GRAPH_FAILED;
  }
  ret = tilingObject.RunKernelTiling();
  return ret;
}

// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(BeamSearchGroup)
    .Tiling(TilingForBeamSearchGroupFunc);
}  // namespace optiling
