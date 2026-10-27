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

#include "cross_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")


namespace optiling {

enum InputIndex {
    QUERY = 0,
    KEY,
    VALUE,
    BLOCK_TABLE,
    ACTUAL_KV_LENS
};

constexpr uint32_t Q_S_BLOCK_TILE = 128;
constexpr uint32_t BLOCK_SIZE = 128;
constexpr int32_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 4; // row * col * blockStackNum
constexpr int32_t SCALE_VALUE_ATTR_INDEX = 0;

class TilingCrossAttentionFunc {
  public:
    explicit TilingCrossAttentionFunc(gert::TilingContext* tiling_context)
        : tiling_context_(tiling_context) {}
    ge::graphStatus RunTiling();
  private:
    CrossAttentionTilingData tiling_data_;
    gert::TilingContext* tiling_context_ = nullptr;
    uint32_t cubeCoreNum;
    uint32_t vecCoreNum;
    uint32_t inputDtype{0};
    ge::graphStatus FillBasicTilingData();
    void FillSplitCoreTilingData();
    void SetWorkspaces();
    uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
};

uint32_t TilingCrossAttentionFunc::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = 128;
    // A trick is used to ensure the qN tile is a even number,
    // thus most tasks have balanced workload between two vec cores,
    // and each vec core possess no more than 64 rows when all-rounded row num is no larger than 128,
    // aiding the coding of rescale block
    uint32_t qNBlockTile = (qRowNumCeil / qSeqlen) / 2 * 2;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

void TilingCrossAttentionFunc::SetWorkspaces()
{
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  size_t systemWorkspaceSize = static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
  size_t userWorkspaceSize = 0;
  auto usedCoreNum = tiling_data_.get_usedCoreNum();
  // Attention occupied space
  // TODO: Only apply for one temporary space, affecting preload function, long sequence scenario needs extra processing
  uint64_t mm1OutSize = usedCoreNum * WORKSPACE_BLOCK_SIZE_DB * 3 * sizeof(float);;
  uint64_t pOutSize = usedCoreNum * WORKSPACE_BLOCK_SIZE_DB * 3 * sizeof(int16_t);
  uint64_t mm2OutSize = usedCoreNum * WORKSPACE_BLOCK_SIZE_DB * 3 * sizeof(float);
  tiling_data_.set_mm1OutSize(mm1OutSize);
  tiling_data_.set_pOutSize(pOutSize);
  tiling_data_.set_mm2OutSize(mm2OutSize);
  tiling_data_.set_wsBlockSizeDb(static_cast<uint64_t>(WORKSPACE_BLOCK_SIZE_DB));
  userWorkspaceSize = mm1OutSize + pOutSize + mm2OutSize;
  size_t* workspace = tiling_context_->GetWorkspaceSizes(1);
  workspace[0] = systemWorkspaceSize + userWorkspaceSize;
}

void TilingCrossAttentionFunc::FillSplitCoreTilingData()
{
  uint32_t totalTaskNum = 0;
  uint32_t groupSize = tiling_data_.get_groupSize();
  uint32_t qSeqlen = tiling_data_.get_qSeqlen();
  uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
  uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
  uint32_t curQNBlockNum = qNBlockNumPerGroup * tiling_data_.get_kvHeads();
  uint32_t curQSBlockTile = Q_S_BLOCK_TILE;
  uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
  uint32_t perBatchTaskNum = curQNBlockNum * curQSBlockNum;
  totalTaskNum = perBatchTaskNum * tiling_data_.get_batch();
  uint32_t usedCoreNum = std::min(totalTaskNum, cubeCoreNum);
  tiling_data_.set_qSBlockTile(curQSBlockTile);
  tiling_data_.set_qNBlockTile(curQNBlockTile);
  tiling_data_.set_qNBlockNumPerGroup(qNBlockNumPerGroup);
  tiling_data_.set_qNBlockNum(curQNBlockNum);
  tiling_data_.set_qSBlockNum(curQSBlockNum);
  tiling_data_.set_perBatchTaskNum(perBatchTaskNum);
  tiling_data_.set_totalTaskNum(totalTaskNum);
  tiling_data_.set_usedCoreNum(usedCoreNum);

  // printf("qSBlockTile %d qNBlockTile %d qNBlockNumPerGroup %d qNBlockNum %d qSBlockNum %d perBatchTaskNum %d totalTaskNum %d usedCoreNum %d\n",
  // curQSBlockTile, curQNBlockTile, qNBlockNumPerGroup, curQNBlockNum, curQSBlockNum, perBatchTaskNum, totalTaskNum, usedCoreNum);
}

ge::graphStatus TilingCrossAttentionFunc::FillBasicTilingData()
{
  auto queryShape = tiling_context_->GetInputShape(QUERY)->GetStorageShape();
  auto keyShape = tiling_context_->GetInputShape(KEY)->GetStorageShape();
  auto blockTableShape = tiling_context_->GetInputShape(BLOCK_TABLE)->GetStorageShape();

  int64_t batch = queryShape.GetDim(0);
  int64_t qSeqlen = queryShape.GetDim(1);
  int64_t qHeadNum = queryShape.GetDim(2);
  int64_t headDim = queryShape.GetDim(3);
  int64_t blockSize = keyShape.GetDim(1);
  int64_t kvHeadNum = keyShape.GetDim(2);
  int64_t maxNumBlocksPerBatch = blockTableShape.GetDim(1);
  float scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * headDim));
  auto attrs = tiling_context_->GetAttrs();
  if (attrs != nullptr) {
    const auto* attr_scale_value = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    if (attr_scale_value != nullptr && *attr_scale_value > 0.0f) {
      scaleValue = *attr_scale_value;
    }
  }
  int32_t groupSize = qHeadNum / kvHeadNum;
  // printf("tiling base info: batch %d qHeads %d kvHeads %d headDim %d groupSize %d qSeqlen %d blockSize %d maxBlocks %d\n",
  // batch, qHeadNum, kvHeadNum, headDim, groupSize, qSeqlen, blockSize, maxNumBlocksPerBatch);

  tiling_data_.set_scaleValue(scaleValue);
  tiling_data_.set_batch(static_cast<uint32_t>(batch));
  tiling_data_.set_qHeads(static_cast<uint32_t>(qHeadNum));
  tiling_data_.set_kvHeads(static_cast<uint32_t>(kvHeadNum));
  tiling_data_.set_headDim(static_cast<uint32_t>(headDim));
  tiling_data_.set_groupSize(static_cast<uint32_t>(groupSize));
  tiling_data_.set_qSeqlen(static_cast<uint32_t>(qSeqlen));
  tiling_data_.set_blockSize(static_cast<uint32_t>(blockSize));
  tiling_data_.set_maxNumBlocksPerBatch(static_cast<uint32_t>(maxNumBlocksPerBatch));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCrossAttentionFunc::RunTiling()
{
  // Get platform hardware information
  auto platform_info =
      platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
  cubeCoreNum = platform_info.GetCoreNumAic();
  vecCoreNum = platform_info.GetCoreNumAiv();

  auto ret = FillBasicTilingData();
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(tiling_context_->GetNodeName(), "fill basic tiling failed.");
    return ge::GRAPH_FAILED;
  }

  FillSplitCoreTilingData();
  SetWorkspaces();

  // Save tilingData
  tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                            tiling_context_->GetRawTilingData()->GetCapacity());
  tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
  tiling_context_->SetBlockDim(tiling_data_.get_usedCoreNum());

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  TilingCrossAttentionFunc tilingObject(context);
  auto ret = tilingObject.RunTiling();

  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(context->GetNodeName(), "CrossAttention tiling failed.");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CrossAttention)
    .Tiling(TilingFunc);
}
