/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "ecr_token_routing_statistics_tiling.h"

#include <algorithm>
#include <cstdint>

#include "ecr_token_routing_statistics_contract.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

ge::graphStatus CheckInputs(gert::TilingContext* context) {
  for (size_t i = 0; i < ops::ecr_statistics::kInputCount; ++i) {
    const auto* desc = context->GetInputDesc(i);
    if (desc == nullptr ||
        desc->GetDataType() != ops::ecr_statistics::InputType(i) ||
        desc->GetStorageFormat() != ge::FORMAT_ND) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  if (context == nullptr || context->GetPlatformInfo() == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto* prob_shape =
      context->GetInputShape(ops::ecr_statistics::kInputProb);
  const auto* tau_val_shape =
      context->GetInputShape(ops::ecr_statistics::kInputTauVal);
  const auto* tau_idx_shape =
      context->GetInputShape(ops::ecr_statistics::kInputTauIdx);
  if (prob_shape == nullptr || tau_val_shape == nullptr ||
      tau_idx_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ops::ecr_statistics::Contract contract;
  if (!ops::ecr_statistics::BuildContract(context,
                                          prob_shape->GetOriginShape(),
                                          &tau_val_shape->GetOriginShape(),
                                          &tau_idx_shape->GetOriginShape(),
                                          false,
                                          contract)) {
    return ge::GRAPH_FAILED;
  }
  if (CheckInputs(context) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  EcrTokenRoutingStatisticsTilingData tiling;
  tiling.set_s(static_cast<uint32_t>(contract.s));
  tiling.set_e(static_cast<uint32_t>(contract.e));
  tiling.set_b(contract.b);
  tiling.set_total_tasks(contract.total_tasks);
  tiling.set_seq_base(contract.seq_base);
  tiling.set_seq_len(contract.seq_len);
  tiling.set_task_base(contract.task_base);

  auto* raw_tiling = context->GetRawTilingData();
  if (raw_tiling == nullptr || raw_tiling->GetData() == nullptr ||
      raw_tiling->GetCapacity() < tiling.GetDataSize()) {
    return ge::GRAPH_FAILED;
  }
  tiling.SaveToBuffer(raw_tiling->GetData(), raw_tiling->GetCapacity());
  raw_tiling->SetDataSize(tiling.GetDataSize());

  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  const int64_t tasks = static_cast<int64_t>(contract.total_tasks);
  const int64_t cores = static_cast<int64_t>(platform.GetCoreNumAiv());
  const int64_t block_dim =
      std::max<int64_t>(1, std::min<int64_t>(cores, tasks));
  context->SetBlockDim(static_cast<uint32_t>(block_dim));
  context->SetTilingKey(0);

  size_t* workspace = context->GetWorkspaceSizes(1);
  if (workspace == nullptr) {
    return ge::GRAPH_FAILED;
  }
  workspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EcrTokenRoutingStatistics).Tiling(TilingFunc);
}  // namespace optiling
