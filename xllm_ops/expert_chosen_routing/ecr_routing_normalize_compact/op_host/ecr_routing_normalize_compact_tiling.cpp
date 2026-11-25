/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "ecr_routing_normalize_compact_tiling.h"

#include <algorithm>
#include <cstdint>

#include "ecr_routing_normalize_compact_contract.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

ge::graphStatus CheckInputs(gert::TilingContext* context,
                            const ops::ecr_compact::Contract& contract) {
  for (size_t i = 0; i < ops::ecr_compact::kInputCount; ++i) {
    const auto* desc = context->GetInputDesc(i);
    const auto* shape = context->GetInputShape(i);
    if (desc == nullptr || shape == nullptr) {
      return ge::GRAPH_FAILED;
    }
    if (desc->GetDataType() != ops::ecr_compact::InputType(i) ||
        desc->GetStorageFormat() != ge::FORMAT_ND) {
      return ge::GRAPH_FAILED;
    }
    const int64_t first = i < 3 ? contract.e : contract.s;
    const int64_t second =
        i == 0 ? contract.s : static_cast<int64_t>(contract.b);
    if (!ops::ecr_compact::Matches(
            shape->GetOriginShape(), first, second, i >= 3, false)) {
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
  const auto* prob_shape = context->GetInputShape(ops::ecr_compact::kInputProb);
  if (prob_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ops::ecr_compact::Contract contract;
  if (!ops::ecr_compact::BuildContract(
          context, prob_shape->GetOriginShape(), false, contract)) {
    return ge::GRAPH_FAILED;
  }
  if (CheckInputs(context, contract) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  EcrRoutingNormalizeCompactTilingData tiling;
  tiling.set_s(static_cast<uint32_t>(contract.s));
  tiling.set_e(static_cast<uint32_t>(contract.e));
  tiling.set_b(contract.b);
  tiling.set_capacity_c(static_cast<uint32_t>(contract.capacity_c));
  tiling.set_top_p(contract.top_p);
  tiling.set_eps(contract.eps);
  tiling.set_s3_total_tasks(contract.s3_total_tasks);
  tiling.set_seq_base(contract.seq_base);
  tiling.set_seq_len(contract.seq_len);
  tiling.set_seq_k(contract.seq_k);
  tiling.set_c_offset(contract.c_offset);
  tiling.set_s3_task_base(contract.s3_task_base);

  auto* raw_tiling = context->GetRawTilingData();
  if (raw_tiling == nullptr || raw_tiling->GetData() == nullptr ||
      raw_tiling->GetCapacity() < tiling.GetDataSize()) {
    return ge::GRAPH_FAILED;
  }
  tiling.SaveToBuffer(raw_tiling->GetData(), raw_tiling->GetCapacity());
  raw_tiling->SetDataSize(tiling.GetDataSize());

  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  const int64_t tasks = contract.e * static_cast<int64_t>(contract.b);
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

IMPL_OP_OPTILING(EcrRoutingNormalizeCompact).Tiling(TilingFunc);
}  // namespace optiling
