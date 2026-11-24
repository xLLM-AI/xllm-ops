/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <cstring>

#include "mega_kda_prefill_policy.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
bool match(const gert::Shape& shape, std::initializer_list<int64_t> dims) {
  if (shape.GetDimNum() != dims.size()) {
    return false;
  }
  size_t i = 0;
  for (int64_t dim : dims) {
    if (shape.GetDim(i++) != dim) {
      return false;
    }
  }
  return true;
}
}  // namespace
static ge::graphStatus mega_kda_prefill_tiling(gert::TilingContext* context) {
  using namespace kda_prefill;
  const auto* attrs = context->GetAttrs();
  if (!attrs || !context->GetPlatformInfo()) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < 3; ++i) {
    if (!attrs->GetInt(i)) {
      return ge::GRAPH_FAILED;
    }
  }
  auto shape = [&](size_t i) {
    return i == static_cast<uint32_t>(Input::CONV_BIAS)
               ? context->GetOptionalInputShape(i)
               : context->GetRequiredInputShape(i);
  };
  for (size_t i = 0; i < static_cast<uint32_t>(Input::INPUT_COUNT); ++i) {
    if (i != static_cast<uint32_t>(Input::CONV_BIAS) && !shape(i)) {
      return ge::GRAPH_FAILED;
    }
  }
  const auto& qkv = shape(static_cast<uint32_t>(Input::QKV))->GetStorageShape();
  const auto& conv =
      shape(static_cast<uint32_t>(Input::CONV_STATE_IN))->GetStorageShape();
  const auto& ssm =
      shape(static_cast<uint32_t>(Input::SSM_STATE_IN))->GetStorageShape();
  const auto& cu =
      shape(static_cast<uint32_t>(Input::CU_SEQLENS))->GetStorageShape();
  if (qkv.GetDimNum() != 2 || conv.GetDimNum() != 3 || ssm.GetDimNum() != 4 ||
      cu.GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  TilingData t{cu.GetDim(0) - 1,
               qkv.GetDim(0),
               ssm.GetDim(1),
               conv.GetDim(0),
               *attrs->GetInt(static_cast<uint32_t>(Attr::CONV_OUTPUT_SLOTS)),
               ssm.GetDim(0),
               *attrs->GetInt(static_cast<uint32_t>(Attr::SSM_OUTPUT_SLOTS)),
               platform.GetCoreNumAic(),
               static_cast<uint64_t>(
                   *attrs->GetInt(static_cast<uint32_t>(Attr::FFTS_ADDR)))};
  const bool is_a5 = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
  if (!valid_tiling(t, is_a5)) {
    return ge::GRAPH_FAILED;
  }
  const int64_t c = t.heads * kQkvParts * kHeadDim;
  auto input =
      [&](size_t i, std::initializer_list<int64_t> dims, ge::DataType dtype) {
        const auto* desc = i == static_cast<uint32_t>(Input::CONV_BIAS)
                               ? context->GetOptionalInputDesc(i)
                               : context->GetRequiredInputDesc(i);
        return shape(i) && desc && desc->GetDataType() == dtype &&
               match(shape(i)->GetStorageShape(), dims);
      };
  if (!input(static_cast<uint32_t>(Input::QKV), {t.tokens, c}, ge::DT_BF16) ||
      !input(static_cast<uint32_t>(Input::GATE),
             {t.tokens, t.heads, kHeadDim},
             ge::DT_BF16) ||
      !input(static_cast<uint32_t>(Input::BETA),
             {t.tokens, t.heads},
             ge::DT_FLOAT) ||
      !input(static_cast<uint32_t>(Input::A_LOG), {t.heads}, ge::DT_FLOAT) ||
      !input(static_cast<uint32_t>(Input::GATE_BIAS),
             {t.heads, kHeadDim},
             ge::DT_FLOAT) ||
      !input(static_cast<uint32_t>(Input::CONV_WEIGHT),
             {kConvTaps, c},
             ge::DT_BF16) ||
      (shape(static_cast<uint32_t>(Input::CONV_BIAS)) &&
       !input(static_cast<uint32_t>(Input::CONV_BIAS), {c}, ge::DT_BF16)) ||
      !input(static_cast<uint32_t>(Input::CONV_STATE_IN),
             {t.conv_input_slots, kConvHistory, c},
             ge::DT_BF16) ||
      !input(static_cast<uint32_t>(Input::SSM_STATE_IN),
             {t.ssm_input_slots, t.heads, kHeadDim, kHeadDim},
             ge::DT_FLOAT) ||
      !input(static_cast<uint32_t>(Input::CU_SEQLENS),
             {t.batch + 1},
             ge::DT_INT32)) {
    return ge::GRAPH_FAILED;
  }
  for (uint32_t i : {static_cast<uint32_t>(Input::CONV_READ_INDICES),
                     static_cast<uint32_t>(Input::CONV_WRITE_INDICES),
                     static_cast<uint32_t>(Input::SSM_READ_INDICES),
                     static_cast<uint32_t>(Input::SSM_WRITE_INDICES)}) {
    if (!input(i, {t.batch}, ge::DT_INT32)) {
      return ge::GRAPH_FAILED;
    }
  }
  auto output =
      [&](size_t i, std::initializer_list<int64_t> dims, ge::DataType dtype) {
        const auto* s = context->GetOutputShape(i);
        const auto* d = context->GetOutputDesc(i);
        return s && d && d->GetDataType() == dtype &&
               match(s->GetStorageShape(), dims);
      };
  if (!output(0, {t.tokens, t.heads, kHeadDim}, ge::DT_BF16) ||
      !output(1, {t.conv_output_slots, kConvHistory, c}, ge::DT_BF16) ||
      !output(
          2, {t.ssm_output_slots, t.heads, kHeadDim, kHeadDim}, ge::DT_FLOAT)) {
    return ge::GRAPH_FAILED;
  }
  auto* raw = context->GetRawTilingData();
  auto* workspace = context->GetWorkspaceSizes(1);
  if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(t) || !workspace) {
    return ge::GRAPH_FAILED;
  }
  std::memcpy(raw->GetData(), &t, sizeof(t));
  raw->SetDataSize(sizeof(t));
  const int64_t chunk_size = chunk_size_for_soc(is_a5);
  workspace[0] = Workspace(t, chunk_size).bytes + platform.GetLibApiWorkSpaceSize();
  context->SetTilingKey(is_a5 ? 1 : 2);
  context->SetBlockDim(t.blocks);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MegaKdaPrefill).Tiling(mega_kda_prefill_tiling);
}  // namespace optiling
