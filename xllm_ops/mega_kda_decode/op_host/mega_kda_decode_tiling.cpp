/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <algorithm>
#include <cstring>

#include "mega_kda_decode_policy.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
bool optional(size_t index) {
  return index == static_cast<uint32_t>(mega_kda::Input::CONV_BIAS) ||
         index == static_cast<uint32_t>(mega_kda::Input::ACCEPTED_TOKENS);
}

const gert::StorageShape* input_shape(gert::TilingContext* context,
                                      size_t index) {
  return optional(index) ? context->GetOptionalInputShape(index)
                         : context->GetRequiredInputShape(index);
}

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

static ge::graphStatus mega_kda_decode_tiling(gert::TilingContext* context) {
  using namespace mega_kda;
  const auto* attrs = context->GetAttrs();
  if (!attrs || !context->GetPlatformInfo()) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < 4; ++i) {
    if (!attrs->GetInt(i)) {
      return ge::GRAPH_FAILED;
    }
  }
  for (size_t i = 0; i < static_cast<uint32_t>(Input::INPUT_COUNT); ++i) {
    if (!optional(i) && !input_shape(context, i)) {
      return ge::GRAPH_FAILED;
    }
  }
  const auto& qkv = input_shape(context, static_cast<uint32_t>(Input::QKV))
                        ->GetStorageShape();
  const auto& conv =
      input_shape(context, static_cast<uint32_t>(Input::CONV_STATE_IN))
          ->GetStorageShape();
  const auto& ssm =
      input_shape(context, static_cast<uint32_t>(Input::SSM_STATE_IN))
          ->GetStorageShape();
  const auto& offsets =
      input_shape(context, static_cast<uint32_t>(Input::CU_SEQLENS))
          ->GetStorageShape();
  if (qkv.GetDimNum() != 2 || conv.GetDimNum() != 3 || ssm.GetDimNum() != 4 ||
      offsets.GetDimNum() != 1 || offsets.GetDim(0) < 2) {
    return ge::GRAPH_FAILED;
  }
  TilingData t{offsets.GetDim(0) - 1,
               qkv.GetDim(0),
               ssm.GetDim(1),
               *attrs->GetInt(static_cast<uint32_t>(Attr::MAX_QUERY_TOKENS)),
               conv.GetDim(1),
               conv.GetDim(0),
               *attrs->GetInt(static_cast<uint32_t>(Attr::CONV_OUTPUT_SLOTS)),
               ssm.GetDim(0),
               *attrs->GetInt(static_cast<uint32_t>(Attr::SSM_OUTPUT_SLOTS))};
  const uint64_t key = select_key(
      *attrs->GetInt(static_cast<uint32_t>(Attr::MODE)),
      input_shape(context, static_cast<uint32_t>(Input::ACCEPTED_TOKENS)) !=
          nullptr,
      t);
  if (!key) {
    return ge::GRAPH_FAILED;
  }
  const int64_t c = kQkvParts * t.heads * kHeadDim;
  auto input =
      [&](size_t i, std::initializer_list<int64_t> dims, ge::DataType dtype) {
        const auto* shape = input_shape(context, i);
        const auto* desc = optional(i) ? context->GetOptionalInputDesc(i)
                                       : context->GetRequiredInputDesc(i);
        return shape && desc && desc->GetDataType() == dtype &&
               match(shape->GetStorageShape(), dims);
      };
  if (!input(static_cast<uint32_t>(Input::QKV), {t.tokens, c}, ge::DT_BF16) ||
      !input(static_cast<uint32_t>(Input::GATE),
             {t.tokens, t.heads, kHeadDim},
             ge::DT_BF16) ||
      !input(static_cast<uint32_t>(Input::BETA),
             {t.tokens, t.heads},
             ge::DT_BF16) ||
      !input(static_cast<uint32_t>(Input::A_LOG), {t.heads}, ge::DT_FLOAT) ||
      !input(static_cast<uint32_t>(Input::GATE_BIAS),
             {t.heads, kHeadDim},
             ge::DT_FLOAT) ||
      !input(static_cast<uint32_t>(Input::CONV_WEIGHT),
             {kConvTaps, c},
             ge::DT_BF16) ||
      !input(static_cast<uint32_t>(Input::CONV_STATE_IN),
             {t.conv_input_slots, t.conv_history, c},
             ge::DT_BF16) ||
      !input(static_cast<uint32_t>(Input::SSM_STATE_IN),
             {t.ssm_input_slots, t.heads, kHeadDim, kHeadDim},
             ge::DT_FLOAT) ||
      !input(static_cast<uint32_t>(Input::CU_SEQLENS),
             {t.batch + 1},
             ge::DT_INT32) ||
      !input(static_cast<uint32_t>(Input::CONV_READ_INDICES),
             {t.batch},
             ge::DT_INT32) ||
      !input(static_cast<uint32_t>(Input::CONV_WRITE_INDICES),
             {t.batch},
             ge::DT_INT32) ||
      !input(static_cast<uint32_t>(Input::SSM_READ_INDICES),
             {t.batch, t.max_query_tokens},
             ge::DT_INT32) ||
      !input(static_cast<uint32_t>(Input::SSM_WRITE_INDICES),
             {t.batch, t.max_query_tokens},
             ge::DT_INT32) ||
      (input_shape(context, static_cast<uint32_t>(Input::CONV_BIAS)) &&
       !input(static_cast<uint32_t>(Input::CONV_BIAS), {c}, ge::DT_BF16)) ||
      (input_shape(context, static_cast<uint32_t>(Input::ACCEPTED_TOKENS)) &&
       !input(static_cast<uint32_t>(Input::ACCEPTED_TOKENS),
              {t.batch},
              ge::DT_INT32))) {
    return ge::GRAPH_FAILED;
  }
  auto output =
      [&](size_t i, std::initializer_list<int64_t> dims, ge::DataType dtype) {
        const auto* shape = context->GetOutputShape(i);
        const auto* desc = context->GetOutputDesc(i);
        return shape && desc && desc->GetDataType() == dtype &&
               match(shape->GetStorageShape(), dims);
      };
  if (!output(0, {t.tokens, t.heads, kHeadDim}, ge::DT_BF16) ||
      !output(1, {t.conv_output_slots, t.conv_history, c}, ge::DT_BF16) ||
      !output(
          2, {t.ssm_output_slots, t.heads, kHeadDim, kHeadDim}, ge::DT_FLOAT)) {
    return ge::GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  if (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950 &&
      *attrs->GetInt(static_cast<uint32_t>(Attr::MODE)) == 3) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t cores = platform.GetCoreNumAiv();
  const uint32_t blocks = std::min<int64_t>(cores, t.batch * t.heads * 8);
  auto* raw = context->GetRawTilingData();
  auto* workspace = context->GetWorkspaceSizes(1);
  if (!blocks || !raw || !raw->GetData() || raw->GetCapacity() < sizeof(t) ||
      !workspace) {
    return ge::GRAPH_FAILED;
  }
  std::memcpy(raw->GetData(), &t, sizeof(t));
  raw->SetDataSize(sizeof(t));
  workspace[0] = 0;
  context->SetTilingKey(key);
  context->SetBlockDim(blocks);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MegaKdaDecode).Tiling(mega_kda_decode_tiling);
}  // namespace optiling
