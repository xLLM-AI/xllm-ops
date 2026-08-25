/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include <algorithm>
#include <array>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "mega_gdn_draft_decode_tiling.h"

namespace optiling {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kMaxBatchSize = 32;
constexpr int64_t kMaxNumCacheSlots = 1024;
constexpr uint64_t kRequiredUbBytes = 175360;
constexpr size_t kFlaSsmStateLayoutAttr = 0;
constexpr uint64_t kFlaTilingKey = 1;
constexpr uint64_t kNonFlaTilingKey = 1001;
constexpr uint32_t kAivPerA2A3MixedBlock = 2;
constexpr size_t kA5ReservedWorkspaceBytes = 16 * 1024 * 1024;
constexpr size_t kA5SoftSyncWorkspaceBytes = 4 * 1024;

enum InputIndex : size_t {
  kQkv = 0,
  kZ = 1,
  kB = 2,
  kA = 3,
  kConvWeight = 4,
  kConvState = 5,
  kALog = 6,
  kDtBias = 7,
  kSsmState = 8,
  kReadStateIndices = 9,
  kWriteStateIndices = 10,
  kQCuSeqLens = 11,
  kStateValidityMask = 12,
  kNormWeight = 13,
};

struct ShapeInfo {
  int64_t total_tokens = 0;
  int64_t batch_size = 0;
  int64_t num_k_heads = 0;
  int64_t num_v_heads = 0;
  int64_t conv_tile_count = 0;
};

bool HasShape(const gert::Shape& shape,
              std::initializer_list<int64_t> dims) {
  if (shape.GetDimNum() != dims.size()) {
    return false;
  }
  size_t index = 0;
  for (const int64_t dim : dims) {
    if (shape.GetDim(index++) != dim) {
      return false;
    }
  }
  return true;
}

bool IsSupportedHeadGeometry(int64_t num_k_heads, int64_t num_v_heads) {
  if (num_k_heads < 1 || num_k_heads > 16 ||
      (num_k_heads & (num_k_heads - 1)) != 0 ||
      num_v_heads % num_k_heads != 0) {
    return false;
  }
  const int64_t v_heads_per_k = num_v_heads / num_k_heads;
  return v_heads_per_k >= 1 && v_heads_per_k <= 4;
}

ge::graphStatus GetShapeInfo(gert::TilingContext* context,
                             ShapeInfo& info) {
  constexpr std::array<size_t, 14> kRequiredInputs = {
      kQkv, kZ, kB, kA, kConvWeight, kConvState, kALog, kDtBias,
      kSsmState, kReadStateIndices, kWriteStateIndices, kQCuSeqLens,
      kStateValidityMask, kNormWeight};
  for (const size_t index : kRequiredInputs) {
    if (context->GetInputShape(index) == nullptr ||
        context->GetInputDesc(index) == nullptr) {
      return ge::GRAPH_FAILED;
    }
  }

  const gert::Shape& qkv = context->GetInputShape(kQkv)->GetStorageShape();
  const gert::Shape& z = context->GetInputShape(kZ)->GetStorageShape();
  const gert::Shape& b = context->GetInputShape(kB)->GetStorageShape();
  const gert::Shape& a = context->GetInputShape(kA)->GetStorageShape();
  const gert::Shape& conv_weight =
      context->GetInputShape(kConvWeight)->GetStorageShape();
  const gert::Shape& conv_state =
      context->GetInputShape(kConvState)->GetStorageShape();
  const gert::Shape& a_log =
      context->GetInputShape(kALog)->GetStorageShape();
  const gert::Shape& dt_bias =
      context->GetInputShape(kDtBias)->GetStorageShape();
  const gert::Shape& ssm_state =
      context->GetInputShape(kSsmState)->GetStorageShape();
  const gert::Shape& read_indices =
      context->GetInputShape(kReadStateIndices)->GetStorageShape();
  const gert::Shape& write_indices =
      context->GetInputShape(kWriteStateIndices)->GetStorageShape();
  const gert::Shape& q_cu_seq_lens =
      context->GetInputShape(kQCuSeqLens)->GetStorageShape();
  const gert::Shape& validity =
      context->GetInputShape(kStateValidityMask)->GetStorageShape();
  const gert::Shape& norm_weight =
      context->GetInputShape(kNormWeight)->GetStorageShape();

  if (qkv.GetDimNum() != 2 || z.GetDimNum() != 3 ||
      b.GetDimNum() != 2 || a.GetDimNum() != 2 ||
      conv_weight.GetDimNum() != 2 || conv_state.GetDimNum() != 3 ||
      a_log.GetDimNum() != 1 || dt_bias.GetDimNum() != 1 ||
      ssm_state.GetDimNum() != 4 || read_indices.GetDimNum() != 1 ||
      write_indices.GetDimNum() != 1 || q_cu_seq_lens.GetDimNum() != 1 ||
      validity.GetDimNum() != 1 || norm_weight.GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }

  info.total_tokens = qkv.GetDim(0);
  const int64_t conv_dim = qkv.GetDim(1);
  info.batch_size = read_indices.GetDim(0);
  info.num_v_heads = z.GetDim(1);
  const int64_t qk_width = conv_dim - info.num_v_heads * kHeadDim;
  const int64_t num_state_slots = conv_state.GetDim(0);
  if (info.total_tokens < 1 || info.batch_size < 1 ||
      info.batch_size > kMaxBatchSize ||
      info.total_tokens > 2 * info.batch_size || conv_dim <= 0 ||
      conv_dim % kHeadDim != 0 || qk_width <= 0 ||
      qk_width % (2 * kHeadDim) != 0 ||
      !HasShape(z, {info.total_tokens, info.num_v_heads, kHeadDim}) ||
      !HasShape(b, {info.total_tokens, info.num_v_heads}) ||
      !HasShape(a, {info.total_tokens, info.num_v_heads}) ||
      !HasShape(conv_weight, {4, conv_dim}) || num_state_slots < 1 ||
      num_state_slots > kMaxNumCacheSlots ||
      !HasShape(conv_state, {num_state_slots, 3, conv_dim}) ||
      !HasShape(a_log, {info.num_v_heads}) ||
      !HasShape(dt_bias, {info.num_v_heads}) ||
      !HasShape(ssm_state,
                {num_state_slots,
                 info.num_v_heads,
                 kHeadDim,
                 kHeadDim}) ||
      !HasShape(read_indices, {info.batch_size}) ||
      !HasShape(write_indices, {info.batch_size}) ||
      !HasShape(q_cu_seq_lens, {info.batch_size + 1}) ||
      !HasShape(validity, {info.batch_size}) ||
      !HasShape(norm_weight, {kHeadDim})) {
    return ge::GRAPH_FAILED;
  }

  info.num_k_heads = qk_width / (2 * kHeadDim);
  if (!IsSupportedHeadGeometry(info.num_k_heads, info.num_v_heads)) {
    return ge::GRAPH_FAILED;
  }
  info.conv_tile_count = conv_dim / kHeadDim;
  return ge::GRAPH_SUCCESS;
}

}  // namespace

static ge::graphStatus MegaGdnDraftDecodeTiling(
    gert::TilingContext* context) {
  ShapeInfo info;
  if (GetShapeInfo(context, info) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  auto platform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub_size = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  if (ub_size < kRequiredUbBytes) {
    return ge::GRAPH_FAILED;
  }
  const auto* attrs = context->GetAttrs();
  const bool* fla_ssm_state_layout =
      attrs == nullptr
          ? nullptr
          : attrs->GetAttrPointer<bool>(kFlaSsmStateLayoutAttr);
  if (fla_ssm_state_layout == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const uint32_t aic_core_count = platform.GetCoreNumAic();
  const uint32_t aiv_core_count = platform.GetCoreNumAiv();
  const uint32_t task_count = static_cast<uint32_t>(std::max(
      info.conv_tile_count, info.batch_size * info.num_v_heads));
  const uint32_t used_aiv_cores = std::min(task_count, aiv_core_count);
  const bool is_ascend950 =
      platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
  const uint32_t block_dim =
      is_ascend950
          ? used_aiv_cores
          : platform.CalcTschBlockDim(
                used_aiv_cores, aic_core_count, aiv_core_count);
  if (block_dim == 0) {
    return ge::GRAPH_FAILED;
  }

  MegaGdnDraftDecodeTilingData tiling;
  tiling.set_total_tokens(info.total_tokens);
  tiling.set_batch_size(info.batch_size);
  tiling.set_num_k_heads(info.num_k_heads);
  tiling.set_num_v_heads(info.num_v_heads);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(*fla_ssm_state_layout ? kFlaTilingKey
                                              : kNonFlaTilingKey);
  context->SetBlockDim(block_dim);
  if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  const size_t lib_workspace_size = platform.GetLibApiWorkSpaceSize();
  if (is_ascend950 && lib_workspace_size > kA5ReservedWorkspaceBytes) {
    return ge::GRAPH_FAILED;
  }
  size_t* workspace_sizes = context->GetWorkspaceSizes(1);
  workspace_sizes[0] =
      is_ascend950
          ? kA5ReservedWorkspaceBytes + kA5SoftSyncWorkspaceBytes
          : lib_workspace_size;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaGdnDraftDecode).Tiling(MegaGdnDraftDecodeTiling);

}  // namespace optiling
