/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include <initializer_list>

#include "register/op_def_registry.h"

namespace {
constexpr int64_t kHeadDim = 128;
constexpr int64_t kValueHeads = 24;
constexpr int64_t kChunkSize = 128;

enum InputIndex {
    MIXED_QKV_INDEX = 0,
    Z_INDEX,
    B_INDEX,
    A_INDEX,
    CONV_WEIGHT_INDEX,
    CONV_STATE_INDEX,
    A_LOG_INDEX,
    DT_BIAS_INDEX,
    SSM_STATE_INDEX,
    NORM_WEIGHT_INDEX,
    MASK_LOWER_INDEX,
    MASK_FULL_INDEX,
    MINUS_IDENTITY_INDEX,
    CU_SEQLENS_INDEX,
};

enum OutputIndex {
    PACKED_QKV_INDEX = 0,
    G_INDEX,
    BETA_INDEX,
    INITIAL_STATE_INDEX,
    MEGA_OUT_INDEX,
    G_SUM_INDEX,
    G_T_INDEX,
    BETA_T_INDEX,
    MEGA_A_INDEX,
    A_INV_F32_INDEX,
    A_INV_INDEX,
    W_INDEX,
    U_INDEX,
    H_INDEX,
    V_NEW_INDEX,
    FINAL_STATE_INDEX,
    CONV_STATE_OUT_INDEX,
    SSM_STATE_OUT_INDEX,
    OUT_INDEX,
};

void SetShape(gert::Shape *shape, std::initializer_list<int64_t> dims)
{
    shape->SetDimNum(dims.size());
    size_t index = 0;
    for (auto dim : dims) {
        shape->SetDim(index++, dim);
    }
}
}  // namespace

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *mixedShape = context->GetInputShape(MIXED_QKV_INDEX);
    const gert::Shape *convStateShape = context->GetInputShape(CONV_STATE_INDEX);
    const gert::Shape *ssmStateShape = context->GetInputShape(SSM_STATE_INDEX);
    if (mixedShape == nullptr || convStateShape == nullptr || ssmStateShape == nullptr ||
        mixedShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    const int64_t totalTokens = mixedShape->GetDim(0);
    const int64_t numMatrices = ((totalTokens + kChunkSize - 1) / kChunkSize) * kValueHeads;

    SetShape(context->GetOutputShape(PACKED_QKV_INDEX), {totalTokens, 5120});
    SetShape(context->GetOutputShape(G_INDEX), {1, totalTokens, kValueHeads});
    SetShape(context->GetOutputShape(BETA_INDEX), {1, totalTokens, kValueHeads});
    SetShape(context->GetOutputShape(INITIAL_STATE_INDEX), {1, kValueHeads, kHeadDim, kHeadDim});
    SetShape(context->GetOutputShape(MEGA_OUT_INDEX), {1, totalTokens, kValueHeads, kHeadDim});
    SetShape(context->GetOutputShape(G_SUM_INDEX), {1, totalTokens, kValueHeads});
    SetShape(context->GetOutputShape(G_T_INDEX), {kValueHeads, totalTokens});
    SetShape(context->GetOutputShape(BETA_T_INDEX), {kValueHeads, totalTokens});
    SetShape(context->GetOutputShape(MEGA_A_INDEX), {1, totalTokens, kValueHeads, kChunkSize});
    SetShape(context->GetOutputShape(A_INV_F32_INDEX), {1, totalTokens, kValueHeads, kChunkSize});
    SetShape(context->GetOutputShape(A_INV_INDEX), {1, totalTokens, kValueHeads, kChunkSize});
    SetShape(context->GetOutputShape(W_INDEX), {1, totalTokens, kValueHeads, kHeadDim});
    SetShape(context->GetOutputShape(U_INDEX), {1, totalTokens, kValueHeads, kHeadDim});
    SetShape(context->GetOutputShape(H_INDEX), {numMatrices, kHeadDim, kHeadDim});
    SetShape(context->GetOutputShape(V_NEW_INDEX), {1, totalTokens, kValueHeads, kHeadDim});
    SetShape(context->GetOutputShape(FINAL_STATE_INDEX), {kValueHeads, kHeadDim, kHeadDim});
    *context->GetOutputShape(CONV_STATE_OUT_INDEX) = *convStateShape;
    *context->GetOutputShape(SSM_STATE_OUT_INDEX) = *ssmStateShape;
    SetShape(context->GetOutputShape(OUT_INDEX), {totalTokens, kValueHeads, kHeadDim});
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Qwen35GdnPrefillSuperOp).InferShape(InferShape);
}  // namespace ge
