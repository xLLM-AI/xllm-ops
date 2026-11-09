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

/*!
 * \file split_qkv_norm_rope_infershape.cpp
 * \brief SplitQkvNormRope 形状与 dtype 推导。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t IDX_QKV = 0;
constexpr size_t OUT_Q = 0;
constexpr size_t OUT_K = 1;
constexpr size_t OUT_V = 2;
constexpr int64_t kHeadDim = 128;
constexpr size_t ATTR_NUM_Q_HEADS = 0;
constexpr size_t ATTR_NUM_KV_HEADS = 1;
constexpr size_t ATTR_LAYOUT_BNSD = 4;

void SetOutputShape(gert::Shape *out, int64_t batch, int64_t seqLen, int64_t heads, bool bnsd)
{
    if (bnsd) {
        out->SetDimNum(4);
        out->SetDim(0, batch);
        out->SetDim(1, heads);
        out->SetDim(2, seqLen);
        out->SetDim(3, kHeadDim);
    } else {
        out->SetDimNum(3);
        out->SetDim(0, batch);
        out->SetDim(1, seqLen);
        out->SetDim(2, heads * kHeadDim);
    }
}
}  // namespace

static ge::graphStatus InferShapeSplitQkvNormRope(gert::InferShapeContext *context)
{
    const gert::Shape *qkvShape = context->GetInputShape(IDX_QKV);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);
    OP_CHECK_IF(qkvShape->GetDimNum() != 3, OP_LOGE(context, "qkv must be [B,S,stride]."), return GRAPH_FAILED);

    gert::Shape *qShape = context->GetOutputShape(OUT_Q);
    gert::Shape *kShape = context->GetOutputShape(OUT_K);
    gert::Shape *vShape = context->GetOutputShape(OUT_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);

    const int64_t batch = qkvShape->GetDim(0);
    const int64_t seqLen = qkvShape->GetDim(1);

    int64_t numHeadQ = 1;
    int64_t numHeadK = 1;
    bool layoutBNSD = false;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *nq = attrs->GetAttrPointer<int64_t>(ATTR_NUM_Q_HEADS);
        const int64_t *nk = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS);
        const bool *bnsd = attrs->GetAttrPointer<bool>(ATTR_LAYOUT_BNSD);
        if (nq != nullptr) {
            numHeadQ = *nq;
        }
        if (nk != nullptr) {
            numHeadK = *nk;
        }
        if (bnsd != nullptr) {
            layoutBNSD = *bnsd;
        }
    }

    SetOutputShape(qShape, batch, seqLen, numHeadQ, layoutBNSD);
    SetOutputShape(kShape, batch, seqLen, numHeadK, layoutBNSD);
    SetOutputShape(vShape, batch, seqLen, numHeadK, layoutBNSD);  // v1：Nv == Nk
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSplitQkvNormRope(gert::InferDataTypeContext *context)
{
    const ge::DataType dtype = context->GetInputDataType(IDX_QKV);
    context->SetOutputDataType(OUT_Q, dtype);
    context->SetOutputDataType(OUT_K, dtype);
    context->SetOutputDataType(OUT_V, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SplitQkvNormRope)
    .InferShape(InferShapeSplitQkvNormRope)
    .InferDataType(InferDataTypeSplitQkvNormRope);
}  // namespace ops
