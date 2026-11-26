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

#include "t5_split_qkv_append_cache_tiling.h"
#include "register/op_def_registry.h"

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qkvShape = context->GetInputShape(0);
    const gert::Shape* pastShape = context->GetInputShape(1);
    if (qkvShape == nullptr || pastShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (qkvShape->GetDimNum() != 3 || pastShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = pastShape->GetDim(0);
    const int64_t headNum = pastShape->GetDim(1);
    const int64_t historyLen = pastShape->GetDim(2);
    const int64_t headDim = pastShape->GetDim(3);
    if (qkvShape->GetDim(0) != batch || qkvShape->GetDim(1) != 1 ||
        qkvShape->GetDim(2) != 3 * headNum * headDim) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* queryShape = context->GetOutputShape(0);
    gert::Shape* keyShape = context->GetOutputShape(1);
    gert::Shape* valueShape = context->GetOutputShape(2);
    *queryShape = *pastShape;
    *keyShape = *pastShape;
    *valueShape = *pastShape;
    queryShape->SetDim(2, 1);
    keyShape->SetDim(2, historyLen + 1);
    valueShape->SetDim(2, historyLen + 1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    for (int32_t i = 0; i < 3; ++i) {
        context->SetOutputDataType(i, inputDataType);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(T5SplitQkvAppendCache)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
} // namespace ge
