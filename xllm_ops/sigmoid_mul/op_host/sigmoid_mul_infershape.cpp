/**
 * Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file sigmoid_mul_infershape.cpp
 */
#include "register/op_def_registry.h"

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    // x1 is either the same shape as x2 or [M, 1] broadcast against x2 [M, N].
    if (x2Shape != nullptr && x2Shape->GetShapeSize() > x1Shape->GetShapeSize()) {
        *outShape = *x2Shape;
    } else {
        *outShape = *x1Shape;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP(SigmoidMul).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ge
