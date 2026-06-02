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

#include "select_unshared_kv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    uint32_t layer_num = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(2)));
    for (size_t i = 0; i < layer_num; i++) {
        const gert::Shape* key_block_shape = context->GetDynamicInputShape(X_KEY_BLOCK, i);
        const gert::Shape* value_block_shape = context->GetDynamicInputShape(X_VALUE_BLOCK, i);
        gert::Shape* select_key_shape = context->GetOutputShape(i);
        gert::Shape* select_value_shape = context->GetOutputShape(i + layer_num);
        *select_key_shape = *key_block_shape;
        *select_value_shape = *value_block_shape;
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    uint32_t layer_num = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(2)));
    const auto inputDataType = context->GetDynamicInputDataType(X_KEY_BLOCK, 0);
    for (size_t i = 0; i < layer_num; i++) {
        context->SetOutputDataType(i, inputDataType);
        context->SetOutputDataType(i + layer_num, inputDataType);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SelectUnsharedKV).InferShape(InperShape).InferDataType(InferDataType);
}
