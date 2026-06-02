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

#include "register/op_def_registry.h"
#include "multi_latent_attention_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace ge {
    static ge::graphStatus InferShapeMLA(gert::InferShapeContext *context) {
        int32_t isRing = static_cast<int32_t>(*(context->GetAttrs()->GetInt(7)));
        const gert::Shape *x1_shape = context->GetInputShape(0);
        const gert::Shape *x2_shape = context->GetInputShape(1);

        gert::Shape *out0 = context->GetOutputShape(0);
        gert::Shape *out1 = context->GetOutputShape(1);

        *out0 = *x1_shape;
        if (isRing) {
            // outTensor1  lse
            *out1 = *x1_shape;
            if (x1_shape->GetDimNum() == 2) {
                out1->SetDim(1, 1);
            } else if(x1_shape->GetDimNum() == 3){
                out1->SetDim(2, 1);
            } else {
                out1->SetDim(0, 0);
            }
        } else {
            *out1 = {0};
        }
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus InferShape(gert::InferShapeContext *context) {
        AtbOps::OpParam::MLA::Type type = static_cast<AtbOps::OpParam::MLA::Type>(*(context->GetAttrs()->GetInt(0)));
        switch (type) {
            case AtbOps::OpParam::MLA::SPLIT_CACHE:
                return InferShapeMLA(context);
            default:
                break;
        }
        return ge::GRAPH_FAILED;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto x1_datatype = context->GetInputDataType(1);
        context->SetOutputDataType(0, x1_datatype);
        int32_t isRing = static_cast<int32_t>(*(context->GetAttrs()->GetInt(7)));
        if (!isRing) {
            context->SetOutputDataType(1, x1_datatype);
        }
        return ge::GRAPH_SUCCESS;
    }


    IMPL_OP(MultiLatentAttention).InferShape(InferShape).InferDataType(InferDataType);
}