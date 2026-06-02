/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_grouped_matmul_cpu.cpp
 * \brief
 */
#include "moe_grouped_matmul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape* x_shape = context->GetDynamicInputShape(X_INDEX, 0);
    const gert::Shape* weight_shape = context->GetDynamicInputShape(WEIGHT_INDEX, 0);
    bool transpose_weight = static_cast<bool>(*(context->GetAttrs()->GetAttrPointer<bool>(0)));
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    auto weight_desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    auto weight_format = static_cast<ge::Format>(ge::GetPrimaryFormat(weight_desc->GetStorageFormat()));
    bool weight_nz = weight_format == ge::FORMAT_FRACTAL_NZ;
    int64_t dim_n;
    if (weight_nz) {
      dim_n = transpose_weight ? (weight_shape->GetDim(1) * weight_shape->GetDim(3)) : 
                                 (weight_shape->GetDim(2) * weight_shape->GetDim(4));
    } else {
      dim_n = transpose_weight ? weight_shape->GetDim(1) : weight_shape->GetDim(2);
    }
    y_shape->SetDim(1, dim_n);
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto input_dtype = context->GetDynamicInputDataType(X_INDEX, 0);
    context->SetOutputDataType(0, input_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeGroupedMatmul)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
} // namespace ge
