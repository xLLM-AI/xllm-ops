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


enum InputIndex {
    BEAM_INDEX = 0,
    X_KEY_BLOCK,
    X_VALUE_BLOCK,
    GROUP_TOKEN_NUM
};


constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;
constexpr uint64_t DIM_3 = 3;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    // Get hardware information
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t core_num = ascendcPlatform.GetCoreNumAiv();

    SelectUnsharedKVTilingData tiling;
    const gert::StorageShape* x_key_block_shape = context->GetInputShape(X_KEY_BLOCK);
    uint32_t batch = x_key_block_shape->GetStorageShape().GetDim(0);
    uint32_t beam_size = x_key_block_shape->GetStorageShape().GetDim(1);
    uint32_t max_decode_step = x_key_block_shape->GetStorageShape().GetDim(2);
    uint32_t head_num = x_key_block_shape->GetStorageShape().GetDim(3);
    uint32_t head_dim = x_key_block_shape->GetStorageShape().GetDim(4);

    uint32_t all_task_num = batch * beam_size;
    uint32_t used_core_num = std::min(all_task_num, core_num);
    uint32_t decode_step = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int>(0)));
    uint32_t block_beam_stride = max_decode_step * head_num * head_dim;
    uint32_t block_batch_stride = block_beam_stride * beam_size;

    tiling.set_batch(batch);
    tiling.set_beam_size(beam_size);
    tiling.set_head_num(head_num);
    tiling.set_head_dim(head_dim);
    tiling.set_decode_step(decode_step);
    tiling.set_max_decode_step(max_decode_step);
    tiling.set_used_core_num(used_core_num);
    tiling.set_block_batch_stride(block_batch_stride);
    tiling.set_block_beam_stride(block_beam_stride);

    context->SetBlockDim(used_core_num);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* key_block_shape = context->GetInputShape(X_KEY_BLOCK);
    const gert::Shape* value_block_shape = context->GetInputShape(X_VALUE_BLOCK);
    gert::Shape* select_key_shape = context->GetOutputShape(0);
    gert::Shape* select_value_shape = context->GetOutputShape(1);
    *select_key_shape = *key_block_shape;
    *select_value_shape = *value_block_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(X_KEY_BLOCK);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SelectUnsharedKV : public OpDef {
public:
    explicit SelectUnsharedKV(const char* name) : OpDef(name)
    {
        this->Input("beam_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x_key_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x_value_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_token_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("select_key_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("select_value_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("decode_step").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(SelectUnsharedKV);
}
