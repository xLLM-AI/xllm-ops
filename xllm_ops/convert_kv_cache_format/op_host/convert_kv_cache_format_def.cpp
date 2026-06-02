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

#include "convert_kv_cache_format_tiling.h"
#include "register/op_def_registry.h"


namespace ops {
class ConvertKvCacheFormat : public OpDef {
 public:
  explicit ConvertKvCacheFormat(const char* name) : OpDef(name) {
    this->Input("k_cache_ptr")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("v_cache_ptr")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("kv_cache_offset")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("kv_seq_len")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->Attr("is_prefill").Bool();
    this->Attr("num_kv_heads").Int();
    this->Attr("head_size_k").Int();
    this->Attr("head_size_v").Int();

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(ConvertKvCacheFormat);
}  // namespace ops
