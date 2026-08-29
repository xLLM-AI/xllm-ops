/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class LightningIndexerV2Proto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerV2Proto TearDown" << std::endl;
    }
};

// BSND/BSND, return_value=true, topk=128, mask_mode=3, cmp_ratio=1
TEST_F(LightningIndexerV2Proto, LightningIndexerV2_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LightningIndexerV2",
        // 输入Tensor (11个)
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // k            input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // w            input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                          // cu_seqlens_q input3 (optional, null)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                          // cu_seqlens_k input4 (optional, null)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                          // seqused_q    input5 (optional, null)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                          // seqused_k    input6 (optional, null)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                          // cmp_residual_k input7 (optional, null)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                          // block_table  input8 (optional, null)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                          // output_idx_offset input9 (optional, null)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}                           // metadata     input10 (optional, null)
        },
        // 输出Tensor
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},    // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}     // sparse_values (V2 always FP32)
        },
        // 属性
        {
            {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 8, 1, 128},
        {1, 8, 1, 128}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND/PA_BBND, return_value=false, topk=2048, mask_mode=0, cmp_ratio=1
TEST_F(LightningIndexerV2Proto, LightningIndexerV2_infershape_1)
{
    int64_t cu_seqlens_q_list[] = {0, 64};
    int64_t cu_seqlens_k_list[] = {0, 256};
    gert::InfershapeContextPara infershapeContextPara(
        "LightningIndexerV2",
        // 输入Tensor (11个)
        {
            {{{64, 64, 128}, {64, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},           // q            input0
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},     // k            input1 (PA: block_num, block_size, N2, D)
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // w            input2
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cu_seqlens_q_list},     // cu_seqlens_q input3
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cu_seqlens_k_list},     // cu_seqlens_k input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // seqused_q    input5
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // seqused_k    input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // cmp_residual_k input7
            {{{1, 16}, {1, 16}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table  input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // output_idx_offset input9
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}                         // metadata     input10
        },
        // 输出Tensor
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},    // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}     // sparse_values
        },
        // 属性
        {
            {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 1, 2048},
        {0}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND/TND, return_value=true, topk=64, mask_mode=3, cmp_ratio=4
TEST_F(LightningIndexerV2Proto, LightningIndexerV2_infershape_2)
{
    int64_t cu_seqlens_q_list[] = {0, 32, 48};
    int64_t cu_seqlens_k_list[] = {0, 128, 192};
    gert::InfershapeContextPara infershapeContextPara(
        "LightningIndexerV2",
        // 输入Tensor (11个)
        {
            {{{48, 32, 128}, {48, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // q            input0
            {{{48, 1, 128}, {48, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // k            input1 (TND: compressed S2)
            {{{48, 32}, {48, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // w            input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cu_seqlens_q_list},     // cu_seqlens_q input3
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cu_seqlens_k_list},     // cu_seqlens_k input4
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // seqused_k    input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // cmp_residual_k input7
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // block_table  input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                        // output_idx_offset input9
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}                         // metadata     input10
        },
        // 输出Tensor
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},    // sparse_indices
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}     // sparse_values
        },
        // 属性
        {
            {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {48, 1, 64},
        {48, 1, 64}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// infer dataType
TEST_F(LightningIndexerV2Proto, LightningIndexerV2_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("LightningIndexerV2")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_q = ge::DT_FLOAT16;
        ge::DataType input_k = ge::DT_FLOAT16;
        ge::DataType input_w = ge::DT_FLOAT;
        ge::DataType input_i32 = ge::DT_INT32;
        ge::DataType output_ref0 = ge::DT_INT32;
        ge::DataType output_ref1 = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(11, 2)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_q, &input_k, &input_w,
                                 &input_i32, &input_i32, &input_i32, &input_i32,
                                 &input_i32, &input_i32, &input_i32, &input_i32})
                .NodeAttrs({
                    {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
                    {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                    {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                    {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                    {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                    {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                    {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
                })
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref0);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref1);
    }
}
