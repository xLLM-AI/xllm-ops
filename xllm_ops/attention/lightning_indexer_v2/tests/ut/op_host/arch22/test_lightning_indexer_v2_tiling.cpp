/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
using namespace std;

class LightningIndexerV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerV2Tiling TearDown" << std::endl;
    }
};

// when key layout is not PA_BBND, input block_table must be null (BSND/BSND with block_table)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_0)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // k            input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},            // w            input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input3
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input5
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_k    input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input7
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table  input8 (NOT null, should fail)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}, // output_idx_offset input9
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}  // metadata     input10
        },
        {
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                        // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// key shape[2] is numhead, only support 1 (BSND/PA_BBND, N2=2)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_1)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t seqused_k_list[] = {256, 256};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 16, 64, 128}, {2, 16, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{4, 64, 2, 128}, {4, 64, 2, 128}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},                                                 // k            input1 (N2=2, should fail)
            {{{2, 16, 64}, {2, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // w            input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // cu_seqlens_q input3
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, seqused_k_list}, // cu_seqlens_k input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // seqused_k    input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // cmp_residual_k input7
            {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},                 // block_table  input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // output_idx_offset input9
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}                  // metadata     input10
        },
        {
            {{{2, 16, 1, 2048}, {2, 16, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// topk must > 0 and <= 8192 (topk=10000)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_2)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t seqused_k_list[] = {256, 256};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 16, 64, 128}, {2, 16, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{4, 64, 1, 128}, {4, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // k            input1
            {{{2, 16, 64}, {2, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // cu_seqlens_q input3
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, seqused_k_list},       // cu_seqlens_k input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // seqused_k    input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // cmp_residual_k input7
            {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},                       // block_table  input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // output_idx_offset input9
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}                        // metadata     input10
        },
        {
            {{{2, 16, 1, 10000}, {2, 16, 1, 10000}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10000)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// BSND/BSND success: BF16, B=2, S1=39, N1=64, D=128, topk=2048, mask_mode=3, cmp_ratio=1
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_3)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 203547;
    std::string expectTilingData = "2 167503724608 8796093022272 274877906944 0 3 "
                                   "9223372036854775807 9223372036854775807 1 0 0 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BSND/PA_BBND success: BF16, B=2, S1=39, N1=64, D=128, block_size=16, topk=2048, mask_mode=3
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_4)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t seqused_k_list[] = {16, 16};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}},
             ge::DT_BF16,
             ge::FORMAT_ND}, // k            input1 (block_num=2, block_size=16, N2=1, D=128)
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens_q input3
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, seqused_k_list}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                 // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1090722587;
    std::string expectTilingData = "2 167503724608 8796093022224 274877906944 4294967312 3 "
                                   "9223372036854775807 9223372036854775807 1 0 0 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// topk=2049 on 910B must be rejected: >2048 and not a multiple of 1024
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_5)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2049)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// topk=3000 on 910B must be rejected: >2048 and not a multiple of 1024
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_6)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 3000}, {2, 39, 1, 3000}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3000)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
