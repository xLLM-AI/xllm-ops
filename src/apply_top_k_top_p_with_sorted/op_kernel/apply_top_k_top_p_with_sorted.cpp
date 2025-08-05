/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file apply_top_k_top_p_with_sorted.cpp
 * \brief
 */

#include "apply_top_k_top_p_with_sorted.h"

extern "C" __global__ __aicore__ void apply_top_k_top_p_with_sorted(GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        ApplyTopKTopPWithSorted<float, float, float> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ApplyTopKTopPWithSorted<half, float, half> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ApplyTopKTopPWithSorted<bfloat16_t, float, bfloat16_t> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.Process();
    }
}