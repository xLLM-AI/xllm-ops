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

#ifndef SIGMOID_MUL_TILING_H_
#define SIGMOID_MUL_TILING_H_

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SigmoidMulTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalElements);
TILING_DATA_FIELD_DEF(int64_t, elemsPerCore);
TILING_DATA_FIELD_DEF(int32_t, tileElems);
// Broadcast form: x1 is [M, 1] against x2 [M, N]; out[r][c] = sigmoid(x1[r]) * x2[r][c].
// rows/cols describe the output, rowsPerCore is the row split.
TILING_DATA_FIELD_DEF(int32_t, broadcast);
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SigmoidMul, SigmoidMulTilingData)
} // namespace optiling

#endif // SIGMOID_MUL_TILING_H_
