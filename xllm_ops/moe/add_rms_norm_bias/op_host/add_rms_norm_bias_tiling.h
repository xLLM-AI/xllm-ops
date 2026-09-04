/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_H_
#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddRMSNormBiasTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_row);
TILING_DATA_FIELD_DEF(uint32_t, num_col);
TILING_DATA_FIELD_DEF(uint32_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint32_t, num_col_align);
TILING_DATA_FIELD_DEF(uint32_t, last_block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_loop);
TILING_DATA_FIELD_DEF(uint32_t, last_block_row_loop);
TILING_DATA_FIELD_DEF(uint32_t, row_tail);
TILING_DATA_FIELD_DEF(uint32_t, last_block_row_tail);
TILING_DATA_FIELD_DEF(uint32_t, mul_loop_fp32);
TILING_DATA_FIELD_DEF(uint32_t, mul_tail_fp32);
TILING_DATA_FIELD_DEF(uint32_t, dst_rep_stride_fp32);
TILING_DATA_FIELD_DEF(uint32_t, mul_loop_fp16);
TILING_DATA_FIELD_DEF(uint32_t, mul_tail_fp16);
TILING_DATA_FIELD_DEF(uint32_t, dst_rep_stride_fp16);
TILING_DATA_FIELD_DEF(uint32_t, is_performance);
TILING_DATA_FIELD_DEF(uint32_t, nullptr_beta);
// arch35 RegBase/VF 路径新增字段（其余分支不读）
TILING_DATA_FIELD_DEF(uint32_t, bin_add_quotient);   // 平方和二分累加分界点
TILING_DATA_FIELD_DEF(uint32_t, ub_loop);            // VF 切 D：列方向整块循环数
TILING_DATA_FIELD_DEF(uint32_t, col_bufer_length);   // VF 切 D：列缓冲长度（沿用 ops-nn 字段名）
END_TILING_DATA_DEF;

struct AddRmsNormBiasCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND950;
};

REGISTER_TILING_DATA_CLASS(AddRmsNormBias, AddRMSNormBiasTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_H_
