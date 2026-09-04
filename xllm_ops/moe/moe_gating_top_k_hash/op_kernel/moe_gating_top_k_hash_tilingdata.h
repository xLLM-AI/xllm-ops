/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_GATING_TOP_K_HASH_TILINGDATA_H
#define MOE_GATING_TOP_K_HASH_TILINGDATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// Device-side mirrors of the host tiling records.  The registry build copies
// op_kernel independently from op_host, so these declarations must be visible
// before the kernel templates are parsed.
struct MoeGatingTopKHashTilingData {
    int64_t needCoreNum;
    int64_t rowCount;
    int64_t perCoreRowCount;
    int64_t lastCoreRowCount;
    int64_t expertCount;
    int64_t addBias;
    int64_t k;
    int64_t kGroup;
    int64_t groupCount;
    int64_t perGroupExpertCount;
    int64_t perGroupExpertCountAlign;
    int64_t groupSelectMode;
    int64_t renorm;
    int64_t normType;
    int64_t outFlag;
    int64_t hashFlag;
    int64_t vmsCount;
    float routedScalingFactor;
    float eps;
    int64_t calTmpBufUbSize;
};

struct MoeGatingTopKHashRegbaseTilingData {
    int64_t needCoreNum;
    int64_t rowCount;
    int64_t perCoreRowCount;
    int64_t lastCoreRowCount;
    int64_t expertCount;
    int64_t addBias;
    int64_t k;
    int64_t kGroup;
    int64_t groupCount;
    int64_t perGroupExpertCount;
    int64_t perGroupExpertCountAlign;
    int64_t groupSelectMode;
    int64_t renorm;
    int64_t normType;
    int64_t outFlag;
    int64_t hashFlag;
    int64_t vmsCount;
    float routedScalingFactor;
    float eps;
    AscendC::tiling::SoftMaxTiling softmaxTilingData;
};

// The registry compiler permits one default tiling record per entry point.
// Both host records share this prefix and differ only in the tail, so one
// union-backed device record preserves both serialized layouts. The
// SoftMaxTiling tail is only produced by the ASCEND950 regbase host tiling
// (MoeGatingTopKHashTilingRegbase::IsCapable gates it to ASCEND950), and
// SoftMaxTiling has a non-trivial default constructor, which would delete the
// union's default constructor on the other archs where
// GET_TILING_DATA_WITH_STRUCT default-constructs this record. Keep the plain
// int64 tail there so the record stays trivially constructible.
struct MoeGatingTopKHashRegistryTilingData {
    int64_t needCoreNum;
    int64_t rowCount;
    int64_t perCoreRowCount;
    int64_t lastCoreRowCount;
    int64_t expertCount;
    int64_t addBias;
    int64_t k;
    int64_t kGroup;
    int64_t groupCount;
    int64_t perGroupExpertCount;
    int64_t perGroupExpertCountAlign;
    int64_t groupSelectMode;
    int64_t renorm;
    int64_t normType;
    int64_t outFlag;
    int64_t hashFlag;
    int64_t vmsCount;
    float routedScalingFactor;
    float eps;
#if defined(__DAV_C310__)
    union {
        int64_t calTmpBufUbSize;
        AscendC::tiling::SoftMaxTiling softmaxTilingData;
    };
#else
    int64_t calTmpBufUbSize;
#endif
};

#endif
