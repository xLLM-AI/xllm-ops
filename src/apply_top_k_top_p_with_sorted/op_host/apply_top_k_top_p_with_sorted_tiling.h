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
 * \file apply_top_k_top_p_with_sorted_tiling.h
 * \brief
 */
#ifndef __APPLY_TOP_K_TOP_P_WITH_SORTED_TILINGDATA_H__
#define __APPLY_TOP_K_TOP_P_WITH_SORTED_TILINGDATA_H__

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyTopKTopPWithSortedTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailBatch);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, dataNumInit);
    TILING_DATA_FIELD_DEF(uint32_t, dataNumInitAligned);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElementAligned);
    TILING_DATA_FIELD_DEF(uint32_t, tailUbFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, tailUbFactorElementAligned);
    TILING_DATA_FIELD_DEF(uint32_t, calUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyTopKTopPWithSorted, ApplyTopKTopPWithSortedTilingData)

struct TilingForApplyTopKTopPWithSortedCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

}  // namespace optiling
#endif  // __APPLY_TOP_K_TOP_P_WITH_SORTED_TILINGDATA_H__