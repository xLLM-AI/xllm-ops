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
 * \file apply_top_k_top_p_with_sorted_l0.cpp
 * \brief
 */
#include "apply_top_k_top_p_with_sorted_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ApplyTopKTopPWithSorted);

const aclTensor* ApplyTopKTopPWithSorted(const aclTensor* sortedValue, const aclTensor* sortedIndices,
                                         const aclTensor* p, const aclTensor* k, aclOpExecutor* executor) {
  L0_DFX(ApplyTopKTopPWithSorted, sortedValue, sortedIndices, p, k);
  auto output = executor->AllocTensor(sortedValue->GetViewShape(), sortedValue->GetDataType());
  
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ApplyTopKTopPWithSorted,
                                         OP_INPUT(sortedValue, sortedIndices, p, k),
                                         OP_OUTPUT(output));

  return output;
}
}  // namespace l0op
