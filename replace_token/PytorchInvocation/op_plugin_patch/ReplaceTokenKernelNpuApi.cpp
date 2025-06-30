/**
 * @file AddCustomKernelNpuApi.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_replace_token(const at::Tensor &x, const at::Tensor &y)
{
    at::Tensor result = npu_preparation::apply_tensor_without_format(x); // Create output memory

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnReplaceToken, x, y, result);
    return result;
}
} // namespace op_api
