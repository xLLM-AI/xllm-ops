// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License").
// http://www.apache.org/licenses/LICENSE-2.0

#include <aclnn/acl_meta.h>

#include "test/python_test/pytorch_npu_helper.hpp"

int64_t test_bridge_descriptor_dtype(const at::Tensor& tensor) {
  aclTensor* descriptor = ConvertType(tensor);
  TORCH_CHECK(descriptor != nullptr, "test bridge returned no descriptor");
  aclDataType dtype = ACL_DT_UNDEFINED;
  const int32_t status = aclGetDataType(descriptor, &dtype);
  Release(descriptor);
  TORCH_CHECK(status == 0, "aclGetDataType failed: ", status);
  TORCH_CHECK(ConvertType(tensor.scalar_type()) == dtype,
              "test bridge scalar/tensor mappings disagree");
  return static_cast<int64_t>(dtype);
}
