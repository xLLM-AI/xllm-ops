/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
*/
// Real CANN metadata; no device allocation or launch.
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "xllm_ops/dsa_kv_offloading_plan_nope/op_host/op_api/dsa_kv_offloading_stable_validate.h"

namespace {
class Metadata final {
 public:
  explicit Metadata(std::initializer_list<int64_t> shape,
                    aclDataType type = ACL_INT32) {
    std::vector<int64_t> dims(shape), strides(dims.size());
    int64_t stride = 1;
    for (size_t axis = dims.size(); axis-- > 0;) {
      strides[axis] = stride;
      stride *= dims[axis];
    }
    tensor = aclCreateTensor(dims.data(),
                             dims.size(),
                             type,
                             strides.data(),
                             0,
                             ACL_FORMAT_ND,
                             dims.data(),
                             dims.size(),
                             reinterpret_cast<void*>(0x10000000));
    assert(tensor != nullptr);
  }
  ~Metadata() { assert(aclDestroyTensor(tensor) == ACLNN_SUCCESS); }
  Metadata(const Metadata&) = delete;
  Metadata& operator=(const Metadata&) = delete;
  aclTensor* tensor;
};
}  // namespace

int main() {
  namespace dsa_kv_offloading = xllm_ops::dsa_kv_offloading;
  Metadata host({32, 128, 1, 512}, ACL_BF16);
  Metadata topk({8, 1, 2176});
  Metadata selection({64 * 17, 128, 1, 512}, ACL_BF16);
  Metadata rope({32, 128, 1, 64}, ACL_BF16);
  Metadata current({8, 1, 64}, ACL_BF16);
  Metadata selected_rope({64 * 17, 128, 1, 64}, ACL_BF16);
  assert(dsa_kv_offloading::StableRope(rope.tensor,
                        current.tensor,
                        selected_rope.tensor,
                        host.tensor,
                        topk.tensor,
                        selection.tensor));
  // Latent and rotary payload share the full resident capacity.
  assert(dsa_kv_offloading::StableSelectionRows(selection.tensor) == 64);
  Metadata short_selection({8 * 17, 128, 1, 64}, ACL_BF16);
  assert(!dsa_kv_offloading::StableRope(rope.tensor,
                         current.tensor,
                         short_selection.tensor,
                         host.tensor,
                         topk.tensor,
                         selection.tensor));
  Metadata bad_host({31, 128, 1, 64}, ACL_BF16);
  assert(!dsa_kv_offloading::StableRope(bad_host.tensor,
                         current.tensor,
                         selected_rope.tensor,
                         host.tensor,
                         topk.tensor,
                         selection.tensor));
  Metadata bad_current({64, 1, 64}, ACL_BF16);
  assert(!dsa_kv_offloading::StableRope(rope.tensor,
                         bad_current.tensor,
                         selected_rope.tensor,
                         host.tensor,
                         topk.tensor,
                         selection.tensor));
  Metadata wrong_dtype({64 * 17, 128, 1, 64}, ACL_INT32);
  assert(!dsa_kv_offloading::StableRope(rope.tensor,
                         current.tensor,
                         wrong_dtype.tensor,
                         host.tensor,
                         topk.tensor,
                         selection.tensor));
  std::cout << "DSA_KV_OFFLOADING_STABLE_ROPE_METADATA_CONTRACT_OK (6 assertions)\n";
}
