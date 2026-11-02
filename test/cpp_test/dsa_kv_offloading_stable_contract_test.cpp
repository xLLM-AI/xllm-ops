/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
*/
// Real CANN tensor metadata; no NPU allocations or launch.
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
    for (size_t index = dims.size(); index-- > 0;) {
      strides[index] = stride;
      stride *= dims[index];
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
  Metadata block_table({8, 4}), descriptor({8, 4}), map({8, 8});
  Metadata topk({8, 1, 2176}), append({8}), current({8, 1, 512}, ACL_BF16);
  Metadata selection({64 * 17, 128, 1, 512}, ACL_BF16);
  Metadata resident({64, 1, 2176}), slots({64, 1, 2176}), count({64});
  Metadata indices({8, 1, 2176}), lengths({8}), table({8, 17});
  assert(dsa_kv_offloading::StableCommon(host.tensor,
                          block_table.tensor,
                          descriptor.tensor,
                          map.tensor,
                          topk.tensor,
                          append.tensor,
                          current.tensor,
                          indices.tensor,
                          selection.tensor));
  assert(dsa_kv_offloading::StableState(topk.tensor,
                         selection.tensor,
                         resident.tensor,
                         slots.tensor,
                         count.tensor,
                         indices.tensor,
                         lengths.tensor,
                         table.tensor));
  // Capacity is independent of the eight active query/output rows.
  assert(dsa_kv_offloading::StableSelectionRows(selection.tensor) == 64);
  Metadata short_selection({4 * 17, 128, 1, 512}, ACL_BF16);
  assert(!dsa_kv_offloading::StableCommon(host.tensor,
                           block_table.tensor,
                           descriptor.tensor,
                           map.tensor,
                           topk.tensor,
                           append.tensor,
                           current.tensor,
                           indices.tensor,
                           short_selection.tensor));
  Metadata short_count({8});
  assert(!dsa_kv_offloading::StableState(topk.tensor,
                          selection.tensor,
                          resident.tensor,
                          slots.tensor,
                          short_count.tensor,
                          indices.tensor,
                          lengths.tensor,
                          table.tensor));
  Metadata different_selection({65 * 17, 128, 1, 512}, ACL_BF16);
  assert(!dsa_kv_offloading::StableState(topk.tensor,
                          different_selection.tensor,
                          resident.tensor,
                          slots.tensor,
                          count.tensor,
                          indices.tensor,
                          lengths.tensor,
                          table.tensor));
  Metadata bad_output({64, 1, 2176});
  assert(!dsa_kv_offloading::StableState(topk.tensor,
                          selection.tensor,
                          resident.tensor,
                          slots.tensor,
                          count.tensor,
                          bad_output.tensor,
                          lengths.tensor,
                          table.tensor));
  Metadata partial_page({64 * 17 + 1, 128, 1, 512}, ACL_BF16);
  assert(!dsa_kv_offloading::StableCommon(host.tensor,
                           block_table.tensor,
                           descriptor.tensor,
                           map.tensor,
                           topk.tensor,
                           append.tensor,
                           current.tensor,
                           indices.tensor,
                           partial_page.tensor));
  Metadata bad_map({8, 9});
  assert(!dsa_kv_offloading::StableCommon(host.tensor,
                           block_table.tensor,
                           descriptor.tensor,
                           bad_map.tensor,
                           topk.tensor,
                           append.tensor,
                           current.tensor,
                           indices.tensor,
                           selection.tensor));
  std::cout << "DSA_KV_OFFLOADING_STABLE_METADATA_CONTRACT_OK (9 contract assertions)\n";
}
