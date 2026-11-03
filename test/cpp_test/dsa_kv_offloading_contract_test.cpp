/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
*/
// Actual CANN tensor metadata and production validation helpers; no NPU use.
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "xllm_ops/dsa_kv_offloading_plan_nope/op_host/op_api/dsa_kv_offloading_stable_validate.h"

namespace {
class Metadata {
 public:
  Metadata(std::initializer_list<int64_t> shape,
           aclDataType type = ACL_INT32,
           uintptr_t pointer = 0x10000000,
           bool contiguous = true) {
    std::vector<int64_t> dims(shape), strides(dims.size());
    int64_t stride = contiguous ? 1 : 2;
    for (size_t i = dims.size(); i-- > 0;) {
      strides[i] = stride;
      stride *= dims[i];
    }
    tensor = aclCreateTensor(dims.data(),
                             dims.size(),
                             type,
                             strides.data(),
                             0,
                             ACL_FORMAT_ND,
                             dims.data(),
                             dims.size(),
                             reinterpret_cast<void*>(pointer));
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
  Metadata tokens({4, 1, 2176});
  assert(dsa_kv_offloading::Tensor(tokens.tensor, op::DataType::DT_INT32));
  assert(!dsa_kv_offloading::Tensor(nullptr, op::DataType::DT_INT32));
  assert(!dsa_kv_offloading::Tensor(tokens.tensor, op::DataType::DT_BF16));
  assert(dsa_kv_offloading::Shape(tokens.tensor, {4, 1, 2176}));
  assert(!dsa_kv_offloading::Shape(tokens.tensor, {4, 2176}));
  Metadata strided({4, 1, 2176}, ACL_INT32, 0x20000000, false);
  assert(!dsa_kv_offloading::Tensor(strided.tensor, op::DataType::DT_INT32));
  Metadata noData({4}, ACL_INT32, 0);
  assert(!dsa_kv_offloading::Tensor(noData.tensor, op::DataType::DT_INT32));
  Metadata adjacent({1}, ACL_INT32, 0x10000000 + 4 * 2176 * 4);
  Metadata overlapping({1}, ACL_INT32, 0x10000000 + 4);
  assert(dsa_kv_offloading::Distinct({tokens.tensor, adjacent.tensor}));
  assert(!dsa_kv_offloading::Distinct({tokens.tensor, tokens.tensor}));
  assert(!dsa_kv_offloading::Distinct({tokens.tensor, overlapping.tensor}));
  assert(!dsa_kv_offloading::Distinct({overlapping.tensor, tokens.tensor}));
  Metadata overflow({2}, ACL_INT32, std::numeric_limits<uintptr_t>::max() - 3);
  assert(!dsa_kv_offloading::Distinct({overflow.tensor}));
  Metadata host({8, 128, 1, 512}, ACL_BF16);
  Metadata blockTable({2, 4}), descriptor({4, 4}), map({4, 4});
  Metadata append({4}), current({4, 1, 512}, ACL_BF16);
  Metadata selection({68, 128, 1, 512}, ACL_BF16);
  assert(dsa_kv_offloading::StableCommon(host.tensor,
                                         blockTable.tensor,
                                         descriptor.tensor,
                                         map.tensor,
                                         tokens.tensor,
                                         append.tensor,
                                         current.tensor,
                                         tokens.tensor,
                                         selection.tensor));
  Metadata badMap({4, 5});
  assert(!dsa_kv_offloading::StableCommon(host.tensor,
                                          blockTable.tensor,
                                          descriptor.tensor,
                                          badMap.tensor,
                                          tokens.tensor,
                                          append.tensor,
                                          current.tensor,
                                          tokens.tensor,
                                          selection.tensor));
  Metadata badLatent({8, 128, 1, 256}, ACL_BF16);
  assert(!dsa_kv_offloading::StableCommon(badLatent.tensor,
                                          blockTable.tensor,
                                          descriptor.tensor,
                                          map.tensor,
                                          tokens.tensor,
                                          append.tensor,
                                          current.tensor,
                                          tokens.tensor,
                                          selection.tensor));
  Metadata resident({4, 1, 2176}), slots({4, 1, 2176}), count({4});
  Metadata indices({4, 1, 2176}), lengths({4}), selectionTable({4, 17});
  assert(dsa_kv_offloading::StableState(tokens.tensor,
                                        selection.tensor,
                                        resident.tensor,
                                        slots.tensor,
                                        count.tensor,
                                        indices.tensor,
                                        lengths.tensor,
                                        selectionTable.tensor));
  Metadata shortCount({3});
  assert(!dsa_kv_offloading::StableState(tokens.tensor,
                                         selection.tensor,
                                         resident.tensor,
                                         slots.tensor,
                                         shortCount.tensor,
                                         indices.tensor,
                                         lengths.tensor,
                                         selectionTable.tensor));
  Metadata hr({8, 128, 1, 64}, ACL_BF16);
  Metadata cr({4, 1, 64}, ACL_BF16), sr({68, 128, 1, 64}, ACL_BF16);
  assert(dsa_kv_offloading::StableRope(hr.tensor,
                                       cr.tensor,
                                       sr.tensor,
                                       host.tensor,
                                       tokens.tensor,
                                       selection.tensor));
  Metadata wrongRope({8, 128, 1, 128}, ACL_BF16);
  assert(!dsa_kv_offloading::StableRope(wrongRope.tensor,
                                        cr.tensor,
                                        sr.tensor,
                                        host.tensor,
                                        tokens.tensor,
                                        selection.tensor));
  std::cout
      << "DSA_KV_OFFLOADING_METADATA_CONTRACT_OK (19 contract assertions)\n";
}
