/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Adapted from vllm-ascend csrc/torch_binding.cpp (npu_hc_pre_v3).
==============================================================================*/
#pragma once

namespace xllm_ops_hc_pre_va {
std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_output_tensor(
    const at::Tensor& x,
    int64_t hc_mult) {
  auto xDims = x.dim();
  at::SmallVector<int64_t, 8> y_size;
  at::SmallVector<int64_t, 8> post_size;
  at::SmallVector<int64_t, 8> comb_frag_size;
  if (xDims == 4) {
    auto batch = x.size(0);
    auto size = x.size(1);
    auto d = x.size(3);
    y_size = {batch, size, d};
    post_size = {batch, size, hc_mult};
    comb_frag_size = {batch, size, hc_mult, hc_mult};
  } else if (xDims == 3) {
    auto bs = x.size(0);
    auto d = x.size(2);
    y_size = {bs, d};
    post_size = {bs, hc_mult};
    comb_frag_size = {bs, hc_mult, hc_mult};
  }

  at::Tensor y = at::empty(y_size, x.options().dtype(at::kBFloat16));
  at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
  at::Tensor comb_frag =
      at::empty(comb_frag_size, x.options().dtype(at::kFloat));

  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

at::Tensor construct_hc_pre_pre_output_tensor(const at::Tensor& x,
                                              int64_t hc_mult) {
  at::SmallVector<int64_t, 8> pre_size;
  if (x.dim() == 4) {
    pre_size = {x.size(0), x.size(1), hc_mult};
  } else if (x.dim() == 3) {
    pre_size = {x.size(0), hc_mult};
  }
  return at::empty(pre_size, x.options().dtype(at::kFloat));
}

void check_hc_pre_shape_and_dtype(const at::Tensor& x,
                                  const at::Tensor& hc_fn,
                                  const at::Tensor& hc_scale,
                                  const at::Tensor& hc_base,
                                  const c10::optional<at::Tensor>& pre_mix,
                                  int64_t hc_mult) {
  constexpr int64_t HC_PRE_HC_LIMIT = 4;
  constexpr int64_t HC_PRE_MIX_HC_LIMIT = 24;
  constexpr int64_t HC_SCALE_SIZE = 3;
  auto x_dims = x.dim();
  TORCH_CHECK(x_dims == 3 || x_dims == 4,
              "Input tensor x's dim num should be 3 or 4, actual ",
              x_dims,
              ".");
  for (auto i = 0; i < x_dims; i++) {
    TORCH_CHECK(x.size(i) > 0,
                "Input tensor x's shape should be positive, but x.shape[",
                i,
                "] is ",
                x.size(i),
                ".");
  }

  auto hc = x_dims == 4 ? x.size(2) : x.size(1);
  auto d = x_dims == 4 ? x.size(3) : x.size(2);
  TORCH_CHECK(hc_mult == HC_PRE_HC_LIMIT,
              "hc_mult only supports ",
              HC_PRE_HC_LIMIT,
              ", actual ",
              hc_mult,
              ".");
  TORCH_CHECK(hc == HC_PRE_HC_LIMIT,
              "The hc of x only supports ",
              HC_PRE_HC_LIMIT,
              ", actual ",
              hc,
              ".");
  TORCH_CHECK(hc_fn.dim() == 2,
              "Input tensor hc_fn's dim num should be 2, actual ",
              hc_fn.dim(),
              ".");
  TORCH_CHECK(hc_fn.size(0) == HC_PRE_MIX_HC_LIMIT,
              "The hc_fn.shape[0] only supports ",
              HC_PRE_MIX_HC_LIMIT,
              ", actual ",
              hc_fn.size(0),
              ".");
  TORCH_CHECK(hc_fn.size(1) == hc * d,
              "The hc_fn.shape[1] should be hc * d, actual hc_fn.shape[1] is ",
              hc_fn.size(1),
              ", hc is ",
              hc,
              ", d is ",
              d,
              ".");
  TORCH_CHECK(hc_scale.dim() == 1,
              "Input tensor hc_scale's dim num should be 1, actual ",
              hc_scale.dim(),
              ".");
  TORCH_CHECK(hc_scale.size(0) == HC_SCALE_SIZE,
              "Input tensor hc_scale's shape should be [",
              HC_SCALE_SIZE,
              "], actual [",
              hc_scale.size(0),
              "].");
  TORCH_CHECK(hc_base.dim() == 1,
              "Input tensor hc_base's dim num should be 1, actual ",
              hc_base.dim(),
              ".");
  TORCH_CHECK(hc_base.size(0) == HC_PRE_MIX_HC_LIMIT,
              "The hc_base.shape[0] only supports ",
              HC_PRE_MIX_HC_LIMIT,
              ", actual ",
              hc_base.size(0),
              ".");

  TORCH_CHECK(x.dtype() == at::kBFloat16, "x's dtype should be BFLOAT16.");
  TORCH_CHECK(hc_fn.dtype() == at::kFloat, "hc_fn's dtype should be FLOAT32.");
  TORCH_CHECK(hc_scale.dtype() == at::kFloat,
              "hc_scale's dtype should be FLOAT32.");
  TORCH_CHECK(hc_base.dtype() == at::kFloat,
              "hc_base's dtype should be FLOAT32.");
  if (pre_mix.has_value() && pre_mix->defined()) {
    TORCH_CHECK(pre_mix->dtype() == at::kFloat,
                "pre_mix's dtype should be FLOAT32.");
    TORCH_CHECK(pre_mix->dim() == x_dims - 1,
                "pre_mix's dim num should be ",
                x_dims - 1,
                ", actual ",
                pre_mix->dim(),
                ".");
    for (auto i = 0; i < x_dims - 1; i++) {
      TORCH_CHECK(pre_mix->size(i) == x.size(i),
                  "pre_mix.shape[",
                  i,
                  "] should equal x.shape[",
                  i,
                  "], actual ",
                  pre_mix->size(i),
                  " vs ",
                  x.size(i),
                  ".");
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> run_hc_pre_fusion(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const c10::optional<at::Tensor>& pre_mix,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double norm_eps,
    double hc_eps) {
  auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
  at::Tensor y = std::get<0>(output_tensors);
  at::Tensor post = std::get<1>(output_tensors);
  at::Tensor comb_frag = std::get<2>(output_tensors);
  at::Tensor pre = construct_hc_pre_pre_output_tensor(x, hc_mult);
  EXEC_NPU_CMD(aclnnHcPreVa,
               x,
               hc_fn,
               hc_scale,
               hc_base,
               pre_mix,
               hc_mult,
               hc_sinkhorn_iters,
               hc_eps,
               norm_eps,
               y,
               post,
               comb_frag,
               pre);

  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      y, post, comb_frag, pre);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hc_pre_va_impl_npu(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const c10::optional<at::Tensor>& pre_mix,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double norm_eps,
    double hc_eps) {
  check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult);
  return run_hc_pre_fusion(x,
                           hc_fn,
                           hc_scale,
                           hc_base,
                           pre_mix,
                           hc_mult,
                           hc_sinkhorn_iters,
                           norm_eps,
                           hc_eps);
}

}  // namespace xllm_ops_hc_pre_va
