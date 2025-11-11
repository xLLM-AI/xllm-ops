/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#ifndef BEAM_SEARCH_H
#define BEAM_SEARCH_H

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include "utils_tensor.h"
#include "aclnn_beam_search_group.h"

namespace beam_search {
struct TensorShapes {
  std::vector<std::vector<int64_t>> token_ids_shape;
  std::vector<std::vector<int64_t>> log_probs_shape;
  std::vector<std::vector<int64_t>> top_tokens_shape;
  std::vector<std::vector<int64_t>> top_probs_shape;
  std::vector<std::vector<int64_t>> output_shape;
  std::vector<std::vector<int64_t>> beam_prefix_shape;
};
class BeamSearchBase {
 public:
  using ProbGenerator = std::function<void(int64_t /*n_sequences*/,
                                           int64_t /*top_k*/,
                                           const torch::TensorOptions& /*opt_f32*/,
                                           torch::Tensor& /*log_probs_out*/,
                                           torch::Tensor& /*top_probs_out*/)>;

  BeamSearchBase(int64_t beam_width,
                 int64_t top_k,
                 int64_t request_num,
                 int64_t sequence_length)
      : beam_width_(beam_width),
        top_k_(top_k),
        request_num_(request_num),
        sequence_length_(sequence_length),
        n_sequences_(0) {
    n_sequences_ = request_num_ * beam_width_;
    shapes.token_ids_shape = {{n_sequences_, sequence_length_}};
    shapes.log_probs_shape = {{n_sequences_, 1}};
    shapes.top_tokens_shape = {{n_sequences_, top_k_}};
    shapes.top_probs_shape = {{n_sequences_, top_k_}};
    shapes.output_shape = {{n_sequences_, 1}};
    shapes.beam_prefix_shape = {{n_sequences_, 1}};
  }

  BeamSearchBase(int64_t beam_width,
                 int64_t top_k,
                 int64_t request_num,
                 int64_t sequence_length,
                 ProbGenerator prob_generator)
      : beam_width_(beam_width),
        top_k_(top_k),
        request_num_(request_num),
        sequence_length_(sequence_length),
        n_sequences_(0),
        prob_generator_(std::move(prob_generator)) {
    n_sequences_ = request_num_ * beam_width_;
    shapes.token_ids_shape = {{n_sequences_, sequence_length_}};
    shapes.log_probs_shape = {{n_sequences_, 1}};
    shapes.top_tokens_shape = {{n_sequences_, top_k_}};
    shapes.top_probs_shape = {{n_sequences_, top_k_}};
    shapes.output_shape = {{n_sequences_, 1}};
    shapes.beam_prefix_shape = {{n_sequences_, 1}};
  }

  void set_prob_generator(ProbGenerator gen) { prob_generator_ = std::move(gen); }

  void create_torch_tensors() {
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32);
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32);
    auto opt_f64 = torch::TensorOptions().dtype(torch::kFloat64);

    token_ids = torch::randint(
        /*low=*/4, /*high=*/10, {n_sequences_, sequence_length_}, opt_i32);
    if (prob_generator_) {
      prob_generator_(n_sequences_, top_k_, opt_f32, log_probs, top_probs);
    } else {
      // Default random generation if no generator is provided
      // Generate random log_probs and top_probs in range [-100000, 0]
      log_probs = torch::rand({n_sequences_, 1}, opt_f64) * 100000 - 100000;
      log_probs = log_probs.to(opt_f32);
      top_probs = torch::rand({n_sequences_, top_k_}, opt_f64) * 100000 - 100000;
      top_probs = top_probs.to(opt_f32);
    }
    top_tokens = torch::randint(/*low=*/4, /*high=*/10, {n_sequences_, top_k_}, opt_i32);

    output_token_ids_torch = torch::zeros({n_sequences_, 1}, opt_i32);
    output_token_ids_op = torch::zeros({n_sequences_, 1}, opt_i32);
    output_log_probs_op = torch::zeros({n_sequences_, 1}, opt_f32);
    output_log_probs_torch = torch::zeros({n_sequences_, 1}, opt_f32);
    output_token_index_op = torch::zeros({n_sequences_, 1}, opt_i32);
    output_token_index_torch = torch::zeros({n_sequences_, 1}, opt_i32);
    output_log_probs_torch = torch::zeros({n_sequences_, 1}, opt_f32);
    output_token_ids_op = torch::zeros({n_sequences_, 1}, opt_i32);
    output_token_index_op = torch::zeros({n_sequences_, 1}, opt_i32);
    output_log_probs_op = torch::zeros({n_sequences_, 1}, opt_f32);
    output_beam_count_prefix_sums_torch =
        torch::zeros({n_sequences_, 1}, opt_i32);
    output_beam_count_prefix_sums_op =
        torch::zeros({n_sequences_, 1}, opt_i32);
    output_sequence_torch = torch::zeros({n_sequences_, sequence_length_}, opt_i32);
    output_sequence_op = torch::zeros({n_sequences_, sequence_length_}, opt_i32);
  }

  int32_t beam_width_;
  int32_t top_k_;
  int32_t request_num_;
  int32_t sequence_length_;
  int32_t n_sequences_;
  int32_t current_step_;
  torch::Tensor token_ids;
  torch::Tensor log_probs;
  torch::Tensor top_tokens;
  torch::Tensor top_probs;
  torch::Tensor output_token_ids_torch;
  torch::Tensor output_token_index_torch;
  torch::Tensor output_log_probs_torch;
  torch::Tensor output_token_ids_op;
  torch::Tensor output_token_index_op;
  torch::Tensor output_log_probs_op;
  torch::Tensor output_beam_count_prefix_sums_torch;
  torch::Tensor output_beam_count_prefix_sums_op;
  torch::Tensor output_sequence_torch;
  torch::Tensor output_sequence_op;

  TensorShapes shapes;

 private:
  ProbGenerator prob_generator_{nullptr};
};

class BeamSearchTorch {
 public:
  BeamSearchTorch(BeamSearchBase& base) : base(base) {}
  void process() {
    torch::Tensor expanded_log_probs = base.log_probs.repeat({1, base.top_k_});
    torch::Tensor candidate_scores = (expanded_log_probs + base.top_probs);
    candidate_scores = candidate_scores.view(
        {base.request_num_, base.beam_width_ * base.top_k_});
    auto topk_result =
        at::topk(candidate_scores, base.beam_width_, 1, true, true);
    torch::Tensor topk_scores = std::get<0>(topk_result);
    torch::Tensor topk_indices = std::get<1>(topk_result);

    torch::Tensor selected_beam =
        at::floor_divide(topk_indices, base.top_k_).to(torch::kInt32);
    torch::Tensor selected_within_top =
        at::remainder(topk_indices, base.top_k_).to(torch::kInt32);
    auto device = base.token_ids.device();
    auto options_long =
        torch::TensorOptions().dtype(torch::kLong).device(device);
    torch::Tensor request_ids =
        torch::arange(base.request_num_, options_long).view({-1, 1});
    torch::Tensor base_indices = (request_ids * base.beam_width_);
    torch::Tensor orig_seq_indices =
        (base_indices + selected_beam.to(torch::kLong)).reshape({-1});
    torch::Tensor selected_top =
        base.top_tokens.index_select(0, orig_seq_indices);
    torch::Tensor selected_within_top_flat =
        selected_within_top.reshape({-1}).to(torch::kLong);
    torch::Tensor next_tokens =
        selected_top.gather(1, selected_within_top_flat.unsqueeze(1))
            .reshape({base.request_num_, base.beam_width_})
            .to(torch::kInt32);
    torch::Tensor beam_ids =
        selected_beam.reshape({base.request_num_, base.beam_width_});
    torch::Tensor within_top =
        selected_within_top.reshape({base.request_num_, base.beam_width_});
    torch::Tensor scores =
        topk_scores.to(torch::kFloat32).reshape({base.request_num_, base.beam_width_});

    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    std::vector<torch::Tensor> tokens_sorted_list;
    std::vector<torch::Tensor> scores_sorted_list;
    std::vector<torch::Tensor> index_sorted_list;
    std::vector<torch::Tensor> prefix_list;

    for (int32_t r = 0; r < base.request_num_; ++r) {
      auto beam_ids_r = beam_ids[r].clone();
      auto tokens_r = next_tokens[r].clone();
      auto scores_r = scores[r].clone();
      auto within_r = within_top[r].clone();

      torch::Tensor counts = torch::zeros({base.beam_width_}, opt_i32);
      auto beam_ids_ptr = beam_ids_r.data_ptr<int32_t>();
      auto counts_ptr = counts.data_ptr<int32_t>();
      for (int32_t j = 0; j < base.beam_width_; ++j) {
        int32_t beam = beam_ids_ptr[j];
        if (beam >= 0 && beam < base.beam_width_) {
          counts_ptr[beam] += 1;
        }
      }

      torch::Tensor prefix = torch::cumsum(counts, 0);
      torch::Tensor start = prefix - counts;
      torch::Tensor cursor = start.clone().to(opt_i32);
      auto cursor_ptr = cursor.data_ptr<int32_t>();

      torch::Tensor token_sorted = torch::zeros({base.beam_width_}, opt_i32);
      torch::Tensor score_sorted =
          torch::full({base.beam_width_},
                      -std::numeric_limits<float>::infinity(), opt_f32);
      torch::Tensor index_sorted =
          torch::full({base.beam_width_}, -1, opt_i32);
      auto token_out_ptr = token_sorted.data_ptr<int32_t>();
      auto score_out_ptr = score_sorted.data_ptr<float>();
      auto index_out_ptr = index_sorted.data_ptr<int32_t>();
      auto token_in_ptr = tokens_r.data_ptr<int32_t>();
      auto scores_r_ptr = scores_r.data_ptr<float>();

      for (int32_t j = 0; j < base.beam_width_; ++j) {
        int32_t beam = beam_ids_ptr[j];
        if (beam < 0 || beam >= base.beam_width_) {
          continue;
        }
        int32_t pos = cursor_ptr[beam];
        if (pos >= base.beam_width_) {
          continue;
        }
        cursor_ptr[beam] += 1;
        token_out_ptr[pos] = token_in_ptr[j];
        score_out_ptr[pos] = scores_r_ptr[j];
        index_out_ptr[pos] = r * base.beam_width_ + beam;
      }

      tokens_sorted_list.push_back(token_sorted);
      scores_sorted_list.push_back(score_sorted);
      index_sorted_list.push_back(index_sorted);
      // global prefix sum
      prefix = prefix + r * base.beam_width_;
      prefix_list.push_back(prefix);
    }

    base.output_token_ids_torch =
        torch::stack(tokens_sorted_list).reshape({-1, 1});
    base.output_log_probs_torch =
        torch::stack(scores_sorted_list).reshape({-1, 1});
    base.output_token_index_torch =
        torch::stack(index_sorted_list).reshape({-1, 1});
    base.output_beam_count_prefix_sums_torch =
        torch::stack(prefix_list).reshape({-1, 1});
    // reorder sequences and append next tokens along sequence_length dimension
    torch::Tensor index_flat =
        base.output_token_index_torch.view({-1}).to(torch::kLong);
    torch::Tensor reordered =
        base.token_ids.index_select(/*dim=*/0, index_flat);
    // write next token into the last position of sequence_length dimension
    // cout << "token ids torch " << base.output_token_ids_torch;
    // cout << "reordered " << reordered;
    reordered.select(/*dim=*/1, 1)
        .copy_(base.output_token_ids_torch.view({-1}));
    // cout << "reordered after" << reordered;
   base.output_sequence_torch = reordered;

  }

 private:
  BeamSearchBase& base;
};

class BeamSearchOp {
 public:
  BeamSearchOp(BeamSearchBase& base) : base(base) {}
  void process(aclrtStream stream) {
    this->create_tensors();
    CHECK_ACL_SUCCESS(execute_beam_search_operator(token_ids,
                                                   log_probs,
                                                   top_tokens,
                                                   top_probs,
                                                   output_token_ids,
                                                   output_token_index,
                                                   output_log_probs,
                                                   output_beam_count_prefix_sums,
                                                   stream),
                      "execute_beam_search_operator failed");
  }
  int execute_beam_search_operator(aclTensor* token_ids,
                                   aclTensor* log_probs,
                                   aclTensor* top_tokens,
                                   aclTensor* top_probs,
                                   aclTensor* output_token_ids,
                                   aclTensor* output_token_index,
                                   aclTensor* output_log_probs,
                                   aclTensor* output_beam_count_prefix_sums,
                                   aclrtStream stream);
  void destroy_tensors() {
    aclDestroyTensor(token_ids);
    aclDestroyTensor(log_probs);
    aclDestroyTensor(top_tokens);
    aclDestroyTensor(top_probs);
    aclDestroyTensor(output_token_ids);
    aclDestroyTensor(output_token_index);
    aclDestroyTensor(output_log_probs);
    aclDestroyTensor(output_beam_count_prefix_sums);
  }

 private:
  void create_tensors() {
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.token_ids_shape[0], base.token_ids, &token_ids),
        "create input_a Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.log_probs_shape[0], base.log_probs, &log_probs),
        "create input_b Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.top_tokens_shape[0], base.top_tokens, &top_tokens),
        "create top_tokens Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.top_probs_shape[0], base.top_probs, &top_probs),
        "create top_probs Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(base.shapes.output_shape[0],
                                        base.output_token_ids_op,
                                        &output_token_ids),
        "create output_token_ids_op Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(base.shapes.output_shape[0],
                                        base.output_token_index_op,
                                        &output_token_index),
        "create output_token_index_op Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(base.shapes.output_shape[0],
                                        base.output_log_probs_op,
                                        &output_log_probs),
        "create output_log_probs_op Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.beam_prefix_shape[0],
            base.output_beam_count_prefix_sums_op,
            &output_beam_count_prefix_sums),
        "create output_beam_count_prefix_sums Tensor failed");
  }
  aclTensor* token_ids = nullptr;
  aclTensor* log_probs = nullptr;
  aclTensor* top_tokens = nullptr;
  aclTensor* top_probs = nullptr;
  aclTensor* output_token_ids = nullptr;
  aclTensor* output_token_index = nullptr;
  aclTensor* output_log_probs = nullptr;
  aclTensor* output_beam_count_prefix_sums = nullptr;
  BeamSearchBase& base;
};
int BeamSearchOp::execute_beam_search_operator(aclTensor* token_ids,
                                               aclTensor* log_probs,
                                               aclTensor* top_tokens,
                                               aclTensor* top_probs,
                                               aclTensor* output_token_ids,
                                               aclTensor* output_token_index,
                                               aclTensor* output_log_probs,
                                               aclTensor* output_beam_count_prefix_sums,
                                               aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  auto ret = aclnnBeamSearchGroupGetWorkspaceSize(
      log_probs,
      top_tokens,
      top_probs,
      token_ids,
      2,
      output_token_ids,
      output_token_index,
      output_log_probs,
      output_beam_count_prefix_sums,
      token_ids,
      &workspaceSize,
      &executor);
  CHECK_ACL_SUCCESS(ret, "aclnn_beam_search_operator get workspace size failed");
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL_SUCCESS(ret, "aclnn_beam_search_operator malloc workspace failed");
  }
  ret = aclnnBeamSearchGroup(workspaceAddr,
                             workspaceSize,
                             executor,
                             stream);

  CHECK_ACL_SUCCESS(ret, "aclnn_beam_search_operator execute failed");
  ret = aclrtSynchronizeStream(stream);
  CHECK_ACL_SUCCESS(ret, "aclnn_beam_search_operator synchronize stream failed");

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return ACL_SUCCESS;
}

}  // namespace beam_search

#endif