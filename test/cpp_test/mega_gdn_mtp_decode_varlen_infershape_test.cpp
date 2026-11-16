/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {

constexpr const char* kOpType = "MegaGdnMtpDecodeVarlen";
constexpr std::array<size_t, 4> kOutputInputs = {0, 5, 8, 1};
constexpr std::array<ge::DataType, 13> kInputTypes = {ge::DT_BF16,
                                                      ge::DT_BF16,
                                                      ge::DT_BF16,
                                                      ge::DT_BF16,
                                                      ge::DT_BF16,
                                                      ge::DT_BF16,
                                                      ge::DT_FLOAT,
                                                      ge::DT_FLOAT,
                                                      ge::DT_FLOAT,
                                                      ge::DT_INT32,
                                                      ge::DT_INT32,
                                                      ge::DT_BF16,
                                                      ge::DT_INT32};

class MegaGdnMtpDecodeVarlenInferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* override_library = std::getenv("XLLM_OPS_PROTO_LIBRARY");
    const std::string library =
        override_library != nullptr ? override_library : XLLM_OPS_PROTO_LIBRARY;
    ASSERT_TRUE(std::filesystem::is_regular_file(library)) << library;
    const gert::OppSoDesc package({ge::AscendString(library.c_str())},
                                  ge::AscendString("varlen_infer_test"));
    ASSERT_EQ(registry_.AddSoToRegistry(package), ge::GRAPH_SUCCESS);
    functions_ = registry_.GetOpImpl(kOpType);
    ASSERT_NE(functions_, nullptr);
    ASSERT_NE(functions_->infer_shape, nullptr);
    ASSERT_NE(functions_->infer_datatype, nullptr);

    const std::array<gert::Shape, 13> shapes = {gert::Shape({8, 512}),
                                                gert::Shape({8, 2, 128}),
                                                gert::Shape({8, 2}),
                                                gert::Shape({8, 2}),
                                                gert::Shape({4, 512}),
                                                gert::Shape({4, 10, 512}),
                                                gert::Shape({2}),
                                                gert::Shape({2}),
                                                gert::Shape({32, 2, 128, 128}),
                                                gert::Shape({1}),
                                                gert::Shape({1}),
                                                gert::Shape({128}),
                                                gert::Shape({2})};
    for (size_t index = 0; index < inputs_.size(); ++index) {
      inputs_[index].SetDataType(kInputTypes[index]);
      inputs_[index].SetOriginFormat(ge::FORMAT_ND);
      inputs_[index].SetStorageFormat(ge::FORMAT_ND);
      set_shape(index, shapes[index]);
    }
  }

  void set_shape(size_t index, const gert::Shape& shape) {
    inputs_[index].MutableOriginShape() = shape;
    inputs_[index].MutableStorageShape() = shape;
  }

  gert::ContextHolder<gert::InferShapeContext> shape_context(
      size_t output_count = 4) {
    std::vector<gert::Tensor*> inputs;
    inputs.reserve(inputs_.size());
    for (gert::Tensor& input : inputs_) {
      inputs.emplace_back(&input);
    }
    gert::OpInferShapeContextBuilder builder;
    builder.OpType(ge::AscendString(kOpType))
        .OpName(ge::AscendString("varlen_infer_test"))
        .IONum(inputs.size(), output_count)
        .InputTensors(inputs);
    for (size_t index = 0; index < output_count; ++index) {
      builder.OutputTensorDesc(index,
                               kInputTypes[kOutputInputs[index]],
                               ge::FORMAT_ND,
                               ge::FORMAT_ND);
    }
    return builder.Build();
  }

  void expect_shapes(gert::InferShapeContext* context) const {
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(functions_->infer_shape(context), ge::GRAPH_SUCCESS);
    for (size_t index = 0; index < kOutputInputs.size(); ++index) {
      ASSERT_NE(context->GetOutputShape(index), nullptr);
      EXPECT_EQ(*context->GetOutputShape(index),
                inputs_[kOutputInputs[index]].GetOriginShape());
    }
  }

  gert::OpImplSpaceRegistryV2 registry_;
  const gert::OpImplKernelRegistry::OpImplFunctionsV2* functions_ = nullptr;
  std::array<gert::Tensor, 13> inputs_;
};

TEST_F(MegaGdnMtpDecodeVarlenInferTest, RegistersWithoutHostTensorData) {
  auto holder = shape_context();
  expect_shapes(holder.GetContext());
  for (const gert::Tensor& input : inputs_) {
    EXPECT_EQ(input.GetAddr(), nullptr);
  }
}

TEST_F(MegaGdnMtpDecodeVarlenInferTest, PreservesDynamicPackedAndStateShapes) {
  set_shape(/*index=*/0, gert::Shape({-1, 512}));
  set_shape(/*index=*/1, gert::Shape({-1, 2, 128}));
  set_shape(/*index=*/5, gert::Shape({-1, 10, 512}));
  set_shape(/*index=*/8, gert::Shape({-1, 2, 128, 128}));
  auto holder = shape_context();
  expect_shapes(holder.GetContext());
}

TEST_F(MegaGdnMtpDecodeVarlenInferTest, PreservesUnknownRank) {
  set_shape(/*index=*/0, gert::Shape({-2}));
  auto holder = shape_context();
  expect_shapes(holder.GetContext());
}

TEST_F(MegaGdnMtpDecodeVarlenInferTest, RejectsMissingOutputShape) {
  auto holder = shape_context(/*output_count=*/3);
  gert::InferShapeContext* context = holder.GetContext();
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(functions_->infer_shape(context), ge::GRAPH_FAILED);
}

TEST_F(MegaGdnMtpDecodeVarlenInferTest, InfersMixedOutputDataTypes) {
  gert::OpInferDataTypeContextBuilder builder;
  builder.OpType(ge::AscendString(kOpType))
      .OpName(ge::AscendString("varlen_dtype_test"))
      .IONum(kInputTypes.size(), kOutputInputs.size());
  for (size_t index = 0; index < kInputTypes.size(); ++index) {
    builder.InputTensorDesc(
        index, kInputTypes[index], ge::FORMAT_ND, ge::FORMAT_ND);
  }
  for (size_t index = 0; index < kOutputInputs.size(); ++index) {
    builder.OutputTensorDesc(index, ge::FORMAT_ND, ge::FORMAT_ND);
  }
  auto holder = builder.Build();
  gert::InferDataTypeContext* context = holder.GetContext();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(functions_->infer_datatype(context), ge::GRAPH_SUCCESS);
  for (size_t index = 0; index < kOutputInputs.size(); ++index) {
    EXPECT_EQ(context->GetOutputDataType(index),
              kInputTypes[kOutputInputs[index]]);
  }
}

}  // namespace
