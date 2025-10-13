#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "beam_search.h"
#include "base/utils_tensor.h"

class BeamSearchTest : public ::testing::Test {
protected:
    int32_t deviceId = 9;
    aclrtStream stream1;
    
    void SetUp() override {
        utils::initialize_acl(deviceId, &stream1);
    }

    void TearDown() override {
        aclrtDestroyStream(stream1);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
};

TEST_F(BeamSearchTest, BasicMatmulCorrectness) {
    beam_search::BeamSearchBase inputs(2, 2, 2, 2);
    inputs.create_torch_tensors();
    beam_search::BeamSearchTorch op_torch(inputs);
    op_torch.process();
    beam_search::BeamSearchOp op_cann(inputs);
    op_cann.process(stream1);
    // Compare on CPU to avoid device mismatch
    auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
    auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
    auto out_torch_index_cpu = inputs.output_token_index_torch.to(torch::kCPU);
    auto out_op_index_cpu = inputs.output_token_index_op.to(torch::kCPU);
    auto out_torch_log_cpu = inputs.output_log_probs_torch.to(torch::kCPU).view({-1, 1});
    auto out_op_log_cpu = inputs.output_log_probs_op.to(torch::kCPU).view({-1, 1});
    EXPECT_TRUE(torch::equal(out_torch_cpu, out_torch_cpu))
        << "Custom op output does not match native output";
    op_cann.destroy_tensors();
}
TEST_F(BeamSearchTest, DifferentSizes) {
    std::vector<std::tuple<int, int, int, int>> test_sizes = {
        {2, 2, 2, 2},
        {4, 4, 2, 2},
        {8, 8, 2, 2},
        {16, 16, 2, 2},
        {32, 32, 2, 2},
        {64, 64, 2, 2},
        {128, 128, 2, 2},
        {256, 256, 2, 2},
        {512, 512, 2, 2}
    };
    for (auto [beam_width, top_k, request_num, sequence_length] : test_sizes) {
        beam_search::BeamSearchBase inputs(beam_width, top_k, request_num, sequence_length);
        inputs.create_torch_tensors();
        beam_search::BeamSearchTorch op_torch(inputs);
        op_torch.process();
        beam_search::BeamSearchOp op_cann(inputs);
        op_cann.process(stream1);
        auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
        auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
        auto out_torch_index_cpu = inputs.output_token_index_torch.to(torch::kCPU);
        auto out_op_index_cpu = inputs.output_token_index_op.to(torch::kCPU);
        auto out_torch_log_cpu = inputs.output_log_probs_torch.to(torch::kCPU).view({-1, 1});
        auto out_op_log_cpu = inputs.output_log_probs_op.to(torch::kCPU).view({-1, 1});
        EXPECT_TRUE(torch::equal(out_torch_log_cpu, out_op_log_cpu))
            << "Failed for size: beam_width=" << beam_width << ", top_k=" << top_k << ", request_num=" << request_num << ", sequence_length=" << sequence_length;
        op_cann.destroy_tensors();
    }
}