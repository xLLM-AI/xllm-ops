#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>

#include "multi_latent_attention.h"
#include "base/utils_tensor.h"

class MultiLatentAttentionTest : public ::testing::Test {
protected:
    int32_t deviceId = 15;
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

TEST_F(MultiLatentAttentionTest, BasicAttentionCorrectness) {
    multi_latent_attention::MultiLatentAttentionBase inputs(200, 32, 64, 128);
    inputs.create_torch_tensors();
    multi_latent_attention::MultiLatentAttentionOp op_attention(inputs);
    op_attention.process(stream1);
    multi_latent_attention::MultiLatentAttentionNative native_attention(inputs);
    native_attention.process(stream1);
    float rtol = 1e-5;
    float atol = 1e-3;
    EXPECT_TRUE(torch::allclose(inputs.output_native, inputs.output_op, rtol, atol))
        << "Custom op output does not match native output";
    native_attention.destroyTensors();
    op_attention.destroyTensors();
}

TEST_F(MultiLatentAttentionTest, DifferentSizes) {
    std::vector<std::tuple<int, int, int, int>> test_sizes = {
        {1, 128, 1024, 128},
        {6, 128, 2048, 128},
        {12, 128, 2048, 128},
        {24, 128, 4096, 128},
        {25, 128, 4096, 128},
        {1, 32, 1024, 128},
        {6, 32, 2048, 128},
        {12, 32, 2048, 128},
        {24, 32, 4096, 128},
        {25, 32, 4096, 128},
        {1, 64, 1024, 128},
        {6, 64, 2048, 128},
        {12, 64, 2048, 128},
        {24, 64, 4096, 128},
        {25, 64, 4096, 128}
    };
    
    for (auto [bs, headNum, seqLen, blockSize] : test_sizes) {
        multi_latent_attention::MultiLatentAttentionBase inputs(bs, headNum, seqLen, blockSize);
        inputs.create_torch_tensors();
        
        multi_latent_attention::MultiLatentAttentionOp op_attention(inputs);
        op_attention.process(stream1);
        multi_latent_attention::MultiLatentAttentionNative native_attention(inputs);
        native_attention.process(stream1);
        float rtol = 1e-5;
        float atol = 1e-3;
        EXPECT_TRUE(torch::allclose(inputs.output_native, inputs.output_op, rtol, atol))
            << "Failed for size: batch size=" << bs << ", headNum=" << headNum << ", seqLen=" << seqLen << ", blockSize=" << blockSize;
        native_attention.destroyTensors();
        op_attention.destroyTensors();
    }
}