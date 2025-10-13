#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "beam_search.h"
#include "base/utils_tensor.h"

// class BeamSearchTest : public ::testing::Test {
// protected:
//     int32_t deviceId = 15;
//     aclrtStream stream1;
    
//     void SetUp() override {
//         utils::initialize_acl(deviceId, &stream1);
//     }

//     void TearDown() override {
//         aclrtDestroyStream(stream1);
//         aclrtResetDevice(device#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "beam_search.h"
#include "base/utils_tensor.h"

// class BeamSearchTest : public ::testing::Test {
// protected:
//     int32_t deviceId = 15;
//     aclrtStream stream1;
    
//     void SetUp() override {
//         utils::initialize_acl(deviceId, &stream1);
//     }

//     void TearDown() override {
//         aclrtDestroyStream(stream1);
//         aclrtResetDevice(deviceId);
//         aclFinalize();
//     }
// };

// TEST_F(BeamSearchTest, BasicMatmulCorrectness) {
//     beam_search::BeamSearchBase inputs(2, 2, 2, 2);
//     inputs.create_torch_tensors();
//     // beam_search::BeamSearchTorch op_torch(inputs);
//     // op_torch.process();
//     // print torch values
//     printf("torch value:\n");
//     {
//         std::ostringstream oss; oss << inputs.token_ids; printf("token_ids:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.log_probs; printf("log_probs:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.top_tokens; printf("top_tokens:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.top_probs; printf("top_probs:\n%s\n", oss.str().c_str());
//     }
//     // {
//     //     std::ostringstream oss; oss << inputs.output_token_ids_torch; printf("output_token_ids_torch:\n%s\n", oss.str().c_str());
//     // }
//     beam_search::BeamSearchOp op_cann(inputs);
//     op_cann.process(stream1);
//     // Ensure NPU ops are finished before reading back
//     aclrtSynchronizeStream(stream1);
//     // Compare on CPU to avoid device mismatch
//     // auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
//     // auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
//     // std::cout << "out_op_cpu:\n" << out_op_cpu << std::endl;

//     // std::cout << "less than 1e-5: " << (out_torch_cpu - out_op_cpu).abs().max().item<float>() << std::endl;
//     // EXPECT_TRUE(torch::equal(out_torch_cpu, out_op_cpu))
//     //     << "Custom op output does not match native output";
//     op_cann.destroy_tensors();
// }

// int main() {
//     int deviceId = 15;
//     aclrtStream stream1;
//     utils::initialize_acl(deviceId, &stream1);
//     beam_search::BeamSearchBase inputs(512, 512, 2, 2, deviceId);
//     inputs.create_torch_tensors();
//     beam_search::BeamSearchTorch op_torch(inputs);
//     op_torch.process();
//     cout<<"op_torch done"<<endl;
    
//     beam_search::BeamSearchOp op_cann(inputs);
//     op_cann.process(stream1);
//     aclrtSynchronizeStream(stream1);
    
//     cout<<"op_cann done"<<endl;
//     // cout<< "output_token_ids:\n" << inputs.output_token_ids_op << endl;
//     auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
//     auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
//     auto out_torch_index_cpu = inputs.output_token_index_torch.to(torch::kCPU);
//     auto out_op_index_cpu = inputs.output_token_index_op.to(torch::kCPU);
//     auto out_torch_log_cpu = inputs.output_log_probs_torch.to(torch::kCPU).view({-1, 1});
//     auto out_op_log_cpu = inputs.output_log_probs_op.to(torch::kCPU).view({-1, 1});
//     cout<< "token ids equal: " << torch::equal(out_torch_cpu, out_op_cpu) << endl;
//     cout<< "token index equal: " << torch::equal(out_torch_index_cpu,out_op_index_cpu) <<endl;
//     // cout << "token ids shape : " << out_torch_cpu.sizes() << endl;
//     // cout << "token index shape: " << out_torch_index_cpu.sizes() << endl;
//     // cout << "token log probs shape: " << out_torch_log_cpu.sizes() << endl;
//     auto token_ids_diff_mask = (out_torch_cpu != out_op_cpu);
//     auto token_index_diff_mask = (out_torch_index_cpu != out_op_index_cpu);

//     auto token_ids_diff_indices = torch::nonzero(token_ids_diff_mask);
//     auto token_index_diff_indices = torch::nonzero(token_index_diff_mask);

//     // cout << "Token IDs different count: " << token_ids_diff_indices.size(0) << endl;
//     // cout << "Token Index different count: " << token_index_diff_indices.size(0) << endl;
    
//         // 打印不相等的索引和值
//     if (token_ids_diff_indices.size(0) > 0) {
//         // cout << token_ids_diff_indices <<endl;
//         auto torch_diff_vals = out_torch_cpu.index({token_ids_diff_mask});
//         auto op_diff_vals = out_op_cpu.index({token_ids_diff_mask});
        
//         cout << "Token IDs differences:" << endl;
//         // cout << "  torch values at diff positions: " << torch_diff_vals << endl;
//         // cout << "  op values at diff positions: " << op_diff_vals << endl;
    
//         auto diff_rows = token_ids_diff_indices.select(1, 0);
//         auto diff_cols = token_ids_diff_indices.select(1, 1);
//         auto torch_diff_log_vals = out_torch_log_cpu.index({diff_rows, diff_cols});
//         auto op_diff_log_vals = out_op_log_cpu.index({diff_rows, diff_cols});
//         // cout << "  torch log probs at diff positions: " << torch_diff_log_vals << endl;
//         // cout << "  op log probs at diff positions: " << op_diff_log_vals << endl;
//     }

//     if (token_index_diff_indices.size(0) > 0) {
//         // cout << token_index_diff_indices<<endl;
//         auto torch_diff_vals = out_torch_index_cpu.index({token_index_diff_mask});
//         auto op_diff_vals = out_op_index_cpu.index({token_index_diff_mask});
//         cout << "Token Index differences:" << endl;
//         // cout << "  torch values at diff positions: " << torch_diff_vals << endl;
//         // cout << "  op values at diff positions: " << op_diff_vals << endl;
//     }
//     // cout<< "output_token_index torch:\n" <<out_torch_index_cpu<<endl;
//     cout<< "token log equal: " <<torch::equal(out_torch_log_cpu,out_op_log_cpu) <<endl;
//     op_cann.destroy_tensors();
//     aclrtDestroyStream(stream1);
//     aclrtResetDevice(deviceId);
//     aclFinalize();
//     return 0;
// }
// TEST_F(BeamSearchTest, DifferentSizes) {
//     std::vector<std::tuple<int, int, int, int>> test_sizes = {
//         {2, 2, 2, 2},
//         {2, 2, 2, 2},
//         {2, 2, 2, 2}
//     };
//     for (auto [beam_width, top_k, request_num, sequence_length] : test_sizes) {
//         beam_search::BeamSearchBase inputs(beam_width, top_k, request_num, sequence_length);
//         inputs.create_torch_tensors();
//         beam_search::BeamSearchTorch op_torch(inputs);
//         op_torch.process();
//         beam_search::BeamSearchOp op_cann(inputs);
//         op_cann.process(stream1);
//         EXPECT_TRUE(torch::equal(inputs.output_token_ids_torch, inputs.output_token_ids_op))
//             << "Failed for size: beam_width=" << beam_width << ", top_k=" << top_k << ", request_num=" << request_num << ", sequence_length=" << sequence_length;
//         op_cann.destroy_tensors();
//     }
// }

// int main() {
    
//     beam_search::BeamSearchBase inputs(512, 512, 2, 2);
//     inputs.create_torch_tensors();
//     beam_search::BeamSearchTorch op_torch(inputs);
//     op_torch.process();
//     cout<<"op_torch done"<<endl;
//     int deviceId = 0;
//     aclrtStream stream1;
//     utils::initialize_acl(deviceId, &stream1);
//     beam_search::BeamSearchOp op_cann(inputs);
//     op_cann.process(stream1);
//     aclrtSynchronizeStream(stream1);
    
//     cout<<"op_cann done"<<endl;
//     auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
//     auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
//     // cout<< "out torch:\n" << out_torch_cpu << endl;
//     // cout<< "out op:\n" << out_op_cpu << endl;
//     auto out_torch_index_cpu = inputs.output_token_index_torch.to(torch::kCPU);
//     auto out_op_index_cpu = inputs.output_token_index_op.to(torch::kCPU);
//     // cout<< "out index torch:\n" << out_torch_index_cpu << endl;
//     // cout<< "out index op:\n" << out_op_index_cpu << endl;
//     auto out_torch_log_cpu = inputs.output_log_probs_torch.to(torch::kCPU).view({-1, 1});
//     auto out_op_log_cpu = inputs.output_log_probs_op.to(torch::kCPU).view({-1, 1});
//     // cout<< "out log torch:\n" << out_torch_log_cpu << endl;
//     // cout<< "out log op:\n" << out_op_log_cpu << endl;
//     cout<< "token ids equal: " << torch::equal(out_torch_cpu, out_op_cpu) << endl;
//     cout<< "token index equal: " << torch::equal(out_torch_index_cpu,out_op_index_cpu) <<endl;
//     // cout << "token ids shape : " << out_torch_cpu.sizes() << endl;
//     // cout << "token index shape: " << out_torch_index_cpu.sizes() << endl;
//     // cout << "token log probs shape: " << out_torch_log_cpu.sizes() << endl;
//     auto token_ids_diff_mask = (out_torch_cpu != out_op_cpu);
//     auto token_index_diff_mask = (out_torch_index_cpu != out_op_index_cpu);

//     auto token_ids_diff_indices = torch::nonzero(token_ids_diff_mask);
//     auto token_index_diff_indices = torch::nonzero(token_index_diff_mask);

//     // cout << "Token IDs different count: " << token_ids_diff_indices.size(0) << endl;
//     // cout << "Token Index different count: " << token_index_diff_indices.size(0) << endl;
    
//         // 打印不相等的索引和值
//     if (token_ids_diff_indices.size(0) > 0) {
//         // cout << token_ids_diff_indices <<endl;
//         auto torch_diff_vals = out_torch_cpu.index({token_ids_diff_mask});
//         auto op_diff_vals = out_op_cpu.index({token_ids_diff_mask});
        
//         cout << "Token IDs differences:" << endl;
//         // cout << "  torch values at diff positions: " << torch_diff_vals << endl;
//         // cout << "  op values at diff positions: " << op_diff_vals << endl;
    
//         auto diff_rows = token_ids_diff_indices.select(1, 0);
//         auto diff_cols = token_ids_diff_indices.select(1, 1);
//         auto torch_diff_log_vals = out_torch_log_cpu.index({diff_rows, diff_cols});
//         auto op_diff_log_vals = out_op_log_cpu.index({diff_rows, diff_cols});
//         // cout << "  torch log probs at diff positions: " << torch_diff_log_vals << endl;
//         // cout << "  op log probs at diff positions: " << op_diff_log_vals << endl;
//     }

//     if (token_index_diff_indices.size(0) > 0) {
//         // cout << token_index_diff_indices<<endl;
//         auto torch_diff_vals = out_torch_index_cpu.index({token_index_diff_mask});
//         auto op_diff_vals = out_op_index_cpu.index({token_index_diff_mask});
//         cout << "Token Index differences:" << endl;
//         // cout << "  torch values at diff positions: " << torch_diff_vals << endl;
//         // cout << "  op values at diff positions: " << op_diff_vals << endl;
//     }
//     // cout<< "output_token_index torch:\n" <<out_torch_index_cpu<<endl;
//     cout<< "token log equal: " <<torch::equal(out_torch_log_cpu,out_op_log_cpu) <<endl;
//     op_cann.destroy_tensors();
//     aclrtDestroyStream(stream1);
//     aclrtResetDevice(deviceId);
//     aclFinalize();
//     return 0;
// }Id);
//         aclFinalize();
//     }
// };

// TEST_F(BeamSearchTest, BasicMatmulCorrectness) {
//     beam_search::BeamSearchBase inputs(2, 2, 2, 2);
//     inputs.create_torch_tensors();
//     // beam_search::BeamSearchTorch op_torch(inputs);
//     // op_torch.process();
//     // print torch values
//     printf("torch value:\n");
//     {
//         std::ostringstream oss; oss << inputs.token_ids; printf("token_ids:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.log_probs; printf("log_probs:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.top_tokens; printf("top_tokens:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.top_probs; printf("top_probs:\n%s\n", oss.str().c_str());
//     }
//     // {
//     //     std::ostringstream oss; oss << inputs.output_token_ids_torch; printf("output_token_ids_torch:\n%s\n", oss.str().c_str());
//     // }
//     beam_search::BeamSearchOp op_cann(inputs);
//     op_cann.process(stream1);
//     // Ensure NPU ops are finished before reading back
//     aclrtSynchronizeStream(stream1);
//     // Compare on CPU to avoid device mismatch
//     // auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
//     // auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
//     // std::cout << "out_op_cpu:\n" << out_op_cpu << std::endl;

//     // std::cout << "less than 1e-5: " << (out_torch_cpu - out_op_cpu).abs().max().item<float>() << std::endl;
//     // EXPECT_TRUE(torch::equal(out_torch_cpu, out_op_cpu))
//     //     << "Custom op output does not match native output";
//     op_cann.destroy_tensors();
// }

// int main() {
//     int deviceId = 15;
//     aclrtStream stream1;
//     utils::initialize_acl(deviceId, &stream1);
//     beam_search::BeamSearchBase inputs(512, 512, 2, 2, deviceId);
//     inputs.create_torch_tensors();
//     beam_search::BeamSearchTorch op_torch(inputs);
//     op_torch.process();
//     cout<<"op_torch done"<<endl;
    
//     beam_search::BeamSearchOp op_cann(inputs);
//     op_cann.process(stream1);
//     aclrtSynchronizeStream(stream1);
    
//     cout<<"op_cann done"<<endl;
//     // cout<< "output_token_ids:\n" << inputs.output_token_ids_op << endl;
//     auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
//     auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
//     auto out_torch_index_cpu = inputs.output_token_index_torch.to(torch::kCPU);
//     auto out_op_index_cpu = inputs.output_token_index_op.to(torch::kCPU);
//     auto out_torch_log_cpu = inputs.output_log_probs_torch.to(torch::kCPU).view({-1, 1});
//     auto out_op_log_cpu = inputs.output_log_probs_op.to(torch::kCPU).view({-1, 1});
//     cout<< "token ids equal: " << torch::equal(out_torch_cpu, out_op_cpu) << endl;
//     cout<< "token index equal: " << torch::equal(out_torch_index_cpu,out_op_index_cpu) <<endl;
//     // cout << "token ids shape : " << out_torch_cpu.sizes() << endl;
//     // cout << "token index shape: " << out_torch_index_cpu.sizes() << endl;
//     // cout << "token log probs shape: " << out_torch_log_cpu.sizes() << endl;
//     auto token_ids_diff_mask = (out_torch_cpu != out_op_cpu);
//     auto token_index_diff_mask = (out_torch_index_cpu != out_op_index_cpu);

//     auto token_ids_diff_indices = torch::nonzero(token_ids_diff_mask);
//     auto token_index_diff_indices = torch::nonzero(token_index_diff_mask);

//     // cout << "Token IDs different count: " << token_ids_diff_indices.size(0) << endl;
//     // cout << "Token Index different count: " << token_index_diff_indices.size(0) << endl;
    
//         // 打印不相等的索引和值
//     if (token_ids_diff_indices.size(0) > 0) {
//         // cout << token_ids_diff_indices <<endl;
//         auto torch_diff_vals = out_torch_cpu.index({token_ids_diff_mask});
//         auto op_diff_vals = out_op_cpu.index({token_ids_diff_mask});
        
//         cout << "Token IDs differences:" << endl;
//         // cout << "  torch values at diff positions: " << torch_diff_vals << endl;
//         // cout << "  op values at diff positions: " << op_diff_vals << endl;
    
//         auto diff_rows = token_ids_diff_indices.select(1, 0);
//         auto diff_cols = token_ids_diff_indices.select(1, 1);
//         auto torch_diff_log_vals = out_torch_log_cpu.index({diff_rows, diff_cols});
//         auto op_diff_log_vals = out_op_log_cpu.index({diff_rows, diff_cols});
//         // cout << "  torch log probs at diff positions: " << torch_diff_log_vals << endl;
//         // cout << "  op log probs at diff positions: " << op_diff_log_vals << endl;
//     }

//     if (token_index_diff_indices.size(0) > 0) {
//         // cout << token_index_diff_indices<<endl;
//         auto torch_diff_vals = out_torch_index_cpu.index({token_index_diff_mask});
//         auto op_diff_vals = out_op_index_cpu.index({token_index_diff_mask});
//         cout << "Token Index differences:" << endl;
//         // cout << "  torch values at diff positions: " << torch_diff_vals << endl;
//         // cout << "  op values at diff positions: " << op_diff_vals << endl;
//     }
//     // cout<< "output_token_index torch:\n" <<out_torch_index_cpu<<endl;
//     cout<< "token log equal: " <<torch::equal(out_torch_log_cpu,out_op_log_cpu) <<endl;
//     op_cann.destroy_tensors();
//     aclrtDestroyStream(stream1);
//     aclrtResetDevice(deviceId);
//     aclFinalize();
//     return 0;
// }
// TEST_F(BeamSearchTest, DifferentSizes) {
//     std::vector<std::tuple<int, int, int, int>> test_sizes = {
//         {2, 2, 2, 2},
//         {2, 2, 2, 2},
//         {2, 2, 2, 2}
//     };
//     for (auto [beam_width, top_k, request_num, sequence_length] : test_sizes) {
//         beam_search::BeamSearchBase inputs(beam_width, top_k, request_num, sequence_length);
//         inputs.create_torch_tensors();
//         beam_search::BeamSearchTorch op_torch(inputs);
//         op_torch.process();
//         beam_search::BeamSearchOp op_cann(inputs);
//         op_cann.process(stream1);
//         EXPECT_TRUE(torch::equal(inputs.output_token_ids_torch, inputs.output_token_ids_op))
//             << "Failed for size: beam_width=" << beam_width << ", top_k=" << top_k << ", request_num=" << request_num << ", sequence_length=" << sequence_length;
//         op_cann.destroy_tensors();
//     }
// }

int main() {
    
    beam_search::BeamSearchBase inputs(512, 512, 2, 2);
    inputs.create_torch_tensors();
    beam_search::BeamSearchTorch op_torch(inputs);
    op_torch.process();
    cout<<"op_torch done"<<endl;
    int deviceId = 15;
    aclrtStream stream1;
    utils::initialize_acl(deviceId, &stream1);
    beam_search::BeamSearchOp op_cann(inputs);
    op_cann.process(stream1);
    aclrtSynchronizeStream(stream1);
    
    cout<<"op_cann done"<<endl;
    // cout<< "output_token_ids:\n" << inputs.output_token_ids_op << endl;
    auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
    auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
    auto out_torch_index_cpu = inputs.output_token_index_torch.to(torch::kCPU);
    auto out_op_index_cpu = inputs.output_token_index_op.to(torch::kCPU);
    auto out_torch_log_cpu = inputs.output_log_probs_torch.to(torch::kCPU).view({-1, 1});
    auto out_op_log_cpu = inputs.output_log_probs_op.to(torch::kCPU).view({-1, 1});
    cout<< "token ids equal: " << torch::equal(out_torch_cpu, out_op_cpu) << endl;
    cout<< "token index equal: " << torch::equal(out_torch_index_cpu,out_op_index_cpu) <<endl;
    // cout << "token ids shape : " << out_torch_cpu.sizes() << endl;
    // cout << "token index shape: " << out_torch_index_cpu.sizes() << endl;
    // cout << "token log probs shape: " << out_torch_log_cpu.sizes() << endl;
    auto token_ids_diff_mask = (out_torch_cpu != out_op_cpu);
    auto token_index_diff_mask = (out_torch_index_cpu != out_op_index_cpu);

    auto token_ids_diff_indices = torch::nonzero(token_ids_diff_mask);
    auto token_index_diff_indices = torch::nonzero(token_index_diff_mask);

    // cout << "Token IDs different count: " << token_ids_diff_indices.size(0) << endl;
    // cout << "Token Index different count: " << token_index_diff_indices.size(0) << endl;
    
        // 打印不相等的索引和值
    if (token_ids_diff_indices.size(0) > 0) {
        // cout << token_ids_diff_indices <<endl;
        auto torch_diff_vals = out_torch_cpu.index({token_ids_diff_mask});
        auto op_diff_vals = out_op_cpu.index({token_ids_diff_mask});
        
        cout << "Token IDs differences:" << endl;
        // cout << "  torch values at diff positions: " << torch_diff_vals << endl;
        // cout << "  op values at diff positions: " << op_diff_vals << endl;
    
        auto diff_rows = token_ids_diff_indices.select(1, 0);
        auto diff_cols = token_ids_diff_indices.select(1, 1);
        auto torch_diff_log_vals = out_torch_log_cpu.index({diff_rows, diff_cols});
        auto op_diff_log_vals = out_op_log_cpu.index({diff_rows, diff_cols});
        // cout << "  torch log probs at diff positions: " << torch_diff_log_vals << endl;
        // cout << "  op log probs at diff positions: " << op_diff_log_vals << endl;
    }

    if (token_index_diff_indices.size(0) > 0) {
        // cout << token_index_diff_indices<<endl;
        auto torch_diff_vals = out_torch_index_cpu.index({token_index_diff_mask});
        auto op_diff_vals = out_op_index_cpu.index({token_index_diff_mask});
        cout << "Token Index differences:" << endl;
        // cout << "  torch values at diff positions: " << torch_diff_vals << endl;
        // cout << "  op values at diff positions: " << op_diff_vals << endl;
    }
    // cout<< "output_token_index torch:\n" <<out_torch_index_cpu<<endl;
    cout<< "token log equal: " <<torch::equal(out_torch_log_cpu,out_op_log_cpu) <<endl;
    op_cann.destroy_tensors();
    aclrtDestroyStream(stream1);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}