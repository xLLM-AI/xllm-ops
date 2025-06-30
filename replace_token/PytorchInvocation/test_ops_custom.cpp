#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/npu/NPUBridge.h>
#include <iostream>

// 自定义的replace_token操作声明
torch::Tensor npu_replace_token(const torch::Tensor& x, const torch::Tensor& y);

// 测试类
class TestCustomAdd {
public:
    void test_add_custom() {
        try {
            // 设置NPU设备
            auto device = torch::Device("npu:0");
            
            // 创建输入张量
            auto options = torch::TensorOptions()
                .dtype(torch::kInt32)
                .device(torch::kCPU);
                
            // 生成随机张量，大小为(5,1)
            auto x = torch::randint(-4, 5, {5, 1}, options);
            auto y = torch::randint(1, 5, {5, 1}, options);
            
            // 打印输入张量
            std::cout << "Input x:\n" << x << std::endl;
            std::cout << "Input y:\n" << y << std::endl;
            
            // 将张量移动到NPU设备
            auto x_npu = x.to(device);
            auto y_npu = y.to(device);
            
            // 设置不需要梯度
            x_npu.requires_grad_(false);
            y_npu.requires_grad_(false);
            
            // 执行replace_token操作
            auto output_npu = torch::ops::custom_ops::replace_token(x_npu, y_npu);
            
            // 同步NPU操作
            c10::npu::NPUStream::getCurrentStream().synchronize();
            
            // 将结果移回CPU并打印
            auto output = output_npu.to(torch::kCPU);
            std::cout << "Output:\n" << output << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
};

int main() {
    // 初始化NPU设备
    torch::npu::init_npu();
    
    // 禁用内部格式
    torch::npu::config::allow_internal_format = false;
    
    TestCustomAdd test;
    test.test_add_custom();
    
    return 0;
} 