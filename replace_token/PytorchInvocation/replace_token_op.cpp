#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/npu/NPUBridge.h>

// 声明replace_token的实现
torch::Tensor replace_token_impl(const torch::Tensor& x, const torch::Tensor& y) {
    // 确保输入在NPU上
    TORCH_CHECK(x.device().type() == at::kNPU, "Input tensor 'x' must be on NPU");
    TORCH_CHECK(y.device().type() == at::kNPU, "Input tensor 'y' must be on NPU");
    
    // 调用NPU自定义算子
    return at::npu_replace_token(x, y);
}

// 注册算子
TORCH_LIBRARY(custom_ops, m) {
    m.def("replace_token", replace_token_impl);
} 