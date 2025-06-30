/**
 * @file replace_token.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using variable_list = std::vector<at::Tensor>;

// 为NPU设备注册前向实现
at::Tensor replace_token_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    // 创建输出内存
    at::Tensor result = at::empty_like(self);

    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnReplaceToken, self, other, result);
    return result;
}

// 为NPU设备注册反向实现
std::tuple<at::Tensor, at::Tensor> replace_token_backward_impl_npu(const at::Tensor& grad) {
    at::Tensor result = grad; // 创建输出内存

    return {result, result};
}


// 通过继承torch::autograd::Function类实现前反向绑定
class ReplaceTokenFunction : public torch::autograd::Function<ReplaceTokenFunction> {
    public:
        static at::Tensor forward(AutogradContext *ctx, at::Tensor self, at::Tensor other) {
            at::AutoDispatchBelowADInplaceOrView guard;
            static auto op = torch::Dispatcher::singleton()
                            .findSchemaOrThrow("myops::replace_token", "")
                            .typed<decltype(replace_token_impl_npu)>();

            auto result = op.call(self, other);
            return result;
        }

        static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
            auto grad_output = grad_outputs[0];

            static auto op = torch::Dispatcher::singleton()
                            .findSchemaOrThrow("myops::replace_token_backward", "")
                            .typed<decltype(replace_token_backward_impl_npu)>();
            auto result = op.call(grad_output);
            return {std::get<0>(result), std::get<1>(result)};
        }
};

// 使用的时候调用apply()方法
at::Tensor replace_token_autograd(const at::Tensor& self, const at::Tensor& other) {
    return ReplaceTokenFunction::apply(self, other);
}

// 为NPU设备注册前反向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("replace_token", &replace_token_impl_npu);
    m.impl("replace_token_backward", &replace_token_backward_impl_npu);
}

// 给op绑定NPU的自动求导实现
// 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m) {
    m.impl("replace_token", &replace_token_autograd);
}
