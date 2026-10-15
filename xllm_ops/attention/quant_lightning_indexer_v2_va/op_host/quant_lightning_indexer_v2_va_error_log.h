/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <iostream>

// 说明: 这些错误日志宏不依赖 OP_LOGE。opdev/op_log.h (经 opdev/format_utils.h 传递包含)
// 会把 OP_LOGE 重定义为 errno 语义 (std::to_string(errno)), 与 opname 语义冲突;
// 故此处直接用 std::cerr 输出, 自包含、不受宏遮蔽影响, 且能正确处理 std::string / const char* /
// 数值等各类实参 (operator<< 重载)。

#ifndef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opname, param, reason) \
    do {                                                                \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid argument for " << (param) \
                  << ", reason: " << (reason) << std::endl;             \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPE
#define OP_LOGE_FOR_INVALID_DTYPE(opname, param, actual, expected) \
    do {                                                           \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid dtype for " << (param) \
                  << ", actual: " << (actual) << ", expected: " << (expected) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opname, param, actual, reason) \
    do {                                                                     \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid dtype for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opname, param, actual, reason) \
    do {                                                                      \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid dtypes for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE
#define OP_LOGE_FOR_INVALID_SHAPE(opname, param, actual, expected) \
    do {                                                           \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shape for " << (param) \
                  << ", actual: " << (actual) << ", expected: " << (expected) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opname, param, actual, reason) \
    do {                                                                     \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shape for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opname, param, actual, reason) \
    do {                                                                      \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shapes for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opname, param, actual, expected) \
    do {                                                              \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shape dim for " << (param) \
                  << ", actual: " << (actual) << ", expected: " << (expected) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opname, param, actual, reason) \
    do {                                                                        \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shape dim for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opname, param, actual, reason) \
    do {                                                                         \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shape dims for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE
#define OP_LOGE_FOR_INVALID_SHAPESIZE(opname, param, actual, expected) \
    do {                                                               \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shape size for " << (param) \
                  << ", actual: " << (actual) << ", expected: " << (expected) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opname, param, actual, reason) \
    do {                                                                         \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shape size for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opname, param, actual, reason) \
    do {                                                                          \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid shape sizes for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(opname, param, actual, expected) \
    do {                                                           \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid value for " << (param) \
                  << ", actual: " << (actual) << ", expected: " << (expected) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opname, param, actual, reason) \
    do {                                                                     \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid value for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opname, param, actual, reason) \
    do {                                                                      \
        std::cerr << "[ERROR] [" << (opname) << "] Invalid values for " << (param) \
                  << ", actual: " << (actual) << ", reason: " << (reason) << std::endl; \
    } while (0)
#endif