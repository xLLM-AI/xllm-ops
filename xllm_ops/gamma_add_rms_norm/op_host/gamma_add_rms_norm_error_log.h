/**
 * Copyright (c) 2026 The xLLM Authors. All Rights Reserved.
 */

#pragma once

#include "log/ops_log.h"

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opname, param, actual, reason) \
    OPS_LOG_E(opname, "Invalid shape for %s, actual: %s, reason: %s", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opname, param, actual, reason) \
    OPS_LOG_E(opname, "Invalid shapes for %s, actual: %s, reason: %s", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opname, param, actual, expected) \
    OPS_LOG_E(opname, "Invalid shape dim for %s, actual: %s, expected: %s", param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opname, param, actual, reason) \
    OPS_LOG_E(opname, "Invalid shape dims for %s, actual: %s, reason: %s", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(opname, param, actual, expected) \
    OPS_LOG_E(opname, "Invalid value for %s, actual: %s, expected: %s", param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opname, param, actual, reason) \
    OPS_LOG_E(opname, "Invalid value for %s, actual: %s, reason: %s", param, actual, reason)
#endif
