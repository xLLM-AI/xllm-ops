// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License").
// http://www.apache.org/licenses/LICENSE-2.0

#include "core/platform/platform.h"

// The standalone harness exercises adapters without linking the entire server.
// Device selection and health are checked by the Python runner before NPU use.
bool xllm::Platform::is_ascend950() { return true; }
