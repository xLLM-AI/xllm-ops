#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

# 定义测试参数组合
TEST_PARAMS = {
    # 基础场景
    "li_default_a5":{
        "batch_size": [1],
        "q_seq": [1],
        "k_seq": [256],
        "q_t_size":[1],
        "k_t_size":[256],#压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [512], # 取16的整数倍，最多支持到1024
        "block_num":[8],
        "qk_dtype": [torch.float16],
        "cu_seqlens_q": [None],
        "cu_seqlens_k": [None],
        "seqused_q": [[1]],
        "seqused_k": [[256]],
        "cmp_residual_k": [None],
        "output_idx_offset": [None],
        "layout_q": ["BSND"],
        "layout_k":["PA_BBND"],
        "topk": [128],
        "mask_mode": [0],
        "query_datarange":[[-448,448]],
        "key_datarange":[[-20,20]],
        "weights_datarange":[[-123,123]],
        "cmp_ratio":[1],
        "return_value": [0],
        "max_seqlen_q": [-1]
    },
}

# 按需选择要启用的测试参数（例如默认启用所有）
properties = torch.npu.get_device_properties()
ENABLED_PARAMS = [TEST_PARAMS["li_default_a5"]] 