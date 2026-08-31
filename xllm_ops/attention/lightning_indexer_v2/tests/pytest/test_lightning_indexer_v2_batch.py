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

import itertools
import torch
import result_compare_method
from batch import lightning_indexer_v2_pt_loadprocess
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
TEST_INPUT_PATH = "pt_path"
pt_dir = TEST_INPUT_PATH
result_path = Path('result.xlsx')  # 或使用传入的result_path
result_path = Path('result.xlsx')
device_id = 0
# 支持通过环境变量 LIV2_TESTCASE_PATH 指定单条用例文件，实现进程级隔离执行：
#   - 设置时：仅运行该条用例（配合 batch_isolated_run.sh 每条用例拉起独立进程）
#   - 未设置：回退为原有行为，一次性加载目录下全部用例
single_case_path = os.environ.get("LIV2_TESTCASE_PATH", "").strip()
# flag：是否处于批量隔离模式（由 batch_isolated_run.sh 设置 LIV2_TESTCASE_PATH 触发）
is_isolated_mode = bool(single_case_path)
# flag：运行模式 eager / graph（通过环境变量 LIV2_RUN_MODE 或命令行参数设置，默认 eager）
run_mode = os.environ.get("LIV2_RUN_MODE", "eager").strip().lower()
# 生成所有的组合，并转换为字典列表
locals()["testcase_files"] = []
if single_case_path:
    if not os.path.isfile(single_case_path):
        print(f"错误: 环境变量 LIV2_TESTCASE_PATH 指定的用例文件不存在: {single_case_path}")
    else:
        print(f"单用例隔离模式, 仅执行: {single_case_path}")
        locals()["testcase_files"].append(single_case_path)
elif os.path.isdir(pt_dir):
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {pt_dir}")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:  
            filepath = os.path.join(pt_dir, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {pt_dir}")

def liv2(testcase_files):   # 初始化参数和tensor
    try:
        if run_mode == "graph":
            cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value, output_idx_offset, params = \
                lightning_indexer_v2_pt_loadprocess.test_liv2_process_graph(testcase_files, device_id=device_id)
        else:
            cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value, output_idx_offset, params = lightning_indexer_v2_pt_loadprocess.test_liv2_process(testcase_files, device_id=device_id)
        if npu_result != None:
            result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result, topk_value, output_idx_offset, params, cpu_topk_value, npu_topk_value)
        else:
            result = "Failed"
            fulfill_percent = 0
        
        return_value = params[25]
        if return_value:
            result_return_value, fulfill_precent_return_value = result_compare_method.check_result_return_value(cpu_topk_value, npu_topk_value, params)
            print(f"result_return_value: {result_return_value}")
            print(f"result_return_value: {fulfill_precent_return_value}")
        else:
            result_return_value = "N/A"
            fulfill_precent_return_value = 0
    except Exception as e:
        print("NPU ERROR：", e)
        result = "NPU ERROR"
        fulfill_percent = 0
        result_return_value = "N/A"
        fulfill_precent_return_value = 0
        params = [None] * 27

    row_data = {
        "case_name": Path(testcase_files).stem,
        "batch_size": params[0],
        "q_seq": params[1],
        "k_seq": params[2],
        "q_t_size": params[3],
        "k_t_size": params[4],
        "q_head_num": params[5],
        "k_head_num": params[6],
        "head_dim": params[7],
        "block_size": params[8],
        "block_num": params[9],
        "qk_dtype": params[10],
        "cu_seqlens_q": params[11],
        "cu_seqlens_k": params[12],
        "seqused_q": params[13],
        "seqused_k": params[14],
        "cmp_residual_k": params[15],
        "output_idx_offset": params[16],
        "layout_q": params[17],
        "layout_k": params[18],
        "topk":params[19],
        "mask_mode":params[20],
        "query_datarange":params[21],
        "key_datarange":params[22],
        "weights_datarange":params[23],
        "cmp_ratio":params[24],
        "return_value": params[25],
        "max_seqlen_q":params[26],
        "result":result,
        "fulfill_percent":fulfill_percent,
        "result_return_value": result_return_value,
        "fulfill_percent_return_value": fulfill_precent_return_value
    }

    # 检查文件是否存在
    if result_path.exists():
        # 读取现有数据
        df = pd.read_excel(result_path)
        
        # 检查列名是否一致
        if set(df.columns) != set(row_data.keys()):
            print("警告：变量名与Excel列名不匹配！")
            print(f"Excel列名: {list(df.columns)}")
            print(f"变量名: {list(row_data.keys())}")
            print("请检查变量名或Excel文件")
            return False
        
        # 追加新行
        new_df = pd.DataFrame([row_data])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        # 文件不存在，创建新的DataFrame
        df = pd.DataFrame([row_data])
    
    # 保存到Excel
    df.to_excel(result_path, index=False)
    if result == "NPU ERROR":
        pytest.fail(f"用例执行失败:{Path(testcase_files).stem}")

@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_liv2(testcase_files):   # 初始化参数和tensor
    if is_isolated_mode:
        # 批量隔离模式：shell 层已通过独立 pytest 进程提供进程隔离，内部使用线程池即可
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = executor.submit(liv2, testcase_files)
            for future in as_completed([futures]):
                try:
                    result = future.result()
                except Exception as e:
                    pytest.fail(f"当前用例线程执行失败：{e}")
    else:
        # 非隔离模式（直接 pytest）：使用子进程隔离，防止单条用例崩溃影响整体
        with ProcessPoolExecutor(max_workers=1) as executor:
            # 创建当前用例子进程
            future1 = executor.submit(liv2, testcase_files)
            # 检查退出码
            for future in as_completed([future1]):
                try:
                    result = future.result()
                except Exception as e:
                    pytest.fail(f"❌ 当前用例子进程执行失败：{e}")
