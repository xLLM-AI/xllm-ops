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

import math
import random
import logging
import torch
import datetime
import os
import sys
import ast
import numpy as np
from time import time

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def cal_relative_diff_np_isclose(real_data, expect_data, type_str="fp16"):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


def print_log(data=None, level="INFO"):
    print(
        "[%s] [%s]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            level,
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        "Error Line-----------------------------------------------------------------------------"
    )
    print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i,
                    expect_data[i],
                    real_data[i],
                    abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_3, dot_3, dot_3, dot_3, dot_3)
            )
        elif count > 100:
            break

    print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_idx,
                    expect_data[m_idx],
                    real_data[m_idx],
                    abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                    max_error,
                )
            )
        else:
            break
    print_log(
        "---------------------------------------------------------------------------------------"
    )


def display_output_np_isclose(
    real_data, expect_data, start, end, expect_fp32_data=None
):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])
        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    if expect_fp32_data is not None:
        print_log(
            "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        )
    else:
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def find_batch_and_position(cu_seqlens, x):
    """
    判断x属于哪个batch以及在该batch中的位置

    参数:
        cu_seqlens: 前缀和列表, cu_seqlens[b_idx]表示前(b_idx)个batch的总长度
        x: 需要判断的数值

    返回:
        tuple: (batch_idx, position)
               - batch_idx: 所属的batch索引(从0开始)，超出范围则为-1
               - position: 在该batch中的位置(从0开始), 超出范围则为-1
    """
    if not cu_seqlens:
        return (-1, -1)
    # 遍历前缀和列表查找所属批次
    for batch_idx in range(len(cu_seqlens) - 1):
        # 计算当前批次的起始位置
        start = cu_seqlens[batch_idx]
        # 判断是否在当前批次范围内
        if start <= x < cu_seqlens[batch_idx + 1]:
            # 计算在当前批次中的位置（偏移量）
            position = x - start
            return (batch_idx, position)
    # 超出所有批次范围
    return (-1, -1)


def judge_value_by_isclose(real_data, data_compe):
    atol = 2.5e-05
    rtol = 0.005
    pct_thd = 0.005
    diff_thd = 0.005
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    split_count = int(end - start + 1) if end != start else 1

    if str(real_data.dtype) == "bfloat16":
        # bf16 尾数位少、舍入误差大，误差门限放宽到 1/128（约 0.0078125）
        atol = 0.0001
        rtol = 1.0 / 128
        diff_thd = 1.0 / 128
        diff_result = np.isclose(
            real_data.astype(np.float32),
            data_compe.astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    else:
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    err_idx = np.where(diff_result != np.array((True,)))[0]
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    pct_thd = (1 - pct_thd) * 100.0
    result = True if (fulfill_percent >= pct_thd) else False
    return result


def compare_topk_valid(
    cur_cpu,
    cur_npu,
    topk_value,
    bsn,
    diff_npu,
    diff_cpu,
    cur_npu_output_value=None,
    cur_cpu_output_value=None,
    thres=0.001,
    return_value_flag=False,
    output_idx_offset=None,
    layout_query=None,
    cu_seqlens_q=None,
    q_seq=0,
):
    b_idx, s1_idx, n2_idx = bsn
    max_re = 0.0
    npu_pass = True
    npu_set = set(cur_npu)
    cpu_set = set(cur_cpu)

    is_equivalent = npu_set == cpu_set
    if is_equivalent:
        pass
    else:
        if output_idx_offset is not None:
            if layout_query == "TND":
                cur_prefix = cu_seqlens_q[b_idx]
                offset = np.array(output_idx_offset.cpu()).flatten()[
                    cur_prefix + s1_idx
                ]
            else:
                offset = np.array(output_idx_offset.cpu()).flatten()[
                    b_idx * q_seq + s1_idx
                ]
            offset_mask = cur_cpu != -1
            cur_npu = np.where(offset_mask, cur_npu - offset, cur_npu)
            cur_cpu = np.where(offset_mask, cur_cpu - offset, cur_cpu)
            npu_set = set(cur_npu)
            cpu_set = set(cur_cpu)
        value_bm = topk_value[b_idx, n2_idx, s1_idx, cur_cpu[-1]]
        element_list = topk_value[b_idx, n2_idx, s1_idx, :]
        only_in_npu = npu_set - cpu_set
        only_in_cpu = cpu_set - npu_set
        only_in_npu_list = list(only_in_npu)
        only_in_cpu_list = list(only_in_cpu)
        for diff_idx in range(len(only_in_npu_list)):
            element_npu = element_list[only_in_npu_list[diff_idx]]
            element_cpu = element_list[only_in_cpu_list[diff_idx]]
            npu_ae = abs(element_npu - value_bm)
            cpu_ae = abs(element_cpu - value_bm)
            if value_bm == 0:
                if npu_ae == 0:
                    npu_re = 0.0
                else:
                    npu_re = float("inf")
                if cpu_ae == 0:
                    cpu_re = 0.0
                else:
                    cpu_re = float("inf")
            else:
                npu_re = abs(npu_ae / value_bm)
                cpu_re = abs(cpu_ae / value_bm)
            if npu_re > thres or cpu_re > thres:
                if return_value_flag:
                    if not judge_value_by_isclose(
                        cur_npu_output_value, cur_cpu_output_value
                    ):
                        npu_pass = False
                        diff_npu.append(element_npu)
                        diff_cpu.append(element_cpu)
                        max_re = max(max_re, npu_re, cpu_re)
                else:
                    npu_pass = False
                    diff_npu.append(element_npu)
                    diff_cpu.append(element_cpu)
                    max_re = max(max_re, npu_re)
    return npu_pass, max_re


def compare_return_value(cur_npu_output_value=None, cur_cpu_output_value=None):
    max_re = 0.0
    npu_pass = True
    npu_pass = judge_value_by_isclose(cur_npu_output_value, cur_cpu_output_value)
    return npu_pass, max_re


def trans_tnd_actseq(list):
    list_len = len(list)
    if list_len == 0:
        raise ValueError("TND情况下 act_seq需要必传")
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i + 1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            raise ValueError(f"TND情况下 act_seq_len 为非递减数列 act_seq_len={list}")
    return list_new


def check_result(
    expect,
    result,
    topk_value,
    output_idx_offset,
    params,
    cpu_topk_value,
    npu_topk_value,
):
    (
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        _,
        layout_query,
        layout_key,
        topk,
        mask_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        cmp_ratio,
        return_value,
        max_seqlen_q,
    ) = params

    if isinstance(cu_seqlens_q, int):
        cu_seqlens_q = [cu_seqlens_q]
    elif isinstance(cu_seqlens_q, list):
        cu_seqlens_q = cu_seqlens_q
    elif cu_seqlens_q is not None:
        cu_seqlens_q = ast.literal_eval(cu_seqlens_q)

    if isinstance(cu_seqlens_k, int):
        cu_seqlens_k = [cu_seqlens_k]
    elif isinstance(cu_seqlens_k, list):
        cu_seqlens_k = cu_seqlens_k
    elif cu_seqlens_k is not None:
        cu_seqlens_k = ast.literal_eval(cu_seqlens_k)

    if isinstance(seqused_q, int):
        seqused_q = [seqused_q]
    elif isinstance(seqused_q, list):
        seqused_q = seqused_q
    elif seqused_q is not None:
        seqused_q = ast.literal_eval(seqused_q)

    if isinstance(seqused_k, int):
        seqused_k = [seqused_k]
    elif isinstance(seqused_k, list):
        seqused_k = seqused_k
    elif seqused_k is not None:
        seqused_k = ast.literal_eval(seqused_k)

    npu_pass = True
    max_error = 0
    max_re = 0
    thres = 0.001
    diff_thd = 0.01
    pct_thd = 0.05
    max_diff_hd = 0.1
    rtol = 0.005
    atol = 0.000025
    max_error_idx = 10000000
    cpu_output = expect.cpu().numpy()
    npu_output = result.cpu().numpy()
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    diff_cpu = []
    diff_npu = []

    if layout_query in ["BSND"]:
        sp = (batch_size, q_seq, k_head_num)
        total_rows = batch_size * q_seq * k_head_num
    elif layout_query in ["TND"]:
        sp = (q_t_size, k_head_num)
        total_rows = q_t_size * k_head_num
    else:
        total_rows = 0
        sp = (0, 0)
    print(f"total_line is {total_rows}")
    npu_reshape = npu_output.reshape([total_rows, topk])
    cpu_reshape = cpu_output.reshape([total_rows, topk])
    if return_value:
        npu_topk_value = npu_topk_value.cpu().numpy()
        npu_topk_value = npu_topk_value.reshape([total_rows, topk])
        cpu_topk_value = cpu_topk_value.reshape([total_rows, topk])
    start_time = time()
    invalid_data = cpu_reshape != -1
    valid_lens = invalid_data.sum(axis=-1)  # (total_rows,)
    # 判断有效值部分集合是否相同
    cpu_output_sorted = np.sort(cpu_reshape, axis=1)
    npu_output_sorted = np.sort(npu_reshape, axis=1)
    diff_rows = np.zeros(total_rows, dtype=bool)
    diff_rows |= np.any(
        cpu_output_sorted != npu_output_sorted, axis=1
    )  # 标记存在差异的行
    test_id = []
    rows = []
    if np.any(diff_rows):
        rows = np.where(diff_rows)[0]
    num_rows = len(rows)
    if num_rows:
        print(f"需要进行第二步比较的batch有{num_rows}")
    else:
        print("有效值集合相同，无需进行比较")
    for t_id in rows:
        bsn = np.unravel_index(t_id, sp)
        npu_topk_output_value = None
        cpu_topk_output_value = None
        if layout_query == "TND":
            b_idx, s1_idx = find_batch_and_position(cu_seqlens_q, bsn[0])
            bsn = (b_idx, s1_idx, bsn[-1])
        if return_value:
            cpu_topk_output_value = cpu_topk_value[t_id, :]
            npu_topk_output_value = npu_topk_value[t_id, :]
        npu_pass_t = True
        max_re_t = 0
        valid_len = valid_lens[t_id]
        npu_pass_t, max_re_t = compare_topk_valid(
            cpu_reshape[t_id, :valid_len],
            npu_reshape[t_id, :valid_len],
            topk_value,
            bsn,
            diff_npu,
            diff_cpu,
            npu_topk_output_value,
            cpu_topk_output_value,
            thres,
            return_value,
            output_idx_offset,
            layout_query,
            cu_seqlens_q,
            q_seq,
        )
        if not npu_pass_t:
            npu_pass = False
    end_time = time()
    print(f"耗时：{end_time - start_time:.6f} 秒")
    topk_precision = not diff_npu and not diff_cpu
    if topk_precision:
        print("[success]TopK精度通过, idx不同的地方的value误差在阈值之内")
    else:
        print("[fail]TopK精度失败")
    print(f"npu_pass is {npu_pass}")
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    diff_result = np.isclose(
        real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = np.where(diff_result != np.array((True,)))[0]
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]
    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (npu_pass or topk_precision) else "Failed"
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, pct_thd, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd)
        )
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return result, fulfill_percent


def check_result_return_value(expect, result, params):
    (
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        output_idx_offset,
        layout_query,
        layout_key,
        topk,
        mask_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        cmp_ratio,
        return_value,
        max_seqlen_q,
    ) = params

    if isinstance(cu_seqlens_q, int):
        cu_seqlens_q = [cu_seqlens_q]
    elif isinstance(cu_seqlens_q, list):
        cu_seqlens_q = cu_seqlens_q
    elif cu_seqlens_q is not None:
        cu_seqlens_q = ast.literal_eval(cu_seqlens_q)

    if isinstance(cu_seqlens_k, int):
        cu_seqlens_k = [cu_seqlens_k]
    elif isinstance(cu_seqlens_k, list):
        cu_seqlens_k = cu_seqlens_k
    elif cu_seqlens_k is not None:
        cu_seqlens_k = ast.literal_eval(cu_seqlens_k)

    if isinstance(seqused_q, int):
        seqused_q = [seqused_q]
    elif isinstance(seqused_q, list):
        seqused_q = seqused_q
    elif seqused_q is not None:
        seqused_q = ast.literal_eval(seqused_q)

    if isinstance(seqused_k, int):
        seqused_k = [seqused_k]
    elif isinstance(seqused_k, list):
        seqused_k = seqused_k
    elif seqused_k is not None:
        seqused_k = ast.literal_eval(seqused_k)

    if layout_query == "TND":
        if len(cu_seqlens_q) == batch_size + 1:
            cu_seqlens_q = cu_seqlens_q[1:]
    if layout_key == "TND":
        if len(cu_seqlens_k) == batch_size + 1:
            cu_seqlens_k = cu_seqlens_k[1:]

    npu_pass = True
    max_error = 0
    max_re = 0
    thres = 0.0001
    diff_thd = 0.01
    pct_thd = 0.005
    max_diff_hd = 0.1
    rtol = 0.005
    atol = 0.000025
    max_error_idx = 10000000
    cpu_output = expect.cpu().numpy()
    npu_output = result.cpu().numpy()
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()

    if layout_query in ["BSND"]:
        sp = (batch_size, q_seq, k_head_num)
        total_rows = batch_size * q_seq * k_head_num
    elif layout_query in ["TND"]:
        sp = (q_t_size, k_head_num)
        total_rows = q_t_size * k_head_num
    else:
        total_rows = 0
        sp = (0, 0)
    print(f"total_line is {total_rows}")
    npu_reshape = npu_output.reshape([total_rows, topk])
    cpu_reshape = cpu_output.reshape([total_rows, topk])
    start_time = time()
    invalid_data = cpu_reshape != -1
    valid_lens = invalid_data.sum(axis=-1)  # (total_rows,)
    npu_pass = judge_value_by_isclose(npu_reshape, cpu_reshape)
    end_time = time()
    print(f"耗时：{end_time - start_time:.6f} 秒")
    print(f"npu_pass is {npu_pass}")
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    diff_result = np.isclose(
        real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = np.where(diff_result != np.array((True,)))[0]
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]
    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if npu_pass else "Failed"
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, pct_thd, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd)
        )
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return result, fulfill_percent
