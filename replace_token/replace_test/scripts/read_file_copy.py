#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import argparse
from argparse import ArgumentParser

def gen_golden_data_simple(args_: argparse.Namespace):
    # input_a = np.random.uniform(1, 10, [16, 16]).astype(np.float32)
    groupsize = 256
    global args
    args = args_
    m = args.m
    n = args.n
    k = args.k
    weights_per_token =8
    input_a = np.random.uniform(1, 3, [m, k]).astype(np.int8)
    # input_a = np.ones((m, k), dtype=np.int8)
    sorted_list = np.zeros((m,  weights_per_token), dtype=np.int64)
    for i in range(m):
        # Each token selects 8 different weights from 256 weights
        sorted_list[i] = np.random.choice(groupsize, size=weights_per_token, replace=False)
    # Flatten sorted_list into a one-dimensional array
    sorted_list = sorted_list.flatten()

    # Sort by weight index
    token_idx = np.argsort(sorted_list)//8
    group_matmul_input_a = input_a[token_idx]
    print(group_matmul_input_a.shape)
    s =np.bincount(sorted_list,minlength=256)
    # print(s)
    group_offset = np.cumsum(s).astype(np.int64)
    print(group_offset)
    input_bs = np.zeros((256, k, n), dtype=np.int8)
    goldens = np.zeros((8*m,n),dtype=np.int32)
    preoffset =0
    firstout = 0
    scales = np.zeros((256,n),dtype=np.float16)
    
   
    # scales=float32_to_bf16(scales)
    per_token_scales = np.random.uniform(0,0.5,[m*weights_per_token]).astype(np.float32)
    print(per_token_scales)
    input_bias = np.zeros((groupsize,n),dtype=np.int32)
    for i in range(256):
        scale = np.random.uniform(0,0.5,[n]).astype(np.float16)
        tokens = s[i]
        matmul_a = input_a[token_idx[preoffset:preoffset+tokens]]
        # 将b的值设置为1
        # input_b = np.ones((k, n), dtype=np.int8)
        input_b = np.random.uniform(1, 3, [k, n]).astype(np.int8)
        input_bs[i]=input_b
        if tokens !=0:
            print(i,"\n",input_b)
            print(scale)
            print(per_token_scales)
        group_bias = np.random.uniform(1,3,[n]).astype(np.int32)
        golden = (np.matmul(matmul_a.astype(np.int32), input_b.astype(np.int32)) + group_bias).astype(np.int32)
        start_row = preoffset
        end_row = start_row + tokens
        goldens[start_row:end_row, :] = golden
        preoffset =group_offset[i]
        input_bias[i]=group_bias
        scales[i]=scale
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("output"):
        os.mkdir("output")
    scales.tofile("./input/scales.bin")
    # print("scales")
    # print(scales)
    # print("goldens int result32")
    print(goldens.shape)
    print(goldens)
    goldens.tofile("./input/golden_in.bin")
    # print("goldens")
    # print(goldens)
    preoffset =0
    # print("goldens shape:",goldens.shape)

    for j in range(256):
        tokens = s[j]
        if tokens==0:continue
        # print(j,"\n",scales[j])
        start_row = preoffset
        end_row = start_row + tokens
        # print("input :\n",goldens[start_row:end_row, :])
        # goldens[start_row:end_row, :] = (goldens[start_row:end_row, :].astype(np.float32)* scales[j])
        # print("output: \n",goldens[start_row:end_row, :])
        preoffset =group_offset[j]
    # print("ascendDequant res\n",goldens)
    per_token_scales = per_token_scales.reshape(-1, 1)
    # goldens = np.round(goldens * per_token_scales).astype(np.float16)
    # goldens=float32_to_bf16(goldens)
    
    
    input_a.tofile("./input/input_a.bin")
    input_bs.tofile("./input/input_bs.bin")
    input_bias.tofile("./input/input_bias.bin")
    # print("final goldens")
    # print(goldens)
    # print(m,n,k)
    # goldens.tofile("./output/golden.bin")
    # print("group_offset")
    # print(group_offset)
    # print("group_offset shape:",group_offset.shape)
    group_offset.tofile("./input/group_offset.bin")
    token_idx.tofile("./input/sorted_list.bin")
    group_matmul_input_a.tofile("./input/group_matmul_input_a.bin")
    
    per_token_scales.tofile("./input/per_token_scales.bin")


if __name__ == "__main__":
    parser = ArgumentParser(description="experts numbers")
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="The m size must be set",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4096,
        help="The m size must be set",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7168,
        help="The m size must be set",
    )
    args = parser.parse_args()
    gen_golden_data_simple(args)
