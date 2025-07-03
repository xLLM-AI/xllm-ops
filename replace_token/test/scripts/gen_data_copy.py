#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import sys
def gen_data(shape,forked_token_file_path,last_step_out_put_token_file_path,output_out_file_path):
    m = shape[0]
    forked_token = np.array([-1, -2, -3], dtype=np.int32)
    forked_token.tofile(forked_token_file_path)
    last_step_out_put_token = np.array([129649, 129649, 129649], dtype=np.int64)
    last_step_out_put_token.tofile(last_step_out_put_token_file_path)
    # print("forked_token:",forked_token)
    # print("last_step_out_put_token:",last_step_out_put_token)
    forked_token_copy = forked_token.copy()
    for i in range(shape[0]):
        if forked_token_copy[i] < 0:
            # print("forked_token_copy[i]:", forked_token_copy[i])
            # print("last_step_out_put_token[0-forked_token_copy[i]]:", last_step_out_put_token[0-forked_token_copy[i]-1])
            forked_token_copy[i] = last_step_out_put_token[0-forked_token_copy[i]-1]

    print("原始 forked_token:", forked_token)
    print("处理后的 forked_token_copy:", forked_token_copy)
    print("last_step_out_put_token:",last_step_out_put_token)
    forked_token_copy.tofile(output_out_file_path)
if __name__ == "__main__":
   
    gen_data((3,1),"./input/forked_token_ids.bin","./input/last_step_out_put_token_ids.bin","./input/output_out.bin")
