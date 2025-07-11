#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import sys
def gen_data(m,j,forked_token_file_path,last_step_out_put_token_file_path,output_out_file_path):
    shape = (m,1)
    shape_last_step_out_put_token = (j,1)
    forked_token = np.random.randint(-(j),j,shape,dtype=np.int32)
    forked_token.tofile(forked_token_file_path)
    last_step_out_put_token = np.random.randint(0,j,shape_last_step_out_put_token,dtype=np.int64)
    last_step_out_put_token.tofile(last_step_out_put_token_file_path)
    # print("forked_token:",forked_token)
    # print("last_step_out_put_token:",last_step_out_put_token)
    forked_token_copy = forked_token.copy()
    print("forked_token_copy shape:",forked_token_copy.shape)
    for i in range(shape[0]):
        if forked_token_copy[i] < 0:
            # print("forked_token_copy[i]:", forked_token_copy[i])
            # print("last_step_out_put_token[0-forked_token_copy[i]]:", last_step_out_put_token[0-forked_token_copy[i]-1])
            forked_token_copy[i] = last_step_out_put_token[0-forked_token_copy[i]-1]

    # print("原始 forked_token:", forked_token)
    # print("处理后的 forked_token_copy:", forked_token_copy)
    # print("last_step_out_put_token:",last_step_out_put_token)
    forked_token_copy.tofile(output_out_file_path)
if __name__ == "__main__":
    m=int(sys.argv[1])
    j=int(sys.argv[2])
    gen_data(m,j,"./input/forked_token_ids.bin","./input/last_step_out_put_token_ids.bin","./input/output_out.bin")
