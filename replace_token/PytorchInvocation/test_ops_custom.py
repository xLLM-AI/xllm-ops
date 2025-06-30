#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu
# from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False


class TestCustomAdd():

    def test_add_custom(self):
        size = (5, 1)  # 直接用元组表示尺寸
        x = torch.randint(-4, 5, size, device='cpu', dtype=torch.int32)  # 区间[-4, 4]
        y = torch.randint(1, 5, size, device='cpu', dtype=torch.int32)
        print(x, '\n', y)

        torch.npu.synchronize()
        output = torch_npu.npu_replace_token(x.npu(), y.npu()).cpu()
        torch.npu.synchronize()

        print(output)
        # self.assertRtolEqual(output, x + y)


if __name__ == "__main__":
    # run_tests()
    test = TestCustomAdd()
    test.test_add_custom()
