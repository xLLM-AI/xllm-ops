#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import argparse
import os
import shutil
import glob


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dynamic-dir", help="source compile build dir."
    )
    parser.add_argument(
        "-s", "--src-dir", help="source compile source dir."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()
    src_dir = os.path.realpath(args.src_dir)
    os.makedirs(args.dynamic_dir, exist_ok=True)
    dynamic_dir = os.path.realpath(args.dynamic_dir)
    header_files = glob.glob(os.path.join(src_dir, "*.h"))
    source_files = glob.glob(os.path.join(src_dir, "*.cpp"))
    for header_file in header_files:
        shutil.copy(header_file, dynamic_dir)
    for source_file in source_files:
        shutil.copy(source_file, dynamic_dir)
