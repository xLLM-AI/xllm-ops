#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ====================== 配置区======================
# 需要读取的用例excel表格路径，如下：
PATH1="./excel/*"
# 用例pt的文件存放路径，如下：
PATH2="./pt_path"

# 脚本路径
LIV2_PT_SAVE_SCRIPT="./batch/lightning_indexer_v2_pt_save.py"
TEST_LIV2_BATCH_SCRIPT="test_lightning_indexer_v2_batch.py"
REPLACE_PATH_SCRIPT="./batch/replace_path.py"
TEST_LIV2_SINGLE_SCRIPT="test_lightning_indexer_v2_single.py"

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_LIV2_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 用例批量生成调试
# 用法: run_batch [eager|graph]
#   eager - 直接调用算子（默认）
#   graph - 通过 torch.compile + torchair 后端执行
run_batch() {
    local run_mode="${1:-eager}"
    echo "===== 执行用例批量生成测试 (模式: ${run_mode}) ====="

    echo -e "\n===== 第一步：执行lightning_indexer_v2_pt_save.py ====="
    python3 $LIV2_PT_SAVE_SCRIPT $PATH1 $PATH2
    if [ $? -ne 0 ]; then
        echo "lightning_indexer_v2_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：替换test_lightning_indexer_v2_batch.py中的路径 ====="
    python3 $REPLACE_PATH_SCRIPT $TEST_LIV2_BATCH_SCRIPT $PATH2
    if [ $? -ne 0 ]; then
        echo "替换路径失败，退出"
        exit 1
    fi

    echo -e "\n===== 第三步：执行pytest命令 (LIV2_RUN_MODE=${run_mode}) ====="
    LIV2_RUN_MODE="${run_mode}" python3 -m pytest -rA -s $TEST_LIV2_BATCH_SCRIPT -v -m ci
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    cp test_lightning_indexer_v2_batch.py.bak test_lightning_indexer_v2_batch.py

    echo -e "\n=====执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <command> [run_mode]"
    echo "命令说明："
    echo "  single              执行单算子用例调测"
    echo "  batch [eager|graph] 执行用例批量生成调试"
    echo "                        eager - 直接调用算子（默认）"
    echo "                        graph - torch.compile + torchair 后端"
    echo "  help                显示本帮助信息"
    echo "示例："
    echo "  $0 single        # 执行single模式"
    echo "  $0 batch         # 执行batch模式（默认eager）"
    echo "  $0 batch eager   # 显式指定eager模式"
    echo "  $0 batch graph   # 使用graph模式"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -lt 1 ]; then
 	     echo "错误：必须传入至少一个参数"
    show_help
    exit 1
fi

COMMAND="$1"
RUN_MODE="${2:-eager}"

# 根据参数执行对应函数
case "$COMMAND" in
    single)
        run_single
        ;;
    batch)
        if [[ "$RUN_MODE" != "eager" && "$RUN_MODE" != "graph" ]]; then
            echo "错误：batch 模式仅支持 eager/graph，当前值: $RUN_MODE"
            show_help
            exit 1
        fi
        run_batch "$RUN_MODE"
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知命令 '$COMMAND'"
        show_help
        exit 1
        ;;
esac

exit 0