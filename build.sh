#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

BASE_DIR=$(realpath "$(dirname "$0")")

# 显示帮助信息
show_help() {
    echo "Usage: bash build.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-dir=DIR     Specify build directory (default: build)"
    echo '  -n|--op-name NAME   Only build the specified operator(s), semicolon-separated'
    echo '  -c|--compute-unit   Specifies the chip type. If there are multiple values, separate'
    echo '                       them with semicolons and use quotation marks. Default: ascend910b'
    echo '                       e.g. -c "ascend910b" or -c "ascend910b;ascend310p"'
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash build.sh                                        # Build all operators"
    echo "  bash build.sh -n flash_attention_score_grad          # Build only flash_attention_score_grad"
    echo "  bash build.sh --op-name moe_grouped_matmul           # Build only moe_grouped_matmul"
    echo '  bash build.sh --op-name "op1;op2"                    # Build op1 and op2'
    echo '  bash build.sh -c "ascend910b;ascend310p"             # Build for multiple chip types'
}

# 解析参数
BUILD_DIR=""
OP_NAME=""
COMPUTE_UNIT=""
for arg in "$@"; do
    case "$arg" in
        --build-dir=*)
            BUILD_DIR="${arg#*=}"
            ;;
        --compute-unit=*)
            COMPUTE_UNIT="${arg#*=}"
            ;;
        -n|--op-name)
            # -n 和 --op-name 需要取下一个参数作为算子名，在下面的循环中处理
            ;;
        -c|--compute-unit)
            # -c 需要取下一个参数作为芯片类型，在下面的循环中处理
            ;;
    esac
done

# 使用位置索引处理 -n/--op-name 和 -c/--compute-unit 参数（需要取下一个参数）
idx=1
while [[ $idx -le $# ]]; do
    arg="${!idx}"
    if [[ "$arg" == "-n" || "$arg" == "--op-name" ]]; then
        next_idx=$((idx + 1))
        if [[ $next_idx -le $# ]]; then
            OP_NAME="${!next_idx}"
        fi
    elif [[ "$arg" == "-c" || "$arg" == "--compute-unit" ]]; then
        next_idx=$((idx + 1))
        if [[ $next_idx -le $# ]]; then
            COMPUTE_UNIT="${!next_idx}"
        fi
    fi
    idx=$((idx + 1))
done

# 解析 -h/--help 参数
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            show_help
            exit 0
            ;;
    esac
done

# 默认值：未通过 --build-dir 指定时，优先复用 XLLM_OPS_BUILD_DIR 持久化缓存目录
# （CI / 本地增量编译跨多次构建复用同一 build 目录），否则回落到源码树内的 build。
if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR="${XLLM_OPS_BUILD_DIR:-${BASE_DIR}/build}"
fi
BUILD_DIR=$(realpath -m "$BUILD_DIR")  # 规范化路径（不要求目录存在）
OUTPUT_DIR="${BASE_DIR}/output"

export CC=$(which gcc)
export CXX=$(which g++)

$CC --version
$CXX --version
# 保留 BUILD_DIR 以支持增量编译，仅清理打包产物
rm -rf dist

cd $BASE_DIR/xllm_ops
if [ -n "${OP_NAME}" ]; then
    echo ">>> Building specified operator(s): ${OP_NAME}"
fi

# 确定 SOC_VERSION 列表
if [ -n "${COMPUTE_UNIT}" ]; then
    # 用户通过 -c|--compute-unit 指定芯片类型
    echo ">>> Using specified compute unit(s): ${COMPUTE_UNIT}"
    IFS=';' read -ra SOC_VERSION_LIST <<< "${COMPUTE_UNIT}"
else
    # 自动检测芯片类型
    if [ -z "${SOC_VERSION}" ]; then
        ops=$(python3 -c "import torch; import torch_npu; soc = torch_npu._C._npu_get_soc_version(); ops = 'ascend910b' if soc <= 250 else 'ascend910_93'; print(ops)")
        echo "ascendc build ops: $ops"
        export SOC_VERSION=$ops
    else
        echo "ascendc build ops: $SOC_VERSION"
    fi
    SOC_VERSION_LIST=("${SOC_VERSION}")
fi

# 对每个芯片类型分别调用 build_aclnn.sh
for soc_ver in "${SOC_VERSION_LIST[@]}"; do
    echo ">>> Building for SOC_VERSION=${soc_ver}"
    bash $BASE_DIR/xllm_ops/build_aclnn.sh $BASE_DIR/xllm_ops "$soc_ver" "$BUILD_DIR" "$OP_NAME"
done

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

cp "$BUILD_DIR"/cann-ops-xllm-custom*.run "$OUTPUT_DIR"/

cd  "$OUTPUT_DIR"
OPP_INSTALL_PATH="${ASCEND_HOME_PATH}/opp"
./cann-ops-xllm-custom*.run --install-path="$OPP_INSTALL_PATH"

# -----------------------------------------------------------------------------
# Post-install: 补写 ${OPP_INSTALL_PATH}/vendors/config.ini
#
# 背景：.run 包内置的 install.sh 在使用 --install-path 时，仅生成 bin/set_env.bash，
#       不会创建/更新 vendors/config.ini，导致默认 OPP 加载链找不到该自定义 vendor。
#       此处在外层手动补写 config.ini，使其包含本次安装的 vendor。
# -----------------------------------------------------------------------------
VENDOR_DIR_ROOT="${OPP_INSTALL_PATH}/vendors"
CONFIG_FILE="${VENDOR_DIR_ROOT}/config.ini"

# 自动探测刚刚安装的 vendor 名字（取 vendors/ 下最新修改的子目录名）
if [ -d "${VENDOR_DIR_ROOT}" ]; then
    VENDOR_NAME=$(ls -1t "${VENDOR_DIR_ROOT}" 2>/dev/null \
                  | while read d; do
                        [ -d "${VENDOR_DIR_ROOT}/$d" ] && echo "$d" && break
                    done)
fi

if [ -z "${VENDOR_NAME}" ]; then
    echo "[WARN] cannot detect installed vendor name under ${VENDOR_DIR_ROOT}, skip writing config.ini"
else
    echo "[INFO] post-install: write config.ini for vendor '${VENDOR_NAME}'"
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "load_priority=${VENDOR_NAME}" > "${CONFIG_FILE}"
        chmod 640 "${CONFIG_FILE}" 2>/dev/null || true
        echo "[INFO] created ${CONFIG_FILE}"
    else
        # 已存在：把当前 vendor 合并进 load_priority（若不存在则前置加入）
        found_vendors="$(grep -w 'load_priority' "${CONFIG_FILE}" | cut --only-delimited -d'=' -f2-)"
        if echo ",${found_vendors}," | grep -q ",${VENDOR_NAME},"; then
            echo "[INFO] ${VENDOR_NAME} already in load_priority, skip"
        else
            if [ -z "${found_vendors}" ]; then
                echo "load_priority=${VENDOR_NAME}" > "${CONFIG_FILE}"
            else
                sed -i "s@load_priority=${found_vendors}@load_priority=${VENDOR_NAME},${found_vendors}@g" "${CONFIG_FILE}"
            fi
            echo "[INFO] updated ${CONFIG_FILE}"
        fi
    fi
fi
# -----------------------------------------------------------------------------

