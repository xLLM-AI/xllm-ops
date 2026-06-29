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

export PYTHONUNBUFFERED=1
export VERBOSE=1
BASE_DIR=$(realpath "$(dirname "$0")")

# ============================================================================
# 帮助信息
# ============================================================================
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

# ============================================================================
# 参数解析
# ============================================================================
parse_args() {
    BUILD_DIR=""
    OP_NAME=""
    COMPUTE_UNIT=""

    # 第一轮：解析 --key=value 形式参数
    for arg in "$@"; do
        case "$arg" in
            --build-dir=*)
                BUILD_DIR="${arg#*=}"
                ;;
            --compute-unit=*)
                COMPUTE_UNIT="${arg#*=}"
                ;;
            -n|--op-name)
                # 需要取下一个参数，在第二轮中处理
                ;;
            -c|--compute-unit)
                # 需要取下一个参数，在第二轮中处理
                ;;
        esac
    done

    # 第二轮：使用位置索引处理 -n/-c 等需要取下一个参数的选项
    local idx=1
    while [[ $idx -le $# ]]; do
        local arg="${!idx}"
        if [[ "$arg" == "-n" || "$arg" == "--op-name" ]]; then
            local next_idx=$((idx + 1))
            if [[ $next_idx -le $# ]]; then
                OP_NAME="${!next_idx}"
            fi
        elif [[ "$arg" == "-c" || "$arg" == "--compute-unit" ]]; then
            local next_idx=$((idx + 1))
            if [[ $next_idx -le $# ]]; then
                COMPUTE_UNIT="${!next_idx}"
            fi
        fi
        idx=$((idx + 1))
    done

    # 解析 -h/--help
    for arg in "$@"; do
        case "$arg" in
            -h|--help)
                show_help
                exit 0
                ;;
        esac
    done
}

# ============================================================================
# 自动检测芯片类型
# 返回值通过全局变量 DETECTED_SOC_VERSION 传递
# ============================================================================
detect_soc_version() {
    if [ -n "${SOC_VERSION}" ]; then
        echo ">>> Using env SOC_VERSION: ${SOC_VERSION}"
        DETECTED_SOC_VERSION="${SOC_VERSION}"
        return
    fi

    local ops
    ops=$(python3 -c "import torch; import torch_npu; soc = torch_npu._C._npu_get_soc_version(); ops = 'ascend910b' if soc <= 250 else 'ascend910_93'; print(ops)")
    echo ">>> Auto-detected SOC version: ${ops}"
    export SOC_VERSION="${ops}"
    DETECTED_SOC_VERSION="${ops}"
}

# ============================================================================
# 确定 SOC_VERSION 列表
# 结果存入全局数组 SOC_VERSION_LIST
# ============================================================================
resolve_soc_version_list() {
    if [ -n "${COMPUTE_UNIT}" ]; then
        echo ">>> Using specified compute unit(s): ${COMPUTE_UNIT}"
        IFS=';' read -ra SOC_VERSION_LIST <<< "${COMPUTE_UNIT}"
    else
        detect_soc_version
        SOC_VERSION_LIST=("${DETECTED_SOC_VERSION}")
    fi
}

# ============================================================================
# 环境准备：设置编译器、清理打包产物
# ============================================================================
prepare_build_env() {
    # 默认值：未通过 --build-dir 指定时，优先复用 XLLM_OPS_BUILD_DIR 持久化缓存目录
    # （CI / 本地增量编译跨多次构建复用同一 build 目录），否则回落到源码树内的 build。
    if [ -z "${BUILD_DIR}" ]; then
        BUILD_DIR="${XLLM_OPS_BUILD_DIR:-${BASE_DIR}/build}"
    fi
    BUILD_DIR=$(realpath -m "$BUILD_DIR")
    OUTPUT_DIR="${BASE_DIR}/output"

    export CC=$(which gcc)
    export CXX=$(which g++)
    $CC --version
    $CXX --version

    # 保留 BUILD_DIR 以支持增量编译，仅清理打包产物
    rm -rf dist
}

# ============================================================================
# 执行编译：对每个芯片类型分别调用 build_aclnn.sh
# ============================================================================
run_build() {
    cd "${BASE_DIR}/xllm_ops"

    if [ -n "${OP_NAME}" ]; then
        echo ">>> Building specified operator(s): ${OP_NAME}"
    fi

    resolve_soc_version_list

    for soc_ver in "${SOC_VERSION_LIST[@]}"; do
        echo ">>> Building for SOC_VERSION=${soc_ver}"
        stdbuf -oL bash "${BASE_DIR}/xllm_ops/build_aclnn.sh" "${BASE_DIR}/xllm_ops" "$soc_ver" "$BUILD_DIR" "$OP_NAME"
    done
}

# ============================================================================
# 安装构建产物
# ============================================================================
install_artifacts() {
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
    fi

    cp "$BUILD_DIR"/cann-ops-xllm-custom*.run "$OUTPUT_DIR"/

    cd "$OUTPUT_DIR"
    local OPP_INSTALL_PATH="${ASCEND_HOME_PATH}/opp"
    local VENDOR_PATH="${OPP_INSTALL_PATH}/vendors/custom_xllm_math"
    if [ -d "${VENDOR_PATH}" ]; then
        # The generated installer exits under set -e if a new package adds a
        # top-level op_impl child that the old install tree did not have.
        echo "[INFO] remove stale custom_xllm_math vendor before reinstall"
        rm -rf "${VENDOR_PATH}"
    fi
    ./cann-ops-xllm-custom*.run --install-path="$OPP_INSTALL_PATH"
}

# ============================================================================
# Post-install: 补写 vendors/config.ini
#
# 背景：.run 包内置的 install.sh 在使用 --install-path 时，仅生成 bin/set_env.bash，
#       不会创建/更新 vendors/config.ini，导致默认 OPP 加载链找不到该自定义 vendor。
#       此处在外层手动补写 config.ini，使其包含本次安装的 vendor。
# ============================================================================
update_vendor_config() {
    local opp_install_path="${ASCEND_HOME_PATH}/opp"
    local vendor_dir_root="${opp_install_path}/vendors"
    local config_file="${vendor_dir_root}/config.ini"

    # 自动探测刚刚安装的 vendor 名字（取 vendors/ 下最新修改的子目录名）
    local vendor_name=""
    if [ -d "${vendor_dir_root}" ]; then
        vendor_name=$(ls -1t "${vendor_dir_root}" 2>/dev/null \
                      | while read d; do
                            [ -d "${vendor_dir_root}/$d" ] && echo "$d" && break
                        done)
    fi

    if [ -z "${vendor_name}" ]; then
        echo "[WARN] cannot detect installed vendor name under ${vendor_dir_root}, skip writing config.ini"
        return
    fi

    echo "[INFO] post-install: write config.ini for vendor '${vendor_name}'"

    if [ ! -f "${config_file}" ]; then
        echo "load_priority=${vendor_name}" > "${config_file}"
        chmod 640 "${config_file}" 2>/dev/null || true
        echo "[INFO] created ${config_file}"
    else
        # 已存在：把当前 vendor 合并进 load_priority（若不存在则前置加入）
        local found_vendors
        found_vendors="$(grep -w 'load_priority' "${config_file}" | cut --only-delimited -d'=' -f2-)"
        if echo ",${found_vendors}," | grep -q ",${vendor_name},"; then
            echo "[INFO] ${vendor_name} already in load_priority, skip"
        else
            if [ -z "${found_vendors}" ]; then
                echo "load_priority=${vendor_name}" > "${config_file}"
            else
                sed -i "s@load_priority=${found_vendors}@load_priority=${vendor_name},${found_vendors}@g" "${config_file}"
            fi
            echo "[INFO] updated ${config_file}"
        fi
    fi
}

# ============================================================================
# 主流程
# ============================================================================
main() {
    parse_args "$@"
    prepare_build_env
    run_build
    install_artifacts
    update_vendor_config
}

main "$@"
