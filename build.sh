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
    echo '  --build-transformer  Build ops-transformer third-party ops (default: enabled)'
    echo '  --no-build-transformer  Skip building ops-transformer third-party ops'
    echo '  --no-build-flash     Skip building flash-linear-attention-npu third-party ops'
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
    BUILD_TRANSFORMER="true"

    # 第一轮：解析 --key=value 形式参数
    for arg in "$@"; do
        case "$arg" in
            --build-dir=*)
                BUILD_DIR="${arg#*=}"
                ;;
            --compute-unit=*)
                COMPUTE_UNIT="${arg#*=}"
                ;;
            --build-transformer)
    BUILD_TRANSFORMER="true"
    BUILD_FLASH="true"
                ;;
            --no-build-transformer)
                BUILD_TRANSFORMER="false"
                ;;
            --no-build-flash)
                BUILD_FLASH="false"
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
    # SOC version mapping:
    #   soc <= 250            -> ascend910b   (A2 series)
    #   soc == 260            -> ascend950    (A5 series, e.g. Ascend910_95)
    #   otherwise (> 250)     -> ascend910_93 (A3 series, Ascend910C)
    ops=$(python3 -c "import torch; import torch_npu; soc = torch_npu._C._npu_get_soc_version(); ops = 'ascend910b' if soc <= 250 else ('ascend950' if soc == 260 else 'ascend910_93'); print(ops)")
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
# Catlass 兼容性补丁
#
# 背景：catlass 的 block_epilogue_dequant.hpp 使用 `AscendC::DT_FLOAT` 等写法，
#   但 DT_FLOAT / DT_FLOAT16 / DT_BF16 在 CANN 的 kernel_type.h 中是预处理宏
#   (#define DT_FLOAT 0 ...)，宏无法被命名空间限定，`AscendC::DT_FLOAT` 会被
#   预处理成 `AscendC::0`，编译报 "expected unqualified-id"。
#   此处在构建前就地剥离 AscendC:: 前缀（仅限 build.log 确认的单文件），幂等。
# ============================================================================
patch_catlass_compat() {
    local target="${BASE_DIR}/third_party/catlass/include/catlass/epilogue/block/block_epilogue_dequant.hpp"
    if [[ ! -f "${target}" ]]; then
        echo "[INFO] catlass patch: target not found, skip: ${target}"
        return 0
    fi

    # 仅替换 build.log 确认报错的三种模式，避免扩大修改范围
    # 注意：grep -c 无匹配时退出码为 1 且仍输出 "0"，故用 `|| true` 兜底（勿用 echo 0，
    # 否则会与 grep 的 "0" 拼成 "0\n0"，导致后续 [[ -eq ]] 解析失败）。
    local before
    before=$(grep -cE 'AscendC::DT_(FLOAT16|BF16|FLOAT)' "${target}" 2>/dev/null || true)
    before=${before:-0}
    if [[ "${before}" -eq 0 ]]; then
        echo "[INFO] catlass patch: already patched, skip"
        return 0
    fi

    # 顺序：长串优先，避免 AscendC::DT_FLOAT 误吃 AscendC::DT_FLOAT16
    sed -i \
        -e 's/AscendC::DT_FLOAT16\b/DT_FLOAT16/g' \
        -e 's/AscendC::DT_BF16\b/DT_BF16/g' \
        -e 's/AscendC::DT_FLOAT\b/DT_FLOAT/g' \
        "${target}"

    local after
    after=$(grep -cE 'AscendC::DT_(FLOAT16|BF16|FLOAT)' "${target}" 2>/dev/null || true)
    after=${after:-0}
    echo "[INFO] catlass patch: fixed $((before - after)) occurrence(s) in block_epilogue_dequant.hpp"
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

    # 修正 catlass 第三方依赖的宏不兼容问题（仅限单文件、幂等）
    patch_catlass_compat

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
# 编译 ops-transformer 三方依赖
# 通过 build_ops_transformer.sh 编译 third_party/ops-transformer 子模块,
# SOC 跟随 resolve_soc_version_list 的结果(逗号分隔多值)。
# ============================================================================
build_ops_transformer() {
    if [[ "${BUILD_TRANSFORMER}" != "true" ]]; then
        echo ">>> Skip ops-transformer build (--no-build-transformer)"
        return 0
    fi

    local transformer_script="${BASE_DIR}/build_ops_transformer.sh"
    if [[ ! -f "${transformer_script}" ]]; then
        echo "[WARN] build_ops_transformer.sh not found, skip ops-transformer build: ${transformer_script}"
        return 0
    fi

    # 子模块未初始化时 requirements.txt 缺失，会导致主 build 失败，提前检查并提示
    local transformer_requirements="${BASE_DIR}/third_party/ops-transformer/requirements.txt"
    if [[ ! -f "${transformer_requirements}" ]]; then
        echo "[ERROR] ops-transformer submodule not initialized (missing ${transformer_requirements})"
        echo "        Please run first: git submodule update --init --recursive"
        exit 1
    fi

    # 将 SOC_VERSION_LIST 数组转换为逗号分隔字符串
    local soc_str=""
    local soc_ver
    for soc_ver in "${SOC_VERSION_LIST[@]}"; do
        if [[ -z "${soc_str}" ]]; then
            soc_str="${soc_ver}"
        else
            soc_str="${soc_str},${soc_ver}"
        fi
    done

    # 310P (ascend310p) 目标无 quant_lightning_indexer_v2 等算子注册，
    # 无条件构建 transformer 会导致编译失败，故跳过并提示。
    local has_310p="false"
    for soc_ver in "${SOC_VERSION_LIST[@]}"; do
        if [[ "${soc_ver}" == "ascend310p" ]]; then
            has_310p="true"
            break
        fi
    done
    if [[ "${has_310p}" == "true" ]]; then
        echo ">>> Skip ops-transformer build: ascend310p has no quant_lightning_indexer_v2 ops registered"
        return 0
    fi

    echo ">>> Building ops-transformer for SOC=${soc_str}"
    stdbuf -oL bash "${transformer_script}" --soc="${soc_str}" all
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
        # 注意：只读取非注释行的 load_priority，避免误读注释行。
        local found_vendors
        found_vendors="$(grep -E '^[[:space:]]*load_priority[[:space:]]*=' "${config_file}" | cut --only-delimited -d'=' -f2- | head -n 1)"
        if echo ",${found_vendors}," | grep -q ",${vendor_name},"; then
            echo "[INFO] ${vendor_name} already in load_priority, skip"
        else
            if [ -z "${found_vendors}" ]; then
                # 追加而非截断：避免覆盖已有其它配置行
                echo "load_priority=${vendor_name}" >> "${config_file}"
            else
                # 用 awk 做字面量精确替换，避免 sed 分隔符/正则元字符(&、@、.等)被误解释
                awk -v old="${found_vendors}" -v new="${vendor_name},${found_vendors}" \
                    '{ gsub("load_priority=" old, "load_priority=" new); print }' \
                    "${config_file}" > "${config_file}.tmp" && mv "${config_file}.tmp" "${config_file}"
            fi
            echo "[INFO] updated ${config_file}"
        fi
    fi
}

# ============================================================================
# 构建 flash-linear-attention-npu 三方库
# ============================================================================
build_flash_linear_attention() {
    if [[ "${BUILD_FLASH}" != "true" ]]; then
        echo ">>> Skip flash-linear-attention-npu build (--no-build-flash)"
        return 0
    fi

    local fla_script="${BASE_DIR}/build_flash_linear_attention.sh"
    if [ ! -f "${fla_script}" ]; then
        echo "[WARN] flash-linear-attention-npu build script not found: ${fla_script}, skip"
        return 0
    fi

    # 从 build.sh 的 SOC 解析结果中取第一个芯片类型传给子脚本。
    # SOC_VERSION_LIST 由 run_build 中的 resolve_soc_version_list 填充（本函数在其后调用）。
    local soc="${SOC_VERSION_LIST[0]}"
    if [ -z "${soc}" ]; then
        echo "[WARN] SOC not resolved, flash-linear-attention-npu falls back to sub-script default ascend910_93"
        bash "${fla_script}"
        return 0
    fi

    echo "[INFO] build flash-linear-attention-npu (SOC=${soc})"
    bash "${fla_script}" --soc="${soc}"
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
    build_flash_linear_attention
    build_ops_transformer
}

main "$@"
