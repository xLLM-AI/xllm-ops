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

# 脚本所在目录（xllm-ops 仓库根目录）
BASE_DIR=$(realpath "$(dirname "$0")")
# flash-linear-attention-npu 子模块目录
FLA_DIR="${BASE_DIR}/third_party/flash-linear-attention-npu"

# ============================================================================
# 默认配置
# ============================================================================
# SOC 芯片类型（FLA_NPU_SOC）：ascend910b(A2) / ascend910_93(A3) / ascend950(A5)
SOC="${FLA_NPU_SOC:-ascend910_93}"
# 是否编译，默认编译（1）；--no-build 时跳过编译与安装
BUILD=1
# 是否安装 wheel，默认安装（1）；--no-install 时跳过安装
INSTALL=1
# 正式发布：默认关闭 local version，生成纯版本号（如 26.9.1）。
# 如需开发版本号（+branch.dev<commit>），设置 FLA_NPU_DISABLE_LOCAL_VERSION=FALSE。
FLA_NPU_DISABLE_LOCAL_VERSION="${FLA_NPU_DISABLE_LOCAL_VERSION:-TRUE}"
export FLA_NPU_DISABLE_LOCAL_VERSION
# 需要编译的算子名称（数组）。默认为 KDA 全部算子；置空表示编译全部算子。
# 通过 --ops=op1,op2,... 指定；也可用环境变量 NPU_OPS（逗号分隔）初始化。
NPU_OPS_DEFAULT=(
    chunk_kda_bwd
    chunk_kda_bwd_finalize
    chunk_kda_bwd_intra
    chunk_kda_bwd_prepare
    chunk_kda_bwd_recompute
    chunk_kda_fwd
    chunk_kda_fwd_finalize
    chunk_kda_fwd_prepare
    kda_gate_cumsum
    recurrent_kda
)
if [ -n "${NPU_OPS:-}" ]; then
    IFS=',' read -ra NPU_OPS <<< "${NPU_OPS}"
else
    NPU_OPS=("${NPU_OPS_DEFAULT[@]}")
fi

# ============================================================================
# 帮助信息
# ============================================================================
show_help() {
    echo "Usage: bash build_flash_linear_attention.sh [OPTIONS]"
    echo ""
    echo "安装依赖、编译并安装 flash-linear-attention-npu 三方库。"
    echo ""
    echo "Options:"
    echo "  --soc SOC           SOC 芯片类型（FLA_NPU_SOC），默认 ascend910_93"
    echo "                       A2 -> ascend910b, A3 -> ascend910_93, A5 -> ascend950"
    echo "  --no-build          跳过编译（不执行 build_wheel.py）"
    echo "  --no-install        编译后不安装 wheel"
    echo "  --skip-deps         跳过依赖安装（pip install -r requirements.txt）"
    echo "  --ops=op1,op2,...   指定要编译的算子（逗号分隔，snake 命名）"
    echo "                       默认编译 KDA 全部算子；--ops= 空值表示编译全部算子"
    echo "  -h, --help          显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  bash build_flash_linear_attention.sh                       # 默认 ascend910_93，安装依赖+编译+安装"
    echo "  bash build_flash_linear_attention.sh --soc ascend950       # 指定 A5 芯片"
    echo "  bash build_flash_linear_attention.sh --no-build            # 仅安装依赖，不编译"
    echo "  bash build_flash_linear_attention.sh --no-install          # 编译但不安装 wheel"
    echo "  bash build_flash_linear_attention.sh --skip-deps           # 跳过依赖安装"
}

# ============================================================================
# 参数解析
# ============================================================================
parse_args() {
    while [ "$#" -gt 0 ]; do
        arg="$1"
        case "$arg" in
            --soc=*)
                SOC="${arg#*=}"
                ;;
            --soc)
                shift
                SOC="$1"
                ;;
            --no-build)
                BUILD=0
                ;;
            --no-install)
                INSTALL=0
                ;;
            --skip-deps)
                SKIP_DEPS=1
                ;;
            --ops=*)
                IFS=',' read -ra NPU_OPS <<< "${arg#*=}"
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

# ============================================================================
# 依赖安装
# ============================================================================
install_deps() {
    if [ "${SKIP_DEPS:-0}" = "1" ]; then
        echo "[INFO] skip dependency installation (--skip-deps)"
        return 0
    fi
    echo "[INFO] install dependencies: pip install -r requirements.txt"
    python3 -m pip install -r "${FLA_DIR}/requirements.txt"
}

# ============================================================================
# 编译
# ============================================================================
build_fla() {
    local build_cmd=(python3 scripts/build_wheel.py)
    if [ "${#NPU_OPS[@]}" -gt 0 ]; then
        local ops_list
        ops_list=$(IFS=','; echo "${NPU_OPS[*]}")
        echo "[INFO] build flash-linear-attention-npu (FLA_NPU_SOC=${SOC}, ops=${ops_list})"
        build_cmd+=(--build-args="--ops=${ops_list}")
    else
        echo "[INFO] build flash-linear-attention-npu (FLA_NPU_SOC=${SOC}, all ops)"
    fi
    (
        cd "${FLA_DIR}"
        FLA_NPU_SOC="${SOC}" "${build_cmd[@]}"
    )
}

# ============================================================================
# 安装
# ============================================================================
install_fla() {
    local wheel
    wheel=$(ls -t "${FLA_DIR}/dist/"*.whl 2>/dev/null | head -n 1)
    if [ -z "${wheel}" ]; then
        echo "[ERROR] build artifact wheel not found, please build first"
        exit 1
    fi
    echo "[INFO] install wheel: ${wheel}"
    python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${wheel}"
}

# ============================================================================
# 主流程
# ============================================================================
main() {
    parse_args "$@"

    if [ ! -d "${FLA_DIR}" ]; then
        echo "[ERROR] submodule directory not found: ${FLA_DIR}"
        echo "        please init submodule first: git submodule update --init --recursive third_party/flash-linear-attention-npu"
        exit 1
    fi

    install_deps

    if [ "${BUILD}" = "1" ]; then
        build_fla
        if [ "${INSTALL}" = "1" ]; then
            install_fla
        fi
    else
        echo "[INFO] skip build (--no-build)"
    fi

    echo "[INFO] flash-linear-attention-npu processing done"
}

main "$@"