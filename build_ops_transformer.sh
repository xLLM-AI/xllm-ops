#!/usr/bin/env bash
# =============================================================================
# build_ops_transformer.sh
# 编译 ops-transformer 三方依赖的构建脚本
#
# 主要功能:
#   1. 安装依赖包: 读取 third_party/ops-transformer/requirements.txt 并 pip 安装
#   2. 执行编译:   bash build.sh --pkg --soc={soc} --vendor_name={vendor_name}
#                  --ops={ops} -j{jobs} -O3
#   3. 安装产物:   将编译生成的 *.run 包安装到 ${ASCEND_HOME_PATH}/opp，
#                  并补写 vendors/config.ini（install.sh 在 --install-path 下不写 config.ini）
#
# 用法:
#   bash build_ops_transformer.sh            # 安装依赖 + 编译 + 安装产物 (默认)
#   bash build_ops_transformer.sh install    # 安装依赖 + 安装产物
#   bash build_ops_transformer.sh build      # 仅执行编译
#   bash build_ops_transformer.sh all        # 安装依赖 + 编译 + 安装产物
#   bash build_ops_transformer.sh --soc=ascend950 build   # 指定 SOC 编译
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# 常量定义区 (按需修改)
# -----------------------------------------------------------------------------

# 待编译的算子列表 (数组格式, 使用 snake name)
OPS=(
    quant_lightning_indexer_v2
    quant_lightning_indexer_v2_metadata
    sparse_flash_mla
    sparse_flash_mla_metadata
    mega_moe
    key_pool
    pool_key_indexer
    sparse_flash_attention
)

# 目标 Ascend SoC 芯片型号 (可选: ascend910b / ascend910_93 / ascend950 / ascend960dt 等)
# 可通过 --soc=xxx 参数覆盖, 未指定时使用此默认值
SOC="ascend910_93"

# 自定义算子包 vendor 名称 (编译参数 --vendor_name 使用, CMake 会追加 _transformer 后缀)
VENDOR_NAME="custom"

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 32)"
if [[ "${JOBS}" -lt 32 ]]; then
    JOBS=32
fi

# 编译优化级别 (默认 O3)
OPT_LEVEL="O3"

# 是否将 torch_extension 打包进 .run 安装包 (默认开启)
# 开启时: 先构建 whl 到 torch_extension/dist/, 再由 --pkg 打包进同一个 .run 包
BUILD_TORCH_EXTENSION="true"
# Whether to install build artifacts (default: enabled); --no-install skips it.
INSTALL_ARTIFACTS="true"

# -----------------------------------------------------------------------------
# 路径定义
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_TRANSFORMER_DIR="${SCRIPT_DIR}/third_party/ops-transformer"
REQUIREMENTS_FILE="${OPS_TRANSFORMER_DIR}/requirements.txt"
TORCH_EXTENSION_REQUIREMENTS_FILE="${OPS_TRANSFORMER_DIR}/torch_extension/requirements.txt"

# -----------------------------------------------------------------------------
# 函数: 安装依赖包
# -----------------------------------------------------------------------------
install_deps() {
    echo "============================================================"
    echo "[Step 1/3] Install dependencies"
    echo "============================================================"

    if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
        echo "Error: dependency file not found: ${REQUIREMENTS_FILE}"
        exit 1
    fi

    echo "Reading dependency file: ${REQUIREMENTS_FILE}"
    echo "----------------------------------------"
    cat "${REQUIREMENTS_FILE}"
    echo "----------------------------------------"
    echo "Installing dependencies..."
    pip install -r "${REQUIREMENTS_FILE}"
    echo "Dependencies installed."
    echo ""
}

# -----------------------------------------------------------------------------
# 函数: 执行编译
# -----------------------------------------------------------------------------
build_ops() {
    echo "============================================================"
    echo "[Step 2/3] Build"
    echo "============================================================"

    # 310P (ascend310p) 目标无 quant_lightning_indexer_v2 等算子注册，跳过编译
    if [[ "${SOC}" == "ascend310p" || "${SOC}" == *",ascend310p"* || "${SOC}" == "ascend310p,"* ]]; then
        echo "[WARN] SOC=${SOC} contains ascend310p, no quant_lightning_indexer_v2 ops registered, skip build"
        return 0
    fi

    if [[ ! -d "${OPS_TRANSFORMER_DIR}" ]]; then
        echo "Error: ops-transformer directory not found: ${OPS_TRANSFORMER_DIR}"
        echo "Please run 'git submodule update --init --recursive' first to pull the submodule."
        exit 1
    fi

    # 将 OPS 数组转换为逗号分隔字符串
    local ops_str=""
    local op
    for op in "${OPS[@]}"; do
        if [[ -z "${ops_str}" ]]; then
            ops_str="${op}"
        else
            ops_str="${ops_str},${op}"
        fi
    done

    local build_cmd="bash build.sh --pkg --soc=${SOC} --vendor_name=${VENDOR_NAME} --ops=${ops_str} -j${JOBS} -${OPT_LEVEL}"

    echo "Build directory: ${OPS_TRANSFORMER_DIR}"
    echo "Build command:"
    echo "  ${build_cmd}"
    echo ""

    cd "${OPS_TRANSFORMER_DIR}"

    # 若关闭 torch_extension 打包，清理 dist/ 残留的旧 whl，避免 --pkg 误打包
    if [[ "${BUILD_TORCH_EXTENSION}" != "true" ]]; then
        if [[ -d "${OPS_TRANSFORMER_DIR}/torch_extension/dist" ]]; then
            echo "[INFO] --no-torch-extension: clean stale whl in torch_extension/dist/"
            rm -rf "${OPS_TRANSFORMER_DIR}/torch_extension/dist"
        fi
    fi

    # 若开启 torch_extension 打包, 先单独构建 whl 到 torch_extension/dist/
    # 随后 --pkg 打包时 package.cmake 会自动把 dist 里的 whl 一并打进 .run 包
    if [[ "${BUILD_TORCH_EXTENSION}" == "true" ]]; then
        # 构建 whl 依赖 python 'build' 模块, 缺失时自动安装 torch_extension 依赖
        if ! python3 -c "import build" 2>/dev/null; then
            echo "[INFO] Missing python 'build' module, install torch_extension dependencies..."
            if [[ -f "${TORCH_EXTENSION_REQUIREMENTS_FILE}" ]]; then
                pip install -r "${TORCH_EXTENSION_REQUIREMENTS_FILE}" || { cd "${SCRIPT_DIR}"; exit 1; }
            else
                echo "Error: file not found: ${TORCH_EXTENSION_REQUIREMENTS_FILE}"
                cd "${SCRIPT_DIR}"
                exit 1
            fi
            echo ""
        fi

        echo "[INFO] Build torch_extension whl:"
        echo "  bash build.sh --torch_extension_only --vendor_name=${VENDOR_NAME} --ops=${ops_str}"
        echo ""
        bash build.sh --torch_extension_only --vendor_name="${VENDOR_NAME}" --ops="${ops_str}" || { cd "${SCRIPT_DIR}"; exit 1; }
        echo ""
    fi

    bash build.sh --pkg --soc="${SOC}" --vendor_name="${VENDOR_NAME}" --ops="${ops_str}" -j"${JOBS}" -"${OPT_LEVEL}"
    cd "${SCRIPT_DIR}"

    echo "Build completed."
    echo ""
}

# -----------------------------------------------------------------------------
# 函数: 安装编译产物 (.run 包)
# 将 third_party/ops-transformer/build_out/*.run 安装到 ${ASCEND_HOME_PATH}/opp
# -----------------------------------------------------------------------------
install_package() {
    echo "============================================================"
    echo "[Step 3/3] Install build artifacts"
    echo "============================================================"

    if [ -z "${ASCEND_HOME_PATH}" ]; then
        echo "Error: environment variable ASCEND_HOME_PATH is not set, cannot determine install path."
        exit 1
    fi

    local build_out_dir="${OPS_TRANSFORMER_DIR}/build_out"
    if [[ ! -d "${build_out_dir}" ]]; then
        echo "Error: build output directory not found: ${build_out_dir}"
        exit 1
    fi

    local run_files=("${build_out_dir}"/*.run)
    if [[ ${#run_files[@]} -eq 0 ]]; then
        echo "Error: no .run package found under directory ${build_out_dir}"
        exit 1
    fi
    if [[ ${#run_files[@]} -gt 1 ]]; then
        echo "Error: multiple .run packages found under directory ${build_out_dir}:"
        printf '%s\n' "${run_files[@]}"
        exit 1
    fi

    local run_file="${run_files[0]}"

    # 显式指定 --install-path：当 ASCEND_CUSTOM_OPP_PATH 含冒号(多路径)时，
    # .run 包内置 install.sh 会拒绝自动推断安装路径并要求显式指定。
    local install_path="${ASCEND_HOME_PATH}/opp"

    echo "Install package: ${run_file}"
    echo "Install path: ${install_path}"
    echo ""

    bash "${run_file}" --quiet --install-path="${install_path}"

    # 补写 config.ini：install.sh 在显式 --install-path 时只写 set_env.bash，
    # 不会更新 vendors/config.ini，需在此手动把 custom_transformer 加入 load_priority。
    update_vendor_config "${install_path}"

    echo "Artifacts installed (vendor directory installed, config.ini written)."
    echo ""
}

install_artifacts() {
    if [[ "${INSTALL_ARTIFACTS}" == "true" ]]; then
        install_package
    else
        echo "[INFO] skip artifact installation (--no-install)"
    fi
}

# -----------------------------------------------------------------------------
# 函数: 补写 vendors/config.ini
#
# 背景：.run 包内置 install.sh 在使用 --install-path 时，仅生成 bin/set_env.bash，
#       不会创建/更新 vendors/config.ini，导致默认 OPP 加载链找不到该自定义 vendor。
#       此处在外层手动补写 config.ini，使其包含本次安装的 vendor。
# -----------------------------------------------------------------------------
update_vendor_config() {
    local opp_install_path="$1"
    local vendor_dir_root="${opp_install_path}/vendors"
    local config_file="${vendor_dir_root}/config.ini"

    # vendor 目录名：CMake 打包时会将 install.sh 中的 vendor_name 替换为
    # ${VENDOR_NAME}_transformer（即 custom_transformer）
    local vendor_name="${VENDOR_NAME}_transformer"

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

# -----------------------------------------------------------------------------
# 解析参数: 支持 --soc=xxx 覆盖默认 SOC, 其余位置参数作为 action
# -----------------------------------------------------------------------------
parse_args() {
    ACTION="all"

    for arg in "$@"; do
        case "${arg}" in
            --soc=*)
                SOC="${arg#*=}"
                ;;
            --soc)
                # 需要取下一个参数, 在第二轮中处理
                ;;
            --torch-extension)
                BUILD_TORCH_EXTENSION="true"
                ;;
            --no-torch-extension)
                BUILD_TORCH_EXTENSION="false"
                ;;
            --no-install)
                INSTALL_ARTIFACTS="false"
                ;;
            --install)
                INSTALL_ARTIFACTS="true"
                ;;
            install|build|all)
                ACTION="${arg}"
                ;;
            *)
                echo "Error: unknown argument '${arg}'" >&2
                ACTION="__invalid__"
                ;;
        esac
    done

    # 第二轮: 处理 --soc 后跟空格分隔值的形式
    local idx=1
    while [[ $idx -le $# ]]; do
        local arg="${!idx}"
        if [[ "${arg}" == "--soc" ]]; then
            local next_idx=$((idx + 1))
            if [[ $next_idx -le $# ]]; then
                SOC="${!next_idx}"
            fi
        fi
        idx=$((idx + 1))
    done
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"
    local action="${ACTION}"

    # 将 OPS 数组转换为逗号分隔字符串用于显示
    local ops_str=""
    local op
    for op in "${OPS[@]}"; do
        if [[ -z "${ops_str}" ]]; then
            ops_str="${op}"
        else
            ops_str="${ops_str},${op}"
        fi
    done

    echo "============================================================"
    echo "ops-transformer build script"
    echo "  SOC          : ${SOC}"
    echo "  VENDOR_NAME  : ${VENDOR_NAME}"
    echo "  OPS          : ${ops_str}"
    echo "  JOBS         : ${JOBS}"
    echo "  OPT_LEVEL    : ${OPT_LEVEL}"
    echo "============================================================"
    echo ""

    case "${action}" in
        install)
            install_deps
            install_artifacts
            ;;
        build)
            build_ops
            ;;
        all)
            install_deps
            build_ops
            install_artifacts
            ;;
        *)
            echo "Usage: bash build_ops_transformer.sh [install|build|all] [--soc=xxx] [--torch-extension|--no-torch-extension] [--no-install]"
            echo "  install  Install dependencies and install build artifacts"
            echo "  build    Build only"
            echo "  all      Install dependencies + build + install artifacts (default)"
            echo "  --torch-extension      Package torch_extension whl into the .run package (enabled by default)"
            echo "  --no-torch-extension   Do not package torch_extension whl"
            echo "  --no-install           Do not install built artifacts"
            exit 1
            ;;
    esac

    echo "All done."
}

main "$@"