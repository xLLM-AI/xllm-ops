#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2
BUILD_DIR_ARG=${3:-""}
OP_NAME_ARG=${4:-""}

log() {
    echo "[build_aclnn] $*"
}

find_op_dir() {
    local op_name=$1
    local dir
    for dir in \
        "${ROOT_DIR}/moe/${op_name}" \
        "${ROOT_DIR}/gmm/${op_name}" \
        "${ROOT_DIR}/attention/${op_name}" \
        "${ROOT_DIR}/mc2/${op_name}" \
        "${ROOT_DIR}/ffn/${op_name}" \
        "${ROOT_DIR}/${op_name}" \
        "${ROOT_DIR}/posembedding/${op_name}"; do
        if [[ -d "${dir}" ]]; then
            echo "${dir}"
            return 0
        fi
    done
    find "${ROOT_DIR}/" -maxdepth 3 -type d -name "${op_name}" -print -quit 2>/dev/null
}

get_cann_toolkit_version() {
    local version_file
    local version_line
    for version_file in \
        "${ASCEND_TOOLKIT_HOME}/toolkit/version.info" \
        "${ASCEND_TOOLKIT_HOME}/version.info" \
        "${ASCEND_HOME_PATH}/toolkit/version.info" \
        "${ASCEND_HOME_PATH}/version.info"; do
        if [[ -f "${version_file}" ]]; then
            version_line=$(grep -m1 '^Version=' "${version_file}" 2>/dev/null || true)
            if [[ -n "${version_line}" ]]; then
                echo "${version_line#Version=}" | tr -d '\r\n'
                return 0
            fi
        fi
    done
    return 1
}

# 如果指定了 OP_NAME_ARG，覆盖 CUSTOM_OPS_ARRAY 和 CUSTOM_OPS
if [[ -n "${OP_NAME_ARG}" ]]; then
    log "OP_NAME_ARG=${OP_NAME_ARG} specified, overriding default CUSTOM_OPS"
    # 支持分号分隔的多个算子名
    IFS=';' read -ra CUSTOM_OPS_ARRAY <<< "${OP_NAME_ARG}"
    CUSTOM_OPS="${OP_NAME_ARG}"
fi

dump_selected_ops() {
    local op_name
    local op_dir
    local kernel_cpp_count

    log "resolved SOC_ARG=${SOC_ARG}"
    log "resolved CUSTOM_OPS=${CUSTOM_OPS}"
    log "custom op count=${#CUSTOM_OPS_ARRAY[@]}"
    for op_name in "${CUSTOM_OPS_ARRAY[@]}"; do
        op_dir=$(find_op_dir "${op_name}")
        if [[ -z "${op_dir}" ]]; then
            log "op ${op_name}: dir=<missing>"
            continue
        fi
        kernel_cpp_count=0
        if [[ -d "${op_dir}/op_kernel" ]]; then
            kernel_cpp_count=$(find "${op_dir}/op_kernel" -maxdepth 1 -name '*.cpp' | wc -l | tr -d ' ')
        fi
        log "op ${op_name}: dir=${op_dir} cmake=$([[ -f "${op_dir}/CMakeLists.txt" ]] && echo yes || echo no) op_host_cmake=$([[ -f "${op_dir}/op_host/CMakeLists.txt" ]] && echo yes || echo no) op_kernel_cpp_count=${kernel_cpp_count}"
    done
}

log "start: ROOT_DIR=${ROOT_DIR:-<unset>} SOC_VERSION=${SOC_VERSION:-<unset>} cwd=$(pwd)"
log "env: ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>} ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    log "matched SOC branch: ascend310"
    # ASCEND310P series
    CUSTOM_OPS_ARRAY=(
        "causal_conv1d_v310"
        "recurrent_gated_delta_rule_v310"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend310p"
elif [[ "$SOC_VERSION" =~ ^(ascend)?910b ]]; then
    log "matched SOC branch: ascend910b"
    # ASCEND910B (A2) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/../third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}
    log "catlass include=${ABSOLUTE_CATLASS_PATH}"

    CUSTOM_OPS_ARRAY=(
        "moe_init_routing_custom"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "lightning_indexer_quant"
        "compressor"
        "quant_lightning_indexer"  ## 已在 CANN 中内置，见 opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/quant_lightning_indexer
        "quant_lightning_indexer_metadata"
        "lightning_indexer_quant_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
        "rms_norm_dynamic_quant"
        "inplace_partial_rotary_mul"
        "dispatch_ffn_combine"
        "dequant_swiglu_quant"  ## 已在 CANN 中内置，见 aarch64-linux/include/aclnnop/aclnn_dequant_swiglu_quant.h
        "scatter_nd_update_v2"

        ### JD's in-house operators ####
        "beam_search_group"
        "x_attention"
        "cache_unshared_kv"
        "causal_conv1d"
        "convert_kv_cache_format"
        "beam_search"
        "index_group_matmul"
        "moe_fused_add_topk"
        "moe_fused_reducesum_div"
        "moe_grouped_matmul"
        "moe_grouped_matmul_swiglu_quant"
        "multi_latent_attention"
        "pp_matmul_opt"
        "replace_token"
        "select_unshared_kv"
        "x_attention_tl"
        "x_flash_attention_infer"
        "onerec_final_beam_select"
        "rec_constrained_top_k"
        "mega_chunk_gdn"
        "laser_attention"
        "layer_norm_fwd"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    log "matched SOC branch: ascend910_93"
    # ASCEND910C (A3) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/../third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    # dependency: cann-toolkit file moe_distribute_base.h
    HCCL_STRUCT_FILE_PATH=$(find -L "${ASCEND_TOOLKIT_HOME}" -name "moe_distribute_base.h" 2>/dev/null | head -n1)
    if [ -z "$HCCL_STRUCT_FILE_PATH" ]; then
        echo "cannot find moe_distribute_base.h file in CANN env"
        exit 1
    fi
    # for dispatch_gmm_combine_decode
    yes | cp "${HCCL_STRUCT_FILE_PATH}" "${ROOT_DIR}/../utils/inc/kernel"
    # for dispatch_ffn_combine
    SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
    TARGET_DIR="$SCRIPT_DIR/mc2/dispatch_ffn_combine/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"

    echo "*************************************"
    echo $HCCL_STRUCT_FILE_PATH
    echo "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"

    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"
    # fix usages that still reference the original (un-renamed) symbols
    sed -i 's/using HcclOpParam = HcclOpResParam;/using HcclOpParam = HcclOpResParamCustom;/g' "$TARGET_FILE"
    sed -i 's/(HcclRankRelationResV2 \*)/(HcclRankRelationResV2Custom *)/g' "$TARGET_FILE"

    TARGET_DIR="$SCRIPT_DIR/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"
    sed -i 's/using HcclOpParam = HcclOpResParam;/using HcclOpParam = HcclOpResParamCustom;/g' "$TARGET_FILE"
    sed -i 's/(HcclRankRelationResV2 \*)/(HcclRankRelationResV2Custom *)/g' "$TARGET_FILE"

    TARGET_DIR="$SCRIPT_DIR/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"
    sed -i 's/using HcclOpParam = HcclOpResParam;/using HcclOpParam = HcclOpResParamCustom;/g' "$TARGET_FILE"
    sed -i 's/(HcclRankRelationResV2 \*)/(HcclRankRelationResV2Custom *)/g' "$TARGET_FILE"

    # for dispatch_normal and combine_normal
    TARGET_DIR="$SCRIPT_DIR/mc2/moe_dispatch_normal/op_kernel/utils/"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"

    TARGET_DIR="$SCRIPT_DIR/mc2/moe_combine_normal/op_kernel/utils/"
    echo "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
    
    CUSTOM_OPS_ARRAY=(
        "moe_init_routing_custom"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "lightning_indexer_quant"
        "lightning_indexer_quant_metadata"
        "compressor"
        "quant_lightning_indexer"  ## 已在 CANN 中内置，见 opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/quant_lightning_indexer
        "quant_lightning_indexer_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
        "rms_norm_dynamic_quant"
        "inplace_partial_rotary_mul"
        "dispatch_ffn_combine"
        "dequant_swiglu_quant"  ## 已在 CANN 中内置，见 aarch64-linux/include/aclnnop/aclnn_dequant_swiglu_quant.h
        "scatter_nd_update_v2"

         ### JD's in-house operators ####
        "beam_search_group"
        "x_attention"
        "cache_unshared_kv"
        "causal_conv1d"
        "convert_kv_cache_format"
        "beam_search"
        "index_group_matmul"
        "moe_fused_add_topk"
        "moe_fused_reducesum_div"
        "moe_grouped_matmul"
        "moe_grouped_matmul_swiglu_quant"
        "multi_latent_attention"
        "pp_matmul_opt"
        "replace_token"
        "select_unshared_kv"
        "x_attention_tl"
        "x_flash_attention_infer"
        "onerec_final_beam_select"
        "rec_constrained_top_k"
        "laser_attention"
        "layer_norm_fwd"
    )
    CANN_TOOLKIT_VERSION="$(get_cann_toolkit_version || true)"
    if [[ "${CANN_TOOLKIT_VERSION}" =~ ^([0-9]+)\.([0-9]+) ]]; then
        if [[ "${BASH_REMATCH[1]}" -ge 9 ]]; then
            CUSTOM_OPS_ARRAY+=("mega_chunk_gdn")
        else
            log "skip mega_chunk_gdn for CANN version ${CANN_TOOLKIT_VERSION}"
        fi
    else
        log "skip mega_chunk_gdn because CANN version is unavailable"
    fi
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
elif [[ "$SOC_VERSION" =~ ^ascend950 ]]; then
    log "matched SOC branch: ascend950"
    # ASCEND950 (A5) series, real compile arch: ascend910_95
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/../third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}
    log "catlass include=${ABSOLUTE_CATLASS_PATH}"

    CUSTOM_OPS_ARRAY=(
        "moe_init_routing_custom"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "lightning_indexer_quant" 
        "lightning_indexer_quant_metadata"
        "compressor"
        "quant_lightning_indexer"  ## 已在 CANN 中内置，见 opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/quant_lightning_indexer
        "quant_lightning_indexer_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
        "rms_norm_dynamic_quant"
        "inplace_partial_rotary_mul"
        "dispatch_ffn_combine"
        "dequant_swiglu_quant"  ## 已在 CANN 中内置，删除后会有精度问题，CANN内置见 aarch64-linux/include/aclnnop/aclnn_dequant_swiglu_quant.h
        "scatter_nd_update_v2"

        #  ### JD's in-house operators ####
        "beam_search_group"
        "x_attention"
        "cache_unshared_kv"
        "causal_conv1d"
        "convert_kv_cache_format"
        "beam_search"
        "index_group_matmul"
        "moe_fused_add_topk"
        "moe_fused_reducesum_div"
        "moe_grouped_matmul"
        "moe_grouped_matmul_swiglu_quant"
        "pp_matmul_opt"
        "replace_token"
        "select_unshared_kv"
        # "x_attention_tl"  # A5 kernel not adapted
        "x_flash_attention_infer"
        "onerec_final_beam_select"
        "rec_constrained_top_k"
        "laser_attention"
        "mega_chunk_gdn"  # A5 kernel not adapted: pto-isa MrgSort/Stride errors
        "layer_norm_fwd"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend950"
else
    # others
    # currently, no custom aclnn ops for other series
    log "no custom ACLNN ops configured for SOC_VERSION=${SOC_VERSION}; skip build_aclnn"
    exit 0
fi

# 如果指定了 OP_NAME_ARG，覆盖 CUSTOM_OPS_ARRAY 和 CUSTOM_OPS
if [[ -n "${OP_NAME_ARG}" ]]; then
    log "OP_NAME_ARG=${OP_NAME_ARG} specified, overriding default CUSTOM_OPS"
    # 支持分号分隔的多个算子名
    IFS=';' read -ra CUSTOM_OPS_ARRAY <<< "${OP_NAME_ARG}"
    CUSTOM_OPS="${OP_NAME_ARG}"
fi

dump_selected_ops

# # build custom ops
# cd csrc
# rm -rf build output build_out
# echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
# bash build.sh --pkg --ops="$CUSTOM_OPS" --soc="$SOC_ARG"

# # install custom ops to vllm_ascend/_cann_ops_custom
# ./build/cann-ops-xllm*.run --install-path=$ROOT_DIR/output/_cann_ops_custom


(
  set -euo pipefail
  exec > >(stdbuf -oL cat)

  log "subshell cwd before cd=$(pwd)"
  #cd csrc
  log "subshell cwd after cd=$(pwd)"
  log "cleaning xllm_ops output dirs (keep build dir for incremental compilation)"
  rm -rf -- output build_out

  : "${ROOT_DIR:?ROOT_DIR is not set}"
  : "${CUSTOM_OPS:?CUSTOM_OPS is not set}"
  : "${SOC_VERSION:?SOC_VERSION is not set}"
  : "${SOC_ARG:?SOC_ARG is not set}"

  log "build command: bash build.sh --pkg --ops=\"${CUSTOM_OPS}\" --soc=\"${SOC_ARG}\""
  log "building custom ops ${CUSTOM_OPS} for ${SOC_VERSION}"
  if [ -n "$BUILD_DIR_ARG" ]; then
    bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}" --build-dir="${BUILD_DIR_ARG}"
  else
    bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"
  fi
  log "build.sh finished"

  install_dir="${ROOT_DIR}/output/_cann_ops_custom"
  log "install_dir=${install_dir}"

  mkdir -p -- "$install_dir"

  # 删除 install_dir 下除 .gitkeep 外的所有内容（包含隐藏文件/目录）
  find "$install_dir" -mindepth 1 -maxdepth 1 \
    ! -name '.gitkeep' \
    -exec rm -rf -- {} +

  shopt -s nullglob
  if [ -n "$BUILD_DIR_ARG" ]; then
    runs=("$BUILD_DIR_ARG"/cann-ops-xllm*.run)
  else
    runs=(./build/cann-ops-xllm*.run)
  fi
  shopt -u nullglob

  log "installer candidate count=${#runs[@]}"
  for run_file in "${runs[@]}"; do
    log "installer candidate: $(ls -lh "${run_file}")"
  done

  (( ${#runs[@]} == 1 )) || { echo "ERROR: expected 1 installer, got ${#runs[@]}" >&2; exit 1; }

  chmod +x -- "${runs[0]}" || true
  log "running installer: ${runs[0]}"
#   "${runs[0]}" --install-path="${install_dir}"
  log "installer finished"
  log "installed files under ${install_dir} (maxdepth=4, first 120 entries):"
  { find "${install_dir}" -mindepth 1 -maxdepth 4 -print | sort | head -n 120 | sed 's#^#[build_aclnn] install: #'; } || true
)
