/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "mega_gdn_draft_decode_pto_kernel.h"

struct MegaGdnDraftDecodeTilingData {
  int64_t total_tokens;
  int64_t batch_size;
  int64_t num_k_heads;
  int64_t num_v_heads;
};

template <bool FlaSsmStateLayout>
AICORE PTO_INLINE void RunMegaGdnDraftDecode(
    GM_ADDR qkv,
    GM_ADDR z,
    GM_ADDR b,
    GM_ADDR a,
    GM_ADDR conv_weight,
    GM_ADDR conv_state,
    GM_ADDR a_log,
    GM_ADDR dt_bias,
    GM_ADDR ssm_state,
    GM_ADDR read_state_indices,
    GM_ADDR write_state_indices,
    GM_ADDR q_cu_seq_lens,
    GM_ADDR state_validity_mask,
    GM_ADDR norm_weight,
    GM_ADDR conv_out,
    GM_ADDR conv_state_out,
    GM_ADDR ssm_state_out,
    GM_ADDR out,
    const MegaGdnDraftDecodeTilingData& tiling_data
#if defined(PTO_NPU_ARCH_A5)
    , GM_ADDR soft_sync_workspace
#endif
    ) {
  mega_gdn_draft_decode_pto::Run<FlaSsmStateLayout>(
      reinterpret_cast<__gm__ bfloat16_t*>(qkv),
      reinterpret_cast<__gm__ bfloat16_t*>(z),
      reinterpret_cast<__gm__ bfloat16_t*>(b),
      reinterpret_cast<__gm__ bfloat16_t*>(a),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_weight),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_state),
      reinterpret_cast<__gm__ float*>(a_log),
      reinterpret_cast<__gm__ float*>(dt_bias),
      reinterpret_cast<__gm__ float*>(ssm_state),
      reinterpret_cast<__gm__ int*>(read_state_indices),
      reinterpret_cast<__gm__ int*>(write_state_indices),
      reinterpret_cast<__gm__ int*>(q_cu_seq_lens),
      reinterpret_cast<__gm__ uint8_t*>(state_validity_mask),
      reinterpret_cast<__gm__ bfloat16_t*>(norm_weight),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_out),
      reinterpret_cast<__gm__ bfloat16_t*>(conv_state_out),
      reinterpret_cast<__gm__ float*>(ssm_state_out),
      reinterpret_cast<__gm__ bfloat16_t*>(out),
      static_cast<int32_t>(tiling_data.total_tokens),
      static_cast<int32_t>(tiling_data.batch_size),
      static_cast<int32_t>(tiling_data.num_k_heads),
      static_cast<int32_t>(tiling_data.num_v_heads)
#if defined(PTO_NPU_ARCH_A5)
      , soft_sync_workspace
#endif
      );
}

extern "C" __global__ __aicore__ void mega_gdn_draft_decode(
    GM_ADDR qkv,
    GM_ADDR z,
    GM_ADDR b,
    GM_ADDR a,
    GM_ADDR conv_weight,
    GM_ADDR conv_state,
    GM_ADDR a_log,
    GM_ADDR dt_bias,
    GM_ADDR ssm_state,
    GM_ADDR read_state_indices,
    GM_ADDR write_state_indices,
    GM_ADDR q_cu_seq_lens,
    GM_ADDR state_validity_mask,
    GM_ADDR norm_weight,
    GM_ADDR conv_out,
    GM_ADDR conv_state_out,
    GM_ADDR ssm_state_out,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
#if defined(PTO_NPU_ARCH_A5)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#else
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
  REGISTER_TILING_DEFAULT(MegaGdnDraftDecodeTilingData);
  GET_TILING_DATA_WITH_STRUCT(MegaGdnDraftDecodeTilingData,
                              tiling_data,
                              tiling);
#if defined(PTO_NPU_ARCH_A5)
  GM_ADDR soft_sync_workspace = workspace;
#else
  (void)workspace;
#endif

  if constexpr (TILING_KEY_IS(1)) {
    RunMegaGdnDraftDecode<true>(qkv,
                                z,
                                b,
                                a,
                                conv_weight,
                                conv_state,
                                a_log,
                                dt_bias,
                                ssm_state,
                                read_state_indices,
                                write_state_indices,
                                q_cu_seq_lens,
                                state_validity_mask,
                                norm_weight,
                                conv_out,
                                conv_state_out,
                                ssm_state_out,
                                out,
                                tiling_data
#if defined(PTO_NPU_ARCH_A5)
                                , soft_sync_workspace
#endif
                                );
  } else if constexpr (TILING_KEY_IS(1001)) {
    RunMegaGdnDraftDecode<false>(qkv,
                                 z,
                                 b,
                                 a,
                                 conv_weight,
                                 conv_state,
                                 a_log,
                                 dt_bias,
                                 ssm_state,
                                 read_state_indices,
                                 write_state_indices,
                                 q_cu_seq_lens,
                                 state_validity_mask,
                                 norm_weight,
                                 conv_out,
                                 conv_state_out,
                                 ssm_state_out,
                                 out,
                                 tiling_data
#if defined(PTO_NPU_ARCH_A5)
                                 , soft_sync_workspace
#endif
                                 );
  }
}

// Keep this include last; generated mixed-kernel wrappers call
// matmul::clearWorkspace.
#include "lib/matmul_intf.h"
