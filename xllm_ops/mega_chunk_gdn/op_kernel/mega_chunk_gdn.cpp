// mega_kernel.cpp — GDN Mega-Kernel (group-value / GQA): all PTO stages in one launch
//
// Same pipeline as pto_mega_kernel, but H/Hg are runtime values.
//
// Stages:
//   1. cumsum      (Vec)
//   2. transpose   (Vec)
//   3. kkt         (Cube+Vec)  — K has Hg heads; β,g,A use H value heads
//   4. solve_tril  (Cube)
//   5. wy_fast     (Vec+Cube)
//   6. chunk_h     (Cube+Vec)
//   7. chunk_o     (Cube+Vec)

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
#ifndef GDN_MAX_HEADS
#define GDN_MAX_HEADS 64
#endif
#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#ifndef GDN_KERNEL_NAME
#define GDN_KERNEL_NAME launch_mega_kernel
#endif
// Note the codegen parser does not support arguments of form "type *name", only "type* name"
// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format off

#include "acl/acl.h"
#include "kernel_operator.h"
#include <pto/pto-inst.hpp>
#include "gdn_sync.h"
#include <type_traits>
using namespace pto;

static_assert(std::is_same_v<DTYPE_Q, half> || std::is_same_v<DTYPE_Q, bfloat16_t>,
              "MegaChunkGdn supports FP16 or BF16 input tensors.");

using GdnInputT = DTYPE_Q;
#if defined(PTO_NPU_ARCH_A5)
using GdnComputeT = std::conditional_t<std::is_same_v<GdnInputT, bfloat16_t>, half, GdnInputT>;
#else
using GdnComputeT = GdnInputT;
#endif
using GdnOutputT = DTYPE_Q;

static_assert(sizeof(GdnInputT) == sizeof(GdnComputeT) && sizeof(GdnComputeT) == sizeof(GdnOutputT),
              "MegaChunkGdn requires equally sized input, compute, and output storage types.");

struct MegaChunkGdnKernelTilingData {
    uint32_t block_dim;
    uint32_t num_matrices;
    uint32_t num_heads;
    uint32_t num_key_heads;
    int64_t has_initial_state;
    int64_t batch_size;
    int64_t seq_len;
    int64_t total_tokens;
    uint64_t ffts_addr;
};

// ===================================================================
// Device-only helpers (shared with standard mega-kernel)
// ===================================================================
#ifdef __CCE_AICORE__

template <bool isAIVOnly = true>
AICORE inline void SyncAllImpl()
{
    if constexpr (isAIVOnly) {
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    } else {
        pto::SYNCALL<pto::SyncCoreType::Mix>();
    }
}

template <typename T>
AICORE inline void mega_clean_invalidate_scalar_range(__gm__ T *ptr, int64_t element_count)
{
#if defined(PTO_NPU_ARCH_A5) && defined(__DAV_VEC__)
    if (element_count <= 0) return;
    constexpr uint64_t CACHE_LINE_BYTES = AscendC::CACHE_LINE_SIZE;
    const uint64_t first_addr =
        reinterpret_cast<uint64_t>(ptr) / CACHE_LINE_BYTES * CACHE_LINE_BYTES;
    const uint64_t last_addr =
        (reinterpret_cast<uint64_t>(ptr + element_count - 1) / CACHE_LINE_BYTES) * CACHE_LINE_BYTES;
    AscendC::GlobalTensor<uint8_t> cache_lines;
    cache_lines.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(first_addr));
    for (uint64_t offset = 0; offset <= last_addr - first_addr; offset += CACHE_LINE_BYTES) {
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(cache_lines[offset]);
    }
#else
    (void)ptr;
    (void)element_count;
#endif
}

template <typename T>
AICORE inline void mega_transpose_TH_to_HT(__gm__ T *src, __gm__ T *dst, int64_t T_len, int32_t H)
{
#if defined(__DAV_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t BLOCK = 128;
    constexpr int32_t ES = static_cast<int32_t>(sizeof(T));
    constexpr int32_t MinTransposeCols = 16;
    constexpr int32_t AlignElems = ((32 / ES) > MinTransposeCols) ? (32 / ES) : MinTransposeCols;
    constexpr int32_t HP = ((GDN_MAX_HEADS + AlignElems - 1) / AlignElems) * AlignElems;
    constexpr int32_t SRC_UB = 0;
    constexpr int32_t DST_UB = SRC_UB + BLOCK * HP * ES;
    constexpr int32_t TMP_UB = DST_UB + HP * BLOCK * ES;

    using UBSrcFull =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBSrcDyn =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBDst = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, HP, BLOCK, SLayout::NoneBox, 512>;
    using UBDstDyn = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using UBTmp = Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512>;

    using UBRow = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, 1, BLOCK, SLayout::NoneBox, 512>;
    using UBRowDyn = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;

    using Gm2D = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmSrcS = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GmS1 = pto::Stride<1, 1, 1, 1, 1>;
    GmSrcS src_stride(H);

    UBSrcFull ub_src;
    TASSIGN(ub_src, SRC_UB);
    UBDst ub_dst;
    TASSIGN(ub_dst, DST_UB);
    UBTmp ub_tmp;
    TASSIGN(ub_tmp, TMP_UB);

    int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;

    for (int64_t bi = static_cast<int64_t>(cid); bi < num_tok_blocks; bi += static_cast<int64_t>(block_num)) {
        int64_t t0 = bi * BLOCK;
        int32_t valid = (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);

        {
            Gm2D gs;
            gs.shape[3] = valid;
            gs.shape[4] = H;
            GlobalTensor<T, Gm2D, GmSrcS> gm(src + t0 * H, gs, src_stride);
            UBSrcDyn ld(valid, H);
            TASSIGN(ld, SRC_UB);
            TLOAD(ld, gm);
            if (valid != BLOCK || H != HP) TFILLPAD_INPLACE(ub_src, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TTRANS(ub_dst, ub_src, ub_tmp);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        for (int32_t h = 0; h < H; ++h) {
            Gm1D gs;
            gs.shape[4] = valid;
            GlobalTensor<T, Gm1D, GmS1> gm(dst + h * T_len + t0, gs);
            UBRowDyn st(1, valid);
            TASSIGN(st, DST_UB + h * BLOCK * ES);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

#if defined(PTO_NPU_ARCH_A5)
        if constexpr (std::is_same_v<T, float>) {
            if (H == 2) {
                pipe_barrier(PIPE_ALL);
                mega_clean_invalidate_scalar_range(src + t0 * H, static_cast<int64_t>(valid) * H);
                mega_clean_invalidate_scalar_range(dst + t0, valid);
                mega_clean_invalidate_scalar_range(dst + T_len + t0, valid);
                AscendC::GlobalTensor<T> src_tensor;
                AscendC::GlobalTensor<T> dst_tensor;
                src_tensor.SetGlobalBuffer(src);
                dst_tensor.SetGlobalBuffer(dst);
                for (int32_t t = 0; t < valid; ++t) {
                    for (int32_t h = 0; h < H; ++h) {
                        dst_tensor.SetValue(h * T_len + t0 + t, src_tensor.GetValue((t0 + t) * H + h));
                    }
                }
                mega_clean_invalidate_scalar_range(dst + t0, valid);
                mega_clean_invalidate_scalar_range(dst + T_len + t0, valid);
                pipe_barrier(PIPE_ALL);
            }
        }
#endif
    }
#endif
}

template <typename SrcT, typename DstT>
AICORE inline void mega_cast_elements(__gm__ SrcT *src, __gm__ DstT *dst, int64_t element_count)
{
#if defined(__DAV_VEC__)
    static_assert(!std::is_same_v<SrcT, DstT>, "mega_cast_elements requires distinct source and destination types.");
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t TILE_ELEMENTS = 256;
    constexpr int32_t SRC_UB = 0;
    constexpr int32_t FP32_UB = SRC_UB + TILE_ELEMENTS * static_cast<int32_t>(sizeof(SrcT));
    constexpr int32_t DST_UB = FP32_UB + TILE_ELEMENTS * static_cast<int32_t>(sizeof(float));

    using SrcUB = Tile<TileType::Vec, SrcT, 1, TILE_ELEMENTS, BLayout::RowMajor, 1, TILE_ELEMENTS,
                       SLayout::NoneBox, 512, PadValue::Zero>;
    using DynSrcUB =
        Tile<TileType::Vec, SrcT, 1, TILE_ELEMENTS, BLayout::RowMajor, DYNAMIC, DYNAMIC,
             SLayout::NoneBox, 512, PadValue::Zero>;
    using Fp32UB = Tile<TileType::Vec, float, 1, TILE_ELEMENTS, BLayout::RowMajor, 1, TILE_ELEMENTS,
                        SLayout::NoneBox, 512>;
    using DstUB = Tile<TileType::Vec, DstT, 1, TILE_ELEMENTS, BLayout::RowMajor, 1, TILE_ELEMENTS,
                       SLayout::NoneBox, 512>;
    using DynDstUB =
        Tile<TileType::Vec, DstT, 1, TILE_ELEMENTS, BLayout::RowMajor, DYNAMIC, DYNAMIC,
             SLayout::NoneBox, 512>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = pto::Stride<1, 1, 1, 1, 1>;

    SrcUB src_ub;
    TASSIGN(src_ub, SRC_UB);
    Fp32UB fp32_ub;
    TASSIGN(fp32_ub, FP32_UB);
    DstUB dst_ub;
    TASSIGN(dst_ub, DST_UB);

    const int64_t block_stride = static_cast<int64_t>(block_num) * TILE_ELEMENTS;
    for (int64_t offset = static_cast<int64_t>(cid) * TILE_ELEMENTS; offset < element_count;
         offset += block_stride) {
        const int32_t valid = static_cast<int32_t>(
            (offset + TILE_ELEMENTS <= element_count) ? TILE_ELEMENTS : (element_count - offset));
        {
            Gm1D shape;
            shape.shape[4] = valid;
            GlobalTensor<SrcT, Gm1D, GmS1> gm(src + offset, shape);
            DynSrcUB load(1, valid);
            TASSIGN(load, SRC_UB);
            TLOAD(load, gm);
            if (valid != TILE_ELEMENTS) {
                TFILLPAD_INPLACE(src_ub, load);
            }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if constexpr (std::is_same_v<SrcT, float>) {
            TCVT(dst_ub, src_ub, RoundMode::CAST_RINT);
        } else if constexpr (std::is_same_v<DstT, float>) {
            TCVT(dst_ub, src_ub, RoundMode::CAST_NONE);
        } else {
            TCVT(fp32_ub, src_ub, RoundMode::CAST_NONE);
            gdn_sync::VectorBarrier();
            TCVT(dst_ub, fp32_ub, RoundMode::CAST_RINT);
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        {
            Gm1D shape;
            shape.shape[4] = valid;
            GlobalTensor<DstT, Gm1D, GmS1> gm(dst + offset, shape);
            DynDstUB store(1, valid);
            TASSIGN(store, DST_UB);
            TSTORE(gm, store);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
#else
    (void)src;
    (void)dst;
    (void)element_count;
#endif
}

#endif  // __CCE_AICORE__

// ===================================================================
// Include original kernel implementations in separate namespaces.
// ===================================================================

namespace mk_cumsum {
#include "chunk_cumsum.cpp"
}

namespace mk_kkt {
#include "scaled_dot_kkt.cpp"
}

namespace mk_solve {
#include "tri_inverse_impl.cpp"
}

namespace mk_wy {
#include "wy_fast.cpp"
}

namespace mk_h {
#include "chunk_h.cpp"
}

namespace mk_o {
#include "chunk_o.cpp"
}

#if defined(__DAV_CUBE__)
#define GDN_WY_FAST_CALL wy_fast_kernel_aic
#define GDN_CHUNK_O_CALL chunk_o_kernel_aic
#elif defined(__DAV_VEC__)
#define GDN_WY_FAST_CALL wy_fast_kernel_aiv
#define GDN_CHUNK_O_CALL chunk_o_kernel_aiv
#else
#define GDN_WY_FAST_CALL wy_fast_kernel
#define GDN_CHUNK_O_CALL chunk_o_kernel
#endif

template <typename ComputeT>
AICORE inline void mega_solve_tril(__gm__ ComputeT *out, __gm__ ComputeT *in, __gm__ ComputeT *minus_id,
                                   uint32_t matrix_size, uint32_t num_matrices, uint32_t num_bsnd_heads,
                                   __gm__ int32_t *cu_seqlens, uint32_t is_lower)
{
    if (num_matrices <= get_block_num())
        mk_solve::runKernelTriInvRecUnroll<ComputeT, float, GDN_C, 1, true, ComputeT>(
            out, in, minus_id, num_matrices, num_bsnd_heads, cu_seqlens, is_lower);
    else if (num_matrices <= 2u * get_block_num())
        mk_solve::runKernelTriInvRecUnroll<ComputeT, float, GDN_C, 2, true, ComputeT>(
            out, in, minus_id, num_matrices, num_bsnd_heads, cu_seqlens, is_lower);
    else
        mk_solve::runKernelTriInvRecUnroll<ComputeT, float, GDN_C, 4, true, ComputeT>(
            out, in, minus_id, num_matrices, num_bsnd_heads, cu_seqlens, is_lower);
}

template <typename InputT, typename ComputeT, typename OutputT>
AICORE inline void mega_kernel_impl(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr,
                                    GM_ADDR msk_lower_ptr, GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr,
                                    GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr,
                                    GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr,
                                    GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr,
                                    GM_ADDR h0_ptr, int64_t has_initial_state, GM_ADDR kkt_ws_ptr,
                                    GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr,
                                    GM_ADDR o_ws_qk_ptr, GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr,
                                    GM_ADDR h0_ws_ptr,
                                    int32_t H, uint32_t num_key_heads, int64_t batch_size, int64_t seq_len,
                                    int64_t total_tokens, uint32_t num_matrices, uint64_t ffts_addr)
{
    constexpr int32_t D = GDN_D;
    constexpr int32_t C = GDN_C;
    static_assert(std::is_same_v<InputT, half> || std::is_same_v<InputT, bfloat16_t>);
    static_assert(std::is_same_v<ComputeT, half> || std::is_same_v<ComputeT, bfloat16_t>);
    static_assert(std::is_same_v<OutputT, half> || std::is_same_v<OutputT, bfloat16_t>);
    static_assert(sizeof(InputT) == sizeof(ComputeT) && sizeof(ComputeT) == sizeof(OutputT));

    if (num_key_heads == 0 || (static_cast<uint32_t>(H) % num_key_heads) != 0) {
        return;
    }

    const int64_t qk_elements = total_tokens * static_cast<int64_t>(num_key_heads) * D;
    const int64_t token_head_elements = total_tokens * static_cast<int64_t>(H);
    const int64_t token_vector_elements = token_head_elements * D;
    const int64_t attention_elements = token_head_elements * C;
    const int64_t matrix_elements = static_cast<int64_t>(num_matrices) * D * D;
    const int64_t final_state_elements = batch_size * static_cast<int64_t>(H) * D * D;

    GM_ADDR q_compute_ptr = q_ptr;
    GM_ADDR k_compute_ptr = k_ptr;
    GM_ADDR v_compute_ptr = v_ptr;
    GM_ADDR beta_compute_ptr = beta_ptr;
    GM_ADDR minus_id_compute_ptr = minus_id_ptr;
    GM_ADDR h0_compute_ptr = h0_ptr;

    if constexpr (!std::is_same_v<InputT, ComputeT>) {
        // Reuse outputs that are not live yet. q/k fit in a_inv_f32 because Hg <= H,
        // while beta is consumed from A before KKT overwrites A with its real output.
        q_compute_ptr = A_inv_f32_ptr;
        k_compute_ptr = A_inv_f32_ptr + qk_elements * static_cast<int64_t>(sizeof(ComputeT));
        v_compute_ptr = v_new_ptr;
        beta_compute_ptr = A_ptr;
        minus_id_compute_ptr = o_ws_gated_ptr;
        h0_compute_ptr = h0_ws_ptr;

        mega_cast_elements<InputT, ComputeT>(reinterpret_cast<__gm__ InputT *>(q_ptr),
                                             reinterpret_cast<__gm__ ComputeT *>(q_compute_ptr), qk_elements);
        mega_cast_elements<InputT, ComputeT>(reinterpret_cast<__gm__ InputT *>(k_ptr),
                                             reinterpret_cast<__gm__ ComputeT *>(k_compute_ptr), qk_elements);
        mega_cast_elements<InputT, ComputeT>(reinterpret_cast<__gm__ InputT *>(v_ptr),
                                             reinterpret_cast<__gm__ ComputeT *>(v_compute_ptr),
                                             token_vector_elements);
        mega_cast_elements<InputT, ComputeT>(reinterpret_cast<__gm__ InputT *>(beta_ptr),
                                             reinterpret_cast<__gm__ ComputeT *>(beta_compute_ptr),
                                             token_head_elements);
        mega_cast_elements<InputT, ComputeT>(reinterpret_cast<__gm__ InputT *>(minus_id_ptr),
                                             reinterpret_cast<__gm__ ComputeT *>(minus_id_compute_ptr), C * C);
        if (has_initial_state != 0) {
            mega_cast_elements<InputT, ComputeT>(reinterpret_cast<__gm__ InputT *>(h0_ptr),
                                                 reinterpret_cast<__gm__ ComputeT *>(h0_compute_ptr),
                                                 final_state_elements);
        }
        SyncAllImpl<false>();
    }

    mk_cumsum::cumsum_kernel<C>(reinterpret_cast<__gm__ float *>(g_in_ptr),
                                reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, H,
                                ffts_addr);

#ifdef MEGA_STOP_AFTER_CUMSUM
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_SYNC1
    return;
#endif

    mega_transpose_TH_to_HT<float>(reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                   reinterpret_cast<__gm__ float *>(g_t_ptr), total_tokens, H);
    mega_transpose_TH_to_HT<ComputeT>(reinterpret_cast<__gm__ ComputeT *>(beta_compute_ptr),
                                     reinterpret_cast<__gm__ ComputeT *>(beta_t_ptr), total_tokens, H);

#ifdef MEGA_STOP_AFTER_TRANSPOSE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_kkt::kkt_kernel<ComputeT, D, C>(
        reinterpret_cast<__gm__ ComputeT *>(k_compute_ptr), reinterpret_cast<__gm__ ComputeT *>(beta_t_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_lower_ptr),
        reinterpret_cast<__gm__ ComputeT *>(kkt_ws_ptr), reinterpret_cast<__gm__ ComputeT *>(A_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
        static_cast<uint32_t>(H), num_key_heads, ffts_addr);

#if defined(__DAV_CUBE__)
    pipe_barrier(PIPE_ALL);
    gdn_sync::Wait<PIPE_FIX>(2);
    gdn_sync::Wait<PIPE_FIX>(3);
#endif

#ifdef MEGA_STOP_AFTER_KKT
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mega_solve_tril<ComputeT>(reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
                              reinterpret_cast<__gm__ ComputeT *>(A_ptr),
                              reinterpret_cast<__gm__ ComputeT *>(minus_id_compute_ptr), C, num_matrices, H,
                              reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), 1);

#ifdef MEGA_STOP_AFTER_SOLVE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_CAST
    pipe_barrier(PIPE_ALL);
    return;
#endif

#ifdef MEGA_STOP_AFTER_SYNC_BEFORE_WY
    return;
#endif

    mk_wy::GDN_WY_FAST_CALL<ComputeT, D, C>(
        reinterpret_cast<__gm__ ComputeT *>(k_compute_ptr), reinterpret_cast<__gm__ ComputeT *>(v_compute_ptr),
        reinterpret_cast<__gm__ ComputeT *>(beta_t_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr), reinterpret_cast<__gm__ ComputeT *>(wy_ws_a1_ptr),
        reinterpret_cast<__gm__ ComputeT *>(wy_ws_a2_ptr), reinterpret_cast<__gm__ ComputeT *>(w_ptr),
        reinterpret_cast<__gm__ ComputeT *>(u_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size,
        seq_len, total_tokens, static_cast<uint32_t>(H), num_key_heads, ffts_addr);

#if defined(__DAV_VEC__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        gdn_sync::AllocateVecGm(mk_wy::kWyA2FreeEvent);
        gdn_sync::AllocateVecGm(mk_wy::kWyA1FreeEvent);
    }
#endif

#ifdef MEGA_STOP_AFTER_WY
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_h::chunk_h_kernel<ComputeT, D, C>(
        reinterpret_cast<__gm__ ComputeT *>(k_compute_ptr), reinterpret_cast<__gm__ ComputeT *>(w_ptr),
        reinterpret_cast<__gm__ ComputeT *>(u_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ ComputeT *>(s_ptr), reinterpret_cast<__gm__ ComputeT *>(v_new_ptr),
        reinterpret_cast<__gm__ ComputeT *>(fs_ptr), reinterpret_cast<__gm__ ComputeT *>(h0_compute_ptr),
        has_initial_state, 1, reinterpret_cast<__gm__ ComputeT *>(h_ws_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
        static_cast<uint32_t>(H), num_key_heads, ffts_addr);

#ifdef MEGA_STOP_AFTER_H
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_o::GDN_CHUNK_O_CALL<ComputeT, D, C>(
        reinterpret_cast<__gm__ ComputeT *>(q_compute_ptr), reinterpret_cast<__gm__ ComputeT *>(k_compute_ptr),
        reinterpret_cast<__gm__ ComputeT *>(v_new_ptr), reinterpret_cast<__gm__ ComputeT *>(s_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_full_ptr),
        reinterpret_cast<__gm__ ComputeT *>(o_ws_qk_ptr), reinterpret_cast<__gm__ ComputeT *>(o_ws_qs_ptr),
        reinterpret_cast<__gm__ ComputeT *>(o_ws_gated_ptr), reinterpret_cast<__gm__ ComputeT *>(o_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
        static_cast<uint32_t>(H), num_key_heads, ffts_addr);

#if defined(__DAV_CUBE__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        gdn_sync::Wait<PIPE_FIX>(3);
    }
#endif

    if constexpr (!std::is_same_v<ComputeT, OutputT>) {
        SyncAllImpl<false>();

        // q/k no longer need the a_inv_f32 scratch after chunk_o, so materialize
        // the public FP32 inverse before converting the low-precision outputs in place.
        mega_cast_elements<ComputeT, float>(reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
                                            reinterpret_cast<__gm__ float *>(A_inv_f32_ptr),
                                            attention_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(o_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(o_ptr),
                                              token_vector_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(beta_t_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(beta_t_ptr),
                                              token_head_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(A_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(A_ptr), attention_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(A_inv_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(A_inv_ptr),
                                              attention_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(w_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(w_ptr),
                                              token_vector_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(u_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(u_ptr),
                                              token_vector_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(s_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(s_ptr), matrix_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(v_new_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(v_new_ptr),
                                              token_vector_elements);
        mega_cast_elements<ComputeT, OutputT>(reinterpret_cast<__gm__ ComputeT *>(fs_ptr),
                                              reinterpret_cast<__gm__ OutputT *>(fs_ptr),
                                              final_state_elements);

        SyncAllImpl<false>();
    }
}

#undef GDN_WY_FAST_CALL
#undef GDN_CHUNK_O_CALL

extern "C" __global__ __aicore__ void
GDN_KERNEL_NAME(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr, GM_ADDR msk_lower_ptr,
                GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR initial_state_ptr,
                GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr, GM_ADDR beta_t_ptr, GM_ADDR A_ptr,
                GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr, GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr,
                GM_ADDR v_new_ptr, GM_ADDR fs_ptr, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MegaChunkGdnKernelTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaChunkGdnKernelTilingData, tiling_data, tiling);
    GM_ADDR user_ws = AscendC::GetUserWorkspace(workspace);
    const uint64_t tile_bytes = static_cast<uint64_t>(GDN_C) * GDN_C * sizeof(GdnComputeT);

    GM_ADDR kkt_ws_ptr = user_ws;
    GM_ADDR wy_ws_a1_ptr = kkt_ws_ptr + static_cast<uint64_t>(tiling_data.block_dim) * 2 * tile_bytes;
    GM_ADDR wy_ws_a2_ptr = wy_ws_a1_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR h_ws_ptr = wy_ws_a2_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR o_ws_qk_ptr = h_ws_ptr + static_cast<uint64_t>(tiling_data.block_dim) * 4 * tile_bytes;
    GM_ADDR o_ws_qs_ptr = o_ws_qk_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR o_ws_gated_ptr = o_ws_qs_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR h0_ws_ptr = o_ws_gated_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;

    if (tiling_data.num_heads == 0 || tiling_data.num_heads > GDN_MAX_HEADS) {
        return;
    }

    mega_kernel_impl<GdnInputT, GdnComputeT, GdnOutputT>(
        q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr, msk_full_ptr, minus_id_ptr, cu_seqlens_ptr, o_ptr,
        g_sum_ptr, g_t_ptr, beta_t_ptr, A_ptr, A_inv_f32_ptr, A_inv_ptr, w_ptr, u_ptr, s_ptr, v_new_ptr, fs_ptr,
        initial_state_ptr, tiling_data.has_initial_state, kkt_ws_ptr, wy_ws_a1_ptr, wy_ws_a2_ptr, h_ws_ptr,
        o_ws_qk_ptr, o_ws_qs_ptr, o_ws_gated_ptr, h0_ws_ptr, static_cast<int32_t>(tiling_data.num_heads),
        tiling_data.num_key_heads, tiling_data.batch_size, tiling_data.seq_len, tiling_data.total_tokens,
        tiling_data.num_matrices, tiling_data.ffts_addr);
}

// The CANN wrapper generated for mixed AIC/AIV kernels calls matmul::clearWorkspace
// after including this source. Keep this include after PTO code so CANN's DYNAMIC
// enum does not collide with pto::DYNAMIC in the kernel templates above.
#include "lib/matmul_intf.h"
