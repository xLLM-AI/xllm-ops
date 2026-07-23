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
#include <type_traits>
using namespace pto;

static_assert(std::is_same_v<DTYPE_Q, half> || std::is_same_v<DTYPE_Q, bfloat16_t>,
              "MegaChunkGdn supports FP16 or BF16 compute tensors.");

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

// Sync flag ids are provided by PTO. Keep this kernel on the shared sync
// contract so updates to the runtime synchronization framework stay aligned.
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllImpl()
{
    pipe_barrier(PIPE_ALL);
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
#if defined(__DAV_C220_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_C220_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
}

template <typename T>
AICORE inline void mega_transpose_TH_to_HT(__gm__ T *src, __gm__ T *dst, int64_t T_len, int32_t H)
{
#if defined(__DAV_C220_VEC__)
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
    using GmSrcS = Stride<1, 1, 1, DYNAMIC, 1>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;
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
    }
#endif
}

template <int32_t H, int32_t C>
AICORE inline void mega_cast_fp32_to_dtype_bsnd(__gm__ float *src, __gm__ DTYPE_Q *dst, uint32_t num_matrices,
                                               int64_t total_tokens)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t F32_UB = 0;
    constexpr int32_t F16_UB = C * static_cast<int32_t>(sizeof(float));

    using SrcUB = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512, PadValue::Zero>;
    using DynSrcUB =
        Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstUB = Tile<TileType::Vec, DTYPE_Q, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512>;
    using DynDstUB = Tile<TileType::Vec, DTYPE_Q, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    SrcUB src_ub;
    TASSIGN(src_ub, F32_UB);
    DstUB dst_ub;
    TASSIGN(dst_ub, F16_UB);

    for (uint32_t m = cid; m < num_matrices; m += block_num) {
        uint32_t h = m % static_cast<uint32_t>(H);
        uint32_t chunk_idx = m / static_cast<uint32_t>(H);

        for (int64_t t = 0; t < total_tokens; ++t) {
            int64_t off = t * static_cast<int64_t>(H * C) + static_cast<int64_t>(h * C);

            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<float, Gm1D, GmS1> gm(src + off, gs);
                SrcUB ld;
                TASSIGN(ld, F32_UB);
                TLOAD(ld, gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(dst_ub, src_ub, RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<DTYPE_Q, Gm1D, GmS1> gm(dst + off, gs);
                DstUB st;
                TASSIGN(st, F16_UB);
                TSTORE(gm, st);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
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

#if defined(__DAV_C220_CUBE__)
#define GDN_WY_FAST_CALL wy_fast_kernel_aic
#define GDN_CHUNK_O_CALL chunk_o_kernel_aic
#elif defined(__DAV_C220_VEC__)
#define GDN_WY_FAST_CALL wy_fast_kernel_aiv
#define GDN_CHUNK_O_CALL chunk_o_kernel_aiv
#else
#define GDN_WY_FAST_CALL wy_fast_kernel
#define GDN_CHUNK_O_CALL chunk_o_kernel
#endif

AICORE inline void mega_solve_tril(__gm__ DTYPE_Q *out, __gm__ DTYPE_Q *in, __gm__ DTYPE_Q *minus_id, uint32_t matrix_size,
                                   uint32_t num_matrices, uint32_t num_bsnd_heads, __gm__ int32_t *cu_seqlens,
                                   uint32_t is_lower)
{
    if (num_matrices <= get_block_num())
        mk_solve::runKernelTriInvRecUnroll<DTYPE_Q, float, GDN_C, 1, true, DTYPE_Q>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
    else if (num_matrices <= 2u * get_block_num())
        mk_solve::runKernelTriInvRecUnroll<DTYPE_Q, float, GDN_C, 2, true, DTYPE_Q>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
    else
        mk_solve::runKernelTriInvRecUnroll<DTYPE_Q, float, GDN_C, 4, true, DTYPE_Q>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
}

AICORE inline void mega_kernel_impl(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr,
                                    GM_ADDR msk_lower_ptr, GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr,
                                    GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr,
                                    GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr,
                                    GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr,
                                    GM_ADDR h0_ptr, int64_t has_initial_state, GM_ADDR kkt_ws_ptr,
                                    GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr,
                                    GM_ADDR o_ws_qk_ptr, GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr,
                                    int32_t H, uint32_t num_key_heads, int64_t batch_size, int64_t seq_len,
                                    int64_t total_tokens, uint32_t num_matrices, uint64_t ffts_addr)
{
    constexpr int32_t D = GDN_D;
    constexpr int32_t C = GDN_C;

    if (num_key_heads == 0 || (static_cast<uint32_t>(H) % num_key_heads) != 0) {
        return;
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
    mega_transpose_TH_to_HT<DTYPE_Q>(reinterpret_cast<__gm__ DTYPE_Q *>(beta_ptr),
                                  reinterpret_cast<__gm__ DTYPE_Q *>(beta_t_ptr), total_tokens, H);

#ifdef MEGA_STOP_AFTER_TRANSPOSE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_kkt::kkt_kernel<D, C>(
        reinterpret_cast<__gm__ DTYPE_Q *>(k_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(beta_t_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_lower_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(kkt_ws_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(A_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
        static_cast<uint32_t>(H), num_key_heads, ffts_addr);

#if defined(__DAV_C220_CUBE__)
    pipe_barrier(PIPE_ALL);
    wait_flag_dev(2);
    wait_flag_dev(3);
#endif

#ifdef MEGA_STOP_AFTER_KKT
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mega_solve_tril(reinterpret_cast<__gm__ DTYPE_Q *>(A_inv_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(A_ptr),
                    reinterpret_cast<__gm__ DTYPE_Q *>(minus_id_ptr), C, num_matrices, H,
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

    mk_wy::GDN_WY_FAST_CALL<D, C>(
        reinterpret_cast<__gm__ DTYPE_Q *>(k_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(v_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(beta_t_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(A_inv_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(wy_ws_a1_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(wy_ws_a2_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(w_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(u_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len,
        total_tokens, static_cast<uint32_t>(H), num_key_heads, ffts_addr);

#if defined(__DAV_C220_VEC__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
        wait_flag_dev(4);
    }
#endif

#ifdef MEGA_STOP_AFTER_WY
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_h::chunk_h_kernel<D, C>(
        reinterpret_cast<__gm__ DTYPE_Q *>(k_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(w_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(u_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(s_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(v_new_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(fs_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(h0_ptr), has_initial_state,
        1,
        reinterpret_cast<__gm__ DTYPE_Q *>(h_ws_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size,
        seq_len, total_tokens, static_cast<uint32_t>(H), num_key_heads, ffts_addr);

#ifdef MEGA_STOP_AFTER_H
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_o::GDN_CHUNK_O_CALL<D, C>(
        reinterpret_cast<__gm__ DTYPE_Q *>(q_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(k_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(v_new_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(s_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_full_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(o_ws_qk_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(o_ws_qs_ptr),
        reinterpret_cast<__gm__ DTYPE_Q *>(o_ws_gated_ptr), reinterpret_cast<__gm__ DTYPE_Q *>(o_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
        static_cast<uint32_t>(H), num_key_heads, ffts_addr);

#if defined(__DAV_C220_CUBE__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
    }
#endif
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
    const uint64_t tile_bytes = static_cast<uint64_t>(GDN_C) * GDN_C * sizeof(DTYPE_Q);

    GM_ADDR kkt_ws_ptr = user_ws;
    GM_ADDR wy_ws_a1_ptr = kkt_ws_ptr + static_cast<uint64_t>(tiling_data.block_dim) * 2 * tile_bytes;
    GM_ADDR wy_ws_a2_ptr = wy_ws_a1_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR h_ws_ptr = wy_ws_a2_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR o_ws_qk_ptr = h_ws_ptr + static_cast<uint64_t>(tiling_data.block_dim) * 4 * tile_bytes;
    GM_ADDR o_ws_qs_ptr = o_ws_qk_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;
    GM_ADDR o_ws_gated_ptr = o_ws_qs_ptr + static_cast<uint64_t>(tiling_data.block_dim) * tile_bytes;

    if (tiling_data.num_heads == 0 || tiling_data.num_heads > GDN_MAX_HEADS) {
        return;
    }

    mega_kernel_impl(q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr, msk_full_ptr, minus_id_ptr,
                     cu_seqlens_ptr, o_ptr, g_sum_ptr, g_t_ptr, beta_t_ptr, A_ptr, A_inv_f32_ptr, A_inv_ptr, w_ptr,
                     u_ptr, s_ptr, v_new_ptr, fs_ptr, initial_state_ptr, tiling_data.has_initial_state, kkt_ws_ptr,
                     wy_ws_a1_ptr, wy_ws_a2_ptr, h_ws_ptr, o_ws_qk_ptr, o_ws_qs_ptr, o_ws_gated_ptr,
                     static_cast<int32_t>(tiling_data.num_heads), tiling_data.num_key_heads, tiling_data.batch_size,
                     tiling_data.seq_len, tiling_data.total_tokens, tiling_data.num_matrices, tiling_data.ffts_addr);
}

// The CANN wrapper generated for mixed AIC/AIV kernels calls matmul::clearWorkspace
// after including this source. Keep this include after PTO code so CANN's DYNAMIC
// enum does not collide with pto::DYNAMIC in the kernel templates above.
#include "lib/matmul_intf.h"
