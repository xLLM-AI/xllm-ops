/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include <cstdint>
#ifdef __CCE_AICORE__
#define KDA_HOST_DEVICE __aicore__
#else
#define KDA_HOST_DEVICE
#endif

namespace kda_prefill {
constexpr int64_t kHeadDim = 128;
constexpr int64_t kChunkSize = 128;
constexpr int64_t kAscend910bSmallChunkSize = 64;
constexpr int64_t kQkvParts = 3;
constexpr int64_t kConvTaps = 4;
constexpr int64_t kConvHistory = kConvTaps - 1;
constexpr uint64_t kBf16Bytes = sizeof(uint16_t);
constexpr uint64_t kFloatBytes = sizeof(float);
constexpr uint64_t kWorkspaceAlignment = 512;
constexpr uint64_t kStateElements = kHeadDim * kHeadDim;
constexpr uint64_t kChunkMatrixElements = kChunkSize * kChunkSize;
constexpr uint64_t kChunkKeyElements = kChunkSize * kHeadDim;
constexpr uint64_t kChunkValueElements = kChunkSize * kHeadDim;
// ChunkH: WS_WS[C,V], WS_K[C,K], WS_V[C,V], WS_S/WS_KV[K,V].
constexpr uint64_t kChunkHWorkspaceElements =
    2 * kChunkValueElements + kChunkKeyElements + 2 * kStateElements;
// ChunkO: WS_Q/WS_K[C,K], WS_V/WS_QS/WS_QKV[C,V], WS_S[K,V], WS_QK[C,C].
constexpr uint64_t kChunkOWorkspaceElements =
    2 * kChunkKeyElements + 3 * kChunkValueElements + kStateElements +
    kChunkMatrixElements;
constexpr uint64_t kMaskMatrices = 2;
enum class Input : uint32_t {
  QKV,
  GATE,
  BETA,
  A_LOG,
  GATE_BIAS,
  CONV_WEIGHT,
  CONV_BIAS,
  CONV_STATE_IN,
  SSM_STATE_IN,
  CU_SEQLENS,
  CONV_READ_INDICES,
  CONV_WRITE_INDICES,
  SSM_READ_INDICES,
  SSM_WRITE_INDICES,
  INPUT_COUNT
};
enum class Attr : uint32_t { CONV_OUTPUT_SLOTS, SSM_OUTPUT_SLOTS, FFTS_ADDR };
struct TilingData {
  int64_t batch;
  int64_t tokens;
  int64_t heads;
  int64_t conv_input_slots;
  int64_t conv_output_slots;
  int64_t ssm_input_slots;
  int64_t ssm_output_slots;
  uint64_t blocks;
  uint64_t ffts_addr;
};
static_assert(sizeof(TilingData) == 72, "MegaKdaPrefill tiling ABI changed");

// Shared host/device layout. Every buffer starts on a DMA-safe boundary.
class Workspace final {
 public:
  uint64_t q;
  uint64_t k;
  uint64_t beta;
  uint64_t g;
  uint64_t value;
  uint64_t lower;
  uint64_t inverse;
  uint64_t aqk;
  uint64_t u;
  uint64_t w;
  uint64_t corrected;
  uint64_t output;
  uint64_t snapshots;
  uint64_t initial;
  uint64_t final;
  uint64_t wy_a;
  uint64_t wy_k;
  uint64_t h_ws;
  uint64_t o_ws;
  uint64_t masks;
  uint64_t eye;
  uint64_t bytes;
  KDA_HOST_DEVICE static inline uint64_t align(uint64_t n) {
    return (n + kWorkspaceAlignment - 1) / kWorkspaceAlignment *
           kWorkspaceAlignment;
  }
  KDA_HOST_DEVICE explicit inline Workspace(const TilingData& t,
                                           int64_t chunk_size = kChunkSize)
      : bytes(0) {
    const uint64_t n = static_cast<uint64_t>(t.tokens) * t.heads;
    const uint64_t state =
        static_cast<uint64_t>(t.batch) * t.heads * kStateElements * kFloatBytes;
    const uint64_t chunks =
        (t.tokens + chunk_size - 1) / chunk_size + t.batch - 1;
    q = take(n * kHeadDim * kBf16Bytes);
    k = take(n * kHeadDim * kBf16Bytes);
    beta = take(n * kFloatBytes);
    g = take(n * kHeadDim * kFloatBytes);
    value = take(n * kHeadDim * kFloatBytes);
    lower = take(n * chunk_size * kFloatBytes);
    inverse = take(n * chunk_size * kFloatBytes);
    aqk = take(n * chunk_size * kFloatBytes);
    u = take(n * kHeadDim * kFloatBytes);
    w = take(n * kHeadDim * kFloatBytes);
    corrected = take(n * kHeadDim * kFloatBytes);
    output = take(n * kHeadDim * kFloatBytes);
    snapshots = take(chunks * t.heads * kStateElements * kFloatBytes);
    initial = take(state);
    final = take(state);
    wy_a = take(t.blocks * kChunkMatrixElements * kFloatBytes);
    wy_k = take(t.blocks * kChunkKeyElements * kFloatBytes);
    h_ws = take(t.blocks * kChunkHWorkspaceElements * kFloatBytes);
    o_ws = take(t.blocks * kChunkOWorkspaceElements * kFloatBytes);
    masks = take(kMaskMatrices * kChunkMatrixElements * kFloatBytes);
    eye = take(kChunkMatrixElements * kFloatBytes);
  }

 private:
  KDA_HOST_DEVICE inline uint64_t take(uint64_t n) {
    const uint64_t p = bytes;
    bytes += align(n);
    return p;
  }
};
}  // namespace kda_prefill
#undef KDA_HOST_DEVICE
