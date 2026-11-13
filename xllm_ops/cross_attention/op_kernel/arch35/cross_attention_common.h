/**
 * CrossAttention shared constants and helpers (host / device, no device-only types).
 */
#ifndef CROSS_ATTENTION_COMMON_H
#define CROSS_ATTENTION_COMMON_H

#include <cstdint>
#include "cross_attention_tiling.h"

// ---- Geometry ----
constexpr uint32_t CA_HEAD_DIM = 128;
constexpr uint32_t CA_KV_TILE = 128;
constexpr uint32_t CA_C0 = 16;          // 32 B / sizeof(bf16 | fp16)
constexpr uint32_t CA_BLOCK_CUBE = 16;  // rows per cube fractal
// Max q rows per task; a task is one K/V panel reuse. Must be a multiple of CA_BLOCK_CUBE: the cube
// side issues Mmad / Fixpipe one 16-row fractal at a time and the panel is loaded once for all
// fractals. Larger values amortise the fixed cost but leave fewer tasks for the core makespan.
constexpr uint32_t CA_MAX_R = 64;

// ---- CV direct connect (no GM staging) ----
// S is finished in AIC L0C and fixpipe'd straight into the UB of each AIV sub; AIV's softmax output P
// goes UB -> L1 (the cube A1 window); O accumulates in L0C across beats and is fixpipe'd into UB once
// per task. Each AIV sub therefore owns half of the R rows, split on a 16-row boundary: Fixpipe mSize
// is counted in fractal rows and the L1 copy of P is written as 16-row NZ blocks.
constexpr uint32_t CA_UB_ROWS = CA_MAX_R / 2; // 32: max rows of one sub

// ---- Ring depths (only the rings that carry cross-core handshakes) ----
// CA_S_RING: how many QK beats AIC may run ahead of AIV. CA_P_RING: how many softmax beats AIV may
// run ahead of AIC's PV.
constexpr uint32_t CA_S_RING = 4;
// QK lead over PV (AIC prologue depth). Must stay equal to the S ring depth: a deeper prologue lets QK
// take CUBE issue slots from PV and loses throughput.
constexpr uint32_t CA_QK_LEAD = CA_S_RING;
// P ring = one S ring block: while AIC pre-issues QK, AIV softmaxes the whole block, so P is ready
// before PV starts and PV never stalls on P_READY.
constexpr uint32_t CA_P_RING = 4;

// ---- L1 K/V panel ring slots ----
// Only 160 KB of the 512 KB L1 is used and K/V each have a single slot, so MTE2 has to wait for the
// previous beat's MTE1 read and the GM latency sits directly on the QK/PV chain. K and V are therefore
// multi-slotted and prefetched ahead of use:
//   K: 4 slots = one S ring block, prefetched as a block while QK runs;
//   V: 2 slots, consumed by PV beat by beat with a 2 beat rolling prefetch.
// There are only 8 event ids per direction and K 4 + V 2 + Q 1 + P return 1 fills them exactly (see the
// id table inside the kernel). Two beats of lead still cover one L1 panel of GM latency.
constexpr uint32_t CA_K_L1_SLOTS = 4;
constexpr uint32_t CA_V_L1_SLOTS = 2;

// ---- Numeric constants ----
constexpr float CA_MINV = -3.4e38f;
constexpr float CA_BIGV = 3.0e38f;
// Upper clamp for s*scale. m is fixed at the beat 0 block maximum and reused for the whole task, so a
// later beat can exceed it; the clamp keeps exp() bounded in fp32 and the fp16 result stable after the
// final l normalisation. Tighten it to 30.0f for a wider margin.
constexpr float CA_SOFT_CLAMP = 60.0f;

// ---- Cross-core flag ids ----
// AIC and its AIV pair handshake with mode 4 (intra block, AIC<->AIV event registers), one id per sub:
// every id has exactly one producer and one consumer. mode only selects the set/reset channel; the
// ordering guarantee comes from the flag counters plus the explicit reverse dependencies in the kernel.
constexpr uint16_t CA_FLAG_SUB_STRIDE = 16; // same as the official matmul: sub0 0..15, sub1 16..31
constexpr uint16_t CA_FLAG_S_READY = 0;     // AIC -> AIV(sub): S ring slot written into this sub's UB
constexpr uint16_t CA_FLAG_P_READY = 1;     // AIV(sub) -> AIC: P written into the L1 A1 ring slot
constexpr uint16_t CA_FLAG_O_READY = 2;     // AIC -> AIV(sub): O written into this sub's UB
constexpr uint16_t CA_FLAG_O_FREE = 3;      // AIV(sub) -> AIC: O read, the buffer may be overwritten
// AIV has read the S ring slot into UB, AIC may overwrite it (recycles QK pre-issue credit)
constexpr uint16_t CA_FLAG_S_FREE = 4;      // AIV(sub) -> AIC
// AIC has loaded the P ring slot into L0A, AIV may overwrite it
constexpr uint16_t CA_FLAG_P_FREE = 5;      // AIC -> AIV(sub)

// ---- Static L1 addresses (bytes, LocalTensor bufferOffset semantics) ----
// QK and PV use separate L1/L0 buffers. The two only serialise inside one beat and never wait across
// beats; sharing them would make PV(i) and QK(i+CA_S_RING) contend and cancel the pre-issue win.
constexpr uint32_t CA_A1_ADDR_Q = 0;              // Q  [R x 128], 16 KB
constexpr uint32_t CA_K_SLOT_BYTES = CA_KV_TILE * CA_HEAD_DIM * 2; // 32 KB
constexpr uint32_t CA_B1_ADDR_Q = 32 * 1024;      // K: CA_K_L1_SLOTS slots of 32 KB (32..160 KB)
constexpr uint32_t CA_A1_ADDR_P = 160 * 1024;     // P: CA_P_RING slots, R rows each (160..224 KB)
constexpr uint32_t CA_P_SLOT_BYTES = CA_MAX_R * CA_HEAD_DIM * 2; // 16 KB of T; each sub writes its rows
constexpr uint32_t CA_V_SLOT_BYTES = CA_KV_TILE * CA_HEAD_DIM * 2; // 32 KB
constexpr uint32_t CA_B1_ADDR_P = 256 * 1024;     // V: CA_V_L1_SLOTS slots of 32 KB (256..320 KB)
// L1 use is 272 KB of 512 KB: Q 16 + K 128 + P 64 + V 64, and the static addresses end at 320 KB.
// L0 double buffering: L0B is only 64 KB, so the QK K panel and the PV V panel share one pair of 32 KB
// slots and alternate globally per beat (AIC slot counter), which lets the next L1->L0 copy hide under
// the previous Mmad. L0A is 64 KB: the Q panel (16 KB) is loaded once per task, P has two 16 KB slots.
constexpr uint32_t CA_A2_ADDR_Q = 0;              // Q  [R x 128], loaded once per task
constexpr uint32_t CA_A2_ADDR_P = 32 * 1024;      // P: two 16 KB slots
constexpr uint32_t CA_A2_P_SLOT_BYTES = CA_MAX_R * CA_HEAD_DIM * 2; // 16 KB
constexpr uint32_t CA_B2_SLOT_BYTES = 32 * 1024;  // [128 x 128] of T = 32 KB, two slots
constexpr uint32_t CA_CO1_ADDR_Q = 0;             // L0C: S [R x 128] fp32 = 32 KB, double buffered
constexpr uint32_t CA_CO1_S_SLOT_BYTES = CA_MAX_R * CA_HEAD_DIM * sizeof(float); // 32 KB
constexpr uint32_t CA_CO1_ADDR_P = 64 * 1024;     // L0C: O [R x 128] fp32 = 32 KB

// A1 / L0A window element count: 64 rows x 128 depth
constexpr uint32_t CA_A_ELEMS = CA_MAX_R * CA_HEAD_DIM;
// B1 / L0B window element count: 128 x 128
constexpr uint32_t CA_B_ELEMS = CA_KV_TILE * CA_HEAD_DIM;

// ---- AIV UB layout (byte offsets; both subs use the same offsets on separate UB) ----
// Every region uses explicit byte offsets instead of TPipe / InitBuffer: the Fixpipe destination and
// the regbase __local_mem__ pointers are computed from the offsets, so both sides share one set.
constexpr uint32_t CA_S_SLOT_BYTES = CA_UB_ROWS * CA_HEAD_DIM * sizeof(float); // 16 KB
constexpr uint32_t CA_UB_S = 0;                                        // S ring: CA_S_RING x 16 KB
constexpr uint32_t CA_UB_P = CA_UB_S + CA_S_RING * CA_S_SLOT_BYTES;    // P: two slots of 8 KB of T
// P has two UB slots: P is the last vector write of a beat and is copied to L1 right after it. With one
// slot the next beat's vector write must wait for that copy, serialising V and MTE3; with two slots the
// AIV beat chain becomes max(V, MTE3) instead of V + MTE3.
constexpr uint32_t CA_P_UB_SLOTS = 2;
constexpr uint32_t CA_UB_P_SLOT_BYTES = CA_UB_ROWS * CA_HEAD_DIM * 2;
constexpr uint32_t CA_UB_P_BYTES = CA_P_UB_SLOTS * CA_UB_P_SLOT_BYTES;
constexpr uint32_t CA_UB_O = CA_UB_P + CA_UB_P_BYTES;                  // O: 16 KB of fp32
constexpr uint32_t CA_UB_O_BYTES = CA_UB_ROWS * CA_HEAD_DIM * sizeof(float);
constexpr uint32_t CA_UB_LACC = CA_UB_O + CA_UB_O_BYTES;               // l accumulator: 16 KB
constexpr uint32_t CA_UB_OUT = CA_UB_LACC + CA_UB_O_BYTES;             // output staging: 8 KB of T
constexpr uint32_t CA_UB_MASK = CA_UB_OUT + CA_UB_P_BYTES;             // mask rows: 2 x 512 B
constexpr uint32_t CA_UB_SCAL = CA_UB_MASK + 1024;                     // scalar area: 1024 B
constexpr uint32_t CA_UB_LSUM = CA_UB_SCAL + 1024;                     // row sums: CA_UB_ROWS x 64 float
constexpr uint32_t CA_UB_ML = CA_UB_LSUM + CA_UB_ROWS * 64 * sizeof(float); // merge m/l (2*R per sub)
// Merge accumulator, used only in the kSeg>1 epilogue and sized for this sub's CA_UB_ROWS rows.
constexpr uint32_t CA_UB_MRG = CA_UB_ML + 2048;
// Total UB use is about 140 KB, comfortably inside the 256 KB budget of one sub.
static_assert(CA_UB_MRG + CA_UB_ROWS * CA_HEAD_DIM * sizeof(float) <= 256u * 1024u, "AIV UB over budget");

__aicore__ inline uint32_t CaCeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

__aicore__ inline uint32_t CaAlignUp(uint32_t v, uint32_t a)
{
    return CaCeilDiv(v, a) * a;
}

// ---- Sub row split (AIC Fixpipe mSize and the AIV UB row range must match exactly) ----
__aicore__ inline uint32_t CaSubHalf(uint32_t rows)
{
    return CaAlignUp((rows + 1u) >> 1, CA_BLOCK_CUBE);
}

__aicore__ inline uint32_t CaSubBegin(uint32_t rows, uint32_t sub)
{
    if (sub == 0u) {
        return 0u;
    }
    const uint32_t half = CaSubHalf(rows);
    return half < rows ? half : rows;
}

__aicore__ inline uint32_t CaSubRows(uint32_t rows, uint32_t sub)
{
    const uint32_t begin = CaSubBegin(rows, sub);
    if (begin >= rows) {
        return 0u;
    }
    const uint32_t half = CaSubHalf(rows);
    const uint32_t remain = rows - begin;
    return remain < half ? remain : half;
}

#endif // CROSS_ATTENTION_COMMON_H
