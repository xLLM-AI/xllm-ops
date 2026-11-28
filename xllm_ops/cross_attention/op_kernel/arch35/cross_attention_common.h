/**
 * CrossAttention kernel 公共常量与工具（host / device 共用常量，无 device 专属类型）。
 */
#ifndef CROSS_ATTENTION_COMMON_H
#define CROSS_ATTENTION_COMMON_H

#include <cstdint>
#include "cross_attention_tiling.h"

// ---- 几何常量 ----
constexpr uint32_t CA_HEAD_DIM = 128;
constexpr uint32_t CA_KV_TILE = 128;
constexpr uint32_t CA_C0 = 16;          // 32B / sizeof(bf16|fp16)
constexpr uint32_t CA_BLOCK_CUBE = 16;  // cube 分形行数
// 单任务的 q 行上限（一个任务 = 一次 K/V 面板复用）。必须是 CA_BLOCK_CUBE 的整数倍：
// cube 侧按 16 行一个 fracta 分块做 Mmad / Fixpipe，K/V 面板只搬一次，供所有分块复用。
// R=128 曾试过：固定开销确实摊薄，但 q=64 b=8 形状下任务数 128->64，28 核的
// makespan 从 5 个 64 行任务变成 3 个 128 行任务，负载不均反而净退化 4%，故回到 64。
constexpr uint32_t CA_MAX_R = 64;
constexpr uint32_t CA_R_BLOCKS = CA_MAX_R / CA_BLOCK_CUBE;

// ---- M2：CV 直通（不走 GM staging）+ online softmax ----
// S 在 AIC 的 L0C 里算完直接 Fixpipe 进两个 AIV sub 各自的 UB；AIV 的 softmax 结果 P
// 直接 UB -> L1（cube 的 A1 窗口）；每拍的 PV 也在 L0C 出结果后逐拍 Fixpipe 进 UB 的
// PV 环，AIV 侧维护 O_run = O_run * dm + PV（dm=exp(gm_old-gm_new)，gm 为逐行运行最大），
// O 不再在 L0C 里跨拍累加（L0C 累加器无法按行乘 dm，这是 online 数值口径的必然代价）。
// 因此每个 AIV sub 只承担 R 行的一半，且按 16 行对齐切分（Fixpipe 的 mSize 必须以
// 分形行为单位，L1 侧的 P 也按 16 行一个 NZ 分块写入）。
constexpr uint32_t CA_UB_ROWS = CA_MAX_R / 2; // 32：单个 sub 的最大行数

// ---- 环深（只保留跨核握手的环）----
// S 环深 = AIC 相对 AIV 的 QK 预发射拍数；P 环深 = AIV 的 softmax 相对 AIC 的 PV 的领先拍数。
constexpr uint32_t CA_S_RING = 4;
// QK 相对 PV 的领先拍数（AIC 侧序言深度）。实测只有等于 S 环深时最优：
// 加深到 8 拍会让 QK 抢走 CUBE 上 PV 的发射窗口（kv=700 29.5 -> 32.3 µs）。
constexpr uint32_t CA_QK_LEAD = CA_S_RING;
// P 环 4 拍 = 一个 S 环块：AIC 预发射 QK 期间 AIV 能连续做完整块的 softmax，
// 等 PV 开始时 P 已全部就绪，PV 不会卡在 P_READY 上。
constexpr uint32_t CA_P_RING = 4;

// ---- L1 侧 K/V 面板环槽 ----
// L1 只用了 160 KB / 512 KB，K/V 都只有单槽，MTE2 必须等上一拍 MTE1 读完才能搬下一拍，
// GM 延迟直接串在 QK/PV 链上。这里给 K/V 各开多槽，并把副本提前发出去：
//   K：4 槽 = 一个 S 环块，整块预取，QK 计算期间 MTE2 已经把后面几拍的 K 全部取回；
//   V：2 槽，PV 逐拍消费，提前 2 拍滚动预取。
// 事件 ID 同类型内只有 8 个：K 4 槽 + V 2 槽 + Q 1 个 + P 回程 1 个刚好排满（见 kernel 内 id 表）。
// V 从 3 槽减到 2 槽，是为了把 MTE1_MTE2 / MTE2_MTE1 各空出 id 3 给 P 的 GM->L1 回程用；
// 提前 2 拍（约 2.8 µs）仍然覆盖一个 L1 面板的 GM 时延。
constexpr uint32_t CA_K_L1_SLOTS = 4;
constexpr uint32_t CA_V_L1_SLOTS = 2;

// ---- online softmax：PV 环 ----
// 每拍 PV 在 L0C 出结果后逐拍 Fixpipe 进 AIV 的 UB 环槽，AIV 隔一拍做
// O_run = O_run * dm + PV（dm 按行落在 CA_UB_DM 环槽，与本拍 PV 同号配对）。
// 环深 2：本拍 fixpipe 可以压在 AIV 当前拍 softmax 之下，任务末的补消费见
// CrossAttentionAiv::RunTask 的 OUpdateBeat 收尾。
constexpr uint32_t CA_PV_RING = 2;
constexpr uint32_t CA_PV_SLOT_BYTES = CA_UB_ROWS * CA_HEAD_DIM * sizeof(float); // 16 KB（每 sub）
constexpr uint32_t CA_DM_SLOTS = 2;
constexpr uint32_t CA_DM_SLOT_BYTES = CA_UB_ROWS * 64 * sizeof(float); // 8 KB：每行 64 float，落位口径同 CA_UB_ROWM

// ---- 数值常量 ----
constexpr float CA_MINV = -3.4e38f;
constexpr float CA_BIGV = 3.0e38f;
// P 的上钳跨度：online 口径下 gm 是逐行运行最大，减完之后有效列的 exp 自变量恒 <= 0，
// 这个上钳只剩保险作用（防标量/掩码异常路径），不再是截断手段。无效列一律压 CA_MINV。
// 数值窗口仍按 16bit 可表示域取：
//   fp16：可表示 [2^-24, 65504] => 窗口 [-16.6, +11.09]，取 10（exp=22026）；
//   bf16：窗口约 [-92, +88]，保持 60（exp=1.1e26）不变。
// 只影响 P 的截断行为，不涉及任何地址或环深布局。
constexpr float CA_SOFT_CLAMP_FP16 = 10.0f;
constexpr float CA_SOFT_CLAMP_BF16 = 60.0f;

// ---- 核间 flag id ----
// AIC 与本组 AIV 之间用 mode 4（intra block，AIC↔AIV 事件寄存器）握手，且按 sub 分 id：
// 每个 id 只有一个生产者、一个消费者，配对关系一目了然。注意 mode 只是“置位通道”的选择，
// 真正保证数据先后的是 flag 自身的计数语义 + 各 pipe 的显式反向依赖（见 kernel 内注释）。
constexpr uint16_t CA_FLAG_SUB_STRIDE = 16; // 与官方 matmul 一致：sub0 用 0..15，sub1 用 16..31
constexpr uint16_t CA_FLAG_S_READY = 0;     // AIC -> AIV(sub)：S 环槽已写进该 sub 的 UB
constexpr uint16_t CA_FLAG_P_READY = 1;     // AIV(sub) -> AIC：P 已写进 L1 的 A1 环槽
constexpr uint16_t CA_FLAG_PV_READY = 2;    // AIC -> AIV(sub)：本拍 PV 已 Fixpipe 进该 sub 的 UB 环槽
constexpr uint16_t CA_FLAG_PV_FREE = 3;     // AIV(sub) -> AIC：本拍 PV 已并入 O_run，环槽可覆写
// AIV 已经把 S 环槽读进 UB，AIC 可以覆写该槽（用于 QK 预发射的信用回收）
constexpr uint16_t CA_FLAG_S_FREE = 4;      // AIV(sub) -> AIC
// AIC 已经把 P 的 L1 环槽读进 L0A，AIV 可以覆写该槽
constexpr uint16_t CA_FLAG_P_FREE = 5;      // AIC -> AIV(sub)

// ---- L1 静态地址（字节，LocalTensor 的 bufferOffset 语义）----
// QK 与 PV 各用一组独立的 L1/L0 缓冲：两组只在同一拍内串行，跨拍互不等待，
// 否则 PV(i) 与 QK(i+CA_S_RING) 会争用同一块 L1/L0，把预发射的收益吃掉。
constexpr uint32_t CA_A1_ADDR_Q = 0;              // Q  [R x 128] 16 KB
constexpr uint32_t CA_K_SLOT_BYTES = CA_KV_TILE * CA_HEAD_DIM * 2; // 32 KB
constexpr uint32_t CA_B1_ADDR_Q = 32 * 1024;      // K  CA_K_L1_SLOTS 个 32 KB 槽（32 ~ 160 KB）
constexpr uint32_t CA_A1_ADDR_P = 160 * 1024;     // P  CA_P_RING 个槽，每槽整任务 R 行（160 ~ 224 KB）
constexpr uint32_t CA_P_SLOT_BYTES = CA_MAX_R * CA_HEAD_DIM * 2; // 16 KB（T，两个 sub 各写自己的行）
constexpr uint32_t CA_V_SLOT_BYTES = CA_KV_TILE * CA_HEAD_DIM * 2; // 32 KB
constexpr uint32_t CA_B1_ADDR_P = 256 * 1024;     // V  CA_V_L1_SLOTS 个 32 KB 槽（256 ~ 352 KB）
// L1 占用 272 KB / 512 KB：Q 16 + K 128 + P 64 + V 64（静态地址排到 320 KB 处）。
// L0 双缓冲：L0B 只有 64 KB，QK 的 K 面板与 PV 的 V 面板共用同一对 32 KB 槽，
// 按拍全局奇偶交替（AIC 侧 slot 计数器），这样下一拍的 L1->L0 搬运可以压在上一拍 Mmad 之下。
// L0A 64 KB：Q 面板 16 KB 任务级只装一次；P 面板 2 个 16 KB 槽做双缓冲。
constexpr uint32_t CA_A2_ADDR_Q = 0;              // Q  [R x 128] 16 KB（每任务只装一次）
constexpr uint32_t CA_A2_ADDR_P = 32 * 1024;      // P  2 个 16 KB 槽
constexpr uint32_t CA_A2_P_SLOT_BYTES = CA_MAX_R * CA_HEAD_DIM * 2; // 16 KB
constexpr uint32_t CA_B2_SLOT_BYTES = 32 * 1024;  // [128 x 128] T = 32 KB，共 2 槽
constexpr uint32_t CA_CO1_ADDR_Q = 0;             // L0C：S [R x 128] fp32 = 32 KB，按拍奇偶双缓冲
constexpr uint32_t CA_CO1_S_SLOT_BYTES = CA_MAX_R * CA_HEAD_DIM * sizeof(float); // 32 KB
constexpr uint32_t CA_CO1_ADDR_P = 64 * 1024;     // L0C：PV [R x 128] fp32 = 32 KB，按拍奇偶双缓冲
                                                  // （online：本拍 Mmad 只等上一拍 fixpipe 读完，不跨拍累加）
constexpr uint32_t CA_CO1_P_SLOT_BYTES = CA_CO1_S_SLOT_BYTES; // 32 KB
// L0C 占用 128 KB：S 2 x 32 + PV 2 x 32。

// A1 / L0A 窗口元素数：64 行 x 128 深
constexpr uint32_t CA_A_ELEMS = CA_MAX_R * CA_HEAD_DIM;
// B1 / L0B 窗口元素数：128 x 128
constexpr uint32_t CA_B_ELEMS = CA_KV_TILE * CA_HEAD_DIM;

// ---- AIV 侧 UB 布局（字节偏移，两个 sub 各自一套同样的偏移，物理 UB 相互独立）----
// 全部用显式字节偏移，不用 TPipe / InitBuffer：Fixpipe 的 UB 目的地址、regbase 的
// __local_mem__ 指针都直接按偏移算，便于和 AIC 侧共享同一份常量。
constexpr uint32_t CA_S_SLOT_BYTES = CA_UB_ROWS * CA_HEAD_DIM * sizeof(float); // 16 KB
constexpr uint32_t CA_UB_S = 0;                                        // S 环：CA_S_RING x 16 KB
constexpr uint32_t CA_UB_P = CA_UB_S + CA_S_RING * CA_S_SLOT_BYTES;    // P：2 x 8 KB（T）双槽
// P 的 UB 缓冲双槽：P 是 AIV 每拍的最后一笔向量写，紧接着就要 MTE3 搬去 GM。
// 单槽时下一拍的向量必须等这笔 MTE3 读完才能覆写，V 与 MTE3 硬串行；
// 双槽后下一拍的向量可以压在上一拍 P 的 MTE3 之下，AIV 的每拍链从 V+MTE3 变成 max(V, MTE3)。
constexpr uint32_t CA_P_UB_SLOTS = 2;
constexpr uint32_t CA_UB_P_SLOT_BYTES = CA_UB_ROWS * CA_HEAD_DIM * 2;
constexpr uint32_t CA_UB_P_BYTES = CA_P_UB_SLOTS * CA_UB_P_SLOT_BYTES;
constexpr uint32_t CA_UB_O = CA_UB_P + CA_UB_P_BYTES;                  // O_run：online 累加器：16 KB（fp32）
constexpr uint32_t CA_UB_O_BYTES = CA_UB_ROWS * CA_HEAD_DIM * sizeof(float);
// 原 CA_UB_LACC（l 逐拍元素级累加）移除：online 口径下 l 改按行维护
// （lsum = lsum * dm + rowsum），直接落 CA_UB_LSUM，任务末不再 ReduceRows。
constexpr uint32_t CA_UB_OUT = CA_UB_O + CA_UB_O_BYTES;                // 输出暂存：8 KB（T，区段仍留 16 KB）
constexpr uint32_t CA_UB_MASK = CA_UB_OUT + CA_UB_P_BYTES;             // 上钳掩码行：2 x 512 B
constexpr uint32_t CA_UB_SCAL = CA_UB_MASK + 1024;                     // 标量区：1024 B
constexpr uint32_t CA_UB_LSUM = CA_UB_SCAL + 1024;                     // 行和：CA_UB_ROWS x 64 float
constexpr uint32_t CA_UB_ML = CA_UB_LSUM + CA_UB_ROWS * 64 * sizeof(float); // merge：m/l 区（每 sub 2*R）
// merge 累加器只在 kSeg>1 的收尾阶段用，按本 sub 的 CA_UB_ROWS 行分配
constexpr uint32_t CA_UB_MRG = CA_UB_ML + 2048;
constexpr uint32_t CA_UB_MRG_BYTES = CA_UB_ROWS * CA_HEAD_DIM * sizeof(float);
// 每行 softmax 的运行最大 gm：online 口径每拍折叠更新。按 64 float/行落地，便于用
// DIST_BRC_B32 广播读，布局与 CA_UB_LSUM 的行和同构。新块追加在尾部，已有地址不动。
constexpr uint32_t CA_UB_ROWM = CA_UB_MRG + CA_UB_MRG_BYTES;
constexpr uint32_t CA_UB_ROWM_BYTES = CA_UB_ROWS * 64 * sizeof(float);
constexpr uint32_t CA_UB_ROWM_ELEMS = CA_UB_ROWS * 64; // 每行 64 float（列 0 有效，广播读）
// online 新增区段（追加在尾部）：
//   MASKB：max pass 的掩码行（有效列 CA_BIGV、无效列 CA_MINV），与上钳掩码同拍各建一份
//   PV 环：AIC 每拍 L0C->UB 的 Fixpipe 落点，CA_PV_RING 个 16 KB 槽
//   DM 环：每行 dm=exp(gm_old-gm_new)，CA_DM_SLOTS 个 8 KB 槽（行落位同 CA_UB_ROWM），与 PV 环同号配对
constexpr uint32_t CA_UB_MASKB = CA_UB_ROWM + CA_UB_ROWM_BYTES;
constexpr uint32_t CA_UB_PV = CA_UB_MASKB + 1024;
constexpr uint32_t CA_UB_DM = CA_UB_PV + CA_PV_RING * CA_PV_SLOT_BYTES;
constexpr uint32_t CA_UB_TOTAL = CA_UB_DM + CA_DM_SLOTS * CA_DM_SLOT_BYTES; // 约 213 KB < 256 KB

__aicore__ inline uint32_t CaCeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

__aicore__ inline uint32_t CaAlignUp(uint32_t v, uint32_t a)
{
    return CaCeilDiv(v, a) * a;
}

// ---- sub 行切分（AIC 的 Fixpipe mSize 与 AIV 的 UB 行范围必须完全一致）----
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
