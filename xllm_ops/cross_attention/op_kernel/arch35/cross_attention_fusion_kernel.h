/**
 * CrossAttention kernel 主体：AIC（QK / PV）与 AIV（softmax / 归一化）。
 *
 * M2 形态：CV 直通，S / P / PV 全部不走 GM staging。
 *   QK：cube 结果留在 L0C，直接 Fixpipe 进两个 AIV sub 各自的 UB（S 环）
 *   softmax：AIV 用 RegBase（__VEC_SCOPE__ + RegTensor）在 UB 里做 online 折叠
 *            （gm=max(gm,块最大)，dm=exp(gm_old-gm_new)，P=exp(s*scale-gm)），
 *            P 直接 UB -> L1 的 A1 环槽（CopyUbufToCbuf，行主序转 NZ 分形）
 *   PV：cube 从 L1 的 A1 环槽直接 LoadData 到 L0A；每拍的 PV 单独出 L0C，
 *       逐拍 Fixpipe 进两个 sub 的 UB（PV 环），AIV 隔拍合并
 *       O_run = O_run * dm + PV；任务末 out = O_run / lsum 写回 GM
 * beat 流水（每个 KV tile 一个 beat）：
 *   AIC 按 CA_S_RING 分块预发射 QK，再逐拍 PV（P_READY 驱动）+ PV Fixpipe（PV_READY 驱动）
 *   AIV 逐拍 softmax，P 环 CA_P_RING 深，靠 P_FREE 回收 L1 槽；
 *   PV 环 CA_PV_RING 深，靠 PV_FREE 回收 UB 槽
 * 数值口径：online softmax（对齐 A3 / FlashAttention 的 running-max 合并）——
 *   每行维护运行最大 gm 与行和 lsum，逐拍 lsum = lsum * dm + 本拍行和；
 *   O_run 逐拍 O_run = O_run * dm + PV。exp 自变量恒 <= 0，无上钳截断误差。
 */
#ifndef CROSS_ATTN_ARCH35_FUSION_KERNEL_H
#define CROSS_ATTN_ARCH35_FUSION_KERNEL_H

#include "kernel_operator.h"
#include "cross_attention_common.h"

namespace ca {

// arch35 侧 tiling 结构定义在 Xllm 命名空间（op_kernel/arch35/cross_attention_tiling.h），
// host tiling 序列化与 kernel 读取共用同一 128B POD 布局
using Xllm::CrossAttentionTilingData;

using AscendC::Add;
using AscendC::Cast;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopy;
using AscendC::DataCopyParams;
using AscendC::Duplicate;
using AscendC::Fixpipe;
using AscendC::FixpipeParamsArch3510;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::LoadData;
using AscendC::LoadData2DParams;
using AscendC::LoadData2DParamsV2;
using AscendC::LoadData2dTransposeParams;
using AscendC::LoadDataWithTranspose;
using AscendC::Mmad;
using AscendC::MmadParams;
using AscendC::Muls;
using AscendC::Nd2NzParams;
using AscendC::PipeBarrier;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;

// RegBase 命名空间：AscendC::MicroAPI 只是 AscendC::Reg 的别名，且该别名仅在
// __NPU_ARCH__==3510 的 pass 里生效；这里直接用 Reg，host pass 也能解析类型。
namespace Mapi = AscendC::Reg;

// L0C -> UB 的 Fixpipe 配置：isFixPixelMerge（第二个参数）置 true 走 UB 目的地址
constexpr AscendC::FixpipeConfig CA_CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

// RegBase 的 Cast trait：fp32 -> 16bit。fp16 用 NO_SAT（softmax 输出必然在 [0,1]），
// bf16 用 SAT（官方 arch35 算子的口径）。
constexpr Mapi::CastTrait CA_CAST_F32_TO_FP16 = {Mapi::RegLayout::ZERO, Mapi::SatMode::NO_SAT,
                                                 Mapi::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr Mapi::CastTrait CA_CAST_F32_TO_BF16 = {Mapi::RegLayout::ZERO, Mapi::SatMode::SAT,
                                                 Mapi::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// ---- UB 标量区内部偏移（相对 CA_UB_SCAL）----
constexpr uint32_t CA_SCAL_RED = 0;   // 64 float：ReduceMax/Exp 的落地缓冲
constexpr uint32_t CA_SCAL_TMP = 384; // 32 float：merge 的分母
constexpr uint32_t CA_SCAL_MRG = 512; // 64 float：merge 的组内最大 m*

// 任务描述（AIC / AIV 各自从 tiling 独立解析）
struct CaTaskDesc {
    uint32_t batchIdx;
    uint32_t kvHeadIdx;
    uint32_t qRowTileIdx;
    uint32_t gTileIdx;  // group 内的 head 分档下标（groupSize <= 16 时恒为 0）
    uint32_t kvSegIdx;
    uint32_t nQ;      // 本 tile 实际的 q 行数
    uint32_t g;       // 本 tile 覆盖的 head 数 = min(gPerTile, groupSize - gBase)
    uint32_t rows;    // R = nQ * g
    uint32_t kvLoops; // beat 数
    uint32_t segStartBlock;
    uint32_t segValid;
    uint64_t qHeadBase; // q 行 0、本档首个 head 的元素偏移
};

__aicore__ inline void ParseTask(const __gm__ CrossAttentionTilingData* t, uint32_t taskIdx,
                                 const GlobalTensor<int64_t>& lensGm, CaTaskDesc& d)
{
    const uint32_t kSeg = t->kSeg;
    d.kvSegIdx = taskIdx % kSeg;
    const uint32_t base = taskIdx / kSeg;
    d.gTileIdx = base % t->gTileNum;
    const uint32_t rowBase = base / t->gTileNum;
    d.qRowTileIdx = rowBase % t->qRowTileNum;
    d.kvHeadIdx = (rowBase / t->qRowTileNum) % t->kvHeads;
    d.batchIdx = rowBase / (t->qRowTileNum * t->kvHeads);

    // 本 task 只覆盖 group 内从 gBase 起的 g 个 head（groupSize <= 16 时 gBase = 0、g = groupSize）
    const uint32_t gBase = d.gTileIdx * t->gPerTile;
    d.g = t->groupSize - gBase;
    if (d.g > t->gPerTile) {
        d.g = t->gPerTile;
    }

    const uint32_t qRowStart = d.qRowTileIdx * t->qRowsPerTile;
    d.nQ = t->qSeqlen - qRowStart;
    if (d.nQ > t->qRowsPerTile) {
        d.nQ = t->qRowsPerTile;
    }
    d.rows = d.nQ * d.g;

    d.segStartBlock = d.kvSegIdx * t->mergeSegBlocks;
    uint32_t segBlocksThis = 0;
    if (d.segStartBlock < t->maxNumBlocksPerBatch) {
        segBlocksThis = t->maxNumBlocksPerBatch - d.segStartBlock;
        if (segBlocksThis > t->mergeSegBlocks) {
            segBlocksThis = t->mergeSegBlocks;
        }
    }
    const int64_t kvLen = lensGm.GetValue(static_cast<int64_t>(d.batchIdx));
    int64_t valid = kvLen - static_cast<int64_t>(d.segStartBlock) * static_cast<int64_t>(t->blockSize);
    if (valid < 0) {
        valid = 0;
    }
    const int64_t cap = static_cast<int64_t>(segBlocksThis) * static_cast<int64_t>(t->blockSize);
    if (valid > cap) {
        valid = cap;
    }
    d.segValid = static_cast<uint32_t>(valid);
    d.kvLoops = CaCeilDiv(d.segValid, t->kvTile);

    d.qHeadBase = ((static_cast<uint64_t>(d.batchIdx) * t->qSeqlen + qRowStart) * t->qHeads +
                   static_cast<uint64_t>(d.kvHeadIdx) * t->groupSize + gBase) *
                  t->headDim;
}

// host 的 kSeg 只能按 maxNumBlocksPerBatch（容量）决定，实际 kvLen 小的时候会多切出
// 若干空段，而 kSeg > 1 的代价是两个全局栅栏 + 一拍 merge 往返（实测 b=1 小 kv 约 6~8 µs）。
// 各核用同一份 lens 推出「生效段数」：实际 block 数不足一段时直接退回 kSeg==1 的收尾路径。
__aicore__ inline uint32_t CaEffSeg(const __gm__ CrossAttentionTilingData* t,
                                    const GlobalTensor<int64_t>& lensGm)
{
    const uint32_t kSeg = t->kSeg;
    if (kSeg <= 1u) {
        return 1u;
    }
    uint32_t maxBlocks = 0u;
    for (uint32_t b = 0; b < t->batch; ++b) {
        int64_t len = lensGm.GetValue(static_cast<int64_t>(b));
        if (len < 0) {
            len = 0;
        }
        const uint32_t blocks = CaCeilDiv(static_cast<uint32_t>(len), t->blockSize);
        if (blocks > maxBlocks) {
            maxBlocks = blocks;
        }
    }
    const uint32_t need = CaCeilDiv(maxBlocks, t->mergeSegBlocks);
    if (need <= 1u) {
        return 1u;
    }
    return need < kSeg ? need : kSeg;
}

// ---------------------------------------------------------------------------
// AIC：QK / PV
// ---------------------------------------------------------------------------
template <typename T>
class CrossAttentionAic {
public:
    __aicore__ inline void Init(const __gm__ CrossAttentionTilingData* t, __gm__ uint8_t* query, __gm__ uint8_t* key,
                                __gm__ uint8_t* value, __gm__ uint8_t* blockTable, __gm__ uint8_t* actualKvLens,
                                __gm__ uint8_t* workspace)
    {
        // M2 不再需要 S / O 的 staging workspace；P 也走 CV 通道进 L1，workspace 里只剩 merge 槽
        tiling_ = t;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value));
        btGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
        lensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(actualKvLens));
        coreIdx_ = GetBlockIdx();
        l0Slot_ = 0u;
        pvASlot_ = 0u;
        vHead_ = false;
        effSeg_ = CaEffSeg(t, lensGm_);
    }

    __aicore__ inline void Process()
    {
        PrimeBeatFlags();

        // 任务链按“上一任务尾块 QK 一发出就预约下一任务的前置搬运”组织：
        // 任务 j 的 Q 面板与 K 首窗在任务 j-1 的 PV 期间就已经在 MTE2 上跑，
        // 任务边界的 GM 延迟不再串在 QK 链之前（原来每次切任务要等 Q + 4 个 K 面板落地）。
        // 首个任务同样走这条路：Q / K / V 三段一起发，不再 Q 落地后才发 K。
        CaTaskDesc cur;
        uint32_t taskIdx = coreIdx_;
        bool haveCur = false;
        while (taskIdx < tiling_->totalTaskNum) {
            ParseTask(tiling_, taskIdx, lensGm_, cur);
            if (cur.kvLoops != 0u) {
                haveCur = true;
                break;
            }
            taskIdx += tiling_->usedCoreNum;
        }
        if (haveCur) {
            // 冷启动无任何在途计算，V 首窗可以并进前置搬运一起发（跨任务时不行：V 槽还被本任务占着）
            PrefetchFront(cur, true);
        }
        while (haveCur) {
            uint32_t nextIdx = taskIdx + tiling_->usedCoreNum;
            CaTaskDesc next;
            bool haveNext = false;
            while (nextIdx < tiling_->totalTaskNum) {
                ParseTask(tiling_, nextIdx, lensGm_, next);
                if (next.kvLoops != 0u) {
                    haveNext = true;
                    break;
                }
                nextIdx += tiling_->usedCoreNum;
            }
            RunTask(cur, haveNext ? &next : nullptr);
            cur = next;
            taskIdx = nextIdx;
            haveCur = haveNext;
        }

        // kSeg > 1：AIV 合并前需要所有核的 merge 槽都写完。SyncAll<true>（AIV-only）在本场景
        // 不保证跨核可见，改用 AIC+AIV 全局栅栏：AIV 经本核 AIC 中转，AIC 之间再做全局同步。
        // 实际 block 数不足一段时 effSeg_ 已退回 1，两侧都不再进这条路径。
        if (effSeg_ > 1u) {
            AscendC::SyncAll<false>();
        }
    }

private:
    __aicore__ inline void RunTask(const CaTaskDesc& d, const CaTaskDesc* next)
    {
        const uint32_t loops = d.kvLoops;
        loops_ = loops;
        // Q 面板已由上一任务尾块（或冷启动）搬进 L1，这里只等落地再切 L0A。
        PrepQ(d);
        // V 的首个窗口只在任务头补：之后由 PvBeat 提前 CA_V_L1_SLOTS-1 拍滚动预取。
        if (!vHead_) {
            PrefetchVHead(d);
        }
        vHead_ = false;
        // 滚动调度（单拍粒度，不再按 S 环块两段式）：
        //   序言：把前一个 S 环深的 QK 全部预发射（AIV 同时做 softmax），并在每拍 QK 之后
        //         把 +CA_S_RING 拍的 K 预取进刚被 SplitB 释放的 L1 槽。
        //         CA_QK_LEAD（序言深度）试过 8 拍：QK 把 CUBE 的发射窗口从 PV 前面抢走，
        //         kv=700 从 29.5 退化到 32.3 µs，故维持与 S 环同深（见设计文档 15.11）。
        //   主体：每拍先做 PV(i)，再发 QK(i+CA_QK_LEAD)。QK 始终领先 PV 一个 S 环深，
        //         所以 P(i+1) 早在 PV(i) 之前就已由 AIV 产出，P 的拍内预取可以跨拍连续生效，
        //         不再像按块调度那样在每个块首拍暴露一次 P_READY。
        //         PV 仍连续推进（QK 排在同一拍末尾），不会打断 CUBE 的 PV 链。
        // L0B 的 K/V 面板共用一对槽，slot 由 l0Slot_ 跨 QK/PV 连续交替。
        const uint32_t pro = (loops < CA_QK_LEAD) ? loops : CA_QK_LEAD;
        for (uint32_t j = 0; j < pro; ++j) {
            if (j >= CA_S_RING) {
                // S 槽 j%CA_S_RING 的上一任占用者（第 j-CA_S_RING 拍）已被 AIV 读走
                WaitSFree();
            }
            QkBeat(d, j);
            NotifySReady();
            // K 的 L1 槽在 QkBeat 的 SplitB 之后归还，这里立刻补 +CA_S_RING 拍的那一份
            if (j + CA_S_RING < loops) {
                PrefetchK(d, j + CA_S_RING);
            }
        }
        for (uint32_t i = 0; i < loops; ++i) {
            // 尾段把下一任务的 Q 与 K 首窗压进 MTE2：此时本任务的 K 槽已全部归还，
            // 剩下的 PV 拍覆盖这 144 KB 的 GM 时间。V 不带（槽仍被本任务尾拍占着）。
            if (next != nullptr && i == ((loops >= 2u) ? (loops - 2u) : 0u)) {
                PrefetchFront(*next, false);
            }
            PvBeat(d, i);
            if (i + pro < loops) {
                // 环槽 (i+pro)%CA_S_RING 的上一任占用者（第 i+pro-CA_S_RING 拍）已被 AIV 读走
                WaitSFree();
                QkBeat(d, i + pro);
                NotifySReady();
                if (i + pro + CA_S_RING < loops) {
                    PrefetchK(d, i + pro + CA_S_RING);
                }
            }
        }

        // 信用排空：S_FREE 每拍产出一份，AIC 只在超出环深时消费，任务结束时补齐不足一个环深的部分，
        // 保证计数跨任务守恒。（P_FREE 方向相反：AIC 是生产者，不需要消费。）
        const uint32_t drainS = loops < CA_S_RING ? loops : CA_S_RING;
        for (uint32_t k = 0; k < drainS; ++k) {
            WaitSFree();
        }

        // PV_FREE 与 S_FREE 同构：AIV 每拍产出一份，AIC 只在超出 PV 环深时消费，
        // 任务末补齐不足一个环深的部分，保证计数跨任务守恒（online：PV 逐拍进 UB，
        // O_run 在 AIV 侧维护，任务末不再有 O 投递 / O_READY / O_FREE）。
        const uint32_t drainPV = loops < CA_PV_RING ? loops : CA_PV_RING;
        for (uint32_t k = 0; k < drainPV; ++k) {
            WaitPvFree();
        }
    }

    // ---- AIC 侧核间握手（mode 4 intra block，按 sub 分 id，见 cross_attention_common.h） ----
    __aicore__ inline void NotifySReady()
    {
        CrossCoreSetFlag<4, PIPE_FIX>(CA_FLAG_S_READY);
        CrossCoreSetFlag<4, PIPE_FIX>(CA_FLAG_S_READY + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void NotifyPvReady()
    {
        CrossCoreSetFlag<4, PIPE_FIX>(CA_FLAG_PV_READY);
        CrossCoreSetFlag<4, PIPE_FIX>(CA_FLAG_PV_READY + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void NotifyPFree()
    {
        CrossCoreSetFlag<4, PIPE_MTE1>(CA_FLAG_P_FREE);
        CrossCoreSetFlag<4, PIPE_MTE1>(CA_FLAG_P_FREE + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitPReady()
    {
        // 默认挂在 PIPE_S（标量）上：CV 直通下这两个等待只用于“P 已写完/已读走”的
        // 顺序，实测挂到 PIPE_MTE1 会在多拍形状上引入死锁，退回默认。
        CrossCoreWaitFlag<4>(CA_FLAG_P_READY);
        CrossCoreWaitFlag<4>(CA_FLAG_P_READY + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitPFree()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_P_FREE);
        CrossCoreWaitFlag<4>(CA_FLAG_P_FREE + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitPvFree()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_PV_FREE);
        CrossCoreWaitFlag<4>(CA_FLAG_PV_FREE + CA_FLAG_SUB_STRIDE);
    }

    // AIV 已完成对 S 环槽的读取（两个 sub 各一份信用）
    __aicore__ inline void WaitSFree()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_S_FREE);
        CrossCoreWaitFlag<4>(CA_FLAG_S_FREE + CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline uint64_t KVElemBase(const CaTaskDesc& d, uint32_t beat) const
    {
        const uint32_t logicalBlock = d.segStartBlock + beat;
        const int32_t physBlock =
            btGm_.GetValue(static_cast<int64_t>(d.batchIdx) * tiling_->maxNumBlocksPerBatch + logicalBlock);
        uint64_t off = static_cast<uint64_t>(physBlock) * tiling_->blockSize * tiling_->kvHeads * tiling_->headDim;
        off += static_cast<uint64_t>(d.kvHeadIdx) * tiling_->headDim;
        return off;
    }

    __aicore__ inline void BuildKV(const CaTaskDesc& d, uint32_t beat, LocalTensor<T>& b1, GlobalTensor<T>& src)
    {
        const uint64_t off = KVElemBase(d, beat);
        Nd2NzParams nd;
        nd.ndNum = 1;
        nd.nValue = CA_KV_TILE;
        nd.dValue = CA_HEAD_DIM;
        nd.srcNdMatrixStride = 0;
        nd.srcDValue = tiling_->kvHeads * tiling_->headDim;
        nd.dstNzC0Stride = CA_KV_TILE;
        nd.dstNzNStride = 1;
        nd.dstNzMatrixStride = 0;
        DataCopy(b1, src[off], nd);
    }

    // A 面板的 L1 布局有两种，由 rows 是否能被 CA_BLOCK_CUBE 整除决定：
    //   interleaved（rows%16==0）：整面板当一个大 NZ 矩阵，分形序 k_block*nbCount+m_block。
    //     Nd2Nz 的 dstNzC0Stride 取 rows（d 方向分块步长 = rows 个 32B），一个描述符搬完整个
    //     面板 —— MTE2 的瓶颈是描述符个数（实测约 0.131 µs/条，与字节数无关），所以这样最省。
    //   blocked（尾块 rows%16!=0）：按 16 行一个分块分别搬、块间连续，每块内 dstNzC0Stride=16。
    //     rows 不是 16 的倍数时 d 方向分块步长会落在分形半中间，只能退回这种搬法。
    static constexpr bool CaInterleaved(uint32_t rows)
    {
        return (rows % CA_BLOCK_CUBE) == 0u;
    }

    __aicore__ inline void BuildQ(const CaTaskDesc& d, LocalTensor<T>& a1)
    {
        const uint32_t g = d.g;
        if (CaInterleaved(d.rows)) {
            // 一个 q 行的 g 个头在 GM 里连续（行内步长 headDim），q 行之间隔 qHeads*headDim；
            // 落到 L1 的整面板 NZ 上，每个矩阵 p 的基址就是 p*g 行，矩阵间步长 g*C0 均匀。
            Nd2NzParams nd;
            nd.ndNum = static_cast<uint16_t>(d.nQ);
            nd.nValue = static_cast<uint16_t>(g);
            nd.dValue = CA_HEAD_DIM;
            nd.srcNdMatrixStride = static_cast<uint64_t>(tiling_->qHeads) * tiling_->headDim;
            nd.srcDValue = CA_HEAD_DIM;
            nd.dstNzC0Stride = static_cast<uint16_t>(d.rows);
            nd.dstNzNStride = 1;
            nd.dstNzMatrixStride = g * CA_C0;
            DataCopy(a1, qGm_[d.qHeadBase], nd);
            return;
        }
        const uint32_t perBlock = (g <= CA_BLOCK_CUBE) ? (CA_BLOCK_CUBE / g) : 1u;
        for (uint32_t p = 0; p < d.nQ; ++p) {
            Nd2NzParams nd;
            nd.ndNum = 1;
            nd.nValue = static_cast<uint16_t>(g);
            nd.dValue = CA_HEAD_DIM;
            nd.srcNdMatrixStride = 0;
            nd.srcDValue = CA_HEAD_DIM;
            nd.dstNzC0Stride = CA_BLOCK_CUBE;
            nd.dstNzNStride = 1;
            nd.dstNzMatrixStride = 0;
            const uint32_t blk = p / perBlock;
            const uint32_t pos = p % perBlock;
            const uint32_t dst = blk * CA_BLOCK_CUBE * CA_HEAD_DIM + pos * g * CA_C0;
            const uint64_t off = d.qHeadBase + static_cast<uint64_t>(p) * tiling_->qHeads * tiling_->headDim;
            DataCopy(a1[dst], qGm_[off], nd);
        }
    }

    // 整面板恒等搬运：L1 的 interleaved 面板（Nd2Nz 的 dstNzC0Stride = rows）与 L0A 的
    // Zn 布局逐字节相同，所以一次 M = rows 的 Mmad 只需要一条描述符：
    //   mStep = 全部 m 分块，kStep = 8 个 k 分形，源/目的跨 k 分形的步长同为 nbCount。
    // 和 SplitA(interleaved=true) 的区别只在目的端：后者把每个 16 行分块摊平，供逐块 Mmad。
    __aicore__ inline void SplitAWhole(LocalTensor<T>& a2, LocalTensor<T>& a1, uint32_t nbCount)
    {
        LoadData2DParamsV2 ld;
        ld.mStartPosition = 0;
        ld.kStartPosition = 0;
        ld.mStep = static_cast<uint16_t>(nbCount);
        ld.kStep = CA_HEAD_DIM / CA_C0;
        ld.srcStride = static_cast<int32_t>(nbCount);
        ld.dstStride = static_cast<uint16_t>(nbCount);
        ld.ifTranspose = false;
        LoadData(a2, a1, ld);
    }

    __aicore__ inline void SplitA(LocalTensor<T>& a2, LocalTensor<T>& a1, uint32_t nbCount, bool interleaved)
    {
        // 每 16 行分块一次 L1->L0A：目的端两种布局下都连续（dstStride=1），
        // 源端 interleaved 时同一 m 分块的 8 个 k 分形相隔 nbCount 个分形。
        LoadData2DParamsV2 ld;
        ld.mStartPosition = 0;
        ld.kStartPosition = 0;
        ld.mStep = 1;
        ld.kStep = CA_HEAD_DIM / CA_C0;
        ld.srcStride = static_cast<int32_t>(interleaved ? nbCount : 1u);
        ld.dstStride = 1;
        ld.ifTranspose = false;
        for (uint32_t nb = 0; nb < nbCount; ++nb) {
            const uint32_t src = interleaved ? (nb * CA_BLOCK_CUBE * CA_C0) : (nb * CA_BLOCK_CUBE * CA_HEAD_DIM);
            const uint32_t dst = nb * CA_BLOCK_CUBE * CA_HEAD_DIM;
            LoadData(a2[dst], a1[src], ld);
        }
    }

    // K 按 (kv, hd) 行主序存入 L1 后即等于 (K=hd, N=kv) 的 Zn 布局，直接搬运
    __aicore__ inline void SplitBForQk(LocalTensor<T>& b2, LocalTensor<T>& b1)
    {
        LoadData2DParams ld;
        ld.repeatTimes = (CA_KV_TILE / CA_BLOCK_CUBE) * (CA_HEAD_DIM / CA_C0);
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        LoadData(b2, b1, ld);
    }

    // V 为 (K=kv, N=hd) 行主序，需要 Nz -> Zn 的在途转置
    __aicore__ inline void SplitBForPv(LocalTensor<T>& b2, LocalTensor<T>& b1)
    {
        constexpr uint32_t dstOffset = (CA_HEAD_DIM / CA_BLOCK_CUBE) * CA_BLOCK_CUBE * CA_C0;
        constexpr uint32_t srcOffset = CA_BLOCK_CUBE * CA_C0;
        LoadData2dTransposeParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = CA_HEAD_DIM / CA_C0;
        ld.srcStride = CA_HEAD_DIM / CA_BLOCK_CUBE;
        ld.dstGap = 0;
        ld.dstFracGap = 0;
        for (uint32_t i = 0; i < CA_KV_TILE / CA_BLOCK_CUBE; ++i) {
            LoadDataWithTranspose(b2[i * dstOffset], b1[i * srcOffset], ld);
        }
    }

    // ---- 跨拍反向依赖（AIC 内 L1 / L0 复用） ----
    // L0B 一对槽由 QK 的 K 面板与 PV 的 V 面板共用，按拍全局奇偶交替（l0Slot_）：
    // 下一拍可以往另一个槽里做 L1->L0 搬运，不必等上一拍 Mmad 算完，从而把 MTE1 藏进 Mmad。
    // L0A 的 P 面板同样双槽；Q 面板任务级只装一次，拍内不再产生 A 依赖。
    // 头文件只导出 EVENT_ID0..3，4..7 直接用整型字面量（硬件事件 ID 共 8 个）
    static constexpr int32_t CA_L0B_S0 = EVENT_ID2;
    static constexpr int32_t CA_L0B_S1 = EVENT_ID3;
    static constexpr int32_t CA_PV_A_S0 = EVENT_ID0;
    static constexpr int32_t CA_PV_A_S1 = EVENT_ID1;
    static constexpr int32_t CA_L1_Q = 4;      // Q 的 L1 面板（每任务一次）

    // L1 环槽的两组依赖，同类型内各占 8 个 id。MTE2_MTE1：Q 0 / V 1~2 / K 4~7；
    // MTE1_MTE2：KFree 0,5,6,7 / VFree 1,2 / Q 4。
    // P 走 A5 的 CV 通道（AIV 的 MTE3 直接写 L1），不再是 AIC 的 MTE2 副本，id 3 空闲。
    //   MTE1_MTE2（槽可覆写信用，生产者在 MTE1 读完拍置位，消费者在 MTE2 搬入前置位）
    static constexpr int32_t KFreeId(uint32_t slot)
    {
        return (slot == 0u) ? 5 : ((slot == 1u) ? 6 : ((slot == 2u) ? 7 : 0));
    }
    static constexpr int32_t VFreeId(uint32_t slot) { return 1 + static_cast<int32_t>(slot); }
    //   MTE2_MTE1（本拍副本已落 L1，ED0 留给 Q 的即时屏障）
    static constexpr int32_t KCopyId(uint32_t slot) { return 4 + static_cast<int32_t>(slot); }
    static constexpr int32_t VCopyId(uint32_t slot) { return 1 + static_cast<int32_t>(slot); }

    static constexpr uint32_t KL1Off(uint32_t beat)
    {
        return CA_B1_ADDR_Q + (beat % CA_K_L1_SLOTS) * CA_K_SLOT_BYTES;
    }
    static constexpr uint32_t VL1Off(uint32_t beat)
    {
        return CA_B1_ADDR_P + (beat % CA_V_L1_SLOTS) * CA_V_SLOT_BYTES;
    }

    __aicore__ inline void PrimeBeatFlags()
    {
        SetFlag<HardEvent::MTE1_MTE2>(CA_L1_Q);
        // S 的 L0C 双槽信用：QkBeat 每拍等一份“上一轮 fixpipe 已读完”，首拍由这里预置
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        // PV 的 L0C 双槽信用：PvBeat 每拍等一份“上一轮 PV fixpipe 已读完”，首拍由这里预置
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        // K / V 每个 L1 槽各预置一份“可覆写”信用，供各任务首个预取窗口消费；
        // K/V 每拍都是一次“置位 + 消费”，所以信用数跨任务守恒（见 RunTask / PvBeat）。
        for (uint32_t s = 0; s < CA_K_L1_SLOTS; ++s) {
            SetFlag<HardEvent::MTE1_MTE2>(KFreeId(s));
        }
        for (uint32_t s = 0; s < CA_V_L1_SLOTS; ++s) {
            SetFlag<HardEvent::MTE1_MTE2>(VFreeId(s));
        }
        SetFlag<HardEvent::M_MTE1>(CA_L0B_S0);
        SetFlag<HardEvent::M_MTE1>(CA_L0B_S1);
        SetFlag<HardEvent::M_MTE1>(CA_PV_A_S0);
        SetFlag<HardEvent::M_MTE1>(CA_PV_A_S1);
    }

    static constexpr int32_t L0BSlotId(uint32_t slot) { return (slot == 0u) ? CA_L0B_S0 : CA_L0B_S1; }
    static constexpr int32_t PvASlotId(uint32_t slot) { return (slot == 0u) ? CA_PV_A_S0 : CA_PV_A_S1; }

    // S 环槽（AIV 的两个 sub 用同一套字节偏移，物理 UB 各自独立）
    __aicore__ inline uint32_t UbSlot(uint32_t ring, uint32_t slotBytes) const
    {
        return CA_UB_S + (ring % CA_S_RING) * slotBytes;
    }

    // ---- 前置搬运（只发不等） ----
    // 冷启动与跨任务预取共用：只发 Q 进 L1 与 K 首窗进 L1 环槽，不插任何等待计算的屏障。
    // 前者把 Q 的 GM 延迟与 K 的并行起来，后者把整段 GM 延迟挪到上一任务的 PV 之下。
    // withV 只在冷启动为 true：跨任务时 V 的 3 个槽还被本任务尾拍的 PV 占着。
    __aicore__ inline void PrefetchFront(const CaTaskDesc& d, bool withV)
    {
        LocalTensor<T> a1(TPosition::A1, CA_A1_ADDR_Q, CA_A_ELEMS);
        WaitFlag<HardEvent::MTE1_MTE2>(CA_L1_Q);
        BuildQ(d, a1);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        const uint32_t hi = (d.kvLoops < CA_S_RING) ? d.kvLoops : CA_S_RING;
        for (uint32_t i = 0; i < hi; ++i) {
            const uint32_t ks = i % CA_K_L1_SLOTS;
            WaitFlag<HardEvent::MTE1_MTE2>(KFreeId(ks));
            LocalTensor<T> b1(TPosition::B1, KL1Off(i), CA_B_ELEMS);
            BuildKV(d, i, b1, kGm_);
            SetFlag<HardEvent::MTE2_MTE1>(KCopyId(ks));
        }
        if (withV) {
            PrefetchVHead(d);
        }
    }

    // 单拍 K 预取：滚动调度里每发完一拍 QK，立刻补 +CA_S_RING 拍的那一份。
    // 调用点必须紧跟在 QkBeat 之后——该槽的 MTE1_MTE2 信用刚由那次 SplitB 归还。
    __aicore__ inline void PrefetchK(const CaTaskDesc& d, uint32_t beat)
    {
        const uint32_t ks = beat % CA_K_L1_SLOTS;
        WaitFlag<HardEvent::MTE1_MTE2>(KFreeId(ks));
        LocalTensor<T> b1(TPosition::B1, KL1Off(beat), CA_B_ELEMS);
        BuildKV(d, beat, b1, kGm_);
        SetFlag<HardEvent::MTE2_MTE1>(KCopyId(ks));
    }

    // V 的首个窗口：前 CA_V_L1_SLOTS 拍一次发满，之后由 PvBeat 逐拍滚动预取
    __aicore__ inline void PrefetchVHead(const CaTaskDesc& d)
    {
        const uint32_t vn = (d.kvLoops < CA_V_L1_SLOTS) ? d.kvLoops : CA_V_L1_SLOTS;
        for (uint32_t i = 0; i < vn; ++i) {
            const uint32_t vs = i % CA_V_L1_SLOTS;
            WaitFlag<HardEvent::MTE1_MTE2>(VFreeId(vs));
            LocalTensor<T> b1(TPosition::B1, VL1Off(i), CA_B_ELEMS);
            BuildKV(d, i, b1, vGm_);
            SetFlag<HardEvent::MTE2_MTE1>(VCopyId(vs));
        }
        vHead_ = true;
    }

    // 任务头：等 Q 落 L1（前置搬运已发过），再切到 L0A，之后所有拍直接复用 Q 面板。
    __aicore__ inline void PrepQ(const CaTaskDesc& d)
    {
        LocalTensor<T> a1(TPosition::A1, CA_A1_ADDR_Q, CA_A_ELEMS);
        LocalTensor<T> a2(TPosition::A2, CA_A2_ADDR_Q, CA_A_ELEMS);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        const uint32_t nbCount = CaCeilDiv(d.rows, CA_BLOCK_CUBE);
        if ((d.rows % CA_BLOCK_CUBE == 0u) && (nbCount > 1u)) {
            // 与 QkBeat 的大 M 口径配对：L1 的 interleaved 面板原样搬进 L0A
            SplitAWhole(a2, a1, nbCount);
        } else {
            SplitA(a2, a1, nbCount, CaInterleaved(d.rows));
        }
        SetFlag<HardEvent::MTE1_MTE2>(CA_L1_Q);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
    }

    __aicore__ inline void QkBeat(const CaTaskDesc& d, uint32_t beat)
    {
        const uint32_t slot = l0Slot_;
        const uint32_t ks = beat % CA_K_L1_SLOTS;
        LocalTensor<T> b1(TPosition::B1, KL1Off(beat), CA_B_ELEMS);
        LocalTensor<T> a2(TPosition::A2, CA_A2_ADDR_Q, CA_A_ELEMS);
        LocalTensor<T> b2(TPosition::B2, slot * CA_B2_SLOT_BYTES, CA_B_ELEMS);
        LocalTensor<float> co1(TPosition::CO1, CA_CO1_ADDR_Q + (beat & 1u) * CA_CO1_S_SLOT_BYTES,
                               CA_MAX_R * CA_HEAD_DIM);

        // 本拍的 K 已在 RunTask 的预取窗口里发过，这里只等它落 L1；MTE2 早已跑到当前拍之前。
        WaitFlag<HardEvent::MTE2_MTE1>(KCopyId(ks));

        // 该 L0B 槽的上一任 Mmad 读完后才能覆写
        WaitFlag<HardEvent::M_MTE1>(L0BSlotId(slot));
        SplitBForQk(b2, b1);
        SetFlag<HardEvent::MTE1_MTE2>(KFreeId(ks));
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        // S 在 L0C 里按拍奇偶双槽：本拍 Mmad 只等 beat-2 的 fixpipe 读完同槽。
        // 单槽时这里必须等本拍 fixpipe 把 S 全部搬进 UB 才能进下一拍（每拍固定 0.29 µs 气泡），
        // 双槽后这个等待降到 2 拍之前，实测即可消失。
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        const uint32_t nbCount = CaCeilDiv(d.rows, CA_BLOCK_CUBE);
        // rows 是 16 的整数倍且不止一个分块时，把整个 A 面板当一个大 NZ 矩阵、一次 Mmad 算完：
        // 每拍 8 条 m=16 的 Mmad 各自只有 32 cycle 的数学量，实测单条要 49 cycle
        // （见 profile：aic_mac_time 11.86 µs / 438 条），碎片化的发射开销占掉 1/3。
        // 尾块（rows 不是 16 的倍数）仍退回逐分块，此时 L0C 是并列的 16 行瓦片。
        const bool bigM = (d.rows % CA_BLOCK_CUBE == 0u) && (nbCount > 1u);
        MmadParams mp;
        mp.m = static_cast<uint16_t>(bigM ? d.rows : CA_BLOCK_CUBE);
        mp.n = static_cast<uint16_t>(CA_KV_TILE);
        mp.k = static_cast<uint16_t>(CA_HEAD_DIM);
        if (bigM) {
            mp.cmatrixInitVal = true;
            Mmad(co1, a2, b2, mp);
        } else {
            for (uint32_t nb = 0; nb < nbCount; ++nb) {
                mp.cmatrixInitVal = true;
                Mmad(co1[nb * CA_BLOCK_CUBE * CA_HEAD_DIM], a2[nb * CA_BLOCK_CUBE * CA_HEAD_DIM], b2, mp);
            }
        }
        SetFlag<HardEvent::M_MTE1>(L0BSlotId(slot));
        l0Slot_ ^= 1u;
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

        // S 直接从 L0C 进两个 AIV sub 的 UB：按 sub 行区间分别投递，subBlockId 选择目标 UB。
        // 行区间与 AIV 侧的 CaSubBegin/CaSubRows 完全一致，且都是 16 行对齐。
        const uint32_t ubOff = UbSlot(beat, CA_S_SLOT_BYTES);
        for (uint32_t sub = 0; sub < 2u; ++sub) {
            const uint32_t nRows = CaSubRows(d.rows, sub);
            if (nRows == 0u) {
                continue;
            }
            const uint32_t begin = CaSubBegin(d.rows, sub);
            LocalTensor<float> dst(TPosition::VECCALC, ubOff, CA_UB_ROWS * CA_HEAD_DIM);
            if (bigM) {
                // 大 M：L0C 是一个 rows x 128 的 NZ 矩阵，C0 步长 = rows，sub 一条就能搬完
                FixpipeWide(dst, co1, begin, nRows, sub, CA_KV_TILE, d.rows);
            } else {
                FixpipeRows(dst, co1, begin, nRows, sub, CA_KV_TILE);
            }
        }
        // 只还信用不等：本槽的下一个使用者是 beat+2 的 Mmad，它会在那里等。
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void PvBeat(const CaTaskDesc& d, uint32_t beat)
    {
        const uint32_t a1Off = CA_A1_ADDR_P + (pSeq_ % CA_P_RING) * CA_P_SLOT_BYTES;
        const uint32_t slot = l0Slot_;
        const uint32_t aslot = pvASlot_;
        const uint32_t vs = beat % CA_V_L1_SLOTS;
        LocalTensor<T> a1(TPosition::A1, a1Off, CA_A_ELEMS);
        LocalTensor<T> b1(TPosition::B1, VL1Off(beat), CA_B_ELEMS);
        LocalTensor<T> a2(TPosition::A2, CA_A2_ADDR_P + aslot * CA_A2_P_SLOT_BYTES, CA_A_ELEMS);
        LocalTensor<T> b2(TPosition::B2, slot * CA_B2_SLOT_BYTES, CA_B_ELEMS);
        LocalTensor<float> co1(TPosition::CO1, CA_CO1_ADDR_P + (pSeq_ & 1u) * CA_CO1_P_SLOT_BYTES,
                               CA_MAX_R * CA_HEAD_DIM);
        // P 由 AIV 通过 A5 的 CV 通道直接写进 L1 的 A1 环槽，两个 sub 各写自己的行段。
        // AIC 侧没有 GM 读回与 Nd2Nz：P_READY 一到，L1 里的就是 L0A 要的分形序。
        // L0A 的 P 槽空出来保证 Mmad(beat-2) 已结束，而 SplitA 在 MTE1 上按序，
        // 故 SplitA(beat-4) 必然完成 —— L1 的 P 槽（beat%CA_P_RING）可以安全覆写。
        WaitFlag<HardEvent::M_MTE1>(PvASlotId(aslot));

        // 本拍的 V 由前 CA_V_L1_SLOTS 拍的预取（或任务头窗口）送进 L1
        WaitFlag<HardEvent::MTE2_MTE1>(VCopyId(vs));

        WaitFlag<HardEvent::M_MTE1>(L0BSlotId(slot));
        SplitBForPv(b2, b1);
        // 归还本槽信用，并立刻把它用在 beat+CA_V_L1_SLOTS 拍的预取上：
        // 那一拍要用的正是这个槽，此时 MTE1 已读完，可以安全覆写。
        SetFlag<HardEvent::MTE1_MTE2>(VFreeId(vs));
        const uint32_t pf = beat + CA_V_L1_SLOTS;
        if (pf < loops_) {
            WaitFlag<HardEvent::MTE1_MTE2>(VFreeId(vs));
            BuildKV(d, pf, b1, vGm_);
            SetFlag<HardEvent::MTE2_MTE1>(VCopyId(vs));
        }
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);

        // 本拍的 P 已由 AIV 经 CV 通道写进 L1 环槽，两个 sub 都写完才能读。
        // 等到 SplitA 之前再等，把 AIV 的等待期藏在本拍的 V 搬运 / SplitB 之下。
        WaitPReady();
        const uint32_t nbCount = CaCeilDiv(d.rows, CA_BLOCK_CUBE);
        // P 的 L1 布局跟 QK 的 Q 一样分两种：
        //   rows 是 16 的整数倍且有多块时走「整面板交错」（C0 步长 = rows），配 SplitAWhole
        //   恒等搬运 + 单条 Mmad(m=rows)，与 QK 的大 M 口径一致；
        //   尾块（rows%16 != 0）或单块仍走 16 行分块布局（块基址 nb*16*128，块内 C0 步长 16），
        //   与 CopyPToL1 的写入口径、与旧 GM 回程版的 Nd2Nz 输出逐字节一致。
        const bool wideP = (d.rows % CA_BLOCK_CUBE == 0u) && (nbCount > 1u);
        if (wideP) {
            SplitAWhole(a2, a1, nbCount);
        } else {
            SplitA(a2, a1, nbCount, false);
        }
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        // P 的 L1 槽已经读进 L0A，可以还信用让 AIV 覆写（CA_P_RING 深）
        NotifyPFree();

        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        // online：本拍 PV 单独出一个 L0C 槽（按 pSeq_ 奇偶双缓冲，与 QK 的 S 同构），
        // 不再跨拍累加。本槽的上一任 Mmad 的 fixpipe 读完（FIX_M 信用）才能覆写。
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        // 同一份 V 面板供所有 16 行分块复用；每拍 cmatrixInitVal=true，输出即本拍 PV
        MmadParams mp;
        mp.m = static_cast<uint16_t>(wideP ? d.rows : CA_BLOCK_CUBE);
        mp.n = static_cast<uint16_t>(CA_KV_TILE);
        mp.k = static_cast<uint16_t>(CA_HEAD_DIM);
        if (wideP) {
            // 大 M：一条 Mmad 出整个 rows x 128，PV 在 L0C 里是完整 NZ 矩阵（C0 步长 = rows）
            mp.cmatrixInitVal = true;
            Mmad(co1, a2, b2, mp);
        } else {
            for (uint32_t nb = 0; nb < nbCount; ++nb) {
                mp.cmatrixInitVal = true;
                Mmad(co1[nb * CA_BLOCK_CUBE * CA_HEAD_DIM], a2[nb * CA_BLOCK_CUBE * CA_HEAD_DIM], b2, mp);
            }
        }
        SetFlag<HardEvent::M_MTE1>(L0BSlotId(slot));
        SetFlag<HardEvent::M_MTE1>(PvASlotId(aslot));
        // M_FIX 自配对屏障：本拍 Mmad 落地后 fixpipe 才能读（与 QkBeat 的 EVENT_ID0 同法）
        SetFlag<HardEvent::M_FIX>(EVENT_ID1);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID1);

        // 本拍 PV 直接从 L0C 投递到两个 AIV sub 各自的 UB 环槽（PV_READY 驱动，
        // PV_FREE 按 sub 回收信用；空行 sub 的 AIV 也照常消费/归还，配对不失衡）
        if (beat >= CA_PV_RING) {
            WaitPvFree();
        }
        const uint32_t pvUbOff = CA_UB_PV + (pSeq_ % CA_PV_RING) * CA_PV_SLOT_BYTES;
        for (uint32_t sub = 0; sub < 2u; ++sub) {
            const uint32_t nRows = CaSubRows(d.rows, sub);
            if (nRows == 0u) {
                continue;
            }
            const uint32_t begin = CaSubBegin(d.rows, sub);
            LocalTensor<float> dst(TPosition::VECCALC, pvUbOff, CA_UB_ROWS * CA_HEAD_DIM);
            // 与 wideP 同条件：大 M 单条 Mmad 的 PV 在 L0C 里是完整 NZ 矩阵
            if (wideP) {
                FixpipeWide(dst, co1, begin, nRows, sub, CA_HEAD_DIM, d.rows);
            } else {
                FixpipeRows(dst, co1, begin, nRows, sub, CA_HEAD_DIM);
            }
        }
        NotifyPvReady();
        // 只还信用不等：本槽的下一个使用者是 pSeq_+2 的 Mmad，它会在那里等
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        l0Slot_ ^= 1u;
        pvASlot_ ^= 1u;
        ++pSeq_;
    }

    // L0C 里的每个 16 行分块都是一次独立 Mmad 的结果，在 L0C 中是并列的 16x128 瓦片，
    // 并不是一个 32x128 的 NZ 矩阵。Fixpipe 一次读 mSize 行时会按矩阵内的 m1 跨步索引，
    // mSize>16 就会读到别的瓦片（症状：每 32 行里第二个 16 行块错）。因此按 16 行逐块投递。
    // 大 M 单条 Mmad 的路径改用下面的 FixpipeWide（L0C 是完整 NZ 矩阵，C0 步长 = M）。
    __aicore__ inline void FixpipeRows(LocalTensor<float> dst, LocalTensor<float> co1, uint32_t begin,
                                       uint32_t nRows, uint32_t sub, uint32_t nSize)
    {
        for (uint32_t off = 0; off < nRows; off += CA_BLOCK_CUBE) {
            const uint32_t rowsThis = (nRows - off < CA_BLOCK_CUBE) ? (nRows - off) : CA_BLOCK_CUBE;
            FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.mSize = static_cast<uint16_t>(rowsThis);
            fp.nSize = static_cast<uint16_t>(nSize);
            fp.srcStride = static_cast<uint16_t>(CA_BLOCK_CUBE);
            fp.dstStride = nSize;
            fp.subBlockId = (sub != 0u);
            Fixpipe<float, float, CA_CFG_ROW_MAJOR_UB>(dst[off * nSize], co1[(begin + off) * nSize], fp);
        }
    }

    // 大 M 路径：一次 Mmad 算出完整的 rows x nSize NZ 矩阵，L0C 的 C0 步长就是 mStride(=M)。
    // 每个 sub 只有一条 fixpipe，起始地址是「第 begin 行在第一个 n 分形里的位置」= begin*C0。
    __aicore__ inline void FixpipeWide(LocalTensor<float> dst, LocalTensor<float> co1, uint32_t begin,
                                       uint32_t nRows, uint32_t sub, uint32_t nSize, uint32_t mStride)
    {
        FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
        fp.mSize = static_cast<uint16_t>(nRows);
        fp.nSize = static_cast<uint16_t>(nSize);
        fp.srcStride = static_cast<uint16_t>(mStride);
        fp.dstStride = nSize;
        fp.subBlockId = (sub != 0u);
        Fixpipe<float, float, CA_CFG_ROW_MAJOR_UB>(dst, co1[begin * CA_C0], fp);
    }

    // 任务末不再投递 O：online 口径下 O_run 全程在 AIV 的 UB 里逐拍合并（见 OUpdateBeat），
    // L0C 的 PV 槽逐拍已被 fixpipe 排空，任务边界只剩 PV_FREE 信用补齐（见 RunTask）。

    const __gm__ CrossAttentionTilingData* tiling_;
    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<int32_t> btGm_;
    GlobalTensor<int64_t> lensGm_;
    uint32_t coreIdx_;
    uint32_t l0Slot_;  // L0B 共用槽指针（QK/PV 连续交替）
    uint32_t pvASlot_; // L0A 中 P 面板双槽指针
    uint32_t loops_;   // 当前任务的 KV 拍数（V 滚动预取的边界）
    bool vHead_;       // 本任务的 V 首窗已由 PrefetchFront 发过
    uint32_t effSeg_;  // 按实际 kvLen 推出的生效段数（1 = 不需要 merge）
    // P 的 L1 环槽号必须用跨任务连续的拍号，和 AIV 侧的 pUbSeq_ 一一对应：
    // 若按任务内的 beat 取槽，单拍任务会一直用 0 号槽，而 P_FREE 的等待条件是
    // beat >= CA_P_RING，跨任务时永远不触发 —— AIV 会在 AIC 还没读走上一任务的
    // P 时就把它覆写掉（现象：kv 极小的单拍形状错得最重）。
    uint32_t pSeq_ = 0u;
};

// ---------------------------------------------------------------------------
// AIV：softmax / 归一化 / merge
// ---------------------------------------------------------------------------
template <typename T>
class CrossAttentionAiv {
public:
    __aicore__ inline void Init(const __gm__ CrossAttentionTilingData* t, __gm__ uint8_t* attnOut,
                                __gm__ uint8_t* actualKvLens, __gm__ uint8_t* workspace)
    {
        tiling_ = t;
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attnOut));
        lensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(actualKvLens));
        mergeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        coreIdx_ = GetBlockIdx() >> 1;
        subIdx_ = GetSubBlockIdx();
        // 标量表量在 Init 阶段一次性读入标量寄存器：向量指令的标量操作数直接取 GM 值不可靠
        scaleV_ = t->scaleValue;
        begin_ = 0u;
        nRows_ = 0u;
        pUbSeq_ = 0u;
        effSeg_ = CaEffSeg(t, lensGm_);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t taskIdx = coreIdx_; taskIdx < tiling_->totalTaskNum; taskIdx += tiling_->usedCoreNum) {
            CaTaskDesc d;
            ParseTask(tiling_, taskIdx, lensGm_, d);
            if (d.kvLoops == 0) {
                // m/l 区按 sub 分区，两个 sub 各写自己的区域，不能只让 sub0 写
                if (effSeg_ > 1u) {
                    WriteSentinelSlot(taskIdx, d);
                }
                continue;
            }
            RunTask(d);
        }

        if (effSeg_ > 1u) {
            // 与 AIC 侧成对出现：AIC+AIV 全局栅栏，保证各核 merge 槽已落地
            AscendC::SyncAll<false>();
            for (uint32_t taskIdx = coreIdx_; taskIdx < tiling_->totalTaskNum; taskIdx += tiling_->usedCoreNum) {
                CaTaskDesc d;
                ParseTask(tiling_, taskIdx, lensGm_, d);
                if (d.kvSegIdx == 0) {
                    MergeTask(d);
                }
            }
        }
    }

private:
    // ---- UB 视图（全部按字节偏移构造，不用 TPipe） ----
    __aicore__ inline __local_mem__ float* UbF32(uint32_t off) const
    {
        LocalTensor<float> t(TPosition::VECCALC, off, 8);
        return (__local_mem__ float*)t.GetPhyAddr();
    }

    __aicore__ inline __local_mem__ T* UbT(uint32_t off) const
    {
        LocalTensor<T> t(TPosition::VECCALC, off, 8);
        return (__local_mem__ T*)t.GetPhyAddr();
    }

    __aicore__ inline LocalTensor<float> UbF32Tensor(uint32_t off, uint32_t elems) const
    {
        return LocalTensor<float>(TPosition::VECCALC, off, elems);
    }

    __aicore__ inline LocalTensor<T> UbTTensor(uint32_t off, uint32_t elems) const
    {
        return LocalTensor<T>(TPosition::VECCALC, off, elems);
    }

    // ---- AIV 侧核间握手（mode 4 intra block，只用自己的 sub id） ----
    __aicore__ inline void WaitSReady()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_S_READY + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void NotifyPReady()
    {
        CrossCoreSetFlag<4, PIPE_MTE3>(CA_FLAG_P_READY + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    // S 槽已读进 UB，可以还给 AIC 覆写。调用点必须在 S 的向量读取全部落地之后。
    __aicore__ inline void NotifySFree()
    {
        CrossCoreSetFlag<4, PIPE_V>(CA_FLAG_S_FREE + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitPFree()
    {
        // 默认挂在 PIPE_S 上（同上：挂 MTE3 会引入死锁）。
        CrossCoreWaitFlag<4>(CA_FLAG_P_FREE + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    __aicore__ inline void WaitPvReady()
    {
        CrossCoreWaitFlag<4>(CA_FLAG_PV_READY + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }
    // 本拍 PV 已并入 O_run，AIC 可以覆写该 sub 的 PV 环槽
    __aicore__ inline void NotifyPvFree()
    {
        CrossCoreSetFlag<4, PIPE_V>(CA_FLAG_PV_FREE + static_cast<uint16_t>(subIdx_) * CA_FLAG_SUB_STRIDE);
    }

    // ---- 标量 + 向量混合工具 ----
    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> red = UbF32Tensor(CA_UB_SCAL + CA_SCAL_RED, 64);
        Duplicate(red, x, 64);
        PipeBarrier<PIPE_V>();
        AscendC::Exp(red, red, 64);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return red.GetValue(0);
    }

    __aicore__ inline void ZeroOAcc()
    {
        LocalTensor<float> oAcc = UbF32Tensor(CA_UB_O, CA_UB_ROWS * CA_HEAD_DIM);
        Duplicate(oAcc, 0.0f, CA_UB_ROWS * CA_HEAD_DIM);
        PipeBarrier<PIPE_V>();
    }

    // 每任务的 online 行状态复位：gm 从 CA_MINV 起折，lsum 从 0 起累。
    // 都用向量写（与 SoftmaxRows / StoreML 的 V_S 屏障口径一致，避免标量-向量跨管道竞态）。
    __aicore__ inline void InitRowState()
    {
        LocalTensor<float> rowm = UbF32Tensor(CA_UB_ROWM, CA_UB_ROWM_ELEMS);
        LocalTensor<float> lsum = UbF32Tensor(CA_UB_LSUM, CA_UB_ROWS * 64);
        Duplicate(rowm, CA_MINV, CA_UB_ROWM_ELEMS);
        Duplicate(lsum, 0.0f, CA_UB_ROWS * 64);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunTask(const CaTaskDesc& d)
    {
        begin_ = CaSubBegin(d.rows, subIdx_);
        nRows_ = CaSubRows(d.rows, subIdx_);
        ZeroOAcc();
        InitRowState();
        const uint32_t loops = d.kvLoops;
        // 本任务首拍的跨任务全局拍号（与 AIC 的 pSeq_ 同源，用于 PV / dm 环槽配对）
        const uint32_t pvBase = pUbSeq_;
        for (uint32_t i = 0; i < loops; ++i) {
            SoftmaxBeat(d, i);
        }
        // 收尾：任务最后一拍的 PV 必须并进 O_run——它既补全 O_run / PV_READY / PV_FREE
        // 的配对，又是 AIC 任务末 PV_FREE 补齐能收齐信用的前提。
        OUpdateBeat(pvBase + loops - 1u);
        // P_FREE 是 AIC 每拍产出一份、本侧只在超出环深时消费，任务末补齐
        const uint32_t drainP = loops < CA_P_RING ? loops : CA_P_RING;
        for (uint32_t k = 0; k < drainP; ++k) {
            WaitPFree();
        }

        if (effSeg_ == 1u) {
            WriteOutput(d);
        } else {
            WriteMergeSlot(d);
        }
    }

    // 掩码行：列 [0, nReal) 置 clampVal，其余列置 CA_MINV。每拍建两份：
    //   CA_UB_MASKB（clampVal=CA_BIGV）：max pass 用，无效列压成 CA_MINV 不参与取最大
    //   CA_UB_MASK （clampVal=SoftClampV）：P pass 用，减完 gm 后上钳（online 下恒不生效，纯保险）
    __aicore__ inline void BuildMaskRow(uint32_t ubOff, uint32_t nReal, float clampVal)
    {
        __local_mem__ float* maskPtr = UbF32(ubOff);
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vA, vB;
            uint32_t cFull = 64u;
            // UpdateMask 是「取剩余个数并后置递减」，超过一个向量宽度时行为不确定，
            // 这里显式限幅到 <= 64。
            uint32_t c0 = (nReal < 64u) ? nReal : 64u;
            uint32_t c1 = (nReal > 64u) ? (nReal - 64u) : 0u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            Mapi::MaskReg m0 = Mapi::UpdateMask<float>(c0);
            Mapi::MaskReg m1 = Mapi::UpdateMask<float>(c1);
            // 注意：Duplicate 带 mask 时默认是 ZEROING（未选中 lane 清零），会把
            // 前一次铺好的 CA_MINV 抹成 0；必须显式用 MERGING 保留 dst 的旧值。
            Mapi::Duplicate(vA, CA_MINV, mFull);
            Mapi::Duplicate<float, Mapi::MaskMergeMode::MERGING>(vA, clampVal, m0);
            Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(maskPtr, vA, mFull);
            Mapi::Duplicate(vB, CA_MINV, mFull);
            Mapi::Duplicate<float, Mapi::MaskMergeMode::MERGING>(vB, clampVal, m1);
            Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(maskPtr + 64, vB, mFull);
        }
    }

    // 本 dtype 的 P 保险钳位：online 口径下减完运行 gm 后有效列的自变量恒 <= 0，
    // 这个上钳只在异常路径兜底（见公共头），不再是截断手段。
    __aicore__ inline float SoftClampV() const
    {
        return AscendC::IsSameType<T, half>::value ? CA_SOFT_CLAMP_FP16 : CA_SOFT_CLAMP_BF16;
    }

    // 一拍 online softmax（FA2 口径，逐行折叠，含 1/scale 与两份掩码，见 SoftmaxBeat）：
    //   1) 本拍行最大 lm（BIGV 掩码压无效列），gm = max(gm, lm) 写回 CA_UB_ROWM；
    //      原始差值 (gm_old - gm_new) 按 lane 0 落 CA_UB_DM 环槽暂存；
    //   2) scope 2 BRC 读回差值（跨 scope + V_S 屏障后可见，同旧版 ComputeRowMax ->
    //      SoftmaxRows 读 ROWM 的口径），exp 得 dm，覆写回同一槽供隔拍 OUpdateBeat 广播读；
    //   3) P = exp(scale*S - gm)（减完再 Min 掩码：无效列压 CA_MINV，exp 归零），
    //      转 T 写 P 的 UB 环槽（与改动前口径一致）；
    //   4) lsum = lsum * dm + rowsum(P)，直接落 CA_UB_LSUM（列 0，广播读写口径同 gm），
    //      任务末不再有元素级 lacc / ReduceRows。
    // 必须逐行而不是全块最大：P 是 16bit，块最大当整块偏移时，行间 logit 跨度超过约 16.6
    // 会让整行的 exp 在 fp16 里下溢成 0。
    __aicore__ inline void SoftmaxRows(__local_mem__ float* sPtr, uint32_t pseq)
    {
        __local_mem__ float* lsumPtr = UbF32(CA_UB_LSUM);
        __local_mem__ float* maskPtr = UbF32(CA_UB_MASK);
        __local_mem__ float* maskBPtr = UbF32(CA_UB_MASKB);
        __local_mem__ float* rowmPtr = UbF32(CA_UB_ROWM);
        __local_mem__ float* dmPtr = UbF32(CA_UB_DM + (pseq % CA_DM_SLOTS) * CA_DM_SLOT_BYTES);
        __local_mem__ T* pPtr = UbT(CA_UB_P + (pseq % CA_P_UB_SLOTS) * CA_UB_P_SLOT_BYTES);
        const uint16_t nRows = static_cast<uint16_t>(nRows_);
        const float scale = scaleV_;
        // scope 1：逐行折叠 gm 落 ROWM，(gm_old - gm_new) 的 lane 0 落 dm 槽暂存
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vS0, vS1, vA0, vA1, vNB0, vNB1;
            Mapi::RegTensor<float> vOld, vHm, vDraw, vR;
            uint32_t cFull = 64u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vNB0, maskBPtr);
            Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vNB1, maskBPtr + 64);
            for (uint16_t r = 0; r < nRows; ++r) {
                const uint32_t ro = static_cast<uint32_t>(r) * CA_HEAD_DIM;
                const uint32_t rm = static_cast<uint32_t>(r) * 64;
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vS0, sPtr + ro);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vS1, sPtr + ro + 64);
                Mapi::Muls(vS0, vS0, scale, mFull);
                Mapi::Muls(vS1, vS1, scale, mFull);
                // 本拍行最大：无效列压 CA_MINV，不参与
                Mapi::Min(vA0, vS0, vNB0, mFull);
                Mapi::Min(vA1, vS1, vNB1, mFull);
                Mapi::Max(vA0, vA0, vA1, mFull);
                Mapi::ReduceMax(vR, vA0, mFull);
                // 折叠运行最大：vR 只有 lane 0 有效，vHm 的其余 lane 仍是 gm_old 的广播值；
                // 落 ROWM 的 lane 0 即新 gm，BRC 读回口径只取 lane 0，scope 2 拿到的就是
                // 干净的标量 gm 全 lane 广播。差值同理只保留 lane 0。
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_BRC_B32>(vOld, rowmPtr + rm);
                Mapi::Max(vHm, vOld, vR, mFull);
                Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(rowmPtr + rm, vHm, mFull);
                Mapi::Sub(vDraw, vOld, vHm, mFull);
                Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(dmPtr + rm, vDraw, mFull);
            }
        }
        // scope 1 的写入要跨 scope 才对 scope 2 的 BRC 读可见：SetFlag/WaitFlag V_S 让标量
        // 管道等向量管道排空，后续 vcopy 发射时数据已落地（旧版 ComputeRowMax 同款）。
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        // scope 2：BRC 读回 gm/dm 标量广播，算 P、dm 覆写、lsum 折叠与 P 落 16bit 环槽
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vS0, vS1, vA0, vN0, vN1;
            Mapi::RegTensor<float> vHm, vDm, vLs, vLo;
            Mapi::RegTensor<T> vT0, vT1;
            uint32_t cFull = 64u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vN0, maskPtr);
            Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vN1, maskPtr + 64);
            for (uint16_t r = 0; r < nRows; ++r) {
                const uint32_t ro = static_cast<uint32_t>(r) * CA_HEAD_DIM;
                const uint32_t rm = static_cast<uint32_t>(r) * 64;
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_BRC_B32>(vHm, rowmPtr + rm);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_BRC_B32>(vDm, dmPtr + rm);
                Mapi::Exp(vDm, vDm, mFull);
                // dm 覆写回同槽：全 lane 一致，OUpdateBeat 的 BRC 读不受脏 lane 影响
                Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(dmPtr + rm, vDm, mFull);
                // P = exp(scale*S - gm)：有效列自变量恒 <= 0
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vS0, sPtr + ro);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vS1, sPtr + ro + 64);
                Mapi::Muls(vS0, vS0, scale, mFull);
                Mapi::Muls(vS1, vS1, scale, mFull);
                Mapi::Sub(vS0, vS0, vHm, mFull);
                Mapi::Sub(vS1, vS1, vHm, mFull);
                Mapi::Min(vS0, vS0, vN0, mFull);
                Mapi::Min(vS1, vS1, vN1, mFull);
                Mapi::Exp(vS0, vS0, mFull);
                Mapi::Exp(vS1, vS1, mFull);
                // lsum = lsum * dm + rowsum（ReduceSum 只有 lane 0 有效，落位后按列 0 广播读）
                Mapi::Add(vA0, vS0, vS1, mFull);
                Mapi::ReduceSum(vLs, vA0, mFull);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_BRC_B32>(vLo, lsumPtr + rm);
                Mapi::Mul(vLo, vLo, vDm, mFull);
                Mapi::Add(vLo, vLo, vLs, mFull);
                Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(lsumPtr + rm, vLo, mFull);
                // P 转 16bit 进 UB 环槽
                CaCastF32To(vT0, vS0, mFull);
                CaCastF32To(vT1, vS1, mFull);
                Mapi::DataCopy<T, Mapi::StoreDist::DIST_PACK_B32>(pPtr + ro, vT0, mFull);
                Mapi::DataCopy<T, Mapi::StoreDist::DIST_PACK_B32>(pPtr + ro + 64, vT1, mFull);
            }
        }
        // gm / lsum / dm 都在向量管道里写完：StoreML 与 OUpdateBeat 的后续读
        // （标量 GetValue / 向量广播）依赖这次同步，口径与改动前的 ComputeRowMax 一致。
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    // 隔拍合并：把 pvseq 那一拍的 PV（AIC 逐拍 fixpipe 进 CA_UB_PV 环槽）并入 O_run：
    //   O_run = O_run * dm + PV，dm 用 SoftmaxRows 同拍号写好的 CA_UB_DM 环槽按行广播读。
    // 空行 sub（rows <= 16 时的 sub1）没有向量工作，但 PV_READY / PV_FREE 必须照常消费 /
    // 归还：AIC 的 NotifyPvReady / WaitPvFree 都是按 sub 成对发的，缺一份就会死锁。
    __aicore__ inline void OUpdateBeat(uint32_t pvseq)
    {
        WaitPvReady();
        if (nRows_ != 0u) {
            __local_mem__ float* oPtr = UbF32(CA_UB_O);
            __local_mem__ float* pvPtr = UbF32(CA_UB_PV + (pvseq % CA_PV_RING) * CA_PV_SLOT_BYTES);
            __local_mem__ float* dmPtr = UbF32(CA_UB_DM + (pvseq % CA_DM_SLOTS) * CA_DM_SLOT_BYTES);
            const uint16_t nRows = static_cast<uint16_t>(nRows_);
            __VEC_SCOPE__
            {
                Mapi::RegTensor<float> vO0, vO1, vP0, vP1, vDm;
                uint32_t cFull = 64u;
                Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
                for (uint16_t r = 0; r < nRows; ++r) {
                    const uint32_t ro = static_cast<uint32_t>(r) * CA_HEAD_DIM;
                    Mapi::DataCopy<float, Mapi::LoadDist::DIST_BRC_B32>(vDm, dmPtr + static_cast<uint32_t>(r) * 64);
                    Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vO0, oPtr + ro);
                    Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vO1, oPtr + ro + 64);
                    Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vP0, pvPtr + ro);
                    Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vP1, pvPtr + ro + 64);
                    Mapi::Mul(vO0, vO0, vDm, mFull);
                    Mapi::Mul(vO1, vO1, vDm, mFull);
                    Mapi::Add(vO0, vO0, vP0, mFull);
                    Mapi::Add(vO1, vO1, vP1, mFull);
                    Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(oPtr + ro, vO0, mFull);
                    Mapi::DataCopy<float, Mapi::StoreDist::DIST_NORM>(oPtr + ro + 64, vO1, mFull);
                }
            }
        }
        // PV 槽与 dm 槽都已读完（PIPE_V 上置位保证向量读取完成），AIC 可以覆写
        NotifyPvFree();
    }

    __aicore__ inline void CaCastF32To(Mapi::RegTensor<T>& dst, Mapi::RegTensor<float>& src, Mapi::MaskReg& mask)
    {
        if constexpr (AscendC::IsSameType<T, half>::value) {
            Mapi::Cast<half, float, CA_CAST_F32_TO_FP16>(dst, src, mask);
        } else {
            Mapi::Cast<bfloat16_t, float, CA_CAST_F32_TO_BF16>(dst, src, mask);
        }
    }

    // P：UB 行主序 -> L1 的 A1 环槽，走 A5 的 CV 通道（UB -> L1，硬件是 copy_ubuf_to_cbuf）。
    // 不经 GM、也不需要 TPipe：AIV 的 MTE3 直接把 32B 片段写进 L1 的 NZ 分形。
    // UB 里每个 16 列组是一段连续 32B，相邻行隔 CA_HEAD_DIM*2 = 256B，画面见下面的按列描述符。
    // 落点与改动前 AIC 侧 Nd2Nz 的输出布局逐字节一致，因此 AIC 侧去掉 GM 读回与 Nd2Nz，直接 SplitA。
    __aicore__ inline void CopyPToL1(const CaTaskDesc& d, uint32_t beat, uint32_t pseq)
    {
        // 槽号与 AIC 的 pSeq_ 同源：跨任务连续，否则单拍任务会反复压 0 号槽。
        const uint32_t slot = pseq % CA_P_RING;
        LocalTensor<T> pU(TPosition::VECCALC, CA_UB_P + (pseq % CA_P_UB_SLOTS) * CA_UB_P_SLOT_BYTES,
                          CA_UB_ROWS * CA_HEAD_DIM);
        LocalTensor<T> pL1(TPosition::A1, CA_A1_ADDR_P + slot * CA_P_SLOT_BYTES, CA_A_ELEMS);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // 按 16 列组下发：一条描述符 = 一个 16 行分块里的一个 16 列组。源侧相邻行隔
        // 8 个 32B 块（CA_HEAD_DIM*2 = 256B），目的侧同一分形内 16 个 32B 连续
        // （dstStride = 0），下一个列组换一个分形（+512B = 32 个 32B 块）。
        // 描述符数 = 分块数 x 8，比逐行写法（行数 x 1）少一半，且 MTE3 的 burst 数不变。
        // 落点与逐行写法逐字节一致，AIC 侧仍然直接 SplitA。
        // 大 M 口径（与 PvBeat 的 wideP 同条件）：P 的 L1 布局改成整面板交错，第 g 个
        // 16 列组的分形基址 = g*rows*16，本 sub 的行段在分形里连续。每条描述符仍不跨
        // 16 行（硬件 dst 同分形内连续，跨分形块就要换基址），只是基址按列组整段排。
        const uint32_t nbCount = CaCeilDiv(d.rows, CA_BLOCK_CUBE);
        const bool wideP = (d.rows % CA_BLOCK_CUBE == 0u) && (nbCount > 1u);
        for (uint32_t c = 0; c < CaCeilDiv(nRows_, CA_BLOCK_CUBE); ++c) {
            const uint32_t left = nRows_ - c * CA_BLOCK_CUBE;
            const uint32_t cnt = left < CA_BLOCK_CUBE ? left : CA_BLOCK_CUBE;
            const uint32_t ubBase = c * CA_BLOCK_CUBE * CA_HEAD_DIM;
            const uint32_t rowBase = begin_ + c * CA_BLOCK_CUBE;
            const uint32_t blkBase = wideP ? 0u : (rowBase / CA_BLOCK_CUBE) * CA_BLOCK_CUBE * CA_HEAD_DIM;
            DataCopyParams cp;
            cp.blockCount = static_cast<uint16_t>(cnt);                              // 本分块的行数
            cp.blockLen = 1u;                                                        // 32B = 16 个 T
            cp.srcStride = static_cast<uint16_t>(CA_HEAD_DIM / CA_C0 - 1u);           // 跨行 7 个 32B
            cp.dstStride = 0u;                                                       // 分形内连续
            for (uint32_t g = 0; g < CA_HEAD_DIM / CA_C0; ++g) {
                const uint32_t dstOff = wideP ? (g * d.rows * CA_C0 + rowBase * CA_C0)
                                              : (blkBase + g * CA_BLOCK_CUBE * CA_C0);
                DataCopy(pL1[dstOff], pU[ubBase + g * CA_C0], cp);
            }
        }
        // 8 条描述符全部发完才置 P_READY（flag 挂在 MTE3 上，天然排在它们之后）
        PipeBarrier<PIPE_MTE3>();
        NotifyPReady();
        // 该 UB 槽 CA_P_UB_SLOTS 拍之后才被向量覆写：这里只置信用，本拍不等，
        // 让这笔 MTE3 压在下一拍的向量之下。等待点见 SoftmaxBeat。
        // 用 id 2/3，避开 OUTPUT 缓冲的 MTE3_V(EVENT_ID0)，否则两次 Set 一次 Wait 会合并。
        SetFlag<HardEvent::MTE3_V>((pseq % CA_P_UB_SLOTS) == 0u ? CA_PUB_S0 : CA_PUB_S1);
    }

    __aicore__ inline void SoftmaxBeat(const CaTaskDesc& d, uint32_t beat)
    {
        // P 的 UB 槽号跨任务连续递增：单拍任务下 beat 每任务从 0 重启，
        // 若按 beat 取槽，信用（SetFlag/WaitFlag）会在任务边界上错位。
        const uint32_t pseq = pUbSeq_;
        ++pUbSeq_;
        WaitSReady();
        __local_mem__ float* sPtr = UbF32(CA_UB_S + (beat % CA_S_RING) * CA_S_SLOT_BYTES);

        // 本拍的有效列数（尾块需要掩码）
        const uint32_t nRealBase = beat * CA_KV_TILE;
        uint32_t nReal = (d.segValid > nRealBase) ? (d.segValid - nRealBase) : 0u;
        if (nReal > CA_KV_TILE) {
            nReal = CA_KV_TILE;
        }
        if (nReal == 0u) {
            nReal = 1u; // ParseTask 保证 kvLoops 与 segValid 一致，这里只做兜底
        }

        if (nRows_ != 0u) {
            // online 口径：每拍都要折行最大。两份掩码行各建一份：BIGV 版供 max pass
            //（无效列压 CA_MINV 不参与取最大），SoftClamp 版供 P pass（无效列 exp 归零）
            BuildMaskRow(CA_UB_MASKB, nReal, CA_BIGV);
            BuildMaskRow(CA_UB_MASK, nReal, SoftClampV());
            // 本拍要写的 P 槽上一轮的 MTE3 必须读完（信用在 CA_P_UB_SLOTS 拍之前置位）
            if (pseq >= CA_P_UB_SLOTS) {
                WaitFlag<HardEvent::MTE3_V>((pseq % CA_P_UB_SLOTS) == 0u ? CA_PUB_S0 : CA_PUB_S1);
            }
            SoftmaxRows(sPtr, pseq);
        }
        // S 槽已读完（PIPE_V 上置位，保证向量读取完成），AIC 可以覆写
        NotifySFree();

        // 任务边界已由 RunTask 末尾的 S_FREE / PV_FREE 信用补齐兜住（AIV 把本任务最后一拍的
        // S 与 PV 全部消费完才归还最后的信用，AIC 收到才进下一个任务），所以跨任务不需要
        // 额外的 P_FREE 保护；而按 pseq 判会在多任务之后触发跨任务的等待，mode 4 的 flag
        // 只是事件寄存器，AIC 早期置的位不会累积，实测 b=8 kv=300 直接死锁。
        if (beat >= CA_P_RING) {
            WaitPFree();
        }
        if (nRows_ != 0u) {
            CopyPToL1(d, beat, pseq);
        } else {
            NotifyPReady();
        }

        // 隔拍合并：本拍 P 已发出，把上一拍的 PV 并进 O_run——PV(beat-1) 的 Mmad + fixpipe
        // 链与本拍 softmax 重叠，这里的等待通常已落地；最后一拍在 RunTask 收尾处补。
        if (beat > 0u) {
            OUpdateBeat(pseq - 1u);
        }
    }

    // out = O / l，转 T 落到 CA_UB_OUT
    // 1/l 不再用标量逐行求：直接按行广播 l 做向量 Div，省掉 StoreInverseL 的
    // 32 次 GetValue/SetValue 与两对 V_S / S_V 同步。
    __aicore__ inline void ScaleAndCast()
    {
        __local_mem__ float* oPtr = UbF32(CA_UB_O);
        __local_mem__ float* lsumPtr = UbF32(CA_UB_LSUM);
        __local_mem__ T* outPtr = UbT(CA_UB_OUT);
        const uint16_t nRows = static_cast<uint16_t>(nRows_);
        __VEC_SCOPE__
        {
            Mapi::RegTensor<float> vO0, vO1, vI;
            Mapi::RegTensor<T> vT0, vT1;
            uint32_t cFull = 64u;
            Mapi::MaskReg mFull = Mapi::UpdateMask<float>(cFull);
            for (uint16_t r = 0; r < nRows; ++r) {
                const uint32_t ro = static_cast<uint32_t>(r) * CA_HEAD_DIM;
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_BRC_B32>(vI, lsumPtr + static_cast<uint32_t>(r) * 64);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vO0, oPtr + ro);
                Mapi::DataCopy<float, Mapi::LoadDist::DIST_NORM>(vO1, oPtr + ro + 64);
                Mapi::Div(vO0, vO0, vI, mFull);
                Mapi::Div(vO1, vO1, vI, mFull);
                CaCastF32To(vT0, vO0, mFull);
                CaCastF32To(vT1, vO1, mFull);
                Mapi::DataCopy<T, Mapi::StoreDist::DIST_PACK_B32>(outPtr + ro, vT0, mFull);
                Mapi::DataCopy<T, Mapi::StoreDist::DIST_PACK_B32>(outPtr + ro + 64, vT1, mFull);
            }
        }
    }

    __aicore__ inline uint64_t OutputElem(const CaTaskDesc& d, uint32_t r) const
    {
        const uint32_t gBase = d.gTileIdx * tiling_->gPerTile;
        const uint32_t qHeadIdx = d.kvHeadIdx * tiling_->groupSize + gBase + (r % d.g);
        const uint32_t qRowIdx = d.qRowTileIdx * tiling_->qRowsPerTile + (r / d.g);
        return ((static_cast<uint64_t>(d.batchIdx) * tiling_->qSeqlen + qRowIdx) * tiling_->qHeads + qHeadIdx) *
               tiling_->headDim;
    }

    // 输出行回写 GM：同一 q 行内的 groupSize 个 head 在 GM 里相邻，相邻 q 行之间固定跳
    // qHeads 个 head，因此整段可以用一个多块描述符发完。原来每个 q 行单独一次 DataCopy，
    // q=64 形状每任务 8~16 次 1 KB 小搬运，每次都要付固定的搬运启动开销（实测 0.07~0.2 µs/次），
    // 这是 AIV 侧 MTE3 占比偏高的直接原因之一。
    __aicore__ inline void CopyRowsToGm(const CaTaskDesc& d, const LocalTensor<T>& src, uint32_t begin,
                                        uint32_t rows)
    {
        const uint32_t g = d.g;
        const uint32_t headBytes = tiling_->headDim * sizeof(T);
        const uint32_t blkBytes = g * headBytes;
        const uint32_t gapBytes = (tiling_->qHeads - g) * headBytes;
        if (g != 0u && rows % g == 0u && begin % g == 0u && blkBytes % 32u == 0u && gapBytes % 32u == 0u) {
            DataCopyParams cp;
            cp.blockCount = static_cast<uint16_t>(rows / g);
            cp.blockLen = static_cast<uint16_t>(blkBytes / 32u);
            cp.srcStride = 0; // UB 侧各块连续
            cp.dstStride = static_cast<uint16_t>(gapBytes / 32u);
            DataCopy(outGm_[OutputElem(d, begin)], src, cp);
            return;
        }
        uint32_t i = 0;
        while (i < rows) {
            const uint64_t base = OutputElem(d, begin + i);
            uint32_t n = 1;
            while (i + n < rows &&
                   OutputElem(d, begin + i + n) == base + static_cast<uint64_t>(n) * CA_HEAD_DIM) {
                ++n;
            }
            DataCopy(outGm_[base], src[i * CA_HEAD_DIM], n * CA_HEAD_DIM);
            i += n;
        }
    }

    __aicore__ inline void WriteOutput(const CaTaskDesc& d)
    {
        // online：O_run / lsum 已由逐拍 SoftmaxRows + OUpdateBeat（含 RunTask 收尾补拍）备齐，
        // 不再有 ReduceRows 与 O 缓冲信用；本 sub 分不到行时直接返回（PV 配对应 SoftmaxBeat /
        // OUpdateBeat 已按拍成对处理完）。
        if (nRows_ == 0u) {
            return;
        }
        ScaleAndCast();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        LocalTensor<T> outL = UbTTensor(CA_UB_OUT, CA_UB_ROWS * CA_HEAD_DIM);
        // 同一 q 行内的 groupSize 个 head 在 GM 里相邻，相邻 q 行之间固定跳 qHeads 个 head，
        // 整段用一个多块描述符发完（见 CopyRowsToGm）。
        CopyRowsToGm(d, outL, begin_, nRows_);
        // OUT 缓冲下一任务要被向量覆写：等本轮 MTE3 读完
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline uint32_t TaskIdxOfSeg(const CaTaskDesc& d, uint32_t s) const
    {
        const uint32_t base = ((d.batchIdx * tiling_->kvHeads + d.kvHeadIdx) * tiling_->qRowTileNum +
                               d.qRowTileIdx) *
                              tiling_->gTileNum +
                              d.gTileIdx;
        return base * tiling_->kSeg + s;
    }

    __aicore__ inline uint64_t MergeSlotElem(uint32_t taskIdx) const
    {
        return (static_cast<uint64_t>(taskIdx) * tiling_->mergeSlotBytes) / sizeof(float);
    }

    // m/l 区：两个 AIV sub 各有独立 2*CA_MAX_R 个 float，避免同核 sub0/sub1 互相覆盖
    __aicore__ inline uint64_t MLSlotElem(uint32_t taskIdx) const
    {
        const uint32_t rows = tiling_->rowsPerTask;
        return MergeSlotElem(taskIdx) + static_cast<uint64_t>(rows) * CA_HEAD_DIM +
               static_cast<uint64_t>(subIdx_) * 2ull * CA_MAX_R;
    }

    __aicore__ inline void StoreML(uint32_t taskIdx, const CaTaskDesc& d)
    {
        (void)d;
        LocalTensor<float> mL = UbF32Tensor(CA_UB_ML, 2 * CA_MAX_R);
        LocalTensor<float> lsum = UbF32Tensor(CA_UB_LSUM, CA_UB_ROWS * 64);
        // 每行的 m 取本 sub 的 CA_UB_ROWM（online 逐拍折叠后的最终运行最大）
        LocalTensor<float> rowm = UbF32Tensor(CA_UB_ROWM, CA_UB_ROWS * 64);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t r = 0; r < CA_MAX_R; ++r) {
            float mv = CA_MINV;
            if (r >= begin_ && (r - begin_) < nRows_) {
                mv = rowm.GetValue((r - begin_) * 64);
            }
            mL.SetValue(r, mv); // m：逐行独立，kSeg>1 合并时取组内最大
            float lv = 0.0f;
            if (r >= begin_ && (r - begin_) < nRows_) {
                lv = lsum.GetValue((r - begin_) * 64);
            }
            mL.SetValue(CA_MAX_R + r, lv);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(mergeGm_[MLSlotElem(taskIdx)], mL, 2 * CA_MAX_R);
        // 必须在跨核 SyncAll 之前确保 mL 落地，否则合并方读到 l=0 会把整行清零
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void WriteSentinelSlot(uint32_t taskIdx, const CaTaskDesc& d)
    {
        (void)d;
        LocalTensor<float> mL = UbF32Tensor(CA_UB_ML, 2 * CA_MAX_R);
        for (uint32_t r = 0; r < CA_MAX_R; ++r) {
            mL.SetValue(r, CA_MINV);
            mL.SetValue(CA_MAX_R + r, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(mergeGm_[MLSlotElem(taskIdx)], mL, 2 * CA_MAX_R);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void WriteMergeSlot(const CaTaskDesc& d)
    {
        const uint64_t base = MergeSlotElem(TaskIdxOfSeg(d, d.kvSegIdx));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        LocalTensor<float> oL = UbF32Tensor(CA_UB_O, CA_UB_ROWS * CA_HEAD_DIM);
        // merge 槽里本 sub 的行是连续的，一次 DataCopy 搬完
        DataCopy(mergeGm_[base + static_cast<uint64_t>(begin_) * CA_HEAD_DIM], oL, nRows_ * CA_HEAD_DIM);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        StoreML(TaskIdxOfSeg(d, d.kvSegIdx), d);
    }

    // kSeg > 1：把一个 (batch, kvHead, qRowTile) 下各 segment 的 O / l 用组内最大 m* 归一后合并
    __aicore__ inline void MergeTask(const CaTaskDesc& d)
    {
        const uint32_t begin = CaSubBegin(d.rows, subIdx_);
        const uint32_t nRows = CaSubRows(d.rows, subIdx_);
        LocalTensor<float> mL = UbF32Tensor(CA_UB_ML, 2 * CA_MAX_R);
        // merge 累加器按本 sub 的 nRows 分配，行号用本 sub 内局部下标
        LocalTensor<float> totO = UbF32Tensor(CA_UB_MRG, CA_UB_ROWS * CA_HEAD_DIM);
        LocalTensor<float> oL = UbF32Tensor(CA_UB_O, CA_UB_ROWS * CA_HEAD_DIM);
        LocalTensor<float> mStar = UbF32Tensor(CA_UB_SCAL + CA_SCAL_MRG, CA_MAX_R);
        LocalTensor<float> denL = UbF32Tensor(CA_UB_SCAL + CA_SCAL_TMP, 32);

        // 1) 组内最大值 m*
        for (uint32_t i = 0; i < nRows; ++i) {
            mStar.SetValue(i, CA_MINV);
        }
        for (uint32_t s = 0; s < effSeg_; ++s) {
            const uint64_t mBase = MLSlotElem(TaskIdxOfSeg(d, s));
            DataCopy(mL, mergeGm_[mBase], CA_MAX_R);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t i = 0; i < nRows; ++i) {
                const float mv = mL.GetValue(begin + i);
                if (mv > mStar.GetValue(i)) {
                    mStar.SetValue(i, mv);
                }
            }
        }

        // 2) 累加 O 与分母
        // 累加 O 与分母
        Duplicate(totO, 0.0f, CA_UB_ROWS * CA_HEAD_DIM);
        for (uint32_t i = 0; i < nRows; ++i) {
            denL.SetValue(i, 0.0f);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t s = 0; s < effSeg_; ++s) {
            const uint64_t sBase = MergeSlotElem(TaskIdxOfSeg(d, s));
            const uint64_t mBase = MLSlotElem(TaskIdxOfSeg(d, s));
            DataCopy(oL, mergeGm_[sBase + static_cast<uint64_t>(begin) * CA_HEAD_DIM], nRows * CA_HEAD_DIM);
            DataCopy(mL, mergeGm_[mBase], 2 * CA_MAX_R);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t i = 0; i < nRows; ++i) {
                const float lv = mL.GetValue(CA_MAX_R + begin + i);
                float w = 0.0f;
                if (lv > 0.0f) {
                    w = ExpScalar(mL.GetValue(begin + i) - mStar.GetValue(i));
                }
                denL.SetValue(i, denL.GetValue(i) + w * lv);
                if (w == 0.0f) {
                    Duplicate(oL[i * CA_HEAD_DIM], 0.0f, CA_HEAD_DIM);
                } else {
                    Muls(oL[i * CA_HEAD_DIM], oL[i * CA_HEAD_DIM], w, CA_HEAD_DIM);
                }
            }
            PipeBarrier<PIPE_V>();
            Add(totO, totO, oL, nRows * CA_HEAD_DIM);
            PipeBarrier<PIPE_V>();
            // oL 在下一轮被重新 DataCopy 覆盖：必须等本轮 V 读完，否则丢数据
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        // 3) 归一化并写回
        LocalTensor<T> outL = UbTTensor(CA_UB_OUT, CA_UB_ROWS * CA_HEAD_DIM);
        // 先在 UB 里把本 sub 的所有行都归一化/转型完，再一次性同步给 MTE3：
        // 原来每行一对 V_MTE3 / MTE3_V，32 行就是 64 次往返。
        for (uint32_t i = 0; i < nRows; ++i) {
            const float den = denL.GetValue(i);
            const float inv = den > 0.0f ? (1.0f / den) : 0.0f;
            Muls(totO[i * CA_HEAD_DIM], totO[i * CA_HEAD_DIM], inv, CA_HEAD_DIM);
            Cast(outL[i * CA_HEAD_DIM], totO[i * CA_HEAD_DIM], RoundMode::CAST_RINT, CA_HEAD_DIM);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // 相邻 q 行在 GM 里按固定间隔排布，整段用一个多块描述符发完（见 CopyRowsToGm）
        CopyRowsToGm(d, outL, begin, nRows);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    const __gm__ CrossAttentionTilingData* tiling_;
    GlobalTensor<T> outGm_;
    GlobalTensor<int64_t> lensGm_;
    GlobalTensor<float> mergeGm_;
    uint32_t coreIdx_;
    uint32_t subIdx_;
    uint32_t begin_;
    uint32_t nRows_;
    float scaleV_;
    uint32_t effSeg_;  // 按实际 kvLen 推出的生效段数（1 = 不需要 merge）
    // P 的 UB 双槽信用：事件 id 用 2/3，与 OUTPUT 缓冲的 MTE3_V(EVENT_ID0) 隔离。
    // 同一 id 上 Set/Wait 必须严格交替，所以槽号用跨任务连续的 pUbSeq_ 而不是 beat。
    static constexpr int32_t CA_PUB_S0 = 2;
    static constexpr int32_t CA_PUB_S1 = 3;
    uint32_t pUbSeq_ = 0u;  // 跨任务连续的 P 拍号
};

} // namespace ca

#endif // CROSS_ATTN_ARCH35_FUSION_KERNEL_H
