/**
 * CrossAttention native tiling structs, shared by host and device.
 * Plain POD: no __aicore__ / __gm__ qualifiers, no bool / enum / bitfield / long.
 */
#ifndef CROSS_ATTENTION_ARCH35_NATIVE_TILING_H
#define CROSS_ATTENTION_ARCH35_NATIVE_TILING_H

#include <cstdint>
#include <cstddef>

#define CA_TILING_VERSION 1u

// dtype encoding carried in shape / tiling; enum is not allowed here
#define CA_DTYPE_BF16 0u
#define CA_DTYPE_FP16 1u

namespace Xllm {

struct CrossAttentionShape {
    uint32_t batch;
    uint32_t qSeqlen;
    uint32_t qHeads;
    uint32_t kvHeads;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t dtype; // CA_DTYPE_*
};

struct CrossAttentionTilingData {
    // Shape and scaling
    uint32_t batch;
    uint32_t qHeads;
    uint32_t kvHeads;
    uint32_t groupSize;
    uint32_t headDim;
    uint32_t qSeqlen;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    float scaleValue;
    // Task space
    uint32_t qRowsPerTile;      // q rows per task: (16/groupSize)*4 when groupSize divides 16
    uint32_t qRowTileNum;       // ceil(qSeqlen / qRowsPerTile)
    uint32_t rowsPerTask;       // R limit = qRowsPerTile * groupSize (<= 64)
    uint32_t kvTile;            // 128 (fixed)
    uint32_t kSeg;
    uint32_t mergeSegBlocks;    // ceil(maxNumBlocksPerBatch / kSeg)
    uint32_t totalTaskNum;
    uint32_t usedCoreNum;
    uint32_t dtype;
    // Workspace segments (byte offsets, 64 B aligned)
    uint64_t mergeSlotBytes;    // bytes per slot; 0 when kSeg == 1
    uint64_t sStageOff;
    uint64_t pStageOff;
    uint64_t oStageOff;
    uint64_t userWsBytes;
    uint32_t tilingVersion;
    uint32_t reserved0;
};

static_assert(sizeof(CrossAttentionTilingData) % 8 == 0, "tiling ABI: 8B alignment");

static_assert(sizeof(CrossAttentionTilingData) == 120, "arch35 tiling ABI size");
static_assert(offsetof(CrossAttentionTilingData, mergeSlotBytes) == 72, "arch35 tiling ABI workspace");
static_assert(offsetof(CrossAttentionTilingData, tilingVersion) == 112, "arch35 tiling ABI version");

} // namespace Xllm

#endif // CROSS_ATTENTION_ARCH35_NATIVE_TILING_H
