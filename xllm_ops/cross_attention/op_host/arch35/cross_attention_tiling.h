// Native tiling is also consumed from the OPC kernel-only source tree.
#ifndef CROSS_ATTENTION_ARCH35_HOST_TILING_H
#define CROSS_ATTENTION_ARCH35_HOST_TILING_H

#include "../../op_kernel/arch35/cross_attention_tiling.h"

namespace Xllm {
// Host side tiling. Returns 0 on success, non-zero when the shape is out of range (reason on stderr)
int32_t ComputeCrossAttentionTiling(const CrossAttentionShape& shape, float scaleValue, uint32_t aicCoreNum,
                                    CrossAttentionTilingData& out);

} // namespace Xllm

#endif // CROSS_ATTENTION_ARCH35_HOST_TILING_H
