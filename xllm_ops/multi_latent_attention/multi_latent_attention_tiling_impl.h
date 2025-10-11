#ifndef MLA_TILING_H
#define MLA_TILING_H

#include "multi_latent_attention_tiling_dependency.h"
#include "mla.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"
 
namespace AtbOps {
ge::graphStatus MLATiling(gert::TilingContext *context);
ge::graphStatus GetMLATilingParam(OpParam::MLA param, const MLAInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize);
}

#endif // MLA_TILING_H
