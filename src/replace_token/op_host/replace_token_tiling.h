
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReplaceTokenTilingData)
TILING_DATA_FIELD_DEF(int32_t, sequenceLength);
TILING_DATA_FIELD_DEF(int32_t, max_tokens);
TILING_DATA_FIELD_DEF(int32_t, blength);
// TILING_DATA_FIELD_DEF_ARR(int64_t,256,groupOffset);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReplaceToken, ReplaceTokenTilingData)
} // namespace optiling
