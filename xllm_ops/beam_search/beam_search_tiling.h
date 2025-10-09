#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
    
BEGIN_TILING_DATA_DEF(BeamSearchTilingData)
TILING_DATA_FIELD_DEF(int32_t, num_sequences);
TILING_DATA_FIELD_DEF(int32_t, sequence_length);
TILING_DATA_FIELD_DEF(int32_t, beam_width);
TILING_DATA_FIELD_DEF(int32_t, top_k);
TILING_DATA_FIELD_DEF(int32_t, request_num);
TILING_DATA_FIELD_DEF(int32_t, core_num);
TILING_DATA_FIELD_DEF(int32_t, min_size);
TILING_DATA_FIELD_DEF(int32_t, step_size);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topKTilingData1);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BeamSearch, BeamSearchTilingData)

}  // namespace optiling