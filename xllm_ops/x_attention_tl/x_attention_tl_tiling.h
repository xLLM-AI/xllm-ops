
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(XAttentionTlTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch_size);
  TILING_DATA_FIELD_DEF(uint32_t, num_heads);
  TILING_DATA_FIELD_DEF(uint32_t, head_size);
  TILING_DATA_FIELD_DEF(uint32_t, q_length);
  TILING_DATA_FIELD_DEF(uint32_t, shared_k_length);
  TILING_DATA_FIELD_DEF(uint32_t, unshared_k_length);
  TILING_DATA_FIELD_DEF(uint32_t, beam_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_num);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(XAttentionTl, XAttentionTlTilingData)
}
