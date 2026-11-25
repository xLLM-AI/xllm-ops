/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "ecr_routing_normalize_compact_vf.h"

namespace {

using ecr_routing_normalize_compact_impl::kSimtThreads;

template <typename Tiling>
__aicore__ inline void OutputCompact(GM_ADDR prob_gm,
                                     GM_ADDR tau_val_gm,
                                     GM_ADDR tau_idx_gm,
                                     GM_ADDR sums_gm,
                                     GM_ADDR cnts_gm,
                                     GM_ADDR ids_gm,
                                     GM_ADDR weights_gm,
                                     const Tiling& t) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> scan_buffer;
  pipe.InitBuffer(scan_buffer, 2 * kSimtThreads * sizeof(uint32_t));
  auto scan_tensor = scan_buffer.Get<uint32_t>();
  const uint32_t total_tasks = t.e * t.b;
  const uint32_t core_num = AscendC::GetBlockNum();
  // Tasks own disjoint output slices; same-core VF calls serialize UB reuse.
  for (uint32_t task = AscendC::GetBlockIdx(); task < total_tasks;
       task += core_num) {
    const uint32_t expert = task / t.b;
    const uint32_t sample = task % t.b;
    const uint32_t length = t.seq_len[sample];
    const uint32_t selected_limit = t.seq_k[sample];
    if (length == 0 || selected_limit == 0) {
      continue;
    }
    const uint32_t base = t.seq_base[sample];
    const uint64_t prob_offset = static_cast<uint64_t>(expert) * t.s + base;
    const uint64_t output_offset =
        static_cast<uint64_t>(expert) * t.capacity_c + t.c_offset[sample];
    AscendC::Simt::VF_CALL<ecr_routing_normalize_compact_impl::OutputCompactVf>(
        AscendC::Simt::Dim3(kSimtThreads),
        reinterpret_cast<__gm__ float*>(prob_gm),
        reinterpret_cast<__gm__ float*>(tau_val_gm),
        reinterpret_cast<__gm__ int32_t*>(tau_idx_gm),
        reinterpret_cast<__gm__ float*>(sums_gm),
        reinterpret_cast<__gm__ float*>(cnts_gm),
        reinterpret_cast<__gm__ int32_t*>(ids_gm),
        reinterpret_cast<__gm__ float*>(weights_gm),
        reinterpret_cast<__ubuf__ uint32_t*>(scan_tensor.GetPhyAddr()),
        prob_offset,
        output_offset,
        task,
        base,
        length,
        selected_limit,
        t.eps);
  }
}

}  // namespace

extern "C" __global__ __aicore__ void ecr_routing_normalize_compact(
    GM_ADDR prob,
    GM_ADDR tau_val,
    GM_ADDR tau_idx,
    GM_ADDR sums,
    GM_ADDR cnts,
    GM_ADDR ids,
    GM_ADDR weights,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(t, tiling);
  (void)workspace;
  OutputCompact(prob, tau_val, tau_idx, sums, cnts, ids, weights, t);
}
