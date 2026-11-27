/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

/*!
 * \file rms_norm_channels_common.h
 * \brief Host/kernel shared shape-family routing keys for RmsNormChannels.
 *
 * The host tiling picks exactly one key per launch and the kernel entry
 * dispatches on it.  Keeping the values here guarantees the two sides cannot
 * drift apart.
 */

#ifndef RMS_NORM_CHANNELS_COMMON_H
#define RMS_NORM_CHANNELS_COMMON_H

#include <cstdint>

// C fits in UB: keep the whole [C, Ntile] tile, one input read + one write.
constexpr uint32_t RMS_NORM_CHANNELS_TK_SMALL_C = 1001U;
// Large C with small N: split C over all vector cores, join partial sums.
constexpr uint32_t RMS_NORM_CHANNELS_TK_SPLIT_C = 2001U;
// Large C and large N: stream C chunks twice (sum, then normalize).
constexpr uint32_t RMS_NORM_CHANNELS_TK_STREAM = 3001U;
// C == 1: the reduction is just x * x.
constexpr uint32_t RMS_NORM_CHANNELS_TK_SINGLE_CHANNEL = 4001U;
// Very short N: one SIMT warp per output position.
constexpr uint32_t RMS_NORM_CHANNELS_TK_SIMT_WARP = 5001U;

#endif // RMS_NORM_CHANNELS_COMMON_H
