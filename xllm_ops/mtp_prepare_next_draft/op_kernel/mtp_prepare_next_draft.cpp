/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "kernel_operator.h"
#include "mtp_prepare_next_draft.h"

extern "C" __global__ __aicore__ void mtp_prepare_next_draft(
    GM_ADDR acceptedTokens,
    GM_ADDR acceptedEmbeddings,
    GM_ADDR embeddingPlaceholder,
    GM_ADDR basePositions,
    GM_ADDR baseKvSeqLens,
    GM_ADDR blockTables,
    GM_ADDR draftTokenIds,
    GM_ADDR draftEmbeddings,
    GM_ADDR draftPositions,
    GM_ADDR draftKvSeqLens,
    GM_ADDR draftCacheSlots,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  MtpPrepareNextDraftKernel<DTYPE_ACCEPTED_EMBEDDINGS> op;
  op.Init(acceptedTokens,
          acceptedEmbeddings,
          embeddingPlaceholder,
          basePositions,
          baseKvSeqLens,
          blockTables,
          draftTokenIds,
          draftEmbeddings,
          draftPositions,
          draftKvSeqLens,
          draftCacheSlots,
          &tiling_data);
  op.Process();
}
