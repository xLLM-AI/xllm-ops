/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_fused_reducesum_div.h
 * \brief
 */
#include "kernel_operator.h"

// Architecture isolation via folder-separated adapters instead of scattered macros.
// Each ArchAdapter exposes the same interface but encapsulates the per-chip hazard-sync
// and buffer strategy:
//   - arch35/arch35.h : ascend950 (A5, __NPU_ARCH__ == 3510). Async V/S/MTE pipes and a
//     ReduceSum that truly consumes its sharedTmpBuffer, so explicit syncs and a dedicated
//     work buffer are mandatory.
//   - arch32/arch32.h : A2 (__CCE_AICORE__ == 220) and A3 (__NPU_ARCH__ == 3003/3113).
//     TQue EnQue/DeQue already order the pipes and ReduceSum ignores the temp buffer, so
//     the syncs are no-ops and no work buffer is allocated.
// To support a new chip, add a new arch<XX>/ folder with its own ArchAdapter and route it
// here; the compute logic below stays untouched.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/arch35.h"
#else
#include "arch32/arch32.h"
#endif

constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t SINGLE_CONST_TWO = 2;
using namespace AscendC;

namespace kernels {
    template <typename INPUT_TP>
    class MoeFusedReducesumDiv {
    public:
        __aicore__ MoeFusedReducesumDiv() {};
    
        __aicore__ inline void init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const MoeFusedReducesumDivTilingData &tiling) {
            this->tiling = tiling;
            int32_t core_id = GetBlockIdx();

            n = tiling.n;
            n_ailgn = tiling.nAlign;
            big_core_num = tiling.bigCoreNum;
            little_core_num = tiling.littleCoreNum;
            avg_core_calc_rows = tiling.avgCoreCalcRows;
            single_core_calc_rows = tiling.singleCoreCalcRows;

            single_core_calc_rows /= BUFFER_NUM;

            if (big_core_num == 0) {
                core_offset = core_id * avg_core_calc_rows;
                core_calc_rows = avg_core_calc_rows;
            } else if (core_id < big_core_num && big_core_num != 0) {
                core_offset = core_id * (avg_core_calc_rows + 1);
                core_calc_rows = avg_core_calc_rows + 1;
            } else {
                core_offset = big_core_num * (avg_core_calc_rows + 1) + (core_id - big_core_num) * avg_core_calc_rows;
                core_calc_rows = avg_core_calc_rows;
            }
            tail = core_calc_rows % single_core_calc_rows;
            loop = core_calc_rows / single_core_calc_rows;

            

            inputGM.SetGlobalBuffer((__gm__ INPUT_TP*)input);
            outputGM.SetGlobalBuffer((__gm__ INPUT_TP*)output);

            pipe.InitBuffer(inputQUE, BUFFER_NUM, single_core_calc_rows * n_ailgn * sizeof(float));
            pipe.InitBuffer(outputQUE, BUFFER_NUM, single_core_calc_rows * n_ailgn * sizeof(float));
            pipe.InitBuffer(sumBUF, single_core_calc_rows / BUFFER_NUM * sizeof(float));
            // Separate staging buffer for the 16-bit input so the widening Cast reads
            // from a distinct buffer instead of casting in-place within inputQUE (which
            // caused overlapping source/destination byte ranges on ascend950).
            pipe.InitBuffer(castBUF, single_core_calc_rows * n_ailgn * sizeof(int16_t));
            // Dedicated ReduceSum sharedTmpBuffer. Whether it is actually allocated is
            // decided by the per-architecture ArchAdapter (A5 needs it, A2/A3 skip it to
            // keep more UB free). See arch35.h / arch32.h.
            ArchAdapter::InitWorkBuf<float>(pipe, workBUF, single_core_calc_rows * n_ailgn);
        }
    
        __aicore__ inline void process() {
            int32_t core_id = GetBlockIdx();
            loop *= BUFFER_NUM;
            for (int32_t block_id = 0; block_id < loop; block_id++){
                processOne(block_id, single_core_calc_rows);
            }
            if (tail != 0){
                processOne(loop, tail);
            }
        }

        __aicore__ inline void processOne(int32_t block_id, int32_t calc_rows) {
	        LocalTensor<float> inputLocal = inputQUE.AllocTensor<float>();
            // load
            DataCopyExtParams dataCopyParams;
            DataCopyPadExtParams<INPUT_TP> padParams;
            dataCopyParams.blockCount = calc_rows;
            dataCopyParams.blockLen = n * sizeof(INPUT_TP);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = n_ailgn - n;
            SetPadValue(0);

            LocalTensor<INPUT_TP> castSrc = castBUF.Get<INPUT_TP>();
            if (sizeof(INPUT_TP) == SINGLE_CONST_TWO) {
                // 16-bit input: stage into the separate castBUF, then widen-cast to fp32.
                DataCopyPad(castSrc, inputGM[(core_offset + block_id * single_core_calc_rows) * n], dataCopyParams, padParams);
                // castBUF is a plain TBuf outside the inputQUE pipeline, so the widening
                // Cast (vector) below must wait for this DataCopyPad (MTE2). The adapter
                // inserts the MTE2->V sync only where the pipeline does not already order it.
                ArchAdapter::SyncMte2ToV();
            } else {
                // fp32 input: load directly into the fp32 compute buffer.
                DataCopyPad(inputLocal.ReinterpretCast<INPUT_TP>(), inputGM[(core_offset + block_id * single_core_calc_rows) * n], dataCopyParams, padParams);
            }

            inputQUE.EnQue(inputLocal);
            inputLocal = inputQUE.DeQue<float>();

            // cast
            // The 16-bit input was staged into a SEPARATE castBUF buffer above, then cast
            // into the fp32 inputLocal buffer here. Using two independent buffers avoids
            // the in-place tail->head overlap hazard on ascend950 where the fp32
            // destination would otherwise overwrite unread 16-bit source bytes and
            // corrupt every other element.
            if (sizeof(INPUT_TP) == SINGLE_CONST_TWO) {
                Cast(inputLocal, castSrc, RoundMode::CAST_NONE, calc_rows * n_ailgn);
                PipeBarrier<PIPE_V>();
            }
            
            inputQUE.EnQue(inputLocal);
            // compute
            inputLocal = inputQUE.DeQue<float>();
            LocalTensor<float> sumLocal = sumBUF.Get<float>();
            // On architectures that need it (A5) workBUF is initialized and serves as the
            // ReduceSum scratch; otherwise it stays empty and ReduceScratch falls back to
            // outputLocal. Acquiring the tensor is only valid when the buffer was allocated.
            LocalTensor<float> workLocal;
            if constexpr (ArchAdapter::kNeedWorkBuf) {
                workLocal = workBUF.Get<float>();
            }
            LocalTensor<float> outputLocal = outputQUE.AllocTensor<float>();

            for(int32_t i=0; i<calc_rows; i++){
                // Pick the ReduceSum sharedTmpBuffer per architecture: A5 uses the dedicated
                // workLocal (ReduceSum truly consumes it), A2/A3 reuse outputLocal as scratch.
                ReduceSum(sumLocal[i], inputLocal[i * n_ailgn],
                          ArchAdapter::ReduceScratch(workLocal, outputLocal)[i * n_ailgn], n);
                // ReduceSum (V) writes the row sum, GetValue below is a scalar read. The
                // adapter inserts a V->S sync where the pipes are asynchronous (A5).
                ArchAdapter::SyncVToS();
                Muls<float>(outputLocal[i * n_ailgn], inputLocal[i * n_ailgn], (float)1.0/sumLocal.GetValue(i), n);
            }

            outputQUE.EnQue(outputLocal);
            outputLocal = outputQUE.DeQue<float>();
            LocalTensor<INPUT_TP> castDst = castBUF.Get<INPUT_TP>();
            if (sizeof(INPUT_TP) == SINGLE_CONST_TWO) {
                // Narrow-cast fp32 -> 16-bit into the SEPARATE castBUF to avoid the
                // in-place source/destination overlap on ascend950 that otherwise
                // corrupts every other output element.
                Cast(castDst, outputLocal, RoundMode::CAST_RINT, calc_rows * n_ailgn);
                // Vector(Cast) writes castDst, then MTE3(DataCopyPad) reads it. castBUF is
                // a TBuf (no queue) so a V->MTE3 sync is required on async-pipe chips; a
                // bare PipeBarrier<PIPE_V> would NOT block the MTE3 pipe.
                ArchAdapter::SyncVToMte3();
            }

            DataCopyExtParams dataCopyParamsOUT;
            dataCopyParamsOUT.blockCount = calc_rows;
            dataCopyParamsOUT.blockLen = n * sizeof(INPUT_TP);
            dataCopyParamsOUT.srcStride = (n_ailgn - n) * sizeof(INPUT_TP);
            dataCopyParamsOUT.dstStride = 0;
            // Use dataCopyParamsOUT (not the load params): the UB result is laid out with
            // n_ailgn stride per row, so srcStride must skip the (n_ailgn - n) padding
            // elements between rows when copying the valid n elements out to GM.
            if (sizeof(INPUT_TP) == SINGLE_CONST_TWO) {
                DataCopyPad(outputGM[(core_offset + block_id * single_core_calc_rows) * n], castDst, dataCopyParamsOUT);
            } else {
                DataCopyPad(outputGM[(core_offset + block_id * single_core_calc_rows) * n], outputLocal.ReinterpretCast<INPUT_TP>(), dataCopyParamsOUT);
            }
            inputQUE.FreeTensor(inputLocal);
            outputQUE.FreeTensor(outputLocal);
        }
    
    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inputQUE;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outputQUE;
        TBuf<TPosition::VECCALC> sumBUF;
        TBuf<TPosition::VECCALC> castBUF;
        TBuf<TPosition::VECCALC> workBUF;
        
        GlobalTensor<INPUT_TP> inputGM;
        GlobalTensor<INPUT_TP> outputGM;
    
        MoeFusedReducesumDivTilingData tiling;

        int32_t n;
        int32_t n_ailgn;
        int32_t big_core_num;
        int32_t little_core_num;
        int32_t avg_core_calc_rows;
        int32_t single_core_calc_rows;

        int32_t core_offset;
        int32_t core_calc_rows;
        int32_t tail;
        int32_t loop; 
    };
}
