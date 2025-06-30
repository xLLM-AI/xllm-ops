/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

 #include "kernel_operator.h"
 using namespace AscendC;
 class ReplaceToken  {
 public:
   __aicore__ inline ReplaceToken() {}
   __aicore__ inline void Init(GM_ADDR forkedTokenIds, GM_ADDR lastStepOutPutTokenIds, GM_ADDR Out, int32_t sequenceLength) {
     forkedTokenIdsGM.SetGlobalBuffer((__gm__ int32_t *)forkedTokenIds);
     lastStepOutPutTokenIdsGM.SetGlobalBuffer((__gm__ int64_t *)lastStepOutPutTokenIds);
     outGM.SetGlobalBuffer((__gm__ int32_t *)Out);
     this->sequenceLength = sequenceLength;
     pipe.InitBuffer(vecOutQue,1,sequenceLength*sizeof(int32_t));
     pipe.InitBuffer(vecLastIn,1,sequenceLength*sizeof(int64_t));
     pipe.InitBuffer(vecInQue,1,sequenceLength*sizeof(int32_t));
     pipe.InitBuffer(tmpBuf,sequenceLength*sizeof(int32_t));
   }
   __aicore__ inline void Process() {
     LocalTensor<int32_t> forkedIn = vecInQue.AllocTensor<int32_t>();
     LocalTensor<int64_t> lastIn = vecLastIn.AllocTensor<int64_t>();
     DataCopyPadExtParams<int32_t> forkedInpadParams;
     forkedInpadParams.isPad = true;
     forkedInpadParams.paddingValue = 0;
     DataCopyExtParams forkedInParams;
     forkedInParams.blockLen = sequenceLength * sizeof(int32_t);
     forkedInParams.blockCount = 1;
     forkedInParams.srcStride = 0;
     forkedInParams.dstStride = 0;
     AscendC::DataCopyPad(forkedIn, forkedTokenIdsGM, forkedInParams, forkedInpadParams);
     DataCopyExtParams lastInParams;
     lastInParams.blockLen = sequenceLength * sizeof(int64_t);
     lastInParams.blockCount = 1;
     lastInParams.srcStride = 0;
     lastInParams.dstStride = 0;
     DataCopyPadExtParams<int64_t> lastInpadParams;
     lastInpadParams.isPad = true;
     lastInpadParams.paddingValue = 0;
     AscendC::DataCopyPad(lastIn,lastStepOutPutTokenIdsGM,lastInParams,lastInpadParams);
     vecInQue.EnQue(forkedIn);
     vecLastIn.EnQue(lastIn);
     LocalTensor<int32_t> forkedInTmp = vecInQue.DeQue<int32_t>();
     LocalTensor<int32_t> forkedOut = vecOutQue.AllocTensor<int32_t>();
     LocalTensor<int64_t> lastStepOutPutTokenIdsInUb = vecLastIn.DeQue<int64_t>();
     Muls(forkedOut, forkedInTmp, int32_t(1), sequenceLength);
     for(int32_t i=0;i<sequenceLength;i++){
      if(forkedOut.GetValue(i)<0){
        forkedOut.SetValue(i,static_cast<int32_t>(lastStepOutPutTokenIdsInUb.GetValue((0-forkedOut.GetValue(i))-1)));
      }
     }
     AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(sequenceLength * sizeof(int32_t)), 0, 0, 0};
     AscendC::DataCopyPad(outGM, forkedOut, copyParams);
     vecInQue.FreeTensor(forkedInTmp);
     vecOutQue.FreeTensor(forkedOut);
     vecLastIn.FreeTensor(lastStepOutPutTokenIdsInUb);
   }
 private:
  AscendC::GlobalTensor<int32_t> forkedTokenIdsGM;
  AscendC::GlobalTensor<int64_t> lastStepOutPutTokenIdsGM;
  AscendC::GlobalTensor<int32_t> outGM;
  int32_t sequenceLength;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> vecOutQue;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> vecInQue;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> vecLastIn;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
 };
 
