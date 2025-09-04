/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file pp_mat_mul_default.cc
 * \brief
 */
#include "pp_mat_mul_default.h"
#include "pp_mat_mul_common_tiling.h"
#include "pp_mat_mul_tiling.h"
#include <unordered_map>
// #include "matmul_v3_tiling.h"
// #include "tiling/tiling_templates_registry.h"
#include "register/op_def_registry.h"
// #include "cube_tiling_runtime.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
// #include "op_log.h"
#define OP_LOGD(x, y, z) std::cout << x << " " << y << " " << z << std::endl;

// using namespace optiling::pp_mat_mul; // todo：使用别的地方定义在pp_mat_mul的东西

namespace {
constexpr uint64_t L1_DESCALE_BUFFER_SIZE_MAX = 6144;
constexpr uint64_t CONST_3 = 3;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_16 = 16;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_256 = 256;
constexpr uint64_t CONST_512 = 512;

constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t COMPUTE_TYPE_IDX_CANN = 3;
constexpr size_t COMPUTE_TYPE_IDX_ATB = 10;
constexpr size_t EINSUM_MODE = 4;

constexpr uint32_t DTYPE_BIT_COUNT = 2;
constexpr uint32_t FORMAT_BIT_COUNT = 1;
constexpr uint32_t COMPUTE_TYPE_BIT_COUNT = 3;
constexpr uint32_t MAX_ATTRS_NUM = 4;
constexpr uint32_t EN_SHUFFFLE_IDX_ATB = 7;

/**
 * 获取矩阵A中的m值
 */
uint64_t GetATensorM(gert::Shape inputAStorageShape, bool transA) {
    size_t inputADimNums = inputAStorageShape.GetDimNum();
    if (inputADimNums == DIM_2) {
        return transA ? inputAStorageShape[1] : inputAStorageShape[0];
    } else if (inputADimNums == DIM_3) {
        return transA ? inputAStorageShape[DIM_2] : inputAStorageShape[1];
    }
    return 0;
}

/**
 * 获取矩阵A中的k值
 */
uint64_t GetATensorK(gert::Shape inputAStorageShape, bool transA) {
    size_t inputADimNums = inputAStorageShape.GetDimNum();
    if (inputADimNums == DIM_2) {
        return transA ? inputAStorageShape[0] : inputAStorageShape[1];
    } else if (inputADimNums == DIM_3) {
        return transA ? inputAStorageShape[1] : inputAStorageShape[DIM_2];
    }
    return 0;
}

/**
 * 获取矩阵C中的n值
 */
uint64_t GetCTensorN(gert::Shape outputCStorageShape) {
    size_t outputCDimNums = outputCStorageShape.GetDimNum();
    if (outputCDimNums == DIM_2) {
        return outputCStorageShape[1];
    } else if (outputCDimNums == DIM_3) {
        return outputCStorageShape[DIM_2];
    }
    return 0;
}
}

namespace optiling {
namespace pp_mat_mul {

void PpMatmulDefaultTilingData::SetBaseShape(uint64_t batchSize, uint64_t m, uint64_t k, uint64_t n) {
    opShape.batchSize = batchSize;
    opShape.m = m;
    opShape.k = k;
    opShape.n = n;
}

void PpMatmulDefaultTilingData::SetBaseOp(uint64_t coreNum, uint64_t l0cSize, uint64_t mBase, uint64_t nBase, const MatMulInfo &mmInfo, bool isAscend310P) {
    opShape.m0 = mBase;
    opShape.n0 = nBase;
    mLoop = CeilDiv(opShape.m, opShape.m0);
    nLoop = CeilDiv(opShape.n, opShape.n0);
    coreLoop = opShape.batchSize * mLoop * nLoop;
    if (!isAscend310P && mLoop == 1 && mmInfo.transB && static_cast<uint64_t>(coreLoop % coreNum) < 
        static_cast<uint64_t>(coreNum / CONST_4) * CONST_3) {
        mBase = RoundUp(opShape.m, CONST_16);
        opShape.m0 = mBase;
        uint64_t maxN0 = l0cSize / (mBase * sizeof(float));
        uint64_t x = CeilDiv(opShape.n, coreNum);
        uint64_t y = CeilDiv(x, maxN0);
        nBase = RoundUp(CeilDiv(x, y), CONST_16);
        uint64_t rqdL0CSize = mBase * nBase * sizeof(float);
        if (rqdL0CSize < l0cSize &&
            (mBase + nBase) * CONST_256 * sizeof(uint16_t) < L1AB_PINGPONG_BUFFER_SIZE) {
            opShape.n0 = nBase;
            nLoop = CeilDiv(opShape.n, opShape.n0);
            coreLoop = opShape.batchSize * nLoop;
        }
    }
    blockDim = std::min(coreLoop, coreNum);
}

void PpMatmulDefaultTilingData::End(const MatMulInfo &mmInfo, bool isAscend310P) {
    uint64_t shapeSum = opShape.m0 + opShape.n0;
    if (!isAscend310P) {
        uint64_t k0Max = shapeSum == 0UL
                        ? L1AB_PINGPONG_BUFFER_SIZE
                        : static_cast<uint64_t>(static_cast<float>(L1AB_PINGPONG_BUFFER_SIZE)
                            / (shapeSum * mmInfo.inDtype));
        opShape.k0 = k0Max < CUBE_BLOCK_SIZE ? RoundDown(k0Max, BLOCK_SIZE) : RoundDown(k0Max, CUBE_BLOCK_SIZE);
        if (opShape.k0 > CONST_512) {
            opShape.k0 = RoundDown(opShape.k0, CONST_512);
        }
    } else {
        uint32_t k0Max = (shapeSum == 0) ? UB_LIMIT_SIZE_910A : (UB_LIMIT_SIZE_910A / shapeSum);
        opShape.k0 = k0Max < CUBE_BLOCK_SIZE ? k0Max / BLOCK_SIZE * BLOCK_SIZE : \
            k0Max / CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE; // k0Max less than 256, matrix 16
    }
        kLoop = CeilDiv(opShape.k, opShape.k0);
}

bool PpMatMulDefault::GetMatMulInfo()
{
    size_t indexA = 0;
    size_t indexB = indexA + static_cast<size_t>(1);
    size_t indexBias = indexB + static_cast<size_t>(1);
    size_t idxC = 0;

    auto inputDType = context_->GetInputDesc(indexA)->GetDataType();
    //都得校验是否为null
    auto inputAStorageShape = context_->GetInputShape(indexA)->GetStorageShape();
    auto outputCStorageShape = context_->GetOutputShape(idxC)->GetStorageShape();
    size_t inputADimNums = inputAStorageShape.GetDimNum();
    matMulInfo_.formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(indexA)->GetStorageFormat()));
    matMulInfo_.formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(indexB)->GetStorageFormat()));
    matMulInfo_.formatC = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(indexA)->GetStorageFormat()));
    matMulInfo_.transA = *context_->GetAttrs()->GetAttrPointer<bool>(indexA);
    matMulInfo_.transB = *context_->GetAttrs()->GetAttrPointer<bool>(indexB);

    matMulInfo_.isInt8 = (inputDType == ge::DT_INT8);
    matMulInfo_.dtypeA = context_->GetInputDesc(indexA)->GetDataType();
    matMulInfo_.dtypeB = context_->GetInputDesc(indexB)->GetDataType();
    matMulInfo_.dtypeC = context_->GetOutputDesc(idxC)->GetDataType();
    matMulInfo_.biasFlag = context_->GetOptionalInputDesc(indexBias) != nullptr;
    auto attrs = context_->GetAttrs();
    size_t numAttrs = attrs->GetAttrNum();
    auto computeTypeValue = 0;
    if (numAttrs <= MAX_ATTRS_NUM) {
        computeTypeValue = *context_->GetAttrs()->GetAttrPointer<int>(COMPUTE_TYPE_IDX_CANN);
    } else {
        computeTypeValue = *context_->GetAttrs()->GetAttrPointer<int>(COMPUTE_TYPE_IDX_ATB);
    }

    if (hardwareInfo_.socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        if (inputADimNums == DIM_3) {
            matMulInfo_.batchSize = 1;
        } else if (inputADimNums == DIM_4)
        {
            matMulInfo_.batchSize = static_cast<uint32_t>(inputAStorageShape[0]);
        }
    } else {
        if (computeTypeValue == EINSUM_MODE) {
            matMulInfo_.batchSize = static_cast<uint32_t>(inputAStorageShape[1]);
        } else {
            if (inputADimNums == DIM_2) { // 2: [M, K]
                matMulInfo_.batchSize = 1;
            } else if (inputADimNums == DIM_3) { // 3: [B, M, K]
                matMulInfo_.batchSize = static_cast<uint32_t>(inputAStorageShape[0]);
            }
        }
    }
    size_t index0 = 0;
    size_t index1 = 1;
    size_t index2 = 2;
    if (numAttrs <= MAX_ATTRS_NUM) {
        auto inputAOriginShape = context_->GetInputShape(indexA)->GetOriginShape();
        auto outputCOriginShape = context_->GetOutputShape(idxC)->GetOriginShape();
        matMulInfo_.m = GetATensorM(inputAOriginShape, matMulInfo_.transA);
        matMulInfo_.k = GetATensorK(inputAOriginShape, matMulInfo_.transA);
        matMulInfo_.n = GetCTensorN(outputCOriginShape);
    } else if (matMulInfo_.formatC == ge::Format::FORMAT_ND && computeTypeValue != EINSUM_MODE) {
        matMulInfo_.m = GetATensorM(inputAStorageShape, matMulInfo_.transA);
        matMulInfo_.k = GetATensorK(inputAStorageShape, matMulInfo_.transA);
        matMulInfo_.n = GetCTensorN(outputCStorageShape);
    } else {
        auto oriShape = static_cast<const int64_t*>(attrs->GetAttrPointer<gert::ContinuousVector>(2)->GetData());
        matMulInfo_.m = oriShape[index0];
        matMulInfo_.k = oriShape[index1];
        matMulInfo_.n = oriShape[index2]; 
    }
    return true;
}

void PpMatMulDefault::GetHardwareInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        // OP_LOGE("[PpMatMul]", "platformInfo is nullptr");
        return;
    }
    HardwareInfo hardwareInfo;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", hardwareInfo.socVersionStr);

    hardwareInfo.coreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAic());
    hardwareInfo.socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, hardwareInfo.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, hardwareInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, hardwareInfo.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, hardwareInfo.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, hardwareInfo.l0cSize);
    hardwareInfo_ = hardwareInfo;
}

bool PpMatMulDefault::GetTilingKey() {
    std::unordered_map<platform_ascendc::SocVersion, uint32_t> archTypeMap = {
        {platform_ascendc::SocVersion::ASCEND310P, 0},
        {platform_ascendc::SocVersion::ASCEND910, 0},
        {platform_ascendc::SocVersion::ASCEND310B, 0},
        {platform_ascendc::SocVersion::ASCEND910B, 1}
    };

    std::unordered_map<ge::Format, uint32_t> formatMap = {
        {ge::Format::FORMAT_ND, 0}, 
        {ge::Format::FORMAT_FRACTAL_NZ, 1}};
    std::unordered_map<ge::DataType, uint32_t> dtypeMap = {{ge::DT_INT8, 0}, {ge::DT_FLOAT16, 1},
                                                           {ge::DT_BF16, 2}, {ge::DT_FLOAT, 3}};

    // OP_TILING_CHECK(formatMap.find(matMulInfo_.formatB) == formatMap.end(),
    //                 CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "unsupported format of matrix B."),
    //                 return false);
    // OP_TILING_CHECK(formatMap.find(matMulInfo_.formatA) == formatMap.end(),
    //                 CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "unsupported format of matrix A."),
    //                 return false);
    // OP_TILING_CHECK(archTypeMap.find(hardwareInfo_.socVersion) == archTypeMap.end(),
    //                 CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "unsupported platform."),
    //                 return false);
    size_t attrsNum = context_->GetAttrs()->GetAttrNum();
    auto computeTypeValue = 0;
    if (attrsNum <= MAX_ATTRS_NUM) {
        computeTypeValue = *context_->GetAttrs()->GetAttrPointer<int>(COMPUTE_TYPE_IDX_CANN);
    } else {
        computeTypeValue = *context_->GetAttrs()->GetAttrPointer<int>(COMPUTE_TYPE_IDX_ATB);
    }
    uint32_t tilingKey = archTypeMap[hardwareInfo_.socVersion];
    tilingKey = (tilingKey << DTYPE_BIT_COUNT) + dtypeMap[matMulInfo_.dtypeA];
    tilingKey = (tilingKey << DTYPE_BIT_COUNT) + dtypeMap[matMulInfo_.dtypeB];
    tilingKey = (tilingKey << DTYPE_BIT_COUNT) + dtypeMap[matMulInfo_.dtypeC];
    tilingKey = (tilingKey << FORMAT_BIT_COUNT) + formatMap[matMulInfo_.formatA];
    tilingKey = (tilingKey << FORMAT_BIT_COUNT) + formatMap[matMulInfo_.formatB];
    tilingKey = (tilingKey << FORMAT_BIT_COUNT) + formatMap[matMulInfo_.formatC];
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(matMulInfo_.transA);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(matMulInfo_.transB);
    tilingKey = (tilingKey << COMPUTE_TYPE_BIT_COUNT) + static_cast<uint32_t>(computeTypeValue);
    ppMatmulDefaultTilingData_.tilingKey = tilingKey;
    // OP_LOGD(context_->GetNodeName(), "tilingKey: %d.", tilingKey);
    return true;
}

bool PpMatMulDefault::GetMatMulTilingData()
{
    ppMatmulDefaultTilingData_.SetBaseShape(matMulInfo_.batchSize, matMulInfo_.m, matMulInfo_.k, matMulInfo_.n);
    OpShape opShape = ppMatmulDefaultTilingData_.opShape;
    if (opShape.m < opShape.n) {
        TilingFunc<false, OpShape, PpMatmulDefaultTilingData, HardwareInfo, MatMulInfo>(opShape, ppMatmulDefaultTilingData_, hardwareInfo_, matMulInfo_);
    } else {
        TilingFunc<true, OpShape, PpMatmulDefaultTilingData, HardwareInfo, MatMulInfo>(opShape, ppMatmulDefaultTilingData_, hardwareInfo_, matMulInfo_);
    }
    Swizzl<PpMatmulDefaultTilingData>(ppMatmulDefaultTilingData_);
    ppMatmulDefaultTilingData_.End(matMulInfo_, hardwareInfo_.socVersion == platform_ascendc::SocVersion::ASCEND310P);
    return true;
}

void PpMatMulDefault::DoTiling() {
    std::map<ge::DataType, float> dTypeMap = {{ge::DT_INT8, 1.0}, 
                                              {ge::DT_FLOAT16, 2.0},
                                              {ge::DT_BF16, 2.0},
                                              {ge::DT_FLOAT, 4.0}};
    auto inputDType = context_->GetInputDesc(0)->GetDataType();
    matMulInfo_.inDtype = dTypeMap[inputDType];
    GetHardwareInfo();
    GetMatMulInfo();
    GetTilingKey();
    GetMatMulTilingData();
    auto attrs = context_->GetAttrs();
    size_t enShuffleKIdx = DIM_2;
    if (attrs->GetAttrNum() > MAX_ATTRS_NUM) {
        enShuffleKIdx = EN_SHUFFFLE_IDX_ATB;
    }
    ppMatmulDefaultTilingData_.enShuffleK = *attrs->GetAttrPointer<bool>(enShuffleKIdx);
    PrintTiling();
}

void PpMatMulDefault::PrintTiling() {
    // OP_LOGD(context_->GetNodeName(), "PpMatMul batchSize: %ld.", ppMatmulDefaultTilingData_.opShape.batchSize);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul m: %ld.", ppMatmulDefaultTilingData_.opShape.m);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul k: %ld.", ppMatmulDefaultTilingData_.opShape.k);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul n: %ld.", ppMatmulDefaultTilingData_.opShape.n);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul m0: %ld.", ppMatmulDefaultTilingData_.opShape.m0);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul k0: %ld.", ppMatmulDefaultTilingData_.opShape.k0);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul n0: %ld.", ppMatmulDefaultTilingData_.opShape.n0);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul mLoop: %ld.", ppMatmulDefaultTilingData_.mLoop);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul kLoop: %ld.", ppMatmulDefaultTilingData_.kLoop);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul nLoop: %ld.", ppMatmulDefaultTilingData_.nLoop);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul coreLoop: %ld.", ppMatmulDefaultTilingData_.coreLoop);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul swizzlCount: %ld.", ppMatmulDefaultTilingData_.swizzlCount);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul tilingKey: %d.", ppMatmulDefaultTilingData_.tilingKey);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul blockDim: %ld.", ppMatmulDefaultTilingData_.blockDim);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul swizzlDirect: %ld.", ppMatmulDefaultTilingData_.swizzlDirect);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul splitk: %ld.", ppMatmulDefaultTilingData_.splitk);
    // OP_LOGD(context_->GetNodeName(), "PpMatMul enShuffleK: %ld.", ppMatmulDefaultTilingData_.enShuffleK);
}

ge::graphStatus PpMatMulDefault::PostTiling() {
    context_->SetTilingKey(ppMatmulDefaultTilingData_.tilingKey);
    context_->SetBlockDim(ppMatmulDefaultTilingData_.blockDim);

    PpMatmulTilingData tilingData;
    tilingData.set_batch(ppMatmulDefaultTilingData_.opShape.batchSize);
    tilingData.set_m(ppMatmulDefaultTilingData_.opShape.m);
    tilingData.set_k(ppMatmulDefaultTilingData_.opShape.k);
    tilingData.set_n(ppMatmulDefaultTilingData_.opShape.n);
    tilingData.set_m0(ppMatmulDefaultTilingData_.opShape.m0);
    tilingData.set_k0(ppMatmulDefaultTilingData_.opShape.k0);
    tilingData.set_n0(ppMatmulDefaultTilingData_.opShape.n0);
    tilingData.set_mLoop(ppMatmulDefaultTilingData_.mLoop);
    tilingData.set_kLoop(ppMatmulDefaultTilingData_.kLoop);
    tilingData.set_nLoop(ppMatmulDefaultTilingData_.nLoop);
    tilingData.set_coreLoop(ppMatmulDefaultTilingData_.coreLoop);
    tilingData.set_swizzlCount(ppMatmulDefaultTilingData_.swizzlCount);
    tilingData.set_tilingKey(ppMatmulDefaultTilingData_.tilingKey);
    tilingData.set_blockDim(ppMatmulDefaultTilingData_.blockDim);
    tilingData.set_swizzlDirect(ppMatmulDefaultTilingData_.swizzlDirect);
    tilingData.set_splitk(ppMatmulDefaultTilingData_.splitk);
    tilingData.set_enShuffleK(ppMatmulDefaultTilingData_.enShuffleK);
    
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace pp_mat_mul
} // namespace optiling