#include "../iterator.h"
/////////////////////////////////////////////////////
// l0c_to_l1
/////////////////////////////////////////////////////

// Partial specialization ZN, half, int32_t
template <ArchType ArchTag>
struct l0c_to_l1<ArchTag, DataFormat::ZN, half, int32_t> {
    using ElementOut = half;
    using ElementIn = int32_t;
    __aicore__ l0c_to_l1(AscendC::LocalTensor<ElementOut> l1Tensor,
                         AscendC::LocalTensor<ElementIn> l0cTensor,
                         AscendC::LocalTensor<uint64_t> deqTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t mTileCeil,
                         uint32_t nActual)
    {
        constexpr uint32_t BLOCK_NUM = 16;
        constexpr uint32_t BLOCK_SIZE = 32;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // A5 架构：使用 FixpipeParamsArch3510
        // VDEQF16 是 Vector Quant 模式，需要传入 cbufWorkspace (deqTensor)
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE),
            0,
            mTileCeil - static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE) *
                            sizeof(ElementOut) / sizeof(ElementIn));
        intriParams.quantPre = QuantMode_t::VDEQF16;
        AscendC::Fixpipe<half, int32_t, AscendC::CFG_NZ>(l1Tensor, l0cTensor, deqTensor, intriParams);
#else
        // 其他架构：保持原代码
        AscendC::FixpipeParams<ElementIn> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE),
            0,
            mTileCeil - static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE) *
                            sizeof(ElementOut) / sizeof(ElementIn));
        intriParams.nz2ndParams = {false, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::VDEQF16};
        AscendC::Fixpipe(l1Tensor, l0cTensor, deqTensor, intriParams);
#endif
    };
};