#include "../iterator.h"

constexpr uint32_t BLOCK_NUM = 16;
constexpr uint32_t BLOCK_SIZE_INT8 = 32;

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, half, float> {
    /**
     * @brief Copy data from L0C buffer to global memory, partial specialized for
     *
     * @param gmTensor the destination tensor on global memory, which is stored in ND format.
     * @param l0cTensor the source tensor on L0C buffer, which is stored in FRACTAL_NZ format.
     * @param mTileActual the m-direction size of the matrix in L0C buffer.
     * @param nTileActual the n-direction size of the matrix in L0C buffer.
     * @param srcStride the source stride between the adjacent fractal matrics along n-direction in unit of C0_SIZE.
     * @param dstStride the leading dimension of the destination matrix in unit of element.
     */
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<half> gmTensor,
                         AscendC::LocalTensor<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // A5 architecture: use FixpipeParamsArch3510<ROW_MAJOR>, F322F16 mode does not require cbufWorkspace
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> intriParams(
            nTileActual,  // nSize
            mTileActual,  // mSize
            srcStride,    // srcStride
            dstStride);   // dstStride
        intriParams.quantPre = QuantMode_t::F322F16;
        AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#elif defined(__DAV_C220_CUBE__)
        // V220 architecture (A2/A3): use FixpipeParamsV220
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::F322F16;
        AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        // Other architectures: use FixpipeParams
        AscendC::FixpipeParams<float> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::F322F16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, half, int32_t> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<half> gmTensor,
                         AscendC::LocalTensor<int32_t> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride)
    {
        // VDEQF16 mode requires cbufWorkspace, but L0C->GM path does not support this mode
        // For A5 and V220 architectures, use scalar quantization DEQF16 instead
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> intriParams(
            nTileActual,  // nSize
            mTileActual,  // mSize
            srcStride,    // srcStride
            dstStride);   // dstStride
        // A5 L0C->GM path does not support VDEQF16 (Vector Quant), use DEQF16 (Scalar Quant) instead
        // Note: deqScalar needs to set quantization parameters
        intriParams.quantPre = QuantMode_t::DEQF16;
        // TODO: set deqScalar quantization parameters
        // intriParams.deqScalar = scaleValue;
        AscendC::Fixpipe<half, int32_t, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#elif defined(__DAV_C220_CUBE__)
        // V220 architecture (A2/A3): use FixpipeParamsV220
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::VDEQF16;
        AscendC::Fixpipe<half, int32_t, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        // Other architectures: VDEQF16 mode not supported yet
        AscendC::FixpipeParams<int32_t> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::VDEQF16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, __bf16, float> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<__bf16> gmTensor,
                         AscendC::LocalTensor<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // A5 architecture: use FixpipeParamsArch3510<ROW_MAJOR>, F322BF16 mode
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> intriParams(
            nTileActual,  // nSize
            mTileActual,  // mSize
            srcStride,    // srcStride
            dstStride);   // dstStride
        intriParams.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<__bf16, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#elif defined(__DAV_C220_CUBE__)
        // V220 architecture (A2/A3): use FixpipeParamsV220
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<__bf16, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        // Other architectures: use FixpipeParams
        AscendC::FixpipeParams<float> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::F322BF16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

// Partial specialization ND, float
template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<float> gmTensor,
                         AscendC::LocalTensor<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // A5 architecture: use FixpipeParamsArch3510<ROW_MAJOR>, NoQuant mode
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> intriParams(
            nTileActual,  // nSize
            mTileActual,  // mSize
            srcStride,    // srcStride
            dstStride);   // dstStride
        intriParams.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#elif defined(__DAV_C220_CUBE__)
        // V220 architecture (A2/A3): use FixpipeParamsV220
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        // Other architectures: use FixpipeParams
        AscendC::FixpipeParams<float> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::NoQuant};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::NZ, half, float> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<half> gmTensor,
                         AscendC::LocalTensor<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // A5 architecture: use FixpipeParamsArch3510<NZ>, NZ2NZ mode
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> intriParams(
            nTileActual,  // nSize
            mTileActual,  // mSize
            srcStride,    // srcStride
            dstStride - (nTileActual * sizeof(half) / sizeof(float)));  // dstStride unit is element
        intriParams.quantPre = QuantMode_t::F322F16;
        AscendC::Fixpipe<half, float, AscendC::CFG_NZ>(gmTensor, l0cTensor, intriParams);
#elif defined(__DAV_C220_CUBE__)
        // V220 architecture (A2/A3): use FixpipeParamsV220
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::F322F16;
        AscendC::Fixpipe<half, float, AscendC::CFG_NZ>(gmTensor, l0cTensor, intriParams);
#else
        // Other architectures: use FixpipeParams
        AscendC::FixpipeParams<float> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride - (nTileActual * sizeof(half) / sizeof(float)));
        intriParams.quantParams = {QuantMode_t::F322F16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, int32_t, int32_t> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<int32_t> gmTensor,
                         AscendC::LocalTensor<int32_t> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // A5 architecture: use FixpipeParamsArch3510<ROW_MAJOR>, NoQuant mode
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> intriParams(
            nTileActual,  // nSize
            mTileActual,  // mSize
            srcStride,    // srcStride
            dstStride);   // dstStride
        intriParams.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe<int32_t, int32_t, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#elif defined(__DAV_C220_CUBE__)
        // V220 architecture (A2/A3): use FixpipeParamsV220
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe<int32_t, int32_t, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        // Other architectures: use FixpipeParams
        AscendC::FixpipeParams<int32_t> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(int32_t) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::NoQuant};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};