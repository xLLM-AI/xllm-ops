#include <climits>
#include <stdio.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include "acl/acl.h"
#include "aclnn_compressor.h"

// ---- Case (non-OVERLAP {128,1,512}, consistent with compressor_gen.py) ----
static const int64_t B = 1;
static const int64_t S = 128;
static const int64_t SR = 1;              // ceil(S/CMP_RATIO)
static const int64_t HIDDEN = 1024;
static const int64_t HEAD_DIM = 512;
static const int64_t COFF_D = 512;        // coff*head_dim
static const int64_t CMP_RATIO = 128;
static const int64_t COFF = 1;
static const int64_t ROPE_HEAD_DIM = 64;
static const int64_t BLOCK_SIZE = 128;
static const int64_t BLOCK_NUM = 1;
static const double NORM_EPS = 1e-6;
static const int64_t ROTARY_MODE = 1;     // HALF
static const bool ENABLE_GRAD = false;
static const char* DATA_DIR = "/tmp/compressor_data/";

static aclDataType g_dtype = ACL_FLOAT16;

static std::vector<char> ReadBin(const std::string &name) {
  std::ifstream f(std::string(DATA_DIR) + name, std::ios::binary | std::ios::ate);
  if (!f) { printf("open %s FAILED\n", name.c_str()); return {}; }
  std::streamsize sz = f.tellg(); f.seekg(0, std::ios::beg);
  std::vector<char> buf(sz); f.read(buf.data(), sz); return buf;
}

std::tuple<aclTensor*, void*> CreateTensor(size_t size, std::vector<int64_t> shape,
    aclDataType dType, const void* hostData = nullptr) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
  void* dev = nullptr;
  auto ret = aclrtMalloc(&dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) { printf("aclrtMalloc %d\n", ret); return {nullptr, nullptr}; }
  aclTensor* t = aclCreateTensor(shape.data(), shape.size(), dType, strides.data(), 0,
      aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), dev);
  if (t == nullptr) { aclrtFree(dev); return {nullptr, nullptr}; }
  if (hostData) aclrtMemcpy(dev, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  return {t, dev};
}

// 16-bit(fp16/bf16) bit pattern -> float32
static float ToF32(uint16_t v, bool isBf16) {
  uint32_t u;
  if (isBf16) {
    u = static_cast<uint32_t>(v) << 16;
  } else {
    uint32_t sign = (v >> 15) & 0x1;
    uint32_t exp = (v >> 10) & 0x1f;
    uint32_t man = v & 0x3ff;
    if (exp == 0) {
      if (man == 0) { u = sign << 31; }
      else {
        exp = 127 - 15 + 1;
        while ((man & 0x400) == 0) { man <<= 1; exp--; }
        man &= 0x3ff;
        u = (sign << 31) | (exp << 23) | (man << 13);
      }
    } else if (exp == 0x1f) {
      u = (sign << 31) | (0xff << 23) | (man << 13);
    } else {
      u = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
  }
  float f; memcpy(&f, &u, 4); return f;
}

int main(int argc, char** argv) {
  bool isBf16 = (argc > 1 && std::string(argv[1]) == "bf16");
  g_dtype = isBf16 ? ACL_BF16 : ACL_FLOAT16;
  size_t elemSz = 2;

  int32_t deviceId = 0; aclrtStream stream;
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) { printf("aclInit %d\n", ret); return -1; }
  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) { printf("aclrtSetDevice %d\n", ret); return -1; }
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) { printf("aclrtCreateStream %d\n", ret); return -1; }

  auto xBuf = ReadBin("x.bin");
  auto wkvBuf = ReadBin("wkv.bin");
  auto wgateBuf = ReadBin("wgate.bin");
  auto apeBuf = ReadBin("ape.bin");
  auto kvStateBuf = ReadBin("kv_state.bin");
  auto scoreStateBuf = ReadBin("score_state.bin");
  auto kvBlockTableBuf = ReadBin("kv_block_table.bin");
  auto scoreBlockTableBuf = ReadBin("score_block_table.bin");
  auto normWBuf = ReadBin("norm_weight.bin");
  auto ropeCosBuf = ReadBin("rope_cos.bin");
  auto ropeSinBuf = ReadBin("rope_sin.bin");
  auto goldenBuf = ReadBin("golden_cmp_kv.bin");
  if (xBuf.empty() || wkvBuf.empty() || goldenBuf.empty() || apeBuf.empty()) {
    printf("read bin FAILED\n"); return -1;
  }

  // ---- input tensors ----
  aclTensor *x, *wkv, *wgate, *kvState, *scoreState, *ape, *normW, *ropeSin, *ropeCos;
  void *xDev, *wkvDev, *wgateDev, *kvStateDev, *scoreStateDev, *apeDev, *normWDev, *ropeSinDev, *ropeCosDev;
  std::tie(x, xDev) = CreateTensor(xBuf.size(), {B, S, HIDDEN}, g_dtype, xBuf.data());
  std::tie(wkv, wkvDev) = CreateTensor(wkvBuf.size(), {COFF_D, HIDDEN}, g_dtype, wkvBuf.data());
  std::tie(wgate, wgateDev) = CreateTensor(wgateBuf.size(), {COFF_D, HIDDEN}, g_dtype, wgateBuf.data());
  // kv_state/score_state: float32, in-place, PageAttention: [blockNum, blockSize, coffD]
  std::tie(kvState, kvStateDev) = CreateTensor(kvStateBuf.size(), {BLOCK_NUM, BLOCK_SIZE, COFF_D}, ACL_FLOAT, kvStateBuf.data());
  std::tie(scoreState, scoreStateDev) = CreateTensor(scoreStateBuf.size(), {BLOCK_NUM, BLOCK_SIZE, COFF_D}, ACL_FLOAT, scoreStateBuf.data());
  std::tie(ape, apeDev) = CreateTensor(apeBuf.size(), {CMP_RATIO, COFF_D}, ACL_FLOAT, apeBuf.data());
  std::tie(normW, normWDev) = CreateTensor(normWBuf.size(), {HEAD_DIM}, g_dtype, normWBuf.data());
  std::tie(ropeSin, ropeSinDev) = CreateTensor(ropeSinBuf.size(), {B, SR, ROPE_HEAD_DIM}, g_dtype, ropeSinBuf.data());
  std::tie(ropeCos, ropeCosDev) = CreateTensor(ropeCosBuf.size(), {B, SR, ROPE_HEAD_DIM}, g_dtype, ropeCosBuf.data());

  // kv/score block_table: int32, shape (batchSize, maxBlockNumPerBatch)
  const int64_t MAX_BLOCK = (S + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 1
  aclTensor *kvBlockTable, *scoreBlockTable;
  void *kvBlockTableDev, *scoreBlockTableDev;
  std::tie(kvBlockTable, kvBlockTableDev) = CreateTensor(kvBlockTableBuf.size(), {B, MAX_BLOCK}, ACL_INT32, kvBlockTableBuf.data());
  std::tie(scoreBlockTable, scoreBlockTableDev) = CreateTensor(scoreBlockTableBuf.size(), {B, MAX_BLOCK}, ACL_INT32, scoreBlockTableBuf.data());

  // ---- output tensors ----
  // cmp_kv: BSH -> (B, SR, HEAD_DIM); when enable_grad=false the 4 auxiliary outputs have shape=0, use {1} as placeholder
  aclTensor *cmpKv, *wkvProj, *softmaxRes, *normX, *normRstd;
  void *cmpKvDev, *wkvProjDev, *softmaxResDev, *normXDev, *normRstdDev;
  int64_t cmpKvElems = B * SR * HEAD_DIM;
  std::tie(cmpKv, cmpKvDev) = CreateTensor(cmpKvElems * elemSz, {B, SR, HEAD_DIM}, g_dtype);
  std::tie(wkvProj, wkvProjDev) = CreateTensor(elemSz, {1}, g_dtype);
  std::tie(softmaxRes, softmaxResDev) = CreateTensor(elemSz, {1}, g_dtype);
  std::tie(normX, normXDev) = CreateTensor(elemSz, {1}, g_dtype);
  std::tie(normRstd, normRstdDev) = CreateTensor(elemSz, {1}, g_dtype);

  // ---- aclnnCompressor (25 params: kv_state/score_state are in-place Ref, passed only once) ----
  aclOpExecutor* executor = nullptr; uint64_t wsSize = 0; void* ws = nullptr;
  ret = aclnnCompressorGetWorkspaceSize(
      x, wkv, wgate, kvState, scoreState, ape, normW, ropeSin, ropeCos,
      kvBlockTable, scoreBlockTable, nullptr, nullptr, nullptr,  // kvBT, scoreBT, cuSeqlens, seqused, startPos
      ROPE_HEAD_DIM, CMP_RATIO, COFF, NORM_EPS, ROTARY_MODE, ENABLE_GRAD,
      cmpKv, wkvProj, softmaxRes, normX, normRstd,
      &wsSize, &executor);
  if (ret != ACL_SUCCESS) { printf("Compressor GetWorkspaceSize FAILED ret=%d\n", ret); return -1; }
  printf("Compressor GetWorkspaceSize OK, ws=%lu\n", wsSize);
  if (wsSize > 0) { ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { printf("malloc ws %d\n", ret); return -1; } }
  ret = aclnnCompressor(ws, wsSize, executor, stream);
  if (ret != ACL_SUCCESS) { printf("Compressor Execute FAILED ret=%d\n", ret); return -1; }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) { printf("Compressor Sync %d\n", ret); return -1; }
  printf("Compressor Execute OK\n");

  // ---- read back cmp_kv and compare ----
  std::vector<uint16_t> outHost(cmpKvElems);
  ret = aclrtMemcpy(outHost.data(), cmpKvElems * elemSz, cmpKvDev, cmpKvElems * elemSz, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) { printf("memcpy out %d\n", ret); return -1; }

  const uint16_t* golden = reinterpret_cast<const uint16_t*>(goldenBuf.data());
  double maxAbs = 0.0, maxRel = 0.0;
  int failCnt = 0;
  double atol = 2e-2, rtol = 2e-2;
  for (int64_t i = 0; i < cmpKvElems; ++i) {
    float g = ToF32(golden[i], isBf16);
    float o = ToF32(outHost[i], isBf16);
    double ad = std::fabs(g - o);
    double rd = ad / (std::fabs(g) + 1e-6);
    if (ad > maxAbs) maxAbs = ad;
    if (rd > maxRel) maxRel = rd;
    if (ad > atol && rd > rtol) failCnt++;
  }
  printf("==== cmpKvElems=%ld maxAbs=%.5f maxRel=%.5f fail=%d ====\n", (long)cmpKvElems, maxAbs, maxRel, failCnt);
  printf(failCnt == 0 ? "RESULT: PASS\n" : "RESULT: FAIL\n");

  aclDestroyTensor(x); aclDestroyTensor(wkv); aclDestroyTensor(wgate);
  aclDestroyTensor(kvState); aclDestroyTensor(scoreState); aclDestroyTensor(ape);
  aclDestroyTensor(normW); aclDestroyTensor(ropeSin); aclDestroyTensor(ropeCos);
  aclDestroyTensor(cmpKv); aclDestroyTensor(wkvProj); aclDestroyTensor(softmaxRes);
  aclDestroyTensor(normX); aclDestroyTensor(normRstd);
  aclDestroyTensor(kvBlockTable); aclDestroyTensor(scoreBlockTable);
  aclrtFree(xDev); aclrtFree(wkvDev); aclrtFree(wgateDev);
  aclrtFree(kvStateDev); aclrtFree(scoreStateDev); aclrtFree(apeDev);
  aclrtFree(normWDev); aclrtFree(ropeSinDev); aclrtFree(ropeCosDev);
  aclrtFree(cmpKvDev); aclrtFree(wkvProjDev); aclrtFree(softmaxResDev);
  aclrtFree(normXDev); aclrtFree(normRstdDev);
  aclrtFree(kvBlockTableDev); aclrtFree(scoreBlockTableDev);
  if (ws) aclrtFree(ws);
  aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize();
  return 0;
}