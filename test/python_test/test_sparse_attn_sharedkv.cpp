#include <climits>
#include <stdio.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include "acl/acl.h"
#include "aclnn_sparse_attn_sharedkv.h"
#include "aclnn_sparse_attn_sharedkv_metadata.h"

// ---- Case (SWA single-path, consistent with sparse_attn_sharedkv_gen.py) ----
static const int64_t B = 1;
static const int64_t S1 = 4;
static const int64_t S2 = 16;
static const int64_t N1 = 64;       // qHead hard constraint =64
static const int64_t KV_N = 1;      // kvHead hard constraint =1
static const int64_t D = 512;       // headDim hard constraint =512
static const int64_t BLOCK_SIZE = 16;
static const int64_t BLOCK_NUM = 1;   // B*ceil(S2/BLOCK_SIZE)=1
static const int64_t MAX_BLOCKS = 1;
static const int64_t ORI_MASK_MODE = 4;
static const int64_t CMP_MASK_MODE = 3;
static const int64_t ORI_WIN_LEFT = 127;
static const int64_t ORI_WIN_RIGHT = 0;
static const int64_t CMP_RATIO = 1;
static const int64_t ORI_KV_STRIDE = 0;
static const int64_t CMP_KV_STRIDE = 0;
static const int64_t ORI_TOP_K = 512;
static const int64_t CMP_TOP_K = 512;
static const double SOFTMAX_SCALE = 1.0 / 22.627416997969522;  // 1/sqrt(512)
static std::string layoutQ = "BSND";
static std::string layoutKv = "PA_ND";
static const int64_t SAS_META_SIZE = 1024;
static const char* DATA_DIR = "/tmp/sparse_attn_sharedkv_data/";

// dtype: 0=fp16, 1=bf16 (determined by command line argv[1]; default fp16)
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
  size_t elemSz = 2;  // fp16/bf16 are both 2 bytes

  int32_t deviceId = 0; aclrtStream stream;
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) { printf("aclInit %d\n", ret); return -1; }
  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) { printf("aclrtSetDevice %d\n", ret); return -1; }
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) { printf("aclrtCreateStream %d\n", ret); return -1; }

  auto qBuf = ReadBin("q.bin");
  auto kvBuf = ReadBin("kv_pa.bin");
  auto sinksBuf = ReadBin("sinks.bin");
  auto btBuf = ReadBin("block_table.bin");
  auto susedBuf = ReadBin("seqused_kv.bin");
  auto goldenBuf = ReadBin("golden_out.bin");
  if (qBuf.empty() || kvBuf.empty() || goldenBuf.empty() || btBuf.empty() || susedBuf.empty()) {
    printf("read bin FAILED\n"); return -1;
  }

  // ================= Step 1: metadata op (25 params) =================
  // wrapper unconditionally applies Contiguous to optional inputs, nullptr will crash; pass non-null cu_seqlens/seqused tensors
  // cu_seqlens prefix sum: length B+1, [0, S1](q), [0, S2](kv); seqused: length B
  std::vector<int32_t> cuSeqQHost(B + 1, 0), cuSeqOriKvHost(B + 1, 0), cuSeqCmpKvHost(B + 1, 0);
  for (int64_t i = 0; i < B; ++i) { cuSeqQHost[i + 1] = cuSeqQHost[i] + (int32_t)S1;
    cuSeqOriKvHost[i + 1] = cuSeqOriKvHost[i] + (int32_t)S2;
    cuSeqCmpKvHost[i + 1] = cuSeqCmpKvHost[i] + (int32_t)S2; }
  std::vector<int32_t> susedQHost(B, (int32_t)S1);
  aclTensor *cuSeqQT_m, *cuSeqOriKvT_m, *cuSeqCmpKvT_m, *susedQT_m, *susedKvT_m, *metaT;
  void *cuSeqQDev_m, *cuSeqOriKvDev_m, *cuSeqCmpKvDev_m, *susedQDev_m, *susedKvDev_m, *metaDev;
  std::tie(cuSeqQT_m, cuSeqQDev_m) = CreateTensor((B + 1) * sizeof(int32_t), {B + 1}, ACL_INT32, cuSeqQHost.data());
  std::tie(cuSeqOriKvT_m, cuSeqOriKvDev_m) = CreateTensor((B + 1) * sizeof(int32_t), {B + 1}, ACL_INT32, cuSeqOriKvHost.data());
  std::tie(cuSeqCmpKvT_m, cuSeqCmpKvDev_m) = CreateTensor((B + 1) * sizeof(int32_t), {B + 1}, ACL_INT32, cuSeqCmpKvHost.data());
  std::tie(susedQT_m, susedQDev_m) = CreateTensor(B * sizeof(int32_t), {B}, ACL_INT32, susedQHost.data());
  std::tie(susedKvT_m, susedKvDev_m) = CreateTensor(susedBuf.size(), {B}, ACL_INT32, susedBuf.data());
  std::tie(metaT, metaDev) = CreateTensor(SAS_META_SIZE * sizeof(int32_t), {SAS_META_SIZE}, ACL_INT32);

  aclOpExecutor* mExec = nullptr; uint64_t mWs = 0; void* mWsPtr = nullptr;
  ret = aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
      cuSeqQT_m, cuSeqOriKvT_m, cuSeqCmpKvT_m, susedQT_m, susedKvT_m,
      N1, KV_N, D, B, S1, S2,
      ORI_TOP_K, CMP_TOP_K, CMP_RATIO,
      ORI_MASK_MODE, CMP_MASK_MODE, ORI_WIN_LEFT, ORI_WIN_RIGHT,
      &layoutQ[0], &layoutKv[0], true, false,
      metaT, &mWs, &mExec);
  if (ret != ACL_SUCCESS) { printf("Metadata GetWorkspaceSize FAILED ret=%d\n", ret); return -1; }
  if (mWs > 0) { ret = aclrtMalloc(&mWsPtr, mWs, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { printf("malloc mWs %d\n", ret); return -1; } }
  ret = aclnnSparseAttnSharedkvMetadata(mWsPtr, mWs, mExec, stream);
  if (ret != ACL_SUCCESS) { printf("Metadata Execute FAILED ret=%d\n", ret); return -1; }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) { printf("Metadata Sync %d\n", ret); return -1; }
  printf("Metadata OK\n");

  // ================= Step 2: main op (27 params) =================
  aclTensor *query, *oriKv, *oriBt, *susedKvT, *sinksT, *attnOut, *lseOut;
  void *qDev, *kvDev, *btDev, *susedKvDev, *sinksDev, *outDev, *lseDev;
  std::tie(query, qDev) = CreateTensor(qBuf.size(), {B, S1, N1, D}, g_dtype, qBuf.data());
  // paged ori_kv: [blockNum, blockSize, kvHead=1, D]
  std::tie(oriKv, kvDev) = CreateTensor(kvBuf.size(), {BLOCK_NUM, BLOCK_SIZE, KV_N, D}, g_dtype, kvBuf.data());
  std::tie(oriBt, btDev) = CreateTensor(btBuf.size(), {B, MAX_BLOCKS}, ACL_INT32, btBuf.data());
  std::tie(susedKvT, susedKvDev) = CreateTensor(susedBuf.size(), {B}, ACL_INT32, susedBuf.data());
  std::tie(sinksT, sinksDev) = CreateTensor(sinksBuf.size(), {N1}, ACL_FLOAT, sinksBuf.data());

  int64_t outElems = B * S1 * N1 * D;
  std::tie(attnOut, outDev) = CreateTensor(outElems * elemSz, {B, S1, N1, D}, g_dtype);
  int64_t lseElems = B * S1 * N1;
  std::tie(lseOut, lseDev) = CreateTensor(lseElems * sizeof(float), {B, S1, N1}, ACL_FLOAT);

  aclOpExecutor* executor = nullptr; uint64_t wsSize = 0; void* ws = nullptr;
  // 27 params: q, oriKv, cmpKv, oriSparseIdx, cmpSparseIdx, oriBt, cmpBt,
  //         cuSeqQ, cuSeqOriKv, cuSeqCmpKv, sequsedQ, sequsedKv, sinks, metadata,
  //         softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriKvStride, cmpKvStride,
  //         oriWinLeft, oriWinRight, layoutQ, layoutKv, returnSoftmaxLse, attnOut, lseOut
  ret = aclnnSparseAttnSharedkvGetWorkspaceSize(
      query, oriKv, nullptr, nullptr, nullptr, oriBt, nullptr,
      nullptr, nullptr, nullptr, nullptr, susedKvT, sinksT, metaT,
      SOFTMAX_SCALE, CMP_RATIO, ORI_MASK_MODE, CMP_MASK_MODE, ORI_KV_STRIDE, CMP_KV_STRIDE,
      ORI_WIN_LEFT, ORI_WIN_RIGHT, &layoutQ[0], &layoutKv[0], false,
      attnOut, lseOut, &wsSize, &executor);
  if (ret != ACL_SUCCESS) { printf("Main GetWorkspaceSize FAILED ret=%d\n", ret); return -1; }
  printf("Main GetWorkspaceSize OK, ws=%lu\n", wsSize);
  if (wsSize > 0) { ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { printf("malloc ws %d\n", ret); return -1; } }
  ret = aclnnSparseAttnSharedkv(ws, wsSize, executor, stream);
  if (ret != ACL_SUCCESS) { printf("Main Execute FAILED ret=%d\n", ret); return -1; }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) { printf("Main Sync %d\n", ret); return -1; }

  std::vector<uint16_t> outHost(outElems);
  ret = aclrtMemcpy(outHost.data(), outElems * elemSz, outDev, outElems * elemSz, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) { printf("memcpy out %d\n", ret); return -1; }

  const uint16_t* golden = reinterpret_cast<const uint16_t*>(goldenBuf.data());
  double maxAbs = 0.0, maxRel = 0.0;
  int failCnt = 0;
  double atol = 2e-2, rtol = 2e-2;
  for (int64_t i = 0; i < outElems; ++i) {
    float g = ToF32(golden[i], isBf16);
    float o = ToF32(outHost[i], isBf16);
    double ad = std::fabs(g - o);
    double rd = ad / (std::fabs(g) + 1e-6);
    if (ad > maxAbs) maxAbs = ad;
    if (rd > maxRel) maxRel = rd;
    if (ad > atol && rd > rtol) failCnt++;
  }
  printf("==== outElems=%ld maxAbs=%.5f maxRel=%.5f fail=%d ====\n", (long)outElems, maxAbs, maxRel, failCnt);
  printf(failCnt == 0 ? "RESULT: PASS\n" : "RESULT: FAIL\n");

  aclDestroyTensor(query); aclDestroyTensor(oriKv); aclDestroyTensor(oriBt);
  aclDestroyTensor(susedKvT); aclDestroyTensor(sinksT); aclDestroyTensor(attnOut); aclDestroyTensor(lseOut);
  aclDestroyTensor(susedKvT_m); aclDestroyTensor(metaT);
  aclDestroyTensor(cuSeqQT_m); aclDestroyTensor(cuSeqOriKvT_m);
  aclDestroyTensor(cuSeqCmpKvT_m); aclDestroyTensor(susedQT_m);
  aclrtFree(qDev); aclrtFree(kvDev); aclrtFree(btDev); aclrtFree(susedKvDev);
  aclrtFree(sinksDev); aclrtFree(outDev); aclrtFree(lseDev);
  aclrtFree(susedKvDev_m); aclrtFree(metaDev);
  aclrtFree(cuSeqQDev_m); aclrtFree(cuSeqOriKvDev_m);
  if (mWsPtr) aclrtFree(mWsPtr);
  if (ws) aclrtFree(ws);
  aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize();
  return 0;
}