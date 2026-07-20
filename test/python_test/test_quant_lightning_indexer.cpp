#include <climits>
#include <stdio.h>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include "acl/acl.h"
#include "aclnn_quant_lightning_indexer.h"
#include "aclnn_quant_lightning_indexer_metadata.h"

// ---- Case (PA_BSND, consistent with quant_lightning_indexer_gen.py) ----
static const int64_t batchSize = 1;
static const int64_t qSeq = 4;
static const int64_t kSeq = 128;      // = block_size (1 block)
static const int64_t blockSize = 128;
static const int64_t blockNum = 1;
static const int64_t kMaxBlockPerBatch = 1;
static const int64_t numHeadsQ = 64;
static const int64_t numHeadsK = 1;
static const int64_t headDim = 128;
static const int64_t queryQuantMode = 0;
static const int64_t keyQuantMode = 0;
static const int64_t sparseCount = 8;
static const int64_t sparseMode = 3;
static const int64_t preToken = INT64_MAX;
static const int64_t nextToken = INT64_MAX;
static const int64_t cmpRatio = 1;
static const int64_t stride = 1;
static const int64_t scaleStride = 1;
static const bool returnValues = false;
static std::string layoutQuery = "BSND";
static std::string layoutKey = "PA_BSND";
static const int64_t QLI_META_SIZE = 1024;
static const char* DATA_DIR = "/tmp/quant_lightning_indexer_data/";

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

int main() {
  int32_t deviceId = 0; aclrtStream stream;
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) { printf("aclInit %d\n", ret); return -1; }
  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) { printf("aclrtSetDevice %d\n", ret); return -1; }
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) { printf("aclrtCreateStream %d\n", ret); return -1; }

  auto qBuf = ReadBin("query.bin");
  auto kBuf = ReadBin("key.bin");
  auto wBuf = ReadBin("weights.bin");
  auto qsBuf = ReadBin("q_scale.bin");
  auto ksBuf = ReadBin("k_scale.bin");
  auto aslqBuf = ReadBin("aslq.bin");
  auto aslkBuf = ReadBin("aslk.bin");
  auto btBuf = ReadBin("block_table.bin");
  auto goldenBuf = ReadBin("golden_indices.bin");
  if (qBuf.empty() || kBuf.empty() || goldenBuf.empty() || btBuf.empty()) { printf("read bin FAILED\n"); return -1; }

  // ================= Step 1: metadata op =================
  aclTensor *aslqT, *aslkT, *metaT;
  void *aslqDev, *aslkDev, *metaDev;
  std::tie(aslqT, aslqDev) = CreateTensor(aslqBuf.size(), {batchSize}, ACL_INT32, aslqBuf.data());
  std::tie(aslkT, aslkDev) = CreateTensor(aslkBuf.size(), {batchSize}, ACL_INT32, aslkBuf.data());
  std::tie(metaT, metaDev) = CreateTensor(QLI_META_SIZE * sizeof(int32_t), {QLI_META_SIZE}, ACL_INT32);

  int64_t maxSeqQ = qSeq, maxSeqK = kSeq * cmpRatio;
  aclOpExecutor* mExec = nullptr; uint64_t mWs = 0; void* mWsPtr = nullptr;
  ret = aclnnQuantLightningIndexerMetadataGetWorkspaceSize(
      aslqT, aslkT, numHeadsQ, numHeadsK, headDim, queryQuantMode, keyQuantMode,
      batchSize, maxSeqQ, maxSeqK, &layoutQuery[0], &layoutKey[0],
      sparseCount, sparseMode, preToken, nextToken, cmpRatio,
      metaT, &mWs, &mExec);
  if (ret != ACL_SUCCESS) { printf("Metadata GetWorkspaceSize FAILED ret=%d\n", ret); return -1; }
  if (mWs > 0) { ret = aclrtMalloc(&mWsPtr, mWs, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { printf("malloc mWs %d\n", ret); return -1; } }
  ret = aclnnQuantLightningIndexerMetadata(mWsPtr, mWs, mExec, stream);
  if (ret != ACL_SUCCESS) { printf("Metadata Execute FAILED ret=%d\n", ret); return -1; }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) { printf("Metadata Sync %d\n", ret); return -1; }
  printf("Metadata OK\n");

  // ================= Step 2: main op =================
  aclTensor *query, *key, *weights, *qScale, *kScale, *btT, *idxOut, *valOut;
  void *qDev, *kDev, *wDev, *qsDev, *ksDev, *btDev, *idxDev, *valDev;
  std::tie(query, qDev) = CreateTensor(qBuf.size(), {batchSize, qSeq, numHeadsQ, headDim}, ACL_INT8, qBuf.data());
  // paged key: [blockNum, blockSize, kH, hd]
  std::tie(key, kDev) = CreateTensor(kBuf.size(), {blockNum, blockSize, numHeadsK, headDim}, ACL_INT8, kBuf.data());
  std::tie(weights, wDev) = CreateTensor(wBuf.size(), {batchSize, qSeq, numHeadsQ}, ACL_FLOAT16, wBuf.data());
  std::tie(qScale, qsDev) = CreateTensor(qsBuf.size(), {batchSize, qSeq, numHeadsQ}, ACL_FLOAT16, qsBuf.data());
  // paged k_scale: [blockNum, blockSize, kH]
  std::tie(kScale, ksDev) = CreateTensor(ksBuf.size(), {blockNum, blockSize, numHeadsK}, ACL_FLOAT16, ksBuf.data());
  std::tie(btT, btDev) = CreateTensor(btBuf.size(), {batchSize, kMaxBlockPerBatch}, ACL_INT32, btBuf.data());

  int64_t idxCount = batchSize * qSeq * numHeadsK * sparseCount;
  std::tie(idxOut, idxDev) = CreateTensor(idxCount * sizeof(int32_t),
      {batchSize, qSeq, numHeadsK, sparseCount}, ACL_INT32);
  std::tie(valOut, valDev) = CreateTensor(idxCount * sizeof(float),
      {batchSize, qSeq, numHeadsK, sparseCount}, ACL_FLOAT);

  aclOpExecutor* executor = nullptr; uint64_t wsSize = 0; void* ws = nullptr;
  ret = aclnnQuantLightningIndexerGetWorkspaceSize(
      query, key, weights, qScale, kScale,
      aslqT, aslkT, btT, metaT,
      queryQuantMode, keyQuantMode, &layoutQuery[0], &layoutKey[0],
      sparseCount, sparseMode, preToken, nextToken, cmpRatio,
      returnValues, stride, scaleStride,
      idxOut, valOut, &wsSize, &executor);
  if (ret != ACL_SUCCESS) { printf("Main GetWorkspaceSize FAILED ret=%d\n", ret); return -1; }
  printf("Main GetWorkspaceSize OK, ws=%lu\n", wsSize);
  if (wsSize > 0) { ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { printf("malloc ws %d\n", ret); return -1; } }
  ret = aclnnQuantLightningIndexer(ws, wsSize, executor, stream);
  if (ret != ACL_SUCCESS) { printf("Main Execute FAILED ret=%d\n", ret); return -1; }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) { printf("Main Sync %d\n", ret); return -1; }

  std::vector<int32_t> idxHost(idxCount);
  ret = aclrtMemcpy(idxHost.data(), idxCount * sizeof(int32_t), idxDev,
      idxCount * sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) { printf("memcpy out %d\n", ret); return -1; }

  const int32_t* golden = reinterpret_cast<const int32_t*>(goldenBuf.data());
  int total = 0, setMatch = 0;
  for (int s = 0; s < qSeq; ++s) {
    const int32_t* npuRow = &idxHost[s * sparseCount];
    const int32_t* gRow = &golden[s * sparseCount];
    printf("s=%d NPU   =", s); for (int j = 0; j < sparseCount; ++j) printf(" %d", npuRow[j]); printf("\n");
    printf("s=%d golden=", s); for (int j = 0; j < sparseCount; ++j) printf(" %d", gRow[j]); printf("\n");
    for (int j = 0; j < sparseCount; ++j) {
      total++;
      for (int k = 0; k < sparseCount; ++k) if (npuRow[j] == gRow[k]) { setMatch++; break; }
    }
  }
  printf("==== set-match %d/%d ====\n", setMatch, total);
  printf(setMatch == total ? "RESULT: PASS\n" : "RESULT: FAIL\n");

  aclDestroyTensor(query); aclDestroyTensor(key); aclDestroyTensor(weights);
  aclDestroyTensor(qScale); aclDestroyTensor(kScale); aclDestroyTensor(btT);
  aclDestroyTensor(idxOut); aclDestroyTensor(valOut);
  aclDestroyTensor(aslqT); aclDestroyTensor(aslkT); aclDestroyTensor(metaT);
  aclrtFree(qDev); aclrtFree(kDev); aclrtFree(wDev); aclrtFree(qsDev); aclrtFree(ksDev);
  aclrtFree(btDev); aclrtFree(idxDev); aclrtFree(valDev);
  aclrtFree(aslqDev); aclrtFree(aslkDev); aclrtFree(metaDev);
  if (mWsPtr) aclrtFree(mWsPtr);
  if (ws) aclrtFree(ws);
  aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize();
  return 0;
}