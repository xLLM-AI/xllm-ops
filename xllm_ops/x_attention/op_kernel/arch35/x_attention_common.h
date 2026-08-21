
#ifndef X_ATTENTION_COMMON
#define X_ATTENTION_COMMON


constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t CV_RATIO = 2;

namespace SharedInfer {
    constexpr uint16_t SYNC_QK_READY_FLAG[2] = {0, 1};
    constexpr uint16_t SYNC_SOFTMAX_READY_FLAG[3] = {2, 3, 4};
    constexpr uint16_t SYNC_PV_READY_FLAG[2] = {5, 6};
    constexpr uint16_t QK_UB_RELEASE_FLAG[2] = {7, 8};
    constexpr uint16_t PV_UB_RELEASE_FLAG[2] = {9, 10};
    constexpr uint32_t COMPUTE_PIPE_NUM = 3;
    struct TaskArgs {
        int32_t taskId = 0;
        int32_t batchId = 0;
        int32_t qHeadId = 0;
        int32_t kvHeadId = 0;
        int32_t qBlockId = 0;
        int32_t kvBlockId = 0;
        int32_t actualKvLen = 0;
        int32_t blockQLen = 0;
        int32_t blockKvLen = 0;
        int32_t qCoord = 0;
        int32_t kvCoord = 0;
        int32_t qNCoord = 0;
        int32_t kvNCoord = 0;
        bool isFirstKv = false;
        bool isUpdate = false;
        bool isLastKv = false;
        int32_t taskIdMod2 = 0;
        int32_t taskIdMod3 = 0;
        int32_t kvBatchOffset = 0;
        int32_t halfBlockQLen = 0;
        int32_t halfBlockQOffset = 0;
        int32_t maxOutOffset = 0;
    };
}

namespace UnSharedInfer {
    constexpr uint16_t SYNC_QK_READY_FLAG[2] = {0, 1};
    constexpr uint16_t SYNC_SOFTMAX_READY_FLAG[3] = {2, 3, 4};
    constexpr uint16_t QK_UB_RELEASE_FLAG[2] = {5, 6};
    constexpr uint32_t COMPUTE_PIPE_NUM = 3;
    struct TaskArgs {
        int32_t taskId;
        int32_t batchId;
        int32_t cacheBlockId;
        int32_t groupCountBlockId;
        int32_t qCoord;
        int32_t kvCoord;
        int32_t taskIdMod2;
        int32_t taskIdMod3;
        int32_t maxOutOffset = 0;
    };
}

struct XAttnKernelCommonParams {
    GM_ADDR q;
    GM_ADDR sharedK;
    GM_ADDR sharedV;
    GM_ADDR unsharedK;
    GM_ADDR unsharedV;
    GM_ADDR sharedBlockTable;
    GM_ADDR unsharedBlockTable;
    GM_ADDR sharedKvLens; // shared Kv
    GM_ADDR decodeStep;     // unshared kv: 1, 2, 3
    GM_ADDR sharedO;
    GM_ADDR sharedMax;
    GM_ADDR sharedSum;
    GM_ADDR unsharedO;
    GM_ADDR unsharedMax;
    GM_ADDR unsharedSum;
    GM_ADDR o;  // final combine out
    GM_ADDR tiling;

    CATLASS_DEVICE
    XAttnKernelCommonParams() {
    }

    CATLASS_DEVICE
    XAttnKernelCommonParams(
        GM_ADDR q_, GM_ADDR sharedK_, GM_ADDR sharedV_, GM_ADDR unsharedK_, GM_ADDR unsharedV_,
        GM_ADDR sharedBlockTable_, GM_ADDR unsharedBlockTable_, GM_ADDR sharedKvLens_, GM_ADDR decodeStep_,
        GM_ADDR sharedO_, GM_ADDR sharedMax_, GM_ADDR sharedSum_, GM_ADDR unsharedO_, GM_ADDR unsharedMax_,
        GM_ADDR unsharedSum_, GM_ADDR o_, GM_ADDR tiling_)
        : q(q_),
          sharedK(sharedK_),
          sharedV(sharedV_),
          unsharedK(unsharedK_),
          unsharedV(unsharedV_),
          sharedBlockTable(sharedBlockTable_),
          unsharedBlockTable(unsharedBlockTable_),
          sharedKvLens(sharedKvLens_),
          decodeStep(decodeStep_),
          sharedO(sharedO_),
          sharedMax(sharedMax_),
          sharedSum(sharedSum_),
          unsharedO(unsharedO_),
          unsharedMax(unsharedMax_),
          unsharedSum(unsharedSum_),
          o(o_),
          tiling(tiling_)
    {}
};

#endif
