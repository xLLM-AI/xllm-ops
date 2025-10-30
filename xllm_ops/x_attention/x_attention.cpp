#include "kernel_operator.h"
#include "x_attention_catlass_helper.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void x_attention(GM_ADDR query, GM_ADDR shared_key_block, GM_ADDR shared_value_block, 
                        GM_ADDR unshared_key_block, GM_ADDR unshared_value_block, GM_ADDR shared_block_table, 
                        GM_ADDR shared_kv_lens, GM_ADDR decode_step, GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling) {
    // workspace use; [s,p,oTemp,oUpdate,shared_workspace,unshared_workspace]
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);

    GM_ADDR s = workspace;
    GM_ADDR p = workspace + tiling_data.mm1OutSize;
    GM_ADDR oTemp = p + tiling_data.smOnlineOutSize;
    GM_ADDR oUpdate = oTemp + tiling_data.mm2OutSize;
    GM_ADDR shared_workspace = oUpdate + tiling_data.updateSize;
    GM_ADDR unshared_workspace = shared_workspace + tiling_data.sharedWorkspaceSize;
    int64_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

    XAttnKernelParams params{query, shared_key_block, shared_value_block, unshared_key_block, unshared_value_block, 
     shared_block_table, shared_kv_lens, decode_step, s, p, oTemp, oUpdate, shared_workspace, 
         unshared_workspace, attn_out, tiling};
    if (coreIdx < tiling_data.sharedCoreNum) {
        CallSharedInferKernel(params, &tiling_data);
    } else {
        CallUnsharedInferKernel(params, &tiling_data);
    }
    AscendC::SyncAll();
    CallCombineScale(params, &tiling_data);
}
