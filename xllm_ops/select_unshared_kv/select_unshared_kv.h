/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"

namespace kernels {

constexpr uint32_t TOTAL_UB_SIZE = 180 * 1024;
constexpr uint32_t KEY_UB_OFFSET = 0;
constexpr uint32_t KEY_UB_LEN = 80 * 1024;
constexpr uint32_t VALUE_UB_OFFSET = 90 * 1024;
constexpr uint32_t VALUE_UB_LEN = 80 * 1024;
constexpr int8_t TASK_DIRECTION_UP = 1;   // Forward Task，src_beam_index > dst_beam_index
constexpr int8_t TASK_DIRECTION_DOWN = -1; // Reverse Task，src_beam_index < dst_beam_index

using namespace AscendC;

struct beamParams {
    int32_t b_idx{0};   // batch number
    int32_t beam_idx{0};  // beam number
    int32_t beam_token_num{0};  // Number of tokens in Beam
    int32_t beam_inner_offset;  // The offset starting position written by the current beam
    int8_t task_type;
};

template <class T1, class T2>
class SelectUnsharedKVKernel {
public:
    SelectUnsharedKVTilingData tiling_;
    AscendC::TPipe *pipe_ = nullptr;

    GlobalTensor<T1> x_key_block_gm_;
    GlobalTensor<T1> x_value_block_gm_;
    GlobalTensor<T1> select_key_block_gm_;
    GlobalTensor<T1> select_value_block_gm_;
    GlobalTensor<T2> beam_index_gm_;
    GlobalTensor<T2> group_token_num_gm_;

    TBuf<TPosition::VECCALC> ubuf_;

    uint32_t core_id;
    uint32_t used_core_num;
    int32_t global_batch_idx = 0;
    int32_t global_beam_idx;

    __aicore__ inline SelectUnsharedKVKernel(AscendC::TPipe *pipe) {pipe_ = pipe;}

    __aicore__ inline void Init(GM_ADDR beam_index, GM_ADDR x_key_block, GM_ADDR x_value_block,
                                GM_ADDR group_token_num, GM_ADDR workspace,
                                const SelectUnsharedKVTilingData *tiling,
                                GM_ADDR select_key_block, GM_ADDR select_value_block)
    {
        tiling_ = *tiling;
        core_id = GetBlockIdx();
        used_core_num = GetBlockNum();
        beam_index_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2 *>(beam_index));
        group_token_num_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2 *>(group_token_num));
        x_key_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1 *>(x_key_block));
        x_value_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1 *>(x_value_block));
        select_key_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1 *>(select_key_block));
        select_value_block_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1 *>(select_value_block));

        pipe_->InitBuffer(ubuf_, TOTAL_UB_SIZE);
    }

    __aicore__ inline bool calcValidBlock(beamParams &beam_params, int8_t task_type)
    {
        int64_t calc_block = 0;
        int32_t max_beam_token_num = 0;
        while (calc_block < used_core_num) {
            if (global_batch_idx >= tiling_.batch) {
                break;
            }
            uint32_t beam_gm_offset = global_batch_idx * tiling_.beam_size + global_beam_idx;
            // Get the current beamIndex and beamnum
            int32_t beam_token_num = global_beam_idx == 0 ? group_token_num_gm_.GetValue(beam_gm_offset)
                                    : group_token_num_gm_.GetValue(beam_gm_offset) - group_token_num_gm_.GetValue(beam_gm_offset - 1);
            int32_t beam_inner_offset = global_beam_idx == 0 ? 0 : group_token_num_gm_.GetValue(beam_gm_offset - 1);
            
            // Determine whether there is a task
            bool has_task = true;
            // Determine whether there is no task of the corresponding type
            if ((task_type == TASK_DIRECTION_UP && (global_beam_idx <= beam_inner_offset)) ||
                (task_type == TASK_DIRECTION_DOWN && (global_beam_idx > (beam_inner_offset + beam_token_num - 1)))) {
                has_task = false;
            }

            if (beam_token_num != 0 && has_task) {
                if (calc_block == core_id) {
                    beam_params.b_idx = global_batch_idx;
                    beam_params.beam_idx = global_beam_idx;
                    // Calculate the number of tasks actually executed and the starting offset of the write address based on the task type
                    if (task_type == TASK_DIRECTION_UP) {
                        beam_params.beam_inner_offset = beam_inner_offset;
                        beam_params.beam_token_num = (beam_inner_offset + beam_token_num) > beam_params.beam_idx ?
                                                    (beam_params.beam_idx - 1 - beam_inner_offset + 1) : beam_token_num;
                    } else {
                        beam_params.beam_inner_offset = beam_inner_offset > beam_params.beam_idx ? beam_inner_offset : (beam_params.beam_idx + 1);
                        beam_params.beam_token_num = (beam_inner_offset + beam_token_num - beam_params.beam_inner_offset);
                    }
                    beam_params.task_type = task_type;
                }
                calc_block++;
            }
            
            if (task_type == TASK_DIRECTION_UP) {
                global_beam_idx++;
                if (global_beam_idx > tiling_.beam_size - 1) {
                    global_batch_idx++;
                    global_beam_idx = 0;
                }
            } else {
                global_beam_idx--;
                if (global_beam_idx < 0) {
                    global_batch_idx++;
                    global_beam_idx = tiling_.beam_size - 1;
                }
            }
        }
        return calc_block > 0 ? true : false;
    }

    __aicore__ inline void process()
    {
        AscendC::TEventID event_id_mte2_to_mte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        AscendC::TEventID event_id_mte3_to_mte2 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        // 正序处理+逆序处理
        int8_t loop_num = 2;

        for (size_t loop_idx = 0; loop_idx < loop_num; loop_idx++) {
            int8_t task_type = loop_idx == 0 ? TASK_DIRECTION_UP : TASK_DIRECTION_DOWN;

            global_batch_idx = 0;
            global_beam_idx = loop_idx == 0 ? 0 : (tiling_.beam_size - 1);

            while (true) {
                beamParams beam_params;
                bool has_block = calcValidBlock(beam_params, task_type);

                if (!has_block) {
                    break;
                }

                int32_t src_beam_idx = beam_params.beam_idx;
                uint32_t src_offset = beam_params.b_idx * tiling_.block_batch_stride +
                                    src_beam_idx * tiling_.block_beam_stride;
                LocalTensor<T1> key_ub_ = ubuf_.GetWithOffset<T1>(KEY_UB_LEN / sizeof(T1), KEY_UB_OFFSET);
                LocalTensor<T1> value_ub_ = ubuf_.GetWithOffset<T1>(VALUE_UB_LEN / sizeof(T1), VALUE_UB_OFFSET);
                uint32_t copy_len = tiling_.decode_step * tiling_.head_num * tiling_.head_dim;
                // Processing is done according to the task type
                if (beam_params.beam_token_num > 0) {
                    SetFlag<HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);

                    // Copy key block
                    LocalTensor<T1> key_ub_ = ubuf_.GetWithOffset<T1>(KEY_UB_LEN / sizeof(T1), KEY_UB_OFFSET);
                    LocalTensor<T1> value_ub_ = ubuf_.GetWithOffset<T1>(VALUE_UB_LEN / sizeof(T1), VALUE_UB_OFFSET);
                    DataCopy(key_ub_, x_key_block_gm_[src_offset], copy_len);
                    DataCopy(value_ub_, x_value_block_gm_[src_offset], copy_len);
                }

                // Inter-core read synchronization to prevent overwriting when writing
                CrossCoreSetFlag<0x0, PIPE_MTE2>(0x8);
                CrossCoreWaitFlag(0x8);

                if (beam_params.beam_token_num > 0) {
                    SetFlag<HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
                    WaitFlag<HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
                    for (size_t i = 0; i < beam_params.beam_token_num; i++) {
                        // Move out
                        int32_t dst_beam_idx = beam_params.beam_inner_offset + i;
                        uint32_t dst_offset = beam_params.b_idx * tiling_.block_batch_stride +
                                            dst_beam_idx * tiling_.block_beam_stride;

                        if (src_beam_idx != dst_beam_idx) {
                            DataCopy(select_key_block_gm_[dst_offset], key_ub_, copy_len);
                            DataCopy(select_value_block_gm_[dst_offset], value_ub_, copy_len);
                        }
                    }
                }
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
                CrossCoreWaitFlag(0x8);
            }
        }

    }
};
}