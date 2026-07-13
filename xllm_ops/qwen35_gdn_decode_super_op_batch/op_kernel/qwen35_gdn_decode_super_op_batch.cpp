#include "tl_templates/ascend/common.h"
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

struct Qwen35GdnDecodeSuperOpBatchTilingData {
  int64_t batch_size;
};

extern "C" __global__ __aicore__ void qwen35_gdn_decode_super_op_batch(
    GM_ADDR qkv_handle, GM_ADDR z_handle, GM_ADDR b_handle, GM_ADDR a_handle,
    GM_ADDR conv_weight_handle, GM_ADDR conv_state_handle, GM_ADDR a_log_handle,
    GM_ADDR dt_bias_handle, GM_ADDR ssm_state_handle, GM_ADDR state_indices_handle,
    GM_ADDR norm_weight_handle, GM_ADDR conv_out_handle, GM_ADDR conv_state_out_handle,
    GM_ADDR ssm_state_out_handle, GM_ADDR out_handle, GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if ASCEND_IS_AIC {
    AscendC::SyncAll<false>();
    return;
  }
  if ASCEND_IS_AIV {
  REGISTER_TILING_DEFAULT(Qwen35GdnDecodeSuperOpBatchTilingData);
  GET_TILING_DATA_WITH_STRUCT(Qwen35GdnDecodeSuperOpBatchTilingData,
                              tiling_data,
                              tiling);
  const int32_t batch_size = static_cast<int32_t>(tiling_data.batch_size);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<bfloat16_t> qkv;
  qkv.SetGlobalBuffer((__gm__ bfloat16_t*)qkv_handle);
  AscendC::GlobalTensor<bfloat16_t> z;
  z.SetGlobalBuffer((__gm__ bfloat16_t*)z_handle);
  AscendC::GlobalTensor<bfloat16_t> b;
  b.SetGlobalBuffer((__gm__ bfloat16_t*)b_handle);
  AscendC::GlobalTensor<bfloat16_t> a;
  a.SetGlobalBuffer((__gm__ bfloat16_t*)a_handle);
  AscendC::GlobalTensor<bfloat16_t> conv_weight;
  conv_weight.SetGlobalBuffer((__gm__ bfloat16_t*)conv_weight_handle);
  AscendC::GlobalTensor<bfloat16_t> conv_state;
  conv_state.SetGlobalBuffer((__gm__ bfloat16_t*)conv_state_handle);
  AscendC::GlobalTensor<float> a_log;
  a_log.SetGlobalBuffer((__gm__ float*)a_log_handle);
  AscendC::GlobalTensor<float> dt_bias;
  dt_bias.SetGlobalBuffer((__gm__ float*)dt_bias_handle);
  AscendC::GlobalTensor<float> ssm_state;
  ssm_state.SetGlobalBuffer((__gm__ float*)ssm_state_handle);
  AscendC::GlobalTensor<int> state_indices;
  state_indices.SetGlobalBuffer((__gm__ int*)state_indices_handle);
  AscendC::GlobalTensor<bfloat16_t> norm_weight;
  norm_weight.SetGlobalBuffer((__gm__ bfloat16_t*)norm_weight_handle);
  AscendC::GlobalTensor<bfloat16_t> conv_out;
  conv_out.SetGlobalBuffer((__gm__ bfloat16_t*)conv_out_handle);
  AscendC::GlobalTensor<bfloat16_t> conv_state_out;
  conv_state_out.SetGlobalBuffer((__gm__ bfloat16_t*)conv_state_out_handle);
  AscendC::GlobalTensor<float> ssm_state_out;
  ssm_state_out.SetGlobalBuffer((__gm__ float*)ssm_state_out_handle);
  AscendC::GlobalTensor<bfloat16_t> out;
  out.SetGlobalBuffer((__gm__ bfloat16_t*)out_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto w_half0 = ascend_ub.GetWithOffset<bfloat16_t>(128, 0);
  auto w_half1 = ascend_ub.GetWithOffset<bfloat16_t>(128, 256);
  auto w_half2 = ascend_ub.GetWithOffset<bfloat16_t>(128, 512);
  auto w_half3 = ascend_ub.GetWithOffset<bfloat16_t>(128, 768);
  auto w0 = ascend_ub.GetWithOffset<float>(128, 1024);
  auto w1 = ascend_ub.GetWithOffset<float>(128, 1536);
  auto w2 = ascend_ub.GetWithOffset<float>(128, 2048);
  auto w3 = ascend_ub.GetWithOffset<float>(128, 2560);
  auto hist_half0 = ascend_ub.GetWithOffset<bfloat16_t>(128, 3072);
  auto hist_half1 = ascend_ub.GetWithOffset<bfloat16_t>(128, 3328);
  auto hist_half2 = ascend_ub.GetWithOffset<bfloat16_t>(128, 3584);
  auto x_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 3840);
  auto hist0 = ascend_ub.GetWithOffset<float>(128, 4096);
  auto hist1 = ascend_ub.GetWithOffset<float>(128, 4608);
  auto hist2 = ascend_ub.GetWithOffset<float>(128, 5120);
  auto x_fp32 = ascend_ub.GetWithOffset<float>(128, 5632);
  auto conv_acc = ascend_ub.GetWithOffset<float>(128, 6144);
  auto conv_tmp = ascend_ub.GetWithOffset<float>(128, 6656);
  auto conv_y = ascend_ub.GetWithOffset<float>(128, 7168);
  auto y_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 7680);
  auto save_half0 = ascend_ub.GetWithOffset<bfloat16_t>(128, 7936);
  auto save_half1 = ascend_ub.GetWithOffset<bfloat16_t>(128, 8192);
  auto save_half2 = ascend_ub.GetWithOffset<bfloat16_t>(128, 8448);
  auto q_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 8704);
  auto k_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 8960);
  auto a_half = ascend_ub.GetWithOffset<bfloat16_t>(1, 9216);
  auto b_half = ascend_ub.GetWithOffset<bfloat16_t>(1, 9248);
  auto q_fp32 = ascend_ub.GetWithOffset<float>(128, 9280);
  auto k_fp32 = ascend_ub.GetWithOffset<float>(128, 9792);
  auto scalar = ascend_ub.GetWithOffset<float>(1, 10304);
  auto scalar2 = ascend_ub.GetWithOffset<float>(1, 10336);
  auto norm_sq = ascend_ub.GetWithOffset<float>(128, 10368);
  auto norm_val = ascend_ub.GetWithOffset<float>(1, 43648);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(32768, 10880);
  auto scalar_tmp = ascend_ub.GetWithOffset<float>(1, 43680);
  auto norm_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 143552);
  auto z_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 143808);
  auto weight_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 144064);
  auto norm_fp32 = ascend_ub.GetWithOffset<float>(128, 144320);
  auto z_fp32 = ascend_ub.GetWithOffset<float>(128, 144832);
  auto weight_fp32 = ascend_ub.GetWithOffset<float>(128, 145344);
  auto square_fp32 = ascend_ub.GetWithOffset<float>(128, 145856);
  auto rms = ascend_ub.GetWithOffset<float>(1, 146368);
  auto gate_fp32 = ascend_ub.GetWithOffset<float>(128, 146400);
  auto final_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 146912);
  auto v_half = ascend_ub.GetWithOffset<bfloat16_t>(64, 43712);
  auto h_vec = ascend_ub.GetWithOffset<float>(8192, 43840);
  auto v_fp32 = ascend_ub.GetWithOffset<float>(64, 76608);
  auto k_1d = ascend_ub.GetWithOffset<float>(128, 76864);
  auto broadcast_buf = ascend_ub.GetWithOffset<float>(8192, 77376);
  auto compute_buf = ascend_ub.GetWithOffset<float>(8192, 110144);
  auto pred = ascend_ub.GetWithOffset<float>(64, 142912);
  auto delta = ascend_ub.GetWithOffset<float>(64, 143168);
  auto out_half = ascend_ub.GetWithOffset<bfloat16_t>(64, 143424);
  auto vid = AscendC::GetSubBlockIdx();
  tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(w_half0[0], conv_weight[((cid * 256) + (vid * 128))], 20480, 1, 128, bfloat16_t(0.000000e+00f));
  tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(w_half1[0], conv_weight[(((cid * 256) + (vid * 128)) + 5120)], 20480, 1, 128, bfloat16_t(0.000000e+00f));
  tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(w_half2[0], conv_weight[(((cid * 256) + (vid * 128)) + 10240)], 20480, 1, 128, bfloat16_t(0.000000e+00f));
  tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(w_half3[0], conv_weight[(((cid * 256) + (vid * 128)) + 15360)], 20480, 1, 128, bfloat16_t(0.000000e+00f));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
  AscendC::Cast(w0[0], w_half0[0], AscendC::RoundMode::CAST_NONE, 128);
  AscendC::Cast(w1[0], w_half1[0], AscendC::RoundMode::CAST_NONE, 128);
  AscendC::Cast(w2[0], w_half2[0], AscendC::RoundMode::CAST_NONE, 128);
  AscendC::Cast(w3[0], w_half3[0], AscendC::RoundMode::CAST_NONE, 128);
  for (int32_t batch_idx = 0; batch_idx < 4; ++batch_idx) {
    AscendC::PipeBarrier<PIPE_ALL>();
    if (batch_idx < batch_size) {
      AscendC::PipeBarrier<PIPE_ALL>();
      int32_t state_idx = state_indices.GetValue(batch_idx);
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(hist_half0[0], conv_state[(((state_idx * 15360) + (cid * 256)) + (vid * 128))], 122880, 1, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(hist_half1[0], conv_state[((((state_idx * 15360) + (cid * 256)) + (vid * 128)) + 5120)], 122880, 1, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(hist_half2[0], conv_state[((((state_idx * 15360) + (cid * 256)) + (vid * 128)) + 10240)], 122880, 1, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(x_half[0], qkv[(((batch_idx * 5120) + (cid * 256)) + (vid * 128))], 20480, 1, 128, bfloat16_t(0.000000e+00f));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
      AscendC::Cast(hist0[0], hist_half0[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::Cast(hist1[0], hist_half1[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::Cast(hist2[0], hist_half2[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::Cast(x_fp32[0], x_half[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(conv_acc[0], w0[0], hist0[0], 128);
      AscendC::Mul(conv_tmp[0], w1[0], hist1[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(conv_acc[0], conv_acc[0], conv_tmp[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(conv_tmp[0], w2[0], hist2[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(conv_acc[0], conv_acc[0], conv_tmp[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::MulAddDst(conv_acc[0], x_fp32[0], w3[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Silu(conv_y[0], conv_acc[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(y_half[0], conv_y[0], AscendC::RoundMode::CAST_RINT, 128);
      AscendC::Cast(save_half0[0], hist1[0], AscendC::RoundMode::CAST_RINT, 128);
      AscendC::Cast(save_half1[0], hist2[0], AscendC::RoundMode::CAST_RINT, 128);
      AscendC::Cast(save_half2[0], x_fp32[0], AscendC::RoundMode::CAST_RINT, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128>(conv_out[(((batch_idx * 5120) + (cid * 256)) + (vid * 128))], y_half[0], 20480, 1, 128);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128>(conv_state_out[(((state_idx * 15360) + (cid * 256)) + (vid * 128))], save_half0[0], 122880, 1, 128);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128>(conv_state_out[((((state_idx * 15360) + (cid * 256)) + (vid * 128)) + 5120)], save_half1[0], 122880, 1, 128);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128>(conv_state_out[((((state_idx * 15360) + (cid * 256)) + (vid * 128)) + 10240)], save_half2[0], 122880, 1, 128);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
  }
  AscendC::SyncAll<false>();
  for (int32_t head_pass = 0; head_pass < 3; ++head_pass) {
    AscendC::PipeBarrier<PIPE_ALL>();
    if ((((head_pass * 20) + cid) - (batch_size * 12)) < 0) {
      AscendC::PipeBarrier<PIPE_ALL>();
      int32_t state_idx_1 = state_indices.GetValue((((head_pass * 5) + (cid / 4)) / 3));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(q_half[0], conv_out[(((((head_pass * 5) + (cid / 4)) / 3) * 5120) + ((((((head_pass * 40) + (cid * 2)) + vid) % 24) / 3) * 128))], 20480, (((((head_pass * 40) + (cid * 2)) + vid) <= 95) ? 1 : 0), 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(k_half[0], conv_out[((((((head_pass * 5) + (cid / 4)) / 3) * 5120) + ((((((head_pass * 40) + (cid * 2)) + vid) % 24) / 3) * 128)) + 1024)], 20480, (((((head_pass * 40) + (cid * 2)) + vid) <= 95) ? 1 : 0), 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 1>(a_half[0], a[(((head_pass * 40) + (cid * 2)) + vid)], 96, (((((head_pass * 40) + (cid * 2)) + vid) <= 95) ? 1 : 0), 1, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 1>(b_half[0], b[(((head_pass * 40) + (cid * 2)) + vid)], 96, (((((head_pass * 40) + (cid * 2)) + vid) <= 95) ? 1 : 0), 1, bfloat16_t(0.000000e+00f));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
      AscendC::Cast(q_fp32[0], q_half[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::Cast(k_fp32[0], k_half[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::Cast(scalar[0], a_half[0], AscendC::RoundMode::CAST_NONE, 1);
      AscendC::Cast(scalar2[0], b_half[0], AscendC::RoundMode::CAST_NONE, 1);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(norm_sq[0], q_fp32[0], q_fp32[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_sum<float,  1,  128,  -1>(norm_val[0], norm_sq[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Adds(norm_val[0], norm_val[0], 1.000000e-06f, 1);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sqrt(norm_val[0], norm_val[0], 1);
      AscendC::PipeBarrier<PIPE_ALL>();
      float q_norm_scalar = norm_val.GetValue(0);
      AscendC::Muls(q_fp32[0], q_fp32[0], 1.0f / q_norm_scalar, 128);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Muls(q_fp32[0], q_fp32[0], 8.838835e-02f, 128);
      }
      AscendC::Mul(norm_sq[0], k_fp32[0], k_fp32[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_sum<float,  1,  128,  -1>(norm_val[0], norm_sq[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Adds(norm_val[0], norm_val[0], 1.000000e-06f, 1);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sqrt(norm_val[0], norm_val[0], 1);
      AscendC::PipeBarrier<PIPE_ALL>();
      float k_norm_scalar = norm_val.GetValue(0);
      AscendC::Muls(k_fp32[0], k_fp32[0], 1.0f / k_norm_scalar, 128);
      tl::ascend::copy_gm_to_ub<float, 1>(scalar_tmp[0], a_log[((((head_pass * 40) + (cid * 2)) + vid) % 24)], 24, 1, 1, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
      AscendC::Exp(scalar_tmp[0], scalar_tmp[0], 1);
      AscendC::PipeBarrier<PIPE_ALL>();
      float exp_a = scalar_tmp.GetValue(0);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
      tl::ascend::copy_gm_to_ub<float, 1>(norm_val[0], dt_bias[((((head_pass * 40) + (cid * 2)) + vid) % 24)], 24, 1, 1, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
      AscendC::PipeBarrier<PIPE_ALL>();
      float x_gate = (scalar.GetValue(0) + norm_val.GetValue(0));
      AscendC::PipeBarrier<PIPE_ALL>();
      if (2.000000e+01f < x_gate) {
        norm_val.SetValue(0, x_gate);
      } else {
        scalar_tmp.SetValue(0, x_gate);
        AscendC::Exp(scalar_tmp[0], scalar_tmp[0], 1);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Adds(scalar_tmp[0], scalar_tmp[0], 1.000000e+00f, 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Ln(scalar_tmp[0], scalar_tmp[0], 1);
        norm_val.SetValue(0, scalar_tmp.GetValue(0));
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      scalar_tmp.SetValue(0, ((exp_a * -1.000000e+00f) * norm_val.GetValue(0)));
      AscendC::Exp(scalar_tmp[0], scalar_tmp[0], 1);
      AscendC::PipeBarrier<PIPE_ALL>();
      float decay = scalar_tmp.GetValue(0);
      AscendC::Sigmoid(scalar2[0], scalar2[0], tmp_ub[0], 1);
      AscendC::PipeBarrier<PIPE_ALL>();
      float beta_gate = scalar2.GetValue(0);
      for (int32_t half_idx = 0; half_idx < 2; ++half_idx) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
        tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(v_half[0], conv_out[(((((((head_pass * 5) + (cid / 4)) / 3) * 5120) + (((((head_pass * 40) + (cid * 2)) + vid) % 24) * 128)) + (half_idx * 64)) + 2048)], 20480, (((((head_pass * 40) + (cid * 2)) + vid) <= 95) ? 1 : 0), 64, bfloat16_t(0.000000e+00f));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3);
        tl::ascend::copy_gm_to_ub<float, 64, 128>(h_vec[0], ssm_state[(((state_idx_1 * 393216) + (((((head_pass * 40) + (cid * 2)) + vid) % 24) * 16384)) + (half_idx * 64))], 128, 128, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::Cast(v_fp32[0], v_half[0], AscendC::RoundMode::CAST_NONE, 64);
        {
        AscendC::Muls(h_vec[0], h_vec[0], decay, 8192);
        }
        tl::ascend::copy_ub_to_ub<float, float, 128>(k_1d[0], k_fp32[0]);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::Broadcast<float, 2, 1, false>(broadcast_buf[0],k_1d[0],tmp_ub,(uint32_t[]){128, 64}, (uint32_t[]){128, 1});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(compute_buf[0], h_vec[0], broadcast_buf[0], 8192);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::reduce_sum<float,  128,  64,  0>(pred[0], compute_buf[0], tmp_ub[0], true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(delta[0], v_fp32[0], pred[0], 64);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(delta[0], delta[0], beta_gate, 64);
        }
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::Broadcast<float, 2, 0, false>(compute_buf[0],delta[0],tmp_ub,(uint32_t[]){128, 64}, (uint32_t[]){1, 64});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(h_vec[0], broadcast_buf[0], compute_buf[0], 8192);
        tl::ascend::copy_ub_to_ub<float, float, 128>(k_1d[0], q_fp32[0]);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::Broadcast<float, 2, 1, false>(broadcast_buf[0],k_1d[0],tmp_ub,(uint32_t[]){128, 64}, (uint32_t[]){128, 1});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(compute_buf[0], h_vec[0], broadcast_buf[0], 8192);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::reduce_sum<float,  128,  64,  0>(pred[0], compute_buf[0], tmp_ub[0], true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(out_half[0], pred[0], AscendC::RoundMode::CAST_RINT, 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<bfloat16_t, bfloat16_t, 64>(norm_half[(half_idx * 64)], out_half[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<float, 64, 128>(ssm_state_out[(((state_idx_1 * 393216) + (((((head_pass * 40) + (cid * 2)) + vid) % 24) * 16384)) + (half_idx * 64))], h_vec[0], 128, 128, 64);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(6);
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(z_half[0], z[(((head_pass * 5120) + (cid * 256)) + (vid * 128))], 12288, 1, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128>(weight_half[0], norm_weight[0], 128, 1, 128, bfloat16_t(0.000000e+00f));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(norm_fp32[0], norm_half[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::Cast(z_fp32[0], z_half[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::Cast(weight_fp32[0], weight_half[0], AscendC::RoundMode::CAST_NONE, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(square_fp32[0], norm_fp32[0], norm_fp32[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_sum<float,  1,  128,  -1>(rms[0], square_fp32[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(rms[0], rms[0], 1.0f / 1.280000e+02f, 1);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Adds(rms[0], rms[0], 1.000000e-06f, 1);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sqrt(rms[0], rms[0], 1);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::PipeBarrier<PIPE_ALL>();
      auto rms_scalar = 1.0f / (float)rms.GetValue(0);
      AscendC::Muls(norm_fp32[0], norm_fp32[0], rms_scalar, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(norm_fp32[0], norm_fp32[0], weight_fp32[0], 128);
      AscendC::Silu(gate_fp32[0], z_fp32[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(norm_fp32[0], norm_fp32[0], gate_fp32[0], 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(final_half[0], norm_fp32[0], AscendC::RoundMode::CAST_RINT, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128>(out[(((head_pass * 5120) + (cid * 256)) + (vid * 128))], final_half[0], 12288, 1, 128);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
  }
  }
}

// The CANN wrapper for mixed AIC/AIV kernels clears the matmul workspace.
#include "lib/matmul_intf.h"
