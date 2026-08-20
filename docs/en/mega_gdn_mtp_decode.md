<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnMtpDecode Operator Design

English | [简体中文](../zh/mega_gdn_mtp_decode.md)

## Product support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 training/inference products | Yes |
| Atlas A2 training/inference products | Yes |

## Overview

`MegaGdnMtpDecode` targets the Multi-Token Prediction (MTP) verification
stage of the Qwen3.5 Gated Delta Network (GDN). A single device invocation
fuses:

1. width-4 causal convolution;
2. Q/K L2 normalization;
3. GDN gates, decay, and recurrent-state updates;
4. one SSM checkpoint write per verified token;
5. RMSNorm and the Z gate.

The operator is intentionally separate from the single-token
`MegaGdnDecode` operator. MTP requires an extended convolution state, a
checkpoint for every token, and independent read and write state slots.
Reusing the single-token ABI would violate those state semantics.

## Symbols and shapes

This document uses the following symbols:

```text
B = batch size
K = number of speculative tokens, 1 <= K <= 16
S = K + 1, the verification length including the correction token
D = head dimension, fixed at 128
NK = number of key heads
NV = number of value heads
C = (2 * NK + NV) * D
L = K + 3 = S + 2, the per-slot convolution-state length
N = number of state slots
```

## Computation

For every `t in [0, K]`, the operator first evaluates the causal
convolution:

```text
conv_t = SiLU(
    W0 * history_t[0] +
    W1 * history_t[1] +
    W2 * history_t[2] +
    W3 * qkv_t)
```

After splitting `conv_t` into Q, K, and V, Q and K are L2-normalized:

```text
q_hat_t = q_t / sqrt(sum(q_t * q_t) + 1e-6) / sqrt(128)
k_hat_t = k_t / sqrt(sum(k_t * k_t) + 1e-6)
```

The GDN gates and recurrent state are then updated as follows:

```text
g_fp32_t = -exp(a_log) * softplus(a_t + dt_bias, threshold=20)
g_t      = FP32(BF16_RINT(g_fp32_t))
decay_t  = exp(g_t)
beta_t   = FP32(BF16_RINT(sigmoid(b_t)))

H_t      = decay_t * H_(t-1)
pred_t   = H_t^T * k_hat_t
delta_t  = (v_t - pred_t) * beta_t
H_t      = H_t + k_hat_t * delta_t^T
o_t      = H_t^T * q_hat_t

out_t = RMSNorm(BF16_RINT(o_t), norm_weight, eps=1e-6)
        * SiLU(z_t)
```

`H_-1` is loaded from the selected read checkpoint. Each `H_t` is written
to its write checkpoint immediately. The next token consumes the state that
remains in UB instead of reloading it from GM.

## Parameters

All tensors use the ND format and must be contiguous.

| Name | Direction | Data type | Shape | Description |
| --- | --- | --- | --- | --- |
| `qkv` | Input | BF16 | `[B,S,C]` | Q/K/V input before convolution |
| `z` | Input | BF16 | `[B,S,NV,D]` | Output gate |
| `b` | Input | BF16 | `[B,S,NV]` | Beta-gate input |
| `a` | Input | BF16 | `[B,S,NV]` | Decay-gate input |
| `convWeight` | Input | BF16 | `[4,C]` | Width-4 depthwise-convolution weights |
| `convState` | Input | BF16 | `[N,L,C]` | Convolution state used for reads |
| `aLog` | Input | FP32 | `[NV]` | Recurrent decay parameter |
| `dtBias` | Input | FP32 | `[NV]` | Recurrent decay bias |
| `ssmState` | Input | FP32 | `[N*S,NV,D,D]` | SSM checkpoints used for reads |
| `readStateIndices` | Input | INT32 | `[B]` | State slot read by each batch item |
| `writeStateIndices` | Input | INT32 | `[B]` | State slot written by each batch item |
| `numAcceptedTokens` | Input | INT32 | `[B]` | Previous accepted count in `[1,S]` |
| `normWeight` | Input | BF16 | `[D]` | RMSNorm weight |
| `convOut` | Output | BF16 | `[B,S,C]` | Convolution output at the BF16 rounding boundary |
| `convStateOut` | Output | BF16 | `[N,L,C]` | Receives updated convolution-state write slots |
| `ssmStateOut` | Output | FP32 | `[N*S,NV,D,D]` | Receives updated per-token write checkpoints |
| `out` | Output | BF16 | `[B,S,NV,D]` | Final output |

## State contract

### SSM checkpoints

`numAcceptedTokens` includes the correction token. The read location is:

```text
read_checkpoint = read_state_id * S + num_accepted_tokens - 1
```

The write location for each token in the current verification is:

```text
write_checkpoint(t) = write_state_id * S + t,  t in [0, K]
```

### Convolution state

Token 0 reads its three-row history from:

```text
read_conv_state[num_accepted_tokens - 1 : num_accepted_tokens + 2]
```

At the end of the invocation, the write slot becomes:

```text
write_conv_state[0:2]   =
    read_conv_state[num_accepted_tokens : num_accepted_tokens + 2]
write_conv_state[2:S+2] = qkv[0:S]
```

The convolution state stores the original `qkv`, not `convOut`.

### Input/output buffers and slot ownership

- input and output state buffers may use different device addresses;
- the kernel reads only read slots in the input buffers and writes only write
  slots in the output buffers;
- with separate output buffers, slots not written by the invocation are
  undefined; callers that need a complete state cache must preserve or
  pre-copy those slots;
- write slots must be unique within a batch;
- read slots may be shared by multiple requests reading the same prefix;
- when `read_state_id == write_state_id`, the kernel must load the initial
  SSM state into UB before checkpoint 0 is written;
- when `read_state_id != write_state_id`, the read slot remains read-only and
  every update is written to the write slot.

The tiling callback validates dtype, rank, and shape only. It does not read
device-tensor values. The caller must validate accepted counts, slot ranges,
write-slot uniqueness, and cross-request ownership before dispatch, avoiding
a device-to-host synchronization.

## Numerical contract

| Stage | Compute/materialization precision |
| --- | --- |
| Convolution accumulation and SiLU | FP32 |
| Convolution to Q/K/V | BF16 RINT |
| Q/K L2Norm | FP32 |
| `g` and `beta` | FP32 compute, BF16 RINT materialization, then FP32 |
| Decay and SSM state | FP32 |
| Recurrent readout to Norm | BF16 RINT |
| RMSNorm and Z gate | FP32 |
| Final output | BF16 RINT |

Optimizations must preserve these rounding boundaries, `epsilon`, the
softplus threshold, and the state-update order.

## Tiling design

### Host validation

The current host tiling supports:

- `1 <= B <= 32`;
- `2 <= S <= 17`, or `1 <= K <= 16`;
- `D = 128`;
- power-of-two `NK` in `[1,16]`;
- `NV % NK == 0` and `1 <= NV/NK <= 4`;
- `1 <= N <= 1024`;
- at least 175,360 bytes of UB.

### Tiling keys

| Key | Condition | Device path |
| ---: | --- | --- |
| 101/102/103/104/105 | `K=1/2/3/4/5` | Corresponding static-K template |
| 108 | `K=8` | Static K8 template |
| 208 | `K=8, B=4, NK=8, NV=24` with sufficient resources | Q/K group cache and deferred Norm |
| 308 | Same shape as key 208, on Ascend 950 with sufficient AIV/UB | A5 two-owner Q/K-group scheduling |
| 210-216 | `K=10-16` with at least 182,816 bytes of UB | Static K with deferred Norm |
| 100 | Other supported K values or insufficient specialized-key resources | Dynamic-K fallback |

Key 308 is compiled and dispatched only under `PTO_NPU_ARCH_A5`. A2 and A3
do not contain this device branch and retain the mixed AIC/AIV path; A5 uses
the AIV-only path. Consequently, A5-specific optimization must not change A2
or A3 device code.

### Block scheduling

Convolution owners are assigned by 128-channel tiles:

```text
conv_tasks = C / 128
```

Recurrent owners are assigned by `batch * value_head`:

```text
recurrent_tasks = B * NV
task_count = max(conv_tasks, recurrent_tasks)
used_aiv = min(task_count, platform_aiv_count)
```

A2/A3 use `CalcTschBlockDim` and `MIX_AIC_1_2`; A5 uses an AIV-only block
dimension. The host sets `schedule_mode=1`. After convolution stores complete,
the owners perform all-AIV synchronization before remapping to recurrent work.

### UB and workspace

The generic host UB reserve is 175,360 bytes; keys 208/308 reserve 187,936
bytes; the largest deferred-Norm path for K10-16 uses 182,816 bytes. The full
FP32 state and primary compute buffers are both `128 x 128` and are reused
across all S updates of a head.

The operator requests only the system workspace returned by
`platform.GetLibApiWorkSpaceSize()`. Q/K/V, gates, and state are not stored in
workspace. Checkpoints are written directly to the target GM state.

## Constraints

- Only the dtypes, ranks, shapes, and contiguous ND tensors in the parameter
  table are supported;
- convolution width must be 4 and `D` must be 128;
- the caller must validate index and accepted-count values;
- two active batch items must not write the same state slot;
- after any state write, execution must not fall back to the unfused chain;
- prefix-cache callers must hold a shared read lease and an exclusive write
  lease before dispatch;
- graph padding must use valid, exclusive sink write slots rather than
  negative or out-of-range indices.

## Invocation and build layout

The operator is invoked through `aclnnMegaGdnMtpDecode`. Its definition,
host tiling, and device kernel are located at:

- [`mega_gdn_mtp_decode_def.cpp`](../../xllm_ops/mega_gdn_mtp_decode/op_host/mega_gdn_mtp_decode_def.cpp)
- [`mega_gdn_mtp_decode_tiling.cpp`](../../xllm_ops/mega_gdn_mtp_decode/op_host/mega_gdn_mtp_decode_tiling.cpp)
- [`mega_gdn_mtp_decode.cpp`](../../xllm_ops/mega_gdn_mtp_decode/op_kernel/mega_gdn_mtp_decode.cpp)

The Python regression suite is
[`test_mega_gdn_mtp_decode.py`](../../test/python_test/test_mega_gdn_mtp_decode.py).
