<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnDecode Operator Design

English | [简体中文](../zh/mega_gdn_decode.md)

## Series documentation

[Core MegaChunkGdn](mega_chunk_gdn.md) |
[Fused Prefill](mega_gdn_prefill_op.md) | [Draft Decode](mega_gdn_draft_decode.md) |
[MTP Verify](mega_gdn_mtp_decode.md)

## Product support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 training/inference products | Yes |
| Atlas A2 training/inference products | Yes |

## Overview

`MegaGdnDecode` targets the single-token decode stage of the Qwen3.5 Gated
Delta Network (GDN). A single device invocation fuses:

1. width-4 causal convolution;
2. Q/K L2 normalization;
3. GDN gates, decay, and recurrent-state updates;
4. RMSNorm and the Z gate;
5. convolution-state and SSM-state updates.

Multi-Token Prediction (MTP) verification is handled by the separate
`MegaGdnMtpDecode` operator. The two operators have different state layouts,
checkpoint strides, and scheduling contracts and must not share an ABI.

## Symbols and shapes

```text
B = batch size
D = head dimension, fixed at 128
NK = number of key heads
NV = number of value heads
C = (2 * NK + NV) * D
N = number of state slots
```

## Computation

For every batch item, the convolution consumes the three history rows from
the read slot:

```text
conv = SiLU(
    W0 * conv_state_read[0] +
    W1 * conv_state_read[1] +
    W2 * conv_state_read[2] +
    W3 * qkv)
```

After splitting `conv` into Q, K, and V, Q and K are L2-normalized:

```text
q_hat = q / sqrt(sum(q * q) + 1e-6) / sqrt(128)
k_hat = k / sqrt(sum(k * k) + 1e-6)
```

Each value head uses its corresponding key head for the recurrent update:

```text
g       = -exp(a_log) * softplus(a + dt_bias, threshold=20)
decay   = exp(g)
beta    = sigmoid(b)

H       = decay * H_read
pred    = H^T * k_hat
delta   = (v - pred) * beta
H_write = H + k_hat * delta^T
o       = H_write^T * q_hat

out = RMSNorm(BF16_RINT(o), norm_weight, eps=1e-6) * SiLU(z)
```

## Parameters

All tensors use the ND format and must be contiguous.

| Name | Direction | Data type | Shape | Description |
| --- | --- | --- | --- | --- |
| `qkv` | Input | BF16 | `[B,C]` | Q/K/V input before convolution |
| `z` | Input | BF16 | `[B,NV,D]` | Output gate |
| `b` | Input | BF16 | `[B,NV]` | Beta-gate input |
| `a` | Input | BF16 | `[B,NV]` | Decay-gate input |
| `convWeight` | Input | BF16 | `[4,C]` | Width-4 depthwise-convolution weights |
| `convState` | Input | BF16 | `[N,3,C]` | Convolution state used for reads |
| `aLog` | Input | FP32 | `[NV]` | Recurrent decay parameter |
| `dtBias` | Input | FP32 | `[NV]` | Recurrent decay bias |
| `ssmState` | Input | FP32 | `[N,NV,D,D]` | SSM state used for reads |
| `readStateIndices` | Input | INT32 | `[B]` | State slot read by each batch item |
| `writeStateIndices` | Input | INT32 | `[B]` | State slot written by each batch item |
| `normWeight` | Input | BF16 | `[D]` | RMSNorm weight |
| `flaSsmStateLayout` | Attribute | BOOL | Scalar | `true` stores `[K,V]`; `false` stores `[V,K]`; defaults to `true` |
| `convOut` | Output | BF16 | `[B,C]` | Convolution output at the BF16 rounding boundary |
| `convStateOut` | Output | BF16 | `[N,3,C]` | Receives updated convolution-state write slots |
| `ssmStateOut` | Output | FP32 | `[N,NV,D,D]` | Receives updated SSM-state write slots |
| `out` | Output | BF16 | `[B,NV,D]` | Final output |

## State contract

### Convolution state

Each state slot stores three consecutive input rows. After an invocation, the
write slot is updated as follows:

```text
write_conv_state[0] = read_conv_state[1]
write_conv_state[1] = read_conv_state[2]
write_conv_state[2] = qkv
```

The convolution state stores the original `qkv`, not `convOut`.

### SSM state

Each batch item loads a complete FP32 state from its read slot, performs one
recurrent update, and stores the result in its write slot:

```text
H_read  = ssm_state[read_state_id]
H_write = updated_gdn_state(H_read)
ssm_state_out[write_state_id] = H_write
```

The recurrent math always uses the logical `H[K,V]` layout. With
`flaSsmStateLayout=true`, GM stores that layout directly; with `false`, GM stores
its transpose `H^T[V,K]`. The non-FLA PTO path uses transposed matvec,
reduction, and rank-one-update formulas directly instead of physically
transposing the `128 x 128` state.

### Input/output buffers and slot ownership

- input and output state buffers may use different device addresses;
- the kernel reads only read slots in the input buffers and writes only write
  slots in the output buffers;
- with separate output buffers, slots not written by the invocation are
  undefined; callers that need a complete state cache must preserve or
  pre-copy those slots;
- write slots must be unique within a batch;
- read slots may be shared by requests reading the same prefix;
- same-slot execution must complete the read-state GM-to-UB load before
  writing the result;
- in a prefix-fork, the read slot remains read-only and every update goes to
  the private write slot.

The tiling callback does not read device index values. The caller must
validate slot ranges, write-slot uniqueness, and cross-request ownership
before dispatch, avoiding device-to-host synchronization.

## Numerical contract

| Stage | Compute/materialization precision |
| --- | --- |
| Convolution accumulation and SiLU | FP32 |
| Convolution to Q/K/V | BF16 RINT |
| Q/K L2Norm | FP32 |
| `g` and `beta` | Computed in FP32 after converting BF16 inputs |
| Decay and SSM state | FP32 |
| Recurrent readout to Norm | BF16 RINT |
| RMSNorm and Z gate | FP32 |
| Final output | BF16 RINT/platform-reference-equivalent rounding |

Optimizations must preserve these rounding boundaries, `epsilon`, the
softplus threshold, and the state-update order.

## Tiling design

### Host validation

The current host tiling supports:

- `1 <= B <= 32`;
- `D = 128`;
- power-of-two `NK` in `[1,16]`;
- `NV % NK == 0` and `1 <= NV/NK <= 4`;
- `1 <= N <= 1024`;
- a convolution-state length fixed at 3.

### Tiling keys

| Key | Condition | Device path |
| ---: | --- | --- |
| 2 | `B=1` | Static batch-one fast path |
| 1 | `B>1` | Dynamic-batch path |
| 12 | `B=1` with non-FLA state layout | Batch-one non-FLA path |
| 11 | `B>1` with non-FLA state layout | Dynamic-batch non-FLA path |

Tiling data contains only `batch_size`, `num_k_heads`, and `num_v_heads`.

### Block scheduling

Convolution work is assigned in 128-channel tiles, while recurrent work is
assigned by `batch * value_head`:

```text
conv_tasks = C / 128
head_tasks = B * NV
```

A2/A3 use `MIX_AIC_1_2` and `CalcTschBlockDim`; A5 uses an AIV-only block
dimension. For the A2/A3 `B=1` fast path, each mixed block contains two AIVs,
so convolution contributes `(conv_tasks + 1) / 2` to block planning. Other
cases use `max(conv_tasks, head_tasks)`. The host sets `schedule_mode=1`.

After convolution stores complete, the kernel performs cross-AIV
synchronization before remapping channel owners to recurrent-head owners. A5
and A2/A3 implementations are selected by compile-time architecture guards;
A5-specific numerical or instruction optimizations must not alter A2/A3.

### UB and workspace

Each recurrent owner keeps a full FP32 `128 x 128` state and an equally sized
compute buffer in UB and reuses Q/K, gate, reduction, and Norm scratch.

The operator requests only the system workspace returned by
`platform.GetLibApiWorkSpaceSize()`. Business intermediates and state are not
stored in workspace.

## Constraints

- Only the dtypes, ranks, shapes, and contiguous ND tensors in the parameter
  table are supported;
- convolution width must be 4 and `D` must be 128;
- callers must keep read and write indices within `[0,N)`;
- two active batch items must not write the same state slot;
- after any state write, execution must not fall back to the unfused chain;
- prefix-cache callers must hold a shared read lease and an exclusive write
  lease before dispatch.

## Invocation and build layout

The operator is invoked through `aclnnMegaGdnDecode`. Its definition, host
tiling, and device kernel are located at:

- [`mega_gdn_decode_def.cpp`](../../xllm_ops/mega_gdn_decode/op_host/mega_gdn_decode_def.cpp)
- [`mega_gdn_decode_tiling.cpp`](../../xllm_ops/mega_gdn_decode/op_host/mega_gdn_decode_tiling.cpp)
- [`mega_gdn_decode.cpp`](../../xllm_ops/mega_gdn_decode/op_kernel/mega_gdn_decode.cpp)

The Python regression suite is
[`test_mega_gdn_decode.py`](../../test/python_test/test_mega_gdn_decode.py).
