<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaChunkGdn Operator Design

English | [简体中文](../zh/mega_chunk_gdn.md)

## Series documentation

[Fused Prefill](mega_gdn_prefill_op.md) |
[Single-token Decode](mega_gdn_decode.md) | [Draft Decode](mega_gdn_draft_decode.md) |
[MTP Verify](mega_gdn_mtp_decode.md)

## Product support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 training/inference products | Yes |
| Atlas A2 training/inference products | Yes |

## Overview

`MegaChunkGdn` is the chunk GDN core operator for the Prefill path. The caller
must prepare the causal-convolution result, Q/K L2 normalization, and gates
before invoking the operator. It processes 128 tokens per chunk and fuses
cumulative decay, KKT, lower-triangular inversion, the WY transform, state
recurrence, and output computation.

Use `MegaGdnPrefillOp` when convolution, cache updates, and gated RMSNorm must
also be fused.

## Symbols and shapes

```text
B  = number of sequences described by cu_seqlens
T  = total number of packed tokens
D  = head dimension, fixed at 128
Ck = chunk size, fixed at 128
NK = number of key heads
NV = number of value heads
M  = NV * sum_i ceil(sequence_length_i / Ck)
```

Inputs use a packed BSND layout with a fixed batch axis:

| Name | Direction | Data type | Shape | Description |
| --- | --- | --- | --- | --- |
| `q` | Input | FP16/BF16 | `[1,T,NK,D]` | L2-normalized query |
| `k` | Input | FP16/BF16 | `[1,T,NK,D]` | L2-normalized key |
| `v` | Input | FP16/BF16 | `[1,T,NV,D]` | Value |
| `g` | Input | FP32 | `[1,T,NV]` | Log-decay gate |
| `beta` | Input | FP16/BF16 | `[1,T,NV]` | Update gate |
| `maskLower` | Input | FP32 | `[128,128]` | Strictly lower-triangular mask |
| `maskFull` | Input | FP32 | `[128,128]` | Lower-triangular mask including the diagonal |
| `minusIdentity` | Input | FP16/BF16 | `[128,128]` | Matrix whose diagonal is `-1` |
| `cuSeqlens` | Input | INT32 | `[B+1]` | Packed sequence boundaries; first value 0 and last value T |
| `initialState` | Input | FP16/BF16 | `[B,NV,D,D]` | Initial state of every sequence |
| `numMatrices` | Attribute | INT64 | Scalar | Must equal `M`; the Python binding requires a positive value |
| `hasInitialState` | Attribute | BOOL | Scalar | Whether to load `initialState` |
| `fftsAddr` | Attribute | INT64 | Scalar | FFTS address on A2/A3; unused on A5 |

Q/K/V/beta must use the same compute dtype. The host requires
`1 <= NV <= 64`, `NK > 0`, and `NV % NK == 0`; the last dimensions of Q and V
must both be 128.

## Outputs

The raw ACLNN interface returns 12 tensors. Apart from `out` and `finalState`,
these are internal results of the fused pipeline; a higher-level wrapper will
usually expose only a subset.

| Name | Data type | Shape | Description |
| --- | --- | --- | --- |
| `out` | Same as V | `[1,T,NV,D]` | Core output before an external `D^-0.5` scale |
| `gSum` | FP32 | `[1,T,NV]` | Cumulative gate for every sequence and chunk |
| `gT` | FP32 | `[NV,T]` | Transposed gate |
| `betaT` | Compute dtype | `[NV,T]` | Transposed beta |
| `a` | Compute dtype | `[1,T,NV,128]` | Chunk lower-triangular matrix |
| `aInvF32` | FP32 | `[1,T,NV,128]` | FP32 inversion intermediate |
| `aInv` | Compute dtype | `[1,T,NV,128]` | Inverse lower-triangular matrix |
| `w` | Compute dtype | `[1,T,NV,D]` | WY intermediate W |
| `u` | Compute dtype | `[1,T,NV,D]` | WY intermediate U |
| `h` | Compute dtype | `[M,D,D]` | State of every chunk |
| `vNew` | Compute dtype | `[1,T,NV,D]` | Corrected value |
| `finalState` | Compute dtype | `[B,NV,D,D]` | Final state of every sequence |

## Computation

Within every chunk, the operator first computes the cumulative gate and builds
a strictly lower-triangular system:

```text
G_t       = cumsum(g)_t
decay_ij  = exp(G_i - G_j), i >= j
A         = -tril((beta * K) @ K^T * decay, diagonal=-1)
A_inv     = inverse(I - A)
```

It then evaluates the corrected value in WY form and updates the recurrent
state chunk by chunk:

```text
V_new = chunk_update(A_inv, K, V, beta, G)
O_i   = recurrent_readout(Q_i, K_i, V_new_i, H_previous, G_i)
H_i   = recurrent_state_update(H_previous, K_i, V_new_i, G_i)
```

`cuSeqlens` isolates sequences from one another. A final chunk shorter than 128
tokens is padded internally and never crosses a sequence boundary.

## State contract

- with `hasInitialState=false`, the kernel uses a zero state and does not read
  the value of `initialState`;
- with `hasInitialState=true`, `initialState[b]` corresponds to packed sequence
  b;
- `finalState[b]` is the state after the last valid token of that sequence;
- the state head axis is expanded by value head, while Q/K map to value heads
  according to `NV/NK`;
- the caller-provided `numMatrices` must equal the total number of chunks
  described by `cuSeqlens`.

## Data-type contract

The shared kernel represents its compute type with the independent `ComputeT`
alias:

```text
GDN_COMPUTE_DTYPE defaults to DTYPE_Q
ComputeT = GDN_COMPUTE_DTYPE
```

The standalone `MegaChunkGdn` therefore preserves its existing behavior: FP16
inputs use FP16 and BF16 inputs use BF16. However, the public Q type and the
internal compute type are semantically separate in the source. Fused Prefill
can override `GDN_COMPUTE_DTYPE` explicitly without presenting its internal
type as the public `DTYPE_Q`.

## Tiling, synchronization, and workspace

When `numMatrices` is omitted, the host uses `ceil(T/128) * NV` as a compatibility
fallback. For packed multi-sequence invocations, the caller must still pass the
accurate value obtained by rounding every sequence length independently.

A2/A3 use FFTS for synchronization between MIX AIC/AIV stages. A5 uses the full
1:2 MIX block grid and PTO `SYNCALL<SyncCoreType::Mix>`. The host restricts the
block count to a factor of the number of value heads so that chunks belonging
to the same head retain a stable owner.

User workspace contains per-block temporary regions for KKT, WY, H/O, a
16-MiB-aligned region for H workspace, and an 8-MiB phase reserve, followed by
CANN system workspace. The caller must use the size returned by
`aclnnMegaChunkGdnGetWorkspaceSize` and must not infer it from output shapes.

## Invocation and build layout

The operator is invoked through `aclnnMegaChunkGdn`. Its definition, host
tiling, and device kernel are located at:

- [`mega_chunk_gdn_def.cpp`](../../xllm_ops/mega_chunk_gdn/op_host/mega_chunk_gdn_def.cpp)
- [`mega_chunk_gdn_tiling.cpp`](../../xllm_ops/mega_chunk_gdn/op_host/mega_chunk_gdn_tiling.cpp)
- [`mega_chunk_gdn.cpp`](../../xllm_ops/mega_chunk_gdn/op_kernel/mega_chunk_gdn.cpp)

The Python wrapper and regression suite are located at:

- [`custom_ops.py`](../../test/python_test/custom_ops.py)
- [`test_mega_chunk_gdn.py`](../../test/python_test/test_mega_chunk_gdn.py)
