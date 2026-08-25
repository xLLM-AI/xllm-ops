<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnPrefillOp Operator Design

English | [简体中文](../zh/mega_gdn_prefill_op.md)

## Series documentation

[Core MegaChunkGdn](mega_chunk_gdn.md) |
[Single-token Decode](mega_gdn_decode.md) | [Draft Decode](mega_gdn_draft_decode.md) |
[MTP Verify](mega_gdn_mtp_decode.md)

## Product support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 training/inference products | Yes |
| Atlas A2 training/inference products | Yes |

## Overview

`MegaGdnPrefillOp` targets packed variable-length Prefill for the Qwen3.5 GDN.
A single device invocation fuses:

1. width-4 causal convolution and Conv-cache updates;
2. splitting packed Q/K/V and applying Q/K L2 normalization;
3. decay-gate and beta-gate preparation;
4. the KKT, solve, WY, and H/O pipeline of `MegaChunkGdn`;
5. reading the SSM cache and writing the final state;
6. RMSNorm and the Z gate.

Unlike `MegaChunkGdn`, this operator accepts mixed BF16 QKV before convolution
and directly returns the BF16 normalized hidden state required by the model.

## Symbols and shapes

```text
B  = batch size
T  = total number of packed tokens
D  = head dimension, fixed at 128
NK = number of key heads
NV = number of value heads
C  = (2 * NK + NV) * D
N  = number of Conv-state slots
R  = checkpoint stride
M  = NV * sum_i ceil(sequence_length_i / 128)
```

All tensors use the ND format and must be contiguous.

| Name | Direction | Data type | Shape | Description |
| --- | --- | --- | --- | --- |
| `mixed_qkv` | Input | BF16 | `[T,C]` | Packed Q/K/V before convolution |
| `b` | Input | BF16 | `[T,NV]` | Beta-gate input |
| `a` | Input | BF16 | `[T,NV]` | Decay-gate input |
| `z` | Input | BF16 | `[T,NV,D]` | Output gate |
| `conv_weight` | Input | BF16 | `[4,C]` | Depthwise-convolution weights |
| `conv_state` | Input | BF16 | `[N,R+2,C]` | Conv state/cache |
| `a_log` | Input | FP32 | `[NV]` | Recurrent decay parameter |
| `dt_bias` | Input | FP32 | `[NV]` | Recurrent decay bias |
| `conv_state_read_indices` | Input | INT32 | `[B]` | Conv read slot; a negative value means no initial state |
| `conv_state_write_indices` | Input | INT32 | `[B]` | Conv write slot |
| `ssm_state_read_indices` | Input | INT32 | `[B]` | Absolute read index in the first SSM-cache dimension; a negative value means a zero state |
| `ssm_state_write_indices` | Input | INT32 | `[B]` | Absolute write index in the first SSM-cache dimension |
| `ssm_cache` | Input | FP32 | `[N*R,NV,D,D]` | Flattened checkpoint cache |
| `mask_lower` | Input | FP32 | `[128,128]` | Strictly lower-triangular mask |
| `mask_full` | Input | FP32 | `[128,128]` | Lower-triangular mask including the diagonal |
| `minus_identity` | Input | BF16 | `[128,128]` | Matrix whose diagonal is `-1` |
| `cu_seqlens` | Input | INT32 | `[B+1]` | Packed sequence boundaries |
| `norm_weight` | Input | BF16 | `[D]` | RMSNorm weight |
| `ffts_addr` | Attribute | INT64 | Scalar | FFTS address on A2/A3; must be zero on A5 |
| `num_matrices` | Attribute | INT64 | Scalar | Must equal `M` |
| `norm_output` | Output | BF16 | `[T,NV,D]` | Gated RMSNorm output |
| `conv_state_out` | Output | BF16 | `[N,R+2,C]` | Updated Conv cache |
| `ssm_cache_out` | Output | FP32 | `[N*R,NV,D,D]` | Updated SSM cache |

The host supports `NV` values of `1/2/3/4/6/8/12/16/24/32/48/64` and requires
`NK > 0` and `NV % NK == 0`. `num_matrices` must be positive, be a multiple of
`NV`, and not exceed `T*NV`.

## Computation

Convolution and Q/K normalization are evaluated as follows:

```text
conv_t = SiLU(W0*x_(t-3) + W1*x_(t-2) + W2*x_(t-1) + W3*x_t)
q_hat  = q / sqrt(sum(q*q) + 1e-6)
k_hat  = k / sqrt(sum(k*k) + 1e-6)
```

The gates are:

```text
g    = -exp(a_log) * softplus(a + dt_bias, threshold=20)
beta = sigmoid(b)
```

The operator then executes chunk GDN and produces the final output:

```text
core = ChunkGdn(q_hat, k_hat, v, g, beta, initial_state)
out  = RMSNorm(core, norm_weight, eps=1e-6) * SiLU(z)
```

## Conv-state contract

- `conv_state_read_indices[b] < 0` means that the sequence has no history, so
  convolution starts from zero history;
- every write index must be in `[0,N)`;
- read and write indices may be equal or may use separate slots for a prefix
  fork;
- if a read slot in a multi-batch invocation is also written by another
  sequence, the kernel first preserves a compact snapshot to avoid a
  write-after-read hazard;
- when `R>1`, the state length remains `R+2`; the checkpoint tail beyond the
  three Conv-history rows is preserved according to the read/write-slot
  contract;
- when output and input buffers differ, the caller must pre-preserve slots not
  written by the invocation.

## SSM-cache contract

An SSM index is an absolute index into the first dimension of `ssm_cache`, not
a Conv slot ID. With a typical MTP layout, the checkpoint base of logical slot
`s` is `s*R`:

```text
ssm_read_index  = read_slot  * R
ssm_write_index = write_slot * R
```

Prefill writes only the final state of each sequence at
`ssm_state_write_indices[b]`; all other checkpoints remain unchanged. A
negative read index selects a zero state. Every write index must be valid and
unique within an invocation.

## Data-type contract

Public tensor boundaries are fixed at BF16/FP32, while chunk GDN explicitly
uses FP16 internally:

```text
GDN_PREFILL_COMPUTE_DTYPE = half
GDN_COMPUTE_DTYPE         = GDN_PREFILL_COMPUTE_DTYPE
ComputeT                  = GDN_COMPUTE_DTYPE
GDN_PUBLIC_DTYPE          = DTYPE_MIXED_QKV  // BF16
```

`mixed_qkv`, `minus_identity`, and the final output retain the public BF16 ABI,
while packed Q/K/V, beta, KKT, WY, and state intermediates in workspace use
FP16.

## Tiling, synchronization, and workspace

The host partitions token/channel work according to `T` and `C`, while
`cu_seqlens` isolates the ragged batch. The current implementation uses
128-token chunks; `num_matrices` determines the number of chunk states.

A2/A3 use FFTS to synchronize AIC/AIV stages, and the ACLNN wrapper obtains
`ffts_addr` from the runtime. A5 uses the complete physical MIX block grid and
PTO `SYNCALL<SyncCoreType::Mix>`. It does not consume an FFTS address, so the
wrapper sets the attribute to zero. The A5-specific path is isolated by
`GDN_PREFILL_TARGET_A5` and is not compiled into the A2/A3 branch.

This fused operator requires business workspace for the Conv snapshot, packed
compute QKV, gates, KKT, WY, H/O, and final-state intermediates. Workspace grows
with `T`, `NV`, `M`, and the block count. The caller must use the size returned
by the ACLNN GetWorkspaceSize API and must not reuse the `MegaChunkGdn`
workspace formula.

## Constraints

- `cu_seqlens` must be monotonic, start at zero, and end at T;
- read indices may repeat, but write indices must be unique;
- a negative read index means no initial state; a negative write index is
  invalid;
- `conv_state.shape[1]` must equal `ssm_cache.shape[0]/N + 2`;
- the caller is responsible for the contents of the masks and
  `minus_identity`;
- graph padding must allocate a valid sink write slot and must not rely on an
  out-of-range index;
- after state updates begin, execution must not fall back to a decomposed path
  that writes the same slot again.

## Invocation and build layout

The operator is invoked through `aclnnMegaGdnPrefillOp`. Its definition, host
tiling, and device kernel are located at:

- [`mega_gdn_prefill_op_def.cpp`](../../xllm_ops/mega_gdn_prefill_op/op_host/mega_gdn_prefill_op_def.cpp)
- [`mega_gdn_prefill_op_tiling.cpp`](../../xllm_ops/mega_gdn_prefill_op/op_host/mega_gdn_prefill_op_tiling.cpp)
- [`mega_gdn_prefill_op.cpp`](../../xllm_ops/mega_gdn_prefill_op/op_kernel/mega_gdn_prefill_op.cpp)
- [`gdn_prefill_frontend.h`](../../xllm_ops/mega_gdn_prefill_op/op_kernel/gdn_prefill_frontend.h)

The Python regression and performance scripts are located at:

- [`test_mega_gdn_prefill_op.py`](../../test/python_test/test_mega_gdn_prefill_op.py)
- [`benchmark_mega_gdn_prefill_op.py`](../../test/python_test/benchmark_mega_gdn_prefill_op.py)
