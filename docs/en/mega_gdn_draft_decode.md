<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnDraftDecode Operator Design

English | [简体中文](../zh/mega_gdn_draft_decode.md)

## Series documentation

[Core MegaChunkGdn](mega_chunk_gdn.md) |
[Fused Prefill](mega_gdn_prefill_op.md) | [Single-token Decode](mega_gdn_decode.md) |
[MTP Verify](mega_gdn_mtp_decode.md)

## Current enablement status

The current Qwen3.5 MTP draft path uses Full Attention and does not execute GDN
Draft Decode. Consequently, the current inference path does not dispatch
`MegaGdnDraftDecode`, and this path is inactive. The remainder of this document
records the interface and state contract of the reserved GDN Draft Decode
implementation for standalone validation or future model integration. The
table below indicates implementation support only; it does not mean that the
operator is enabled in the current model path.

## Product support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 training/inference products | Yes |
| Atlas A2 training/inference products | Yes |

## Overview

`MegaGdnDraftDecode` targets packed ragged decode for an MTP draft model. Each
sequence may contain zero, one, or two tokens in one invocation. The operator
fuses causal convolution, Q/K normalization, GDN recurrent updates, RMSNorm,
and the Z gate, and stores only the final Conv/SSM state of each non-empty
sequence.

Its key difference from `MegaGdnMtpDecode` is the state contract. Draft input
uses `qCuSeqLens` to describe ragged zero-to-two-token sequences and does not
store per-token checkpoints. MTP Verify consumes a fixed `K+1` tokens and
stores a checkpoint for every token.

## Symbols and shapes

```text
B  = batch size, 1 <= B <= 32
T  = total number of packed tokens, 1 <= T <= 2*B
D  = head dimension, fixed at 128
NK = number of key heads
NV = number of value heads
C  = (2 * NK + NV) * D
N  = number of state slots, 1 <= N <= 1024
```

All tensors use the ND format and must be contiguous.

| Name | Direction | Data type | Shape | Description |
| --- | --- | --- | --- | --- |
| `qkv` | Input | BF16 | `[T,C]` | Packed Q/K/V before convolution |
| `z` | Input | BF16 | `[T,NV,D]` | Output gate |
| `b` | Input | BF16 | `[T,NV]` | Beta-gate input |
| `a` | Input | BF16 | `[T,NV]` | Decay-gate input |
| `convWeight` | Input | BF16 | `[4,C]` | Depthwise-convolution weights |
| `convState` | Input | BF16 | `[N,3,C]` | Conv history |
| `aLog` | Input | FP32 | `[NV]` | Recurrent decay parameter |
| `dtBias` | Input | FP32 | `[NV]` | Recurrent decay bias |
| `ssmState` | Input | FP32 | `[N,NV,D,D]` | SSM state |
| `readStateIndices` | Input | INT32 | `[B]` | Read slot |
| `writeStateIndices` | Input | INT32 | `[B]` | Write slot |
| `qCuSeqLens` | Input | INT32 | `[B+1]` | Packed token boundaries |
| `stateValidityMask` | Input | BOOL | `[B]` | Whether to load the initial Conv/SSM state for each sequence |
| `normWeight` | Input | BF16 | `[D]` | RMSNorm weight |
| `flaSsmStateLayout` | Attribute | BOOL | Scalar | `true` stores `[K,V]`; `false` stores `[V,K]`; defaults to `true` |
| `convOut` | Output | BF16 | `[T,C]` | BF16 convolution output |
| `convStateOut` | Output | BF16 | `[N,3,C]` | Final Conv history of every non-empty sequence |
| `ssmStateOut` | Output | FP32 | `[N,NV,D,D]` | Final SSM state of every non-empty sequence |
| `out` | Output | BF16 | `[T,NV,D]` | Gated RMSNorm output |

Head geometry requires a power-of-two `NK` in `[1,16]`, `NV % NK == 0`, and
`1 <= NV/NK <= 4`.

## Packed-sequence contract

The caller must ensure that:

```text
qCuSeqLens[0] = 0
qCuSeqLens[B] = T
0 <= qCuSeqLens[b+1] - qCuSeqLens[b] <= 2
```

The host constrains only the aggregate size through `T <= 2*B`; it does not
read boundary values stored on the device. The caller must therefore validate
monotonicity, the final value, and every per-sequence length. A zero-length
sequence produces no output and does not write Conv or SSM state.

## State contract

`stateValidityMask[b]` controls both initial Conv and SSM state:

- `false`: ignore the contents of the read slot and use zero Conv history and
  zero SSM state;
- `true`: load both kinds of state from `readStateIndices[b]`;
- after a non-empty sequence completes, write its final state to
  `writeStateIndices[b]`;
- read and write slots may be equal; a prefix fork may read a shared slot and
  write a private slot;
- write slots must be unique within an invocation.

Conv state always stores the three most recent original `qkv` inputs, not
`convOut`. The SSM state remains in UB for all packed tokens of a sequence and
is written back only at the end of that sequence; no intermediate checkpoint
is generated.

## Numerical contract

Every valid token uses the same mathematics as the Decode path:

```text
conv   = SiLU(W0*h0 + W1*h1 + W2*h2 + W3*qkv)
q_hat  = q / sqrt(sum(q*q) + 1e-6) / sqrt(128)
k_hat  = k / sqrt(sum(k*k) + 1e-6)
g      = -exp(a_log) * softplus(a + dt_bias, threshold=20)
beta   = sigmoid(b)
H      = recurrent_update(H, q_hat, k_hat, v, g, beta)
out    = RMSNorm(BF16_RINT(readout), norm_weight, 1e-6) * SiLU(z)
```

Convolution accumulation, Q/K normalization, gates, SSM state, and RMSNorm use
FP32. The convolution hand-off, activation inputs, and final output use BF16
rounding boundaries.

## Tiling, synchronization, and workspace

Tiling keys are:

| Key | Condition |
| ---: | --- |
| 1 | FLA state layout |
| 1001 | non-FLA state layout |

Convolution is divided into `C/128` channel tiles, while recurrent work is
divided into `B*NV` head tasks. The actual task count is the maximum of the two.
A2/A3 use MIX AIC/AIV blocks; A5 uses AIV-only blocks.

A2/A3 use CANN system workspace. A5 uses a 4-KiB software-synchronization area
at the beginning of workspace and reserves 16 MiB of address space. This is
internal workspace returned by operator tiling, not an additional input tensor,
and it does not require a change to the public ABI of the Python test helper.

## Constraints

- at least 175,360 bytes of UB must be available;
- read and write indices must be in `[0,N)`; even when validity is false, the
  read index should remain a valid placeholder to avoid differences between
  future host and device validation;
- when input and output state buffers differ, the caller must preserve slots
  not written by the invocation;
- a zero-length padding sequence currently writes no state; write slots of all
  non-empty sequences must be unique;
- Draft and Verify operators must not write the same state slot concurrently.

## Invocation and build layout

The operator is invoked through `aclnnMegaGdnDraftDecode`. Its definition,
host tiling, and device kernel are located at:

- [`mega_gdn_draft_decode_def.cpp`](../../xllm_ops/mega_gdn_draft_decode/op_host/mega_gdn_draft_decode_def.cpp)
- [`mega_gdn_draft_decode_tiling.cpp`](../../xllm_ops/mega_gdn_draft_decode/op_host/mega_gdn_draft_decode_tiling.cpp)
- [`mega_gdn_draft_decode.cpp`](../../xllm_ops/mega_gdn_draft_decode/op_kernel/mega_gdn_draft_decode.cpp)
- [`mega_gdn_draft_decode_pto_kernel.h`](../../xllm_ops/mega_gdn_draft_decode/op_kernel/mega_gdn_draft_decode_pto_kernel.h)
