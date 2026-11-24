<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaKdaPrefill Operator Design

English | [简体中文](../zh/mega_kda_prefill.md)

## Related operators

[MegaKdaDecode](mega_kda_decode.md)

## Product support

| Product | Build target (`-c`) | Supported modes |
| --- | --- | --- |
| Ascend 910B (A2) | `ascend910b` | Prefill |
| Ascend 910C (A3) | `ascend910_93` | Prefill |
| Ascend 950 (A5) | `ascend950` | Prefill |

## Function

One device call processes packed variable-length sequences for GLM-5.3-Flash. It applies four-tap causal convolution, SiLU, Q/K normalization, vector gating and chunked KDA recurrence. It produces one attention output per token and the final Conv and SSM states of each nonempty sequence.

## Symbols and shapes

`B` is the request count, `T` the total packed token count and `H` the local head count (1–128). Q, K and V share head dimension `D=128`. The `C=3*H*D` channels pack all Q heads, then all K heads, then all V heads. `Ci` and `Co` are Conv input and output pool capacities. `Si` and `So` are the corresponding SSM capacities.

## Computation

For one token and one head, `x[0:4]` contains three raw QKV history rows and the current projection, oldest first. The state `S` is stored in `[V,K]` order and the decay vector broadcasts across its K dimension. Platform-specific rounding is described below.

```text
conv = SiLU(conv_bias + sum(conv_weight[j] * x[j] for j in 0..3))
q, k, v = split_QKV(conv)
q_hat = normalize(q) / sqrt(D)
k_hat = normalize(k)
g = -5 * sigmoid(exp(a_log) * (gate + gate_bias))
b = beta
S_decay = S * exp(g)[None, :]
delta = b * (v - S_decay @ k_hat)
S = S_decay + delta[:, None] * k_hat[None, :]
output = BF16(S @ q_hat)
```

Pass the FP32 weight `torch.sigmoid(beta_raw.float())` as `beta`. Prefill uses this value directly in the state update.

## Parameters

All tensors are contiguous ND tensors on the same NPU.

| Name | Direction | Dtype | Shape | Description |
| --- | --- | --- | --- | --- |
| `qkv` | Input | BF16 | `[T,C]` | Raw Q/K/V projection |
| `gate` | Input | BF16 | `[T,H,D]` | Raw vector gate projection |
| `beta` | Input | FP32 | `[T,H]` | Weight after sigmoid |
| `a_log` | Input | FP32 | `[H]` | Gate scale parameter per head |
| `gate_bias` | Input | FP32 | `[H,D]` | Vector gate bias |
| `conv_weight` | Input | BF16 | `[4,C]` | Four depthwise taps, oldest to newest |
| `conv_bias` | Input | BF16 | `[C]` | Optional, pass `None` for zero bias |
| `conv_state_in` | Input | BF16 | `[Ci,3,C]` | Raw QKV history input pool |
| `ssm_state_in` | Input | FP32 | `[Si,H,D,D]` | SSM input pool, trailing dimensions [V,K] |
| `cu_seqlens` | Input | INT32 | `[B+1]` | Nondecreasing packed offsets from 0 to T |
| `conv_read_indices` | Input | INT32 | `[B]` | Conv source slot per request |
| `conv_write_indices` | Input | INT32 | `[B]` | Conv destination slot per request |
| `ssm_read_indices` | Input | INT32 | `[B]` | Initial SSM slot per sequence |
| `ssm_write_indices` | Input | INT32 | `[B]` | Final SSM slot per sequence |
| `conv_output_slots` | Attribute | INT | `scalar` | Co, inferred by the binding from the output |
| `ssm_output_slots` | Attribute | INT | `scalar` | So, inferred by the binding from the output |
| `ffts_addr` | Attribute | INT | `scalar` | Synchronization address, queried by the binding on 910B/910C and set to zero on 950 |
| `output` | Output | BF16 | `[T,H,D]` | Attention output per token |
| `conv_state_out` | Output | BF16 | `[Co,3,C]` | Updated raw QKV history |
| `ssm_state_out` | Output | FP32 | `[So,H,D,D]` | Updated [V,K] state |

## State contract

Conv and SSM are addressed independently. Read slot `-1` starts that history or state at zero and the sequence is processed normally. For each nonempty sequence, the Conv destination receives the last three raw QKV rows, including prior history for short sequences. The SSM destination receives the state after the last token. A write slot of `-1` skips that state write. Empty sequences write no state and untouched destinations retain their contents.

Slot 0 is valid and multiple sequences may share a source slot. Nonempty sequences must use unique active destinations within each output pool. An MTP caller supplies its most recent three Conv history rows for prefill and retains additional speculative history and checkpoints separately.

The caller allocates all three outputs. Their storage must be disjoint from all inputs and from one another. The input and output pools may have different capacities. The device reads offsets, slot tables and any accepted counts at execution time. They may change between graph replays and must remain stable during a call.

## Numerical contract

Both kernel paths convert BF16 Conv inputs and history directly to FP32, accumulate four taps, apply FP32 SiLU and round the activation to BF16. Q/K normalization implements `x / sqrt(sum(x*x) + 1e-6)` and rounds the result to BF16. The 910B/910C path uses an FP32 reciprocal square root. The 950 implementation scales large rows to handle their range. Gate prefix sums, beta and the public SSM state use FP32.

## Tiling design

The 910B/910C path uses 64-token chunks for all supported head counts. The general 950 path uses 128-token chunks. Both compute state and output through KKT, inverse, WY and chunk H/O stages, transposing between internal KV and public VK state layouts. 950 uses direct recurrence for `B=1, H=8, 1≤T≤4`.

The 910B/910C Conv frontend uses 16-, 32- or 64-row tiles. Its first three tokens per sequence use the history-aware path. Subsequent tokens use tiles and the tail stores valid rows only. The gate prefix scan runs sequentially in FP32. MTE3 stores complete before the buffer is reused.

## Constraints

Require `1≤T≤131072`, `1≤B≤1024`, and 1–1,048,576 slots per state pool. Computed workspace is limited to 32 GiB. Actual allocation depends on available HBM. The device checks offsets, slot ranges and duplicate destinations before publishing any output. Invalid metadata returns without publication, so callers should validate it before submission.

## Invocation and build

Build the default operator set from the repository root in a configured CANN environment. Selecting only KDA with `--op-name` in a clean build leaves the upstream registration step dependent on missing `inner` and `exc` INI files. The Python test binding infers the output pool capacities from the supplied tensors.

```bash
bash build.sh
```

To target 910C explicitly, run `bash build.sh -c ascend910_93`. The KDA package was cross-compiled for `Ascend910_9391` on x86_64 Linux with CANN 9.0 and installed successfully in an isolated directory. Accuracy and performance tests on 910C hardware are pending.

The ACLNN entry points are `aclnnMegaKdaPrefillGetWorkspaceSize` and `aclnnMegaKdaPrefill`. The example assumes the current NPU is selected and the contiguous inputs and output buffers follow the table above.

```python
import torch
import custom_ops_lib

beta = torch.sigmoid(beta_raw.float())
output, conv_out, ssm_out = custom_ops_lib.mega_kda_prefill(
    qkv, gate, beta, a_log, gate_bias, conv_weight, conv_bias,
    conv_state_in, ssm_state_in, cu_seqlens,
    conv_read_indices, conv_write_indices, ssm_read_indices, ssm_write_indices,
    output, conv_state_out, ssm_state_out,
)
```

In GLM integrations that expose the KDA backend switch, `XLLM_KDA_KERNEL=mega_kda` selects fused operators for supported layouts. The default backend is `fla_npu`. Non-MTP engines (verify width 1) can keep FLA prefill with `XLLM_KDA_PREFILL_BACKEND=fla`. In that configuration, `XLLM_KDA_DECODE_NORM=public` selects 910B/910C compatibility decode (mode 3). The MTP integration maps supported V3 snapshots to mode 1 checkpoint tables and routes other layouts to FLA V3 before execution.

[Operator definition](../../xllm_ops/mega_kda_prefill/op_host/mega_kda_prefill_def.cpp) · [Host tiling](../../xllm_ops/mega_kda_prefill/op_host/mega_kda_prefill_tiling.cpp) · [Device kernel](../../xllm_ops/mega_kda_prefill/op_kernel/mega_kda_prefill.cpp) · [Python binding](../../test/python_test/mega_kda_ops.cpp) · [NPU tests](../../test/python_test/test_mega_kda_prefill.py)
