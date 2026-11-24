<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaKdaDecode Operator Design

English | [简体中文](../zh/mega_kda_decode.md)

## Related operators

[MegaKdaPrefill](mega_kda_prefill.md)

## Product support

| Product | Build target (`-c`) | Supported modes |
| --- | --- | --- |
| Ascend 910B (A2) | `ascend910b` | Ordinary decode (0), MTP verify (1), compatibility decode (3) |
| Ascend 910C (A3) | `ascend910_93` | Ordinary decode (0), MTP verify (1), compatibility decode (3) |
| Ascend 950 (A5) | `ascend950` | Ordinary decode (0), MTP verify (1) |

## Function

One device call applies the four-tap causal convolution, SiLU, Q/K normalization and KDA recurrence for GLM-5.3-Flash. Ordinary decode advances each request by one token. MTP verify restores the accepted checkpoint from the previous step and writes a checkpoint for each new query token. The caller supplies the accepted count.

## Symbols and shapes

`B` is the request count, `T` the total packed token count and `H` the local head count (1–128). Q, K and V share head dimension `D=128`. The `C=3*H*D` channels pack all Q heads, then all K heads, then all V heads. `Ci` and `Co` are Conv input and output pool capacities. `Si` and `So` are the corresponding SSM capacities.

`M=max_query_tokens` is the query capacity per request and `L` the number of history rows per Conv slot. Ordinary and compatibility decode require `T=B, M=1, L=3`. In MTP each request may have 0 to `M` query tokens, with `T>0, 1≤M≤17, L=M+2`. For example, MTP=3 verifies the current token and three draft tokens with `M=4, L=6`.

## Computation

For one token and one head, `x[0:4]` contains three raw QKV history rows and the current projection, oldest first. The state `S` is stored in `[V,K]` order and the decay vector broadcasts across its K dimension. Platform-specific rounding is described below.

```text
conv = SiLU(conv_bias + sum(conv_weight[j] * x[j] for j in 0..3))
q, k, v = split_QKV(conv)
q_hat = normalize(q) / sqrt(D)
k_hat = normalize(k)
g = -5 * sigmoid(exp(a_log) * (gate + gate_bias))
b = sigmoid(float32(beta))
S_decay = S * exp(g)[None, :]
delta = b * (v - S_decay @ k_hat)
S = S_decay + delta[:, None] * k_hat[None, :]
output = BF16(S @ q_hat)
```

Pass the BF16 projection **before** sigmoid as `beta`. Ordinary decode, MTP verify and compatibility decode all convert this input to FP32 and apply sigmoid once in the kernel.

## Parameters

All tensors are contiguous ND tensors on the same NPU.

| Name | Direction | Dtype | Shape | Description |
| --- | --- | --- | --- | --- |
| `qkv` | Input | BF16 | `[T,C]` | Raw Q/K/V projection |
| `gate` | Input | BF16 | `[T,H,D]` | Raw vector gate projection |
| `beta` | Input | BF16 | `[T,H]` | Projection before sigmoid |
| `a_log` | Input | FP32 | `[H]` | Gate scale parameter per head |
| `gate_bias` | Input | FP32 | `[H,D]` | Vector gate bias |
| `conv_weight` | Input | BF16 | `[4,C]` | Four depthwise taps, oldest to newest |
| `conv_bias` | Input | BF16 | `[C]` | Optional, pass `None` for zero bias |
| `conv_state_in` | Input | BF16 | `[Ci,L,C]` | Raw QKV history input pool |
| `ssm_state_in` | Input | FP32 | `[Si,H,D,D]` | SSM input pool, trailing dimensions [V,K] |
| `cu_seqlens` | Input | INT32 | `[B+1]` | Nondecreasing packed offsets from 0 to T |
| `conv_read_indices` | Input | INT32 | `[B]` | Conv source slot per request |
| `conv_write_indices` | Input | INT32 | `[B]` | Conv destination slot per request |
| `ssm_read_indices` | Input | INT32 | `[B,M]` | Candidate checkpoint input slots per request |
| `ssm_write_indices` | Input | INT32 | `[B,M]` | Checkpoint destination per query token |
| `num_accepted_tokens` | Input | INT32 | `[B]` | Required in MTP (1 to M). Pass `None` in modes 0 and 3 |
| `mode` | Attribute | INT | `scalar` | 0, 1 or 3, default 0 |
| `max_query_tokens` | Attribute | INT | `scalar` | Query capacity M, default 1 |
| `conv_output_slots` | Attribute | INT | `scalar` | Co, inferred by the binding from the output |
| `ssm_output_slots` | Attribute | INT | `scalar` | So, inferred by the binding from the output |
| `output` | Output | BF16 | `[T,H,D]` | Attention output per token |
| `conv_state_out` | Output | BF16 | `[Co,L,C]` | Updated raw QKV history |
| `ssm_state_out` | Output | FP32 | `[So,H,D,D]` | Updated [V,K] state |

## State contract

**Ordinary decode (modes 0 and 3).** Each request reads three rows from `conv_read_indices[b]` and the SSM state from `ssm_read_indices[b,0]`. Its Conv destination receives the last two history rows and the current raw QKV. Its SSM destination receives the updated state. Use `cu_seqlens=[0,1,…,B]`.

**MTP verify (mode 1).** Let `a=num_accepted_tokens[b]` and `n=cu_seqlens[b+1]-cu_seqlens[b]`. Slice ends below are exclusive. Empty requests do no work.

```text
initial_ssm = ssm_state_in[ssm_read_indices[b, a-1]]
history = conv_state_in[conv_read_indices[b]]
initial_conv = history[a-1:a+2]
conv_state_out[conv_write_indices[b], :n+2] = concat(history[a:a+2], query_qkv)
ssm_state_out[ssm_write_indices[b, j]] = state_after_query_token_j  (0 ≤ j < n)
```

The writes above apply to valid destinations. The unused Conv tail keeps its prior contents. With MTP=3 and `a=2`, restore SSM from column 1 (zero-based) and Conv from history rows 1, 2 and 3. A one-token verification still uses mode 1.

**Slots.** Slot 0 is valid. If either the Conv or the selected SSM read slot is `-1`, the request is inactive and its output rows retain their previous contents. For a cold start, supply real zero-initialized read slots. A write slot of `-1` skips only that write. Read slots may be shared. Active destinations within each output pool must be unique, including per-token checkpoints.

The caller allocates all three outputs. Their storage must be disjoint from all inputs and from one another. The input and output pools may have different capacities. The device reads offsets, slot tables and any accepted counts at execution time. They may change between graph replays and must remain stable during a call.

## Numerical contract

Four-tap convolution accumulates in FP32. SiLU executes in FP32 and its output rounds to BF16 before the Q/K/V split. The input conversion and pre-SiLU rounding depend on the platform and mode.

| Platform and mode | Conv input and history | Accumulator before SiLU |
| --- | --- | --- |
| 910B/910C, 0 / 1 | BF16 → FP16 → FP32 | FP32 → BF16 → FP32 |
| 910B/910C, 3 | BF16 → FP32 | FP32 → BF16 → FP32 |
| 950, 0 | BF16 → FP32 | FP32 |
| 950, 1 | BF16 → FP16 → FP32 | FP32 → BF16 → FP32 |

The 910B/910C path computes the FP32 row norm `sqrt(sum(x*x))` and multiplies by its reciprocal for positive norms. Zero rows stay zero. 950 uses `x / sqrt(sum(x*x) + 1e-6)`. Both scale Q after normalization. Beta sigmoid, vector decay and the SSM state use FP32 in every mode. Modes 0 and 1 compute the gate as `-5 / denominator`. Mode 3 on 910B/910C takes the reciprocal first, then multiplies by `-5`.

## Tiling design

Host selects the tiling key from the explicit mode. Ordinary decode uses key 1, MTP verify uses key 2 and 910B/910C compatibility decode uses key 3. Every request in a launch uses the same mode. On 910B/910C, AIV tasks split by request, head and V rows. A task owns 64 rows when `2*B*H` covers all launched AIV blocks, otherwise 32. These variants use 120,032 and 62,368 bytes of explicit UB respectively. Only the first V shard writes Conv state. There is no user workspace. With 64-row shards, modes 0 and 3 reduce two groups of 32 rows and mode 1 reduces one group of 64.

## Constraints

Host validates mode, capacity, dtype and static shapes. The scheduler validates device index ranges and unique write slots. Ordinary decode also requires two-dimensional SSM slot tables `[B,1]`. Use prepacked Conv weights in `[4,C]` order. Weights in `[C,1,4]` order can be converted once after loading and TP sharding.

## Invocation and build

Build the default operator set from the repository root in a configured CANN environment. Selecting only KDA with `--op-name` in a clean build leaves the upstream registration step dependent on missing `inner` and `exc` INI files. The Python test binding infers the output pool capacities from the supplied tensors.

```bash
bash build.sh
```

To target 910C explicitly, run `bash build.sh -c ascend910_93`. The KDA package was cross-compiled for `Ascend910_9391` on x86_64 Linux with CANN 9.0 and installed successfully in an isolated directory. Accuracy and performance tests on 910C hardware are pending.

The ACLNN entry points are `aclnnMegaKdaDecodeGetWorkspaceSize` and `aclnnMegaKdaDecode`. The example assumes the current NPU is selected and the contiguous inputs and output buffers follow the table above.

```python
import custom_ops_lib

output, conv_out, ssm_out = custom_ops_lib.mega_kda_decode(
    qkv, gate, beta_raw.contiguous(), a_log, gate_bias, conv_weight, conv_bias,
    conv_state_in, ssm_state_in, cu_seqlens,
    conv_read_indices, conv_write_indices, ssm_read_indices, ssm_write_indices,
    num_accepted_tokens=None, mode=0, max_query_tokens=1,
    output=output, conv_state_out=conv_state_out, ssm_state_out=ssm_state_out,
)
```

The example uses ordinary decode. For MTP=3 pass `mode=1, max_query_tokens=4` and a device `num_accepted_tokens` tensor. Both SSM tables have shape `[B,4]` and the Conv history length is 6.

In GLM integrations that expose the KDA backend switch, `XLLM_KDA_KERNEL=mega_kda` selects fused operators for supported layouts. The default backend is `fla_npu`. Non-MTP engines (verify width 1) can keep FLA prefill with `XLLM_KDA_PREFILL_BACKEND=fla`. In that configuration, `XLLM_KDA_DECODE_NORM=public` selects 910B/910C compatibility decode (mode 3). The MTP integration maps supported V3 snapshots to mode 1 checkpoint tables and routes other layouts to FLA V3 before execution.

[Operator definition](../../xllm_ops/mega_kda_decode/op_host/mega_kda_decode_def.cpp) · [Host tiling](../../xllm_ops/mega_kda_decode/op_host/mega_kda_decode_tiling.cpp) · [Device kernel](../../xllm_ops/mega_kda_decode/op_kernel/mega_kda_decode.cpp) · [Python binding](../../test/python_test/mega_kda_ops.cpp) · [NPU tests](../../test/python_test/test_mega_kda_decode.py)
