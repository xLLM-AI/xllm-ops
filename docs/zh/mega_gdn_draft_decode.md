<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnDraftDecode 算子设计

[English](../en/mega_gdn_draft_decode.md) | 简体中文

## 系列文档

[核心 MegaChunkGdn](mega_chunk_gdn.md) |
[融合 Prefill](mega_gdn_prefill_op.md) | [单 token Decode](mega_gdn_decode.md) |
[MTP Verify](mega_gdn_mtp_decode.md)

## 当前启用状态

当前 Qwen3.5 的 MTP draft 路径采用 Full Attention，不执行 GDN Draft Decode，
因此现有推理链路不会调度 `MegaGdnDraftDecode`，该路径当前不生效。本文其余内容
记录预留 GDN Draft Decode 实现的接口和 state 合同，供独立验证或后续模型适配
使用。下表仅表示算子实现具备对应架构支持，不表示当前模型链路已经启用。

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`MegaGdnDraftDecode` 面向 MTP draft 模型的 packed ragged decode。一次调用中
每条序列可以包含 0、1 或 2 个 token；算子融合 causal convolution、Q/K
归一化、GDN recurrent 更新、RMSNorm 与 Z gate，并且只保存每条非空序列的
最终 Conv/SSM state。

它与 `MegaGdnMtpDecode` 的关键区别是：Draft 输入由 `qCuSeqLens` 描述 ragged
0～2 token，不保存逐 token checkpoint；MTP Verify 输入为固定 `K+1` token，
并为每个 token 保存 checkpoint。

## 符号与 Shape

```text
B  = batch size，1 <= B <= 32
T  = packed token 总数，1 <= T <= 2*B
D  = head dimension，固定为 128
NK = key head 数
NV = value head 数
C  = (2 * NK + NV) * D
N  = state slot 数，1 <= N <= 1024
```

所有 Tensor 使用 ND 格式且必须 contiguous。

| 参数名 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `qkv` | 输入 | BF16 | `[T,C]` | packed 卷积前 Q/K/V |
| `z` | 输入 | BF16 | `[T,NV,D]` | 输出 gate |
| `b` | 输入 | BF16 | `[T,NV]` | beta gate 输入 |
| `a` | 输入 | BF16 | `[T,NV]` | decay gate 输入 |
| `convWeight` | 输入 | BF16 | `[4,C]` | depthwise Conv 权重 |
| `convState` | 输入 | BF16 | `[N,3,C]` | Conv history |
| `aLog` | 输入 | FP32 | `[NV]` | recurrent decay 参数 |
| `dtBias` | 输入 | FP32 | `[NV]` | recurrent decay bias |
| `ssmState` | 输入 | FP32 | `[N,NV,D,D]` | SSM state |
| `readStateIndices` | 输入 | INT32 | `[B]` | read slot |
| `writeStateIndices` | 输入 | INT32 | `[B]` | write slot |
| `qCuSeqLens` | 输入 | INT32 | `[B+1]` | packed token 边界 |
| `stateValidityMask` | 输入 | BOOL | `[B]` | 是否加载该序列的初始 Conv/SSM state |
| `normWeight` | 输入 | BF16 | `[D]` | RMSNorm 权重 |
| `flaSsmStateLayout` | 属性 | BOOL | 标量 | `true` 为 `[K,V]`，`false` 为 `[V,K]`；默认 `true` |
| `convOut` | 输出 | BF16 | `[T,C]` | BF16 Conv 输出 |
| `convStateOut` | 输出 | BF16 | `[N,3,C]` | 每条非空序列的最终 Conv history |
| `ssmStateOut` | 输出 | FP32 | `[N,NV,D,D]` | 每条非空序列的最终 SSM state |
| `out` | 输出 | BF16 | `[T,NV,D]` | gated RMSNorm 输出 |

Head 几何要求 `1 <= NK <= 16` 且 `NK` 为 2 的幂，`NV % NK == 0`，
并且 `1 <= NV/NK <= 4`。

## Packed 序列合同

调用侧必须保证：

```text
qCuSeqLens[0] = 0
qCuSeqLens[B] = T
0 <= qCuSeqLens[b+1] - qCuSeqLens[b] <= 2
```

Host 只通过 `T <= 2*B` 约束总体规模，不读取 device 上的边界值，因此单调性、
末值和每条序列长度必须由调用侧校验。长度为 0 的序列不产生输出，也不写
Conv/SSM state。

## State 合同

`stateValidityMask[b]` 同时控制 Conv 和 SSM 初始 state：

- `false`：忽略 read slot 的内容，使用零 Conv history 和零 SSM state；
- `true`：从 `readStateIndices[b]` 加载两类 state；
- 非空序列结束后，将最终 state 写入 `writeStateIndices[b]`；
- read/write slot 可以相同；prefix fork 可以读 shared slot、写 private slot；
- 同一调用中的 write slot 必须唯一。

Conv state 始终保存最近三个原始 `qkv` 输入，而不是 `convOut`。SSM state 在一条
序列的所有 packed token 间保留于 UB，只在序列结束时写回，不生成中间
checkpoint。

## 数值合同

每个有效 token 的数学与 Decode 路径一致：

```text
conv   = SiLU(W0*h0 + W1*h1 + W2*h2 + W3*qkv)
q_hat  = q / sqrt(sum(q*q) + 1e-6) / sqrt(128)
k_hat  = k / sqrt(sum(k*k) + 1e-6)
g      = -exp(a_log) * softplus(a + dt_bias, threshold=20)
beta   = sigmoid(b)
H      = recurrent_update(H, q_hat, k_hat, v, g, beta)
out    = RMSNorm(BF16_RINT(readout), norm_weight, 1e-6) * SiLU(z)
```

Conv 累加、Q/K 归一化、gate、SSM state 和 RMSNorm 使用 FP32；Conv hand-off、
激活输入和最终输出使用 BF16 舍入点。

## Tiling、同步与 Workspace

Tiling key：

| Key | 条件 |
| ---: | --- |
| 1 | FLA state layout |
| 1001 | non-FLA state layout |

Conv 按 `C/128` 个 channel tile 分配，recurrent 按 `B*NV` 个 head task 分配，
实际 task 数取两者最大值。A2/A3 使用 MIX AIC/AIV block；A5 使用 AIV-only
block。

A2/A3 使用 CANN system workspace。A5 在 workspace 开头使用 4 KiB 的
software-sync 区，并保留 16 MiB 地址空间；这是算子 tiling 返回的内部
workspace，不是额外输入 tensor，也不要求修改 Python test helper 的公共 ABI。

## 约束说明

- UB 必须不少于 175,360B；
- read/write index 必须位于 `[0,N)`；validity 为 false 时 read 值仍应提供合法
  占位，避免未来 host/device 校验差异；
- 输入和输出 state buffer 分离时，调用侧负责保留未写 slot；
- 长度为 0 的 padding 序列当前不写 state；所有非空序列的 write slot 必须唯一；
- 不允许 Draft 和 Verify 两个算子并发写同一 state slot。

## 调用与构建

算子通过 ACLNN 接口 `aclnnMegaGdnDraftDecode` 调用。主要实现位于：

- [`mega_gdn_draft_decode_def.cpp`](../../xllm_ops/mega_gdn_draft_decode/op_host/mega_gdn_draft_decode_def.cpp)
- [`mega_gdn_draft_decode_tiling.cpp`](../../xllm_ops/mega_gdn_draft_decode/op_host/mega_gdn_draft_decode_tiling.cpp)
- [`mega_gdn_draft_decode.cpp`](../../xllm_ops/mega_gdn_draft_decode/op_kernel/mega_gdn_draft_decode.cpp)
- [`mega_gdn_draft_decode_pto_kernel.h`](../../xllm_ops/mega_gdn_draft_decode/op_kernel/mega_gdn_draft_decode_pto_kernel.h)
