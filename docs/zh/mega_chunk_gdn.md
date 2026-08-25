<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaChunkGdn 算子设计

[English](../en/mega_chunk_gdn.md) | 简体中文

## 系列文档

[融合 Prefill](mega_gdn_prefill_op.md) |
[单 token Decode](mega_gdn_decode.md) | [Draft Decode](mega_gdn_draft_decode.md) |
[MTP Verify](mega_gdn_mtp_decode.md)

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`MegaChunkGdn` 是 Prefill 路径的 chunk GDN 核心算子。调用侧需要先完成
causal convolution、Q/K L2 归一化以及 gate 准备；算子以 128 token 为一个
chunk，融合 cumulative decay、KKT、下三角求逆、WY 变换、state 递推和输出
计算。

需要同时融合 Conv、cache 更新和 gated RMSNorm 时，应使用
`MegaGdnPrefillOp`。

## 符号与 Shape

```text
B  = cu_seqlens 描述的序列数
T  = packed token 总数
D  = head dimension，固定为 128
Ck = chunk size，固定为 128
NK = key head 数
NV = value head 数
M  = NV * sum_i ceil(sequence_length_i / Ck)
```

输入使用带固定 batch 轴的 packed BSND 布局：

| 参数名 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `q` | 输入 | FP16/BF16 | `[1,T,NK,D]` | 已完成 L2 归一化的 query |
| `k` | 输入 | FP16/BF16 | `[1,T,NK,D]` | 已完成 L2 归一化的 key |
| `v` | 输入 | FP16/BF16 | `[1,T,NV,D]` | value |
| `g` | 输入 | FP32 | `[1,T,NV]` | log decay gate |
| `beta` | 输入 | FP16/BF16 | `[1,T,NV]` | update gate |
| `maskLower` | 输入 | FP32 | `[128,128]` | 严格下三角 mask |
| `maskFull` | 输入 | FP32 | `[128,128]` | 含对角线的下三角 mask |
| `minusIdentity` | 输入 | FP16/BF16 | `[128,128]` | 对角线为 `-1` 的矩阵 |
| `cuSeqlens` | 输入 | INT32 | `[B+1]` | packed 序列边界，首项为 0、末项为 T |
| `initialState` | 输入 | FP16/BF16 | `[B,NV,D,D]` | 每条序列的初始 state |
| `numMatrices` | 属性 | INT64 | 标量 | 应为 `M`；Python binding 要求大于 0 |
| `hasInitialState` | 属性 | BOOL | 标量 | 是否加载 `initialState` |
| `fftsAddr` | 属性 | INT64 | 标量 | A2/A3 FFTS 地址；A5 不使用 |

Q/K/V/beta 必须使用相同的 compute dtype。Host 要求 `1 <= NV <= 64`、
`NK > 0`、`NV % NK == 0`，并要求 Q/V 的最后一维为 128。

## 输出说明

ACLNN 原始接口输出 12 个 tensor。除 `out` 和 `finalState` 外，其余为融合
流水内部结果；高层 wrapper 通常只暴露其中一部分。

| 参数名 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- |
| `out` | 与 V 相同 | `[1,T,NV,D]` | 未应用外部 `D^-0.5` scale 的核心输出 |
| `gSum` | FP32 | `[1,T,NV]` | 分序列、分 chunk 的累计 gate |
| `gT` | FP32 | `[NV,T]` | 转置 gate |
| `betaT` | compute dtype | `[NV,T]` | 转置 beta |
| `a` | compute dtype | `[1,T,NV,128]` | chunk 下三角矩阵 |
| `aInvF32` | FP32 | `[1,T,NV,128]` | FP32 求逆中间量 |
| `aInv` | compute dtype | `[1,T,NV,128]` | 下三角逆矩阵 |
| `w` | compute dtype | `[1,T,NV,D]` | WY 中间量 W |
| `u` | compute dtype | `[1,T,NV,D]` | WY 中间量 U |
| `h` | compute dtype | `[M,D,D]` | 每个 chunk 的 state |
| `vNew` | compute dtype | `[1,T,NV,D]` | 校正后的 value |
| `finalState` | compute dtype | `[B,NV,D,D]` | 每条序列的最终 state |

## 计算概要

每个 chunk 内先计算累计 gate，并构造严格下三角系统：

```text
G_t       = cumsum(g)_t
decay_ij  = exp(G_i - G_j), i >= j
A         = -tril((beta * K) @ K^T * decay, diagonal=-1)
A_inv     = inverse(I - A)
```

随后通过 WY 形式求得校正后的 value，并按 chunk 更新 recurrent state：

```text
V_new = chunk_update(A_inv, K, V, beta, G)
O_i   = recurrent_readout(Q_i, K_i, V_new_i, H_previous, G_i)
H_i   = recurrent_state_update(H_previous, K_i, V_new_i, G_i)
```

不同序列之间由 `cuSeqlens` 隔离；最后一个不足 128 token 的 chunk 在内部补齐，
不会越过序列边界。

## State 合同

- `hasInitialState=false` 时，kernel 使用零 state，不读取 `initialState` 的值；
- `hasInitialState=true` 时，`initialState[b]` 对应第 b 条 packed 序列；
- `finalState[b]` 是该序列最后一个有效 token 后的 state；
- state 的 head 轴按 value head 展开，Q/K 按 `NV/NK` 映射到 value head；
- 调用侧提供的 `numMatrices` 必须与 `cuSeqlens` 描述的 chunk 总数一致。

## 数据类型合同

共享 kernel 使用独立的 `ComputeT` 表示计算类型：

```text
GDN_COMPUTE_DTYPE 默认值 = DTYPE_Q
ComputeT = GDN_COMPUTE_DTYPE
```

普通 `MegaChunkGdn` 因而保持 FP16 输入走 FP16、BF16 输入走 BF16 的现有行为，
但公共 Q 类型和内部计算类型在代码语义上已经分离。融合 Prefill 可以显式覆盖
`GDN_COMPUTE_DTYPE`，而不把内部类型伪装成公共 `DTYPE_Q`。

## Tiling、同步与 Workspace

`numMatrices` 未显式给出时，Host 使用 `ceil(T/128) * NV` 作为兼容 fallback；
packed 多序列调用仍应由调用侧传入按每条序列分别取整后的准确值。

A2/A3 使用 FFTS 完成 MIX AIC/AIV 的阶段同步。A5 使用完整 1:2 MIX block 和
PTO `SYNCALL<SyncCoreType::Mix>`；Host 将 block 数限制为 value head 数的因子，
使同一 head 的 chunk 保持稳定 owner。

用户 workspace 包含 KKT、WY、H/O 等 per-block 临时区以及 H workspace 的
16 MiB 对齐区和 8 MiB phase 余量，最终再加 CANN system workspace。调用侧
必须使用 `aclnnMegaChunkGdnGetWorkspaceSize` 返回的大小，不能自行按输出 shape
估算。

## 调用与构建

算子通过 ACLNN 接口 `aclnnMegaChunkGdn` 调用。主要实现位于：

- [`mega_chunk_gdn_def.cpp`](../../xllm_ops/mega_chunk_gdn/op_host/mega_chunk_gdn_def.cpp)
- [`mega_chunk_gdn_tiling.cpp`](../../xllm_ops/mega_chunk_gdn/op_host/mega_chunk_gdn_tiling.cpp)
- [`mega_chunk_gdn.cpp`](../../xllm_ops/mega_chunk_gdn/op_kernel/mega_chunk_gdn.cpp)

Python wrapper 和回归测试位于：

- [`custom_ops.py`](../../test/python_test/custom_ops.py)
- [`test_mega_chunk_gdn.py`](../../test/python_test/test_mega_chunk_gdn.py)
