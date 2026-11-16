<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnMtpDecodeVarlen 算子介绍

[English](../en/mega_gdn_mtp_decode_varlen.md) | 简体中文

## 功能说明

`MegaGdnMtpDecodeVarlen` 是面向 Ascend A3 的 `MegaGdnMtpDecode` 变长前缀版本。
各请求的有效 token 按 packed 布局存放，算子复用原有融合 causal convolution、
Q/K 归一化、gate、FP32 recurrent 更新、逐 token checkpoint 和 RMSNorm/SiLU
实现。原有 dense 算子及其 ABI 保持不变。

## Shape 与参数

本文使用以下符号：

- `S`：固定的 checkpoint 步长，取值范围为 `[2,17]`。
- `B`：请求数量，取值范围为 `[1,128]`。
- `T`：packed 图输入的 token 容量。
- `N`：状态槽位数量。
- `D=128`；`NK` 和 `NV` 分别为 key head 和 value head 数量。
- `C=(2*NK+NV)*D`。

各参数的数据类型和 Shape 如下：

| 参数 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- |
| `qkv` | BF16 | `[T,C]` | packed Q/K/V 输入 |
| `z` | BF16 | `[T,NV,D]` | 输出 gate |
| `a`、`b` | BF16 | `[T,NV]` | recurrent gate 输入 |
| `convWeight` | BF16 | `[4,C]` | 宽度为 4 的卷积权重 |
| `normWeight` | BF16 | `[D]` | RMSNorm 权重 |
| `aLog`、`dtBias` | FP32 | `[NV]` | recurrent 参数 |
| `convState` / `convStateOut` | BF16 | `[N,S+2,C]` | 输入与输出共享存储的卷积状态 |
| `ssmState` / `ssmStateOut` | FP32 | `[N*S,NV,D,D]` | 输入与输出共享存储的 SSM checkpoints |
| `stateIndices` | INT32 | `[B]` | Device 上的请求状态槽位索引 |
| `numAcceptedTokens` | INT32 | `[B]` | Device 上的上一轮接受数量 |
| `cuSeqLens` | INT32 | `[B+1]` | Device 上单调不减的 packed 前缀端点 |
| `convOut` | BF16 | `[T,C]` | 卷积输出 |
| `out` | BF16 | `[T,NV,D]` | 算子输出 |

## 变长前缀与状态槽位约束

当前第 `i` 个请求的有效前缀长度为 `cuSeqLens[i+1]-cuSeqLens[i]`，取值范围为
`[0,S]`。所有端点均在 `[0,T]` 内；第一个端点可以非零，因此多个请求分块可以
访问同一份 packed 输入。

零长度请求不读取或修改状态。算子只写入被引用的 packed 输出行；调用方负责初始化
padding 行，不相交的请求分块可以共享输出缓冲区。所有分块中的有效请求必须各自
独占一个 `stateIndices` 槽位。零长度请求不访问状态，因此可以共享索引。

上一轮接受数量 `accepted=numAcceptedTokens[i]` 的取值范围始终为 `[1,S]`，
不依赖本轮前缀长度。恢复状态时：

- SSM 从 `stateIndices[i]*S+accepted-1` 处恢复。
- 卷积历史从同一槽位的 `accepted-1:accepted+2` 行恢复。

对本轮每个有效 token `t`，写入 `stateIndices[i]*S+t` 对应的 SSM checkpoint；
卷积状态的 `0:2` 行保留上一轮已接受前缀末尾的历史，本轮原始 QKV 写入
`2:2+prefixLength` 行。未写入的缓存行保持不变。

算子只有一组状态索引，读写始终使用同一槽位，不支持独立的读写索引映射或跨槽位
状态迁移。batch 中的请求顺序可以变化，但每个请求必须保留自己的槽位。该约束
避免了跨请求覆盖源状态，不需要额外的跨请求状态快照或状态复制。

`flaSsmStateLayout=true` 时，SSM 状态采用 `[K,V]` 布局；`false` 时采用
`[V,K]` 布局，与原有算子一致。

## ACL Graph 回放

前缀端点和上一轮接受数量均为 Device 输入。在已捕获的 Shape 和上述约束范围内，
回放时修改这些输入即可改变实际循环边界，无需重新捕获图，也无需将长度读回 Host。
