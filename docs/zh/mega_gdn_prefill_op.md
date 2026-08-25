<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnPrefillOp 算子设计

[English](../en/mega_gdn_prefill_op.md) | 简体中文

## 系列文档

[核心 MegaChunkGdn](mega_chunk_gdn.md) |
[单 token Decode](mega_gdn_decode.md) | [Draft Decode](mega_gdn_draft_decode.md) |
[MTP Verify](mega_gdn_mtp_decode.md)

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`MegaGdnPrefillOp` 面向 Qwen3.5 GDN 的 packed variable-length Prefill。
算子在一次设备调用中融合：

1. 宽度为 4 的 causal convolution 和 Conv cache 更新；
2. packed Q/K/V 拆分以及 Q/K L2 归一化；
3. decay gate 与 beta gate 准备；
4. `MegaChunkGdn` 的 KKT、solve、WY、H/O 流水；
5. SSM cache 读取和最终 state 写回；
6. RMSNorm 与 Z gate。

与 `MegaChunkGdn` 相比，该算子接收卷积前的 BF16 mixed QKV，并直接输出模型
需要的 BF16 normalized hidden state。

## 符号与 Shape

```text
B  = batch size
T  = packed token 总数
D  = head dimension，固定为 128
NK = key head 数
NV = value head 数
C  = (2 * NK + NV) * D
N  = Conv state slot 数
R  = checkpoint stride
M  = NV * sum_i ceil(sequence_length_i / 128)
```

所有 Tensor 使用 ND 格式且必须 contiguous。

| 参数名 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `mixed_qkv` | 输入 | BF16 | `[T,C]` | 卷积前的 packed Q/K/V |
| `b` | 输入 | BF16 | `[T,NV]` | beta gate 输入 |
| `a` | 输入 | BF16 | `[T,NV]` | decay gate 输入 |
| `z` | 输入 | BF16 | `[T,NV,D]` | 输出 gate |
| `conv_weight` | 输入 | BF16 | `[4,C]` | depthwise Conv 权重 |
| `conv_state` | 输入 | BF16 | `[N,R+2,C]` | Conv state/cache |
| `a_log` | 输入 | FP32 | `[NV]` | recurrent decay 参数 |
| `dt_bias` | 输入 | FP32 | `[NV]` | recurrent decay bias |
| `conv_state_read_indices` | 输入 | INT32 | `[B]` | Conv read slot；负数表示无初始 state |
| `conv_state_write_indices` | 输入 | INT32 | `[B]` | Conv write slot |
| `ssm_state_read_indices` | 输入 | INT32 | `[B]` | SSM cache 第一维的绝对 read index；负数表示零 state |
| `ssm_state_write_indices` | 输入 | INT32 | `[B]` | SSM cache 第一维的绝对 write index |
| `ssm_cache` | 输入 | FP32 | `[N*R,NV,D,D]` | flattened checkpoint cache |
| `mask_lower` | 输入 | FP32 | `[128,128]` | 严格下三角 mask |
| `mask_full` | 输入 | FP32 | `[128,128]` | 含对角线的下三角 mask |
| `minus_identity` | 输入 | BF16 | `[128,128]` | 对角线为 `-1` 的矩阵 |
| `cu_seqlens` | 输入 | INT32 | `[B+1]` | packed 序列边界 |
| `norm_weight` | 输入 | BF16 | `[D]` | RMSNorm 权重 |
| `ffts_addr` | 属性 | INT64 | 标量 | A2/A3 FFTS 地址；A5 必须为 0 |
| `num_matrices` | 属性 | INT64 | 标量 | 必须为 `M` |
| `norm_output` | 输出 | BF16 | `[T,NV,D]` | gated RMSNorm 输出 |
| `conv_state_out` | 输出 | BF16 | `[N,R+2,C]` | 更新后的 Conv cache |
| `ssm_cache_out` | 输出 | FP32 | `[N*R,NV,D,D]` | 更新后的 SSM cache |

Host 支持的 `NV` 为 `1/2/3/4/6/8/12/16/24/32/48/64`，并要求
`NK > 0`、`NV % NK == 0`。`num_matrices` 必须为正数、是 `NV` 的倍数且不
超过 `T*NV`。

## 计算公式

Conv 和 Q/K 归一化为：

```text
conv_t = SiLU(W0*x_(t-3) + W1*x_(t-2) + W2*x_(t-1) + W3*x_t)
q_hat  = q / sqrt(sum(q*q) + 1e-6)
k_hat  = k / sqrt(sum(k*k) + 1e-6)
```

gate 为：

```text
g    = -exp(a_log) * softplus(a + dt_bias, threshold=20)
beta = sigmoid(b)
```

随后执行 chunk GDN，最后计算：

```text
core = ChunkGdn(q_hat, k_hat, v, g, beta, initial_state)
out  = RMSNorm(core, norm_weight, eps=1e-6) * SiLU(z)
```

## Conv state 合同

- `conv_state_read_indices[b] < 0` 表示该序列没有历史，Conv 从零 history 开始；
- write index 必须位于 `[0,N)`；
- read 与 write 可以相同，也可以用于 prefix fork 的不同 slot；
- 多 batch 中若某个 read slot 同时被另一序列写入，kernel 会先保存紧凑快照，
  避免写后读污染；
- `R>1` 时，state 长度固定为 `R+2`，除三行 Conv history 外的 checkpoint tail
  会按 read/write slot 合同保留；
- 输出 buffer 与输入 buffer 分离时，调用侧负责预先保留未写 slot。

## SSM cache 合同

SSM index 是 `ssm_cache` 第一维的绝对 index，不是 Conv slot id。常见 MTP
布局中，逻辑 slot `s` 的 checkpoint 起点为 `s*R`：

```text
ssm_read_index  = read_slot  * R
ssm_write_index = write_slot * R
```

Prefill 只在 `ssm_state_write_indices[b]` 写入该序列的最终 state，其他
checkpoint 保持不变。read index 为负数时使用零 state。write index 必须合法且
同一调用中唯一。

## 数据类型合同

公共 tensor 边界固定为 BF16/FP32，但 chunk GDN 内部显式使用 FP16：

```text
GDN_PREFILL_COMPUTE_DTYPE = half
GDN_COMPUTE_DTYPE         = GDN_PREFILL_COMPUTE_DTYPE
ComputeT                  = GDN_COMPUTE_DTYPE
GDN_PUBLIC_DTYPE          = DTYPE_MIXED_QKV  // BF16
```

`mixed_qkv`、`minus_identity` 和最终输出仍遵守公共 BF16 ABI，workspace 中的
packed Q/K/V、beta、KKT、WY 和 state 中间量使用 FP16。

## Tiling、同步与 Workspace

Host 根据 `T` 和 `C` 划分 token/channel 任务，并使用 `cu_seqlens` 隔离 ragged
batch。当前构建使用 128-token chunk；`num_matrices` 决定 chunk state 数量。

A2/A3 使用 FFTS 跨 AIC/AIV 同步，ACLNN wrapper 从 runtime 获取
`ffts_addr`。A5 使用完整物理 MIX block 集和 PTO
`SYNCALL<SyncCoreType::Mix>`，不消费 FFTS 地址，因此 wrapper 将属性置为 0。
A5 专用路径由 `GDN_PREFILL_TARGET_A5` 隔离，不进入 A2/A3 编译分支。

该融合算子需要业务 workspace 保存 Conv snapshot、packed compute QKV、gate、
KKT、WY、H/O 和最终 state 中间量。workspace 随 `T`、`NV`、`M` 和 block 数
增长，必须使用 ACLNN GetWorkspaceSize 的返回值，不得复用
`MegaChunkGdn` 的 workspace 公式。

## 约束说明

- `cu_seqlens` 必须单调、首项为 0、末项为 T；
- read index 可以重复，write index 必须唯一；
- 负 read index 只表示无初始 state，负 write index 非法；
- `conv_state.shape[1]` 必须等于 `ssm_cache.shape[0]/N + 2`；
- mask 和 `minus_identity` 的内容由调用侧保证；
- graph padding 必须分配合法 sink write slot，不能依赖越界 index；
- state 开始更新后不得回退到拆分链再次写同一 slot。

## 调用与构建

算子通过 ACLNN 接口 `aclnnMegaGdnPrefillOp` 调用。主要实现位于：

- [`mega_gdn_prefill_op_def.cpp`](../../xllm_ops/mega_gdn_prefill_op/op_host/mega_gdn_prefill_op_def.cpp)
- [`mega_gdn_prefill_op_tiling.cpp`](../../xllm_ops/mega_gdn_prefill_op/op_host/mega_gdn_prefill_op_tiling.cpp)
- [`mega_gdn_prefill_op.cpp`](../../xllm_ops/mega_gdn_prefill_op/op_kernel/mega_gdn_prefill_op.cpp)
- [`gdn_prefill_frontend.h`](../../xllm_ops/mega_gdn_prefill_op/op_kernel/gdn_prefill_frontend.h)

Python 回归与性能脚本位于：

- [`test_mega_gdn_prefill_op.py`](../../test/python_test/test_mega_gdn_prefill_op.py)
- [`benchmark_mega_gdn_prefill_op.py`](../../test/python_test/benchmark_mega_gdn_prefill_op.py)
