<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnMtpDecode 算子设计

[English](../en/mega_gdn_mtp_decode.md) | 简体中文

## 系列文档

[核心 MegaChunkGdn](mega_chunk_gdn.md) |
[融合 Prefill](mega_gdn_prefill_op.md) | [单 token Decode](mega_gdn_decode.md) |
[Draft Decode](mega_gdn_draft_decode.md)

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`MegaGdnMtpDecode` 面向 Qwen3.5 Gated Delta Network（GDN）的多 Token
预测（Multi-Token Prediction，MTP）校验阶段。算子在一次设备调用中融合：

1. 宽度为 4 的 causal convolution；
2. Q/K L2 归一化；
3. GDN gate、decay 和 recurrent state 更新；
4. 每个校验 token 的 SSM checkpoint 写回；
5. RMSNorm 和 Z gate。

该算子与单 token 的 `MegaGdnDecode` 保持独立。MTP 需要扩展的 Conv state、
逐 token SSM checkpoint，以及独立的读/写 state slot；复用单 token ABI 会破坏
状态语义。

## 符号与 Shape

本文使用以下符号：

```text
B = batch size
K = speculative token 数，1 <= K <= 16
S = K + 1，包含 correction token 的校验序列长度
D = head dimension，固定为 128
NK = key head 数
NV = value head 数
C = (2 * NK + NV) * D
L = K + 3 = S + 2，Conv state 的槽内长度
N = state slot 数
```

## 计算公式

对校验序列中的每个 `t ∈ [0, K]`，先计算 causal convolution：

```text
conv_t = SiLU(
    W0 * history_t[0] +
    W1 * history_t[1] +
    W2 * history_t[2] +
    W3 * qkv_t)
```

将 `conv_t` 拆分为 Q、K、V 后，对 Q/K 做 L2 归一化：

```text
q_hat_t = q_t / sqrt(sum(q_t * q_t) + 1e-6) / sqrt(128)
k_hat_t = k_t / sqrt(sum(k_t * k_t) + 1e-6)
```

GDN gate 和 recurrent state 更新为：

```text
g_fp32_t = -exp(a_log) * softplus(a_t + dt_bias, threshold=20)
g_t      = FP32(BF16_RINT(g_fp32_t))
decay_t  = exp(g_t)
beta_t   = FP32(BF16_RINT(sigmoid(b_t)))

H_t      = decay_t * H_(t-1)
pred_t   = H_t^T * k_hat_t
delta_t  = (v_t - pred_t) * beta_t
H_t      = H_t + k_hat_t * delta_t^T
o_t      = H_t^T * q_hat_t

out_t = RMSNorm(BF16_RINT(o_t), norm_weight, eps=1e-6)
        * SiLU(z_t)
```

`H_-1` 从所选 read checkpoint 读取；每次得到 `H_t` 后，立即写入相应的
write checkpoint。下一个 token 继续使用 UB 中的 state，不从 GM 重读。

recurrent 数学统一使用逻辑布局 `H[K,V]`。`flaSsmStateLayout=true` 时 GM
直接保存该布局；`false` 时 GM 保存其转置 `H^T[V,K]`。non-FLA PTO 直接使用：

```text
pred = H_stored * k
H_stored += delta * k^T
out = H_stored * q
```

因此不需要执行 `128 x 128` 物理转置。两种布局使用相同的 checkpoint 索引、
Conv state 和 BF16 舍入点。

## 参数说明

所有 Tensor 使用 ND 格式且必须 contiguous。

| 参数名 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `qkv` | 输入 | BF16 | `[B,S,C]` | 未执行卷积的 Q/K/V 输入 |
| `z` | 输入 | BF16 | `[B,S,NV,D]` | 输出 gate |
| `b` | 输入 | BF16 | `[B,S,NV]` | beta gate 输入 |
| `a` | 输入 | BF16 | `[B,S,NV]` | decay gate 输入 |
| `convWeight` | 输入 | BF16 | `[4,C]` | 宽度为 4 的 depthwise Conv 权重 |
| `convState` | 输入 | BF16 | `[N,L,C]` | 用于读取的 Conv state |
| `aLog` | 输入 | FP32 | `[NV]` | recurrent decay 参数 |
| `dtBias` | 输入 | FP32 | `[NV]` | recurrent decay bias |
| `ssmState` | 输入 | FP32 | `[N*S,NV,D,D]` | 用于读取的 SSM checkpoints |
| `readStateIndices` | 输入 | INT32 | `[B]` | 每个 batch 项读取的 state slot |
| `writeStateIndices` | 输入 | INT32 | `[B]` | 每个 batch 项写入的 state slot |
| `numAcceptedTokens` | 输入 | INT32 | `[B]` | 上一轮接受数，取值范围 `[1,S]` |
| `normWeight` | 输入 | BF16 | `[D]` | RMSNorm 权重 |
| `flaSsmStateLayout` | 属性 | BOOL | 标量 | `true` 表示 `[K,V]`，`false` 表示 `[V,K]`，默认 `true` |
| `convOut` | 输出 | BF16 | `[B,S,C]` | 保留 BF16 舍入点的 Conv 输出 |
| `convStateOut` | 输出 | BF16 | `[N,L,C]` | 接收更新后 write slot 的 Conv state |
| `ssmStateOut` | 输出 | FP32 | `[N*S,NV,D,D]` | 接收更新后 write slot 的逐 token checkpoints |
| `out` | 输出 | BF16 | `[B,S,NV,D]` | 最终输出 |

## 状态合同

### SSM checkpoint

`numAcceptedTokens` 包含 correction token。读取位置为：

```text
read_checkpoint = read_state_id * S + num_accepted_tokens - 1
```

本轮每个 token 的写入位置为：

```text
write_checkpoint(t) = write_state_id * S + t,  t in [0, K]
```

### Conv state

本轮 token 0 从 read slot 的以下三行开始：

```text
read_conv_state[num_accepted_tokens - 1 : num_accepted_tokens + 2]
```

本轮结束后，write slot 更新为：

```text
write_conv_state[0:2]   =
    read_conv_state[num_accepted_tokens : num_accepted_tokens + 2]
write_conv_state[2:S+2] = qkv[0:S]
```

Conv state 保存原始 `qkv`，不是 `convOut`。

### 输入/输出 buffer 与 slot 所有权

- 输入和输出 state buffer 可以使用不同的设备地址；
- kernel 只从输入 buffer 的 read slot 读取，只向输出 buffer 的 write slot 写入；
- 使用独立输出 buffer 时，未被本次调用写入的其他 slot 内容不定义；如果调用方
  需要完整 state cache，必须自行保留或预拷贝这些 slot；
- 同一 batch 中的 write slot 必须唯一；
- read slot 可以重复，以支持多个请求读取同一个 shared prefix；
- `read_state_id == write_state_id` 时，kernel 必须先将初始 SSM state 搬入
  UB，再写 checkpoint 0；
- `read_state_id != write_state_id` 时，read slot 保持只读，所有更新只写入
  write slot。

tiling 回调只校验 dtype、rank 和 shape，不读取 device tensor 的值。accepted
范围、slot 范围、write 唯一性和跨请求所有权必须由调用侧在下发前检查，避免
引入 device-to-host 同步。

## 数值合同

| 阶段 | 计算/物化精度 |
| --- | --- |
| Conv 累加和 SiLU | FP32 |
| Conv 到 Q/K/V | BF16 RINT |
| Q/K L2Norm | FP32 |
| `g` 与 `beta` | FP32 计算，BF16 RINT 物化，再转换回 FP32 |
| decay 与 SSM state | FP32 |
| recurrent readout 到 Norm | BF16 RINT |
| RMSNorm 和 Z gate | FP32 |
| 最终输出 | BF16 RINT |

优化不得改变这些舍入点、`epsilon`、softplus threshold 或 state 更新顺序。

## Tiling 设计

### Host 校验

当前 Host tiling 支持：

- `1 <= B <= 32`；
- `2 <= S <= 17`，即 `1 <= K <= 16`；
- `D = 128`；
- `1 <= NK <= 16` 且 `NK` 为 2 的幂；
- `NV % NK == 0` 且 `1 <= NV/NK <= 4`；
- `1 <= N <= 1024`；
- UB 不小于 175,360B。

### Tiling key

| Key | 条件 | Device 路径 |
| ---: | --- | --- |
| 101/102/103/104/105 | `K=1/2/3/4/5` | 对应静态 K 模板 |
| 108 | `K=8` | 静态 K8 模板 |
| 208 | `K=8, B=4, NK=8, NV=24` 且资源满足 | Q/K group cache + deferred Norm |
| 308 | 与 key 208 相同，且为 Ascend 950、AIV/UB 满足 | A5 two-owner Q/K group 调度 |
| 210～216 | `K=10～16` 且 UB 不小于 182,816B | 静态 K + deferred Norm |
| 100 | 其他受支持 K，或专用 key 的资源不足 | 动态 K fallback |
| 1101～1105、1108 | non-FLA 且 `K=1～5、8` | non-FLA 静态 K 路径 |
| 1100 | 其他受支持的 non-FLA K | non-FLA 动态 K 路径 |

key 308 只在 `PTO_NPU_ARCH_A5` 下编译和调度。A2/A3 不包含该 device
分支，继续使用 mixed AIC/AIV 路径；A5 使用 AIV-only 路径。因此 A5 专用
优化不能改变 A2/A3 的 device 代码。

non-FLA layout 当前不使用仅针对 FLA 调优的 Q/K group cache、deferred Norm
和 A5 two-owner 特化。

### Block 调度

Conv 按 128 个 channel 分配 owner：

```text
conv_tasks = C / 128
```

Recurrent 按 `batch * value_head` 分配 owner：

```text
recurrent_tasks = B * NV
task_count = max(conv_tasks, recurrent_tasks)
used_aiv = min(task_count, platform_aiv_count)
```

A2/A3 通过 `CalcTschBlockDim` 使用 `MIX_AIC_1_2`；A5 使用 AIV-only
block dim。Host 设置 `schedule_mode=1`。Conv 写回后完成 MTE3，并在 owner
切换前执行 all-AIV 同步。

### UB 与 Workspace

通用路径的 Host UB reserve 为 175,360B；key 208/308 为 187,936B；
K10～16 deferred-Norm 最重路径为 182,816B。完整 FP32 state 和主要 compute
buffer 都是 `128 x 128`，在同一 head 的 S 次更新间复用。

算子仅申请 `platform.GetLibApiWorkSpaceSize()` 返回的 system workspace，
不使用 workspace 保存 Q/K/V、gate 或 state。checkpoint 直接写入目标 GM
state。

## 约束说明

- 仅支持参数表列出的 dtype、rank、shape 和 contiguous ND Tensor；
- 只支持宽度为 4 的 Conv 和 `D=128`；
- 调用侧必须保证 indices 和 accepted 的值域正确；
- 禁止两个活跃 batch 项写同一个 state slot；
- 不允许在算子已经写入部分 state 后回退到拆分小算子链；
- Prefix Cache 场景必须在调用前取得 shared read lease 和 exclusive write
  lease；
- Graph padding 必须使用合法且独占的 sink write slot，不能使用负数或越界
  index。

## 调用与构建

算子通过 ACLNN 接口 `aclnnMegaGdnMtpDecode` 调用。算子定义、Host tiling
和 device kernel 分别位于：

- [`mega_gdn_mtp_decode_def.cpp`](../../xllm_ops/mega_gdn_mtp_decode/op_host/mega_gdn_mtp_decode_def.cpp)
- [`mega_gdn_mtp_decode_tiling.cpp`](../../xllm_ops/mega_gdn_mtp_decode/op_host/mega_gdn_mtp_decode_tiling.cpp)
- [`mega_gdn_mtp_decode.cpp`](../../xllm_ops/mega_gdn_mtp_decode/op_kernel/mega_gdn_mtp_decode.cpp)

Python 回归测试位于
[`test_mega_gdn_mtp_decode.py`](../../test/python_test/test_mega_gdn_mtp_decode.py)。
