<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnDecode 算子设计

[English](../en/mega_gdn_decode.md) | 简体中文

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`MegaGdnDecode` 面向 Qwen3.5 Gated Delta Network（GDN）的单 token
decode 阶段。算子在一次设备调用中融合：

1. 宽度为 4 的 causal convolution；
2. Q/K L2 归一化；
3. GDN gate、decay 和 recurrent state 更新；
4. RMSNorm 和 Z gate；
5. Conv state 与 SSM state 更新。

多 Token 预测（MTP）校验由独立的 `MegaGdnMtpDecode` 处理。两个算子的
state layout、checkpoint stride 和调度合同不同，不能混用 ABI。

## 符号与 Shape

```text
B = batch size
D = head dimension，固定为 128
NK = key head 数
NV = value head 数
C = (2 * NK + NV) * D
N = state slot 数
```

## 计算公式

对每个 batch 项，使用 read slot 中的三行历史状态计算：

```text
conv = SiLU(
    W0 * conv_state_read[0] +
    W1 * conv_state_read[1] +
    W2 * conv_state_read[2] +
    W3 * qkv)
```

将 `conv` 拆分为 Q、K、V 后，对 Q/K 做 L2 归一化：

```text
q_hat = q / sqrt(sum(q * q) + 1e-6) / sqrt(128)
k_hat = k / sqrt(sum(k * k) + 1e-6)
```

每个 value head 使用对应的 key head，执行 recurrent 更新：

```text
g       = -exp(a_log) * softplus(a + dt_bias, threshold=20)
decay   = exp(g)
beta    = sigmoid(b)

H       = decay * H_read
pred    = H^T * k_hat
delta   = (v - pred) * beta
H_write = H + k_hat * delta^T
o       = H_write^T * q_hat

out = RMSNorm(BF16_RINT(o), norm_weight, eps=1e-6) * SiLU(z)
```

## 参数说明

所有 Tensor 使用 ND 格式且必须 contiguous。

| 参数名 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `qkv` | 输入 | BF16 | `[B,C]` | 未执行卷积的 Q/K/V 输入 |
| `z` | 输入 | BF16 | `[B,NV,D]` | 输出 gate |
| `b` | 输入 | BF16 | `[B,NV]` | beta gate 输入 |
| `a` | 输入 | BF16 | `[B,NV]` | decay gate 输入 |
| `convWeight` | 输入 | BF16 | `[4,C]` | 宽度为 4 的 depthwise Conv 权重 |
| `convState` | 输入 | BF16 | `[N,3,C]` | 用于读取的 Conv state |
| `aLog` | 输入 | FP32 | `[NV]` | recurrent decay 参数 |
| `dtBias` | 输入 | FP32 | `[NV]` | recurrent decay bias |
| `ssmState` | 输入 | FP32 | `[N,NV,D,D]` | 用于读取的 SSM state |
| `readStateIndices` | 输入 | INT32 | `[B]` | 每个 batch 项读取的 state slot |
| `writeStateIndices` | 输入 | INT32 | `[B]` | 每个 batch 项写入的 state slot |
| `normWeight` | 输入 | BF16 | `[D]` | RMSNorm 权重 |
| `convOut` | 输出 | BF16 | `[B,C]` | 保留 BF16 舍入点的 Conv 输出 |
| `convStateOut` | 输出 | BF16 | `[N,3,C]` | 接收更新后 write slot 的 Conv state |
| `ssmStateOut` | 输出 | FP32 | `[N,NV,D,D]` | 接收更新后 write slot 的 SSM state |
| `out` | 输出 | BF16 | `[B,NV,D]` | 最终输出 |

## 状态合同

### Conv state

每个 state slot 保存连续三个历史输入。调用完成后，write slot 更新为：

```text
write_conv_state[0] = read_conv_state[1]
write_conv_state[1] = read_conv_state[2]
write_conv_state[2] = qkv
```

Conv state 保存原始 `qkv`，不是 `convOut`。

### SSM state

每个 batch 项从 `readStateIndices` 指定的 slot 读取完整 FP32 state，完成
一次 recurrent 更新后写入 `writeStateIndices` 指定的 slot：

```text
H_read  = ssm_state[read_state_id]
H_write = updated_gdn_state(H_read)
ssm_state_out[write_state_id] = H_write
```

### 输入/输出 buffer 与 slot 所有权

- 输入和输出 state buffer 可以使用不同的设备地址；
- kernel 只从输入 buffer 的 read slot 读取，只向输出 buffer 的 write slot 写入；
- 使用独立输出 buffer 时，未被本次调用写入的其他 slot 内容不定义；如果调用方
  需要完整 state cache，必须自行保留或预拷贝这些 slot；
- 同一 batch 中的 write slot 必须唯一；
- read slot 可以重复，以支持多个请求读取同一个 shared prefix；
- same-slot 场景必须先完成 read state 的 GM→UB，再写回；
- prefix-fork 场景中 read slot 保持只读，所有更新只写入 private write slot。

tiling 回调不读取 device index tensor 的值。slot 范围、write 唯一性和跨请求
所有权必须由调用侧在下发前校验，避免 device-to-host 同步。

## 数值合同

| 阶段 | 计算/物化精度 |
| --- | --- |
| Conv 累加和 SiLU | FP32 |
| Conv 到 Q/K/V | BF16 RINT |
| Q/K L2Norm | FP32 |
| `g` 与 `beta` | BF16 输入转换为 FP32 后计算 |
| decay 与 SSM state | FP32 |
| recurrent readout 到 Norm | BF16 RINT |
| RMSNorm 和 Z gate | FP32 |
| 最终输出 | BF16 RINT/与平台参考路径一致的舍入 |

优化不得改变这些舍入点、`epsilon`、softplus threshold 或 state 更新顺序。

## Tiling 设计

### Host 校验

当前 Host tiling 支持：

- `1 <= B <= 32`；
- `D = 128`；
- `1 <= NK <= 16` 且 `NK` 为 2 的幂；
- `NV % NK == 0` 且 `1 <= NV/NK <= 4`；
- `1 <= N <= 1024`；
- Conv state 长度固定为 3。

### Tiling key

| Key | 条件 | Device 路径 |
| ---: | --- | --- |
| 2 | `B=1` | Batch-one 静态快路径 |
| 1 | `B>1` | 动态 batch 路径 |

Tiling data 只保存 `batch_size`、`num_k_heads` 和 `num_v_heads`。

### Block 调度

Conv 按 128 个 channel 分配任务，recurrent 按 `batch * value_head` 分配任务：

```text
conv_tasks = C / 128
head_tasks = B * NV
```

A2/A3 使用 `MIX_AIC_1_2` 和 `CalcTschBlockDim`；A5 使用 AIV-only
block dim。对 A2/A3 的 `B=1` 快路径，一个 mixed block 包含两个 AIV，
因此 Conv 任务数按 `(conv_tasks + 1) / 2` 计入 block 规划。其他场景使用
`max(conv_tasks, head_tasks)`。Host 设置 `schedule_mode=1`。

Conv 写回完成后执行跨 AIV 同步，再从 channel owner 切换为 recurrent head
owner。A5 与 A2/A3 的 device 实现使用编译期架构门禁；A5 专用数值或指令
优化不得改变 A2/A3 路径。

### UB 与 Workspace

每个 recurrent owner 在 UB 中保留一个完整的 FP32 `128 x 128` state 和
一个同尺寸 compute buffer，并复用 Q/K、gate、归约和 Norm scratch。

算子仅申请 `platform.GetLibApiWorkSpaceSize()` 返回的 system workspace，
不使用 workspace 保存业务中间量或 state。

## 约束说明

- 仅支持参数表列出的 dtype、rank、shape 和 contiguous ND Tensor；
- 只支持宽度为 4 的 Conv 和 `D=128`；
- 调用侧必须保证 read/write index 在 `[0,N)`；
- 禁止两个活跃 batch 项写同一个 state slot；
- 不允许在算子已经写入部分 state 后回退到拆分小算子链；
- Prefix Cache 场景必须在调用前取得 shared read lease 和 exclusive write
  lease。

## 调用与构建

算子通过 ACLNN 接口 `aclnnMegaGdnDecode` 调用。算子定义、Host tiling 和
device kernel 分别位于：

- [`mega_gdn_decode_def.cpp`](../../xllm_ops/mega_gdn_decode/op_host/mega_gdn_decode_def.cpp)
- [`mega_gdn_decode_tiling.cpp`](../../xllm_ops/mega_gdn_decode/op_host/mega_gdn_decode_tiling.cpp)
- [`mega_gdn_decode.cpp`](../../xllm_ops/mega_gdn_decode/op_kernel/mega_gdn_decode.cpp)

Python 回归测试位于
[`test_mega_gdn_decode.py`](../../test/python_test/test_mega_gdn_decode.py)。
