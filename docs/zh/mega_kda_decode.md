<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaKdaDecode 算子设计

[English](../en/mega_kda_decode.md) | 简体中文

## 系列文档

[MegaKdaPrefill](mega_kda_prefill.md)

## 产品支持情况

| 产品 | 构建目标（`-c`） | 支持的模式 |
| --- | --- | --- |
| Ascend 910B（A2） | `ascend910b` | 普通 Decode（0）、MTP verify（1）、兼容 Decode（3） |
| Ascend 910C（A3） | `ascend910_93` | 普通 Decode（0）、MTP verify（1）、兼容 Decode（3） |
| Ascend 950（A5） | `ascend950` | 普通 Decode（0）、MTP verify（1） |

## 功能说明

一次设备调用完成 GLM-5.3-Flash KDA 的四抽头因果卷积、SiLU、Q/K 归一化和递推。普通 Decode 为每个请求处理一个 token。MTP verify 从上轮已接受的检查点恢复，并为本轮每个查询 token 写出新的检查点。接受数由调用方提供。

## 符号与 Shape

`B` 为请求数，`T` 为本次打包的 token 总数，`H` 为本地 head 数（1–128）。Q、K、V 的 head 维度均为 `D=128`。通道数 `C=3*H*D`，按全部 Q、全部 K、全部 V 排列。Conv 的读池、写池容量分别为 `Ci`、`Co`，SSM 的为 `Si`、`So`。

`M=max_query_tokens` 为每条请求的查询容量，`L` 为每个 Conv 槽的历史行数。普通及兼容 Decode 要求 `T=B, M=1, L=3`。MTP 允许每条序列有 0 到 `M` 个查询 token，且 `T>0, 1≤M≤17, L=M+2`。例如 MTP=3 校验当前 token 和三个草稿 token，使用 `M=4, L=6`。

## 计算公式

以下为单个 token、单个 head 的数学表达式。`x[0:4]` 按从旧到新的顺序包含三行原始 QKV 历史和当前投影。`S` 按 `[V,K]` 存储，向量衰减沿 K 维广播。平台舍入点见下文。

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

参数 `beta` 传 sigmoid **之前**的 BF16 投影 `beta_raw`。普通 Decode、MTP verify 和兼容模式都在核内转 FP32 并做一次 sigmoid。

## 参数说明

所有张量均为同一 NPU 上的连续 ND 张量。

| 参数名 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `qkv` | 输入 | BF16 | `[T,C]` | 原始 Q/K/V 投影 |
| `gate` | 输入 | BF16 | `[T,H,D]` | 原始向量门控投影 |
| `beta` | 输入 | BF16 | `[T,H]` | sigmoid 前的投影 |
| `a_log` | 输入 | FP32 | `[H]` | 每个 head 的门控缩放参数 |
| `gate_bias` | 输入 | FP32 | `[H,D]` | 向量门控偏置 |
| `conv_weight` | 输入 | BF16 | `[4,C]` | 从旧到新的四组逐通道权重 |
| `conv_bias` | 输入 | BF16 | `[C]` | 可选，传 `None` 表示零偏置 |
| `conv_state_in` | 输入 | BF16 | `[Ci,L,C]` | 原始 QKV 历史读池 |
| `ssm_state_in` | 输入 | FP32 | `[Si,H,D,D]` | SSM 读池，末两维为 [V,K] |
| `cu_seqlens` | 输入 | INT32 | `[B+1]` | 打包偏移，单调不减，首项 0、末项 T |
| `conv_read_indices` | 输入 | INT32 | `[B]` | 每个请求的 Conv 读槽 |
| `conv_write_indices` | 输入 | INT32 | `[B]` | 每个请求的 Conv 写槽 |
| `ssm_read_indices` | 输入 | INT32 | `[B,M]` | 每个请求的候选检查点读槽 |
| `ssm_write_indices` | 输入 | INT32 | `[B,M]` | 每个查询 token 的检查点写槽 |
| `num_accepted_tokens` | 输入 | INT32 | `[B]` | MTP 必传（1 到 M）。模式 0、3 传 `None` |
| `mode` | 属性 | INT | `scalar` | 0、1 或 3，默认 0 |
| `max_query_tokens` | 属性 | INT | `scalar` | 查询容量 M，默认 1 |
| `conv_output_slots` | 属性 | INT | `scalar` | Co，绑定从输出张量推导 |
| `ssm_output_slots` | 属性 | INT | `scalar` | So，绑定从输出张量推导 |
| `output` | 输出 | BF16 | `[T,H,D]` | 逐 token 的注意力输出 |
| `conv_state_out` | 输出 | BF16 | `[Co,L,C]` | 更新后的原始 QKV 历史 |
| `ssm_state_out` | 输出 | FP32 | `[So,H,D,D]` | 更新后的 [V,K] 状态 |

## 状态合同

**普通 Decode（模式 0、3）。** 每条请求从 `conv_read_indices[b]` 读三行历史，从 `ssm_read_indices[b,0]` 读 SSM。Conv 写槽保存最近两行历史和当前原始 QKV，SSM 写槽保存本 token 更新后的状态。`cu_seqlens` 必须为 `[0,1,…,B]`。

**MTP verify（模式 1）。** 令 `a=num_accepted_tokens[b]`，`n=cu_seqlens[b+1]-cu_seqlens[b]`。以下切片右端开区间。空请求跳过全部读写。

```text
initial_ssm = ssm_state_in[ssm_read_indices[b, a-1]]
history = conv_state_in[conv_read_indices[b]]
initial_conv = history[a-1:a+2]
conv_state_out[conv_write_indices[b], :n+2] = concat(history[a:a+2], query_qkv)
ssm_state_out[ssm_write_indices[b, j]] = state_after_query_token_j  (0 ≤ j < n)
```

上述写回仅针对有效写槽，Conv 尾部保留原值。以 MTP=3、`a=2` 为例，初态取 SSM 槽表第 1 列（从 0 计），Conv 使用历史的第 1、2、3 行。即使本轮仅有一个查询 token，仍使用模式 1。

**槽位。** 槽 0 有效。Conv 或选中的 SSM 读槽为 `-1` 时整条请求跳过，输出行保留原值。冷启动需分配真实的全零读槽。写槽为 `-1` 仅跳过该次状态写回。多个请求可共享读槽，同一输出池内的有效写目标必须唯一，包括不同 token 的检查点。

三个输出均由调用方预分配，彼此以及与所有输入之间均不能共享存储。读写池的容量可各自设置。偏移、槽表和接受数（如适用）在设备端读取，可在两次图回放之间更新，单次调用期间须保持稳定。

## 数值合同

四抽头 Conv 累加和 SiLU 用 FP32，SiLU 输出舍入 BF16 后拆分 Q/K/V。各平台的 Conv 输入和 SiLU 前舍入点如下。

| 平台与模式 | Conv 输入和历史 | SiLU 前的累加结果 |
| --- | --- | --- |
| 910B/910C, 0 / 1 | BF16 → FP16 → FP32 | FP32 → BF16 → FP32 |
| 910B/910C, 3 | BF16 → FP32 | FP32 → BF16 → FP32 |
| 950, 0 | BF16 → FP32 | FP32 |
| 950, 1 | BF16 → FP16 → FP32 | FP32 → BF16 → FP32 |

910B/910C 计算 FP32 行范数 `sqrt(sum(x*x))`，正范数取倒数相乘，零行保持为零。950 使用 `x / sqrt(sum(x*x) + 1e-6)`。两者均在归一化后缩放 Q。所有模式的 beta sigmoid、向量衰减和 SSM 状态使用 FP32。模式 0、1 的 gate 用 `-5 / denominator`，910B/910C 模式 3 先求分母倒数再乘 `-5`。

## Tiling 设计

Host 根据显式模式选择 tiling key。普通 Decode 为 1，MTP verify 为 2，910B/910C 兼容 Decode 为 3。一次调用的所有请求共享模式。910B/910C 按请求、head 和 V 行分配 AIV 任务。若 `2*B*H` 足以覆盖已启动的 AIV block，每个任务处理 64 行，否则处理 32 行。两种分片分别使用 120,032 和 62,368 字节显式 UB，仅首个 V 分片写 Conv 状态，无用户 workspace。64 行分片中，模式 0、3 的归约分两组 32 行，模式 1 使用一组 64 行。

## 约束说明

Host 校验模式、容量、dtype 和静态 shape。调度器负责验证设备索引范围和写槽唯一性。普通 Decode 的两张 SSM 槽表也必须保留第二维 `[B,1]`。Conv 权重使用预打包的 `[4,C]`，从 `[C,1,4]` 转换时可在加载权重并完成 TP 切分后做一次。

## 调用与构建

在已配置 CANN 的仓库根目录使用默认算子集合构建。若在干净目录中用 `--op-name` 只选 KDA，主干的注册信息生成步骤仍依赖 `inner` 和 `exc` 两组 INI。测试用 Python 绑定从输出张量推导状态池容量。

```bash
bash build.sh
```

构建 910C 版本时，运行 `bash build.sh -c ascend910_93` 可显式指定目标。KDA 包已在 x86_64 Linux 上使用 CANN 9.0 针对 `Ascend910_9391` 交叉编译，并在隔离目录安装成功。910C 真机精度和性能待验证。

ACLNN 入口为 `aclnnMegaKdaDecodeGetWorkspaceSize` 和 `aclnnMegaKdaDecode`。以下示例假设已选定当前 NPU，且已按参数表准备连续的输入和输出缓冲区。

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

上例为普通 Decode。MTP=3 时改传 `mode=1, max_query_tokens=4` 及设备上的 `num_accepted_tokens`，两张 SSM 槽表均为 `[B,4]`，Conv 历史长度为 6。

在提供 KDA 后端开关的 GLM 接入版本中，`XLLM_KDA_KERNEL=mega_kda` 为支持的布局选择融合路径，默认后端为 `fla_npu`。非 MTP（verify width 为 1）可设置 `XLLM_KDA_PREFILL_BACKEND=fla` 保留 FLA Prefill，此时 910B/910C 上的 `XLLM_KDA_DECODE_NORM=public` 选择兼容 Decode（模式 3）。MTP 接入将支持的 V3 快照映射到模式 1 的检查点槽表，其他布局在执行前选择 FLA V3。

[算子定义](../../xllm_ops/mega_kda_decode/op_host/mega_kda_decode_def.cpp) · [Host tiling](../../xllm_ops/mega_kda_decode/op_host/mega_kda_decode_tiling.cpp) · [Device kernel](../../xllm_ops/mega_kda_decode/op_kernel/mega_kda_decode.cpp) · [Python 绑定](../../test/python_test/mega_kda_ops.cpp) · [NPU 测试](../../test/python_test/test_mega_kda_decode.py)
