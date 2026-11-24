<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaKdaPrefill 算子设计

[English](../en/mega_kda_prefill.md) | 简体中文

## 系列文档

[MegaKdaDecode](mega_kda_decode.md)

## 产品支持情况

| 产品 | 构建目标（`-c`） | 支持的模式 |
| --- | --- | --- |
| Ascend 910B（A2） | `ascend910b` | Prefill |
| Ascend 910C（A3） | `ascend910_93` | Prefill |
| Ascend 950（A5） | `ascend950` | Prefill |

## 功能说明

一次设备调用处理 GLM-5.3-Flash 的一批变长序列，完成四抽头因果卷积、SiLU、Q/K 归一化、向量门控和分块 KDA 递推。返回每个 token 的注意力输出，以及每条非空序列的最终 Conv 和 SSM 状态。

## 符号与 Shape

`B` 为请求数，`T` 为本次打包的 token 总数，`H` 为本地 head 数（1–128）。Q、K、V 的 head 维度均为 `D=128`。通道数 `C=3*H*D`，按全部 Q、全部 K、全部 V 排列。Conv 的读池、写池容量分别为 `Ci`、`Co`，SSM 的为 `Si`、`So`。

## 计算公式

以下为单个 token、单个 head 的数学表达式。`x[0:4]` 按从旧到新的顺序包含三行原始 QKV 历史和当前投影。`S` 按 `[V,K]` 存储，向量衰减沿 K 维广播。平台舍入点见下文。

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

参数 `beta` 传调用方算好的 FP32 权重 `torch.sigmoid(beta_raw.float())`。Prefill 直接将它用于状态更新。

## 参数说明

所有张量均为同一 NPU 上的连续 ND 张量。

| 参数名 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `qkv` | 输入 | BF16 | `[T,C]` | 原始 Q/K/V 投影 |
| `gate` | 输入 | BF16 | `[T,H,D]` | 原始向量门控投影 |
| `beta` | 输入 | FP32 | `[T,H]` | sigmoid 后的权重 |
| `a_log` | 输入 | FP32 | `[H]` | 每个 head 的门控缩放参数 |
| `gate_bias` | 输入 | FP32 | `[H,D]` | 向量门控偏置 |
| `conv_weight` | 输入 | BF16 | `[4,C]` | 从旧到新的四组逐通道权重 |
| `conv_bias` | 输入 | BF16 | `[C]` | 可选，传 `None` 表示零偏置 |
| `conv_state_in` | 输入 | BF16 | `[Ci,3,C]` | 原始 QKV 历史读池 |
| `ssm_state_in` | 输入 | FP32 | `[Si,H,D,D]` | SSM 读池，末两维为 [V,K] |
| `cu_seqlens` | 输入 | INT32 | `[B+1]` | 打包偏移，单调不减，首项 0、末项 T |
| `conv_read_indices` | 输入 | INT32 | `[B]` | 每个请求的 Conv 读槽 |
| `conv_write_indices` | 输入 | INT32 | `[B]` | 每个请求的 Conv 写槽 |
| `ssm_read_indices` | 输入 | INT32 | `[B]` | 每条序列的 SSM 初态读槽 |
| `ssm_write_indices` | 输入 | INT32 | `[B]` | 每条序列的 SSM 终态写槽 |
| `conv_output_slots` | 属性 | INT | `scalar` | Co，绑定从输出张量推导 |
| `ssm_output_slots` | 属性 | INT | `scalar` | So，绑定从输出张量推导 |
| `ffts_addr` | 属性 | INT | `scalar` | 同步地址，绑定在 910B/910C 查询，在 950 上传 0 |
| `output` | 输出 | BF16 | `[T,H,D]` | 逐 token 的注意力输出 |
| `conv_state_out` | 输出 | BF16 | `[Co,3,C]` | 更新后的原始 QKV 历史 |
| `ssm_state_out` | 输出 | FP32 | `[So,H,D,D]` | 更新后的 [V,K] 状态 |

## 状态合同

Conv 和 SSM 独立寻址。读槽 `-1` 将对应的历史或 SSM 初始化为零，序列继续计算。每条非空序列在 Conv 写槽保存最近三行原始 QKV（短序列接续输入历史），在 SSM 写槽保存最后一个 token 的状态。写槽 `-1` 仅丢弃该类状态的写回。空序列不写状态，未写的输出槽保留原值。

槽 0 有效。多条序列可共享读槽。同一输出池中，非空序列的有效写槽必须唯一。MTP 模型做 Prefill 时传入最近三行 Conv 历史，额外的投机历史和检查点由调用方保存。

三个输出均由调用方预分配，彼此以及与所有输入之间均不能共享存储。读写池的容量可各自设置。偏移、槽表和接受数（如适用）在设备端读取，可在两次图回放之间更新，单次调用期间须保持稳定。

## 数值合同

两条内核路径将 BF16 Conv 输入和历史直接转为 FP32，四抽头累加后执行 FP32 SiLU，结果舍入 BF16。Q/K 按 `x / sqrt(sum(x*x) + 1e-6)` 归一化并舍入 BF16。910B/910C 的归一化使用 FP32 倒数平方根，950 的实现对大值行做范围缩放。gate 前缀和、beta 与对外 SSM 状态均为 FP32。

## Tiling 设计

910B/910C 对所有 H 使用 64-token chunk，950 通用路径使用 128-token chunk。两者通过 KKT、求逆、WY 以及 chunk H/O 计算状态和输出，在内部 KV 与接口 VK 布局之间转置。950 在 `B=1, H=8, 1≤T≤4` 时直接做递推。

910B/910C Conv 前端使用 16、32 或 64 行的分块。每条序列的前三个 token 读取历史，后续使用分块路径，尾块只写有效行。gate 前缀和按序用 FP32 扫描，缓冲区复用前完成 MTE3 写出。

## 约束说明

要求 `1≤T≤131072`、`1≤B≤1024`，各状态池容量为 1–1,048,576 槽，计算出的 workspace 上限为 32 GiB。实际可分配容量取决于可用 HBM。设备在写出前检查偏移、索引和重复写目标，元数据无效时直接返回，调用方应预先校验。

## 调用与构建

在已配置 CANN 的仓库根目录使用默认算子集合构建。若在干净目录中用 `--op-name` 只选 KDA，主干的注册信息生成步骤仍依赖 `inner` 和 `exc` 两组 INI。测试用 Python 绑定从输出张量推导状态池容量。

```bash
bash build.sh
```

构建 910C 版本时，运行 `bash build.sh -c ascend910_93` 可显式指定目标。KDA 包已在 x86_64 Linux 上使用 CANN 9.0 针对 `Ascend910_9391` 交叉编译，并在隔离目录安装成功。910C 真机精度和性能待验证。

ACLNN 入口为 `aclnnMegaKdaPrefillGetWorkspaceSize` 和 `aclnnMegaKdaPrefill`。以下示例假设已选定当前 NPU，且已按参数表准备连续的输入和输出缓冲区。

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

在提供 KDA 后端开关的 GLM 接入版本中，`XLLM_KDA_KERNEL=mega_kda` 为支持的布局选择融合路径，默认后端为 `fla_npu`。非 MTP（verify width 为 1）可设置 `XLLM_KDA_PREFILL_BACKEND=fla` 保留 FLA Prefill，此时 910B/910C 上的 `XLLM_KDA_DECODE_NORM=public` 选择兼容 Decode（模式 3）。MTP 接入将支持的 V3 快照映射到模式 1 的检查点槽表，其他布局在执行前选择 FLA V3。

[算子定义](../../xllm_ops/mega_kda_prefill/op_host/mega_kda_prefill_def.cpp) · [Host tiling](../../xllm_ops/mega_kda_prefill/op_host/mega_kda_prefill_tiling.cpp) · [Device kernel](../../xllm_ops/mega_kda_prefill/op_kernel/mega_kda_prefill.cpp) · [Python 绑定](../../test/python_test/mega_kda_ops.cpp) · [NPU 测试](../../test/python_test/test_mega_kda_prefill.py)
