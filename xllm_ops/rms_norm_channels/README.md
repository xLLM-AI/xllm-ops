# RmsNormChannels

Ascend950 通道轴 RMSNorm：对连续 `[B, C, T, H, W]` 张量的通道维（dim 1）归一化，
不做转置、不产生额外 contiguous 拷贝。

```text
N = T * H * W
ms[b, t, h, w] = (1 / C) * sum_c value[b, c, t, h, w]^2
out[b, c, t, h, w] = value[b, c, t, h, w] * rsqrt(ms + epsilon) * gamma[c]
```

归约与调制在 FP32 中完成，最后按 value 的 dtype 写回。支持 FP16 / BF16 / FP32，
`gamma` 为长度 C 的 1D 张量且与 `value` 同 dtype，`epsilon` 为有限非负 float。

## Tiling 分支

host tiling 按 shape 家族选一个 tiling key，kernel 入口据此分发（见
`op_kernel/rms_norm_channels_common.h`）：

| key | 名称 | 适用 shape | 策略 |
|---|---|---|---|
| 1001 | SMALL_C | C <= 512 | 整个 `[C, Ntile]` tile 常驻 UB，一次读入一次写出 |
| 2001 | SPLIT_C | C >= 1024 且 N <= 1024，或 C in [256, 512] 且 B=1 且 N <= 160 | C 按核切分，各核写部分平方和，`SyncAll` 后合并并归一化 |
| 3001 | STREAM | C > 512 且不满足 SPLIT_C | 沿 C 分块流式两遍（求和、归一化） |
| 4001 | SINGLE_CHANNEL | C == 1 | 退化为 `x * rsqrt(x*x + eps) * gamma` |
| 5001 | SIMT_WARP | C <= 256 且 N <= 16 且 B*N <= 8192 | 一个 SIMT warp 处理一个输出位置，lane 切 C |

这些阈值来自 Ascend950PR 上的实测 A/B，调整前需要重新对比。SPLIT_C 使用跨核
`SyncAll` 屏障，因此 host 侧会设置 batch mode 调度并保证 block dim 不超过物理核数。

## 文件

```text
rms_norm_channels/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   ├── rms_norm_channels_def.cpp
│   ├── rms_norm_channels_infershape.cpp
│   ├── rms_norm_channels_tiling.h
│   └── rms_norm_channels_tiling.cpp
└── op_kernel/
    ├── rms_norm_channels.cpp
    ├── rms_norm_channels_common.h
    └── rms_norm_channels_kernel_impl.h
```

## 构建

已加入 Ascend950 默认构建列表，与其他算子一起构建：

```bash
bash build.sh -c ascend950 --no-build-transformer --no-build-flash
```

只生成单算子包：

```bash
cd xllm_ops
bash build.sh --pkg --ops=rms_norm_channels --soc=ascend950 -j8
```

生成的公开接口为 `aclnnRmsNormChannelsGetWorkspaceSize` 与 `aclnnRmsNormChannels`。

## Torch 调用

`test/python_test` 的 `custom_ops_lib` 已提供包装：

```python
import custom_ops

out = custom_ops.rms_norm_channels_npu(value, gamma, eps=1e-12)
```

## 限制

- 仅支持连续 rank-5 `[B, C, T, H, W]` 输入，T/H/W 必须为静态正数。
- `gamma` 必须为 1D、长度等于 C，且与 `value` 同 dtype。
- 仅注册 ascend950（arch35）。
