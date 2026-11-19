# AdaRmsNorm

Ascend950 上的融合 RMS 归一化算子：

```text
r = x * rsqrt(mean(x * x, axis=-1, keepdim=True) + epsilon)
out = r * (1 + scale) + shift
```

传入可选 `weight[D]` 时，先执行 `r *= weight`。归约与调制使用 FP32，仅最终输出转换到 x 的 dtype。

支持连续 ND 的 FP16、BF16、FP32，x 为 `[N,D]` 或 `[B,S,D]`，`1 <= D <= 8192`。scale、shift 和 weight 与 x 同 dtype；scale/shift 最后一维为 D，支持共享和完整广播。rank3 的 `[B,D]` 按 Ada 语义解释为 `[B,1,D]`，即使 B=S 也沿 batch 广播。epsilon 必须转换为有限正 FP32 值。

## 构建

在仓库根目录使用现有构建入口，和其余Ascend950算子一并构建：

```bash
bash build.sh -c ascend950 --no-build-transformer --no-build-flash
```

AdaRmsNorm 已进入 Ascend950 默认构建列表。开发时仅生成单算子包，可从 `xllm_ops` 目录调用其原有入口：

```bash
bash build.sh --pkg --ops=ada_rms_norm --soc=ascend950 -j8
```

本算子通过仓库 `op_host` 自动发现，使用统一 ACLNN 生成和自定义 OPP 包构建体系。生成的公开接口为 `aclnnAdaRmsNormGetWorkspaceSize` 与 `aclnnAdaRmsNorm`。

单算子包用于隔离验证；正式安装使用包含所需其他算子的完整包。

## Torch 调用

使用仓库原有 `test/python_test/setup.py` 构建 `custom_ops_lib`，按同目录现有测试方式加载 `custom_ops`：

```python
import custom_ops

out = custom_ops.ada_rms_norm_npu(x, scale, shift, weight=None, eps=1e-6)
```

Python封装与 `beam_search_group_npu` 一样调用 `custom_ops_lib`，C++适配函数 `ada_rms_norm_impl_npu` 直接定义在 `RegisterOps.cpp` 中。也可调用 `custom_ops_lib.ada_rms_norm(...)`，其可选weight与eps缺省值保持一致。

`RegisterOps.cpp` 只保留一个执行适配函数。shape、dtype和广播约束由ACLNN/Tiling检查；scalar epsilon在转换为FP32前检查，空输出直接返回。

`op_kernel/ada_rms_norm.h` 集中定义相关常量、kernel内部参数结构和计算类。普通Host编译只读取共享定义，设备编译读取计算实现。

## 验证

```bash
pytest -q test/python_test/test_ada_rms_norm.py
```

测试覆盖原始 profiling 的21个BF16 shape、FP16/FP32、归约尾部、Ada广播、可选gamma、空输入、参数拒绝及Python封装与C++绑定调用的一致性。

极大输入的平方或其和可能超过FP32范围；当前不保证该数值域内的稳定归一化。

## 已验证结果

在指定A5容器、CANN9.2、torch2.7.1/torch_npu2.7.1.post10环境中：

- 仓库自定义OPP包构建和隔离安装通过，正式ACLNN两阶段接口可加载。
- 仓库原有NpuExtension构建通过，`custom_ops`与`custom_ops_lib`调用入口可用。
- 166个功能测试通过，覆盖全部21个profiling shape；无跳过。
- 与原直调版本对照的33组输入全部输出逐字节一致，覆盖profiling shape、三种dtype及可选gamma。
- 与现有 `ReplaceToken` 合编通过，并在同一包、同一进程成功调用两者。
