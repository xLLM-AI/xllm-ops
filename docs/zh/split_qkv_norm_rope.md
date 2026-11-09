<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# SplitQkvNormRope 算子设计

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |

仅 Ascend 950（`DAV_3510`，arch35）实现，其余代际不提供兼容分支。

## 功能说明

`SplitQkvNormRope` 把 JoyOV2 DiT 注意力前的三步融合成一个 kernel：

```text
1. SplitQKV : qkv[B,S,(Nq+2*Nk)*D + tail] -> q / k / v
2. QK-Norm  : q、k 各自按 head 做 RMSNorm（gamma 独立）
3. RoPE     : q、k 做 NeoX 半分化旋转（cos/sin 只取前 D/2 维）
```

融合收益来自两点：把 29 个 kernel 收成 1 个（省 launch 与调度），以及把中间张量
从约 350 MB 压到 73 MB（只保留 qkv 输入与 q/k/v 输出）。

与上游 Python 的对应关系：

| 上游 | 本算子 |
| --- | --- |
| `q, k, v = qkv.split(...)` | 输入行内前 `(Nq+2*Nk)*D` 列按 head 切分 |
| `norm_q(x)` / `norm_k(x)` | `1/sqrt(mean(x^2)+eps) * gamma`，fp32 全链 |
| `apply_rotary_neox` | `[x1*c - x2*s, x2*c + x1*s]`，`x1=x[0:64]`、`x2=x[64:128]` |
| `k.repeat_interleave(rep, dim=2)` | `gqa_expand=true` 时在写出阶段重复发 DMA |

`gqa_expand` 是运行时开关：`false` 时 k/v 只写 `Nk` 个头（交给 FA 的原生 GQA），
`true` 时扩张成 `Nq` 个头（兜底），同一份二进制支持两种模式。

## 符号与形状

```text
B, S      = batch / sequence
D         = head_dim，固定 128
Nq, Nk, Nv= q / k / v 头数，v1 约束 Nv == Nk，Nq % Nk == 0，Nq+Nk+Nv <= 16
stride    = qkv 行跨度（元素），>= (Nq+Nk+Nv)*D，尾部 tail 列被忽略
ropeStride= cos/sin 行跨度（元素），取值 [64, 128]；只用前 64 维
```

| 张量 | 形状 | dtype |
| --- | --- | --- |
| `qkv` | `[B, S, stride]` | bf16 / fp16 |
| `q_gamma` | `[D]` | 同 `qkv` |
| `k_gamma` | `[D]` | 同 `qkv` |
| `cos` | `[B, S, ropeStride]` | 同 `qkv` |
| `sin` | `[B, S, ropeStride]` | 同 `qkv` |
| `q`（BSND） | `[B, S, Nq*D]` | 同 `qkv` |
| `q`（BNSD） | `[B, Nq, S, D]` | 同 `qkv` |
| `k`/`v`（BSND） | `[B, S, NkvOut*D]` | 同 `qkv` |
| `k`/`v`（BNSD） | `[B, NkvOut, S, D]` | 同 `qkv` |

`NkvOut = gqa_expand ? Nq : Nk`。

## 接口

aclnn 接口由 opbuild 依据 OpDef **自动生成**（本算子不手写 `op_api/`，
与 `replace_token`、`beam_search` 一致）：

```cpp
// 生成物，签名由 OpDef 的 input/attr/output 顺序决定
aclnnStatus aclnnSplitQkvNormRopeGetWorkspaceSize(
    const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma,
    const aclTensor *cos, const aclTensor *sin,
    int64_t numHeadQ, int64_t numHeadK, float epsilon,
    bool gqaExpand, bool layoutBNSD,
    aclTensor *qOut, aclTensor *kOut, aclTensor *vOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnSplitQkvNormRope(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

工程约定（踩过的坑，接入时注意）：

- **tilingKey 必须是 0**：kernel 只有一个入口（无 `TILING_KEY_IS` 分支），
  框架生成的 `kernelList` 入口名是 `<op>_0`，`SetTilingKey` 与之不符会报
  `funcEntry is invalid`。
- **dtype 用 `DTYPE_QKV` 取**：bf16/fp16 由框架拆成两个 bin，
  kernel 侧写 `KernelSplitQkvNormRope<DTYPE_QKV>` 即可，不要自己按 key 分派。
- **改 kernel 源码必须清 `build/binary`**：框架把 `op_kernel/` 拷到
  `build/binary/<soc>/src/<op>/op_kernel/` 并打 `.done` 标记，该拷贝只做一次；
  只删那一层会导致 kernel 二进制不再产出。

| 属性 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `num_q_heads` | Int | 必填 | `Nq` |
| `num_kv_heads` | Int | 必填 | `Nk`（`Nv = Nk`） |
| `epsilon` | Float | 1e-6 | RMSNorm eps |
| `gqa_expand` | Bool | false | k/v 是否扩张到 `Nq` 头 |
| `layout_bnsd` | Bool | false | 输出布局 `[B,H,S,D]` / `[B,S,H*D]` |

约束（host 侧校验，失败返回 `ACLNN_ERR_PARAM_INVALID`）：

- `D == 128`，`stride >= (Nq+2*Nk)*D`，`ropeStride ∈ [64,128]`；
- `Nq > 0`、`Nk > 0`、`Nq % Nk == 0`、`Nq+Nk+Nv <= 16`；
- 输入输出 dtype 一致，`q_gamma`/`k_gamma` 长度为 `D`；
- UB 需求 `<= 253952 B`（主场景实测 44544 B）。

## 算法

对第 `t` 个 token 的第 `h` 个 q/k 头（`x = qkv[t, h*D : (h+1)*D]`，fp32）：

```text
sumsq  = Σ_{i=0}^{D-1} x_i^2
scale  = 1 / sqrt(sumsq / D + eps)
y_i    = x_i * scale * gamma_i
x1     = y[0:64],  x2 = y[64:128]
out    = [ x1*cos - x2*sin ,  x2*cos + x1*sin ]
```

`v` 头不做 norm/rope，由一次多行 UB→UB `DataCopy` 原样搬进输出 tile（不经 fp32）。

## 内存布局与 32 B 对齐

输入行跨度 `stride*2` 通常不是 32 B 整数倍（主场景 `1286*2 = 2572 B`，
`2572 = 4*643`）。这带来两个层次的影响，必须分别处理：

1. **tile 起点**：`DataCopyPad` 只要求 GM 源地址 1 B 对齐，因此逐 token 搬即可；
2. **tile 内部行距**：VF 的 `LoadAlign/StoreAlign` 要求 UB 地址 **32 B 对齐**。
   若 UB 行距直接用 `stride`，只有每 4 行中的 1 行正确（`4*5144 B = 643*32`），
   其余行读错位，且慢约 1e4 倍。

因此 UB 侧引入 padding 行距：

```text
padStride      = RoundUp(stride, 16)      # 1286 -> 1296
bf16 行距      = padStride * 2 = 2592 B   # 2592 = 81*32
```

搬入按 token 逐个 `DataCopyPad`（`blockLen = stride*2`），硬件把每块右侧补 dummy
到 32 B 对齐，正好填满 padding 区；VF 内部按 token 分段访问。

## Tiling

`tokenTile` 固定 8（`stride*2` 与 32 B 的对齐周期决定），核间按 token 连续段切分：

```text
blockDim     = min(aivNum, ceil(totalTokens / 8))
tokenPerCore = RoundUp(ceil(totalTokens / blockDim), 8)
blockDim     = ceil(totalTokens / tokenPerCore)      # 回算
lastBlockTokenNum = totalTokens - tokenPerCore*(blockDim-1)
```

ACLNN 路径 `aivNum = ascendcPlatform.GetCoreNumAiv() = 54`，主场景
（`S = 12915`）：`blockDim = 54`、`tokenPerCore = 240`、`lastBlockTokenNum = 195`，
每核 30 个 tile。

TilingData 字段：`batch / seqLen / numHeadQ / numHeadK / numHeadV / headDim /
qkvStride / padStride / ropeStride / totalTokens / tokenPerCore / tokenTile /
lastBlockTokenNum / eps / reciprocal / flags / blockDim`。
`flags` 用位域替代 tilingKey 做布局与 GQA 分支：`bit0 = gqa_expand`、`bit1 = layout_bnsd`；
tilingKey 只区分 dtype（`0 = bf16`、`1 = fp16`）。

## Kernel 实现要点

- **编程模型**：norm + rope 走 SIMD-Regbase（`__simd_vf__` + `Reg::*`），单一 VF、单一 callee 边界。
- **数据入口**：q/k 与 cos/sin 直接从 bf16/fp16 的输入 tile 载入
  （`LoadAlign<T, DIST_UNPACK_B16>` + `Cast<float, T, layoutZMrgZ>`），
  UB 侧不保存 fp32 副本。
- **两级扫描**：pass1 只算每 `(token, head)` 的 sumsq（`ReduceSum` +
  `StoreAlign<DIST_FIRST_ELEMENT_B32>`），pass2 再算 norm+rope，整个 tile 只 1 次 `LocalMemBar`。
- **scale 链**：每 tile 做一次，段数按 `ceil(tokenTile*(Nq+Nk)/64)` 运行期确定
  （`Muls→Adds→Sqrt→Div`，950 无 `Reg::Rsqrt`），per-head 只留 1 次
  `LoadAlign<DIST_BRC_B32>` 广播。末段多出的 lane 只写回 `sumsq`，pass2 不会回读。
- **γ 折进 cos/sin**：`a=γ1·cos、c=γ1·sin、d=γ2·cos、b=-γ2·sin`，每 head 少 2 次乘法。
- **RoPE 融合**：每半 1 次 `Mul` + 1 次 `MulAddDst`。
- **输出 pack-store**：`Cast<T,float,RndRSatNS>` + `StoreAlign<T, DIST_PACK_B32>`
  直接把 fp32 结果打包进输出 tile，无整趟 cast-out；目标行距用输出 tile 行距
  `numHeadTotal*128`，不是 q/k 行宽。
- **v 直通**：一次多行 UB→UB `DataCopy` 完成，不经 fp32、不进 VF。
- **三级流水**：`inQkv_/inCos_/inSin_/outTile_` 全部深度 2，
  循环体为「预取 tile k+1 → 计算 tile k → 写出 tile k-1」，
  使 `gqa_expand`（3 倍 k/v 写出）的 mte3 与计算重叠。
- **单一路径**：任意 `Nq+Nk <= 16` 走同一份 VF，无兜底分支；无 fp32 中间缓冲，
  UB 只占 `inQkv ×2 + cos/sin ×2 + gamma ×2 + outTile ×2 + sumsq`（主场景 44544 B）。
- **布局与 GQA 均为运行期参数**：同一份二进制支持 BSND/BNSD、`gqa_expand` 0/1。

## 精度

fp32 中间量、bf16/fp16 输入输出，判据按 dtype：

| dtype | 判据 |
| --- | --- |
| bf16 | `|a-b| <= 2^-7*|b| + 2^-7` |
| fp16 | `|a-b| <= 2^-10*|b| + 2^-10` |

`v` 是纯拷贝，必须逐字节相等。主场景实测：q `max_abs = 1.56e-2`、
k `1.55e-2`、v `0`。

## 性能

平台 `Ascend950PR_9579`，UB 253952 B；向量主频随负载在 1.25–1.6 GHz 之间漂移，
下表为**同一窗口内多次采集的中位**（每次 launch 取 kernel 时间与 `aiv_total_cycles`）。

### 主场景（ACLNN 路径，`S=12915`、`[6,2,2]`、stride 1286、bf16）

| 指标 | 值 |
| --- | ---: |
| kernel 时间 | **22.34 us** |
| `aiv_total_cycles` | **1,754,621** |
| 有效带宽 | `72.8 MB / 22.34 us ≈ 3.26 TB/s` |

### 各用例（同一 kernel 的直调 harness，`aclrtEvent` 多次平均、整 kernel）

| 用例 | 形状 | 耗时 |
| --- | --- | ---: |
| 主场景 BSND | `S=12915` `[6,2,2]` stride 1286 | **26.2 us** |
| BNSD | 同上 | 24.5 us |
| `gqa_expand` | 同上 | 26.8 us |
| `S=1` | `[6,2,2]` | 2.18 us |
| `S=7` fp16 | `[6,2,2]` | 2.28 us |
| `[8,1,1]` | `S=414` | 4.49 us |
| `[8,2,2]` | `S=259` stride 1536 | 3.74 us |
| `ropeStride=128` | `S=1024` | 5.25 us |

> 直调 harness 用 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 报 56 核，
> ACLNN 用 `GetCoreNumAiv()` 报 54 核，两者每核 tile 数不同，故主场景
> ACLNN 22.3 us 与直调 26.2 us 不直接可比，比较应各自看同路径数字。

### 瓶颈

向量流水（`vec`）是第一约束，`mte2`（GM→UB 搬入）第二。`gqa_expand` 的 k/v 三倍写出
由三级流水覆盖，不构成瓶颈（写带宽已接近上限但已被隐藏），该用例的耗时与不扩张时相当。

## 测试

`test/python_test/test_split_qkv_norm_rope.py`，覆盖 11 组形状 × 2 种 dtype（共 22 条）：
主场景、BNSD、`gqa_expand`、无 tail、单 token、尾 tile、文本侧 `[8,1,1]`/`[8,2,2]`、
`B>1`、`ropeStride=128`、BNSD+gqa。参考实现用 torch 逐元素复算；
`v` 用 `atol=rtol=0` 校验纯拷贝语义。

端到端验证（编译算子包 → 安装 → 编译 python 扩展 → pytest）：

```bash
# 1) 改过 kernel 必须先清缓存，框架对 op_kernel 的拷贝只做一次
rm -rf xllm_ops/build/binary xllm_ops/build/split_qkv_norm_rope xllm_ops/build_out xllm_ops/output

# 2) 重编算子包并安装
cd xllm_ops
bash build_aclnn.sh "$PWD" ascend950 "" split_qkv_norm_rope
mkdir -p output/_cann_ops_custom
bash build/cann-ops-xllm-custom_linux-x86_64.run --install-path=output/_cann_ops_custom

# 3) 使能 vendor 环境后编译扩展并跑用例
source output/_cann_ops_custom/vendors/custom_xllm_math/bin/set_env.bash
cd ../test/python_test
python3 setup.py build bdist_wheel && pip3 install --force-reinstall --no-deps dist/*.whl
python3 -m pytest -q test_split_qkv_norm_rope.py
```

上板结果（`Ascend950PR_9579`，单卡）：**22 passed**。
