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
- UB 需求 `<= 253952 B`（主场景实测 175104 B）。

## 算法

对第 `t` 个 token 的第 `h` 个 q/k 头（`x = qkv[t, h*D : (h+1)*D]`，fp32）：

```text
sumsq  = Σ_{i=0}^{D-1} x_i^2
scale  = 1 / sqrt(sumsq / D + eps)
y_i    = x_i * scale * gamma_i
x1     = y[0:64],  x2 = y[64:128]
out    = [ x1*cos - x2*sin ,  x2*cos + x1*sin ]
```

`v` 头不做 norm/rope，只按 dtype 原样搬（fp32 中转后回写，因输入行距非 32B 对齐，
无法直接 DMA 到输出布局）。

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
fp32 行距      = padStride * 4 = 5184 B   # 5184 = 162*32
```

搬入按 token 逐个 `DataCopyPad`（`blockLen = stride*2`），硬件把每块右侧补 dummy
到 32 B 对齐，正好填满 padding 区；`Cast` 同样按 token 分段。

## Tiling

`tokenTile` 固定 8（`stride*2` 与 32 B 的对齐周期决定），核间按 token 连续段切分：

```text
blockDim     = min(aivNum, ceil(totalTokens / 8))
tokenPerCore = RoundUp(ceil(totalTokens / blockDim), 8)
blockDim     = ceil(totalTokens / tokenPerCore)      # 回算
lastBlockTokenNum = totalTokens - tokenPerCore*(blockDim-1)
```

主场景（`S = 12915`、56 AIV）：`blockDim = 56`、`tokenPerCore = 232`、
`lastBlockTokenNum = 155`，每核 29 个 tile。

TilingData 字段：`batch / seqLen / numHeadQ / numHeadK / numHeadV / headDim /
qkvStride / padStride / ropeStride / totalTokens / tokenPerCore / tokenTile /
lastBlockTokenNum / eps / reciprocal / flags / blockDim`。
`flags` 用位域替代 tilingKey 做布局与 GQA 分支：`bit0 = gqa_expand`、`bit1 = layout_bnsd`；
tilingKey 只区分 dtype（`0 = bf16`、`1 = fp16`）。

## Kernel 实现要点

- **编程模型**：norm + rope 走 SIMD-Regbase（`__simd_vf__` + `Reg::*`）。
  `Reg::ReduceSum` + `StoreAlign<DIST_FIRST_ELEMENT_B32>` 落 sumsq，
  `LoadAlign<DIST_BRC_B32>` 广播回寄存器，`Muls→Adds→Sqrt→Div` 求 scale
  （950 无 `Reg::Rsqrt`）。
- **两级跳过写**：pass1 只算每 `(token, head)` 的 sumsq，pass2 再算 norm+rope，
  整个 tile 只需 1 次 `LocalMemBar`。
- **RoPE 融合**：高半用 `MulAddDst`（`dst += src0*src1`）把 `mul + add` 压成一条。
- **三级流水**：`inQkv_/inCos_/inSin_/outTile_` 全部深度 2，
  循环体为「预取 tile k+1 → 计算 tile k → 写出 tile k-1」，
  使 `gqa_expand`（3 倍 k/v 写出）的 mte3 与计算重叠。

## 精度

fp32 中间量、bf16/fp16 输入输出，判据按 dtype：

| dtype | 判据 |
| --- | --- |
| bf16 | `|a-b| <= 2^-7*|b| + 2^-7` |
| fp16 | `|a-b| <= 2^-10*|b| + 2^-10` |

`v` 是纯拷贝，必须逐字节相等。主场景实测：q `max_abs = 1.56e-2`、
k `1.55e-2`、v `0`。

## 性能

平台 `Ascend950PR_9579`，56 AIV，UB 253952 B，向量频率约 1592 MHz。
数据由同一 kernel 的直调 harness 采集（`aclrtEvent` 20 次平均、整 kernel）；
走 aclnn 路径时只多一次算子调度开销。

| 用例 | 形状 | 耗时 |
| --- | --- | ---: |
| 主场景 BSND | `S=12915` `[6,2,2]` stride 1286 | **38.0 us** |
| BNSD | 同上 | 38.0 us |
| gqa_expand | 同上 | 38.7 us |
| 无 tail 列 | stride 1280 | 37.8 us |
| `S=1614` | `[6,2,2]` | 8.4 us |
| `S=1` | `[6,2,2]` | 2.75 us |
| `S=7` fp16 | `[6,2,2]` | 2.95 us |
| `[8,1,1]` | `S=414` | 4.8 us |
| `[8,2,2]` | `S=259` stride 1536 | 4.3 us |
| `B=2` | `S=1024` | 9.7 us |
| `ropeStride=128` | `S=1024` | 6.9 us |
| BNSD + gqa | `S=1024` | 7.1 us |

主场景有效带宽 `72.8 MB / 38.0 us ≈ 1.92 TB/s`。msprof 采集（7 组 aic-metrics）：
`vec_ratio 93%`、`mte2 23%`、`mte3 15%`、头开销 2.1%，即**向量流水是主约束**。

### ACLNN 路径实测（msprof，8 次 launch 取中位数）

采集口径：单算子直调（S=12915、`[6,2,2]`、stride 1286、bf16、warmup 3 + 计时 5），
7 组 `--aic-metrics` + 1 次 sample-based。

| 指标 | 中位数 | 判读 |
|---|---:|---|
| Task Duration | **38.42 us** | 首次冷启动 47.85 us（L2 冷），稳态 37.4–39.9 |
| `aiv_time` | 38.18 us | 头开销 0.80 us（1.7%） |
| `aiv_total_cycles` | 3,388,354 | 反推主频 ≈ **1585 MHz**（满频） |
| `aiv_vec_ratio` | **92.9%**（35.35 us） | **向量流水是主约束** |
| `aiv_scalar_ratio` | 36.6%（14.01 us） | 地址/循环标量，与 VEC 并行 |
| `aiv_mte2_ratio` | 26.0%（9.97 us） | 搬入已被双缓冲盖住 |
| `aiv_mte3_ratio` | 19.1%（7.28 us） | 搬出同上 |
| `aiv_icache_miss` | 0% | 无取指瓶颈 |
| `vec_bank_cflt` | **92.6%** | 向量访问 UB 几乎总在冲突 |

**与直调路径对比**：同一 kernel 直调测得 38.0 us / 3.37 M cycles，aclnn 路径
38.42 us / 3.39 M cycles，差 1%；host 侧 `perf_counter` 44.0 us/iter < kernel 时间，
说明 aclnn 的调度与 launch 完全被 kernel 掩盖，**融合成 1 个算子的收益没有被封装吃掉**。

**瓶颈判定**：单个 tile 的向量指令约 2088 条（cast-in 168 + pass1 512 + pass2 1216
+ v 直通 32 + cast-out 160），每核 29 tile ≈ 6.05 万条；38.2 us × 1.59 GHz ≈ 6.07 万
cycle，即**向量流水已跑满 ~1.0 指令/cycle**。所以：

### 已落地的优化（round1，`ops-perf-optimize` 工作流）

冻结 5 条 case 走完「分析 → 3 方案隔离实现 → 同一 msprof 协议各采一次 → 唯一终选」：

| 方案 | c1_main | c2_bnsd | c3_gqa | c4_single | c5_fp16 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 42.460 | 42.415 | 42.442 | 7.043 | 7.873 | — |
| **vf-instr-cut** | **35.951** | **36.089** | **40.302** | 7.232 | 8.145 | **选中**：3 提升 / 0 回退，有效几何加速 1.135 |
| partial-tile | 42.553 | 51.310 | 42.600 | 6.540 | 8.137 | 淘汰（c2 +21%） |
| mte-pipeline | 50.675 | 49.875 | 50.621 | 7.556 | 8.611 | 淘汰（cycles +18.4%） |

实现（已合并进本工程 `op_kernel/split_qkv_norm_rope.h`）：

1. pass1 每 (t,h) 9 → 6 条：`Mul` 起手 + 一次 `MulAddDst` 累加；
2. sumsq 行距 `kMaxHeads(16) → Nq+Nk(=8)`，整 tile 64 个 sumsq 占满一个寄存器宽度；
3. scale 链从 per-head 提到**每 tile 一次 64-lane**，per-head 只留 1 次 `DIST_BRC_B32` 载入；
4. γ 折进 cos/sin：`a=γ1·cos、c=γ1·sin、d=γ2·cos、b=-γ2·sin`；
5. RoPE 每半 1 `Mul` + 1 `MulAddDst`。

运行期兜底（`fastScale_ && !smallTile_` 才走快路径，否则 `SqnrNormRopeVfLegacy` = 基线 VF 逐字副本）：

- `Nq+Nk != 8`（`[8,1,1]`/`[8,2,2]`）：64-lane 前提不成立；
- 每核只有 1 个 tile（`S=1`、`S=7`）：tile 级 prologue 摊不掉。

> 教训：兜底最初用 tensor 接口"等价改写"基线语义，实测 c4/c5 **+156%/+168%**；必须逐字复用同一份 VF
> 源码（现在 c4/c5 输出与基线**逐字节相同**）。

cycles `3,682,564 → 3,004,483`（**−18.4%**），`vec 0.841→0.670`、`mte2 0.683→0.786`：
**瓶颈已从向量发射迁移到 GM→UB 搬入**。合并后回归：直调 8/8 PASS、ACLNN `pytest` 22/22 PASS。

### round2 三项小改进（pack-store 直写，已合并）

| # | 改动 | 收益来源 |
|---|---|---|
| 1 | 输出改寄存器 pack-store 直写（`Cast<T,float,RndRSatNS>` + `StoreAlign<T,DIST_PACK_B32>`） | 整趟 cast-out（160 条/tile）消失 |
| 2 | v 不再经过 fp32：快路径一次多行 UB→UB `DataCopy`（32B 块单位：`blockLen=16, srcGap=65, dstGap=64`） | VF 少 4 条/head，省 v 的 cast-in |
| 3 | 快路径 cast-in 只转 q/k 头（`(Nq+Nk)*128`） | 少 40 条/tile |

实现注意：pack-store 目标行距必须用**输出 tile 行距** `numHeadTotal*128`（不是 q/k 行宽 1024），
否则 q/k 覆盖 v 区；`LayoutZMrgZRndRSatNS` 需写 `AscendC::LayoutZMrgZRndRSatNS`；
v 拷贝必须发生在 `inQkv_.FreeTensor(xb)` 之前，回落路径仍走 fp32（Legacy VF 不变）。

| 指标（ACLNN 同身份，8 次中位） | 原始 | round1 | **round2** |
|---|---:|---:|---:|
| Task Duration | 38.42 us | 29.50 us | **24.23 us** |
| cycles | 3,388,354 | 2,559,250 | **2,069,996** |
| vec 时间 / 占比 | 35.35 / 0.929 | 25.21 / 0.862 | **20.98 / 0.876** |
| scalar | 14.01 / 0.366 | 15.72 / 0.534 | 15.70 / **0.654** |
| mte2 | 9.97 / 0.260 | 14.19 / 0.486 | 11.28 / 0.471 |
| mte3 | 7.28 / 0.191 | 7.67 / 0.265 | 8.24 / **0.344** |

**相对原始：时间 −36.9%、cycles −38.9%**。回归：直调 8/8 PASS、ACLNN 22/22 PASS（`v` 逐字节相同）。
瓶颈：`vec 0.876` 仍是第一约束，`scalar 0.654` 升为第二，下一步压 scalar（copy-out per-head 循环、
pass2 的 q/k 两段控制流）。

### scalar 管道调查（消融实验，未采纳）

| 实验 | 结果 | 结论 |
|---|---|---|
| 注入 256 次/tile 空转标量循环（写回 padding，结果不变） | c1 **24.72 → 40.01 us（+62%）** | **scalar 在关键路径、无余量**（≈3.2 cycle/iter） |
| 主场景 (Nq,Nk)=(6,2) head 循环改编译期常量、8×8 全展开 | 24.57 us，**0% 收益**，`.text` 增大 | scalar 开销**不在循环控制/地址乘** |

折算：每 tile 约 14 次 `DataCopyPad` + 10 次 `Cast` + ~20 次队列操作，`845 cycle/tile`
≈ **每调用 19 cycle 固定开销**。要继续压 scalar 只能减少「每 tile 的 API 调用次数」：

- cast-in 折进 VF（`DIST_UNPACK_B16`）：省 8 次 Cast，但每 head +4 条向量指令（+17% vec，已 0.876）→ 净负，不采纳。**该结论已被下一节推翻**：整体 −13~15% cycles（ACLNN 同身份 A/B）。
- 加大 `tokenTile` 8→16 摊薄固定开销：见下一节复核结论，**结构性放不下**。

### round3：快路径切 UNPACK 通路（已合并）

上一节的"cast-in 折进 VF 净负"判断只看了指令数。实测把快路径 VF 换成
`LoadAlign<T, DIST_UNPACK_B16>` + `Cast<float,T,layoutZMrgZ>` 后，整趟 cast-in（每 token 1024 元素）
与 `cos/sin` 的 fp32 副本一起消失，收益远大于每 head 多出的 2 条 Cast：

| 身份 | cycles | aiv_us | vec | scalar | mte2 |
|---|---:|---:|---:|---:|---:|
| pack（round2） | 2,056,916 | 23.55 | 0.883 | 0.658 | 0.409 |
| round3（UNPACK） | 1,779,198 | 25.47 | 0.837 | 0.622 | 0.558 |

第二轮顺序互换复核：2,071,702 → 1,761,314（−15.0%）。`aiv_us` 不可比（向量频率在 1247–1559 MHz 间漂移），
以 `aiv_total_cycles` 为准。

另外删除 `smallTile_` 单 tile 回落门槛（`kSingleTileMax`）：UNPACK 通路的 prologue 小了，
单 tile 场景反而更快（直调同窗口：S=1 `2.95 → 2.34 us`，S=7 fp16 `3.13 → 2.37 us`）。
现在分派只剩 `fastScale_`（`Nq+Nk != 8` 时才走 Legacy VF）。

`tokenTile=16` 复核：即便完全删掉回落路径，`inQkv_`(164,864 B) + cos/sin(8,192) + `outTile_`(98,304)
+ gamma/sumsq 仍需 **273,408 B > 253,952 B**；`outTile_` 降为深度 1 会与 `CopyOut(k-1)` 互等。
**结论：在 D=128、Nq+Nk+Nv=12 下 `tokenTile=16` 结构性放不下**，除非改成"16 token 计算 + 8 token 分段写出"。

功能验收：`pytest test_split_qkv_norm_rope.py` **22/22 PASS**。

### round4：放开 8 头限制（所有形状走同一条快路径）

`Nq+Nk != 8`（文本侧 `[8,1,1]`、`[8,2,2]`）此前走逐字复用的 Legacy VF，本轮改动完全不覆盖。
把 scale 链的段数改成运行期 `segs = ceil(kTokenTile*(Nq+Nk)/64)`（末段空 lane 只写回 `sumsq`，
pass2 只按 `(t,h)` 单点广播，不回读），即可删掉 Legacy VF、`fastScale_` 分派和四块 fp32 scratch
（`bufXf_`/`bufOutF_`/`bufCos_`/`bufSin_`），UB 需求 175,104 → **44,544 B**。

实测（空闲卡 NPU 3，每侧 3 个独立 msprof 会话 × 100 reps，取会话中位的中位）：

| case | 优化前 us | 优化后 us | cycles 变化 |
|---|---:|---:|---:|
| 1（主场景） | 29.33 | 26.16 | −30.0% |
| 2（BNSD） | 29.54 | 24.50 | −30.4% |
| 3（GQA 扩张） | 30.06 | 26.83 | −26.0% |
| 6（S=1） | 2.99 | 2.18 | −32.4% |
| 7（S=7 fp16） | 3.12 | 2.28 | −31.8% |
| 8（[8,1,1]） | 5.20 | 4.49 | −22.6% |
| 9（[8,2,2]） | 4.75 | 3.74 | −27.9% |
| 13（rope_stride=128） | 6.34 | 5.25 | −22.5% |

GQA 扩张的写带宽已用探针量化：把扩张写出头数从 6 降到 2（结果错误，仅测量）只省 3% cycles，
说明 3 倍 k/v 写量基本被流水掩盖，重组写模式没有价值；用例 3 的收益来自向量侧。

测量标准：本机 `npu-smi` 无 aicore 锁频接口（`-p` 参数被拒），改用空闲卡 + 每侧 3 会话 × 100 reps、
轮转顺序、取中位，并同时报 `aiv_total_cycles` 交叉验证。




- 剩下只有 2.8 us（7.4%）不属于 VEC，是 prologue/epilogue 与冷 L2；
- `vec_bank_cflt 92.6%` 说明这 1 IPC 里大量周期花在 UB 冲突上，**降冲突比降指令数更划算**；
- 想再快只能减少向量访存次数或改善 UB bank 分布，而不是加带宽（main_mem 远未饱和）。

**下一步候选（按性价比排序，均未实施）**

| # | 手段 | 预期 |
|---|---|---|
| 1 | 调 `padStride` 让行首相位避开 bank 冲突（现在 `RoundUp(stride,16)=1296`，改为 32/64 对齐试 A/B，看 `vec_bank_cflt` 是否下降） | 若冲突是真因，−10% 量级 |
| 2 | 输出改为寄存器直接 pack-store（`StoreAlign<T, DIST_PACK_B32>`），去掉 `bufOutF_` 与整趟 cast-out（160 条/tile） | −2~4% |
| 3 | pass1/pass2 拆成两个 `__simd_vf__`（不省指令，只改寄存器分配与调度） | 低，需 A/B |
| 4 | `sumsq`、`bufXf_`/`bufOutF_` 起始地址对齐到 1024 B，减少跨 bank 边界 | 待测 |

### 已实测无效或有害的优化方向

记录在此以免重复踩坑：

| 方向 | 结果 |
| --- | --- |
| pass2 按 gamma 拆成「q 组 / k 组」两个 callee，gamma 常驻寄存器 | 慢 1.5x（活跃寄存器翻倍） |
| 保留 x1/x2 跨 pass，省 2 次 load/head | 慢 2.6x（`LocalMemBar` 从 1 次/tile 变 64 次/tile，单次约 55 cycle） |
| v 走 bf16 `DataCopy(UB→UB)` 直通 | 慢约 8 us（8 次调用开销大于搬运量） |
| 把 `1/sqrt` 改成两次 `Div` | 慢约 1.8x 的 VF 段（fp32 Div 代价高） |
| MulAddDst 融合 RoPE 高半 | 有效，cycles −2.3% |
| 四个队列深度 2 + 三级流水 | 有效，gqa 路径 −10~13%，其余 −2% |

> 注意：该卡向量频率会在运行中从约 1592 MHz 掉到约 840 MHz（`npu-smi` 显示
> `Warning`）。**跨时间窗比较墙上时间无效**，需要比较 `aiv_total_cycles`
> （时钟无关）或同一窗口内交替 A/B。

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
