# Multi-Latent Attention (MLA) 算子实现分析(BF16 数据类型）

> 本文档分析 `xllm_ops/multi_latent_attention` 算子在 Ascend AscendC 平台上、**数据类型为 BF16** 时的实现。
> - **Host 侧**:聚焦 Tiling 切分策略(tiling 参数、核数计算、任务数、多核分配),与 INT8 基本一致,差异点单独标注。
> - **Kernel 侧**:聚焦 **BF16 数据类型**的实现(业务处理流程、函数逻辑、数据获取、地址计算),重点对比与 INT8 的差异。
> - 参考文档:`multi_latent_attention_desc.md`(INT8 版本)。

---

## 1. 算子概述

Multi-Latent Attention(MLA)是 DeepSeek 系列模型使用的注意力机制,核心特点是把 KV Cache 压缩到一个低秩的隐空间(latent),从而大幅降低 KV Cache 显存占用。本算子实现的是 **decode(增量推理)阶段**的 MLA,基于 **PagedAttention** 的 block_table 机制管理 KV Cache。

算子采用 Ascend **MIX AIC/AIV** 架构(`KERNEL_TYPE_MIX_AIC_1_2`,即 1 个 Cube 核搭配 2 个 Vector 核):

- **Cube 侧(AIC)**:类 `MLAttentionDecoderAic`,负责两次矩阵乘 —— QK^T(mm1)与 PV(mm2)。
- **Vector 侧(AIV)**:类 `MLADecoderAiv`,负责 Softmax 与 flash-attention 在线累加(online rescale)。**BF16 下不含反量化(DeQuant)/量化(Quant)步骤。**

两侧通过 Workspace 上的中间 GM buffer 和跨核同步原语(FftsCrossCoreSync)协作,形成流水:
`QK^T(Cube) → Softmax(Vector) → PV(Cube) → Online Rescale 输出(Vector)`。

### 输入/输出

| 序号 | 名称 | 说明（BF16 场景） |
|------|------|------|
| 0 | query | Q 主体(**bf16**,hidden=576,含 nope 512 + rope 64) |
| 1 | queryRope | Q 的 rope 部分(bf16,hidden=64) |
| 2 | kvCache | KV Cache 主体(**bf16**,支持 ND / NZ 格式) |
| 3 | kvCacheRope | KV Cache 的 rope 部分(bf16) |
| 4 | block_tables | PagedAttention 块表 |
| 5 | contextLens | KV 序列长度 |
| 6 | mask | 注意力 mask |
| 7 | qSeqlen | Q 序列长度 |
| 8 | qkDescale | **BF16 场景不使用**（无 QK 反量化） |
| 9 | pvDescale | **BF16 场景不使用**（无 PV 反量化） |
| 10 | attenOut | 注意力输出(bf16) |
| 11 | lseOut | log-sum-exp 输出(ring 场景) |

> BF16 场景下 `qkDescale`/`pvDescale` 两个量化 scale 输入不参与计算(全程无量化)。

### 数学定义与含义

本算子在 **decode 阶段**为每个 query token 计算一次标准的缩放点积注意力(scaled dot-product attention),但 K/V 来自 MLA 压缩的低秩隐空间,并按 PagedAttention 组织。

**1) 基础注意力公式**

对第 `h` 个 head、当前 query 向量 `q_h`(与其历史 KV 序列 `K_h, V_h`,长度 = 上下文长度 `L`):

```
Attn_h = softmax( (q_h · K_hᵀ) / √d + mask ) · V_h
```

其中 `d` 为 head 维度,缩放系数 `tor = 1/√d`(host 侧算好写入 tiling 的 `TILING_TOR`)。

**2) MLA 的 rope 拼接**

MLA 把 Q/K 拆成**压缩主体**(nope,hidden=512)与 **rope 位置编码部分**(hidden=64),QK^T 分数是两部分之和:

```
score = q_nope · k_nopeᵀ + q_rope · k_ropeᵀ
```

**BF16 下 nope 与 rope 不再拆成两条不同精度的 MMA**:因为主体本身就是浮点,rope 也是浮点,二者可以拼成 **hidden=576** 的统一 bf16 矩阵乘一次算出,无需像 INT8 那样把主体走 int8、rope 单独走 float 再相加(见 §5.3、§6.4)。

**3) BF16 下的等价计算（无量化）**

Q、K、P 全程以 bf16 存储,矩阵乘在 float(fp32)域累加,直接得到结果,无 scale 还原:

```
score  = (Q_bf16 · K_bf16ᵀ)            # bf16 × bf16 → float(hidden=576 一次算完，含 rope)
P      = softmax(score × tor + mask)    # 概率 ∈ [0,1]，float 域
Attn_h = (P_bf16 · V_bf16)              # bf16 × bf16 → float，直接累加
```

即 **没有 DeQuant(QK)/Requant(P)/DeQuant(PV) 三个量化点**;所有矩阵乘输入为 bf16、累加为 float，Softmax 全程 float,概率转回 bf16 仅用一次 `Cast`(不带 scale)。

**4) Flash-Attention 在线累加(online softmax)**

与 INT8 完全一致。KV 按 block(block_size=64)逐段计算,采用 flash-attention 的在线归约。设历史最大值 `gm`、历史分母 `gl`、历史加权输出 `go`,新 block 的局部最大 `hm`、局部行和 `ll`、局部输出 `lo`:

```
m_new = max(gm, hm)
dm    = exp(gm - m_new)          # 历史项 rescale 因子
gl    = dm · gl + ll             # 分母(归一化因子)累加
go    = dm · go + lo             # 分子(∑ P·V)累加
gm    = m_new
```

全部 block 处理完后归一化输出:

```
attenOut_h = go / gl
lseOut_h   = gm + log(gl)        # ring/分布式场景需要的 log-sum-exp
```

---

## 2. 算子注册与数据类型

算子注册见 `op_host/multi_latent_attention_def.cpp`。**BF16 场景**的关键特征:

- `query`、`kvCache` 数据类型为 `DT_BF16`;
- `kvCache` 的 Format 可为 `FORMAT_ND`(TILING_KEY 1)或 `FORMAT_FRACTAL_NZ`(TILING_KEY 17);
- `queryRope`/`kvCacheRope` 同为 bf16。

对应的模板实例化(见 `op_kernel/multi_latent_attention.cpp`):

```cpp
// TILING_KEY 1: bf16(IN) + bf16(OUT), ND 格式
MLAttentionDecoderAic<TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, ND_FORMAT>
// TILING_KEY 17: bf16(IN) + bf16(OUT), NZ 格式
MLAttentionDecoderAic<TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, NZ_FORMAT>
```

模板参数含义:输入类型 `__bf16`、rope 类型 `__bf16`、输出类型 `__bf16`、KV 类型 `__bf16`、输入格式 `ND_FORMAT/NZ_FORMAT`。**注意 INT8 用 5 个不同的类型参数(int8/half/half/int8),而 BF16 五个数据类型参数全部是 `__bf16`。**

### 2.1 AttentionType 类型萃取(BF16 vs INT8）

`AttentionType<>` 特化决定 mm1/mm2 的中间累加类型(见 `multi_latent_attention.h`):

| 成员 | BF16 (= HALF) | INT8 |
|------|--------------|------|
| mm1OutputType / mm1CopyType | `float` | `int32_t` |
| mm2OutputType / mm2CopyType | `float` | `int32_t` |
| mmBiasType / mmScaleType | `float` | `float` |

**BF16 与 fp16(HALF)的类型萃取完全相同**:两次矩阵乘的输出/搬运类型均为 `float`,即 bf16×bf16 累加到 float,不存在 int32 量化域。这是 BF16 与 INT8 在 kernel 层最根本的区别。

---

## 3. TilingKey 生成规则

见 `MLATiling()` → `GenTilingKey()`(`op_host/multi_latent_attention_tiling_impl.cpp`):

```cpp
uint32_t dataType = static_cast<int32_t>(mmInfo.type);
uint32_t tilingKey = dataType
                   + (mmInfo.kNz      << 4)   // KV 是否 NZ 格式
                   + (mmInfo.mtpTp1Flag << 2) // 是否 MTP/TP1 分支(numHeads==128)
                   + (param.isRing    << 5);  // 是否 ring attention
```

其中 `dataType` 取值(`GetTilingKeyTypeBase()`):

| type 值 | 枚举 | 含义 |
|---------|------|------|
| 0 | TILING_HALF_DATA | fp16 |
| **1** | **TILING_BF16_DATA** | **bf16** |
| 2 | TILING_INT8_HALF_DATA | int8 输入 / fp16 输出 |
| 3 | TILING_INT8_BF16_DATA | int8 输入 / bf16 输出 |

BF16 判定:当 `query` 为 bf16 时 `dataType = 1`。

**BF16 常见 TILING_KEY 组合**(`dataType=1`):

| TILING_KEY | 组合 | 计算式 |
|------------|------|--------|
| 1 | bf16 + ND | `1` |
| 17 | bf16 + NZ | `1 + (1<<4)` |
| 5 | bf16 + ND + TP1 | `1 + (1<<2)` |
| 21 | bf16 + NZ + TP1 | `1 + (1<<4) + (1<<2)` |
| 33 | bf16 + ND + ring | `1 + (1<<5)` |
| 49 | bf16 + NZ + ring | `1 + (1<<4) + (1<<5)` |
| 37 / 53 | bf16 + ring + TP1(ND/NZ) | `+(1<<2)` |

**与 INT8 的关键区别**:INT8 恒走 18/19(强制 NZ、不支持 TP1);**BF16 支持 ND 与 NZ 两种格式,且支持 MTP/TP1 分支**(`mtpTp1Flag = (numHeads == 128) && (type < 2)`,BF16 的 type=1 < 2 满足条件)。

---

## 4. Host 侧 Tiling 实现

Host Tiling 逻辑(`op_host/`)在 BF16 与 INT8 之间**基本一致**,仅在 workspace 各段的数据类型/字节大小与 hidden 维度上有差异。核心常量:

| 常量 | 值 | 含义 |
|------|-----|------|
| TILING_HEAD_SIZE | 15 | tiling 头部字段数 |
| TILING_PARA_SIZE | 8 | 每个 batch 的字段数 |
| BATCH_MLA | 32 | 典型 batch |
| BLOCK_DIM_MLA | 20 | batch==32 时固定 20 个 Cube 核 |
| M_LIMIT | 128 | 单次处理的 M 上限 |

### 4.1 核数与任务数

- `totalTaskNum = Σ qSeqLen`(decode 阶段每个 batch 的 qSeqLen 通常为 1,故 ≈ batch)。
- `blockDim = GetCoreNumAic()`;当 `batch == 32` 时固定使用 20 个核。
- 任务按 **round-robin** 方式在核间轮转分配。

### 4.2 tiling 数据布局

- **头部 15 项**:全局参数(numHead、hidden、tor、block_size、page 相关等)。
- **每 batch 8 项**:`qSeqLen`、`kvSeqlen`,以及 query / kvCache / block_table 三组地址的高低 32 位。地址随 batch 逐个累加算出。

### 4.3 Workspace 分段

Workspace 划分为 6 段中间 GM buffer,BF16 场景各段按 **浮点(float/bf16)** 大小分配(INT8 场景 s_gm 走 int32、p_gm 走 int8):

| 段 | 名称 | 用途 | BF16 类型 |
|----|------|------|-----------|
| 1 | s_gm | QK^T 分数 | float |
| 2 | s_rope_out_gm | (INT8 专用 rope 分数) | **BF16 不使用** |
| 3 | p_gm | Softmax 概率 | bf16(OUT_DTYPE) |
| 4 | o_tmp_gm | PV 中间输出 | float |
| 5 | go_gm | 在线累加输出 | float |
| 6 | tmp_gm | 临时 buffer | float |

> BF16 下 `s_rope_out_gm` 段不参与计算 —— rope 已并入 hidden=576 的统一 MMA,分数直接落 `s_gm`。

---

## 5. Kernel 入口与 BF16 数据流

### 5.1 入口分发

`op_kernel/multi_latent_attention.cpp` 中 `extern "C"` 入口按 `TILING_KEY_IS` 分发。BF16 分支:

```cpp
if (TILING_KEY_IS(1) || TILING_KEY_IS(17) /* ND / NZ */) {
    // AIC: MLAttentionDecoderAic<TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, ND/NZ_FORMAT>
    // AIV: MLADecoderAiv<TILING_BF16_DATA, __bf16, __bf16 [, ring, BlockStack, TP1]>
}
```

内核首先解析 6 段 workspace GM 地址,再按 Cube / Vector 角色进入各自主循环。

### 5.2 BF16 数据流(对比 INT8）

```
                 ┌──────────────── AIC (Cube) ────────────────┐
 query/kvCache ─▶│ QK^T:  bf16 × bf16 → float (hidden=576,     │─▶ s_gm(float)
   (bf16)        │        nope+rope 一次 MMA 算完)             │
                 └────────────────────────────────────────────┘
                                    │ QK_READY
                                    ▼
                 ┌──────────────── AIV (Vector) ──────────────┐
 s_gm(float) ───▶│ SoftmaxStage1: 读 float s_gm → ×tor → +mask │─▶ p_gm(bf16)
                 │   → rowmax → flash(max/dm) → exp             │
                 │   → Cast(float→bf16)   ★无 Requant           │
                 └────────────────────────────────────────────┘
                                    │ SOFTMAX_READY
                                    ▼
                 ┌──────────────── AIC (Cube) ────────────────┐
 p_gm(bf16) ────▶│ PV:  bf16 × bf16 → float                    │─▶ o_tmp_gm(float)
                 └────────────────────────────────────────────┘
                                    │ UPDATE_READY
                                    ▼
                 ┌──────────────── AIV (Vector) ──────────────┐
 o_tmp_gm ──────▶│ SoftmaxStage2: 读 float o_tmp → online       │─▶ attenOut(bf16)
                 │   rescale(gl=dm·gl+ll / go=dm·go+lo)         │─▶ lseOut(ring)
                 │   → 末 block go/gl → Cast   ★无 DeQuant       │
                 └────────────────────────────────────────────┘
```

**与 INT8 流程图的三处删减**:
1. QK^T 后**没有** DeQuant(int32×qkDescale→float);BF16 直接产出 float。
2. SoftmaxStage1 后**没有** Requant(×1/127 → int8);BF16 直接 Cast float→bf16。
3. PV 后**没有** DeQuant(int32×pvDescale→float);BF16 直接产出 float。

### 5.3 hidden 维度

BF16 场景 `hidden_size = 576`(nope 512 + rope 64,一并做 MMA);INT8 场景 `hidden_size = 512`(rope 64 单独走 float MMA 落 `s_rope_gm`)。`n_loop = (cur_kv_seqlen + pp_n - 1) / pp_n`。

---

## 6. AIC(Cube 侧)`MLAttentionDecoderAic`

### 6.1 SetArgs / Run

`Run()` 按 round-robin 领取本核负责的 (batch, task) 任务,循环调用 `InnerRunCubeMLA()`。

### 6.2 Q 地址与 L1 搬运

- 用 tiling 中每 batch 的地址高/低 32 位拼接出 64 位 Q / kvCache / block_table GM 地址。
- 将 Q 搬入 L1。

### 6.3 n_loop 与 block_table 定位

按 KV 序列长度切成 `n_loop` 个 block,通过 `block_table` 定位每个 KV block 在 Cache 中的物理页。

### 6.4 CUBE1(QK^T,BF16 关键差异）

BF16 走 **embed_split 5 段**(4×128 + 64 = 576),但**统一用 bf16 mmad 一次算完**:

```
非 INT8 / BF16 分支(multi_latent_attention.h L947-1003):
  for idx in [0..4]:        # 5 段 embed_split,累加到同一 L0C
      mmad(bf16 × bf16 → float accumulate)
  idx == 4 结束后: 一次 l0c_to_gm 写 s_gm(float)
  ★无 rope 独立分支、★无 dequant
```

对比 INT8:INT8 在 `idx == 3` 时把主体分数(int32)写 `s_gm`,`idx == 4` 单独把 rope 部分以 float MMA 写 `s_rope_gm`,后续由 Vector 侧 DeQuant 再相加。**BF16 因全程浮点,rope 直接并入统一 MMA,少一次 GM 往返与一个量化点。**

### 6.5 CUBE2(PV)

用 `LoadDataWithTranspose` 把概率 `p_gm`(bf16)与 V(bf16)做矩阵乘,累加到 float 落 `o_tmp_gm`。

### 6.6 跨核同步

`FftsCrossCoreSync` 依次发出 `QK_READY → SOFTMAX_READY → UPDATE_READY`,与 Vector 侧握手。

---

## 7. AIV(Vector 侧)`MLADecoderAiv`

`InnerRunVectorChange()` 将 head 按 `sub_block_idx`(0/1)分给两个 Vector 核各半,用 `n_loop + 1` 的软流水、`n_idx % 2` ping-pong buffer,使 Stage1 与 Stage2 错位一拍并行。

### 7.1 SoftmaxStage1(BF16 分支,`multi_latent_attention.h` L2517-2796）

```
BF16 (else 分支):
  gm_to_ub  : 直接把 float 的 s_gm 搬入 ls32_ubuf   ★无 DeQuantPerHeadImpl、★无 s_rope_gm 相加
  mask      : DataCopy + Cast(载入 mask)
  muls(tor) : ls × tor(分 FLOAT_VECTOR_SIZE 段 + 尾段)
  mask Add  : + mask
  ReduceMaxRepeatM : 行最大 lm
  flash     : n_idx!=0 → hm=max(lm,gm), dm=exp(gm-hm); else hm=lm; gm=hm
  TensorSubValueRepeatM : ls - hm
  exp_v     : exp(ls - hm)
  conv_v    : float → OUT_DTYPE(bf16)          ★无 QuantPerTokenImpl(不乘 1/127、不转 int8)
  ub_to_gm  : 写 p_gm(bf16)
  ReduceSumRepeatM : 行和 ll
```

对比 INT8:INT8 分支先做 `DeQuantPerHeadImpl`(s_gm×qkDescale→float)、把 `s_rope_gm` 以 float 载入相加,末尾用 `QuantPerTokenImpl`(×1/127 转 int8)。**BF16 三处量化相关操作全部省去。**

### 7.2 SoftmaxStage2MLAHeadLoop(BF16 分支,L2798-3157）

```
n_idx != 0:
  gm_to_ub  : 读 o_tmp_gm(float)作为 lo           ★BF16 无 DeQuantPerHeadImpl
  head_loop_idx==0: exp(dm); gl = dm·gl + ll
  brcb dm → tv;  go = go·dm_block(分段 mul_v);  go = go + lo
n_idx == 0:
  gl = ll;  gm_to_ub 读 o_tmp_gm 作为 go           ★BF16 无 DeQuant

末 block (n_idx == n_loop-1):
  gl_block brcb;  go = go / gl_block(div_v 分段)
  conv_v : go(float) → OUT_DTYPE(bf16)
  DataCopyPad 写 o_gm(head_res / numhead_per_process / tail 三段)
  IS_RING: ln(gl) + gm → lse → conv_v → ub_to_gm_align 写 lse_gm
否则 head_loop>1: ub_to_gm 写 go_gm
```

**BF16 全程 float 在线 rescale,无 PV 反量化步骤。** `process_row_num = 16` 分块,ring 场景额外输出 `lseOut`。

### 7.3 TP1 路径

`numHeads == 128` 时走 TP1 特化:`SoftmaxStage2MLAHeadLoopTP1` / `TailSoftmaxStage2MLAHeadLoopTP1` / `SoftmaxGatherTP1`,以及 `OnlineSoftmaxStage1<float,float,IN_DTYPE,IN_DTYPE,MASK_TYPE_NONE>`(`multi_latent_attention_npu.h`)。**该路径 BF16 可用,INT8 不支持。**

---

## 8. 小结:BF16 与 INT8 的核心差异

| 维度 | BF16 | INT8 |
|------|------|------|
| mm1/mm2 累加类型 | float(bf16×bf16→float) | int32(int8×int8→int32) |
| 三量化点 | **全部无** | DeQuant(QK) / Requant(P,×1/127) / DeQuant(PV) |
| hidden_size | **576**(nope+rope 一体 MMA) | 512(rope 64 独立 float MMA) |
| s_rope_gm 段 | 不使用 | 使用(rope 分数) |
| QK^T | 5 段 embed_split 统一 mmad,一次写 s_gm | idx==3 写 s_gm、idx==4 rope 单独写 s_rope_gm |
| SoftmaxStage1 输出 | conv_v float→bf16 写 p_gm | Requant ×1/127 → int8 |
| SoftmaxStage2 | float online rescale,无 DeQuant | DeQuant PV×pvDescale + online rescale |
| 支持格式 | ND(1) / NZ(17) | 强制 NZ(18/19) |
| TP1 分支 | 支持(5/21/37/53) | 不支持 |
| qkDescale/pvDescale | 不使用 | 必需 |

**一句话总结**:BF16 路径相较 INT8 —— **去掉全部三个量化点、hidden 统一为 576(rope 并入主 MMA)、矩阵乘直接 bf16×bf16→float 累加、Softmax 全程 float,概率仅用一次不带 scale 的 Cast 转回 bf16**。数学上与标准缩放点积注意力 + flash online softmax 完全等价,计算链路比 INT8 更短更直接。