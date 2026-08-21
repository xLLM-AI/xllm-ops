# Multi-Latent Attention (MLA) 算子实现分析

> 本文档分析 `xllm_ops/multi_latent_attention` 算子在 Ascend AscendC 平台上的实现。
> - **Host 侧**:聚焦 Tiling 切分策略(tiling 参数、核数计算、任务数、多核分配)。
> - **Kernel 侧**:聚焦 **INT8 数据类型**的实现(业务处理流程、函数逻辑、数据获取、地址计算)。

---

## 1. 算子概述

Multi-Latent Attention(MLA)是 DeepSeek 系列模型使用的注意力机制,核心特点是把 KV Cache 压缩到一个低秩的隐空间(latent),从而大幅降低 KV Cache 显存占用。本算子实现的是 **decode(增量推理)阶段**的 MLA,基于 **PagedAttention** 的 block_table 机制管理 KV Cache。

算子采用 Ascend **MIX AIC/AIV** 架构(`KERNEL_TYPE_MIX_AIC_1_2`,即 1 个 Cube 核搭配 2 个 Vector 核):

- **Cube 侧(AIC)**:类 `MLAttentionDecoderAic`,负责两次矩阵乘 —— QK^T(mm1)与 PV(mm2)。
- **Vector 侧(AIV)**:类 `MLADecoderAiv`,负责反量化(DeQuant)、Softmax、量化(Quant)与 flash-attention 在线累加(online rescale)。

两侧通过 Workspace 上的中间 GM buffer 和跨核同步原语(FftsCrossCoreSync)协作,形成流水:
`QK^T(Cube) → Softmax(Vector) → PV(Cube) → Online Rescale 输出(Vector)`。

### 输入/输出

| 序号 | 名称 | 说明 |
|------|------|------|
| 0 | query | Q 主体(INT8 场景为 int8,hidden=512) |
| 1 | queryRope | Q 的 rope 部分(float/half/bf16,hidden=64) |
| 2 | kvCache | KV Cache 主体(INT8 场景 int8,NZ 格式) |
| 3 | kvCacheRope | KV Cache 的 rope 部分 |
| 4 | block_tables | PagedAttention 块表 |
| 5 | contextLens | KV 序列长度 |
| 6 | mask | 注意力 mask |
| 7 | qSeqlen | Q 序列长度 |
| 8 | qkDescale | QK^T 反量化 scale(per-head float) |
| 9 | pvDescale | PV 反量化 scale(per-head float) |
| 10 | attenOut | 注意力输出 |
| 11 | lseOut | log-sum-exp 输出(ring 场景) |

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

因此 kernel 里主体走 int8 量化 MMA,rope 部分单独走 float MMA,二者在 softmax 前相加(见 §5.3、§7.3)。

**3) INT8 量化下的等价计算**

主体 Q、K、P 均以 int8 存储,矩阵乘在 int32 域累加,再用 per-head/per-token scale 还原:

```
q_nope · k_nopeᵀ ≈ (Q_int8 · K_int8ᵀ) × qkDescale        # int32 → float,per-head 反量化
score            = 上式 + q_rope · k_ropeᵀ                # rope 恒为 float
P                = softmax(score × tor + mask)             # 概率 ∈ [0,1]
P_int8           = round(P × 127)                          # per-token 量化(scale = 1/127)
Attn_h           ≈ (P_int8 · V_int8) × pvDescale × (1/127) # int32 → float,per-head 反量化
```

即三个量化点:**DeQuant(QK)→ Requant(P)→ DeQuant(PV)**;rope 分支始终保持浮点精度。

**4) Flash-Attention 在线累加(online softmax)**

由于 KV 按 block(block_size=64)逐段计算,采用 flash-attention 的在线归约,避免一次性物化整条注意力矩阵。设历史最大值 `gm`、历史分母 `gl`、历史加权输出 `go`,新 block 的局部最大 `hm`、局部行和 `ll`、局部输出 `lo`:

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

算子注册见 `op_host/multi_latent_attention_def.cpp`。其中 query/kvCache 支持多种数据类型组合,**INT8 场景**的关键特征:

- `query`、`kvCache` 数据类型为 `DT_INT8`;
- `kvCache` 的 Format 为 `FORMAT_FRACTAL_NZ`(NZ 格式);
- `queryRope`/`kvCacheRope` 仍为浮点(fp16 或 bf16),rope 部分不量化。

对应的模板实例化(见 `op_kernel/multi_latent_attention.cpp`):

```cpp
// TILING_KEY 18: int8(IN) + fp16(OUT)
MLAttentionDecoderAic<TILING_INT8_DATA, int8_t, half, half, int8_t, NZ_FORMAT>
// TILING_KEY 19: int8(IN) + bf16(OUT)
MLAttentionDecoderAic<TILING_INT8_DATA, int8_t, __bf16, __bf16, int8_t, NZ_FORMAT>
```

模板参数含义:输入类型 `int8_t`、输出类型 `half/__bf16`、中间/bias 类型、量化类型 `int8_t`、输入格式 `NZ_FORMAT`。

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
| 1 | TILING_BF16_DATA | bf16 |
| 2 | TILING_INT8_HALF_DATA | int8 输入 / fp16 输出 |
| 3 | TILING_INT8_BF16_DATA | int8 输入 / bf16 输出 |

INT8 判定:当 `query` 不是 bf16/fp16 时进入 INT8 分支;再看 `queryRope` 是 fp16(→ type=2)还是 bf16(→ type=3)。

由于 INT8 的 KV Cache 强制 NZ(`kNz=1`,`<<4` 即 +16),最终 **INT8 走 TILING_KEY 18(fp16 输出)/ 19(bf16 输出)**:
- `2 + (1<<4) = 18`
- `3 + (1<<4) = 19`

INT8 不支持 MTP/TP1(`mtpTp1Flag` 要求 `type < 2`),因此 INT8 恒走非 TP1 的 `Run()` 路径。

---

## 4. Host 侧 Tiling 切分策略

Tiling 的入口是 `MLATiling()`,主要逻辑分布在:
- `op_host/multi_latent_attention_tiling_impl.cpp` —— 主入口、信息采集、TilingKey、Workspace 计算。
- `op_host/multi_latent_attention_tiling_dependency.cpp` —— tiling 参数填充、核数与任务分配。

### 4.1 关键常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `TILING_HEAD_SIZE` | 15 | tiling 头部(公共参数)占用的 uint32 个数 |
| `TILING_PARA_SIZE` | 8 | 每个 batch 任务参数占用的 uint32 个数 |
| `TILING_PARA_SIZE_TP1` | 4 | TP1 分支每个 task 的参数个数 |
| `BATCH_MLA` | 32 | 特殊 batch 数(触发固定核数) |
| `BLOCK_DIM_MLA` | 20 | batch==32 时固定使用的核数 |
| `M_LIMIT` | 128 | numHeads==128 时走 MTP/TP1 分支 |
| `PP_MM` | {16,32,...,128} | M 方向分块候选 |
| `QN_TILE_LIST` | {128,64,32,16,8,1} | Q head 方向的分块候选 |

### 4.2 信息采集(GetMLANdInfo)

从 `TilingContext` 提取形状与属性:
- **NZ 判定**:`kNz = (kvCache 末维 == 16 或 32) ? 1 : 0`。INT8 KV Cache 为 NZ,`kNz=1`。
- **embeddingSize / blockSize**:NZ 格式下 `embeddingSize = dim3 * dim1`,`blockSize = dim2`;ND 格式下取原始维度。
- **batch** = `kvSeqLen.size()`(KV 序列条数)。
- **numHeads** = 属性 `headSize`;**kvHeads** = `kvHead`(≤0 则等于 numHeads)。
- **mtpTp1Flag** = `(numHeads == 128) && (type < 2)` —— INT8 恒为 false。

### 4.3 任务数(totalTaskNum)计算

```cpp
if (mmInfo.qSeqLen != nullptr) {
    // 所有 batch 的 qSeqLen 之和
    mmInfo.totalTaskNum = accumulate(qSeqLen, qSeqLen + batch, 0);
} else {
    mmInfo.totalTaskNum = batch;   // decode 每 batch 一个 task
}
```

decode 场景每个 batch 的 qSeqLen 通常为 1,因此 **totalTaskNum 一般等于 batch**。该值写入 tiling 头部 `TILING_TASK_NUM` 供 kernel 侧划分 process。

### 4.4 核数(blockDim)计算

核数计算见 `MLATiling()` 与 `GetMLATilingParam()`:

```cpp
auto blockDim = ascendcPlatform.GetCoreNumAic();     // 默认取平台 AIC 核数
...
// 非 TP1 分支
blockDim = mmInfo.batch == BATCH_MLA ? BLOCK_DIM_MLA : blockDim;
```

- 默认 `blockDim` 取硬件 **AIC 核数**(`GetCoreNumAic()`)。
- **特殊优化**:当 `batch == 32` 时,固定使用 `BLOCK_DIM_MLA = 20` 个核 —— 针对该 batch 规模做过负载均衡调优。
- 最终 `context->SetBlockDim(blockDim)` 下发。由于是 MIX 架构,该 blockDim 表示 AIC 核数,对应 2×blockDim 个 AIV 核。

### 4.5 Tiling 参数布局

Tiling data 的整体布局(`tilingParam` 指针):

```
[0..5]   : 6 个 uint64 的 workspace 段大小(占 6*2 个 uint32)
[6*2..]  : tiling 头部(TILING_HEAD_SIZE=15 个 uint32)
           + 每 batch 参数(TILING_PARA_SIZE=8 个 uint32)× batch
```

**tiling 头部字段**(`GetTilingHead()`,下标见 `_dependency.cpp`):

| 下标 | 字段 | 含义 |
|------|------|------|
| 0 | TILING_BATCH | batch 数 |
| 1 | TILING_NUMHEADS | numHeads |
| 2 | TILING_HEADDIM | embeddingSize |
| 3 | TILING_NUMBLOKS | numBlocks |
| 4 | TILING_BLOCKSIZE | blockSize |
| 5 | TILING_MAXBLOCKS | maxNumBlocksPerQuery |
| 6 | TILING_TOR | 缩放系数 tor(float 位模式) |
| 7 | TILING_KVHEADS | kvHeads |
| 8 | TILING_HEADSIZE | =15(头部大小) |
| 9 | TILING_PARASIZE | 每 task 参数大小(8 或 TP1 的 4) |
| 12 | TILING_MASK_TYPE_ND | maskType |
| 13 | TILING_TASK_NUM | totalTaskNum |
| 14 | TILING_MAX_KV_SEQ_LEN | maxKVseqlen |

**每 batch 参数字段**(`GetNdMLATiling()` + `GetAddrOffsetMLA()`,偏移 `tilingOffset = 15 + 8*seqIdx`):

| 偏移 | 字段 | 含义 |
|------|------|------|
| +0 | qSeqLen | 该 batch 的 Q 序列长度 |
| +1 | kvSeqlen | 该 batch 的 KV 序列长度 |
| +2/+3 | addrQSeqOffset 高/低 32 位 | Q/O 的累积地址偏移(64 位拆分) |
| +4/+5 | addrOSeqOffset 高/低 32 位 | 输出地址偏移 |
| +6/+7 | addrMaskOffset 高/低 32 位 | mask 地址偏移 |

地址偏移**逐 batch 累加**:
```cpp
addrQSeqOffset    += numHeads * qSeqLen;
addrOSeqOffset    += numHeads * embeddingSize * qSeqLen;
addrMaskOffset    += qSeqLen * maxKVseqlen;
```
kernel 侧读取时把高低 32 位重新拼成 64 位地址,再乘以每 head 的 element 数得到实际 GM 偏移。

### 4.6 多核任务分配

- Host 侧仅确定 **核数(blockDim)** 与 **总任务数(totalTaskNum)** 以及每个 batch 的参数/地址偏移。
- **实际的 task→core 映射在 kernel 侧动态完成**:每个核用自身 `block_idx`(0..blockDim-1)以 `blockDim` 为步长循环领取 process(见 §6 的 `Run()`),即典型的 **round-robin 静态均分**。
- INT8 场景总 process = `q_block(每 batch 内 head 分块数) × batch`,由各 AIC/AIV 核以 `block_idx` 起步、步长 `blockDim` 遍历。

### 4.7 Workspace 切分

`MLATiling()` 计算 6 段 workspace(`workspaceParam[0..5]`),INT8(isQuant)与浮点分配不同:

| 段 | 变量 | INT8(isQuant) | 浮点 |
|----|------|---------------|------|
| 0 | s_gm | basicWorkSpaceFloat | float×2 |
| 1 | s_rope_out_gm | basicWorkSpaceFloat | 512 |
| 2 | p_gm | basicWorkSpaceInt8 | half×2 |
| 3 | o_tmp_gm | basicWorkSpaceInt8×2 | float×2 |
| 4 | go_gm | basicWorkSpaceFloat | float |
| 5 | tmp_gm | tailWorkSpaceFloat | float |

其中 `basicWorkSpace* = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLen`,即按核数 double-buffer 分配。INT8 的 p_gm/o_tmp_gm 用 int(int8/int32)存储,总 usrSize 再加系统 workspace。

---

## 5. Kernel 入口与 INT8 整体业务流程

### 5.1 Kernel 入口

入口函数 `multi_latent_attention()`(`op_kernel/multi_latent_attention.cpp`)按顺序完成:

1. 声明 MIX 任务类型 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`。
2. 从 workspace 头部解析 6 段中间 GM 地址(段大小由 host 的 `workspaceParam[0..5]` 给出):

```cpp
GM_ADDR s_gm         = usrWorkspace;                       // QK^T 结果(INT8 为 int32)
GM_ADDR s_rope_out_gm= s_gm + workspaceParam[0];           // QK rope 部分(float)
GM_ADDR p_gm         = s_rope_out_gm + workspaceParam[1];  // softmax 概率(INT8 为 int8)
GM_ADDR o_tmp_gm     = p_gm + workspaceParam[2];           // PV 结果(INT8 为 int32)
GM_ADDR go_gm        = o_tmp_gm + workspaceParam[3];        // online 累加输出(float)
GM_ADDR tmp_gm       = go_gm + workspaceParam[4];           // 临时 buffer
```

3. tiling 参数区从 `tiling + sizeof(uint64)*6` 开始(跳过 6 段 workspace 大小)。
4. 按 `TILING_KEY` 分发。INT8 走 **18(int8→fp16)/ 19(int8→bf16)**:
   - `__DAV_C220_CUBE__`(AIC):实例化 `MLAttentionDecoderAic<...>`,调 `SetArgs()` → `Run()`。
   - `__DAV_C220_VEC__`(AIV):实例化 `MLADecoderAiv<...>`,调 `SetArgs()` → `Run()`。

### 5.2 INT8 类型映射(AttentionType<INT8>)

INT8 场景下 mm1/mm2 的关键类型:
- **mm1/mm2 Output/CopyType = `int32_t`**:两次矩阵乘(int8×int8)累加结果均为 int32。
- **mmBias/mmScaleType = `float`**:反量化 scale 为 float。
- 输入 `IN_DTYPE = int8_t`,rope 部分 `IN_ROPE_DTYPE = half/bf16`(不量化)。

### 5.3 INT8 完整数据流

一次注意力计算(单个 KV block)的 INT8 数据流如下:

```
                         ┌─────────────── AIC (Cube) ───────────────┐
 Q(int8,512) ─┐          │ CUBE1: QK^T                               │
 K(int8,512) ─┼─ mmad ──►│  int8 × int8 → int32  ──► s_gm (int32)     │
 Qrope(fp) ───┤          │ rope: float MMA        ──► s_rope_gm (float)│
 Krope(fp) ───┘          └───────────────────────────────────────────┘
                                          │ FftsCrossCoreSync(QK_READY)
                                          ▼
                         ┌─────────────── AIV (Vector) ──────────────┐
                         │ SoftmaxStage1:                            │
                         │  DeQuantPerHead(s_gm × qk descale) → float │
                         │  + s_rope_gm(float) → muls(tor) → mask     │
                         │  → rowmax → flash max/dm → exp             │
                         │  → QuantPerToken(× 1/127 scale) → int8      │
                         │  → p_gm (int8)                            │
                         └───────────────────────────────────────────┘
                                          │ FftsCrossCoreSync(SOFTMAX_READY)
                                          ▼
                         ┌─────────────── AIC (Cube) ───────────────┐
 p(int8) ─┐              │ CUBE2: PV                                 │
 K^T(int8)┼─ mmad ──────►│  int8 × int8 → int32  ──► o_tmp_gm (int32) │
          └              │  (K 用 LoadDataWithTranspose 转置)         │
                         └───────────────────────────────────────────┘
                                          │ FftsCrossCoreSync(UPDATE_READY)
                                          ▼
                         ┌─────────────── AIV (Vector) ──────────────┐
                         │ SoftmaxStage2MLAHeadLoop:                 │
                         │  DeQuant(o_tmp_gm × pv descale) → float    │
                         │  online rescale:                          │
                         │   dm = exp(gm - hm)                       │
                         │   gl = dm*gl + ll ; go = go*dm + lo        │
                         │  最后一个 block: go/gl → 输出 o_gm          │
                         └───────────────────────────────────────────┘
```

**INT8 相比浮点的三个量化点**:
1. **DeQuant(QK)**:CUBE1 产出的 int32 乘 `qkDescale`(per-head float)还原为 float。
2. **Requant(P)**:softmax 概率 P 用 per-token scale(`quantMax=1/127`)量化回 int8,供 CUBE2 用 int8 做 PV。
3. **DeQuant(PV)**:CUBE2 产出的 int32 乘 `pvDescale`(per-head float)还原为 float。

**rope 部分始终走 float 独立 MMA**,不参与量化,在 SoftmaxStage1 中与 DeQuant 后的主体结果相加。

hidden_size:INT8 主体 =512(rope 的 64 单独处理),浮点场景为 576(512+64)。

---

## 6. AIC(Cube 侧)函数细节 —— MLAttentionDecoderAic

Cube 侧类 `MLAttentionDecoderAic` 负责两次矩阵乘。核心执行流程:`SetArgs()` 保存 GM 指针与参数 → `Run()` 以 `block_idx` 为起点、`blockDim` 为步长 round-robin 领取 process → 每个 process 调 `InnerRunCubeMLA()` 完成 QK^T(CUBE1)与 PV(CUBE2)。

### 6.1 SetArgs / Run

- **SetArgs**:保存 q_gm、q_rope_gm、ctkv_gm、ctkv_rope_gm、block_tables_gm、o_gm 以及 6 段 workspace GM 指针;从 `tiling_para_gm` 读头部公共参数(batch、numHeads、embeddingSize、blockSize、maxNumBlocksPerQuery、tor 等)。
- **Run**:以 `block_idx`(核号)为起点、`blockDim` 为步长遍历 process。每个 process 用其 batch 的 `offset_tiling = TILING_HEAD_SIZE + TILING_PARA_SIZE * seqIdx` 定位到该 batch 参数,调用 `InnerRunCubeMLA()`。

### 6.2 InnerRunCubeMLA —— QK^T 与 PV

单次处理一个 process(某 batch 的一段 head)。

#### (1) Q 地址计算

从 tiling 参数区读 Q 的 64 位地址偏移(高低 32 位拼接),再换算成 element 偏移:

```cpp
uint64_t addr_q_scalar = ((uint64_t)addr_q_high32 << 32) | addr_q_low32;
uint64_t q_offset      = addr_q_scalar * 512 + start_head * 512;  // INT8 主体 hidden=512
uint64_t q_rope_offset = addr_q_scalar * 64  + start_head * 64;   // rope hidden=64
```

INT8 主体 `hidden_size = 512`(浮点为 576),rope 部分独立按 64 计算偏移;`start_head` 为该 process 负责的起始 head。

#### (2) Q 搬入 L1

- `cur_q_seqlen == 1`(纯 decode):用 `gm_to_l1` 直接搬入 L1。
- 否则用 `Nd2NzParams` 做 ND→NZ 转换搬入;head 数超过阈值时逐 seqlen 分批搬。

#### (3) n_loop 循环 —— 遍历 KV block

对每个 KV block:
1. **block_table 定位**:通过 `block_tables_gm` 找到该逻辑 block 对应 KV Cache 的物理 block 号,算出 `kv_offset`。
2. **K / K_rope 搬入 L1**:INT8 KV 为 NZ 格式,走 NZ→NZ 的 `gm_to_l1` 搬运。

#### (4) CUBE1:QK^T(embed_split 分段)

hidden 128 方向切 5 段(前 4 段各 128,第 5 段为 rope 的 64),逐段:
- L1→L0A(Q)、L1→L0B(K);
- **INT8**:`mmad<..., int8_t, int8_t, int32_t, false>`(int8×int8→int32)累加到 `mm1_l0c`,`init` 标志在 `embed_split_idx == 0` 时置位。

```cpp
if constexpr (tilingKeyType == TILING_INT8_DATA) {
    mmad<..., IN_DTYPE, IN_DTYPE, mm1OutputType, false>(   // int8×int8→int32
        mm1_l0c, l0a, l0b, m, qk_round_n, embed_split_size, embed_split_idx == 0);
}
```

- **INT8 特殊分段**:`embed_split_idx == 3` 时把当前 int32 累加结果 `l0c_to_gm` 写到 `s_gm`;第 5 段(`idx == 4`)单独做 **rope 部分的 float MMA**,结果 `l0c_to_gm` 写到 `s_rope_gm`(float)。

```cpp
l0c_to_gm<..., mm1CopyType, mm1OutputType>(s_gm_tensor[...], mm1_l0c, ...);   // int32 主体
mmad<..., IN_ROPE_DTYPE, IN_ROPE_DTYPE, float, false>(...);                    // rope float
l0c_to_gm<..., float, float>(s_rope_gm_tensor[...], ...);
```

#### (5) CUBE2:PV(n_idx != 0 时)

第一个 KV block 之后开始做上一 block 的 PV(与当前 block 的 QK 流水重叠):
- **K 转置**:用 `LoadDataWithTranspose` 把 K 从 L1 转置进 L0B;
- **P 搬入**:softmax 输出的 p 从 `p_gm`(int8)搬进 L0A(NZ→ZZ);
- **mmad**:`int8 × int8 → int32`,结果 `l0c_to_gm` 写到 `o_tmp_gm`(int32)。

```cpp
mmad<..., IN_DTYPE, IN_DTYPE, mm2OutputType, false>(   // int8×int8→int32
    mm2_l0c, l0a_p, l0b_kT, m, embed_split_size, qk_n_2, 1);
l0c_to_gm<..., mm2CopyType, mm2OutputType>(o_tmp_gm_tensor[...], mm2_l0c, ...);
```

#### (6) 同步

Cube 与 Vector 通过 `FftsCrossCoreSync` 跨核同步:CUBE1 完成发 `QK_READY_DECODER`;等 Vector 的 `SOFTMAX_READY_DECODER` 后才做 CUBE2;PV 完成发 `UPDATE_READY_DECODER`。核内用 `SET_FLAG/WAIT_FLAG`(MTE2/MTE1/M/FIX)与 `PIPE_BARRIER` 保证 L1/L0A/L0B/L0C 的 ping-pong(16384 偏移)读写顺序。

---

## 7. AIV(Vector 侧)函数细节 —— MLADecoderAiv

Vector 侧类 `MLADecoderAiv` 承担反量化、Softmax、量化与 flash-attention 在线累加,是 INT8 精度处理的核心。两个 Vector 核(`sub_block_idx` = 0/1)各处理一半 head。

### 7.1 InnerRunVectorChange —— AIV 主控

```cpp
uint32_t sub_head_num = (sub_block_idx == 1) ? (cur_head_num - cur_head_num/2) : cur_head_num/2;
uint32_t sub_m        = sub_head_num * cur_q_seqlen;
o_offset = addr_o_scalar + start_head*embedding_size + sub_block_idx*cur_head_num/2*embedding_size;
```

- **head 切分**:`sub_block_idx` 0/1 各处理 `cur_head_num/2` 个 head;`sub_m = sub_head_num * cur_q_seqlen` 是本核处理的行数。
- **o_offset**:输出地址按 `start_head` 与 `sub_block_idx` 偏移,两核写不同 head 区间。
- **n_loop 循环**(按 `block_size = 64` 切 KV,循环 `n_loop + 1` 次做软件流水):

```cpp
for (n_idx = 0; n_idx < n_loop + 1; n_idx++) {
    if (n_idx != n_loop) {                       // Stage1:当前 block 的 softmax
        WaitFlagDev(QK_READY_DECODER);           // 等 Cube 的 QK^T 完成
        WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
        SoftmaxStage1(p_gm[...], s_gm[...], s_rope_gm[...], mask_gm[...], ...);  // ping-pong(n_idx%2)
        FftsCrossCoreSync<PIPE_MTE3,2>(SOFTMAX_READY_DECODER);  // 通知 Cube 做 PV
        SET_FLAG(MTE3, MTE2, EVENT_ID3);
    }
    if (n_idx != 0) {                            // Stage2:上一 block 的 online rescale
        WaitFlagDev(UPDATE_READY_DECODER);       // 等 Cube 的 PV 完成
        uint32_t head_loop = (sub_m + process_row_num - 1) / process_row_num;   // process_row_num=16
        for (uint32_t hl = 0; hl < head_loop; ++hl) {
            SoftmaxStage2MLAHeadLoop(o_tmp_gm[...], go_gm[...], o_gm[o_offset + ...], ...);
        }
    }
}
```

- **ping-pong**:`n_idx % 2` 交替使用不同的 ubuf/gm 偏移(`dm32_ubuf`/`ll_ubuf`/`pm32_ubuf` 两组),使相邻 block 的 Stage1/Stage2 可重叠。
- Stage1 与 Stage2 在同一次循环里错位一拍:`n_idx` 做当前 block 的 Stage1,同时做上一 block(`n_idx-1`)的 Stage2。

### 7.2 DeQuantPerHeadImpl —— QK 反量化

把 CUBE1 的 int32 结果按 per-head 的 `qkDescale` 还原为 float:

```cpp
// 1. descale 搬入 ub;int32 结果搬入 ub
// 2. Cast int32 → float
Cast(float_ub, int32_ub, RoundMode::CAST_NONE, ...);
// 3. 逐 head 乘 descale(broadcast 到该 head 的所有列)
TensorMulRepeatM(float_ub, float_ub, descale_ub, ...);
```

每个 head 有独立 descale,因此按 head 循环做 broadcast 乘法。

### 7.3 SoftmaxStage1 —— DeQuant + rope + flash softmax + Requant

单个 KV block 的 softmax,输出量化后的 int8 概率 P:

1. **DeQuant + rope 合并**:`DeQuantPerHeadImpl(s_gm × qkDescale)` 得主体 float,再加上 `s_rope_gm`(float,rope 部分)。
2. **缩放 + mask**:`Muls(x, tor)`(tor 为 1/√d 缩放系数),再叠加 `mask_gm`。
3. **行最大 + flash 更新**:`ReduceMax` 求当前 block 行最大 `hm`;与历史最大 `gm` 比较更新,`dm = exp(gm - hm)` 作为历史部分的 rescale 因子。
4. **exp**:`Exp(p, x - hm)` 得未归一化概率;`ll = rowsum(p)` 为当前 block 行和。
5. **Requant(P)**:调 `QuantPerTokenImpl` 把 float 概率按 per-token scale(`quantMax = 1/127`)量化成 int8,写入 `p_gm` 供 CUBE2 使用。

### 7.4 QuantPerTokenImpl —— P 的 per-token 量化

```cpp
// scale = 1/127(per-token);float → int8
Muls(x, x, scale);                                  // 乘 1/127
Cast(half_ub, x, RoundMode::CAST_NONE, ...);        // float → half
Cast(int8_ub, half_ub, RoundMode::CAST_RINT, ...);  // half → int8(四舍五入)
```

概率恒为正且 ≤1,用固定 `1/127` scale 映射到 int8 范围,再由 CUBE2 用 int8×int8 做 PV。

### 7.5 SoftmaxStage2MLAHeadLoop —— PV 反量化 + online rescale

对 CUBE2 的 int32 PV 结果做反量化并做 flash-attention 在线累加:

1. **DeQuant(PV)**:`o_tmp_gm`(int32)Cast→float 后乘 per-head `pvDescale`。
2. **online rescale**(flash-attention 累加):

```
dm = exp(gm - hm)               // 历史 rescale 因子(Stage1 已算)
gl = dm * gl + ll               // 更新分母(行和)
go = go * dm + lo               // 更新分子(加权 V 累加)
```

3. **收尾输出**:遍历到最后一个 block 后,`o = go / gl` 得到归一化注意力输出,`Cast` 成输出类型(fp16/bf16)写到 `o_gm[o_offset]`;`head_loop` 按 `process_row_num = 16` 行分块处理,避免 UB 溢出。ring 场景另写 `lseOut`。

---

## 8. 小结

- **架构**:MIX 1 Cube + 2 Vector,Cube 管两次 matmul,Vector 管量化/softmax/累加,靠 workspace GM + FftsCrossCoreSync 流水协作。
- **Host Tiling**:核数默认 `GetCoreNumAic()`,`batch==32` 固定 20 核;任务数 = ΣqSeqLen(decode≈batch);tiling 由 15 项头部 + 每 batch 8 项(含地址高低 32 位)构成;task→core 在 kernel 侧 round-robin。
- **INT8 全链路**:QK(int8×int8→int32)→ DeQuant(×qkDescale)+ rope(float)→ flash softmax → Requant(×1/127→int8)→ PV(int8×int8→int32)→ DeQuant(×pvDescale)→ online rescale → 输出。三个量化点 + rope 独立 float 路径是 INT8 与浮点实现的核心差异。