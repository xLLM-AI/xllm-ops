# ECR 专家选择路由三算子设计

本文描述 `ecr_expert_selection_threshold`（S2）、`ecr_token_routing_statistics`（S3）、
`ecr_routing_normalize_compact`（S4）三个 aclnn 自定义算子的设计与实现。三个算子按顺序
组合，完成一次「专家选择路由」：S2 逐样本求 top-p 阈值，S3 按阈值统计每个 token 被
选中的次数与概率和，S4 把选中的 token 压紧成 `[E, C]` 的稠密输出。

迁移过程、用例矩阵与性能对比见 [迁移与验证报告](ecr_routing_migration_report.md)。

## 1. 数据流

```text
prob [E, S] fp32
      │
      ▼
┌─────────────────────────────────────┐
│ S2 ecr_expert_selection_threshold   │  per (expert, sample)
│   tau_val [E,B] fp32                │
│   tau_idx [E,B] int32（样本内下标） │
│   n_gt    [E,B] int32               │
└─────────────────────────────────────┘
      │ tau_val / tau_idx
      ▼
┌─────────────────────────────────────┐
│ S3 ecr_token_routing_statistics     │  per (token tile, all experts)
│   sums [S] fp32                     │
│   cnts [S] fp32                     │
└─────────────────────────────────────┘
      │ sums / cnts
      ▼
┌─────────────────────────────────────┐
│ S4 ecr_routing_normalize_compact    │  per (expert, sample)
│   ids     [E, C] int32              │
│   weights [E, C] fp32               │
└─────────────────────────────────────┘
```

三个算子在同一个 NPU stream 上顺序下发，中间结果不出设备。`B` 是样本数
（`seq_lengths` 的元素个数，`B=1` 表示单样本），`S` 是所有样本长度之和，
`C = Σ_b k_b` 是 S4 的输出容量。

## 2. 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR / Ascend 950DT | √ |
| Atlas A3 训练 / 推理系列产品 | × |
| Atlas A2 训练 / 推理系列产品 | × |

三个算子只注册了 `ascend950` 的 AICore 配置，kernel 均为 `KERNEL_TYPE_AIV_ONLY`，
不使用 Cube。

## 3. 公共约定

### 3.1 形状与属性契约

| 项 | 约束 |
| --- | --- |
| `prob` | `[E, S]`，fp32，ND，连续；`1 ≤ E ≤ 64`，`0 ≤ S ≤ INT32_MAX` |
| `seq_lengths` | `ListInt` 属性；`≤ 32` 项；每项 `0 ≤ len ≤ 2097152`；`Σ len == S` |
| `B` | `len(seq_lengths) > 1` 时为项数，否则为 1 |
| `top_p` | `float`，`0 < top_p ≤ 1`，有限值 |
| `eps` | 仅 S4；`float`，`> 0`，有限值 |
| `tau_val` / `tau_idx` | `[E, B]`；fp32 / int32；由 S2 产生 |
| `sums` / `cnts` | `[S]` fp32；由 S3 产生 |

`seq_lengths` 为 `[]`、`()`、`[S]` 三者等价，都表示「单样本、长度 S、忽略这一项的值」。
`B > 1` 时 `seq_lengths[i]` 是第 i 个样本在 `prob` 列方向上的长度，样本按给出顺序首尾相接。

### 3.2 每样本阈值个数 k 与输出容量 C

Host 侧用 double 计算，保证与 Python 参考实现逐位一致：

```text
k_b = 0                                          , len_b == 0
k_b = clamp(ceil(len_b * double(float(top_p)) - 1e-12), 1, len_b)    , len_b > 0

C = Σ_b k_b
```

`-1e-12` 用来吸收 `len * top_p` 在 double 下的向上误差，避免 `len=4096, top_p=0.125`
被算成 513。每个样本的 `k_b` 写进 tiling，S2 用它做第 k 大，S4 用它做容量。

### 3.3 选择谓词

S3 与 S4 必须给出同一套选中集合，否则 `cnts` 与 `ids` 的行数会对不上。统一写成：

```text
selected(p, token, tau_val, tau_idx) =
        p > tau_val
     or (p == tau_val and token_local <= tau_idx)
```

`token_local` 是**样本内**下标（`0 ≤ token_local < len_b`），不是全局 token 下标。
S2 选取 `tau_idx` 时保证：对每个 `(expert, sample)`，满足谓词的 token 恰好有 `k_b` 个，
且平局时优先取下标更小的 token。因此 S4 每个样本写出的有效槽位数严格等于 `k_b`，
正好填满 `C` 中的那一段；`ids` 里不会出现「跳号」。

### 3.4 空样本与退化输入

| 情形 | S2 输出 | S3 | S4 |
| --- | --- | --- | --- |
| `len_b == 0` | `(2.0, -1, 0)` | 该样本不产生 tile | 该样本容量 0，不占输出槽 |
| `len_b > 0` 但正数个数 `< k_b` | `(0.0, -1, n_positive)` | 谓词恒假，`sums/cnts` 为 0 | 该 `(e, b)` 的空槽写 `-1 / 0` |
| `S == 0` | 输出 `[E, B]` 的 `(2.0,-1,0)` | 输出空 `[0]`，不 launch | 绑定层直接返回 `[E,0]`，不 launch |
| 外部传入不一致的 `tau_*` | — | 按谓词如实统计 | 超出 `k_b` 的选中被截断，不足的槽保留 `-1 / 0` |

`2.0` 是「大于任何合法概率」的哨兵值，使空样本在 S3/S4 中自然不选中任何 token；
`0.0 / -1` 则用于「阈值跌破 0」的退化情形（概率全为 0 或非正）。

### 3.5 执行模型

- 三个 op 都是异步接口：aclnn 的 `GetWorkspaceSize` + 两次 launch 形态，由框架插入同步。
- Workspace 只申请 `GetLibApiWorkSpaceSize()`，kernel 本身不使用 workspace。
- `block_dim = min(平台 AIV 核数, 任务数)`，任务按 `block_idx, +core_num` 跨步分配。
- tiling 数据全部是固定长度数组（每样本 32 项上限），不依赖动态内存。

## 4. S2 `ecr_expert_selection_threshold`

### 4.1 接口

| 方向 | 名称 | 类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| 输入 | `prob` | float | `[E, S]` | 归一化后的专家概率 |
| 属性 | `seq_lengths` | ListInt | — | 样本切分 |
| 属性 | `top_p` | float | — | 阈值比例 |
| 输出 | `tau_val` | float | `[E, B]` | 第 k 大的值（阈值） |
| 输出 | `tau_idx` | int32 | `[E, B]` | 阈值的样本内 token 下标 |
| 输出 | `n_gt` | int32 | `[E, B]` | 严格大于阈值的元素个数 |

aclnn 签名：`aclnnEcrExpertSelectionThreshold(prob, seqLengths, topP, tauVal, tauIdx, nGt)`。

### 4.2 算法

单个任务 `(e, b)` 在 `prob[e, base_b : base_b + len_b]` 上执行四个阶段。

**阶段 0：正数计数。** 用 VF `CountCmpVf<GT>` 统计 `p > 0` 的个数。
若不足 `k_b`，直接输出 `(0.0, -1, count)`：此时不存在第 k 大的正数，S3/S4 会自然全不选中。

**阶段 1：位序二分。** 概率已归一化到 `[0,1]`，而非负 fp32 的 IEEE 位模式与数值序单调
一致，因此可以直接在 32 位整数域上二分：

```text
lo = 0x00000000 (0.0)     hi = 0x3F800000 (1.0)
invariant:  count(p > hi) < k  ≤  count(p > lo)
loop (≤ 31 次，直到 hi - lo ≤ 1):
    mid = lo + (hi - lo) / 2
    c = count(p > bits_to_float(mid))
    if c >= k:  lo = mid
    else:       hi = mid
tau_val = bits_to_float(hi);  n_gt = count(p > tau_val)
```

结束时 `hi` 是「使计数跌破 k 的最小值」，即第 k 大的值；`n_gt < k` 是该值左侧的严格大于
个数。用整数域二分而不是浮点二分，避免「mid 落在两个相邻 float 之间」导致的死循环，
也让结果不受 `top_p` 大小影响（`1e-15` 与 `1.0` 走同一条路径）。

**候选压缩。** 每轮二分都要数一遍整行，长序列代价高。当某轮出现
`n_cand = count(>lo) - count(>hi) ≤ 8192` 且 `n_cand * 2 ≤ len` 时，说明真正的答案只可能
落在这 `n_cand` 个元素里：用 VF `CandCompactVf` 把 `(lo, hi]` 区间的元素压紧到 UB 候选
缓冲，之后每轮只数候选（`c = base_gt + count(cand > mid)`），不再触碰原始行。
行短于 32768 时直接从已驻留的 UB 行里压缩；行更长时分块流式压缩，候选写入 UB。

**阶段 2：平局定位。** 需要确定「第 `need = k - n_gt` 个等于 `tau_val` 的元素」。
从行首按 4096 元素分段扫描，`EqIdxCompactVf` 用 `Squeeze` 把等值元素的下标按升序写入
索引缓冲，写入字节数从 AR 寄存器读回；累加各段计数，落在哪一段就取该段第
`need - acc - 1` 个下标。这保证了平局时返回**更小的 token 下标**。

**输出。** 三个标量写到 `[E*B]` 展开后的位置 `e * B + b`。float 与 int32 输出复用同一
32 字节 UB 缓冲，靠 `S_MTE3 → MTE3_S` 事件对交替，保证标量写与 DMA 读不重叠。

### 4.3 并行与任务划分

- 任务数 `E * B`，一核一个任务，跨步分配；`E=48, B=1` 时 48 个任务铺到 48 个核。
- 任务之间读写互不相交：每个任务只写自己那个 `(e,b)` 的 3 个标量。
- 任务内是串行的（二分本身有依赖），但向量宽度 64，`len=4096` 时一次 VF 调用处理 64 个
  元素、单轮 64 次循环。

### 4.4 UB 布局

| 缓冲 | 大小 | 用途 |
| --- | --- | --- |
| `prob_buf` | 32768 × 4 B = 128 KB | 驻留行 / 流式分块 |
| `cand_buf` | (8192 + 64) × 4 B ≈ 32.3 KB | 二分候选 |
| `eq_buf` | (4096 + 64) × 4 B ≈ 16.3 KB | 等值下标输出 |
| `red_buf` / `out_buf` | 各 32 B | 归约结果 / 标量输出 |

合计约 177 KB，单核一份。预留 64 元素是给 `StoreUnAlign` 的尾部整寄存器写留余量。

### 4.5 同步与流水

每个 VF 调用后按依赖关系插事件对，全部是「发出即等待」的局部依赖，不跨任务悬挂：

| 依赖 | 事件 |
| --- | --- |
| VF 结果被标量读回（AR 计数） | `V_S` |
| DMA 覆写 VF 已读的输入缓冲 | `V_MTE2` |
| DMA 覆写 VF 正在写的输出（下一任务清零） | `MTE3_V`（相邻任务间） |
| 标量写 UB → DMA 搬出 | `S_MTE3`、`MTE3_S` |

驻留路径（`len ≤ 32768`）整行只搬一次，后续二分与平局定位都复用 UB；非驻留路径每段
搬入后立即消费，段间用 `V_MTE2` 保护。

### 4.6 复杂度

设 `n = len_b`，`c = n_cand`：

- 时间：最坏 `O(31 · n / 64)` 次向量比较；触发压缩后降为 `O(31 · c / 64 + n / 64)`。
  典型随机数据下 `c` 约为 `n · (top_p 附近密度)`，远小于 `n`，所以长序列主要代价是一次
  整行扫描 + 一次性候选压缩 + 一次平局扫描。
- 空间：每任务 UB 固定，与 `n` 无关。

## 5. S3 `ecr_token_routing_statistics`

### 5.1 接口

| 方向 | 名称 | 类型 | Shape |
| --- | --- | --- | --- |
| 输入 | `prob` | float | `[E, S]` |
| 输入 | `tau_val` | float | `[E, B]` |
| 输入 | `tau_idx` | int32 | `[E, B]` |
| 属性 | `seq_lengths` | ListInt | — |
| 输出 | `sums` | float | `[S]` |
| 输出 | `cnts` | float | `[S]` |

aclnn 签名：`aclnnEcrTokenRoutingStatistics(prob, tauVal, tauIdx, seqLengths, sums, cnts)`。

### 5.2 算法

以 **token tile** 为单位，tile 宽 512：

```text
sums[x] = Σ_e  p[e,x] * selected(p[e,x], x_local, tau_val[e,b], tau_idx[e,b])
cnts[x] = Σ_e      1    * selected(...)
```

kernel 启动时把整张 `[E, B]` 的阈值表一次搬入 UB（最多 `64×32` 项，各 8 KB），
并构建 512 长的 `ramp = [0,1,...,511]`。之后每个 tile：

1. 定位 tile 属于哪个样本：`task_base[b] ≤ task < task_base[b] + ceil(len_b/512)`；
2. 把 `sum`、`cnt` 两个 512 长累加器清零；
3. 对 `e = 0 … E-1` 依次：
   - 搬入 `prob[e, tok_start : tok_start+count]`（≤512 个 float）；
   - VF `AccumVf` 计算谓词并累加：
     `gt = p > tau`，`eq = p == tau`，`le = ramp ≤ tau_idx - tile_local`，
     `sel = gt + (eq and le)`，`sum += p * sel`，`cnt += sel`；
4. 两个累加器一次搬回 `sums/cnts` 的 `[tok_start, tok_start+count)`。

平局判定用 `ramp + tile_local ≤ tau_idx` 的等价变形 `ramp ≤ tau_idx - tile_local`，
避免每 lane 加偏移；`idx_limit` 用 float 表示是安全的（样本内下标 ≤ 2097152 < 2^24，
float 可精确表示）。

累加顺序固定为 `e = 0 … E-1`，与参考实现一致，`cnts` 逐元素完全相等，`sums` 只有浮点
累加误差（实测完全相等，测试仍按 `1e-6` 容差判定以留余量）。

### 5.3 并行与任务划分

- 任务数 `Σ_b ceil(len_b / 512)`，跨步分配到各核。
- 长序列天然切得好：`S=131072` 有 256 个 tile，可铺满所有核；`S=17` 只有 1 个 tile，
  只用一个核，但数据量本身就小。
- `E` 方向不并行：同一 tile 的所有专家累加到同一对 sum/cnt 上，串行避免原子写。

### 5.4 UB 布局

| 缓冲 | 大小 | 用途 |
| --- | --- | --- |
| `tau_val` / `tau_idx` | 64×32×(4+4) B = 16 KB | 阈值表，整表驻留 |
| `ramp` | 512 × 4 B = 2 KB | 0..511 下标斜坡 |
| `prob` | 512 × 4 B = 2 KB | 当前专家的 tile |
| `sum` / `cnt` | 各 512 × 4 B | 累加器 |

### 5.5 同步

- 阈值表搬入后 `MTE2_S`（标量取阈值）+ `MTE2_V`（VF 读平局判定不需要表，但保持对称）。
- `ramp` 标量写入后 `S_V`。
- 每个专家：`MTE2_V`（DMA → VF），VF 后 `V_MTE2`（保护 `prob` 不被下一专家覆写）。
- 输出前 `V_MTE3`，输出后 `MTE3_V`（下一 tile 清零累加器前必须等 DMA 读完）。

## 6. S4 `ecr_routing_normalize_compact`

### 6.1 接口与容量

| 方向 | 名称 | 类型 | Shape |
| --- | --- | --- | --- |
| 输入 | `prob` | float | `[E, S]` |
| 输入 | `tau_val` | float | `[E, B]` |
| 输入 | `tau_idx` | int32 | `[E, B]` |
| 输入 | `sums` | float | `[S]` |
| 输入 | `cnts` | float | `[S]` |
| 属性 | `seq_lengths` | ListInt | — |
| 属性 | `top_p` | float | — |
| 属性 | `eps` | float | — |
| 输出 | `ids` | int32 | `[E, C]` |
| 输出 | `weights` | float | `[E, C]` |

aclnn 签名：
`aclnnEcrRoutingNormalizeCompact(prob, tauVal, tauIdx, sums, cnts, seqLengths, topP, eps, ids, weights)`。

`C = Σ k_b` 在 host tiling 与 Python 绑定里各算一次，必须完全一致（同一套 double 公式）。
样本 b 占用列区间 `[c_offset[b], c_offset[b] + k_b)`。因为 S2 的构造保证每样本恰好选中
`k_b` 个 token，正常输入下这段会被写满；`-1 / 0` 只出现在退化输入或外部传入不一致阈值的
情形。

输出语义：

```text
ids[e, c]     = base_b + token_local            （全局 token 下标）
weights[e, c] = p[e, token] / max(sums[token], eps) * sqrt(cnts[token])
```

`weights` 先除后开方再乘，三步各自一次 fp32 舍入，与上游实现一致；测试按
`|Δ| ≤ 1e-6 + 1e-5·|ref|` 判定，`ids` 逐元素完全相等。

### 6.2 算法

每个 `(expert, sample)` 任务用 SIMT VF（1024 线程）执行一次保序压紧：

1. **初始化**：线程按 `tid < k_b` 把本样本那一段 `k_b` 个槽位写成 `ids=-1, weights=0`，
   保证未用槽有确定值；`__syncthreads()`。
2. **分段计数**：把 `[0, len_b)` 均分成 1024 段，线程 `tid` 负责
   `[tid*seg, min((tid+1)*seg, len))`，数段内满足谓词的 token 数。段是**连续且递增**的，
   这是保序的前提。
3. **独占前缀和**：Hillis–Steele 在 UB 上做 1024 元素的扫描，读写双缓冲（`2×1024×4 B`）
   交替，每轮 `__syncthreads()`；空段线程同样参与所有 barrier（没有提前返回），
   避免同步错位。
4. **定位与写回**：`output_index = prefix[tid] - own_count`，再次遍历本段，命中时写
   `ids/weights`，并把 `output_index` 自增。跨段拼接后恰好是 token 升序，因此输出数组
   的顺序与「按 token 下标升序压紧」一致。
5. **防御截断**：`output_index < k_b` 才写，防止外部传入的阈值使选中数超过容量。

### 6.3 为什么用 SIMT 而不是纯 VF

S4 需要「按谓词生成递增下标并写到数据相关的位置」，本质是带前缀和的 scatter。用 VF
`squeeze` 逐块输出也能做，但需要额外的跨块偏移管理，而且 1024 线程在长序列（`S` 达
131072）下每线程只需处理 128 个 token，前缀和代价（10 轮屏障）可以接受。相比之下 S2
的平局定位只需要「取第 need 个」，不需要全局前缀，因此用 VF + 分段计数更轻。

### 6.4 并行与任务划分

- 任务数 `E * B`，跨步分配到各核；同一核内任务串行，复用同一份 8 KB 扫描缓冲。
- 任务间输出完全不相交（列区间由 `c_offset[b]` 划分，行由 `e` 划分）。
- SIMT 线程数固定 1024，`length < 1024` 时每个线程至多 1 个 token，前缀和仍有 10 轮
  屏障，这是短序列下 S4 固定开销的主要来源。

### 6.5 复杂度

- 时间：`O(len_b / 1024 + log2(1024))` 每个任务，即两遍扫描 + 10 轮屏障。
- 空间：UB 固定 8 KB。

## 7. 数值合同与一致性

| 项目 | 约定 |
| --- | --- |
| 计算精度 | 全程 fp32；kernel 只有一条代码路径，不按 dtype 分支 |
| 输入概率 | 必须已归一化到 `[0, 1]`；负值视为未选中，`>1` 超出阈值搜索域，不保证语义 |
| S2 位精确性 | 二分在整数位模式域进行，`tau_val` 与参考实现逐位相同 |
| S3 累加顺序 | 固定 `e = 0 … E-1`，`cnts` 完全相等，`sums` 只有累加舍入误差 |
| S4 保序 | 连续分段 + 独占前缀和，输出恒为 token 升序 |
| 判定 | S2 三输出、S3 `cnts`、S4 `ids` 逐元素相等；S3 `sums`、S4 `weights` 用 `1e-6 + 1e-5·abs(ref)` |

三个算子均已与上游 bisheng 直调实现做过逐用例 A/B（21 个基础用例 × fp32/bf16 变体），
结论见迁移报告 §5。

## 8. 约束与错误处理

Host 侧 `BuildContract` 是唯一的契约检查点，infershape、tiling、binding 共用同一份逻辑：

| 条件 | 结果 |
| --- | --- |
| `prob` 非二维 / `E ∉ [1,64]` / `E == 0` | `GRAPH_FAILED` |
| `S < 0` 或 `S > INT32_MAX` | `GRAPH_FAILED` |
| `len(seq_lengths) > 32` | `GRAPH_FAILED` |
| 任一 `len` 为负或 `> 2097152` | `GRAPH_FAILED` |
| `Σ len ≠ S` | `GRAPH_FAILED` |
| `top_p` 非有限 / `∉ (0,1]`；`eps ≤ 0` | `GRAPH_FAILED` |
| `tau_val` / `tau_idx` 不是 `[E,B]` | `GRAPH_FAILED`（S3/S4） |
| S4 其余输入形状与 `[E,S]`/`[S]` 不符 | `GRAPH_FAILED` |
| dtype / format 非预期 | `GRAPH_FAILED` |

绑定层另有更早的 `TORCH_CHECK`（rank、`E` 范围、`seq_lengths` 求和、`tau_*` 形状），
错误信息直接指出期望值，便于 Python 侧定位。

两个实现细节：

- aclnn 的 attr 转换不接受空 `int[]`，因此绑定层把 `seq_lengths ∈ {[], ()}` 改写为
  `[S]` 再下发；语义完全等价。
- S4 容量 `C == 0` 只可能来自 `S == 0`，此时没有槽可写，绑定层直接返回空输出、不
  launch（与上游 `S=0` 直接返回的行为一致）。

## 9. 性能特征

实测条件：Ascend950PR，CANN 9.2.1；fp32 变体，`warmup=20`、`iters=100`，
NPU 6 / NPU 7 各 6 轮共 12 轮，aclnn 取轮间中位、上游取轮间最优（加速比为下界）。
完整 21 用例逐条数据见迁移报告 §6。

| 用例 | S | E | S2 aclnn / 上游 (ms) | S3 aclnn / 上游 (ms) | S4 aclnn / 上游 (ms) |
| --- | --- | --- | --- | --- | --- |
| F1 | 4096 | 48 | 0.0544 / 0.0999 | 0.0502 / 0.1128 | 0.0534 / 0.1373 |
| L2 | 131072 | 48 | 0.2477 / 0.3812 | 0.0651 / 0.1412 | 0.3826 / 0.7510 |
| ZERO | 17 | 3 | 0.0447 / 0.1005 | 0.0423 / 0.1137 | 0.0445 / 0.1405 |

读法：

- **短序列由固定路径开销主导**：S=17 时三者 aclnn 都在 0.04–0.05 ms，主要是
  `GetWorkspaceSize` + launch；上游因为每次调用都有一次同步 H2D 拷 tiling，稳定在
  0.10–0.14 ms。
- **S2 随 S 线性上升**：`S=131072` 时 0.25 ms。候选压缩前每一轮二分都要扫整行，加上
  正数计数和平局定位，长序列上是多次整行扫描；这是三算子中唯一在长序列上明显变慢的。
- **S3 几乎不随 S 变化**：`S=131072` 也只有 0.065 ms，因为 tile 并行度随 S 增长，
  单核工作量恒定（512 token × 64 专家）。
- **S4 的长序列由带宽主导**：`S=131072` 时 0.38 ms，要读 `prob`、写 `ids/weights`，
  与上游差距收窄到约 2×。

## 10. 集成与调用

### 10.1 编译与安装

- 算子目录：`xllm_ops/expert_chosen_routing/{ecr_expert_selection_threshold,
  ecr_token_routing_statistics, ecr_routing_normalize_compact}`，三者互不共享文件。
- `xllm_ops/build_aclnn.sh` 的 `ascend950` 列表包含三个 op 名。
- `cmake/func.cmake` 的 host 侧 glob 包含
  `expert_chosen_routing/**/op_host/CMakeLists.txt`。
- 产物打入 `cann-ops-xllm-custom_linux-aarch64.run`，安装后作为 vendor
  `custom_xllm_math` 提供 aclnn 接口。

### 10.2 Python 调用

```python
import torch, torch_npu
import custom_ops

prob = torch.softmax(torch.randn(48, 4096, dtype=torch.float32).npu(), dim=1)
tau_val, tau_idx, n_gt = custom_ops.ecr_expert_selection_threshold_npu(
    prob, seq_lengths=(), top_p=0.125)
sums, cnts = custom_ops.ecr_token_routing_statistics_npu(
    prob, tau_val, tau_idx, seq_lengths=())
ids, weights = custom_ops.ecr_routing_normalize_compact_npu(
    prob, tau_val, tau_idx, sums, cnts, seq_lengths=(), top_p=0.125, eps=1e-9)
```

多样本（变长）时把 `seq_lengths` 换成样本长度列表，例如 `[1024, 1024, 1024, 1024]`。
`prob` 必须是 NPU 上的 fp32 连续二维张量，绑定层不做 dtype 转换和 `.contiguous()`。
注意 `custom_ops.py` 只存在于源码树的 `test/python_test` 下，wheel 里只有 `custom_ops_lib`；
外部工程直接调 `custom_ops_lib.ecr_*`，或把 `custom_ops.py` 一起拷走。
更完整的调用说明与错误信息见迁移报告 §7。

## 11. 已知限制与后续工作

1. **S2 输入域**：二分限定 `[0, 1]`，依赖调用方已完成 softmax/归一化；若概率可能 `>1`，
   需要扩展上界或先做 clamp。
2. **S2 长序列**：`len > 32768` 时无法整行驻留，每轮二分都要重新流式扫描；候选压缩已把
   后续轮次降到候选规模，但首轮扫描仍是线性。若后续 `S` 继续增长，可考虑一次搬入
   「部分和直方图」加速首轮定位。
3. **S3 专家串行**：同一 tile 的 `E` 个专家串行累加，`E=64` 时长序列 tile 的向量利用率
   取决于 tile 数；当前实测已能满足要求，未做专家维并行。
4. **S4 短序列固定开销**：1024 线程 + 10 轮屏障对小 `len` 偏重；若线上以短序列为主，
   可以按 `len` 选择线程数或退回 VF 路径。
5. **性能口径**：当前对比是端到端调用延迟，不是纯 kernel 时间；如需纯 kernel 数据，
   建议用 `msprof` 单独采集。
