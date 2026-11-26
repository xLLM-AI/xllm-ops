# t5_split_qkv_append_cache

T5 单 token 解码阶段的 QKV 拆分与 KV cache 追加算子（Ascend C / ACLNN）。

## 功能

输入融合的 `qkv[B,1,3*H*D]`（末维依次为全部 Q、全部 K、全部 V）与两份历史缓存
`pastKey/pastValue[B,H,L,D]`，输出：

- `query[B,H,1,D]`：独立连续的 Q；
- `key/value[B,H,L+1,D]`：前 L 个 token 逐位复制历史缓存，末尾 token 取 qkv 的 K/V 段。

全部输出重新分配、互不重叠，不修改输入；L=0 时不读取历史缓存。FP16/BF16 均按原始
16 位表示搬运，保留 ±0、Inf、NaN 位模式。

## 接口

| 参数 | 方向 | Shape | dtype |
| --- | --- | --- | --- |
| qkv | 输入 | [B,1,3*H*D] | FP16 / BF16 |
| pastKey | 输入 | [B,H,L,D] | 同 qkv |
| pastValue | 输入 | [B,H,L,D] | 同 qkv |
| query | 输出 | [B,H,1,D] | 同 qkv |
| key | 输出 | [B,H,L+1,D] | 同 qkv |
| value | 输出 | [B,H,L+1,D] | 同 qkv |
| headNum | 属性 | int | 等于 H（缺省 1） |
| headDim | 属性 | int | 等于 D（缺省 1） |

约束：1<=B<=64、1<=H<=128、1<=D<=256、0<=L<=8192，三输出元素总数不超过 2^28；
输入连续、同设备、同 dtype，ND 布局。workspace 恒为 0。

## 目录

```text
t5_split_qkv_append_cache/
├── op_host/
│   ├── t5_split_qkv_append_cache_def.cpp     # OpDef 注册
│   ├── t5_split_qkv_append_cache_proto.cpp   # InferShape / InferDataType
│   ├── t5_split_qkv_append_cache_tiling.h    # tiling data 定义
│   └── t5_split_qkv_append_cache_tiling.cpp  # 参数校验、分核
└── op_kernel/
    ├── t5_split_qkv_append_cache.cpp         # kernel 入口
    └── t5_split_qkv_append_cache.h           # 16 bit 搬运内核
```

## 实现要点

- 每块最多搬运 32768 个 uint16 元素；Q 按 batch 分块，K/V 按 head 分块，
  `blockDim = min(48, jobs)`，各核按连续工作项区间划分。
- 双 UB 槽位乒乓：先发起下一块的 MTE2 读，再等待上一块的 MTE3 写，
  让搬运两个方向重叠。
- 历史段按 L2 预算拆分 cached / streaming 两类输出，避免新分配的输出把
  仍需重读的历史 cache line 挤掉。
- 非对齐 D 与 L=0 均由同一套地址公式处理，无特殊分支的额外 kernel。

## 验证

910C（Ascend910_9382）/ CANN 9.1.0 / 驱动 25.5.t6.b001 实测：

- 精度：35/35 公开用例逐位一致，aclnn 与直调结果一致；
- 性能：10/10 通过，单次 `run()` 中位耗时 21.7–61.4 μs（基线 64.5–204.5 μs，
  2.96–3.39x）；
- 内存：10/10，峰值增量等于输出字节数，workspace = 0；
- 工程：24 条额外种子/偏移输入、84 条非法输入、双 stream、连续追加、
  输出生命周期全部通过。

## 构建

```bash
bash build.sh --pkg --soc=ascend910_93 --vendor_name=custom --ops=t5_split_qkv_append_cache -j16
```

安装后通过生成的 ACLNN 两阶段接口调用：
`aclnnT5SplitQkvAppendCacheGetWorkspaceSize` / `aclnnT5SplitQkvAppendCache`。
