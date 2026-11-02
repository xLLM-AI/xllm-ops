<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# DSA KV Offloading 算子设计

[English](../en/dsa_kv_offloading_c0.md) | 简体中文

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |

## 功能说明

DSA KV offloading 系列算子把一批 query 的逻辑 top-k token 规划到 selection 槽，再把当前 KV
与历史 miss 搬到设备侧 selection payload，供后续 sparse attention 读取。
算子只负责规划与搬运，不实现 sparse attention 或 top-k indexer。

四个 AscendC kernel 共用 `common/include/kernel/dsa_kv_offloading_plan_serve.h` 中的唯一
规划/搬运实现，由编译期参数 `ExportPlan`、RoPE 维度和 `ImmutablePlan` 选择行为：

| Kernel | 职责 |
| --- | --- |
| `DsaKvOffloadingPlanNope` | NoPE 规划：逻辑 token 映射到 selection 槽 |
| `DsaKvOffloadingServeNope` | NoPE 搬运：历史读取、当前 append、miss 填充、发布 attention 索引 |
| `DsaKvOffloadingPlanRope` | RoPE 规划，并导出本步只读 plan |
| `DsaKvOffloadingServeRope` | RoPE 搬运，可复用首层 plan |

三个公开 ACLNN 入口：

| 入口 | 功能 | 内部设备阶段 |
| --- | --- | --- |
| `aclnnDsaKvOffloadingUpdateNope` | NoPE full DSA | `DsaKvOffloadingPlanNope` → `DsaKvOffloadingServeNope` |
| `aclnnDsaKvOffloadingUpdateRopeWithPlan` | RoPE full DSA | `DsaKvOffloadingPlanRope` → `DsaKvOffloadingServeRope` |
| `aclnnDsaKvOffloadingServeRopePlanned` | RoPE shared DSA，复用首层 plan | `DsaKvOffloadingServeRope` |

NoPE 入口始终输出 compact 索引并保留 inactive resident，不存在
plain/compact/stable 三套公开 ABI。`aclnnDsaKvOffloadingSelectionCapacity()` 返回物理
selection 容量 2176。

## 符号与形状

```text
Q = 本次 query 行数
C = resident storage 行容量（C ≥ Q）
P = Host 物理页数
R = request/cache 行数
B = block-table stride
W = 最大 current window 宽度
K = 物理 top-k 宽度，固定为 2176
```

| 名称 | 方向 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `hostLatent` | 输入/输出 | BF16 | `[P,128,1,512]` | Host 映射的设备视图；历史读、当前 append 写 |
| `blockTable` | 输入 | INT32 | `[R,B]` | request 逻辑页 → Host 物理页 |
| `descriptor` | 输入 | INT32 | `[Q,4]` | `[cache_row, current_window_length, visible_kv_length, state_row]` |
| `currentMap` | 输入 | INT32 | `[Q,W]` | 可见 current offset → 本批 `currentLatent` 行 |
| `topk` | 输入 | INT32 | `[Q,1,K]` | 本 query 的逻辑 token ID |
| `appendSlots` | 输入 | INT32 | `[Q]` | 当前行的 Host 物理 token 槽；负值禁写 |
| `currentLatent` | 输入 | BF16 | `[Q,1,512]` | 本次新算出的 HBM latent |
| `selection` | 输入/输出 | BF16 | `[C*17,128,1,512]` | 每 state row 的设备 selection payload |
| `indices` | 输出 | INT32 | `[Q,1,K]` | 最终局部 selection 槽；无效项为 `-1` |
| `lengths` | 输出 | INT32 | `[Q]` | 各 query 有效项数量 |
| `selectionTable` | 输出 | INT32 | `[Q,17]` | 各 query 引用的 state row selection 页表 |
| `residentTokens` | 输入/输出 | INT32 | `[C,1,K]` | 每 state row 降序 token ID |
| `residentSlots` | 输入/输出 | INT32 | `[C,1,K]` | resident ID 对应的 selection 槽 |
| `residentCount` | 输入/输出 | INT32 | `[C]` | 每 state row resident 有效长度 |
| `compactIndices` | 输出 | INT32 | `[Q,1,K]` | NoPE 专用；有效槽按原逻辑顺序紧凑到前缀，尾部 `-1` |
| `hostRope/currentRope/selectionRope` | 输入/输出 | BF16 | latent 形状末维改为 64 | 仅 RoPE 入口，与 latent 共用槽 |
| `planTokens/planSlots` | 输出 | INT32 | `[Q,1,K]` | RoPE Update 输出；shared Serve 只读 |

NoPE Update 使用 15 个 tensor；RoPE Update 使用 19 个 tensor；
`aclnnDsaKvOffloadingServeRopePlanned` 使用 12 个只读 tensor，不读写 owner 的 resident
state 与 attention metadata。

## aclnn 接口

三个入口均采用两段式调用：

```text
aclnnDsaKvOffloadingUpdateNopeGetWorkspaceSize(...)
aclnnDsaKvOffloadingUpdateNope(workspace, workspaceSize, executor, stream)

aclnnDsaKvOffloadingUpdateRopeWithPlanGetWorkspaceSize(...)
aclnnDsaKvOffloadingUpdateRopeWithPlan(workspace, workspaceSize, executor, stream)

aclnnDsaKvOffloadingServeRopePlannedGetWorkspaceSize(...)
aclnnDsaKvOffloadingServeRopePlanned(workspace, workspaceSize, executor, stream)
```

完整声明见 `xllm_ops/dsa_kv_offloading_plan_nope/op_host/op_api/aclnn_dsa_kv_offloading_*.h`。

## 实现说明

### 规划语义

- `-1`：无效 token。
- `planSlots ≥ 0`：历史命中，无需重搬。
- `planSlots ≤ -2`：miss，目标槽为 `-value-2`。

NoPE 是单消费者：Plan 将负 miss 写入 `indices`，Serve 搬完后原地发布非负槽。
`compactIndices` 在规划时直接解码为非负槽，不与 positional `indices` alias；
holes 后的有效项及重复逻辑 token 的顺序、出现次数均保留，不去重。

RoPE Update 将负 miss 写入独立 `planSlots`，attention 的 `indices` 单独解码。
所有 shared 层只读 plan，不清除 miss 标记，确保每层加载自己的 KV 内容。

共享的是 indexer top-k/逻辑槽规划，不是不同层的 KV 内容。plan 只在本步有效；
所有消费者完成前不得覆盖 plan、修改对应 descriptor/currentMap 或推进 owner 状态。

### current、dummy 与 Graph

- current window `[visible-window, visible)` 的每个 offset 必须映射到真实 HBM
  current 行，并且始终刷新 HBM→selection，不因 token ID 命中 resident 而跳过。
- Host append 与 miss 数量无关：每个非负 append 槽都写当前行。当前 miss 直接
  读 HBM，不等待其他 core 的 Host append。
- dummy 为 visible=window=0、append=-1，仍执行设备 metadata 更新，不清 live
  resident 或 payload；dummy→busy→dummy 只更新固定地址 tensor 的值。
- 一个 launch 内 active state_row 必须在 `[0,C)` 且互不重复；active append 槽
  必须有效且互不冲突。有效 top-k 可以重复；逻辑 token 小于 2^24；
  `residentCount` 在 `[0,K]`。
- 请求槽跨请求重用前必须清 `residentCount`；禁止并发调用写同一 selection/state。

## 约束说明

- 仅支持 Atlas A3（`ascend910_93`）。
- BF16 latent/RoPE，INT32 metadata。
- 连续 ND，head 数 `H=1`；page=128，latent=512，RoPE=64。
- 物理 top-k 宽度 `K=2176`。其他逻辑宽度由调用方在设备端 pad 到 K，sentinel
  为 `-1`；有效项可以位于 holes 后面，不能按前缀截断。
- 每个 query/state row 独立预留 17 页 selection。
- 所有参数由调用方预分配；shape、dtype、ND 连续性及跨参数地址重叠会被检查。
- Host tensor 必须使用映射后的设备地址，placement 为 device；调用方需持有原始
  分配与映射，直到所有 stream/Graph 访问结束。
- 同一调用的不同参数不可重叠。shared 调用可只读复用同一个 plan，但每层必须有
  独立的 Host/current/selection payload。
- ACL Tensor 使用 canonical ND 描述：view shape 与 storage shape 都传本次逻辑
  tensor 的 sizes，offset 为零；不得把整个 backing allocation 容量作为 storage
  shape 传入。

## 调用与构建

算子通过 ACLNN 接口 `aclnnDsaKvOffloadingUpdateNope`、`aclnnDsaKvOffloadingUpdateRopeWithPlan` 和
`aclnnDsaKvOffloadingServeRopePlanned` 调用。算子定义、Host tiling 和 device kernel 位于：

- [`dsa_kv_offloading_plan_nope_def.cpp`](../../xllm_ops/dsa_kv_offloading_plan_nope/op_host/dsa_kv_offloading_plan_nope_def.cpp)
- [`dsa_kv_offloading_plan_nope_tiling.cpp`](../../xllm_ops/dsa_kv_offloading_plan_nope/op_host/dsa_kv_offloading_plan_nope_tiling.cpp)
- [`dsa_kv_offloading_plan_nope.cpp`](../../xllm_ops/dsa_kv_offloading_plan_nope/op_kernel/dsa_kv_offloading_plan_nope.cpp)
- [`dsa_kv_offloading_plan_rope`](../../xllm_ops/dsa_kv_offloading_plan_rope)
- [`dsa_kv_offloading_serve_nope`](../../xllm_ops/dsa_kv_offloading_serve_nope)
- [`dsa_kv_offloading_serve_rope`](../../xllm_ops/dsa_kv_offloading_serve_rope)

通过仓库标准入口构建这四个 kernel：

```bash
bash build.sh --pkg \
  -n "dsa_kv_offloading_plan_nope;dsa_kv_offloading_plan_rope;dsa_kv_offloading_serve_nope;dsa_kv_offloading_serve_rope" \
  -c ascend910_93
```

回归测试位于
[`test_dsa_kv_offloading_default_build.py`](../../test/python_test/test_dsa_kv_offloading_default_build.py)、
[`test_dsa_kv_offloading.py`](../../test/python_test/test_dsa_kv_offloading.py) 和
[`dsa_kv_offloading_contract_test.cpp`](../../test/cpp_test/dsa_kv_offloading_contract_test.cpp)。
