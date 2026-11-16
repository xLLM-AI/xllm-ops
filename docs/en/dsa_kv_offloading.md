<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# DSA KV Offloading Operator Design

[简体中文](../zh/dsa_kv_offloading.md) | English

## Product Support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 training/Atlas A3 inference series | √ |
| Atlas A2 training/Atlas A2 inference series | × |

## Overview

The DSA KV offloading operators plan a batch of query logical top-k tokens into selection slots,
then move the current KV and historical misses into a device-side selection
payload for the following sparse attention. The operators only plan and move
data; they do not implement sparse attention or the top-k indexer.

Four AscendC kernels share the single planning/movement implementation in
`xllm_ops/dsa_kv_offloading_common/op_kernel/dsa_kv_offloading_plan_serve.h`.
Each kernel explicitly includes this directory and declares the shared operator
dependency. The header is also installed beside each kernel so installed packages
can resolve it without the original source tree. The compile-time parameters
`ExportPlan`, the RoPE dimension, and `ImmutablePlan` select the behavior:

| Kernel | Responsibility |
| --- | --- |
| `DsaKvOffloadingPlanNope` | NoPE planning: map logical tokens to selection slots |
| `DsaKvOffloadingServeNope` | NoPE movement: read history, append current KV, fill misses, publish attention indices |
| `DsaKvOffloadingPlanRope` | RoPE planning and per-step read-only plan export |
| `DsaKvOffloadingServeRope` | RoPE movement, optionally reusing the first-layer plan |

Three public ACLNN entries:

| Entry | Function | Internal device stages |
| --- | --- | --- |
| `aclnnDsaKvOffloadingUpdateNope` | NoPE full DSA | `DsaKvOffloadingPlanNope` → `DsaKvOffloadingServeNope` |
| `aclnnDsaKvOffloadingUpdateRopeWithPlan` | RoPE full DSA | `DsaKvOffloadingPlanRope` → `DsaKvOffloadingServeRope` |
| `aclnnDsaKvOffloadingServeRopePlanned` | RoPE shared DSA, reuses the first-layer plan | `DsaKvOffloadingServeRope` |

The NoPE entry always emits compact indices and retains inactive residents;
there are no separate plain/compact/stable public ABIs.
`aclnnDsaKvOffloadingSelectionCapacity()` returns the physical selection capacity, 2176.

## Symbols and Shapes

```text
Q = number of query rows
C = resident storage row capacity (C >= Q)
P = number of host physical pages
R = number of request/cache rows
B = block-table stride
W = maximum current window width
K = physical top-k width, fixed at 2176
```

| Name | Direction | Dtype | Shape | Description |
| --- | --- | --- | --- | --- |
| `hostLatent` | in/out | BF16 | `[P,128,1,512]` | Device view of mapped host memory; historical read, current append write |
| `blockTable` | in | INT32 | `[R,B]` | Request logical page to host physical page |
| `descriptor` | in | INT32 | `[Q,4]` | `[cache_row, current_window_length, visible_kv_length, state_row]` |
| `currentMap` | in | INT32 | `[Q,W]` | Visible current offset to this batch's `currentLatent` row |
| `topk` | in | INT32 | `[Q,1,K]` | Logical token IDs of this query |
| `appendSlots` | in | INT32 | `[Q]` | Host physical token slot of the current row; negative disables the write |
| `currentLatent` | in | BF16 | `[Q,1,512]` | Latent produced in this step |
| `selection` | in/out | BF16 | `[C*17,128,1,512]` | Per-state-row device selection payload |
| `indices` | out | INT32 | `[Q,1,K]` | Final local selection slots; invalid entries are `-1` |
| `lengths` | out | INT32 | `[Q]` | Valid entry count per query |
| `selectionTable` | out | INT32 | `[Q,17]` | Selection page table referenced by each query |
| `residentTokens` | in/out | INT32 | `[C,1,K]` | Descending token IDs per state row |
| `residentSlots` | in/out | INT32 | `[C,1,K]` | Selection slots of the resident IDs |
| `residentCount` | in/out | INT32 | `[C]` | Valid resident length per state row |
| `compactIndices` | out | INT32 | `[Q,1,K]` | NoPE only; valid slots packed to the prefix in logical order, `-1` suffix |
| `hostRope/currentRope/selectionRope` | in/out | BF16 | Same as latent, last dim 64 | RoPE entries only; share slots with latent |
| `planTokens/planSlots` | out | INT32 | `[Q,1,K]` | RoPE Update output; shared Serve reads only |

NoPE Update uses 15 tensors; RoPE Update uses 19 tensors;
`aclnnDsaKvOffloadingServeRopePlanned` uses 12 read-only tensors and does not touch the
owner's resident state or attention metadata.

## aclnn Interfaces

All three entries use the two-stage call:

```text
aclnnDsaKvOffloadingUpdateNopeGetWorkspaceSize(...)
aclnnDsaKvOffloadingUpdateNope(workspace, workspaceSize, executor, stream)

aclnnDsaKvOffloadingUpdateRopeWithPlanGetWorkspaceSize(...)
aclnnDsaKvOffloadingUpdateRopeWithPlan(workspace, workspaceSize, executor, stream)

aclnnDsaKvOffloadingServeRopePlannedGetWorkspaceSize(...)
aclnnDsaKvOffloadingServeRopePlanned(workspace, workspaceSize, executor, stream)
```

Full declarations are in `xllm_ops/dsa_kv_offloading_plan_nope/op_host/op_api/aclnn_dsa_kv_offloading_*.h`.

## Implementation

### Planning Semantics

- `-1`: invalid token.
- `planSlots >= 0`: historical hit, no movement needed.
- `planSlots <= -2`: miss; the target slot is `-value-2`.

NoPE has a single consumer: Plan writes negative misses into `indices`, and
Serve publishes non-negative slots in place after the copy. `compactIndices` is
decoded to non-negative slots during planning and does not alias the positional
`indices`; valid entries after holes and duplicate logical tokens keep their
order and occurrence count.

RoPE Update writes negative misses into the separate `planSlots`, and attention
`indices` are decoded separately. Every shared layer reads the plan only and
never clears miss markers, so each layer loads its own KV content.

What is shared is the indexer top-k/logical-slot plan, not KV content across
layers. A plan is valid only for the current step; it must not be overwritten,
nor the owner state advanced, before all consumers finish.

### Current, Dummy, and Graph

- Every offset in the current window `[visible-window, visible)` must map to a
  real HBM current row and is always refreshed HBM to selection, even when the
  token ID hits the resident.
- Host append is independent of the miss count: every non-negative append slot
  writes the current row. Current misses read HBM directly.
- A dummy has visible=window=0 and append=-1; it still updates device metadata
  and must not clear live residents or payloads. dummy to busy to dummy only
  changes values at fixed addresses.
- Within one launch, active state rows must be in `[0,C)` and unique; active
  append slots must be valid and non-conflicting. Valid top-k entries may
  repeat, logical tokens are below 2^24, and `residentCount` is in `[0,K]`.
- `residentCount` must be cleared before a request slot is reused. Concurrent
  calls must not write the same selection/state.

## Constraints

- Atlas A3 (`ascend910_93`) only.
- BF16 latent/RoPE, INT32 metadata.
- Contiguous ND, `H=1`; page=128, latent=512, RoPE=64.
- Physical top-k width `K=2176`. Other logical widths are padded to K on device
  by the caller with a `-1` sentinel; valid entries may appear after holes and
  must not be truncated as a prefix.
- Each query/state row reserves 17 selection pages independently.
- All parameters are preallocated by the caller; shape, dtype, ND contiguity,
  and cross-parameter address overlap are checked.
- Host tensors must use the mapped device address and device placement; the
  caller owns the allocation and mapping until all stream/Graph accesses end.
- Parameters of one call must not overlap. Shared calls may reuse one plan
  read-only, but each layer needs its own host/current/selection payload.
- ACL tensors use canonical ND descriptions: both view and storage shape carry
  this call's logical sizes with zero offset; the whole backing allocation size
  must not be passed as storage shape.

## Build and Test

The operators are invoked through `aclnnDsaKvOffloadingUpdateNope`,
`aclnnDsaKvOffloadingUpdateRopeWithPlan`, and `aclnnDsaKvOffloadingServeRopePlanned`. Operator
definitions, host tiling, and device kernels are in:

- [`dsa_kv_offloading_plan_nope_def.cpp`](../../xllm_ops/dsa_kv_offloading_plan_nope/op_host/dsa_kv_offloading_plan_nope_def.cpp)
- [`dsa_kv_offloading_plan_nope_tiling.cpp`](../../xllm_ops/dsa_kv_offloading_plan_nope/op_host/dsa_kv_offloading_plan_nope_tiling.cpp)
- [`dsa_kv_offloading_plan_nope.cpp`](../../xllm_ops/dsa_kv_offloading_plan_nope/op_kernel/dsa_kv_offloading_plan_nope.cpp)
- [`dsa_kv_offloading_plan_rope`](../../xllm_ops/dsa_kv_offloading_plan_rope)
- [`dsa_kv_offloading_serve_nope`](../../xllm_ops/dsa_kv_offloading_serve_nope)
- [`dsa_kv_offloading_serve_rope`](../../xllm_ops/dsa_kv_offloading_serve_rope)

Build the four kernels through the standard entry point:

```bash
bash build.sh \
  -n "dsa_kv_offloading_plan_nope;dsa_kv_offloading_plan_rope;dsa_kv_offloading_serve_nope;dsa_kv_offloading_serve_rope" \
  -c ascend910_93
```

Regression tests are in
[`test_dsa_kv_offloading_default_build.py`](../../test/python_test/test_dsa_kv_offloading_default_build.py),
[`test_dsa_kv_offloading_kernel_build.py`](../../test/python_test/test_dsa_kv_offloading_kernel_build.py),
[`test_dsa_kv_offloading.py`](../../test/python_test/test_dsa_kv_offloading.py),
and
[`dsa_kv_offloading_contract_test.cpp`](../../test/cpp_test/dsa_kv_offloading_contract_test.cpp),
[`dsa_kv_offloading_stable_contract_test.cpp`](../../test/cpp_test/dsa_kv_offloading_stable_contract_test.cpp),
and
[`dsa_kv_offloading_stable_rope_contract_test.cpp`](../../test/cpp_test/dsa_kv_offloading_stable_rope_contract_test.cpp).
