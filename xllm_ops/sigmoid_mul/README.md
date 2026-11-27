## Overview

`SigmoidMul` fuses `torch.sigmoid(x1) * x2` into a single AIV kernel for bf16 tensors, so the
intermediate `sigmoid(x1)` never round-trips through HBM.

## Operator Description

- Inputs: `x1`, `x2` — bf16, 2-D. Two supported forms:
  - elementwise: both `[M, N]`, same shape;
  - broadcast: `x1` is `[M, 1]` against `x2` `[M, N]`, `out[r][c] = sigmoid(x1[r]) * x2[r][c]`.
  `N` is expected to be 1024 / 2048 / 4096, `M` is arbitrary. Anything else is rejected by tiling.
- Output: `out` — bf16, the shape of the wider input.
- Formula: `out = 1 / (1 + exp(-x1)) * x2`, evaluated in fp32 and rounded back to bf16.
- aclnn API: `aclnnSigmoidMulGetWorkspaceSize(x1, x2, out, &workspaceSize, &executor)` +
  `aclnnSigmoidMul(workspace, workspaceSize, executor, stream)`.

## Design

The two forms are separate kernels picked by tiling key, so neither pays for the other's shape
handling:

| key | branch | entry |
|---|---|---|
| 1000 | same-shape elementwise | `op_kernel/sigmoid_mul.h` |
| 1001 | `[M, 1]` broadcast | `op_kernel/sigmoid_mul_bcast.h` |

`op_host/CMakeLists.txt` declares both keys (`add_ops_tiling_keys`); without that the build emits a
single unkeyed binary and the runtime rejects the launch with `ACLNN_ERR_RUNTIME_ERROR`.

### Key 1000 — elementwise

- Flat elementwise kernel over `M * N` elements. Each core claims at least 4096 elements
  (`blockDim = min(aivNum, ceil(total / 4096))`, per-core length aligned to 256 elements), so
  `M <= 32` shapes land on a handful of AIVs instead of fanning out over all 48 — the kernel
  finishes in well under a microsecond at those sizes, so extra cores only buy per-core prologue
  and UB footprint. Large shapes still saturate every AIV.
- The tile is capped by the per-core length, so a core that owns a single tile reserves UB for
  exactly that tile (and stays on the aligned fast copy path) instead of allocating the full
  8K-element working set for a few hundred elements.
- UB budget: 20 B/element — `x1`/`x2`/`out` double-buffered bf16 (12 B) plus two fp32 scratch
  buffers (8 B). With 196 KB UB the tile settles at 8192 elements.
- Software pipeline: the MTE2 copy-in of tile `t+1` overlaps the vector work of tile `t`, and the
  MTE3 store of tile `t` overlaps the vector work of tile `t+1`.
- `x2` is lifted to fp32 first and its VECIN slot is released immediately (`FreeTensor` before the
  sigmoid sequence), so the MTE2 side gets the buffer back a full sigmoid sequence early. The
  multiply also stays in fp32, keeping the only rounding at the final cast to bf16.
- Tails that are not tile-aligned go through `DataCopyPad`, so arbitrary `N` (and `M`) work even
  though only 1024/2048/4096 are tuned.

### Key 1001 — `[M, 1]` broadcast

- One core owns whole rows (`rowsPerCore = ceil(M / blockDim)`, `blockDim` capped by both the
  per-core element floor and `M`), so no row is ever split between cores and `x1` never needs
  cross-core reduction.
- Each core first computes the sigmoid of its whole row range in one pass — up to 512 rows
  (`kRowChunk`) at a time, bf16 `x1` cast to fp32 and pushed through
  `Muls(-1) → Exp → Adds(1) → Reciprocal` once per row — into a UB fp32 scalar table. A
  `SetFlag/WaitFlag<HardEvent::V_S>` pair makes those values visible to scalar reads.
- Then the rows are flattened into a single tile sequence `(row, tile) -> index` and streamed with
  prefetch-one-ahead, so the copy-in of the next tile overlaps compute across a row boundary
  instead of draining the pipeline at every row end. Tiles that straddle nothing are full
  `min(tileElems, cols)` element copies; the row tail goes through `DataCopyPad`.
- Compute per tile: `s = sigTable[tile / tilesPerRow]` read as a scalar, `x2` cast to fp32,
  `Muls(f, f, s)`, cast back to bf16. Because the sigmoid is a scalar multiply, the broadcast
  costs one scalar read per tile, not an fp32 sigmoid per element.
- UB budget: 12 B/element — `x2`/`out` double-buffered bf16 (8 B) plus one fp32 `x2` scratch
  (4 B), and the only extra allocation is the 512-row fp32 sigmoid table (2 KB).

## Build

```bash
cd xllm_ops
bash build_aclnn.sh "$PWD" ascend910_93 <build_dir> sigmoid_mul
bash <build_dir>/cann-ops-xllm-custom_linux-aarch64.run --quiet --install-path=<opp_dir>
source <opp_dir>/vendors/custom_xllm_math/bin/set_env.bash
```

## Test

`test/cpp_test/sigmoid_mul_bench.cpp` checks accuracy against a double-precision golden and times
the fused op against `aclnnSigmoid` + `aclnnMul` (both re-plan per call, as a framework does):

```bash
g++ -O2 -std=c++17 sigmoid_mul_bench.cpp -o sigmoid_mul_bench \
    -I<opp_dir>/vendors/custom_xllm_math/op_api/include -I$ASCEND_HOME_PATH/include \
    -L$ASCEND_HOME_PATH/lib64 -L<opp_dir>/vendors/custom_xllm_math/op_api/lib \
    -lcust_opapi -lopapi_math -lopapi_nn -lnnopbase -lascendcl -lpthread
./sigmoid_mul_bench --device 0 --iters 50 --warmup 10 --repeat 5
```

## Measured performance (Ascend 910_93, CANN 9.2.0, bf16)

Best of 5 batches, 50 iterations each; `fused_GB/s` counts 6 B/element, `unfused_GB/s` 10 B/element.

| shape | fused (us) | unfused (us) | speedup | fused GB/s | unfused GB/s |
|---|---|---|---|---|---|
| 1x1024 | 5.86 | 29.85 | 5.09x | 1.0 | 0.3 |
| 1x2048 | 5.64 | 29.90 | 5.31x | 2.2 | 0.7 |
| 1x4096 | 6.07 | 31.75 | 5.23x | 4.0 | 1.3 |
| 2x2048 | 5.50 | 30.60 | 5.56x | 4.5 | 1.3 |
| 4x2048 | 5.51 | 29.49 | 5.35x | 8.9 | 2.8 |
| 8x2048 | 6.10 | 28.94 | 4.75x | 16.1 | 5.7 |
| 16x1024 | 5.56 | 29.26 | 5.26x | 17.7 | 5.6 |
| 16x2048 | 5.63 | 30.11 | 5.35x | 34.9 | 10.9 |
| 16x4096 | 6.00 | 29.01 | 4.84x | 65.6 | 22.6 |
| 32x1024 | 5.89 | 29.38 | 4.98x | 33.4 | 11.2 |
| 32x2048 | 5.78 | 30.85 | 5.34x | 68.1 | 21.2 |
| 32x4096 | 5.73 | 30.54 | 5.33x | 137.2 | 42.9 |
| 64x1024 | 5.62 | 31.05 | 5.53x | 70.0 | 21.1 |
| 64x2048 | 5.90 | 30.25 | 5.13x | 133.4 | 43.3 |
| 64x4096 | 5.99 | 28.71 | 4.80x | 262.7 | 91.3 |
| 256x1024 | 6.07 | 28.16 | 4.64x | 259.0 | 93.1 |
| 256x2048 | 6.42 | 28.00 | 4.36x | 490.3 | 187.2 |
| 256x4096 | 7.08 | 28.51 | 4.03x | 888.3 | 367.7 |
| 1024x1024 | 7.51 | 29.14 | 3.88x | 837.8 | 359.8 |
| 1024x2048 | 9.73 | 28.74 | 2.95x | 1293.5 | 729.6 |
| 1024x4096 | 14.08 | 28.97 | 2.06x | 1786.8 | 1447.6 |
| 4096x1024 | 14.06 | 29.00 | 2.06x | 1789.3 | 1446.3 |
| 4096x2048 | 22.37 | 36.24 | 1.62x | 2250.1 | 2314.4 |
| 4096x4096 | 39.97 | 66.80 | 1.67x | 2518.5 | 2511.4 |

Broadcast (`x1` is `[M, 1]`), same measurement setup:

| shape | fused (us) | fused GB/s |
|---|---|---|
| 1x2048(b) | 5.28 | 2.3 |
| 16x1024(b) | 5.48 | 35.9 |
| 16x2048(b) | 5.48 | 71.7 |
| 16x4096(b) | 5.36 | 146.7 |
| 32x2048(b) | 5.17 | 152.4 |
| 64x2048(b) | 5.06 | 311.4 |
| 4096x2048(b) | 23.15 | 2174.4 |

The broadcast path carries no unfused column: the competing pair would be a sigmoid over `[M, 1]`
(32 KB at most here) followed by an `[M, N]` multiply, so its cost is a multiply plus one extra
dispatch. At `4096x2048` the broadcast kernel is 23.15 us against 22.37 us for the same-size flat
case — the row-wise pipeline lands within 4 % of the flat stream while doing 1/2048 of the sigmoid
work.

Accuracy: `max_rel = 6.55e-3`, `max_abs = 8.98e-3`, 0 elements outside bf16 tolerance
(`rtol = 1/64`, `atol = 1/256`) over the shapes above and the unaligned ones
(`1x1`, `1x7`, `1x255`, `1x256`, `3x1000`, `5x257`, `17x2049`, `127x4097`, `255x1023`,
`2049x1024`, `8193x2048`) plus the broadcast unaligned ones (`5x257(b)`, `16x257(b)`,
`16x1000(b)`, `17x2049(b)`, `255x1023(b)`).

Large shapes hit ~2.5 TB/s of effective HBM traffic, i.e. the same ceiling as the unfused pair, and
the fused/unfused ratio converges to the 5 B/3 B traffic ratio (1.67x).

`M <= 32` sits entirely on the fixed ~5.2 us aclnn call + launch floor: 16x2048 moves 96 KB in
total (~0.04 us of HBM time), so its kernel time is lost in the dispatch cost and every row from
1x1024 to 32x4096 measures the same. Sweeping the per-core floor over 1024 / 2048 / 4096 / 8192
(32 / 16 / 8 / 4 AIVs for 16x2048) produced no difference outside run-to-run noise, so that range
is tuned for core occupancy — a handful of AIVs per trivial shape, the full AIV count for real
work — rather than for latency, and the kernel's UB is sized to the per-core length so a 1-tile
core never reserves the 8K-element working set (164 KB) for a few hundred elements.

Broadcast shapes up to 64x2048 sit on the same floor: `sigmoid` over `[M, 1]` costs nothing, so
their fused time is indistinguishable from the elementwise rows of the same output size.
