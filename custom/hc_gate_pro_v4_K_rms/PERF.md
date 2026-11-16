# hc_gate_pro (v4-K + RmsNorm fused) performance measurement notes

> Environment: A5-179.64.14.2, docker `zhaoxingcheng_test`, Ascend950PR, CANN 9.2.0
> Inside the container you must first run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (the default 9.1.0 fails to compile)
> Runtime environment variables: `TILE_FWK_DEVICE_ID=<idle device id> PTO_TILE_LIB_CODE_PATH=third_party/pto-isa`

## 1. Baseline (total measured device time from the production op_summary CSV)

| Chain | t=1 (decode) | t=13 (prefill) |
|----|--------------|----------------|
| 5-op chain | 19.65us | 22.24us |
| **6-kernel chain** (5-op + RmsNorm; RmsNorm t=1 p50 2.548us / t=13 p50 4.150us) | **22.20us** | **26.39us** |

## 2. Final measurement (2026-09-21, quiet window on device 0,
device time from `torch_npu.profiler` kernel_details.csv, 100 samples, no outliers)

| Metric | t=1 (decode) | t=13 (prefill) |
|------|--------------|----------------|
| min / p10 | 7.33 / 7.63us | 9.82 / 10.06us |
| **p50 / mean** | **7.98 / 8.05us** | **10.44 / 10.50us** |
| max | 9.14us | 11.71us |
| vs the 6-kernel baseline | **x2.78** | **x2.53** |

Precision: 11/11 (t in {1,2,4,5,8,9,13,16,17,64,100}) plus 4/4 for
no-initial-value dependence (NaN prefill); all four outputs
y_norm/rstd/post/comb pass their per-output tolerances against the golden
chain (5-op golden + torch RmsNorm).

## 3. Version evolution data (profiler p50 metric)

| Version | t=1 | t=13 | Key change |
|------|-----|------|----------|
| v1 pure Vector | 8.04us | 14.32us | 32 AIV dot products, 2 barriers |
| v2 Cube+Vector | 11.19us | 14.67us | 16+32 cores, 3 barriers, cast goes through GM |
| v3 Mhc pipeline | 13.52us | 15.54us | hc_fn read directly as fp32, serial y d-loop |
| v4-K (DC=128, single buffer) | 11.28us | 13.15us | bf16 hc_fn, 28-core K-split, 1 barrier, FMA |
| v4-N | 13.76us | 16.93us | N-split over 8 rows (3-core data-feeding bottleneck, dropped) |
| v4-K (DC=512, double buffer) | 7.00us | 8.94us | d-loop 32 -> 8 iterations + MTE/V overlap |
| **v4-K + RmsNorm fused** | **7.98us** | **10.44us** | Two-pass y stream fusion (+0.98/+1.50us, saves the standalone RmsNorm's 2.5/4.2us) |

Reference point: MhcPreSinkhorn (AscendC reference implementation) 8.73/10.35us
-- the fused version wins on both shapes.

## 4. cannsim simulation traces (record + report)

Collection commands (inside the container, from the repo root):

```bash
cannsim record 'TILE_FWK_DEVICE_ID=0 PTO_TILE_LIB_CODE_PATH=third_party/pto-isa \
  python3 <driver script>' -s Ascend950 -n 0 -o /tmp/cannsim_out/
cannsim report -e /tmp/cannsim_out/npusim_<timestamp>_TILE_FWK_DEVICE_ID_0 -n 0
```

- Driver script: direct kernel invocation, 3 runs each for t=1 / t=13
  (pre-allocated workspaces, no profiler/golden, so no extra kernels leak
  into the recording).
- The simulator segfaults during teardown (`libpem_davinci.so`, a simulator
  bug) -- all kernels do finish and the recording is complete, so the report
  is generated afterwards with the second command (requires
  `pip install plotly`).
- Artifacts: `report/index.html` (trace summary charts),
  `results/kernel_{0..5}_reports/` (aicore_utilization.json /
  trace_core0.json (openable in perfetto) / IPC charts),
  `record/instr.bin` (instruction-level recording).

Traces of the fused version (MIX block 0 = 1 AIC + 2 AIV, timestamps at the
real core clock):

| shape | AIC (Cube) | AIV0 | AIV1 |
|-------|-----------|------|------|
| t=1 warm | 44.7% (2.7us) | **99.7% (6.0-6.3us)** | 44.2% (2.7us) |
| t=13 warm | 31.7% (2.3us) | **99.7% (7.1-7.2us)** | 96.5% (6.9-7.0us) |

Conclusion: the critical path is AIV0 (two y passes + gates + sinkhorn) while
the AIC is 55-68% idle; at t=1 AIV1 is only 44% busy -- a follow-up could move
the d-chunks of y pass2 onto AIV1.
The complete simulation data (tar.gz + trace charts) is in the archive
directory `cannsim_rms/`.

## 5. Measurement methodology notes

1. **Device-time metric**: `kernel_details.csv` from `torch_npu.profiler`
   (same metric as the production op_summary); event timing under the
   PyPTO-Pro JIT is limited by the host launch rate (an empty kernel takes
   ~39us/call) and does not reflect device time.
2. **Shared-device interference**: devices 2/3 often host other processes
   (100% util) and samples get polluted by preemption (outliers of
   100-500us) -- take p50 on a quiet device, otherwise take min/p10 of the
   clean cluster.
3. **Archived clean CSVs**: `prof/v4k_rms_fused_clean/{t1,t13}_kernel_details.csv`
   (device 0, 100 samples).
4. **Precision re-runs**: the full test script (11 precision cases + NaN
   prefill for no-initial-value dependence) is archived as
   `test_hc_gate_pro.py`; the reference chain is the (FROZEN) 5-op golden in
   `custom/hc_gate/hc_gate_golden.py` plus torch RmsNorm
   (`rsqrt(mean(y_bf16^2)+1e-6)`, `y*rstd*gamma.float()`).