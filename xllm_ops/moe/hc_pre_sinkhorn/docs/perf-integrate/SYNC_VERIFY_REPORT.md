# hc_pre_sinkhorn 同步到 my-xllm-ops — 精度与性能验证

日期：2026-08-21  
来源：`/home/h00801112/codex/xllm-ops/xllm_ops/moe/hc_pre_sinkhorn`  
目标：`/home/h00801112/codex/my-xllm-ops/xllm-ops/xllm_ops/moe/hc_pre_sinkhorn`

## 1. 同步内容

| 文件 | 变更 | md5 |
|------|------|-----|
| `op_kernel/hc_pre_sinkhorn_regbase_base.h` | SoA CombFrag、MultiCopy 转置、ldva TransDataTo5HD + `S_V`/`V_S` | `f152f40d187453990e9c9c882aa38358`（两边一致） |
| `op_kernel/hc_pre_sinkhorn_regbase_perf.h` | `useCombSoa` 分发；`hcScale` 仍 `GetValue()` | `8754c1d245ccdc4f21a389d501e4ce6f`（两边一致） |
| `op_host/CMakeLists.txt` | opc 增加 `-Wno-constant-conversion` | 已同步 |

未改：`hc_pre_sinkhorn.cpp`、tiling、def、proto、membase kernel。

## 2. 编译说明

`my-xllm-ops/xllm-ops` 全新 `build.sh` 会重新下载 `json` / `abseil`，gitcode 返回 403，cmake 失败。  
本次用 dest 源码经 opc 编出 `HcPreSinkhorn_20e241df….o`，装入 `${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math`。host/tiling/aclnn 与源仓一致，无需重编。

## 3. 精度

| 套件 | 结果 |
|------|------|
| `test/python_test/test_hc_pre_sinkhorn.py`（10 func） | **10/10 PASS** |
| cann-bench 20 case（y / post / comb） | **20/20 PASS** |

阈值：y bf16 rtol/atol=1e-2；post fp32 1e-4；comb fp32 1e-3。

SoA 路径最大 abs：y ≤ 1.5e-5，post ≤ 2.4e-7，comb ≤ 2.1e-7。与源仓 xllm-ops 合入结果一致。

## 4. 性能（HcPreSinkhorn kernel µs）

设备：Ascend950PR。口径：torch_npu profiler。  
对照：源仓合入实测、A5 `mhc_sinkhorn` baseline。

| case | bs | M | iter | my-xllm us | xllm-ops us | A5 us | vs A5 |
|------|----|---|------|------------|-------------|-------|-------|
| 1 | 1 | 4 | 20 | 3.142 | 3.138 | 5.36 | 1.71x |
| 2 | 64 | 4 | 20 | 4.546 | 4.650 | 5.82 | 1.28x |
| 3 | 512 | 4 | 20 | 5.512 | 5.505 | 8.47 | 1.54x |
| 4 | 1024 | 4 | 20 | 5.407 | 5.418 | 10.00 | 1.85x |
| 5 | 4096 | 4 | 20 | 6.978 | 6.944 | 10.00 | 1.43x |
| 6 | 16384 | 4 | 20 | 12.884 | 12.828 | 15.90 | 1.23x |
| 7 | 1024 | 2 | 20 | 4.640 | 4.653 | 10.00 | 2.16x |
| 8 | 1024 | 3 | 20 | 5.323 | 5.310 | 10.00 | 1.88x |
| 9 | 1024 | 6 | 20 | 18.812 | 18.768 | 10.00 | 0.53x |
| 10 | 1024 | 8 | 20 | 20.786 | 20.850 | 10.00 | 0.48x |
| 11 | 1024 | 12 | 20 | 24.255 | 24.337 | 10.00 | 0.41x |
| 12 | 1024 | 16 | 20 | 29.423 | 29.455 | 15.50 | 0.53x |
| 13 | 1024 | 4 | 1 | 4.635 | 4.646 | 4.17 | 0.90x |
| 14 | 1024 | 4 | 5 | 4.521 | 4.548 | 5.65 | 1.25x |
| 15 | 1024 | 4 | 40 | 6.498 | 6.578 | 10.00 | 1.54x |
| 16 | 1024 | 4 | 20 | 5.760 | 5.745 | 10.00 | 1.74x |
| 17 | 1024 | 4 | 20 | 5.376 | 5.376 | 10.00 | 1.86x |
| 18 | 1 | 2 | 20 | 3.070 | 3.053 | 10.00 | 3.26x |
| 19 | 1024 | 4 | 20 | 5.524 | 5.427 | 10.00 | 1.81x |
| 20 | 1024 | 4 | 20 | 5.724 | 5.763 | 10.00 | 1.75x |

- 与源仓 xllm-ops：**数值对齐**（差 < 0.15 µs，测量抖动）
- vs A5：**15/20 更快**，大 batch M≤4 与源仓同档（case 6 ≈ 12.9 µs）
- M≥6 仍走 AoS，与 A5 比偏慢（算子还含 pre/y/post，A5 仅 Sinkhorn）

墙钟（aclnn 下发）约 260–400 µs，不能当 kernel 时间。

原始数据：`docs/perf-integrate/my_xllm_20cases_perf.json`

## 5. 结论

代码已同步到 my-xllm-ops，精度 10/10 + 20/20 通过，性能与源仓合入结果一致。
