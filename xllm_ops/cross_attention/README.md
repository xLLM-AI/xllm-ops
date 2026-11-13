# CrossAttention Ascend950 集成

2026-11-12 将 `work/cross_attention` 已验收的 native CV 直通实现接入本目录的 ACLNN 入口。A5 / Ascend950 / DAV_3510 使用 `arch35`；其他 SoC 沿用既有 CATLASS 路径。

## 实现

- `op_kernel/arch35/cross_attention_fusion_kernel.h`：已验收的 AIC/AIV kernel 主体，包含 L1/UB 直通、分段合并与环形同步；本次仅适配 include 路径并保留项目 license。
- `op_kernel/arch35/cross_attention_common.h`：与直调版本逐字一致，地址和环深常量保持原值。
- `op_kernel/arch35/cross_attention_tiling.h`：host/device 共用 native POD，`namespace Xllm` 避免与自动生成的 legacy tiling 类冲突。大小 120 bytes，`mergeSlotBytes` offset 72，`tilingVersion` offset 112。定义放在 kernel 目录，供 OPC 的 kernel-only 源码副本使用。
- `op_host/arch35/cross_attention_tiling.cpp`：原 tiling 计算，增加 `Xllm` namespace。
- `op_host/arch35/cross_attention_tiling_arch35.cpp`：形状检查、native tiling 到注册 schema 的序列化、workspace 及 blockDim 设置。注册类为 `CrossAttentionArch35TilingData`，key 为 3510。
- `op_kernel/cross_attention.cpp`：A5 分支登记并 dispatch key 3510 / MIX AIC:AIV=1:2，通过 `GetUserWorkspace` 取得用户 workspace；调用 native AIC/AIV helper。

`op_host/CMakeLists.txt` 在 Ascend950 构建时设置正式编译选项 `-DENABLE_CV_COMM_VIA_SSBUF=true`。P 的布局选择、AIC/AIV 同步协议及核心计算均与验收版本一致；源码没有实验开关。

## 构建

在 Linux 工程根目录执行，固定 CANN 与 device visibility：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
bash xllm_ops/build.sh --pkg --ops=cross_attention --soc=ascend950 -j8
```

框架的 `*_src_copy.done` 缓存不会自动随 kernel 子目录内容更新。修改后应使用新的 build-dir，或使对应源码复制及 binary 构建标记失效后再编译。

## 本轮验证

环境：`a5host4` / 容器 `ware` / CANN 9.1.0 / `ASCEND_RT_VISIBLE_DEVICES=1`。

| 检查 | 结果 |
| --- | --- |
| Ascend950 完整 `.run` 包构建 | exit 0 |
| fp16 / bf16 binary metadata | 均为 key 3510 / MIX 1:2 |
| native tiling 对冻结 reference、注册 schema 对 native POD | 8400/8400 PASS |
| 新包 ACLNN 全入口，原 41 用例及 Golden | 41/41 PASS，0 failed，0 env errors |
| 精度阈值 | bf16 atol=rtol=1e-2；fp16 atol=rtol=1e-3，与原测试一致 |
| kernel / common / Compute / POD 等价性 | PASS |
| 本地与远程验证副本的完整目录 SHA256 | 一致 |

ACLNN 精度测试使用临时 C++ bridge 调用包内 `aclnnCrossAttentionGetWorkspaceSize` / `aclnnCrossAttention`，每次调用同步后释放元数据与 workspace。该 bridge 仅用于精度，时延不作为性能结论。2026-11-13 用同一入口对本包做了全用例实测：41 条精度用例逐条测得 kernel 耗时（连续下发，200 次/轮、5 轮取中位数），q64 b8 kv700 为 26.02 µs（单次含同步 32.86 µs），与历史直调 26.3–26.4 µs 一致；完整表见 `work/cross_attention/tmp/xllm_sync_20261112_090814/性能实测-全用例.md`。复用同一 aclOpExecutor 重复下发时，b1 q3 kv300/460/700、b2 q3 kv1200、b1 q8 kv1536 会退化到约 1.04 s/次（结果正确，仅时延），每次调用重建 executor 的生产路径不受影响。A2 kernel/helper/common 保持接管时内容，本轮没有 A2 硬件回归。

完整报告、原文件快照及实测日志保存在 `work/cross_attention/tmp/xllm_sync_20261112_090814/`。远程验证副本及构建包位于 `/data/wangguangcai/code/jd/ascend_ca/tmp/xllm_sync_20261112_090814/`。

## 本轮清理（2026-11-13）

代码归属与可读性整理，不改算法、不改 tiling ABI、不改地址与环深常量值。

- namespace：本目录内所有 `ca` 改为 `Xllm`，涉及 `op_kernel/arch35/cross_attention_fusion_kernel.h`、`op_kernel/arch35/cross_attention_tiling.h`、`op_kernel/arch35/cross_attention_fusion_helper.h`、`op_kernel/cross_attention.cpp`、`op_host/arch35/cross_attention_tiling.cpp`、`op_host/arch35/cross_attention_tiling.h`、`op_host/arch35/cross_attention_tiling_arch35.cpp`。CATLASS 路径的 `op_host/cross_attention_tiling.cpp`、`op_kernel/cross_attention_catlass_helper.h` 不在本目录内，未动。
- 注释：`arch35` 下注释全部改为英文，保留 L1/L0/UB 布局与字节偏移、event id 分配、跨核 flag 协议、拍号连续性、`kSeg>1` 全局栅栏等约束说明；删除历史试错叙述、实测性能数字、内部代号与指向设计文档章节的引用。
- 删除的冗余常量：`CA_R_BLOCKS`、`CA_UB_TOTAL` 全仓零引用；`CA_UB_MRG_BYTES` 仅被 `CA_UB_TOTAL` 使用，随之删除，UB 预算校验改为内联 `static_assert`。`CA_UB_MRG` 仍有真实引用，保留。
- 删除的冗余字段：`CrossAttentionShape::numBlocks` 无读取（该结构仅 host 侧构造，不进 tiling ABI）；`cross_attention_tiling_arch35.cpp` 中对 `CrossAttentionShape::batch` 的重复初始化。
- 删除的冗余形参：`StoreML`、`WriteSentinelSlot` 未使用的 `const CaTaskDesc& d` 及其 `(void)d;`，`CopyPToL1` 未使用的 `beat`，调用点已同步。属冗余形参清理，非重构。
- ABI 不变：`CrossAttentionTilingData` 仍为 120 bytes，`mergeSlotBytes` offset 72，`tilingVersion` offset 112。

### 本轮验证

只在缓存标记失效后重编才有意义：仅让源码复制标记失效时，`gen/cross_attention_ascend950_{0,1}.done` 会让 kernel 编译步骤跳过，包内 kernel `.o` 仍是旧文件。本轮同时失效源码复制标记与这两个编译标记后重编。

| 检查 | 结果 |
| --- | --- |
| Ascend950 完整 `.run` 包构建 | exit 0，8 s；包内 kernel `.o` 时间戳刷新为本次构建 |
| 41 用例精度 | 41/41 PASS，0 failed，0 env errors |
| q64 b8 kv700 kernel 耗时（bf16 / fp16） | 25.99 / 25.98 us；基线 25.95 / 25.97 us |
| 41 用例逐条 kernel 耗时对比基线 | 40 组可比数据平均 +0.23%，单形状最大 6.9%，与同代码重复测量的噪声同量级 |
| 重编后与旧二进制逐条对比 | 平均 +0.08%，最大 5.1%，确认为测量噪声 |
| 冗余符号扫描 | `arch35` 下 205 个命名空间级声明（常量 / using / 类型 / 函数 / POD 字段）全部有真实引用；编译无 unused 告警 |
