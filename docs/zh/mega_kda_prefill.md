# MegaKdaPrefill

参考 `mega_gdn_prefill_op` 的注册、构建和单次混合核启动结构，新增 GLM
vector KDA prefill。完整参数和边界见 [英文接口文档](../en/mega_kda_prefill.md)。

融合范围为 Conv/SiLU、Q/K L2 归一化、vector gate、chunk cumsum，以及
KKT、FP32 inverse、WY、H、O。保留 FP32 A2、K_eff/W 和状态计算；不包含输出
norm/projection，不修改模型路由，也不提供 decode/MTP 兼容分支。

## 输入输出

- BF16 原始 QKV `[T,3*H*128]` 和原始 gate `[T,H,128]`。
- beta 是 FP32 **sigmoid 后的结果** `[T,H]`，不是原始投影。
- `a_log [H]`、`gate_bias [H,128]` 为 FP32，gate 下界固定为 -5。
- Conv 权重 `[4,3*H*128]`，可选 bias `[3*H*128]`，均为 BF16。
- Conv 输入/输出池分别为 `[Ci,3,3*H*128]`、`[Co,3,3*H*128]` BF16。
- SSM 输入/输出池分别为 `[Si,H,V=128,K=128]`、`[So,H,128,128]` FP32。
- 输出 `[T,H,128]` 为 BF16，三个输出均由调用者提前分配。
- `cu_seqlens [B+1]` 和四组独立读写索引 `[B]` 都是 Device int32。

读索引 `-1` 表示零初态，写索引 `-1` 表示不发布该类状态；空序列不写状态，
未写槽位保持原值。输出与任意输入及彼此必须没有存储别名。不保留原地更新路径。
同类状态池的有效写槽位必须唯一。索引和 offsets 每次执行从 Device 读取，
支持 Graph replay 改变其内容，但一次调用期间不可修改。

## 验证边界

支持 Ascend910B、chunk/head dimension 128、Conv width 4。
与既有 MegaGDN 绑定一致，调用前须将输入所在 NPU 设为当前设备。
CPU golden 使用 FP64 递推和明确的 BF16 舍入边界，正常值相对误差、近零值
绝对误差及超限比例分别报告。测试包含空序列、尾块、独立池容量、输入不变、
未写区保持，以及 100 轮固定输入/独立推进状态的图回放。

当前单算子 smoke 门槛：以 `abs(golden)=1e-3` 分组，正常值相对误差阈值
为 1%，近零值绝对误差阈值为 `1e-5`，两类超限比例各不超过 0.1%。这不等于
所有元素误差均小于 1%，也不代表模型精度验收。1k 至 32k 各档另检验 20 轮
固定输入图回放的逐比特一致性；TP1、2k 加强到 1000 轮，以覆盖 chunk 状态
缓冲区复用引起的延迟错误。每轮分别检查输出、Conv state 和 SSM state，
按原始字节比较，区分正零与负零。
输出和 golden 必须同 shape、同 dtype；输出、golden 或派生误差出现 NaN/Inf
一律失败。CPU 漏洞回归见 `test_mega_kda_accuracy_checks.py`。

TP shape 测试按模型的 64 个 KDA heads 设置：TP1/2/4/8/16/32/64 对应
每卡 H=64/32/16/8/4/2/1。每档交叉覆盖 1k 至 32k、带/不带 Conv bias 的
尾块与空序列，以及 fixed100/carry100 图回放；CPU golden 和 Host policy
也覆盖这 7 档 head 数。可用 `pytest -k tp` 选择矩阵，`-k tp8` 选择单档。
这是单卡执行的分片 shape 验证，不是多卡 TP/HCCL 或整网验收。

torch_npu 2.9 的 NPUGraph 每次回放前使用
两个 Fill 核更新 RNG seed 和 offset；它们属于框架回放开销，不是本算子的
workspace 或状态初始化，算子中不包含这类 Fill 实现。纯 kernel 耗时应单列，
完整图调用性能对比应计入框架开销。

只有 CPU 测试或编译通过不能宣称 NPU 精度通过。历史模型结果不能替代
本次接入的精度、性能或模型验收；正式性能收益另测。
