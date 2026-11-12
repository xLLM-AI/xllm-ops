# MegaKdaDecode

GLM-5.3-Flash vector KDA 的实验性 Conv + recurrent decode 融合算子。
接入结构参考 `mega_gdn_prefill_op`，包含独立的 OpDef、Shape 推导、Host tiling、
PTO Device 入口、CMake 构建和 ACLNN/Python 测试绑定。尚未接入 xLLM 模型调用路径。

## 入口与状态

- 仅一个 `MegaKdaDecode` 算子，Host 根据显式 mode 设置 tilingKey。
- `mode=0/key=1` 为普通单 token decode；`mode=1/key=2` 为 MTP verify。
  单 token 的 MTP 仍走 key 2，不根据 token 数量猜测语义。
- Conv 与 SSM 各自有独立输入池、输出池和读写映射，输入输出容量允许不同。
  Python 绑定拒绝输出与输入共享 storage，Device 不修改输入状态或 QKV。
- accepted count、offsets 和 slot 映射均为 Device tensor，不固化为 Host tiling 属性。
  MTP 使用 `accepted-1` 恢复 SSM checkpoint 与 Conv history；采样接受判定不在算子内。
- slot 0 有效，读 slot -1 表示 inactive，写 slot -1 表示丢弃该项状态写回。
  调度器必须保证有效写目标唯一；Host 不通过 D2H 校验动态映射。

## 支持范围

目前仅注册 Ascend 910B。BF16 输入、FP32 `[slots,H,V,K]` SSM，D=128，
H=1..128，MTP query capacity=1..17。模型 MTP=K 表示 K 个草稿 token，
验证时还包含当前 token，因此 capacity=K+1；模型 MTP1..16 对应 capacity2..17。
capacity=1 仍支持单 token MTP verify。普通 decode 的 Conv history 长度为 3，
MTP 为 capacity+2。Conv 权重按 MegaGDN 的 `[4,C]` 布局预打包，不能逐次转置。

与 `MegaKdaPrefill` 不同，decode 的 beta 是 sigmoid 前的 BF16 投影，sigmoid
在算子内部计算；prefill 接收 sigmoid 后的 FP32 beta。decode 的读 slot -1
表示 inactive，不是 prefill 的零初态。阶段切换时必须分别遵守各自接口契约。

数学边界保留 SiLU Conv、BF16 Conv 输出舍入、Q/K L2 norm、KDA 向量 gate、
sigmoid beta 与逐 token FP32 状态更新，不复用 GDN 的标量 decay 公式。
每个 task 负责 32 行 V，使用 AIV 与 62,368 字节显式 UB，无用户 workspace。
V 行分片只改变任务粒度，不改变数学计算或状态契约；不代表所有 batch、head 数
和 MTP capacity 均已获得性能收益。

## 验证边界

构建入口为 `bash build.sh --op-name mega_kda_decode`。
测试为 `test/python_test/test_mega_kda_decode.py`，未设置 `MEGA_KDA_TEST_NPU`
时只执行 CPU 合同测试；真机测试需显式选择独占 NPU。
测试包含 fixed20、独立 CPU/NPU carry20、动态 accepted/mapping Graph replay、
padding、不同状态池容量和 alias 拒绝。
fixed20 使用图模式并按原始字节比较，区分正零与负零。
CPU golden 的 Conv、归一化、gate 和递推使用 FP64，保留激活 FP16 转换与
公开 BF16/FP32 输出边界。shape/dtype 不匹配或输出、golden、派生误差出现
NaN/Inf 一律失败。按 `abs(golden)=1e-3` 分组报告正常值相对误差、近零值
绝对误差及超限比例，作为额外诊断；原逐元素验收阈值不变：输出
`rtol=1e-2, atol=1e-3`，状态 `rtol=1e-3, atol=1e-4`。

编译成功、CPU golden 自检成功都不等于原生算子精度通过。仍需完成真实 ACLNN/Graph
状态保持验证、冻结 Triton 基线对齐、预热收敛和正式性能验收，之后才能打开模型路由。
目前不声明超过 Triton 20%，也不声明整网精度通过。

TP 矩阵按 64 个 KDA heads 切分，覆盖 TP1/2/4/8/16/32/64，对应每卡
H=64/32/16/8/4/2/1。每档交叉普通 decode 与全部整数 query capacity1..17，
覆盖模型 MTP1..16，包含 CPU
golden、dense fixed20 图回放、动态元数据 replay20 和独立状态 carry20。
原 batch/ragged/padding 用例保留，Host policy 也覆盖全部 head 数。
可用 `pytest -k tp` 选择矩阵，`-k tp8` 选择单档；这只是单卡分片 shape
测试，不代表多卡 TP 通信、模型显存容量或整网验证通过。

完整参数、状态前置条件和复现命令见 [英文接口合同](../en/mega_kda_decode.md)。
