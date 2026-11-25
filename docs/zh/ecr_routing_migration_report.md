# ecr 路由三算子迁移与验证报告

本报告整理 `ecr_expert_selection_threshold`（S2）、`ecr_token_routing_statistics`（S3）、`ecr_routing_normalize_compact`（S4）三个算子从内核直调（`<<<>>>`）形态迁移为 CANN aclnn 自定义算子的过程、调测记录、用例与精度判定、以及性能对比结果。

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 迁移算子 | 3 个（S2 / S3 / S4），各自独立目录树，算子之间无共享文件 |
| 目标平台 | Ascend950PR（dav-3510），CANN 9.2.1，aarch64 |
| 编译产物 | `cann-ops-xllm-custom_linux-aarch64.run`，安装为 vendor `custom_xllm_math` |
| 精度 | 3 个测试文件共 **138 项全部通过**（4 卡） |
| 与上游一致性 | S2 三个输出逐元素完全一致；S3 `sums/cnts` 完全一致；S4 `ids` 完全一致、`weights` 在 1e-6 内一致 |
| 端到端延迟 | 21 个用例逐条 A/B（12 轮取值）：中位 S2 2.14×、S3 2.53×、S4 2.91×；最长序列回落到 1.5×，详见 §6 |

## 2. 环境与约束

| 项 | 值 |
|---|---|
| 跳板机后 SSH 主机 | `11.169.208.39:20022`（内网地址 `192.168.24.82`） |
| 容器 | `docker_wjh_001`（`cann:9.2.1-950-openeuler24.03-py3.12-devel`） |
| 工作目录 | `/home/w00501444/LanguageTest`（宿主 ↔ 容器 bind mount，同路径） |
| 工具链 | CANN 9.2.1（`/usr/local/Ascend/cann-9.2.1`）、Python 3.12.16、torch 2.10.0+cpu、torch_npu 2.10.0.post7 |
| 编译工具 | gcc/g++ 12.3.1（容器内 oe2403sp4）、bisheng（上游基线）、ccec（aclnn kernel） |
| 卡约束 | 只使用 NPU 0-3；NPU 0/1 长期有其他用户进程占用，性能测量择空闲卡（多为 NPU 2/3）进行 |

所有编译与运行都在容器内执行；宿主机上的 `ascend-toolkit/latest` 指向 CANN 9.3.0 且 Python 缺 `regex`，在宿主机直接编译会失败。

## 3. 迁移方案

### 3.1 形态差异

| 维度 | 上游（迁移前） | 本仓库（迁移后） |
|---|---|---|
| 形态 | bisheng 编译的独立 torch 扩展 `.so`，`torch.ops.npu.*` | CANN aclnn 自定义算子，注册进 vendor |
| kernel | `op_kernel/*_kernel.cpp` 内 `<<<block_dim, nullptr, stream>>>` 直调 + host tiling 结构体 | `op_kernel/*.cpp`（AscendC 入口）+ `*_vf.h`（RegBase SIMD 主体，几乎原样复用） |
| host | torch 扩展里做校验、tiling、`aclrtMemcpy` 下发 tiling | `op_host/`：`*_def.cpp`（算子原型）、`*_infershape.cpp`、`*_tiling.cpp`、`*_contract.h`（形状/属性契约） |
| 调用 | `torch.ops.npu.<op>(...)` | `aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`（通过 `custom_ops_lib` 绑定） |

### 3.2 目录结构与隔离

```
xllm_ops/expert_chosen_routing2/
├── ecr_expert_selection_threshold/
│   ├── CMakeLists.txt
│   ├── op_host/{_def.cpp, _infershape.cpp, _tiling.cpp, _tiling.h, _contract.h}
│   └── op_kernel/{ecr_expert_selection_threshold.cpp, _vf.h}
├── ecr_token_routing_statistics/      # 同上结构，独立一份
└── ecr_routing_normalize_compact/     # 同上结构，独立一份
```

三个算子按"完全隔离、不共享文件"的要求各自持有一份共性的契约/校验代码，任一算子可单独编译、单独删除，不影响另外两个。

### 3.3 算子接口

| 算子 | 输入 | 输出 | 属性 |
|---|---|---|---|
| S2 `EcrExpertSelectionThreshold` | `prob` fp32 [E,S] | `tau_val` fp32 [E,B]、`tau_idx` int32 [E,B]、`n_gt` int32 [E,B] | `seq_lengths` listInt、`top_p` float |
| S3 `EcrTokenRoutingStatistics` | `prob` fp32 [E,S]、`tau_val` fp32 [E,B]、`tau_idx` int32 [E,B] | `sums` fp32 [S]、`cnts` fp32 [S] | `seq_lengths` listInt |
| S4 `EcrRoutingNormalizeCompact` | `prob` fp32 [E,S]、`tau_val`、`tau_idx`、`sums` fp32 [S]、`cnts` fp32 [S] | `ids` int32 [E,C]、`weights` fp32 [E,C]，`C = Σ ceil(len·top_p)` | `seq_lengths` listInt、`top_p` float、`eps` float |

约束：E ∈ [1,64]，B ≤ 32，单样本长度 ≤ 2097152，`seq_lengths` 为空或单元素表示全局单样本（该值本身被忽略，按 S 处理），否则各段必须非负且和为 S。所有张量 ND 格式、支持动态 shape 与动态 rank。

### 3.4 迁移步骤（每个算子重复一遍）

1. 复用上游 `*_vf.h` 的寄存器级实现，只把它从"直调 kernel 内联"改为 `op_kernel/*.cpp` 中的 AscendC kernel 入口调用。
2. 把上游 host 侧的参数/形状校验抽成 `*_contract.h` 的 `BuildContract`，由 infershape 与 tiling 共用，保证两条路径对同一输入给出同一结论。
3. `*_def.cpp` 按上游语义声明输入/输出/属性；`*_tiling.cpp` 将契约结果（E/S/B、各段 base/len/k、top_p、eps）写入 tiling 结构体。
4. 在 `test/python_test/RegisterOps.cpp` 增加 aclnn 绑定，按 `EcrCompactCapacity` 之类的语义在 host 侧先算出输出容量（S4 的 C 需要提前分配）。
5. 补 `test/python_test/test_*.py`：与独立的 numpy 稳定排序 oracle、torch golden 双向对比。
6. 编包安装，上机跑用例；与上游扩展做逐元素比对后再采性能。

## 4. 调测过程与问题记录

按出现顺序，问题、根因与处理如下。

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | 远程命令通道不可用（MCP ssh 服务未连上） | 会话内 MCP 工具不可达 | 改用 WinSCP `call` + `put/get` 的封装通道（`call` 命令不能加引号，否则参数被转义破坏） |
| 2 | 配置阶段卡在下载 makeself / nlohmann json / abseil / protobuf | 内网无外网 DNS，`third_party` 需离线准备 | 从同机已有副本拷贝 `makeself`、`pkg/*.tar.gz`、`json/include` |
| 3 | `ModuleNotFoundError: No module named 'regex'` | 构建脚本依赖 `regex`，镜像未装 | 从同机其他 Python 环境拷入容器 site-packages |
| 4 | `No SOURCES given to target: cust_opmaster` | `cmake/func.cmake` 用 `**` glob 发现算子，而 CMake 的 `**` 只匹配一层，`expert_chosen_routing2/<op>/op_host` 永远匹配不到 | 在 `func.cmake` 增加 `expert_chosen_routing2/**/op_host/CMakeLists.txt` 显式 glob |
| 5 | `expected primary-expression before 'float'`（S2/S4 contract） | 模板函数内 `const auto* attrs = ctx->GetAttrs()` 是依赖类型，成员模板调用需消歧 | 改为 `attrs->template GetAttrPointer<float>(...)` |
| 6 | `No rule to make target 'autogen/inner/aic-ascend950-ops-info.ini'` | 用 `-n` 只编部分算子时没有 inner/exc 组算子，opbuild 不产出这两个 ini，但 `generate_ops_info` 把它们写进 `DEPENDS` | `func.cmake` 中改为仅当对应组非空时才依赖该 ini（`parse_ini_to_json.py` 本身会跳过不存在的文件） |
| 7 | 内核编译 `StoreAlign: expects an lvalue for 3rd argument` | 把 `reg::CreateMask<...>()` 临时量直接传给 `MaskReg&` 形参，ccec 不接受 | 先赋给 `reg::MaskReg` 变量再传 |
| 8 | 改了内核但报错行号/内容仍是旧代码 | 增量构建时 `build/binary/ascend950/src/` 里的源码副本不刷新 | 改内核后先删 `build/binary/ascend950/{src,gen}` 再编 |
| 9 | 在宿主机编译，配置期报 `regex` 缺失、CANN 版本变 9.3.0 | 构建必须在容器内执行（容器 CANN 9.2.1 + 完整 Python 依赖） | 统一 `docker exec docker_wjh_001 bash -lc 'cd /home/... && bash build.sh ...'` |
| 10 | 容器内无 pytest | 镜像未安装且无外网 | 从本机 tileLang 环境的 deps 中抽 `pytest/_pytest/iniconfig/pluggy/packaging/py` 到独立目录，用 `PYTHONPATH` 引入，不改系统环境 |
| 11 | S4 用例 14 个失败：`Failed to add attr[0] ... CheckNnopbaseAddIntArrayAttr` | S4 绑定把空 `seq_lengths` 直接作为 int 数组属性下发，aclnn 属性转换拒绝空数组；S2/S3 绑定此前已做 `[S]` 替换，S4 漏了 | `RegisterOps.cpp` 中 S4 同样把空列表改写为 `[S]`（与契约中"count ≤ 1 视为全局单样本"语义一致） |
| 12 | 首次性能测量结果抖动剧烈（上游 0.1–1.4 ms） | NPU 0/1 上有其他用户进程占用，且上游每次调用都做同步 H2D 拷贝，对主机负载敏感 | 换空闲卡复测，并按卡确认无其他进程后再计时 |
| 13 | 逐用例 benchmark 把 aclnn 与上游**逐次**背靠背交替下发（同一进程内两种实现的内存块交错申请释放）时，跑十几分钟后报 `AI Core error`：MTE 访问非法 GM 地址、`Page fault ... valid_range_size=25165824` | 两套实现用不同的分配器（aclnn/aclrtMalloc 与 torch 缓存分配器）。逐次交错时，一方申请的内存可能落在另一方刚释放、尚未重新映射的地址窗口上，内核访问到的是"半释放"的地址 | 改成**块级交替**：一个用例内 aclnn 跑完整块、golden 跑完整块，只在轮次之间翻转先后次序，不再逐次交错；块级交替下 12 轮无异常。单独用任一后端连跑 3000 次也不复现，说明触发条件是两套分配器混用，而非单个算子的问题 |

## 5. 用例与精度

### 5.1 用例矩阵

三个算子共用同一套 21 个用例（各自文件内独立定义）：

| ID | S | E | 源精度 | top_p | 分段 lengths | 特殊构造 |
|---|---|---|---|---|---|---|
| F1 | 4096 | 48 | fp32 | 0.125 | 单样本 | — |
| F2 | 4096 | 48 | fp16 | 0.125 | 单样本 | 源值按 fp16 量化 |
| F3 | 4096 | 48 | bf16 | 0.125 | 单样本 | 源值按 bf16 量化 |
| F4 | 4096 | 48 | fp32 | 0.125 | [1024,1024,1024,1024] | B=4 |
| F5 | 4096 | 48 | fp32 | 0.125 | [512,1024,1536,1024] | 不均分 4 段 |
| F6 | 40960 | 48 | bf16 | 0.125 | 单样本 | 长序列 |
| F7 | 4096 | 48 | fp32 | 0.125 | 单样本 | 30% 重复值（平局） |
| F8 | 4096 | 48 | fp32 | 1.0 | 单样本 | 全选 |
| F9 | 4097 | 48 | fp32 | 0.125 | 单样本 | 非对齐长度 |
| F10 | 4096 | 33 | fp32 | 0.125 | 单样本 | 非 32 对齐 E |
| L1 | 65537 | 33 | fp32 | 0.125 | 单样本 | 大于 65536 |
| L2 | 131072 | 48 | bf16 | 0.125 | 单样本 | 长序列 + 重复值 |
| L3 | 131073 | 33 | fp16 | 0.125 | [65537,65536] | 两段 + 奇数长度 |
| L4 | 40961 | 33 | fp32 | 1.0 | 单样本 | 全选 + 非对齐 |
| ZERO | 17 | 3 | fp32 | 0.125 | 单样本 | prob 全 0 |
| TIES | 33 | 3 | fp32 | 0.125 | 单样本 | prob 全 0.5（全平局） |
| EMPTY | 0 | 3 | fp32 | 0.125 | 单样本 | S=0 空张量 |
| EMPTY_MIX | 17 | 3 | fp32 | 0.125 | [0,8,0,9] | 空段与非空段混合 |
| TINY | 17 | 3 | fp32 | 1e-15 | [8,9] | 极小 top_p（每段至少选 1） |
| ONE | 1 | 1 | fp32 | 1.0 | 单样本 | 最小合法形状 |
| SINGLE_IGNORED | 17 | 3 | fp32 | 0.125 | [999] | 单元素 lengths 被忽略 |

补充说明：算子 I/O 契约是 fp32（`tau_idx/ids/n_gt` 为 int32）。矩阵中标注 fp16/bf16 的用例，是把源 logits 先量化到 fp16/bf16 再转回 fp32 后交给算子，用于覆盖这些格式可表示的数值分布，并非算子本身支持半精度输入。

每个测试文件除上述 21 个用例的 NPU 与 golden 两轮（42 项）外，还包含：

- `seq_lengths` 三种书写形式（Python list / int32 CPU tensor / int64 CPU tensor）等价性 3 项；
- 非法 lengths（各段之和 ≠ S）必须在下发前被拒绝 1 项。

即每个算子 46 项，合计 138 项。

### 5.2 精度判据

| 算子 | 判据 |
|---|---|
| S2 | `tau_val / tau_idx / n_gt` 与 numpy 稳定排序 oracle **逐元素完全相等**（含 dtype 与 shape）；另外反向校验由 `(tau_val, tau_idx)` 推出的选择掩码与 oracle 选中的 token 集合完全一致；零值/平局/空样本走 2.0/-1/0 哨兵约定 |
| S3 | `sums` 逐元素完全相等，`cnts` 逐元素完全相等（计数不允许近似） |
| S4 | `ids` 逐元素完全相等；`weights` 满足 `\|Δ\| ≤ 1e-6 + 1e-5·\|ref\|` |
| 与上游交叉验证 | 同一输入下：S2 三输出 `torch.equal`；S3 `sums/cnts` `torch.equal`；S4 `ids` `torch.equal`、`weights` 满足 1e-6 绝对/相对容差 |

`tau_idx` 的平局规则是"取更小 token"，`tauIdx` 为样本内局部下标；S4 输出的 `ids` 为全局 token 下标、未使用槽位写 -1/0。

### 5.3 结果

```
test_ecr_expert_selection_threshold.py    46 passed
test_ecr_token_routing_statistics.py      46 passed
test_ecr_routing_normalize_compact.py     46 passed
-----------------------------------------------
合计                                     138 passed in 26.09s
```

运行条件：容器内 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`。

## 6. 性能对比

### 6.1 方法

- 上游基线在同一台机器上按原始工程用 bisheng（`--npu-arch=dav-3510`）编译出 `libecr_*_ops.so`，与本仓库 aclnn 算子在同一进程、同一张卡上对比。
- **测量的用例集合就是 §5.1 的 21 个用例**，不是挑几个规模：每个用例对 S2/S3/S4 各测一次，逐用例做 A/B，因此性能表可以直接和精度矩阵按行对照。
- 计时前先逐用例做输出一致性校验（判据同 5.2），不一致的用例标记 `[WAIVED]` 跳过，避免"基线算错但数字好看"。本轮 21 × 3 个组合全部通过，无一豁免。
- 单次计时：`warmup=20`、`iters=100`，每次调用前后各打一个 `torch.npu.Event`，取 p50；一个用例内某个后端跑完一整块，块与块之间翻转先后次序（偶数轮 aclnn 先，奇数轮 golden 先），不做逐次交错（原因见 §4 问题 13）。
- 每用例每后端采 **12 轮**（两次完整运行 × 6 轮），共 21 × 3 × 2 × 12 = 1512 组计时。
- 取值口径：
  - **aclnn 取 12 轮 p50 的中位数**。小 kernel 下偶发一次事件低读（例如 `TIES`/S4 最好一轮 0.0105 ms，中位 0.0446 ms），取最优会虚高。
  - **golden 取 12 轮中的最优 p50**。该机器共享，上游扩展每次调用都做一次同步 `aclrtMemcpy(H2D)` 拷 tiling，主机负载只会让它**单向变慢**；12 轮 p50 的极差在 47/63 个「用例 × 算子」组合上超过 2×（41/63 超过 3×），最高 13.7×（表内"轮间极差"列）。取中位等于把别人的负载算成我们的收益，取最优轮代表上游在安静窗口下的能力，是对基线最有利的口径——**因此表内加速比是下界**。
- 卡选择：`npu-smi` 确认空闲后使用 NPU 3（NPU 0/1 有他人训练进程，NPU 2 被占 34 GB）。

### 6.2 结果

`aclnn p50` = 12 轮中位；`golden p50` = 12 轮最优；`轮间极差` = golden 最差轮 / golden 最优轮，用于看该用例的噪声水位。

#### S2 `ecr_expert_selection_threshold`

| 用例 | S | E | 源精度 | top_p | 分段 | aclnn p50 (ms) | golden p50 (ms) | golden 轮间极差 | 加速 |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 4096 | 48 | fp32 | 0.125 | - | 0.0545 | 0.1632 | 5.02× | 3.00× |
| F2 | 4096 | 48 | fp16 | 0.125 | - | 0.0422 | 0.1982 | 4.16× | 4.69× |
| F3 | 4096 | 48 | bf16 | 0.125 | - | 0.0425 | 0.2317 | 2.33× | 5.45× |
| F4 | 4096 | 48 | fp32 | 0.125 | 1024×4 | 0.0437 | 0.3889 | 2.05× | 8.89× |
| F5 | 4096 | 48 | fp32 | 0.125 | 512,1024,1536,1024 | 0.0439 | 0.1024 | 6.14× | 2.33× |
| F6 | 40960 | 48 | bf16 | 0.125 | - | 0.0650 | 0.1080 | 2.04× | 1.66× |
| F7 | 4096 | 48 | fp32 | 0.125 | - | 0.0425 | 0.0907 | 7.01× | 2.14× |
| F8 | 4096 | 48 | fp32 | 1.0 | - | 0.0431 | 0.0919 | 4.93× | 2.13× |
| F9 | 4097 | 48 | fp32 | 0.125 | - | 0.0456 | 0.0958 | 1.67× | 2.10× |
| F10 | 4096 | 33 | fp32 | 0.125 | - | 0.0459 | 0.0966 | 1.16× | 2.10× |
| L1 | 65537 | 33 | fp32 | 0.125 | - | 0.0707 | 0.1210 | 4.11× | 1.71× |
| L2 | 131072 | 48 | bf16 | 0.125 | - | 0.2426 | 0.3859 | 1.01× | 1.59× |
| L3 | 131073 | 33 | fp16 | 0.125 | 65537,65536 | 0.1307 | 0.1932 | 1.98× | 1.48× |
| L4 | 40961 | 33 | fp32 | 1.0 | - | 0.0494 | 0.0947 | 1.16× | 1.92× |
| ZERO | 17 | 3 | fp32 | 0.125 | - | 0.0431 | 0.0874 | 5.58× | 2.03× |
| TIES | 33 | 3 | fp32 | 0.125 | - | 0.0457 | 0.0980 | 5.17× | 2.14× |
| EMPTY | 0 | 3 | fp32 | 0.125 | - | 0.0436 | 0.1029 | 4.75× | 2.36× |
| EMPTY_MIX | 17 | 3 | fp32 | 0.125 | 0,8,0,9 | 0.0442 | 0.1010 | 5.07× | 2.29× |
| TINY | 17 | 3 | fp32 | 1e-15 | 8,9 | 0.0423 | 0.1037 | 3.77× | 2.45× |
| ONE | 1 | 1 | fp32 | 1.0 | - | 0.0445 | 0.0975 | 7.59× | 2.19× |
| SINGLE_IGNORED | 17 | 3 | fp32 | 0.125 | 999 | 0.0414 | 0.0917 | 2.75× | 2.21× |

**S2 汇总：min 1.48× / 中位 2.14× / max 8.89×（21 个用例）。**

#### S3 `ecr_token_routing_statistics`

| 用例 | S | E | 源精度 | top_p | 分段 | aclnn p50 (ms) | golden p50 (ms) | golden 轮间极差 | 加速 |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 4096 | 48 | fp32 | 0.125 | - | 0.0569 | 0.2513 | 4.96× | 4.42× |
| F2 | 4096 | 48 | fp16 | 0.125 | - | 0.0516 | 0.2312 | 4.87× | 4.48× |
| F3 | 4096 | 48 | bf16 | 0.125 | - | 0.0398 | 0.1028 | 10.78× | 2.58× |
| F4 | 4096 | 48 | fp32 | 0.125 | 1024×4 | 0.0419 | 0.2856 | 4.10× | 6.82× |
| F5 | 4096 | 48 | fp32 | 0.125 | 512,1024,1536,1024 | 0.0412 | 0.1248 | 7.92× | 3.03× |
| F6 | 40960 | 48 | bf16 | 0.125 | - | 0.0487 | 0.1049 | 2.33× | 2.15× |
| F7 | 4096 | 48 | fp32 | 0.125 | - | 0.0405 | 0.1023 | 6.69× | 2.53× |
| F8 | 4096 | 48 | fp32 | 1.0 | - | 0.0412 | 0.1021 | 5.19× | 2.48× |
| F9 | 4097 | 48 | fp32 | 0.125 | - | 0.0441 | 0.1078 | 2.78× | 2.44× |
| F10 | 4096 | 33 | fp32 | 0.125 | - | 0.0436 | 0.1123 | 4.40× | 2.57× |
| L1 | 65537 | 33 | fp32 | 0.125 | - | 0.0467 | 0.1066 | 8.57× | 2.28× |
| L2 | 131072 | 48 | bf16 | 0.125 | - | 0.0614 | 0.1223 | 3.88× | 1.99× |
| L3 | 131073 | 33 | fp16 | 0.125 | 65537,65536 | 0.0612 | 0.1203 | 3.86× | 1.97× |
| L4 | 40961 | 33 | fp32 | 1.0 | - | 0.0439 | 0.0984 | 1.48× | 2.24× |
| ZERO | 17 | 3 | fp32 | 0.125 | - | 0.0400 | 0.0998 | 9.86× | 2.49× |
| TIES | 33 | 3 | fp32 | 0.125 | - | 0.0432 | 0.1233 | 8.61× | 2.86× |
| EMPTY | 0 | 3 | fp32 | 0.125 | - | 0.0249 | 0.0267 | 1.37× | 1.07× |
| EMPTY_MIX | 17 | 3 | fp32 | 0.125 | 0,8,0,9 | 0.0418 | 0.1139 | 6.79× | 2.72× |
| TINY | 17 | 3 | fp32 | 1e-15 | 8,9 | 0.0400 | 0.1090 | 10.08× | 2.73× |
| ONE | 1 | 1 | fp32 | 1.0 | - | 0.0420 | 0.1025 | 7.87× | 2.44× |
| SINGLE_IGNORED | 17 | 3 | fp32 | 0.125 | 999 | 0.0397 | 0.1036 | 5.02× | 2.61× |

**S3 汇总：min 1.07× / 中位 2.53× / max 6.82×（21 个用例）。**

#### S4 `ecr_routing_normalize_compact`

| 用例 | S | E | 源精度 | top_p | 分段 | aclnn p50 (ms) | golden p50 (ms) | golden 轮间极差 | 加速 |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 4096 | 48 | fp32 | 0.125 | - | 0.0557 | 0.2470 | 3.99× | 4.43× |
| F2 | 4096 | 48 | fp16 | 0.125 | - | 0.0421 | 0.2484 | 4.76× | 5.90× |
| F3 | 4096 | 48 | bf16 | 0.125 | - | 0.0427 | 0.2295 | 5.82× | 5.37× |
| F4 | 4096 | 48 | fp32 | 0.125 | 1024×4 | 0.0506 | 0.2050 | 6.14× | 4.05× |
| F5 | 4096 | 48 | fp32 | 0.125 | 512,1024,1536,1024 | 0.0435 | 0.1472 | 4.76× | 3.38× |
| F6 | 40960 | 48 | bf16 | 0.125 | - | 0.1133 | 0.2239 | 5.85× | 1.98× |
| F7 | 4096 | 48 | fp32 | 0.125 | - | 0.0426 | 0.1241 | 3.17× | 2.91× |
| F8 | 4096 | 48 | fp32 | 1.0 | - | 0.0429 | 0.1263 | 1.46× | 2.94× |
| F9 | 4097 | 48 | fp32 | 0.125 | - | 0.0455 | 0.1244 | 1.85× | 2.73× |
| F10 | 4096 | 33 | fp32 | 0.125 | - | 0.0457 | 0.1249 | 1.37× | 2.73× |
| L1 | 65537 | 33 | fp32 | 0.125 | - | 0.1791 | 0.3483 | 1.88× | 1.94× |
| L2 | 131072 | 48 | bf16 | 0.125 | - | 0.3837 | 0.7500 | 1.01× | 1.95× |
| L3 | 131073 | 33 | fp16 | 0.125 | 65537,65536 | 0.3646 | 0.5441 | 1.20× | 1.49× |
| L4 | 40961 | 33 | fp32 | 1.0 | - | 0.3312 | 0.6655 | 1.00× | 2.01× |
| ZERO | 17 | 3 | fp32 | 0.125 | - | 0.0425 | 0.1244 | 4.25× | 2.93× |
| TIES | 33 | 3 | fp32 | 0.125 | - | 0.0446 | 0.1244 | 4.35× | 2.79× |
| EMPTY | 0 | 3 | fp32 | 0.125 | - | 0.0251 | 0.0028 | 13.72× | 0.11× † |
| EMPTY_MIX | 17 | 3 | fp32 | 0.125 | 0,8,0,9 | 0.0445 | 0.1398 | 1.84× | 3.14× |
| TINY | 17 | 3 | fp32 | 1e-15 | 8,9 | 0.0441 | 0.1347 | 8.47× | 3.05× |
| ONE | 1 | 1 | fp32 | 1.0 | - | 0.0440 | 0.1249 | 3.16× | 2.84× |
| SINGLE_IGNORED | 17 | 3 | fp32 | 0.125 | 999 | 0.0419 | 0.1353 | 1.52× | 3.23× |

**S4 汇总：min 0.11×（†）/ 中位 2.91× / max 5.90×（21 个用例）。**

† `EMPTY`（S=0）是唯一 aclnn 慢于上游的用例：上游对 S=0 直接返回，不建输出、不下发 kernel（2.8 µs）；aclnn 仍要走 `GetWorkspaceSize` + launch（25 µs）。这是退化输入下的固定开销，不代表常规路径，不计入结论。

### 6.3 结论与解读

- 按用例中位看，aclnn 形态在三个算子上都稳定快于上游直调形态：**S2 2.14×、S3 2.53×、S4 2.91×**。
- **序列越长，优势越小**：S2/S4 的最长用例 `L3`（S=131073，fp16，两段）只有 1.48×/1.49×，`L2`（S=131072，bf16）1.59×/1.95×；S3 的 `L3` 为 1.97×。S4 的 `L2/L3/L4` 三个长用例最明显——aclnn 侧本身就要 0.33–0.38 ms，固定路径开销已可忽略。
- **短序列、B>1 的用例优势最大**：S2 `F4` 8.89×、S3 `F4` 6.82×、S3 `F1/F2` 4.4–4.5×。上游按段串行下发、每段都带一次 host 侧的 tiling 同步拷贝，段数越多总开销越大；aclnn 侧整批只有一次 launch。其中 `F4`（4 段等长 1024）在上游侧尤其慢（S2 0.389 ms，是 `F1` 单样本的 2.4 倍），而同样 4 段的 `F5`（512/1024/1536/1024）只有 0.102 ms，说明上游开销还依赖分段的具体形态；aclnn 侧两者都在 0.044 ms，对这种差异不敏感。该差异的具体机制本轮未继续定位。
- `F1`–`F10`（常规形状）30 个组合落在 **1.66×–8.89×**，中位 **2.73×**（去掉 B=4 的 `F4` 后为 1.66×–5.90×）；结果与 dtype（fp32/fp16/bf16）、`top_p`（0.125 / 1.0 / 1e-15）、数据构造（全 0、全平局、含空段、单样本超长）都无关——说明优势是路径级的、对业务形态不敏感。
- 上游 golden 的轮间极差在 47/63 个「用例 × 算子」组合上超过 2×，最高 13.7×，这与 §4 问题 12 同源（同步 H2D + 共享机器）。所以这类数据**必须在多轮里取噪声下界**，单轮数字不可引用。
- **差距来源是调用路径而不是算法**：两者 kernel 来自同一份 `vf.h`，算术完全相同。上游扩展每次调用都新建全部输出张量，并把 tiling 结构体用 `aclrtMemcpy(HOST_TO_DEVICE)` **同步**拷到设备，这会阻塞主机线程且对主机负载敏感；aclnn 路径的 tiling 在 host 侧算好后随 workspace/launch 一起下发，没有这次同步拷贝。这也解释了为什么 S 越大优势越小：kernel 时间占比升高，固定路径开销被摊薄。
- 本表衡量的是**端到端单次调用延迟**，不是纯 kernel 时间；若只看纯 kernel，两者应基本一致。

## 7. Python 调用方式

### 7.1 调用层次

| 层 | 入口 | 适用场景 |
|---|---|---|
| CANN aclnn C 接口 | `aclnnEcrExpertSelectionThreshold` / `aclnnEcrTokenRoutingStatistics` / `aclnnEcrRoutingNormalizeCompact` | C++ 侧直接集成，需自己调 `GetWorkspaceSize` + 两次 launch |
| torch 扩展（C++ 绑定） | `custom_ops_lib.ecr_expert_selection_threshold(prob, seq_lengths, top_p)` 等 | **部署推荐**：pip 安装 wheel 后即可 `import custom_ops_lib` |
| Python 包装 | `custom_ops.ecr_expert_selection_threshold_npu(...)` 等 | 源码树内使用；负责把 `seq_lengths` 的 Tensor 形式转成 list |

注意：`test/python_test/setup.py` 只编译扩展 `custom_ops_lib`（`packages=find_packages()` 为空），**`custom_ops.py` 不会打进 wheel**。外部工程要么直接调 `custom_ops_lib`，要么把 `custom_ops.py` 一起拷过去。

### 7.2 前置条件

1. 安装本仓库产出的 run 包，vendor `custom_xllm_math` 就位（否则 aclnn 接口找不到）。
2. `torch` + `torch_npu` 可用，NPU 设备可见（示例用 `npu:0`，可用前 `torch.npu.set_device(0)`）。
3. 编译并安装 Python 绑定：

```bash
cd test/python_test
bash build_and_run.sh          # 产物：dist/custom_ops-1.0-*.whl
pip3 install --force-reinstall dist/custom_ops-1.0-*.whl
```

4. 输入张量：`prob` 必须是 **NPU 上、fp32、连续（contiguous）、二维 `[E, S]`**。绑定层不做 dtype 转换、不做 `.contiguous()`，传 CPU 张量或转置视图会出错或读到错位数据。

### 7.3 三个接口

```python
import torch, torch_npu          # noqa: F401
import custom_ops_lib as ops     # 或 import custom_ops（源码树内）

# S2 阈值检测
tau_val, tau_idx, n_gt = ops.ecr_expert_selection_threshold(
    prob,                 # [E,S] fp32，NPU，连续
    seq_lengths,          # list[int]：[] / [S] / B 段长度（B ≤ 32，和 == S）
    top_p,                # float，0 < top_p ≤ 1
)
# tau_val [E,B] fp32、tau_idx [E,B] int32、n_gt [E,B] int32

# S3 逐 token 统计（消费 S2 的输出）
sums, cnts = ops.ecr_token_routing_statistics(
    prob, tau_val, tau_idx,
    seq_lengths,          # 与 S2 相同
)
# sums [S] fp32、cnts [S] fp32

# S4 归一化 + 压缩输出（消费 S2/S3 的输出）
ids, weights = ops.ecr_routing_normalize_compact(
    prob, tau_val, tau_idx, sums, cnts,
    seq_lengths,          # 与 S2 相同
    top_p,                # 与 S2 取值一致
    eps,                  # 通常 1e-9
)
# ids [E,C] int32、weights [E,C] fp32，未使用的槽位为 -1 / 0
```

源码树内还有一层更薄的 Python 包装，签名一致，区别只是 `seq_lengths` 也接受 `torch.Tensor`（内部 `.cpu().tolist()`）：

```python
import custom_ops
tau_val, tau_idx, n_gt = custom_ops.ecr_expert_selection_threshold_npu(prob, [], 0.5)
```

约束（绑定层 `TORCH_CHECK`，越界直接抛 `RuntimeError`）：

| 项 | 限制 |
|---|---|
| `prob` | rank 2，`E ∈ [1, 64]`，fp32，NPU，连续 |
| `seq_lengths` | 元素个数 ≤ 32；每个 ≥ 0；**和必须等于 `S`**；S3 另外限制单段 ≤ 2097152 |
| 单样本写法 | `[]`、`()`、`[S]` 等价——长度列表为空或只有 1 项时，按「一个覆盖全部 S 的全局样本」处理（若只给 1 项，其数值被忽略） |
| `top_p` | 原样传给 aclnn，内部按 `float32` 取整；`0 < top_p ≤ 1` |
| S4 输出容量 | `C = Σ_samples min(len, max(1, ceil(len × float32(top_p) − 1e-12)))`，空样本贡献 0；`S=0` 时返回 `[E,0]` 而不下发 kernel |

调用是**异步**的（只做 launch，不在调用点同步）；在主机侧读结果前需要 `torch.npu.synchronize()` 或 `.cpu()`。

### 7.4 示例一：最小可运行

```python
import torch
import torch_npu                      # noqa: F401 必须 import 才能用 npu
import custom_ops_lib as ops

torch.npu.set_device(0)
E, S, top_p = 4, 8, 0.5

prob = torch.rand(E, S, dtype=torch.float32, device="npu:0").contiguous()

tau_val, tau_idx, n_gt = ops.ecr_expert_selection_threshold(prob, [], top_p)
sums, cnts = ops.ecr_token_routing_statistics(prob, tau_val, tau_idx, [])
ids, weights = ops.ecr_routing_normalize_compact(
    prob, tau_val, tau_idx, sums, cnts, [], top_p, 1e-9)

print(tau_val.shape, tau_val.dtype)   # torch.Size([4, 1]) float32
print(tau_idx.dtype, n_gt.dtype)      # int32 int32
print(ids.shape, ids.dtype)           # torch.Size([4, 4]) int32
print(weights.shape, weights.dtype)   # torch.Size([4, 4]) float32
print(ids.cpu())
```

`top_p=0.5`、`S=8` 时每个专家选 `ceil(8×0.5)=4` 个 token，因此 `C=4`；`ids[e]` 是这 4 个 token 的**全局下标，按 token 顺序升序**。

### 7.5 示例二：多样本分段（含空段）

```python
E, top_p = 3, 0.5
lengths = [0, 8, 0, 9]                # 4 段，S = 17，其中第 0、2 段为空
prob = torch.rand(E, sum(lengths), dtype=torch.float32, device="npu:0")

tau_val, tau_idx, n_gt = ops.ecr_expert_selection_threshold(prob, lengths, top_p)
# B = 4；空样本列固定为 tau_val=2.0、tau_idx=-1、n_gt=0
assert torch.all(tau_val[:, [0, 2]] == 2.0)
assert torch.all(tau_idx[:, [0, 2]] == -1)

sums, cnts = ops.ecr_token_routing_statistics(prob, tau_val, tau_idx, lengths)
ids, weights = ops.ecr_routing_normalize_compact(
    prob, tau_val, tau_idx, sums, cnts, lengths, top_p, 1e-9)

# C = 0 + ceil(8×0.5)=4 + 0 + ceil(9×0.5)=5 = 9
print(ids.shape)                      # torch.Size([3, 9])
print(ids[0, :4].tolist())            # 第 1 段的 4 个下标，落在 [0, 8)
print(ids[0, 4:].tolist())            # 第 3 段的 5 个下标，落在 [8, 17)
```

### 7.6 示例三：接到已有 routing scoring 上

如果已有 `logits`（`[S, E]`），只需转置并保证连续；三段按 S2 → S3 → S4 顺序串起来：

```python
def ecr_routing(logits, seq_lengths=(), top_p=0.125, eps=1e-9):
    """logits: [S,E] fp32 on NPU -> (ids [E,C], weights [E,C])."""
    prob = logits.t().contiguous()     # [E,S]
    tau_val, tau_idx, _ = ops.ecr_expert_selection_threshold(prob, seq_lengths, top_p)
    sums, cnts = ops.ecr_token_routing_statistics(prob, tau_val, tau_idx, seq_lengths)
    return ops.ecr_routing_normalize_compact(
        prob, tau_val, tau_idx, sums, cnts, seq_lengths, top_p, eps)

ids, weights = ecr_routing(torch.rand(8192, 48, device="npu:0"), top_p=0.125)
print(ids.shape)                       # [48, 1024]
```

三段之间的 `tau_val/tau_idx/sums/cnts` 必须来自**同一份** `prob` 和**同一个** `top_p`；不要跨 batch 复用。

### 7.7 语义约定

| 项 | 约定 |
|---|---|
| 每段选几个 | `k = min(len, max(1, ceil(len × top_p)))`，空段 `k=0`（非空段至少选 1 个，哪怕 `top_p` 极小） |
| `tau_idx` | **sample-local** 下标（段内偏移），不是全局 token 下标 |
| 平局 | 取更小下标（stable sort，等价于「值相等时选前面的 token」） |
| `n_gt` | 严格大于 `tau_val` 的 token 个数 |
| S3 口径 | 某 token 被某专家选中 ⟺ `prob > tau_val` 或（`prob == tau_val` 且段内下标 ≤ `tau_idx`）；`sums[token]` 为选中它的概率之和，`cnts[token]` 为选中它的专家数 |
| S4 `ids` / `weights` | `ids` 是**全局** token 下标，按 token 顺序排列；`weights = prob / max(sums, eps) × sqrt(cnts)`；未使用槽位 `ids=-1`、`weights=0` |
| `S=0` | S2 返回 `[E,B]` 的哨兵，S3/S4 返回空张量；不报错 |

### 7.8 常见报错对照

| 报错 | 原因 |
|---|---|
| `seq_lengths must sum to prob.size(1)=S, got X` | 分段长度之和与 `prob.size(1)` 不一致（注意空段要显式写 0） |
| `prob.size(0) must be in [1,64]` | 专家数 `E` 越界 |
| `seq_lengths supports at most 32 samples` | 段数超过 32 |
| `tau_val must be [E,B]=[e,b]` | S3 收到的 `tau_val/tau_idx` 是别的 `prob`/分段算出来的 |
| `CheckNnopbaseAddIntArrayAttr` 相关 | 旧版本绑定把空 `seq_lengths` 当 int 数组属性下发；当前绑定已改写为 `[S]`，若仍出现说明用的是旧 `.so` |
| 结果全错但没有异常 | `prob` 非连续（例如直接传 `logits.T`），绑定不做隐式拷贝——加 `.contiguous()` |

## 8. 复现步骤

```bash
# 0) 进入容器
docker exec -it docker_wjh_001 bash -lc 'cd /home/w00501444/LanguageTest/xllm-ops && bash'

# 1) 编译并安装三个算子到 CANN（容器内）
bash build.sh --no-build-transformer --no-build-flash \
  -n "ecr_expert_selection_threshold;ecr_token_routing_statistics;ecr_routing_normalize_compact" \
  -c ascend950
# 产物：xllm_ops/output/_cann_ops_custom，安装后 vendor 目录
#   /usr/local/Ascend/cann-9.2.1/opp/vendors/custom_xllm_math

# 2) 编译并安装 Python 绑定
cd test/python_test && bash build_and_run.sh

# 3) 功能验证（只用前 4 卡）
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=<pytest 目录> python3 -m pytest -q \
  test_ecr_expert_selection_threshold.py \
  test_ecr_token_routing_statistics.py \
  test_ecr_routing_normalize_compact.py

# 4) 性能对比（需先按上游工程编出三个 libecr_*_ops.so）
#    21 个用例 × 3 算子 × 2 后端逐条跑，逐用例先校验输出再计时
export ASCEND_RT_VISIBLE_DEVICES=3
python3 -u bench_ecr_routing.py --warmup 20 --iters 100 --rounds 6 \
  --csv bench_matrix.csv \
  --upstream-s2 <...>/libecr_expert_selection_threshold_ops.so \
  --upstream-s3 <...>/libecr_token_routing_statistics_ops.so \
  --upstream-s4 <...>/libecr_routing_normalize_compact_ops.so
# 建议跑两遍（12 轮），aclnn 取轮间中位、golden 取轮间最优，理由见 6.1
```

注意：修改 `op_kernel/` 下的代码后，重新编译前要删除 `build/binary/ascend950/src` 与 `gen`，否则会继续用旧的源码副本。

## 9. 交付物与改动清单

新增（未跟踪）：

- `xllm_ops/expert_chosen_routing2/ecr_expert_selection_threshold/`（9 个文件）
- `xllm_ops/expert_chosen_routing2/ecr_token_routing_statistics/`（9 个文件）
- `xllm_ops/expert_chosen_routing2/ecr_routing_normalize_compact/`（9 个文件）
- `test/python_test/test_ecr_expert_selection_threshold.py`
- `test/python_test/test_ecr_token_routing_statistics.py`
- `test/python_test/test_ecr_routing_normalize_compact.py`
- `test/python_test/bench_ecr_routing.py`

修改：

- `cmake/func.cmake` — 增加分组目录 glob；ops-info 依赖按组是否为空条件化（支持 `-n` 只编部分算子）
- `test/python_test/RegisterOps.cpp` — S2/S3/S4 的 aclnn 绑定；S4 空 `seq_lengths` 改写为 `[S]`
- `test/python_test/custom_ops.py` — 三个算子的 Python 包装
- `xllm_ops/build_aclnn.sh` — 把三个算子加入 ascend950 的默认算子清单

上述新增目录内的 `*_contract.h` 与 `ecr_expert_selection_threshold_vf.h` 已包含第 4 节问题 5、7 的修正（依赖模板消歧、`MaskReg` 左值），无需在本文件之外再改。

性能原始数据（未入库，留存在调测机）：`/home/w00501444/LanguageTest/bench_matrix_block6{,b}.{csv,log}`，两次运行共 1512 行 `(case, op, backend, round, mean/p50/p90/best)` 记录；§6 的表格全部由这两份 CSV 聚合得到。

## 10. 已知限制与后续建议

1. 本次只在 Ascend950PR + CANN 9.2.1 上验证；其它 SOC/CANN 版本需重新确认 `reg::` API 与 `StoreAlign` 等签名差异。
2. 算子仅支持 fp32 输入；半精度支持需要在内核里补类型分支，本轮范围外。
3. 性能数据取自共享机器，上游扩展含同步 H2D 拷贝，单轮数字不可引用：同一用例 12 轮 p50 的极差可达 13.7×。本轮用"aclnn 取中位、golden 取最优轮"得到加速比下界；若需要纯 kernel 对比，建议后续用 `msprof` 采集单算子 kernel 时间。
4. 两套实现（aclnn 分配器 / torch 缓存分配器）**不能逐次交错下发**，否则长时间运行会出 AI Core 非法 GM 访问（§4 问题 13）。benchmark 采用块级交替；若后续把两套实现合进同一个线上进程，需确认内存分配器统一或显式隔离。
5. 该机器没有 pytest，当前依赖外部目录 `PYTHONPATH` 注入；建议后续在镜像里固化测试依赖，或把用例改成可直接 `python3` 运行的脚本。
