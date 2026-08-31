# lightning_indexer_v2算子测试框架

## 功能说明

基于pytest测试框架，实现lightning_indexer_v2算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能
- **双模式执行隔离**：支持直接pytest多进程执行和shell层进程隔离两种批量模式
- **batch_exec 模式**：根据 Excel 表格筛选已有 pt 文件直接执行 NPU 测试（跳过 pt 生成步骤）
- **性能采集**：支持挂载msprof采集算子性能数据并汇总输出
- **运行模式切换**：支持eager直接调用和graph（torch.compile + torchair）两种算子调用模式

## 当前实现范围

### 参数限制

- **数据格式**:
  - **query_layout**：BSND、TND
  - **key_layout**: PA_BBND

- **运行模式**:
   - **eager**：直接调用 `torch.ops.cann_ops_transformer.lightning_indexer`
   - **graph**：通过 `torch.compile` + `torchair` 后端编译执行（需torchair支持）

### 环境配置

#### 前置要求

1、 确认TorchNPU为最新版本  
2、 参考Attention融合算子Experimental使用说明激活CANN包和自定义算子包

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run_sh                                  # 执行脚本，支持single/batch/batch_exec三种命令
- batch_isolated_run.sh                        # 批量隔离执行脚本（shell层进程隔离+msprof性能采集）
- lightning_indexer_v2_golden.py               # cpu侧算子golden实现
- lightning_indexer_v2_acl_graph.py            # graph模式torchair后端实现
- result_compare_method.py                     # cpu golden与npu输出精度对比
- collect_perf_data.py                         # msprof性能数据收集与汇总
- pytest.ini                                   # 创建测试标记

单用例测试：

- test_lightning_indexer_v2_single.py       # pytest测试单用例运行主程序 
- test_lightning_indexer_v2_paramset.py     # 单用例入参配置

批量测试：

- test_lightning_indexer_v2_batch.py        # 用例批量测试主程序并生成excel文件保存结果
- ./batch/lightning_indexer_v2_pt_loadprocess.py    # 读取pt文件并调用算子获取npu输出
- ./batch/lightning_indexer_v2_pt_save.py           # 读取excel表格批量生成用例pt文件
- ./batch/list_pt_from_excel.py                     # 从Excel提取Testcase_Name并按名匹配pt文件（batch_exec模式用）
- ./batch/replace_path.py                           # test_lightning_indexer_v2_batch.py占位符替换
 
## 架构说明

- **single 模式**：`liv2_output_acl_graph` 调用 `liv2_output_single(is_batch=True)` 即时生成数据 → `torch.compile` + `torchair` 执行
- **batch 模式**：从 Excel 生成 .pt 文件 → 读取 pt 执行 NPU 测试和精度对比 → 生成 result.xlsx
- **batch_exec 模式**：从 Excel 筛选已有 .pt 文件 → 跳过 pt 生成 → 直接执行 NPU 测试和精度对比 → 生成 result.xlsx
- 两路共用 `_liv2_prepare_tensors_and_metadata` 和 `_liv2_run_compiled_graph`，统一使用 `fullgraph=False`

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_lightning_indexer_v2_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### 用例的批量生成与测试

##### 方式A：test_run.sh 批量执行

1、excel路径下存放用例excel表格

2、修改test_run.sh配置区中的DEFAULT_EXCEL（Excel表格路径）和DEFAULT_PT_PATH（pt文件存放路径），或通过命令行参数指定

3、执行指令：

``` bash
bash test_run.sh batch                                    # 使用配置区默认值（eager 模式）
bash test_run.sh batch -E ./excel/test_cases.xlsx        # 指定 Excel 路径
bash test_run.sh batch -S Sheet1 -P ./pt_path            # 指定 Sheet 和 pt 目录
bash test_run.sh batch -M graph                          # graph 模式
bash test_run.sh batch -E ./excel/my.xlsx -S Sheet2 -M graph  # 组合使用
bash test_run.sh batch graph                             # 兼容旧用法：直接跟 run_mode
```

4、配置区默认值：

| 变量 | 默认值 | 命令行参数 | 说明 |
|---|---|---|---|
| DEFAULT_EXCEL | `./excel/test_cases.xlsx` | `-E` | Excel 用例表格路径（**必须指定具体文件名**，不支持通配符如 `./excel/*`） |
| DEFAULT_PT_PATH | `./pt_path` | `-P` | pt 文件存放目录 |
| （无） | `Sheet1` | `-S` | Excel Sheet 页名 |
| （无） | `eager` | `-M` | 运行模式（eager/graph） |

#### 根据 Excel 表格筛选已有 pt 批量执行（batch_exec 模式）
>
> 仅重新执行 NPU 测试和精度对比，不重新生成 pt 文件。适用于已有 pt 文件、只需更新精度结果的场景。

1、确认 pt_path 下已有对应 .pt 文件（可通过 `batch` 模式或手工生成）

2、执行指令：

``` bash
bash test_run.sh batch_exec                                    # 默认: excel/test_cases.xlsx, Sheet1, pt_path
bash test_run.sh batch_exec -E ./excel/test_cases.xlsx         # 指定 Excel 路径
bash test_run.sh batch_exec -S Sheet1 -P ./pt_path             # 指定 Sheet 和 pt 目录
bash test_run.sh batch_exec -E ./excel/test_cases.xlsx -M graph  # graph 模式
```

3、执行流程：

- 从 Excel 表格读取 `Testcase_Name` 列
- 按 `<Testcase_Name>.pt` 在 pt_path 下匹配对应的 .pt 文件
- 仅对匹配到的 .pt 文件执行 NPU 测试和精度对比
- 生成 `result.xlsx` 测试结果表格
- 如果 Excel 中某条用例无对应的 .pt 文件，会输出警告并跳过该用例

4、与 `batch` 模式的区别：

|  | `batch` | `batch_exec` |
|---|---|---|
| pt 生成 | 每次重新生成 | 跳过 |
| 执行速度 | 较慢（含 pt 生成） | 较快 |
| 适用场景 | 首次运行 / 参数变更 | 精度复测 / 仅 NPU 结果更新 |

##### 方式B：手工分步执行

1、生成pt文件：

``` bash
python3 batch/lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path
python3 batch/lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path --sheet Sheet1  # 指定 Sheet 页
```

2、替换测试脚本路径：

``` bash
python3 batch/replace_path.py test_lightning_indexer_v2_batch.py pt_path
```

3、执行测试：

``` bash
python3 -m pytest -rA -s test_lightning_indexer_v2_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

4、恢复测试脚本：

``` bash
cp test_lightning_indexer_v2_batch.py.bak test_lightning_indexer_v2_batch.py
```

##### 方式C：批量隔离执行（推荐用于性能采集）

对每条用例单独拉起一个pytest进程，实现进程间完全隔离，避免单条用例崩溃影响其他用例。

``` bash
bash batch_isolated_run.sh ./pt_path 0         # 不采集性能
bash batch_isolated_run.sh ./pt_path 1         # 采集性能（挂载msprof）
bash batch_isolated_run.sh ./pt_path 0 graph   # graph模式 + 不采集性能
bash batch_isolated_run.sh ./pt_path 1 graph   # graph模式 + 性能采集
```

## Excel 用例表格式

`excel/test_cases.xlsx` 需包含以下列（Sheet1）：

| 列名 | 类型 | 示例 |
|---|---|---|
| Testcase_Name | str | `test_case_01` |
| batch_size | int | `8` |
| q_seq | int | `15` |
| k_seq | int | `111` |
| q_t_size | int | `8` |
| k_t_size | int | `15` |
| q_head_num | int | `64` |
| k_head_num | int | `1` |
| head_dim | int | `128` |
| block_size | int | `512` |
| block_num | int | `8` |
| qk_dtype | str | `FLOAT8_E4M3FN` / `INT8` / `HIFLOAT8` |
| cu_seqlens_q | None/str | `None` 或 `"[0, 1]"` |
| cu_seqlens_k | None/str | `None` 或 `"[0, 1]"` |
| seqused_q | None/str | `None` 或 `"[3,3,3,3,3,3,3,3]"` |
| seqused_k | str | `"[28,24,80,96,47,76,0,111]"` |
| cmp_residual_k | None/str | `None` 或 `"[0,0,0,0,0,0,0,0]"`（cmp_ratio>1时必填）|
| max_seqlen_q | int | `-1` |
| layout_q | str | `BSND` / `TND` |
| layout_k | str | `PA_BBND` |
| topk | int | `512` |
| mask_mode | int | `0` / `3` |
| query_datarange | str | `"[-448,448]"` |
| key_datarange | str | `"[-20,20]"` |
| weights_datarange | str | `"[-123,123]"` |
| cmp_ratio | int | `1` / `4` |
| return_value | int | `0` / `1` |
| output_idx_offset | None/str | `None` 或列表字符串 |

**注意事项**：

- `cmp_ratio > 1`且`sparse_mode != 0`时，`cmp_residual_k`必填（长度=batch_size的列表）
   
## 输出文件

| 文件 | 说明 |
|---|---|
| `result.xlsx` | 测试结果（精度、参数等） |
| `result_perf.xlsx` | 测试结果 + 性能数据（仅msprof模式） |
| `batch_summary.log` | 批量执行详细日志 |
| `batch_fail_list.log` | 失败用例清单 |
| `PROF_*/` | msprof性能原始数据目录 |
