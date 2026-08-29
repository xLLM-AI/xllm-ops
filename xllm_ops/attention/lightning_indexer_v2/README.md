# LightningIndexerV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：LightningIndexerV2基于一系列操作得到每一个token对应的top-k个位置。

- 计算公式：

  $$
  Top-k \left\{  \left[ 1 \left] \mathop{{}}\nolimits_{{1 \times \text{ }g}}\text{@} \left[  \left( W\text{@} \left[ 1 \left] \mathop{{}}\nolimits_{{1\text{ } \times \text{ }S\mathop{{}}\nolimits_{{k}}}} \left) \text{ } \odot \text{ }ReLU \left( Q\mathop{{}}\nolimits_{{index}}\text{@}K\mathop{{}}\nolimits_{{T}}^{{index}} \left)  \left]  \right\} \right. \right. \right. \right. \right. \right. \right. \right. \right. \right.
  $$

- 主要计算过程为：

  1. 将某个token对应的输入参数`q`（$Q_{index}\in\R^{g\times d}$）乘以给定上下文`k`（$K_{index}\in\R^{S_{k}\times d}$），得到相关性。
  2. 通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
  3. 将其与权重系数`w`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$sparseIndices$，并输出对应的$sparseValues$，作为Attention的输入。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
    <td>q</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>k</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>w</td>
    <td>输入</td>
    <td>公式中的输入W。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>cuSeqlensQOptional</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中q的有效token数的累加和</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>cuSeqlensKOptional</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中k的有效token数的累加和。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
    <td>sequsedQOptional</td>
    <td>输入</td>
    <td>不同Batch中q的真实使用长度。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
    <td>sequsedKOptional</td>
    <td>输入</td>
    <td>不同Batch中k的真实使用长度。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
    <td>cmpResidualKOptional</td>
    <td>输入</td>
    <td>表示k压缩前token数量除以cmpRatio的余数。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>blockTableOptional</td>
    <td>输入</td>
    <td>表示PageAttention中KV存储使用的block映射表。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
    <td>outputIdxOffsetOptional</td>
    <td>输入</td>
    <td>表示topK结果输出索引所需要加上的偏移。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
  <td>metadataOptional</td>
  <td>输入</td>
  <td>LightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。</td>
  <td>INT32</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>topk</td>
  <td>输入</td>
  <td>topK阶段需要保留的block数量。</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>maxSeqlenQ</td>
  <td>输入</td>
  <td>q的最大序列长度。</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>layoutQ</td>
  <td>输入</td>
  <td>用于标识输入q的数据排布格式。</td>
  <td>STRING</td>
  <td>-</td>
  </tr>
  <tr>
  <td>layoutK</td>
  <td>输入</td>
  <td>用于标识输入k的数据排布格式。</td>
  <td>STRING</td>
  <td>-</td>
  </tr>
  <tr>
  <td>maskMode</td>
  <td>输入</td>
  <td>表示mask的模式。</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>cmpRatio</td>
  <td>输入</td>
  <td>用于稀疏计算，表示k的压缩倍数。</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>returnValue</td>
  <td>输入</td>
  <td>代表是否需要返回Indices对应的Values值。</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>sparseIndices</td>
  <td>输出</td>
  <td>公式中的Indices输出。</td>
  <td>INT32</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>sparseValues</td>
  <td>输出</td>
  <td>公式中的Indices对应的Values输出。</td>
  <td>FLOAT</td>
  <td>ND</td>
  </tr>
  </tbody>
   </table>

  - q、k、w参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
  - 使用S1和S2分别表示q和k的输入样本序列长度，N1和N2分别表示q和k对应的多头数，k表示最后选取的索引个数。参数q中的D和参数k中的D值相等为128。T1和T2分别表示q和k的输入样本序列长度的累加和。

## 约束说明

- 参数q的N支持1~64，k的N支持1。
- headdim支持128。
- pa_kv_cache支持0轴非连续；pa_block_size支持1~1024，满足block大小32B对齐。
- 参数q、k的数据类型应保持一致。
- sparseIndices无效部分填-1；sparseValues无效部分填-inf。
- 传入的cmpResidualKOptional中的每一个元素的值都应小于传入的压缩率cmpRatio。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>:
  - topk取值范围当前仅支持[1, 2048]，以及3072、4096、5120、6144、7168、8192。
  - 当前不支持sequsedQOptional、outputIdxOffsetOptional、maxSeqlenQ功能，不建议传入这些参数。
  - 当传入的参数layoutK为PA_BBND时，必须传入sequsedKOptional；当layoutK不为PA_BBND时，不支持sequsedKOptional功能，不建议传入该参数。
- <term>Ascend 950PR/Ascend 950DT</term>:
  - 当传入的参数layoutQ为BSND时，不支持传入cuSeqlensQOptional；当layoutK为BSND或PA_BBND时，不支持传入cuSeqlensKOptional。
  - 当传入参数outputIdxOffsetOptional时，只支持大于0的索引偏移值；且应满足约束：加上传入的索引偏移值后，得到的sparseIndice值不超过INT32的最大值。
  - 当传入的参数layoutQ为TND时，必须传入cuSeqlensQOptional，如果也传入sequsedQOptional，应保证由sequsedQOptional传入的各个batch的query长度不超过根据cuSeqlensQOptional计算出的各个batch的q序列长度。当某个batch由sequsedQOptional传入的q序列长度seqlen1小于由cuSeqlensQOptional计算出的query长度seqlen2时，会启用TND Padding功能，将该batch的seqlen2与seqlen1差值部分的query输出的sparseIndices和sparseValues全部置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。
  - 当传入的cmpRatio > 1且maskMode = 3时，必须传入cmpResidualKOptional，其余情况不传入。
  - 当layoutK为PA_BBND时，必须传入sequsedKOptional。
  - 参数metadataOptional必须传入，且shape固定为[1024]。

## 调用示例

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_lightning_indexer_v2](./examples/test_aclnn_lightning_indexer_v2.cpp) | 通过[aclnnLightningIndexerV2](./docs/aclnnLightningIndexerV2.md)接口方式调用LightningIndexerV2算子。 |
| PyTorch API | - | 通过[torch.ops.cann_ops_transformer.lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/lightning_indexer.md)接口调用LightningIndexerV2算子。|
