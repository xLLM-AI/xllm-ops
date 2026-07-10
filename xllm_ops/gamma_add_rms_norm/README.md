# GammaAddRmsNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。GammaAddRmsNorm算子将RmsNorm前的Add算子融合起来，并通过`addGammaOffset`参数控制是否在kernel内完成Gemma风格的`gamma + 1`缩放，减少搬入搬出操作。
- 计算公式：

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
	  g_i = gamma_i + (addGammaOffset ? 1 : 0)
	  $$

	  $$
	  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>x1</td>
      <td>输入</td>
      <td>用于Add计算的第一个输入，对应公式中的`x1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>用于Add计算的第二个输入，对应公式中的`x2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示RmsNorm的缩放因子（权重），对应公式中的`g`。shape需要与`x1`后几维保持一致，后几维为x1需要norm的维度。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
	    <tr>
	      <td>epsilon</td>
	      <td>可选属性</td>
	      <td><ul><li>添加到分母中的值，以确保数值稳定，用于防止除0错误，值需要大于等于零，对应公式中的eps。</li><li>默认值为1e-6。</li></ul></td>
	      <td>FLOAT</td>
	      <td>-</td>
	    </tr>
	    <tr>
	      <td>addGammaOffset</td>
	      <td>可选属性</td>
	      <td><ul><li>控制是否在kernel内对gamma执行Gemma风格的`gamma + 1`。</li><li>默认值为false。</li></ul></td>
	      <td>BOOL</td>
	      <td>-</td>
	    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示最后的输出，Device侧的aclTensor，对应公式中的`RmsNorm(x)`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td>表示归一化后的标准差，对应公式中的`Rms(x)`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示Add计算的结果，对应公式中的`x`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>：参数`x1`、`x2`、`gamma`、`y`、`x`的数据类型不支持BFLOAT16。

  - Kirin X90/Kirin 9030处理器系列产品：x1、x2、gamma、y和x的数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_gamma_add_rms_norm](examples/test_aclnn_gamma_add_rms_norm.cpp) | 通过[aclnnGammaAddRmsNorm](docs/aclnnGammaAddRmsNorm.md)接口方式调用GammaAddRmsNorm算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/gamma_add_rms_norm_proto.h)构图方式调用GammaAddRmsNorm算子。         |
