# aclnnDequantSituQuant

Supported target: Atlas A3 (`ascend910_93`).

The generated ACLNN declaration is installed as
`op_api/include/aclnnop/aclnn_dequant_situ_quant.h`.

```cpp
aclnnStatus aclnnDequantSituQuantGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weightScaleOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* biasOptional,
    const aclTensor* quantScaleOptional,
    const aclTensor* quantOffsetOptional,
    const aclTensor* groupIndexOptional,
    double beta,
    double linearBeta,
    bool activateLeft,
    char* quantModeOptional,
    const aclTensor* y,
    const aclTensor* scale,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnDequantSituQuant(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

Use the installed generated header as the authoritative declaration. INT32
input requires weight and per-row activation scales. BF16 input omits both.
The output uses INT8 quantization and FP32 per-row scales. See the operator
README and op_host validation for supported optional input combinations.
