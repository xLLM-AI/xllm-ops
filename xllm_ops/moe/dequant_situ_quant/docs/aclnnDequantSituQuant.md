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

## Width and routing constraints

INT32 and BF16 input width must be a positive multiple of 16 (the gate/up
width is a multiple of 8 FP32 elements for 32-byte vector alignment).
For grouped INT32 input, every count must be nonnegative and their sum must
equal the input row count. Invalid device-side counts fail kernel execution;
they are not clipped or treated as valid output. Zero-count groups are valid.
Rows are partitioned by global row positions, including sparse one-row groups.
Optional INT32 bias is streamed one gate/up half at a time so TP1 width 12288
fits A3's 192-KiB UB without changing per-row quantization.
