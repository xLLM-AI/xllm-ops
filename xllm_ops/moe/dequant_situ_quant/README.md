# DequantSituQuant

A3-only fusion of dequantization, Kimi K3 SiTU activation, and per-row dynamic
INT8 quantization. This operator has its own OpDef and ACLNN name; it does not
replace the existing DequantSwigluQuant operator.

For each row, split the input into gate and up halves, selected by
`activate_left`:

```text
gate_out = beta * tanh(gate / beta) * sigmoid(gate)
up_out = linear_beta * tanh(up / linear_beta)   # when linear_beta > 0
activation = gate_out * up_out
```

The input is INT32 `[rows, 2H]` with FP32 weight and activation scales, or
BF16 `[rows, 2H]` with no dequantization scales. Both paths use `dynamic`
quantization and return INT8 `[rows, H]` plus FP32 `[rows]` scales.

INT32 input optionally supports grouped expert weight scales, selected by
INT64 row counts in `group_index`. Empty expert groups have a count of zero.
The existing input checks in op_host
validate scale shapes and optional inputs.

Defaults: `beta=4.0`, `linear_beta=25.0`, `activate_left=true`,
`quant_mode="dynamic"`.

The main xllm repository owns the PyTorch wrapper. Build this operator through
the A3 list in `xllm_ops/build_aclnn.sh` or the main repository's
`python setup.py build`.
