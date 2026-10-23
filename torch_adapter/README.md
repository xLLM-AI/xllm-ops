# MX Torch Adapters

The consumer includes `cmake/mxfp8.cmake` and adds `torch_adapter/` to its
`xllm_ops` host target. The consumer supplies the Ascend950 platform guard and
`core/kernels/npu/aclnn/pytorch_npu_helper.hpp`.

The supplied ACLNN bridge must map `Float8_e8m0fnu` to ACL dtype 37
(`ACL_FLOAT8_E8M0`) for both tensors and scalar-type arguments. Apply
`patches/xllm-e8m0-bridge.patch` at the xLLM root if that bridge still uses the
old positional scalar table. The patch targets xLLM integration commit
`525de6fa`; it must accompany this adapter update in the parent repository.
The Python test bridge includes the same explicit mappings and rejects
unmapped types instead of defaulting them to ACL_FLOAT.

`mx_linear.cpp` is always attached. Its FP4 and FP8 GEMMs require only
`XLLM_HAS_NPU_MX_MATMUL`; the latter also serves FP4's expanded prefill cache.
The FP4 layout helpers use generic Torch-NPU format APIs. Dynamic quant and
the W4 SwiGLU call use their individual probes in partial installations.
RMSNorm and grouped SwiGLU remain behind the full MXFP8 capability flag.
The grouped MX wrapper selects `aclnnGroupedMatmulSwigluQuantWeightNzV2` when
the prepacked weight is `FRACTAL_NZ`; the generic V2 entry rejects that
storage format during workspace validation. Both V2 headers and entry points
are required by the full capability probe.

## Regression Harness

`test/torch_adapter/` builds a standalone module against the actual Torch,
Torch-NPU, CANN and consumer headers. Configure it with
`XLLM_SOURCE_DIR`, `DEPENDENCY_INCLUDE_DIR` and, optionally,
`XLLM_BRIDGE_INCLUDE_DIR` for a patched consumer-header overlay.
Set the same Torch/CANN include environment variables as the xLLM build.

Build three configurations in separate directories:

- Full capability detection.
- `-DXLLM_TORCH_NPU_CANN_HAS_MXFP8=FALSE`, retaining individual op probes.
- The preceding option plus all three individual MX probes set to FALSE.

Run `run_probe.py <build>/mx_adapter_probe.so --device 0` with an explicitly
selected idle, healthy Ascend950. Add `--grouped` for the full build.
The script checks actual ACL tensor descriptor types, numerical W4 matmul,
FP8 prefill expansion/cache matmul, NZ prepack and fused SwiGLU against CPU
references. With `--grouped` it exercises grouped MX twice, once with ND
weights (generic entry) and once with `FRACTAL_NZ` weights (WeightNz entry).
The disabled build's `--disabled` case must reject the GEMM.
