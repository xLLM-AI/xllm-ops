# DSV4.1 HC pre V3 (`hc_pre_va`)

Ported from vllm-ascend `fbb4d867d22095047473e3c7ca9db053f5f4c7b0`:

- Kernel and tiling: `csrc/moe/hc_pre/`.
- Four-output wrapper: `npu_hc_pre_v3_npu` in `csrc/torch_binding.cpp`.
- CPU reference and accuracy tests: `test_npu_hc_pre.py`.

`hc_pre_v3` is the vLLM wrapper name. It invokes the `HcPre` device operator
with optional external `pre_mix` and returns `(y, post, comb_frag, pre)`.
This port renames the device operator to `HcPreVa` / `hc_pre_va`, without
introducing another versioned operator name. Both implicit and external mix
paths return the native FP32 `pre`; no BF16 recomputation is used.

Inputs: BF16 hidden `[..., 4, D]`, FP32 projection `[24, 4*D]`, scale `[3]`,
base `[24]`, optional FP32 `pre_mix` `[..., 4]`. The Python scalar argument
order is `(hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)`; ACLNN takes the
two epsilon arguments in the opposite order, matching the source wrapper.
Outputs: BF16 collapsed `y`, FP32 `post`, `comb_frag`, and `pre`.

The source includes Ascend910B/910_93 and Ascend950 implementations. Runtime
validation in this workspace uses Ascend910_93; it does not establish A5
runtime accuracy. The legacy three-output test cases are migrated through
the V3 interface with `pre_mix=None` and also check the fourth output.

Build the operator with `bash build.sh -n hc_pre_va -c ascend910_93` and
build the test extension with `cd test/python_test && python setup.py
build_ext --inplace`. Configure the generated OPP package before executing
`pytest -q test_hc_pre_va.py`.
