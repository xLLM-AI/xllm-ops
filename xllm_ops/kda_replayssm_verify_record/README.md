# KDA ReplaySSM verify record

The Ascend 910_93 operator computes the verification output and writes
FP32 normalized keys, corrections, and decays to the selected record slots.
The slot's recurrent state is read-only during verification.

Query, key, value, and gate are BF16 `[sequences * 8, 8, 128]`.
Query, key, and value may be dense or views into packed QKV storage. Pass
their shared token stride, in elements, as `qkv_token_stride` (1024 or
3072). The attribute distinguishes the two layouts in the ACLNN tiling cache.

Beta is BF16 `[sequences * 8, 8]`. Supply FP32 `a_log` with shape `[8]`,
FP32 `dt_bias` with shape `[1024]`, INT32 slot indices with shape
`[sequences]`, and a positive query scale.

The state is FP32 `[slots, 8, 128, 128]`. The three record outputs are
FP32 `[slots, 8, 8, 128]`, and the verification output is BF16
`[sequences * 8, 8, 128]`. Invalid slot indices produce zero verification
output and leave the records untouched.
