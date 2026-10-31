# Ascend950 MTP Performance Restoration

The A5 MTP path lost two previously validated optimizations: the B5/NK16/NV48
MTP3 dispatch to RegBase key 403, and checkpoint-store overlap with recurrent
gate/input work. This change restores both in the A5 host predicate and arch35
kernel header. A2/A3 kernel files and the public operator ABI are unchanged.

The A5 header also retains the explicit `event_t` declaration required by
CANN 9.1. It carries forward the verified recurrent implementation, including
RegBase vector fences and the final checkpoint wait before deferred norm.

## Validation

- `python3 test/mega_gdn_mtp_decode/check_a5_dispatch.py` compiles the actual
  dispatch expression and checks 2304 combinations, including non-A5,
  non-FLA, non-deferred, other batch sizes and other head geometries.
- On 2026-10-31, isolated CANN 9.1 OPC compilation of the final PR kernel
  sources passed for Ascend950PR_9599 and retained all 27 tiling keys.
- The resulting device object SHA256 is
  `43ef23278dec72c842ed2ee02198374bb8cb93a2260d5000fa03d2c601830c3a`,
  identical to the candidate exercised in the 2026-10-28 runtime experiment.
- That experiment used a fixed xLLM binary, TP1/C5/MTP3, graph decode,
  16384 prompt tokens, 1024 output tokens and private OPPs. Three arms were
  repeated twice with 20 measured requests per run: 120 requests total.

| Configuration | Output tok/s | Mean TPOT | Mean MTP task time |
| --- | ---: | ---: | ---: |
| Key 203, synchronous checkpoints | 182.4525 | 22.5675 ms | 249.074 us |
| Key 403, synchronous checkpoints | 195.4463 | 21.2369 ms | 137.891 us |
| Key 403, restored overlap/vector path | 200.1188 | 20.1623 ms | 96.993 us |

Short/long-context semantic checks, deterministic generation comparisons,
request/token counts and clean exits passed in the historical experiment.
The combined gain was 9.6827% under that contract. This is not a new runtime
measurement of the complete PR package, full C-Eval, A2/A3 acceptance or a
clean full OPP rebuild. The 2026-10-31 check did not use an NPU or install a
global OPP. Rebuild and validate the final package before production rollout.

## xLLM Integration

The existing `npu_mega_gdn_mtp_decode` wrapper and speculative decode state
handling remain the consumer interface. These MTP optimizations require no
new arguments or Python-model changes. Point the xLLM operator dependency at
this PR, rebuild its OPP and preserve the cache layout/state-index contract.

The `split_qkv_rmsnorm_mrope` TileLang kernel belongs to the xLLM repository
and is independent of this operator change.
