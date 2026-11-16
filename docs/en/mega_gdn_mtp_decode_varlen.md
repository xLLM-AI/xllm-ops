<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnMtpDecodeVarlen

English | [简体中文](../zh/mega_gdn_mtp_decode_varlen.md)

Packed-prefix variant of MegaGdnMtpDecode for Ascend A3. It reuses the same
fused causal convolution, Q/K normalization, gating, FP32 recurrence,
per-token checkpoints, and RMSNorm/SiLU implementation. The original dense
operator and its ABI are unchanged.

## Shapes and state contract

Let S be the fixed checkpoint stride (2 through 17), B the request count
(1 through 128), T the packed graph capacity, D=128, and C=(2*NK+NV)*D.

- qkv: BF16 [T,C]; z: BF16 [T,NV,D]; a and b: BF16 [T,NV].
- convWeight: BF16 [4,C]; normWeight: BF16 [D].
- aLog and dtBias: FP32 [NV].
- convState / convStateOut: aliased BF16 [N,S+2,C].
- ssmState / ssmStateOut: aliased FP32 [N*S,NV,D,D].
- stateIndices, numAcceptedTokens: device INT32 [B].
- cuSeqLens: device INT32 [B+1], monotonically nondecreasing endpoints.
- convOut: BF16 [T,C]; out: BF16 [T,NV,D].

Each current prefix length cuSeqLens[i+1]-cuSeqLens[i] is in [0,S].
Endpoints lie in [0,T]; the initial endpoint may be nonzero, allowing
multiple request chunks to address one packed input. Zero-length requests
do not read or modify state. Only referenced packed output rows are written;
the caller initializes padding rows and may share outputs across disjoint
chunks. Each live request must own a unique stateIndices slot across all
chunks. Zero-length requests may share an index because they never access state.

The previous accepted count remains in [1,S], independently of the current
prefix length. Restore SSM from stateIndices[i]*S+accepted-1 and convolution
history from accepted-1:accepted+2 in the same slot. Write SSM checkpoint stateIndices[i]*S+t
for every current token t, convolution history rows 0:2 from the preceding
accepted history, and current raw QKV into rows 2:2+prefixLength.
Unwritten cache rows remain unchanged. The single index input enforces
same-slot reads and writes; separate read/write mappings and cross-slot state
migration are not supported. Batch order may change while each request keeps
its own slot. No cross-request state snapshot or extra state copy is required.
flaSsmStateLayout=true stores state in [K,V] order; false uses
[V,K], matching the original operator.

Lengths and accepted counts are device inputs: changing them during ACL
graph replay changes actual loop bounds without recapture or Host reads.
