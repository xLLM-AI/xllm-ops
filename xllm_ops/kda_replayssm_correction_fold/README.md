# KDA ReplaySSM correction fold

The Ascend 910_93 operator folds accepted correction records into the
recurrent states of several layers in one ACLNN call.

The four input lists hold independent per-layer allocations. Each state is
FP32 `[slots, 8, 128, 128]`; the key, correction, and decay records are FP32
`[slots, 8, 8, 128]`. The output list must alias the state list. The two
INT32 vectors select the slot and accepted prefix. Slots are unique among
live requests; out-of-range slots and nonpositive acceptance leave state
unchanged. Accepted lengths above eight are clamped.
