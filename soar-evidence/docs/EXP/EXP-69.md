# EXP-69: CI Wait

## Question
Can a 3-way decision (EXECUTE/DENY/WAIT) reduce variance while preserving capture?

## Protocol
- Fixed: v3_runtime, 57-bin MDU
- Varied: CI threshold percentiles (15/85)

## Signals
FE_std, IMM, SG, WAIT rate

## Result
DOMINATES shrinkage on both axes. FE_std 5.7%, IMM 84.3%, SG +74.8, WAIT rate 9.6%.

## Verdict
✅ PASS

## Implication
CI Wait is the final variance solution. Waiting is better than killing.
