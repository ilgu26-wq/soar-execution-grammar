# EXP-68b: Bagging × CI Wait

## Question
Does combining bagging with CI Wait improve over CI Wait alone?

## Protocol
- Fixed: v3_runtime, CI Wait parameters
- Varied: Bagging ON/OFF with CI Wait

## Signals
FE_std, IMM with combined vs CI Wait alone

## Result
CI alone is better than combined. Bagging adds complexity without benefit.

## Verdict
⚠️ SOFT PASS

## Implication
CI Wait is sufficient. Bagging is unnecessary complexity.
