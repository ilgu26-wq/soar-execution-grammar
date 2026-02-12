# EXP-68: Bagged Posterior

## Question
Does bootstrap aggregation of posteriors reduce fold-to-fold variance?

## Protocol
- Fixed: v3_runtime, 57-bin MDU
- Varied: Bootstrap samples, aggregation method

## Signals
FE_std before/after bagging, IMM preservation

## Result
NO EFFECT. Variance is from test fold composition, not posterior uncertainty.

## Verdict
❌ NO EFFECT

## Implication
Bagging doesn't help because the variance source is not what bagging addresses.
