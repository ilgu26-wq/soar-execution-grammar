# EXP-54: Deferred Compute

## Question
Can orbit calculations be deferred until boundary events occur?

## Protocol
- Fixed: Hierarchical pruning from EXP-53
- Varied: Deferred vs eager computation

## Signals
Computation savings %, performance equivalence

## Result
47.8% orbit computation savings through deferred execution. No performance difference.

## Verdict
✅ PASS

## Implication
Deferred compute confirmed. Nearly half of orbit calculations are unnecessary.
