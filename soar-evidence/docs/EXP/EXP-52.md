# EXP-52: Worldline Pruning

## Question
Can dead alpha worldlines be removed from computation?

## Protocol
- Fixed: Core v2, alpha lifecycle tracking
- Varied: Pruning criteria, pruning aggressiveness

## Signals
Computation savings, performance impact of pruning

## Result
Dead worldlines pruned without performance impact. Dead worldlines don't compute.

## Verdict
✅ PASS

## Implication
Worldline pruning is safe and saves computation. Proceed to hierarchical pruning (EXP-53).
