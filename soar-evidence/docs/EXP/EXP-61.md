# EXP-61: Bin Coalescence

## Question
Can borderline bins be merged to reduce complexity?

## Protocol
- Fixed: 57-bin structure, posterior values
- Varied: Merge candidates, merge strategies

## Signals
FE rate before/after merge, capture rate impact

## Result
Merging borderline bins increases FalseExec. Coalescence is destructive.

## Verdict
❌ REJECTED

## Implication
Bins cannot be merged. Each bin carries unique structural information.
