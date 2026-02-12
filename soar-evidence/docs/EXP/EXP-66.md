# EXP-66: Shrinkage Posterior

## Question
Does shrinking the posterior toward the global mean reduce variance?

## Protocol
- Fixed: v3_runtime, 57-bin MDU
- Varied: Shrinkage coefficient (0 to 1)

## Signals
FE_std, IMM capture at different shrinkage levels

## Result
FE_std decreases but IMM also decreases. Pareto frontier mapped — no free lunch.

## Verdict
✅ PASS

## Implication
Shrinkage trades variance for capture. Need a method that improves both axes.
