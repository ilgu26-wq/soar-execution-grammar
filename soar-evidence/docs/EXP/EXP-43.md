# EXP-43: Computation Skip

## Question
Can bins with sufficient DENY evidence skip downstream computation?

## Protocol
- Fixed: Posterior from EXP-41, ZOMBIE from EXP-42
- Varied: Skip threshold, computation savings

## Signals
Computation reduction %, performance preservation

## Result
Dead compute eliminated. Bins with strong DENY evidence skip all downstream processing.

## Verdict
✅ SEALED

## Implication
Computation skip saves resources without affecting outcomes.
