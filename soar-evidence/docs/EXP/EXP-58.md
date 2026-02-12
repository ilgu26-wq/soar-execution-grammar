# EXP-58: World A/B Test

## Question
Does the learned posterior improve performance in a parallel world test?

## Protocol
- Fixed: v3_runtime structure
- Varied: World A (baseline) vs World B (posterior-enhanced)

## Signals
FE rate difference, PnL difference per market/timeframe

## Result
NQ_Tick FE -40.9%p improvement. NQ_1min PnL +$10,426. World B dominates.

## Verdict
✅ PASS

## Implication
Posterior enhancement confirmed across worlds. Production should use World B.
