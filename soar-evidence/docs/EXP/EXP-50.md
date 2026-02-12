# EXP-50: Execution Delay

## Question
Does delaying execution after a boundary signal improve outcomes?

## Protocol
- Fixed: Core v2, Sharp Boundary, IMMORTAL-tight
- Varied: Execution delay: 0, 1, 2, 5 bars

## Signals
PF, Win Rate, Net PnL per delay setting

## Result
Delay has minimal effect. The boundary decision dominates; timing within the window is noise.

## Verdict
✅ PASS

## Implication
Execution timing within the boundary window is not critical. Confirms EXP-48.
