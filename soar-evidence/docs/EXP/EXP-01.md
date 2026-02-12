# EXP-01: RAW vs SOAR

## Question
Does the SOAR execution grammar improve performance compared to raw (unfiltered) signal execution?

## Protocol
- Fixed: Signal generation parameters, market data, tick value
- Varied: Execution mode: RAW (all signals) vs SOAR (grammar-filtered)

## Signals
PF (Profit Factor), Win Rate, Max DD%, Net PnL, Loss Streak

## Result
SOAR reduces max drawdown and consecutive loss streaks while preserving profitable trades. Grammar filtering structurally improves risk-adjusted returns.

## Verdict
✅ PASS

## Implication
SOAR grammar is structurally superior to raw execution. Proceed to component-level validation (EXP-02).
