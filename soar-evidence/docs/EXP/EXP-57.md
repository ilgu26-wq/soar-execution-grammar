# EXP-57: Execution Probability

## Question
Does learned p_exec dominate the fixed Sharp Boundary?

## Protocol
- Fixed: v3_runtime, bin structure
- Varied: Decision source: p_exec vs fixed Sharp Boundary

## Signals
PF, FE rate, capture rate per decision source

## Result
Learned p_exec dominates Sharp Boundary. Posterior-based decisions outperform fixed thresholds.

## Verdict
✅ PASS

## Implication
p_exec is the primary decision variable. Sharp Boundary is the fallback.
