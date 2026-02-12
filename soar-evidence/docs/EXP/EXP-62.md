# EXP-62: Confidence-Gated Execution

## Question
Is the FalseExec curve linear with respect to confidence?

## Protocol
- Fixed: 57-bin MDU, posterior values
- Varied: Confidence threshold sweep

## Signals
FE rate vs confidence curve, IMMORTAL distribution

## Result
Non-linear FE curve. IMMORTAL clusters in infant bins — high confidence ≠ low FE universally.

## Verdict
⚠️ PARTIAL

## Implication
Confidence gating is complex. Simple thresholds don't capture the non-linear structure.
