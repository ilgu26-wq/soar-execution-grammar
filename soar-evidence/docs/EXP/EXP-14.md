# EXP-14: Motion Penalty

## Question
Do motion-based weight penalties improve gate decisions?

## Protocol
- Fixed: Core v2, motion observation from EXP-13
- Varied: Motion penalty strength and threshold

## Signals
PF with/without motion penalty, false execution rate

## Result
Motion-based penalties reduce false executions in high-motion regimes.

## Verdict
✅ PASS

## Implication
Motion penalty added to weight pipeline. Observation → penalty path confirmed.
