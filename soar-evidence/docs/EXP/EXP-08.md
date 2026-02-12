# EXP-08: Boundary Sensitivity

## Question
Are boundary parameters robust under perturbation?

## Protocol
- Fixed: SOAR v2 grammar, market data
- Varied: Boundary threshold, cooldown, energy parameters ±20%

## Signals
PF stability, DD stability across parameter variations

## Result
Performance degrades gracefully under parameter perturbation. No cliff effects.

## Verdict
✅ PASS

## Implication
Boundary parameters are robust. No need for fine-tuning or optimization.
