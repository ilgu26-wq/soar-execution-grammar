# EXP-47: Minimal State Distillation

## Question
Can the feature space be reduced while preserving performance?

## Protocol
- Fixed: Core v2, posterior learning
- Varied: Feature count: 12 → 6, feature selection

## Signals
PF, Win Rate, FE before/after distillation

## Result
12→6 features with performance preserved and slightly improved. Surviving features: E_sign, dE_sign, Shadow, ARG_depth, Regime, AEP_zone.

## Verdict
✅ PASS

## Implication
6 features are the minimal set. No additional features may be added without re-validation.
