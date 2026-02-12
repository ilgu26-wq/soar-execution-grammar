# EXP-65: Walk-Forward Constitution

## Question
Is the 70/30 train/test performance gap a temporal shift or partition artifact?

## Protocol
- Fixed: v3_runtime, posterior
- Varied: Random partition seeds, split ratios

## Signals
Gap magnitude per partition, gap vs seed correlation

## Result
70/30 gap is a partition artifact, not temporal shift. Gap changes with random seed.

## Verdict
✅ PASS

## Implication
Variance is in the data split, not the model. Variance reduction should target partition sensitivity.
