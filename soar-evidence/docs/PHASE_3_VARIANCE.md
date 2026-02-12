# Phase 3 — Variance (EXP-65 ~ EXP-70)

> "Waiting is better than killing." — Waiting is better than killing.

## Summary

Phase 3 addresses the final challenge: reducing fold-to-fold variance while preserving IMMORTAL capture. The key discovery is that variance comes from test fold composition, not posterior estimation — and CI Wait dominates all other methods.

## Key Discoveries

### Walk-Forward Constitution (EXP-65)

The 70/30 train/test gap is a partition artifact, not a temporal shift. When the same data is re-partitioned with different random seeds, the gap changes proportionally. This means the variance is in the data split, not in the model.

### Shrinkage Posterior (EXP-66)

Shrinking the posterior toward the global mean reduces FE_std but also reduces IMM capture. The Pareto frontier was mapped: any shrinkage level trades variance reduction for IMMORTAL loss. There is no free lunch.

### Bagged Posterior (EXP-68, 68b)

Bagging (bootstrap aggregation of posteriors) has NO EFFECT. Variance is not reduced because it originates from test fold composition, not posterior uncertainty. Combining bagging with CI Wait (EXP-68b) performs worse than CI Wait alone — the combination adds complexity without benefit.

### CI Wait (EXP-69)

CI Wait introduces a 3-way decision: EXECUTE / DENY / WAIT. Instead of forcing uncertain bins into binary decisions, bins with confidence intervals spanning the boundary are assigned WAIT — no action taken.

CI Wait uses the 15th and 85th percentiles of the Beta posterior:
- If the 15th percentile > threshold → EXECUTE (confident positive)
- If the 85th percentile < threshold → DENY (confident negative)
- Otherwise → WAIT (insufficient evidence)

Results:
- **FE_std: 5.7%** (down from ~12% baseline)
- **IMM: 84.3%** (maintained above 80% threshold)
- **SG: +74.8** (Sharp Gap preserved)
- **WAIT rate: 9.6%** (within 15% budget)

CI Wait **dominates** shrinkage on both axes of the Pareto frontier: lower variance AND higher IMMORTAL capture. It is the only method that improves both simultaneously.

### Boundary Hysteresis (EXP-70)

Hysteresis (adding a buffer zone around the decision boundary to prevent flip-flopping) has NO EFFECT because the flip rate is already 0.0%. There is no instability to fix. The boundary is already stable.

## Phase 3 Conclusion

The variance problem is solved by CI Wait. The root cause — uncertain bins forced into binary decisions — is addressed directly by allowing a third option: wait.

Final system state:
- v3_runtime frozen
- 57-bin MDU topology sealed
- CI Wait 15/85 thresholds confirmed
- All integrity checks pass
- Reproducibility hash: `9ec07cd1a73ae484`
