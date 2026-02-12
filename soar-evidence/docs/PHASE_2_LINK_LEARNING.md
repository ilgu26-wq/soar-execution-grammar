# Phase 2 — Link & Learning (EXP-41 ~ EXP-64)

> "We learned WHAT to execute, not HOW."

## Summary

Phase 2 connects the structural observations from Phase 1 to actual execution decisions. The key innovation is Bayesian posterior learning: the system learns execution probability per structural bin, not execution strategy. The distinction is critical — learning WHAT to execute preserves the structural laws; learning HOW to execute would violate them.

## Key Discoveries

### Bayesian Posterior (EXP-41 ~ 43)

Threshold learning (EXP-41) introduced Beta posterior per feature bin: each structural coordinate gets a Beta(α, β) distribution that tracks execution success. ZOMBIE learning (EXP-42) sealed the judge calibration. Computation skip (EXP-43) eliminated dead compute — bins with sufficient evidence to DENY skip all downstream computation.

### Energy Conservation (EXP-44 ~ 45)

EXP-44 established the Energy Conservation Law (ECL): energy is conserved within the system. Only execution converts energy to PnL. This means reward shaping is forbidden — any attempt to modify energy through non-execution means violates conservation.

EXP-45 observed exit mechanics — how energy dissipates when an alpha terminates.

### Minimal Distillation (EXP-47)

12 features were distilled to 6 with performance preserved (and slightly improved). The 6 surviving features: E_sign, dE_sign, Shadow, ARG_depth, Regime, AEP_zone. This distillation is a law — no additional features may be added without re-validation.

### Sharp Boundary (EXP-48 ~ 50)

EXP-48 is the most important discovery in Phase 2: **"Execute or not?" dominates "When to execute?"** The binary decision boundary (Sharp Boundary) captures almost all performance. Timing refinement (EXP-50) adds negligible value. IMMORTAL-tight (EXP-49) showed that tightening the IMMORTAL boundary improves precision without sacrificing capture.

### Worldline Pruning (EXP-52 ~ 54)

"Dead worldlines don't compute." EXP-52 introduced worldline pruning — alphas classified as dead are removed from computation. EXP-53 organized this into a 3-tier hierarchy. EXP-54 demonstrated deferred compute: 47.8% of orbit calculations are saved by deferring computation until boundary events occur.

### Global Stress Test (EXP-55)

The v3_runtime was stress-tested across 4 markets, 90K+ bars, and 2,910 trades. NQ achieved ALL PASS. This is the freeze point for v3_runtime.

### Cross-Market & Calibration (EXP-51, 56)

NQ laws are preserved across markets (EXP-51), but ES and BTC require separate calibration. FalseExec is structural, not tunable (EXP-56) — it cannot be reduced by parameter adjustment.

### Posterior Topology (EXP-57 ~ 64)

EXP-57 confirmed that learned p_exec dominates the Sharp Boundary. EXP-58 (World A/B Test) showed dramatic improvements in World B. EXP-59 updated the constitution with a final judgment window ≥ 200.

EXP-60 analyzed posterior shape: 57 bins with 70.2% maturity and 0 UNDECIDED bins.

EXP-61 attempted bin coalescence — **REJECTED**. Merging borderline bins increases FalseExec. EXP-61v2a sealed this as the Forbidden Band law: 57 bins is the Minimal Distinct Unit (MDU). "Cannot merge = structure is already optimal" (Cannot merge = structure is already optimal).

EXP-62 revealed a non-linear FE curve with IMMORTAL clustering in infant bins. EXP-63 mapped infant bin growth: 19% IMMORTAL rate in infant bins reflects structural data scarcity, not poor structure.

EXP-64a/b/c explored observation scheduling, boundary compression, and directed pheromone. Results were mixed: scheduling is safe to p=0.25 for mature bins, but boundary compression and directed pheromone showed no significant effect.

## Phase 2 Conclusion

By the end of Phase 2, the execution system was complete: structural observation → Bayesian posterior → Sharp Boundary → Gate decision. The remaining challenge was fold-to-fold variance — addressed in Phase 3.
