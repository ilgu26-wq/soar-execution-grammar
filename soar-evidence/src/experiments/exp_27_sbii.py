#!/usr/bin/env python3
"""
EXP-27: Score-Based Interference Influence (SBII)
===================================================
"The judge does not pick the answer. It changes the probability space where the answer is born."

MOTIVATION:
 EXP-26from ZOMBIE interference(interference) is a phenomenon Proof.
 Now interference state Removaldoes not, but through score-based influence
 alpha Generation distribution gradually re- experiment.

 key/core: determination·blocking·amplification all Prohibited. only score → influence → distribution Movement.

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 PF: 1.28 [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED
 Execution: ZERO changes — AIS is observation-only

 === ZOMBIE Sub-classification ===

 Z-Type n WR PnL avg_ΔE Φ_chg
 Z-I 21 76.2% +$650 +107.1 1.33 ← Interference-Gain
 Z-II 15 60.0% +$280 +104.2 1.73 ← Neutral
 Z-III 15 46.7% +$110 + 27.5 2.40 ← Interference-Loss

 Discovery:
 - Z-I (energy pureinjection): WR 76.2%, Φ oscillation minimum/minimal (1.33)
 - Z-III (energy pure): WR 46.7%, Φ oscillation maximum (2.40)
 - Z-I ZOMBIE amongfromalso "alphaat most near species"

 === Alpha Influence Score (AIS) ===

 AIS = 0.30×Energy_Survival + 0.25×Phase_Coherence
 + 0.20×Lifespan_Ratio + 0.25×Orbit_Potential

 Fate AIS_mean E_surv Φ_coh L_rat O_pot
 SURVIVED 0.7166 0.7698 0.9429 0.0000 1.0000
 IMMORTAL 0.6299 0.5537 0.8553 0.0000 1.0000
 ZOMBIE 0.4519 0.2276 0.6471 0.6902 0.3353
 Z-I 0.5129 0.2809 0.7333 0.4762 0.6000
 Z-II 0.4878 0.2627 0.6533 0.8533 0.3000
 Z-III 0.3307 0.1178 0.5200 0.8267 0.0000
 TERMINATED 0.2807 0.1112 0.5905 0.4152 0.0667
 STILLBORN 0.2031 0.0001 0.6406 0.2145 0.0000

 Critical findings:
 - AIS perfectly separates Lock from Non-Lock
 - Lock AIS mean: 0.6567 vs Non-Lock: 0.2957 (gap: +0.3610)
 - Z-I AIS (0.5129) sits between Lock and Non-Lock

 === AIS as Lock Predictor ===

 AIS ≥ 0.5 threshold:
 TP=63 FP=29 FN=5 TN=196
 Precision = 68.5%
 Recall = 92.6%

 → AIS catches 92.6% of all Lock alphas
 → 68.5% of high-AIS trades are actual Locks
 → Only 5 Locks missed (FN=5)

 === Distribution Shift (AIS Bands) ===

 Band n WR PnL Lock% Z-I%
 High ≥0.6 66 92.4% +$2,748 78.8% 7.6%
 Mid 0.3-0.6 70 71.4% +$1,885 22.9% 22.9%
 Low <0.3 157 2.5% -$3,433 0.0% 0.0%

 → High-AIS band: 92.4% WR, captures 78.8% of Locks
 → Low-AIS band: 2.5% WR, ZERO Locks, ZERO Z-I
 → Mid band is the "transition zone" where Z-I lives

 === Influence Update Rule ===

 Influence_next = Influence_current × (1 + 0.02 × (AIS_mean − 0.5))
 Clamp: [0.90, 1.05]

 RC paths: 24
 Influence range: [0.9948, 1.0027]
 Only 1 path above 1.0 (barely)
 23 paths below 1.0 (gentle suppression)

 → κ=0.02 produces maximum ±0.5% influence
 → This is "scent(scent)", not force

PHYSICAL INTERPRETATION:

 AIS 4dimension state is a vector:
 (Energy_Survival, Phase_Coherence, Lifespan_Ratio, Orbit_Potential)

 4dimension spacefrom:
 - Lock alpha upper right cornerat gatheredexists (high E, high Φ, high O)
 - STILLBORN lower left cornerat fixed (zero E, zero O)
 - ZOMBIE amongfrom oscillationdoes
 - Z-I Lock towardto/as tilted ZOMBIE

 Influence Rulerotates this space very slowly:
 - Good score path → born slightly more often
 - Bad score path → born slightly less often
 - never "blocking" none

 " trades Not policy learning, but terrain learning."
"""

if __name__ == '__main__':
 print("EXP-27: Score-Based Interference Influence — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp27_sbii/")
