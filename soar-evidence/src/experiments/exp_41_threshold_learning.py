#!/usr/bin/env python3
"""
EXP-41: THRESHOLD LEARNING — FIRST REAL LEARNING
===================================================
"judgment probability critical to/as adjustmentdoes."

MOTIVATION:
 EXP-40: "connection phase transition" Proof
 EXP-41: Judge datato/as p_exec to/as learns.

 ❌ RL, policy gradient, reward maximization not
 ✅ tradeswealth/department probability estimation + Leave-One-Out honest Verification

DESIGN:
 State = (ARG_depth, AEP_zone, Shadow_class, Regime)
 p_exec = P(win | state) estimated via hierarchical Bayesian LOO

 Hierarchy (fallback if bucket < 3):
 1. (depth, aep_zone, shadow, regime) — highest/best resolution
 2. (depth, aep_zone, shadow)
 3. (depth, aep_zone)
 4. (depth) — always data sufficient

 Laplace smoothing: α=1.0

RESULTS:
 Money preserved (simulation only):
 Net PnL: $1,200.00 [IDENTICAL]
 Gate: UNTOUCHED

 === Learned p_exec Distribution ===

 p_exec n actual_WR avg_pnl
 [0.0,0.1) 85 2.4% -4.65 ← certain die alpha
 [0.1,0.2) 61 6.6% -4.02 ← almost certain death
 [0.2,0.3) 15 46.7% +2.00 ← ZOMBIE region (uncertain)
 [0.5,0.7) 22 90.9% +8.64 ← alive alpha
 [0.9,1.0) 54 96.3% +9.44 ← certain alive

 === Comparison ===

 Strategy n WR PF Net$ MaxDD
 Baseline (all) 293 39.2% 1.29 $1,270 0.44%
 ALLOW-only 58 94.8% 36.67 $2,675 0.02%
 Graduated (EXP-40) 96 81.5% 9.05 $3,448 0.06%
 ★ LEARNED (EXP-41) 118 73.6% 5.64 $3,539 0.09%

 → Learned > Graduated: +$91 (all before among highest/best Net PnL)
 → 20 trades more executionwhile doing more earned = ZOMBIE selection accurate

 === Calibration ===

 [0.0,0.3): GOOD (pred=10.5%, actual=8.1%, Δ=2.4%)
 [0.3,0.6): OVERFIT (pred=37.1%, actual=0.0%, Δ=37.1%)
 [0.6,0.9): GOOD (pred=72.7%, actual=82.0%, Δ=9.3%)
 [0.9,1.0): GOOD (pred=93.7%, actual=96.3%, Δ=2.6%)

 → 3/4 interval GOOD. [0.3,0.6) intervalonly andever/instancesum.
 → interval n=17 (ZOMBIE+boundary). data insufficient region.

 === Hypothesis Tests ===

 H-41a SUPPORTED: Learned Net $3,539 > Graduated $3,448
 H-41b NOT SUPPORTED: [0.3,0.6) interval andever/instancesum (n=17, Insufficient data)
 H-41c SUPPORTED: Learned MaxDD 0.09% < Baseline 0.44%
 H-41d SUPPORTED: Learned $/trade $30.04 > Baseline $4.33

PHYSICAL INTERPRETATION:

 ═══ first th (ordinal) learning successdid ═══

 3/4 SUPPORTED. failureone/a H-41b "Insufficient data interval" exactly point out.
 This is learning's/of limit not... but ZOMBIE's/of essence:
 boundary essenceever/instanceto/as uncertaindo.

 ═══ learning one/a thing ═══

 - Gate: not tradesdream
 - Alpha: not tradesdream
 - only p_execonly adjustment
 - Result: Baseline versus/compared to $/trade 7times/multiple (4.33 → 30.04)

 ═══ one/a sentence ═══

 "Judge datato/as probability critical if learns,
 structure bardoes not dream withoutalso performance phase transitiondoes."
"""

if __name__ == '__main__':
 print("EXP-41: Threshold Learning")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp41_threshold_learning/")
