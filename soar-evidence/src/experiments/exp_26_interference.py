#!/usr/bin/env python3
"""
EXP-26: Micro-Macro Orbit Interference
========================================
"ZOMBIE micro orbit's/of failureis it?, macro orbitat 's/ofone/a captureis it?,
 two orbitIs this interference arising from relative phase difference?"

MOTIVATION:
 EXP-25from ZOMBIE per/staralso's/of species Confirmed.
 AOCL commit 100%, dominant orbit ALL ALPHA.
 If so why ATP occurrence?
 Between micro orbit (alpha energy) and macro orbit (market energy)
 relativeever/instance phase(Relative Phase) Measurementby doing Answerdoes.

DEFINITIONS:
 Macro Orbit State M(k):
 cumulative dE over 10-bar window (market energy field)

 Micro Orbit State m(k):
 E_total(k) from alpha energy trajectory

 Relative Phase Φ(k):
 Binary: sign(E_macro) × sign(E_micro) ∈ {-1, 0, +1}
 Continuous: E_micro / max(|E_macro|, 1.0), clipped to [-10, +10]

 Sign-change count:
 Number of Φ sign flips across trade lifetime

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 PF: 1.28 [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED

 === Φ Sign-Change Analysis (key/core discriminator) ===

 Fate mean_Δ ≥2_chg% 0_chg% Φ>0%
 IMMORTAL 0.72 21.3% 61.7% 50.6%
 SURVIVED 0.29 4.8% 85.7% 48.6%
 ZOMBIE 1.76 60.8% 13.7% 49.2%
 TERMINATED 2.10 60.0% 19.0% 48.3%
 STILLBORN 1.80 52.2% 20.3% 54.1%

 Critical discovery:
 - SURVIVED: 85.7% have ZERO sign changes
 → Φ never flips. Macro-micro in permanent phase lock.
 - ZOMBIE: 60.8% have ≥2 sign changes
 → Φ oscillates. Phase alternates between reinforcement and interference.
 - TERMINATED: 60.0% have ≥2 sign changes
 → Same oscillation rate as ZOMBIE, but WR 17.1% vs 62.7%.
 - IMMORTAL: 61.7% have ZERO sign changes
 → Nearly as stable as SURVIVED.

 === ZOMBIE ATP/Revival Phase Dynamics ===

 Phase mean_Φ Φ>0%
 Pre-ATP +0.078 49.0%
 At ATP +0.020 47.1% ← Φ drops (phase misalignment)
 Post-ATP (+1) +0.111 51.1% ← Φ recovers (macro energy arrives)
 Post-ATP (+3) +0.089 48.9%
 At End +0.000 49.0%

 → ATP occurs when Φ drops (micro-macro misalignment)
 → Recovery happens when Φ recovers (macro energy re-enters)

 === ZOMBIE Per-bar Φ Evolution ===

 bar mean_Φ Φ>0%
 1 +0.078 49.0%
 2 +0.059 51.0%
 3 +0.196 58.8% ← peak alignment
 4 -0.039 45.1% ← phase flip
 5 -0.020 47.1%
 6 +0.059 47.1% ← oscillation back
 7 -0.059 41.2% ← another flip
 8 +0.118 51.0%
 9 +0.157 52.9%
 10 +0.000 49.0%

 → Clear oscillation pattern with alternating phase

 === THREE TESTS ===

 Test (A) — micro orbit failure?
 ZOMBIE sign-changes = 1.76 < TERMINATED = 2.10
 ZOMBIE WR = 62.7% >> TERMINATED WR = 17.1%
 → NO. ZOMBIE is not simply a failed micro orbit.

 Test (B) — macro orbit capture?
 SURVIVED: 85.7% zero sign-changes (stable capture)
 ZOMBIE: 13.7% zero sign-changes (unstable)
 → PARTIAL. Not stable capture — ZOMBIE oscillates.

 Test (C) — two orbit's/of interference?
 ZOMBIE oscillates (60.8% ≥2 changes) but survives (WR 62.7%)
 TERMINATED oscillates (60.0% ≥2 changes) but dies (WR 17.1%)
 SURVIVED is stable (4.8% ≥2 changes) and thrives (WR 95.2%)
 → YES. ZOMBIE is interference.
 Same oscillation as TERMINATED, but ZOMBIE receives energy
 at phase-aligned moments while TERMINATED does not.

 ★ VERDICT:
 ZOMBIE interference(interference) phenomenonis.
 micro orbit autonomy losing macro orbit's/of relativeever/instance phaseat 's/ofdo
 temporaryever/instanceto/as Maintained interference state.

 "Same oscillation, but ZOMBIE receives energy at the moment it overlaps with the macro orbit"

PHYSICAL INTERPRETATION:
 SURVIVED = permanentever/instance phase fixed (permanent phase lock)
 IMMORTAL = almost permanentever/instance phase fixed (near-permanent phase lock)
 ZOMBIE = Phase oscillation + energy transfer (oscillating interference)
 TERMINATED = phase oscillation + energy dissipation (oscillating dissipation)
 STILLBORN = phase none (no micro orbit ever formed)

 "alpha micro orbitfrom bornknowonly,
 survival always macro orbitand's/of relativityfrom determinationbecomes."
"""

if __name__ == '__main__':
 print("EXP-26: Micro-Macro Orbit Interference — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp26_interference/")
