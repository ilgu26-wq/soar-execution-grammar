#!/usr/bin/env python3
"""
EXP-28: Influence → Geometry Drift
===================================================
"AIS cumulativeif becomes, alpha Generation space's/of geometric actualto/as Movementdo?"

MOTIVATION:
 EXP-27from AIS score's/of validity Proof.
 EXP-28the causation that scores change the space Proof.
 → "Reinforcementconfirmed as geometric learning, not 'learning'.

DESIGN:
 293 trades → 4 temporal cycles (73-74 trades each)
 Each cycle: compute proposal_share per RC cell → apply influence rule
 Measure: Local Drift, Manifold Rotation, Boundary Compression

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED
 Execution: ZERO changes — geometry is observation-only

 === Local Drift (corr ΔI, ΔP) ===

 Cycle 0→1: +0.0444
 Cycle 1→2: -0.1534
 Cycle 2→3: +0.1160
 Global: +0.0277

 H1 CONFIRMED: Global correlation is positive — influence and distribution Movement same direction.
 only/but, weak correlation. κ=0.02's 'scent' actually works subtly.

 === Manifold Rotation (θ) ===

 Cycle 0→1: 26.48°
 Cycle 1→2: 9.28°
 Cycle 2→3: 18.00°
 Total: 34.12°

 H3 REJECTED (threshold 30° exceeded): Distribution rotates more than expected.
 interpretation: geometric Movement existenceone, κ=0.02at 's/ofone/a thingonly not... but
 market regime itself's/of changealso included. C0→C1 26.48°to/as most large
 (C0 WR 50.7% → C1 WR 34.2% — regime transition).

 Key discovery: θ 0 . distribution certain rotationdoes.
 C1→C2 9.28°to/as most small = Stablefrom rotationalso is small.

 === Boundary Compression ===

 Cycle High≥0.6 Mid Low<0.3 near_0.6 near_0.3
 C0 19.2% 35.6% 45.2% 16.4% 21.9%
 C1 21.9% 16.4% 61.6% 8.2% 20.5%
 C2 23.3% 21.9% 54.8% 13.7% 13.7%
 C3 25.7% 21.6% 52.7% 6.8% 17.6%

 High% drift: 19.2% → 25.7% (+6.5%) ← wins trend
 Low% drift: 45.2% → 52.7% (+7.5%) ← together increase
 Near 0.6: 16.4% → 6.8% (-9.6%) ← boundary is empty

 interpretation: distribution polarization(polarization)becomes exists.
 Highand Low simultaneousat increaseand do, boundary near(0.55-0.65) empties.
 → "boundaryat stayed alphaplural fate/destiny determinationand do exists"
 → This is geometric separation(geometric separation)'s/of evidence

 === Z-I Transition by Cycle ===

 C0: zombies=21 Z-I=10(47.6%) Lock=21(28.8%)
 C1: zombies= 8 Z-I= 1(12.5%) Lock=14(19.2%)
 C2: zombies= 8 Z-I= 5(62.5%) Lock=19(26.0%)
 C3: zombies=14 Z-I= 5(35.7%) Lock=14(18.9%)

 H4 CONFIRMED: Z-I ratio betweenevery/each fluctuationBut/However
 C2from 62.5%until surge — energy injection ZOMBIE specific regimefrom among.

 === Influence Trajectories ===

 Low-AIS pathplural's/of influence cumulative reduction:
 Z_MOMENTUM.HIGH_CURV@DEAD: 1.0000→0.9938→0.9885→0.9849→0.9828
 MICRO_MOMENTUM.ALIGNED@DEAD: 1.0000→0.9939→0.9909→0.9879

 → low/day AIS path betweenevery/each influence reductiondoes.
 → This is "slow geometric Movement"'s/of mechanism.

 === Hypothesis Verdict ===

 H1 corr(ΔI, ΔP) > 0: +0.0277 ✓ CONFIRMED
 H2 Low ΔP ≤ High ΔP: REJECTED (reversal)
 H3 θ > 0 & small: 34.12° ✗ REJECTED (expected secondand)
 H4 Z-I ratio stable/↑: ✓ CONFIRMED

 2/4 confirmed → PARTIAL DRIFT

PHYSICAL INTERPRETATION:

 EXP-28 shows thing:

 1. geometric Movement existencedoes (θ ≠ 0, corr > 0)
 2. However κ=0.02's/of pure influencethan/more than regime change more is large
 3. distribution polarizationbecomes exists (High↑, Low↑, boundary↓)
 4. Low-AIS path's/of influence cumulative reduction clear

 Conclusion: "score space bardream?" → YES, But/However slowly.
 market regime changeand combining operationdoes.
 κ=0.02functions as 'scent' at exactly the intended intensity.
 Stronger and it becomes RL; at this intensity, it is geometric learning.

 "Reinforcementlearning not... but, terrain learningis."
"""

if __name__ == '__main__':
 print("EXP-28: Influence → Geometry Drift — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp28_geometry_drift/")
