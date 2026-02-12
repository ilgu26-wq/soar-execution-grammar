#!/usr/bin/env python3
"""
EXP-35: Frame Cost Law (coordinates cost law)
===================================================
"co-movement frame's/of cheap Stableat large/versus(cost) exists — information loss"
Frame-Information Asymmetry (FIA): Absolutefrom thing vs Comovingfrom death thing

MOTIVATION:
 EXP-34showed asymmetry: Going to Alpha-Comoving makes observation appear 'stable' but
 actual performance gets worse. If so, the cost of that 'stability' should remain in the data.
 this information costto/as quantificationdoes.

DESIGN:
 Per-bar, per-trade computation:

 1. Energy vector magnitude:
 |E_absolute| vs |E_comoving|
 SNR = |E_abs| / |E_com| (signal-to-noise ratio)

 2. Angular change per bar:
 θ_abs = angle between consecutive E vectors in absolute frame
 θ_com = angle between consecutive E vectors in comoving frame
 (comovingfrom |E_com| < 0.05surface/if θ_com = 90° degeneration)

 3. FIA (Frame Information Asymmetry):
 Angular cost = Σθ_com - Σθ_abs
 FIA = angular_cost / (Σθ_abs + Σθ_com)
 FIA > 0 → Comoving more non-/fireStable (information cost occurrence)
 FIA < 0 → Comoving more Stable (cheap Stable)

 4. Degenerate fraction: bars where |E_com| < 0.1
 → comoving vector too because small eachalso Measurement degeneration

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED

 === FIA Overview ===

 Mean FIA: 0.5436 Std: 0.4672
 Range: [-0.57, 1.00]
 Angular cost mean: 114.63°
 Degenerate fraction: 14.1%

 === FIA by Fate ===

 FIA θ_abs θ_com WR
 IMMORTAL 0.745 2.7° 13.0° 95.7%
 SURVIVED 0.854 2.6° 16.2° 95.2%
 ZOMBIE 0.383 15.7° 29.7° 62.7%
 TERMINATED 0.274 16.1° 24.5° 17.1%
 STILLBORN 0.840 2.8° 22.6° 0.0%

 === FIA by Outcome ===

 Winners FIA: 0.650
 Losers FIA: 0.475

 === Quartile Analysis (key/core) ===

 Q1 (FIA -0.57~0.19): WR=23.3% TERM=64.4% IMM=4.1%
 Q2 (FIA 0.19~0.80): WR=46.6% TERM=42.5% IMM=23.3%
 Q3 (FIA 0.80~0.91): WR=61.6% TERM=13.7% IMM=34.2% ← mostever/instance interval
 Q4 (FIA 0.91~1.00): WR=25.7% TERM=23.0% IMM=2.7%

 === 4 hypothesis all NOT SUPPORTED — However half/anti-transition truth ===

 H-35a: TERMINATED FIA=0.274 < IMMORTAL FIA=0.745 → opposite!
 H-35b: Losers FIA=0.475 < Winners FIA=0.650 → opposite!
 H-35c: Alpha-dom FIA=-0.169 < Absolute-dom FIA=0.743 → opposite!
 H-35d: FAILED_OPEN FIA=0.481 < CLOSED_ALPHA FIA=0.699 → opposite!

PHYSICAL INTERPRETATION:

 expectedand exactly opposite's/of Result appeared.
 This is more Answer.

 ═══ key/core Discovery ═══

 1. "successdo alpha never coordinatesfrom Stableand, co-movement coordinatesfrom non-/fireStabledo"
 IMMORTAL: θ_abs = 2.7° (Stable) vs θ_com = 13.0° (non-/fireStable) → FIA = 0.745
 why? alpha entryfrom far leaving because.
 never coordinatesfrom consistent directionto/as MovementBut/However,
 co-movement coordinates(entry diverges and the angle changes from the reference frame.
 → " alpha born where far between."

 2. "failuredo alpha all coordinatesfrom non-/fireStabledo"
 TERMINATED: θ_abs = 16.1° (non-/fireStable) vs θ_com = 24.5° (non-/fireStable) → FIA = 0.274
 both sides non-/fireStable → where lookalso shakes.
 → "bad alpha self whereto/as know does not know."

 3. "Alpha-dominant trades's/of FIA = -0.17 (negative!)"
 negative FIA = Comoving Absolutethan/more than more Stable
 This is EXP-34 Result perfectly explanationdoes:
 The observer went to Alpha-Comoving because Comoving appeared more stable
 But/However that is "alpha not moving because" (entry nearat staying)
 → "following easy alpha failuredoes"

 4. "Q3 interval (FIA 0.80~0.91) mostever/instance"
 WR=61.6%, IMMORTAL=34.2%
 high FIA = never coordinates Stable + co-movement coordinates non-/fireStable = certain Movement among
 Q4from again falls keep = STILLBORN (energy none degeneration)

 ═══ EXP-34 asymmetry do ═══

 EXP-34shows 'Alpha-dominant WR=21.9% vs Absolute-dominant WR=44.1%"
 EXP-35 that cause revealed:

 Alpha-dominant = FIA negative = Comoving more Stable = alpha not movement
 Absolute-dominant = FIA positive = Absolute Stable = alpha certain Movement

 observation co-movement coordinates selectiondoes = alpha not moves = profit does not exist.

 ═══ irreversiblenature/property's/of truly/real cause ═══

 "irreversiblenature/property death trades's/of asymmetry not... but, observation information's/of asymmetryfrom first life long."
 proposition EXP-35from Confirmedbecame.

 concreteever/instanceto/as:
 FIA > 0.8person/of alpha WR 61.6% — never coordinatesfrom Stableto/as Movement among
 FIA < 0.2person/of alpha WR 23.3% — which coordinatesfromalso direction none

 irreversiblenature/property's/of start = observation " alpha will follow number/can exist?"'s/of Answer
 "the moment it changes from 'yes' to 'no'.

 coordinates cost law:
 "Stableto/as coordinates information closing and,
 non-/fireStabledo coordinates truth beforedoes.
 observation uncomfortable coordinatesat stays."
"""

if __name__ == '__main__':
 print("EXP-35: Frame Cost Law — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp35_frame_cost/")
