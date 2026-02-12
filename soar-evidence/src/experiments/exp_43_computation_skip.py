#!/usr/bin/env python3
"""
EXP-43: COMPUTATION SKIP LEARNING
===================================================
"Do not waste computation on already dead trades."

MOTIVATION:
 execution slowed keep: observation 6 layers computed every bar
 bar_evolution, energy, shadow, aep, atp/aoc, central_axis
 but even trades with a 'death signature' are computed to the end.

DESIGN:
 Skip trades:
 - ARG deny_bar=0 (entryalready dead) → skip all layers
 - ARG deny_bar≤2 + depth≥2 (initial death) → 4/6 layer skip
 - ARG depth≥3 (among death) → 5/6 layer skip

 Measurement:
 - total bar×layer computation load vs skip Possible computation load
 - skip target's/of WR (before Confirmed)

RESULTS:
 === Skip Analysis ===

 Total trades: 293
 Skippable: 194 (66.2%)
 Non-skippable: 99 (33.8%)

 Total bar×layer computations: 17,580
 Computations saved by skip: 4,628 (26.3%)

 Skippable WR: 18.0% (die trade)
 Non-skippable WR: 80.8% (alive trade)

 False skips (winning trades incorrectly skipped): 35/194 (18.0%)
 Estimated speedup: 1.36x

 === Hypothesis Tests ===

 H-43a NOT SUPPORTED: Saved 26.3% (< 30% threshold)
 → Entry skip bar=0from 0 computations (already computation before)
 → realever/instance reduction Early skip (bar 1-2)fromonly occurrence
 H-43b SUPPORTED: Skippable WR=18.0% (< 30% — safe to skip)
 H-43c SUPPORTED: Non-skippable WR=80.8% (> 80% — alive trades retained)

PHYSICAL INTERPRETATION:

 ═══ 26.3% sufficient? ═══

 Current 12.1s → ~8.9s after skip (1.36x)
 Meaningful improvement in real-time systems.
 But/However 30% undecidedonly H-43a NOT SUPPORTED.

 ═══ truly/real meaning ═══

 The essence of skipping is not speed, but 'not paying attention to what is dead'":
 - 66.2%'s/of trade already died
 - pluralat observation layer rotate trades waste
 - alive 33.8%at among WR=80.8%

 trades Not speed optimization, but attention allocation optimization.
"""

if __name__ == '__main__':
 print("EXP-43: Computation Skip Learning")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp43_computation_skip/")
