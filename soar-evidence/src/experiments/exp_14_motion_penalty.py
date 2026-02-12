#!/usr/bin/env python3
"""
EXP-14: Motion-Aware Alpha Penalty — purification based on motion quality
================================================================
"Gate looks at survival only. Alpha looks at quality."

Hypothesis:
  EXP-13 mapped where FAST_ADVERSE clusters. EXP-14 applies delayed
  penalty to RC slices where motion_score < 0 (more fast adverse than
  healthy movements).

Motion Score:
  motion_score = healthy_rate - fast_adverse_rate
  Range: [-1, +1]. Negative = dominant fast adverse.

Penalty:
  penalty = clip(0.5 * |motion_score|, 0.0, 0.02)
  new_weight = max(0.80, old_weight - penalty)
  Applied AFTER structural score weights (EXP-10/12), stacked.

Key Design:
  - Gate LOCKED. Not touched.
  - No amplification (weight <= 1.0, penalty only)
  - Delayed learning: computed at end of cycle, applied next run
  - n >= 30 threshold (same as RC-level)
  - Max 2% penalty per cycle (very conservative)

Success Criteria:
  1. Net PnL / DD / Gate = identical (first cycle, no skip on pass 1)
  2. 2-3 RC slices penalized where FAST_ADVERSE dominates
  3. Anti-SOAR consistency: denied WouldWin < 50%
  4. Multi-cycle: FAST_ADVERSE% should decrease, HEALTHY% should increase

Key Findings (Cycle 1):
  - PnL $1,200 / WR 39.2% / PF 1.28 / DD 0.42% — IDENTICAL
  - 4 RC slices penalized out of 6 eligible
  - Worst: MICRO_MOMENTUM.MISALIGNED@TREND (score -0.20, 27.1% healthy, 47.1% FADV)
  - All penalties capped at max 0.02 (conservative)
  - Gate completely untouched
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.live_report import main

if __name__ == '__main__':
    print("=" * 70)
    print("  EXP-14: Motion-Aware Alpha Penalty")
    print("  'Gate is survival, Alpha is quality' — elevation of purification responsibility")
    print("  Reduction only, no amplification, delayed learning")
    print("=" * 70)
    main()
