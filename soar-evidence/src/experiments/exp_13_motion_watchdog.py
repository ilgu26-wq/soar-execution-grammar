#!/usr/bin/env python3
"""
EXP-13: Motion Watchdog — Post-Entry Movement Quality Observer
================================================================
"Monitors 'motion', not PnL"

Hypothesis:
  Gate-allowed trades can still be structurally dead — price doesn't
  move in the signal direction (NO_FOLLOW), reverses immediately
  (FAST_ADVERSE), or has weak Force alignment (LOW_FORCE).
  Mapping WHERE these failures cluster reveals a new attack surface
  that legitimacy alone cannot see.

Motion Tags:
  HEALTHY      — MFE >= 1 tick in 5 bars, no fast adverse
  NO_FOLLOW    — MFE too small (price didn't follow signal)
  FAST_ADVERSE — MAE > 4 ticks in first 3 bars (immediate reversal)
  LOW_FORCE    — dir_consistency < 0.45 at entry
  STALL        — price flat (neither MFE nor MAE significant)

Key Design:
  - Observation ONLY. No weight changes. Gate LOCKED.
  - Motion tags recorded per (alpha, condition, regime)
  - Failure rate gaps show where motion quality diverges by regime

Success Criteria:
  1. Net PnL / DD / Gate = identical
  2. Motion failure rate diverges by regime (gap >= 0.15)
  3. HEALTHY trades show significantly higher WR than failure trades
  4. FAST_ADVERSE is the dominant failure mode (impulse market)

Key Findings:
  - HEALTHY: 38.2%, WR 87.5%, avg PnL +$38.00
  - FAST_ADVERSE: 48.1%, WR 3.5%, avg PnL -$21.13
  - Z_MOMENTUM.HIGH_CURV: failure gap 0.260 (TREND=0.607, DEAD=0.867)
  - MICRO_MOMENTUM.MISALIGNED: highest condition-level failure rate 0.744
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.live_report import main

if __name__ == '__main__':
    print("=" * 70)
    print("  EXP-13: Motion Watchdog")
    print("  'Block if it does not move properly' — mapping the quality of motion")
    print("  Observation only, no behavioral change")
    print("=" * 70)
    main()
