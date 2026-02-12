#!/usr/bin/env python3
"""
EXP-09: Alpha Condition Refinement under Fixed Gate
=====================================================
"Which alpha conditions lose meaning" — only observe legitimacy % change

Hypothesis:
  Same alpha type can have DIFFERENT legitimacy depending on Force state.
  By splitting each alpha into condition sub-tags, we see WHICH conditions
  make an alpha structurally meaningful vs. meaningless.

Method:
  - Alpha types unchanged (6 types)
  - Each gets a condition tag from Force state at generation time
  - Memory tracks legitimacy per (alpha_type, condition) pair
  - Gate is FIXED — zero changes to v2 core
  - Money must remain identical to pre-EXP-09

Condition Splits:
  Z_MOMENTUM     → LOW_CURV / HIGH_CURV    (force curvature)
  RANGE_BREAKOUT  → NARROW_RNG / WIDE_RNG   (channel width)
  MICRO_MOMENTUM  → ALIGNED / MISALIGNED    (dir consistency)
  FLOW_IMBALANCE  → STRONG_F / WEAK_F       (force magnitude)
  MEAN_REVERT     → LOW_CURV / HIGH_CURV    (force curvature)
  VOL_CONTRACTION → STRONG_F / WEAK_F       (force magnitude)

Success Criteria:
  1. Money numbers IDENTICAL to pre-EXP-09
  2. At least 2 conditions show legitimacy delta > 0.05
  3. All Anti-SOAR verdicts remain GATE OK (WR < 50% for denied)
  4. No condition with n >= 100 triggers suppression
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.live_report import main

if __name__ == '__main__':
    print("=" * 70)
    print("  EXP-09: Alpha Condition Refinement under Fixed Gate")
    print("  Alpha condition resolution increase — dimensions only, intensity near 0")
    print("=" * 70)
    main()
