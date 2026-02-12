#!/usr/bin/env python3
"""
EXP-12: Regime × Condition Resolution (dimension expansion, no amplification)
==========================================================
"EXP-10 what whether to reduce learned and,
 EXP-12 when whether to reduce learns."

Hypothesis:
 The same alpha/condition shows entirely different structural legitimacy
 depending on market regime. Separating this dimension reveals when
 to reduce proposals — not just what to reduce.

Mechanism:
 - Key extension: (alpha, condition) → (alpha, condition, regime)
 - RC-level structural score: same formula, same [0.80, 1.00] range
 - Weight lookup priority: RC > condition > default
 - RC min_n = 30 (smaller slices)
 - No amplification (delta clipped to ≤ 0)
 - Gate, execution, size: all LOCKED

Cycle:
 Force → Alpha Generator (regime-aware) → Gate (SOAR v2, untouched) →
 Anti-SOAR (RC-level feedback) → Alpha Memory (RC stats) →
 delayed RC weight update → Alpha Generator (next cycle)

Success Criteria:
 1. Net PnL / DD / Gate = identical to EXP-10
 2. Same alpha.condition shows legitimacy gap ≥ 0.15 across regimes
 3. RC weight updates show regime-specific adjustments
 4. No amplification — all deltas ≤ 0

Key Finding:
 MICRO_MOMENTUM.MISALIGNED: gap = 0.156
 TREND = 0.556 (n=126), DEAD = 0.400 (n=15)
 → Same condition, completely different structural validity by regime
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.live_report import main

if __name__ == '__main__':
 print("=" * 70)
 print(" EXP-12: Regime × Condition Resolution")
 print(" The same alpha shows completely different coherence depending on the regime")
 print(" Open only dimensions, do not increase intensity")
 print("=" * 70)
 main()
