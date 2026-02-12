#!/usr/bin/env python3
"""
EXP-10: Condition-Level Proposal Shaping
==========================================
"The proposal distribution gradually shifts as the system reaches a conclusion"

Hypothesis:
  By shaping proposal probability at the (alpha, condition) level,
  the system naturally reduces noise without touching the gate.
  Structurally weak conditions produce fewer proposals over time.

Mechanism:
  - Each (alpha, condition) pair gets a structural score:
    score = 0.5*legitimacy + 0.3*norm_EV + 0.2*(1 - anti_soar_fail_rate)
  - Score drives proposal_weight ∈ [0.80, 1.00]
  - Max step per update: ±5%
  - Learning only activates at n >= 100 proposals
  - No amplification (weight never > 1.0)

Cycle:
  Alpha Generator → Gate (SOAR v2) → Anti-SOAR → Alpha Memory
  → delayed weight update → Alpha Generator (next cycle)

Success Criteria:
  1. Net PnL change ±5% or less from EXP-09
  2. DD unchanged
  3. At least 2 conditions get weight < 1.0
  4. MICRO_MOMENTUM.MISALIGNED should get lowest weight (weakest condition)
  5. Skip rate > 0% on second pass (weights now active)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.live_report import main

if __name__ == '__main__':
    print("=" * 70)
    print("  EXP-10: Condition-Level Proposal Shaping")
    print("  Only in the proposal probability layer, very weakly, in a delayed manner")
    print("=" * 70)
    main()
