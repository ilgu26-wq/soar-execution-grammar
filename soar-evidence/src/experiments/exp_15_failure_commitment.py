#!/usr/bin/env python3
"""
EXP-15: Failure Commitment Layer (FCL) — Irreversible Trajectory Mark
======================================================================
"Not a Gate above the Gate, but an irreversible judgment record above the Gate"

Hypothesis:
  Some gate-allowed trades enter a "failure trajectory" — multiple
  structural signals of imminent failure fire simultaneously.
  Even if such trades survive by luck, the trajectory observation
  is irreversible and should inform future weight updates.

Key Principle:
  "Just because it survived by luck does not mean it was not in a failure orbit"

FCL Conditions (2+ must fire simultaneously):
  1. FAST_ADVERSE motion tag
  2. MFE/MAE ratio < 0.7
  3. Force sign flip within 3 bars
  4. Adverse speed: MAE at bar 3 > 6 ticks
  5. Low force at entry: dir_consistency < 0.40

Key Design:
  - NOT a gate. Does NOT block execution.
  - Irreversible mark: once committed, never erased
  - "Survived failure" tracked separately from "true failure"
  - Feeds into weight update pipeline as additional penalty evidence

Architecture Position:
  [ Force / Alpha / Regime / Motion ]  ← observation layers
                  ↓
          Alpha Proposal Weights       ← delayed learning
                  ↓
     Failure Commitment Layer (FCL)    ← irreversible mark (HERE)
                  ↓
          SOAR v2 Gate (PURE)          ← survival judgment (LOCKED)
                  ↓
               Execution

Success Criteria:
  1. Net PnL / DD / Gate = identical
  2. FCL commitment rate shows structural pattern (not random)
  3. Committed avg PnL << Normal avg PnL (validates trajectory detection)
  4. Survived failures exist but are rare (validates irreversibility value)

Key Findings (Cycle 1):
  - PnL $1,200 / WR 39.2% / PF 1.28 / DD 0.42% — IDENTICAL
  - 282 failure commitments out of 582 trades (48.5%)
  - Survival rate: 2.8% (8 survived out of 282 committed)
  - Committed avg PnL: -$21.70 vs Normal avg PnL: +$29.03
  - Dominant conditions: FAST_ADVERSE (252), HIGH_ADVERSE_SPEED (251), LOW_MFE_MAE (231)
  - Worst RC: MICRO_MOMENTUM.MISALIGNED@TREND (55.7% commitment rate)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.live_report import main

if __name__ == '__main__':
    print("=" * 70)
    print("  EXP-15: Failure Commitment Layer")
    print("  'Failure orbit entry is irreversible' — judgment record, not Gate")
    print("  No blocking, recording only, irreversible")
    print("=" * 70)
    main()
