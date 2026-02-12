#!/usr/bin/env python3
"""
EXP-16: Alpha Orbit Commitment Layer (AOCL) — RESULTS
========================================================
"Just because it lost by bad luck does not mean it was not in an alpha orbit"

Twin of FCL. Alpha orbit is recorded irreversibly, just as failure orbit is recorded.

DESIGN:
 Alpha Trajectory Conditions (2+ must fire):
 1. HEALTHY_MOTION: Trade shows healthy movement pattern
 2. HIGH_MFE_MAE: MFE/MAE >= 1.5 (favorable excursion dominates)
 3. FORCE_SUSTAINED: No force sign flip within 3 bars
 4. LOW_INITIAL_ADVERSE: MAE at bar 3 < 2 ticks
 5. HIGH_FORCE_ENTRY: dir_consistency >= 0.65

RULES:
 - Gate LOCKED. AOCL does NOT deny trades.
 - AOCL does NOT amplify weights. Never > 1.0.
 - AOCL only records. Next cycle may reduce penalty dampening.
 - Both FCL and AOCL can fire on the same trade (overlap).

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 PF: 1.28 [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED

 AOCL summary:
 Total trades observed: 582 (RC slice observations)
 Alpha orbit commits: 441 (75.8%)
 Alpha orbit WR: 52.4%
 Won (orbit correct): 231, PnL $10,875
 Lost despite orbit: 210, PnL -$5,000

 Condition frequency:
 FORCE_SUSTAINED: 429
 HIGH_FORCE_ENTRY: 318
 HIGH_MFE_MAE: 246
 HEALTHY_MOTION: 222
 LOW_INITIAL_ADVERSE: 218

 ORBIT CLASSIFICATION (dual orbit):
 Failure orbit only: 58 trades → WR 1.7%, avg PnL -$22.46
 Alpha orbit only: 136 trades → WR 80.9%, avg PnL +$33.47
 Both (overlap): 86 trades → (conflicting signals)
 Neither (unclassified): 13 trades → WR 7.7%, avg PnL -$17.50

BREAKTHROUGH INSIGHT:
 Overall WR = 39.2%
 But ALPHA_ORBIT only WR = 80.9% (+$33.47 avg)
 And FAILURE_ORBIT only WR = 1.7% (-$22.46 avg)

 orbit separationif do grammar weak not... but
 orbitit was merely that orbits had not yet been separated Proofdone/become.

 "Alpha is not a winning trade, but a winning orbit."

ARCHITECTURE (final):
 [ Force / Alpha / Regime ] ← observation
 ↓
 [ Motion Observation ] ← movement quality
 ↓
 [ Orbit Classification ]
 ├─ Failure Orbit → FCL commit (irreversible Record)
 └─ Alpha Orbit → AOCL commit (irreversible accumulation)
 ↓
 [ Proposal Weight Update ] ← delayed, next cycle
 ↓
 [ SOAR v2 Gate (PURE) ] ← survival only (LOCKED)
 ↓
 [ Execution ]
"""

if __name__ == '__main__':
 print("EXP-16: Alpha Orbit Commitment Layer — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
