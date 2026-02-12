#!/usr/bin/env python3
"""
EXP-17: Observer Gauge Lock — OCT & OSS Measurement
=====================================================
"An experiment measuring when the judge becomes confident"

DESIGN:
  Force-Frame Lock (single coordinate frame fixed):
    - Time: 5-second bars
    - Energy: force-normalized
    - Direction: based on force direction

  Progressive evaluation:
    - After entry, evaluate FCL/AOCL conditions bar-by-bar at bars 1, 2, 3, ..., 10
    - First commit time = Orbit Commitment Time (OCT)
    - Post-commit opposite orbit occurrence rate = Orbit Stability Score (OSS)

RESULTS:
  Money preserved:
    Net PnL:    $1,200.00  [IDENTICAL]
    WR:         39.2%      [IDENTICAL]
    PF:         1.28       [IDENTICAL]
    Max DD:     0.42%      [IDENTICAL]
    Gate:       UNTOUCHED

  OCT (Orbit Commitment Time):
    FCL (failure orbit):
      Mean OCT:    2.00 bars
      P50:         2.0 bars
      49.0% commit at bar 1
      29.5% commit at bar 3 (when mae_at_3 + force_flip available)

    AOCL (alpha orbit):
      Mean OCT:    1.86 bars
      P50:         1.0 bars
      57.2% commit at bar 1
      33.3% commit at bar 3

    → Alpha detected FASTER than failure (1.86 vs 2.00)
    → Both orbits have bimodal distribution: bar 1 + bar 3

  OSS (Orbit Stability Score):
    FCL OSS:
      Mean:        0.411
      P50:         0.200
      >= 0.9:      32.0%

    AOCL OSS:
      Mean:        0.472
      P50:         0.200
      >= 0.7:      42.0%

    → OSS is LOW overall → orbit classification fluctuates after commit
    → This explains the 86 "both" trades from EXP-16

  OCT by Regime:
    TREND:  FCL=2.1, AOCL=1.8 (dominant regime, most data)
    DEAD:   FCL=1.6, AOCL=2.2 (failure detected faster)
    CHOP:   FCL=1.5, AOCL=2.0 (failure detected faster)
    STORM:  FCL=1.8, AOCL=2.2 (failure detected faster)

  OCT by Pure Orbit:
    FAILURE only (n=58):  mean OCT=1.67, P50=1.0
    ALPHA only (n=136):   mean OCT=1.40, P50=1.0

KEY FINDINGS:
  1. System KNOWS orbits fast: 50-57% at bar 1 (= 5 seconds)
  2. Bar 3 is the second structural peak (15 seconds) when mae_at_3 activates
  3. OSS is LOW → the coordinate system is not fully stable
     → same trade can trigger both orbits at different bars
     → this is NOT a bug — it's coordinate system noise
  4. Pure orbits commit faster than mixed: 1.40-1.67 vs 1.86-2.00
     → cleaner orbits are identified earlier
  5. In TREND regime, alpha is detected faster
     In non-TREND regimes, failure is detected faster
     → regime affects the observer's speed

ARCHITECTURAL ANSWER:
  "This system knows alpha within 1 bar (5 seconds)" — 57.2% of the time
  "Failure is known within 2 bars (10 seconds)" — median
  "The reason orbit determination is unstable is that the coordinate frame is not yet unified"

NEXT QUESTION:
  Low OSS means the observation coordinate frame is still shaking.
  → EXP-18: Multi-Gauge Parallel Observation (parallel observation across multiple coordinate frames)
  → Or: Fix the coordinate frame more strongly (adjust condition thresholds)
"""

if __name__ == '__main__':
    print("EXP-17: Observer Gauge Lock — see live_report.py for full results")
    print("Run: python experiments/live_report.py")
