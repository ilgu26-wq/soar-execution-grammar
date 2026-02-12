#!/usr/bin/env python3
"""
EXP-18a: Gauge Lock v2 — Coordinate System Stabilization
==========================================================
"Orbits become clearer as coordinates are more firmly fixed"

MOTIVATION (from EXP-17):
 OCT was already fast (1.86-2.00 bars).
 But OSS was LOW (0.41-0.47) → orbit fluctuates after commit.
 "What is needed now is not faster detection, but a more stable coordinate frame."

 Parallel coordinate frames (EXP-18b) would cause judgment collapse if done now.
 → The single coordinate frame must be fixed first.

DESIGN: Three stabilization mechanisms

 ① Temporal Lock (temporal alignment):
 - Maintain condition count history per bar
 - Must fire in 67%+ of last 3 bars to qualify as 'stable fire'
 - Momentary firing → treated as coordinate noise

 ② Directional Hysteresis (directional inertia):
 - Only 'dir stable' when force direction is sustained for 2+ consecutive bars
 - Momentary force flip treated as coordinate noise
 - When dir unstable, AOCL condition -1 (conservative handling)

 ③ Orbit Dominance Rule (dominant orbit):
 - Compare FCL fire bars vs AOCL fire bars across the full window
 - Subordinate orbit / dominant orbit ratio < 0.5 → subordinate becomes shadow
 - Shadow event: record only, excluded from orbit determination

 Rules (unchanged):
 - Gate LOCKED
 - Size change ZERO
 - Execution change ZERO
 - Only observation coordinates stabilized

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 PF: 1.28 [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED

 === Stabilized OCT (vs Raw) ===
 FCL OCT: raw=2.00 → stab=2.94 (n=196 vs 200)
 - 50% commit at bar 1 (same as raw)
 - 45.9% at bar 4-5 (temporal lock shifts from bar 3 → bar 4-5)
 AOCL OCT: raw=1.86 → stab=2.73 (n=183 vs 243)
 - 39.9% commit at bar 1 (raw 57.2% → decreased)
 - 60 trades filtered out (unstable commits removed)

 OCT SHIFT ANALYSIS:
 Raw commits quickly by 'instantaneous condition satisfaction'
 Stab requires 'cumulative consistency', slightly slower (+0.87~0.94 bars)
 But the meaning of commit has completely changed

 === Stabilized OSS (key metric) ===
 FCL OSS: raw=0.411 → stab=0.762 (delta=+0.351)
 P50: raw=0.200 → stab=1.000
 >= 0.9: raw=32.0% → stab=63.3%

 AOCL OSS: raw=0.472 → stab=0.657 (delta=+0.184)
 P50: raw=0.200 → stab=1.000
 >= 0.7: raw=42.0% → stab=55.7%

 → FCL OSS: +0.351 improvement (nearly 2x)
 → AOCL OSS: +0.184 improvement
 → P50 from 0.200 → 1.000: median moved to complete stability

 === Orbit Dominance Classification ===
 ALPHA: 98 (33.4%) WR=78.6% PnL=$3,130
 FAILURE: 135 (46.1%) WR= 0.7% PnL=-$3,120
 CONTESTED: 54 (18.4%) WR=68.5% PnL=$1,330
 NEUTRAL: 6 ( 2.0%) WR= 0.0% PnL=-$140

 → Pure ALPHA orbit: 78.6% WR
 → Pure FAILURE orbit: 0.7% WR
 → Separation: 77.9%p (nearly identical to EXP-16's 79.2%p)
 → CONTESTED at 68.5% WR — alpha side dominates even when both fire

 === Both-Orbit Resolution ===
 EXP-16 'Both' trades: 86
 EXP-18a 'CONTESTED' trades: 54
 Resolved by dominance rule: 32 trades (37.2%)
 Total shadow events: 67

 === Directional Stability ===
 Mean dir_stable_bars: 3.55 / 10
 Trades with 0 stable bars: 92 (31.4%)

KEY FINDINGS:
 1. OSS significantly wins: FCL +0.351, AOCL +0.184
 → "coordinates fixed orbit clearly does" Proof

 2. P50 OSS = 1.000
 → In more than half of trades, the opposite orbit completely disappeared after commit

 3. OCT +0.87~0.94 bars increase (2.00→2.94, 1.86→2.73)
 → more not slowonly more certainone/a commit
 → "Knowing for certain rather than knowing quickly"

 4. Dominance Rule resolved 32 of 86 trades (37.2%)
 → Remaining 54 CONTESTED trades truly have 'both properties'

 5. ALPHA orbit WR 78.6%, FAILURE orbit WR 0.7%
 → separationalso Maintained Confirmed

 6. Dir stable bars = 3.55/10
 → Directional inertia is still weak (future observation dimension candidate)

INTERPRETATION:
 "Fixing the coordinates made orbits clearer."

 EXP-17: 'Knows quickly but is unstable' (OSS 0.41)
 EXP-18a: 'Waits a bit longer but is certain' (OSS 0.76)

 This is not a trade-off but a difference in coordinate frame quality.
 Same data, same conditions, same observation — only coordinate fixing changed.
"""

if __name__ == '__main__':
 print("EXP-18a: Gauge Lock v2 — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
