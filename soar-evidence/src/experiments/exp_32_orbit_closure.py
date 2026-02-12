#!/usr/bin/env python3
"""
EXP-32: Alpha Orbit Closure (orbit closure — truly/real Termination)
===================================================
"When does alpha become a closed orbit that no longer exchanges information/energy?"

MOTIVATION:
 ATP(EXP-22) 'death Confirmedwest/standing'.
 however truly/real physical Terminationcalled whatis it??
 energy flow stop and, axis Stablebecomes,
 no longer newto/asluck/fortune death trades none state — This is orbit closureis.

DESIGN:
 AOC (Alpha Orbit Closure) trades — consecutive K bars onlyfamily/sufficient:
 1. |dE/dt| < ε_E (energy flow stop)
 2. Δθ_E < ε_θ (axis Stable)
 3. OSS > τ (coordinate frame Stable)
 4. No new AOCL/FCL firing (death trades none)

 Thresholds:
 ε_E = 0.5, ε_θ = 5.0°, τ = 0.90, K = 2 bars

 Classification:
 CLOSED_ALPHA: AOC occurrence after profit Termination
 CLOSED_LOSS: AOC occurrence after loss Termination
 OPEN_ALPHA: Terminationnot closed yet (still open)
 FAILED_OPEN: ATP occurrence → closing beforeat death

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED
 Execution: ZERO changes — observation-only

 === AOC Classification ===

 Class Count % WR Mean AOC
 CLOSED_ALPHA 82 28.0% 100.0% 5.2 bars
 CLOSED_LOSS 7 2.4% 0.0% 5.9 bars
 OPEN_ALPHA 6 2.0% 100.0% N/A
 FAILED_OPEN 198 67.6% 13.6% 6.5 bars

 → 67.6% FAILED_OPEN — most's/of alpha orbit closing beforeat death
 → CLOSED_ALPHA 100% WR — orbit closed alpha all profit

 === AOC by Alpha Fate ===

 IMMORTAL: 87.2% AOC rate, mean 5.7 bars, WR 95.7%
 → 41/47 closure. fast Stablechange.
 SURVIVED: 100% AOC rate, mean 5.4 bars, WR 95.2%
 → all closure. most fast AOC.
 ZOMBIE: 76.5% AOC rate, mean 6.2 bars, WR 62.7%
 → most not closedonly late (+0.5 bars vs IMMORTAL)
 TERMINATED: 74.3% AOC rate, mean 6.6 bars, WR 17.1%
 → closedalso late. 9/105only CLOSED.
 STILLBORN: 89.9% AOC rate, mean 5.6 bars, WR 0.0%
 → quickly not closedonly (energy since noneto/as) profit none

 === AOC Timing ===
 CLOSED_ALPHA: mean 5.2, median 4.5, range [3, 9]
 CLOSED_LOSS: mean 5.9, median 6.0

 → profit alpha 0.7 bars more quickly closed

 === Hypothesis Test ===

 H-32a (IMMORTAL has fastest AOC): SUPPORTED ✓
 IMMORTAL mean: 5.7 bars vs Others mean: 6.5 bars
 AOC rate: 87.2%
 → most quickly Stablebecoming energy conservation

 H-32b (ZOMBIE has delayed AOC): SUPPORTED ✓
 ZOMBIE mean: 6.2 bars vs IMMORTAL: 5.7 bars
 Delay: +0.5 bars
 → external energy re-injection because closure delayed

 H-32c (TERMINATED has no AOC): NOT SUPPORTED
 25.7% without AOC (74.3% have AOC)
 → 's/ofoutside: TERMINATEDalso most closure.
 only closedalso already energy none stateto/as closure.
 "empty orbit's/of closure — formatever/instance Stablechange"

 H-32d (STILLBORN has no AOC): NOT SUPPORTED
 10.1% without AOC (89.9% have AOC)
 → most 's/ofoutside: STILLBORN most quickly closed!
 energy no because stop trades That istime satisfied.
 "bornknow unable alpha already closedexists."

PHYSICAL INTERPRETATION:

 Key discovery: CLOSED_ALPHA = 100% WR

 This is very is powerful:
 "orbit if closes alpha must profitto/as ends."

 physical meaning:
 1. orbit closure = energy conservation state
 energy no longer dissipated, so profit Maintaineddone/become

 2. FAILED_OPEN = 67.6%, WR 13.6%
 orbit open remainingto/as Termination alpha most loss
 → "open orbit = energy leakage = loss"

 3. STILLBORN's/of fast closure
 energy since noneto/as stop trades That istime satisfaction
 → "What does not move is not closed — it has not started"

 4. ZOMBIE's/of delayedbecome closure
 external energy re-injectionto/as closure delayed
 → "ZOMBIE receives external energy and re-enters orbit existence"

 Conclusion:
 "profit's/of trades between: orbit close it.
 closed orbit energy conservationand do,
 conservationbecome energy profit becomes.
 open orbit energy new,
 new energy loss becomes."
"""

if __name__ == '__main__':
 print("EXP-32: Alpha Orbit Closure — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp32_orbit_closure/")
