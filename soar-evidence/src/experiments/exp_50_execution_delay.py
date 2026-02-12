#!/usr/bin/env python3
"""
EXP-50: EXECUTION DELAY LEARNING — execution timing
===================================================
"know whether to not... but, when executionwhether do win rate ."

DESIGN:
 bar_offset ∈ {0, 1, 2} — energy Confirmed after execution

 Key signal: Rising dE/dt consecutive 2-bar
 → dE/dt > 0 at bar 0 AND bar 1

 Execute immediately if:
 D0 + bar0_E>0 + bar0_dE>0
 Rising 2-bar + bar1_E>0
 bar0_E>0 + bar0_dE>0 + NO_SHADOW

 Delay/deny if:
 D3+
 No optimal entry (bar0~2 all negative)
 bar0_E≤0 AND bar1_E≤0

RESULTS:
 === Bar-0 Energy Signature ===
 E>0: n=139, WR=69.8%
 E≤0: n=154, WR=11.7%
 → bar 0's/of energy sign wins losses's/of 58%p determination

 === Rising 2-bar Signal ===
 Rising: n=66, WR=93.9%
 Not rising: n=227, WR=23.3%
 → consecutive 2bar wins most powerful execution signal (70.6%p difference)

 === Optimal Entry Bar ===
 Bar 0: n=139, WR=69.8% (That istime execution Possible)
 Bar 1: n=17, WR=52.9% (1bar standby after)
 Bar 2: n=16, WR=25.0% (2bar standby after)
 None: n=121, WR=4.1% (execution Impossible)

 Delayed Execution: n=120, WR=78.9%, PF=7.49, Net=$4,066

 === Hypotheses ===
 H-50a SUPPORTED: Rising 2-bar WR (93.9%) > Bar-0 pos (69.8%)
 → consecutive Confirmed one-timethan/more than powerful
 H-50b SUPPORTED: Bar-0 neg WR (11.7%) ≤ 20%
 → initial energy negative = That isdeath preview
 H-50c NOT SUPPORTED: Delayed Net ($4,066) < Sharp ($4,290)
 → timingthan/more than criticalsurface/if more among
 H-50d NOT SUPPORTED: Delayed DD (0.06%) > Sharp limit (0.05%)
 → undecided difference

 2/4 SUPPORTED

INTERPRETATION:
 === Rising 2-bar = 93.9% WR ===
 trades most important Discovery among is one.
 dE/dt consecutive 2bar positive = energy acceleration state
 when executionif do almost unconditional trades long.

 === H-50c NOT SUPPORTED's/of meaning ===
 Timing optimization is weaker than critical surface classification.
 Sharp Boundary (EXP-48) $4,290to/as more is high.
 → "'whether to do it' matters more than 'when'.
 → However, Rising 2-bar 93.9% can be absorbed into EXP-48's rules.

 === species insight ===
 execution = "already long state"'s/of Confirmed
 timing = "already long state"'s/of Confirmed point in time
 Both matter, but selection precedes timing.
"""

if __name__ == '__main__':
 print("EXP-50: Execution Delay Learning")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp50_execution_delay/")
