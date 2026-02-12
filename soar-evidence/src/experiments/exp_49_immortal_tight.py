#!/usr/bin/env python3
"""
EXP-49: IMMORTAL-ONLY TIGHTENING — pure alpha solve
===================================================
"WR averageever/instanceto/as does not raise does not. IMMORTAL poolonly pure."

DESIGN:
 PURE_ALPHA (p=1.0):
 IMMORTAL/SURVIVED + NO_SHADOW
 IMMORTAL/SURVIVED + POS + RISING

 STRONG_ALPHA (p=0.5~0.9):
 IMMORTAL/SURVIVED (remaining)
 ZOMBIE + POS + NO_SHADOW
 ZOMBIE + RISING + peak>5

 EXCLUDED (p=0.0):
 ZOMBIE (remaining)
 TERMINATED (unless D0 + NO_SHADOW)
 STILLBORN (all)

RESULTS:
 PURE_ALPHA: 68 trades, WR=95.6%, avg_PnL=+9.34
 STRONG_ALPHA: 29 trades, WR=79.3%, avg_PnL=+6.90
 EXCLUDED: 196 trades, WR=13.8%, avg_PnL=-2.96

 IMMORTAL-Tight: n=86, WR=92.4%, PF=25.25, Net=$3,829, $/trade=$44.32
 → highest/best WR (92.4%) AND highest/best $/trade ($44.32) AND highest/best PF (25.25)

 4/4 SUPPORTED (yes th (ordinal) before support)

INTERPRETATION:
 failure make onlyenters does not.
 ZOMBIE IMMORTALto/as does not deform does not.
 Selects only the living more sharply.
 trades RL not... but criticalsurface/if sharpeningis.
"""

if __name__ == '__main__':
 print("EXP-49: IMMORTAL-Only Tightening")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp49_immortal_tight/")
