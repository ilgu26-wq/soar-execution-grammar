#!/usr/bin/env python3
"""
EXP-48: SHARP BOUNDARY LEARNING — criticalsurface/if hardening
===================================================
"p_exec smoothly does not raise does not. step functionat near."

❌ ' statesurface/if road probability is high' → Prohibited
✅ ' state already was' → Allowed

DESIGN:
 Bayesian LOO (continuous p) → Deterministic threshold (step function)

 EXECUTE (p=1.0) rules:
 - D0 + NO_SHADOW
 - D0 + POS + RISING
 - D1 + NO_SHADOW + POS
 - D1 + POS + RISING + HIGH AEP (p=0.9)

 DENY (p=0.0) rules:
 - D3+
 - D2 + NEG
 - D2 + POS + SHADOW + FALLING
 - NEG + FALLING + SHADOW

 BOUNDARY (p=0.5): everything else

RESULTS:
 EXECUTE: 90 trades (30.7%) — WR=93.3%, avg_PnL=+9.00
 DENY: 176 trades (60.1%) — WR=8.0%, avg_PnL=-3.84
 BOUNDARY: 27 trades (9.2%) — WR=63.0%, avg_PnL=+4.44

 Sharp Boundary: n=102, WR=89.6%, PF=17.75, Net=$4,290, $/trade=$42.24
 → before experiment highest/best Net ($4,290) AND highest/best WR (89.6%) AND highest/best PF (17.75)

 By Fate:
 IMMORTAL: 46/47 EXECUTE, 0 DENY, 1 BOUNDARY
 STILLBORN: 1/69 EXECUTE, 68 DENY, 0 BOUNDARY
 → Physical threshold classifies Fate almost perfectly

 4/4 SUPPORTED (three/world th (ordinal) before support)

INTERPRETATION:
 trades andever/instancesum not... but physical thresholdis.
 consecutive probability not... but mountain/living Determination.
 "'might win' but 'has already won/died'".
"""

if __name__ == '__main__':
 print("EXP-48: Sharp Boundary Learning")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp48_sharp_boundary/")
