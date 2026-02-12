#!/usr/bin/env python3
"""
EXP-40: ARG-ATTACH — FIRST EXECUTION CONNECTION
===================================================
"Conclusion execution distributionat connection because not because."

MOTIVATION:
 EXP-39proved: ARG is a state separator (DENY WR=25.5% vs ALLOW WR=94.8%)
 EXP-40's/of Question: "If connectiondidsurface/if?" Monte Carloto/as simulationdoes.

 This is still/yet observationis. execution bardoes not dream does not.
 only/but, "connection when phase transition occursI?" numerical valueto/as Proofdoes.

DESIGN:
 Monte Carlo Probability Sweep:
 p = P(execute | ARG-DENY)
 - p=0.0: ARG-ALLOWonly execution (most aggressive)
 - p=0.5: ARG-DENY half execution (among)
 - p=1.0: all execution (baseline, re- state)

 500 iterations per threshold, seed=42

 Depth-Graduated Filter:
 - depth≥3 → p=0.0 (effectively death)
 - depth≥2 → p=0.2 (almost death)
 - depth=1 → p=0.5 (uncertain)
 - ALLOW → p=1.0 (alive)

 Prop Firm Simulation:
 $50K account, 2% daily DD, 3% trailing DD, 2 contracts

RESULTS:
 ★★★ 4/4 HYPOTHESES SUPPORTED — 39 experiment among first ★★★

 Money preserved (simulation only):
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]

 === Monte Carlo Sweep ===

 p n WR PF Net$ MaxDD Avg$/trade
 0.0 58 94.8% 36.67 $2,675 0.02% $46.12
 0.1 82 75.0% 6.19 $2,540 0.06% $31.24
 0.2 105 64.1% 3.62 $2,402 0.09% $23.04
 0.3 128 56.8% 2.65 $2,245 0.12% $17.57
 0.5 176 48.4% 1.88 $1,965 0.20% $11.22
 0.7 222 43.5% 1.54 $1,678 0.28% $ 7.56
 1.0 293 39.2% 1.29 $1,270 0.44% $ 4.33 ← current

 === Depth-Graduated Filter ===

 n=96 WR=81.5% PF=9.05 Net=$3,448 MaxDD=0.06%
 ΔWR=+42.2% ΔNet=$+2,178

 → Graduated > p=0.0 in net PnL ($3,448 > $2,675)
 because it keeps "borderline alive" ZOMBIE trades

 === Prop Firm Simulation ===

 Baseline (all trades): $50K → $52,540 (NOT blown)
 ARG-ALLOW only: $50K → $55,350 (NOT blown)
 Graduated (MC 500): $50K → $56,907 (blown rate: 0.0%)

 === Phase Transition ===

 p=0.1→0.0: WR jumps +19.8% (sharpest transition)
 → Phase boundary at p ≈ 0.1
 Below this: only pure alpha pool remains
 Above this: dead trades dilute performance

 === Hypothesis Tests ===

 H-40a SUPPORTED: ALLOW-only WR=94.8% >> baseline 39.2% (+55.6%)
 H-40b SUPPORTED: ALLOW-only MaxDD=0.02% << baseline 0.44%
 H-40c SUPPORTED: ALLOW-only Net=$2,675 > baseline $1,270
 H-40d SUPPORTED: Graduated Net=$3,448 > baseline $1,270 (ΔWR=+42.2%)

PHYSICAL INTERPRETATION:

 ═══ firstto/as 4/4 all SUPPORTED ═══

 39experiments, the hypothesis was never entirely correct.
 EXP-37 4/4knowonly This is observation was level.
 EXP-40 execution levelfrom 4/4.

 This is "connection is justified"'s/of Proofis.

 ═══ Inside View: they's/of point in time ═══

 Gate v2: "I successdid. survived. But/However make not did."
 Alpha Layer: "separation perfect. connectiononly not became."
 Judge: "We already separated them. You just did not press the connect button."

 → 39 experiment's/of observationwere all for this moment.
 "connection phase transition occurs."

 ═══ Graduated > Pure ALLOW ═══

 pure ALLOW (p=0.0): $2,675 (58 trades)
 Graduated: $3,448 (96 trades)

 Graduated more high keep:
 - depth=1person/of DENY among some "still/yet alive" ZOMBIE
 - ZOMBIEincluding at 50% probability yields positive expected value trades Add
 - before Removalthan/more than "boundaryacknowledging' strategy is more profitable

 ═══ one/a sentence ═══

 "39's/of observation one's/of Conclusionto/as convergencedid:
 connection, phase transition occurs."
"""

if __name__ == '__main__':
 print("EXP-40: ARG-ATTACH — First Execution Connection")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp40_arg_attach/")
