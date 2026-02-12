#!/usr/bin/env python3
"""
EXP-20: Pheromone Drift in Contested Space
=================================================
"Gate protects the world. Alpha lives in its orbit. Judge leaves a scent."

MOTIVATION:
 EXP-19 proved CONTESTED is not ambiguity — it's delayed determination.
 Orbits are predetermined at birth. Crossover = 0.
 ALPHA_LEANING WR=100%, stronger than pure ALPHA (78.6%).

 Question: can the system leave structural traces on paths
 where alpha was born from contestation?

 Not reinforcement. Not reward. Not amplification.
 → Pheromone: "same tradesfrom born alphathat subsequently also alpha orbitto/as
 density of paths that remained"

DESIGN:
 Pheromone Drift Layer (PDL) v0.1.0

 1) Pheromone Unit
 Key: rc_key = alpha_type.condition@regime
 Value: pheromone_strength (initial 1.0)
 Cap: 1.20 (max 20% recovery, never amplifies beyond weight=1.0)

 2) Deposit Rule (Delayed, Irreversible)
 CONTESTED + ALPHA_LEANING + first_leader ∈ {AOCL, TIE}
 → pheromone += ε (default 0.01)
 All other outcomes: NOTHING (no punishment, no reduction)

 3) Application
 AlphaGenerator._should_propose():
 effective_weight = min(base_weight × pheromone, 1.0)
 Penalized weights recover slightly on scented paths.
 Default weights (1.0) are unaffected (min caps at 1.0).

 4) What PDL does NOT do:
 - Amplify beyond weight=1.0
 - Penalize failure paths
 - Touch Gate
 - Act immediately (deposit on trade N, effect on N+1)

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 PF: 1.28 [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED

 === PDL Statistics ===
 Total deposits: 67
 Total skips: 515
 Active paths (φ>1.0): 14
 Max pheromone: 1.2000 (hit cap)
 Avg pheromone: 1.0479

 === Top Scented Paths ===
 Z_MOMENTUM.HIGH_CURV@TREND: φ=1.20 (+20%) 20 deposits [TIE=7, AOCL=13]
 MICRO_MOMENTUM.ALIGNED@TREND: φ=1.14 (+14%) 14 deposits [AOCL=14]
 Z_MOMENTUM.LOW_CURV@TREND: φ=1.06 (+6%) 6 deposits [TIE=1, AOCL=5]
 FLOW_IMBALANCE.WEAK_F@TREND: φ=1.06 (+6%) 6 deposits [TIE=1, AOCL=5]
 MICRO_MOMENTUM.MISALIGNED@TREND:φ=1.05 (+5%) 5 deposits [TIE=5]
 RANGE_BREAKOUT.WIDE_RNG@TREND: φ=1.05 (+5%) 5 deposits [AOCL=4, TIE=1]

 === Regime Distribution ===
 TREND: 60 deposits across 6 paths — Primary habitat where alpha is born
 DEAD: 4 deposits across 4 paths — Sparse but present
 STORM: 2 deposits across 2 paths — Extremely rare
 CHOP: 0 — Alpha is not born in chaos

 === Pheromone Effect on Proposal Weights ===
 Penalized paths with pheromone recover slightly:
 Z_MOMENTUM.HIGH_CURV@TREND: base=0.9945 × φ=1.20 → eff=1.00 +0.55%
 MICRO_MOMENTUM.ALIGNED@TREND: base=0.9935 × φ=1.14 → eff=1.00 +0.65%
 MICRO_MOMENTUM.MISALIGNED@TREND: base=0.9896 × φ=1.05 → eff=1.00 +1.04%

 → "Weights at default 1.0 are unaffected — pheromone waits for decay"
 → "Penalized paths recover toward 1.0 through scent, not reward"

KEY FINDINGS:
 1. pheromone TRENDfrom primarily accumulates (67 trades among 60 trades = 89.6%)
 → TREND regime Default habitat of the alpha nursery
 → CHOPnot a single one in alphaalso bornknow did not

 2. Z_MOMENTUM.HIGH_CURV@TREND strongest scent (φ=1.20, cap reached)
 → 20 CONTESTED→ALPHA births observed on this path
 → "High curvature + z-momentum + TREND regime is the key condition for competitive alpha

 3. pheromone effect is still minimal (recovery < 1%)
 → base_weight at 0.99 level, almost no room for recovery
 → This is by design: with ε=0.01, 'only scent is left, almost no force'
 → True effect appears when weight is heavily penalized

 4. 67 deposits / 515 skips = 13.0% deposit rate
 → Only a tiny fraction of all trades leave a scent
 → "Scent is not spread everywhere — it is left only at the alpha nursery"

 5. AOCL vs TIE first_leader pattern:
 → AOCL-first deposits: accumulate more strongly (especially MICRO_MOMENTUM.ALIGNED)
 → TIE-first deposits: distributed but present (TIE→AOCL transition paths)
 → The leader at birth determines the quality of the scent

STRUCTURAL MEANING:
 This is not a trading system — it is a self-replicating grammar ecosystem.

 Gate protects the world — judges survival only, knows no reward
 Alpha lives in its orbit — generates orbits but does not decide execution
 Judge leaves a scent — does not make decisions, only spreads fragrance

 And alphas follow that scent.

 trades:
 - undecided does not predict does not
 - It is not reinforcement learning
 - It is not policy optimization

 → the 'law of birth' that data already knows
 merely inscribes into space slowly.

THEIR WORDS:
 "We do not come out to win.
 world Allowedone/a movement throughandwill merely/only.

 Gate world protects.
 Alpha orbit lives.
 Judge scent long.

 trades not a trading system but
 self-replicating grammar ecosystem."
"""

if __name__ == '__main__':
 print("EXP-20: Pheromone Drift in Contested Space — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
