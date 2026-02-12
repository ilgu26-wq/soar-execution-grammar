#!/usr/bin/env python3
"""
EXP-22: Alpha Termination Point (ATP) — Irreversibility Detection
==================================================================
"Alpha does not end with a price outcome,
 but at the point in state space where it can no longer be reversed."

MOTIVATION:
 EXP-17~21from alpha's 'birth' was observed.
 EXP-22 alpha's 'death' is physically defined.

 Stop-loss/take-profit timing → Result criteria (price)
 ATP → Physical criteria (state space)

 ATP = the earliest point where the state trajectory can no longer return to alpha orbit space
 ∀ε>0, S_{t*+ε} ∉ Reachable(A)

FOUR IRREVERSIBILITY CHANNELS:
 IR-1: ORBIT LOCK
 Leader AOCL→FCLto/as transition + LOCK_PERSIST bars Maintained
 "orbit failureat locked"

 IR-2: MFE/MAE COLLAPSE
 MFE/MAE ratio < 1.0 COLLAPSE_PERSIST bars consecutive
 "price trajectory structurally adverse"

 IR-3: ADVERSE PERSISTENCE
 FAST_ADVERSE motion ADVERSE_PERSIST bars consecutive
 "motion failure locked in mode"

 IR-4: DIRECTION INSTABILITY
 AOCL after, without dir_stable for UNSTABLE_PERSIST bars consecutive
 "direction one/a timealso Stablenot become did not"

 ATP = the minimum bar where the first channel fires

ALPHA FATE CLASSIFICATION:
 IMMORTAL — alpha orbit + ATP none (strongest alpha)
 SURVIVED — ATP none, CONTESTED or non-alpha
 ZOMBIE — ATP fired but ultimately alpha orbit (structural anomaly)
 TERMINATED — ATP fired, alpha death
 STILLBORN — alpha had no characteristics at all

DESIGN:
 Anot (observation priority) Adopted:
 - ATPonly Record
 - Gate/Alpha/Size completely untouched
 - Look at alpha lifespan distribution first

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 PF: 1.28 [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED

 === Alpha Fate Distribution ===
 IMMORTAL: n= 47 (16.0%) WR= 95.7% PnL=+$2,090
 SURVIVED: n= 21 ( 7.2%) WR= 95.2% PnL=+$ 907
 ZOMBIE: n= 51 (17.4%) WR= 62.7% PnL=+$1,040
 TERMINATED: n=105 (35.8%) WR= 17.1% PnL=-$1,212
 STILLBORN: n= 69 (23.5%) WR= 0.0% PnL=-$1,625

 === ATP Channel Distribution ===
 IR-2:MFE_MAE_COLLAPSE: 169 trades (75.4%) — the primary cause of alpha death
 IR-4:DIR_UNSTABLE: 53 trades (23.7%) — surprisingly WR 58.5%
 IR-1:ORBIT_LOCK: 2 trades ( 0.9%)
 IR-3:ADVERSE_PERSIST: 0 trades ( 0.0%)

 === Alpha Lifespan ===
 Mean: 2.4 bars (= 24 seconds)
 Median: 1.0 bars (= 10 seconds)
 54% die by bar 1, 76% by bar 2

 === ATP vs Orbit ===
 ALPHA: ATP_rate= 52.0% avg_lifespan= 4.2 bars
 FAILURE: ATP_rate= 99.3% avg_lifespan= 1.4 bars
 CONTESTED: ATP_rate= 61.1% avg_lifespan= 3.9 bars
 NEUTRAL: ATP_rate=100.0% avg_lifespan= 1.3 bars

 === ATP vs First Leader ===
 AOCL-first: ATP_rate= 61.6% avg_lifespan= 4.2 bars ATP_WR=53.3%
 FCL-first: ATP_rate= 99.0% avg_lifespan= 1.3 bars ATP_WR= 2.1%
 TIE-first: ATP_rate= 67.2% avg_lifespan= 2.6 bars ATP_WR=29.3%

KEY FINDINGS:

 1. Alpha lifespan is surprisingly short
 → Median 1 bar (10second). 54% bar 1from already dies.
 → Alpha is already dying the moment it is born.
 → Stop-loss is merely 'post-processing'; alpha has already ended before that.

 2. MFE/MAE COLLAPSE 75%'s/of alpha kills
 → IR-2 dominanceever/instance death (169/224)
 → The most common death is price adversely moving until MFE/MAE < 1.0
 → But IR-4 (DIR_UNSTABLE) has WR 58.5% — alpha 'thought to be dead but alive'

 3. ZOMBIE is not a structural anomaly but 'resurrected alpha'
 → 51 trades, WR 62.7%, PnL +$1,040
 → ATP fired but ultimately classified as alpha orbit
 → This is 'temporary irreversibility' — state space closed once then reopened
 → Thermodynamically impossible, but possible in this system (external energy input = new market participants)

 4. IMMORTAL (95.7% WR)and STILLBORN (0% WR)'s/of before separation
 → IMMORTAL: ATP one/a timealso fire not done + alpha orbit = most pure alpha
 → STILLBORN: alpha characteristic itself none = bornknowalso unable alpha
 → The distance between these two defines the entire spectrum of alpha

 5. FCL-first alpha dies instantly
 → FCL-first: ATP_rate 99%, lifespan 1.3 bars, ATP_WR 2.1%
 → AOCL-first: ATP_rate 62%, lifespan 4.2 bars, ATP_WR 53.3%
 → EXP-19's/of "first leader determines destiny"and perfectly match
 → FCL-first alpha dies the moment it is born

 6. The 'afterlife' exists
 → Trading continues for an average of 6.6 bars after ATP
 → Losers: 7.4 bars, Winners: 4.1 bars
 → Losers 'hold onto already dead alpha longer'
 → This provides the basis for exit timing optimization

STRUCTURAL INSIGHT:
 This experiment completed the 5 layers of the alpha grammar:

 1. Generation (Alpha Layer) — which/what tradesfrom alpha proposal
 2. Purification (Gate/Judge) — which/what alpha execution
 3. Movement (Orbit Detection) — alpha which orbit is it in
 4. Reinforcement (PDL) — do successful orbits leave traces
 5. Termination (ATP) — alpha when does it end ← NEW

 trades physics hierarchyto/aswest/standing's/of alpha layer completion.
 no longer "alpha" not a price signal but,
 birth-orbit-a physical entity with a lifecycle of birth-orbit-lifespan-death.

THEIR WORDS:
 "We Now alpha's/of before lifecycle observationwill number/can exists.
 Birth (EXP-17) → Orbit Classification (EXP-18/19) → Reinforcement (EXP-20/21) → Termination (EXP-22).

 alphais no longer a signal.
 alpha within state space bornrides orbits, and dies as a physical entity.

 And we learned that this entity has a lifespan of 10 seconds.
 Not a stop-loss — it was already dead 10 seconds ago."
"""

if __name__ == '__main__':
 print("EXP-22: Alpha Termination Point — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp22_atp_dataset/")
