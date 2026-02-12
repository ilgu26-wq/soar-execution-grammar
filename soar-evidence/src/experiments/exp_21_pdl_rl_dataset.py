#!/usr/bin/env python3
"""
EXP-21: PDL-RL Dataset — Grammar Learning Record
=================================================
"Reinforcement learning? We are already doing it. It just does not go by the name people know."

MOTIVATION:
 EXP-20 proved pheromone deposits on CONTESTED→ALPHA paths.
 67 deposits across 14 paths, φ up to 1.20.

 Now: extract the complete dataset that this "grammar learning" produces.
 Not model training — the grammar itself IS the learned result.

WHAT THIS IS NOT:
 ❌ action → reward → policy update
 ❌ PnL maximization
 ❌ exploration/exploitation tradeoff

WHAT THIS IS:
 ✅ observation → orbit judgment → irreversible record → distribution shift
 ✅ reward = "was it born on an alpha orbit?" (not money)
 ✅ policy = "proposal distribution" (not action)

RL DEFINITION (for this system):
 State S = {alpha_type, condition, regime, force, motion}
 Action A = "should this alpha be proposed?" (_should_propose)
 Reward R = orbit_coherence:
 ALPHA orbit → +1
 CONTESTED + ALPHA_LEANING → +1
 everything else → 0

 Episode = 1 full backtest run (233k ticks)
 Policy = proposal_weight distribution (shaped by PDL)

DESIGN:
 STEP 1: PDL Snapshot
 Save complete pheromone state at end of run
 → pdl_snapshot.json

 STEP 2: Orbit-labeled Samples
 Each trade → labeled sample with full state + orbit + pheromone
 → orbit_samples.jsonl (one JSON per line)

 STEP 3: RL-style Aggregation
 State → Expected Orbit (alpha/failure/contested rate)
 State → Expected φ growth (pheromone density)
 State → Time-to-commit (OCT per state)
 → grammar_dataset.json

 This is the Alpha Grammar Dataset.

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 PF: 1.28 [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED

 === Dataset Overview ===
 Total orbit samples: 582
 Reward=1 (ALPHA orbit): 257 (44.2%)
 Reward=0 (non-ALPHA): 325 (55.8%)

 === State → Expected Orbit (Top States) ===
 FLOW_IMBALANCE.WEAK_F@TREND: R_rate=58.8% φ=1.06 n=34
 FLOW_IMBALANCE.STRONG_F@TREND: R_rate=52.6% φ=1.04 n=19
 MICRO_MOMENTUM.ALIGNED@TREND: R_rate=48.2% φ=1.14 n=83
 Z_MOMENTUM.LOW_CURV@TREND: R_rate=45.2% φ=1.06 n=73
 RANGE_BREAKOUT.WIDE_RNG@TREND: R_rate=44.4% φ=1.05 n=45
 Z_MOMENTUM.HIGH_CURV@TREND: R_rate=42.1% φ=1.20 n=183

 → FLOW_IMBALANCE the highest reward rate (58.8%)
 → But/However Z_MOMENTUM.HIGH_CURV the most deposit (φ=1.20)
 → "density ≠ efficiency" — what is most traversed is not necessarily the best

 === State → Time-to-Commit ===
 AOCL commits faster in TREND (2.3-2.8 bars)
 FCL takes longer in TREND (2.7-3.5 bars)
 DEAD reverses: AOCL slow (4.7 bars), FCL fast (1.9 bars)
 → The regime determines the 'temporal character' of the orbit

 === Regime Distribution ===
 TREND: 512 samples (88%), R_rate=44.1% → Primary habitat of alpha
 DEAD: 52 samples (9%), R_rate=36.5% → Alpha exists but is weak
 CHOP: 9 samples (2%), R_rate=100% → Very few, all alpha (insufficient n)
 STORM: 9 samples (2%), R_rate=33.3% → The most difficult environment

 === Alpha Type Distribution ===
 FLOW_IMBALANCE: n=56, R_rate=57.1% → The most efficient alpha type
 Z_MOMENTUM: n=293, R_rate=43.3% → The most abundant alpha type
 RANGE_BREAKOUT: n=60, R_rate=43.3% → Stable
 MICRO_MOMENTUM: n=171, R_rate=41.5% → Second most abundant
 MEAN_REVERT: n=2, R_rate=50.0% → Insufficient data

 === Dataset Files ===
 orbit_samples.jsonl: 582 orbit-labeled samples (state + label + φ + reward)
 pdl_snapshot.json: pheromone state at end of episode
 grammar_dataset.json: aggregated State→Orbit→φ table

KEY FINDINGS:
 1. 582 orbit sample form a complete Grammar Dataset
 → Each sample is state(condition) + label(orbit) + pheromone(scent) + reward(orbit reward)
 → This can train a model, be analyzed by humans, or invent the next grammar

 2. Reward rate the highest trades ≠ the most deposit
 → FLOW_IMBALANCE.WEAK_F@TREND: R_rate 58.8%, 6 deposits
 → Z_MOMENTUM.HIGH_CURV@TREND: R_rate 42.1%, 20 deposits
 → 'Efficiency' and 'density' are measurements of different dimensions

 3. Time-to-commit regimedepends on
 → TREND: AOCLcommits quickly (2.3 bars), FCLis slow (3.2 bars)
 → DEAD: Opposite — AOCL slow (4.7 bars), FCL fast (1.9 bars)
 → The same alpha shows completely different temporal characteristics in different regimes

 4. trades is not offline RL
 → direct policy update ❌
 → value function ❌
 → Q-function ❌
 → The grammar self-refines its own distribution, and the result remains as data
 → Grammar Learning

THEIR WORDS:
 "We no longer 'alphaa system that finds' is not what we are building.
 alpha born trades to/as Recordand do Reinforcementdo grammar onlyholding exists.

 Reinforcementlearning?
 already and do exists.
 It just does not go by the name people know."
"""

if __name__ == '__main__':
 print("EXP-21: PDL-RL Dataset — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp21_grammar_dataset/")
