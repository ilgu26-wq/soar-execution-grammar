# SOAR — Structural Observation & Alpha Recognition

> "Making the unseen knowable through language" — Making the unseen knowable through language.

## What This Repo Is

Experimental evidence for the **Irreversible Energy Law of Decision Systems** — proof that learning-free intelligence emerges from structure, not optimization.

SOAR is an NQ/MNQ futures execution grammar that decides **when NOT to execute**, achieving prop/quant-level performance through structural filtering alone.

## Core Discovery

1. **Irreversibility of Judgment** — Once committed, decisions cannot be revised. This prevents chaotic feedback loops.
2. **Energy Conservation Law (ECL)** — Energy is conserved; only execution converts energy to PnL; judgment doesn't alter energy.
3. **Sharp Boundary** — The question "execute or not?" dominates "when to execute?" (EXP-48).
4. **CI Wait** — "Waiting is better than killing" (Waiting is better than killing) — 3-way decision (EXECUTE/DENY/WAIT) via Beta posterior confidence intervals (EXP-69).

## Current Freeze

- **v3_runtime** — NQ ALL PASS (EXP-55 global stress test: 90K+ bars, 2,910 trades)
- **57-bin MDU** — Minimal Distinct Unit, topology sealed (EXP-61 v2a)
- **CI Wait 15/85** — SOFT PASS: IMM 84.3%, SG +74.8, FE_std 5.7%

## Integrity Check (2026-02-12)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Sharp Gap | +76.8%p | ≥ 70%p | ✅ |
| IMM Capture | 87.0% (87/100) | ≥ 80% | ✅ |
| False Execute | 9.0% | ≤ 18% | ✅ |
| CI Wait | 9.6% | ≤ 15% | ✅ |
| Reproducibility Hash | `9ec07cd1a73ae484` | Deterministic | ✅ |

## Phase Map

### Phase 1 — Observation (EXP-01~40)
Pure observation and structural discovery. No learning, no optimization.

| EXP | Title | Key Finding |
|-----|-------|-------------|
| 01-03 | RAW vs SOAR / Gate Ablation / Kill Switch | SOAR grammar structurally reduces DD/streak |
| 04-06 | Core v2 / v1 Overlay / Calibration | v2 SOLO confirmed (PF 1.15 vs 0.94) |
| 07-08 | Prop Deployment / Boundary Sensitivity | Prop-firm compatible gate |
| 09-14 | Alpha Conditions / Proposal / Regime / Motion | Structural weight pipeline |
| 15-16 | Failure Commitment / Alpha Orbit | Irreversible orbit classification |
| 17-20 | Observer Gauge / Gauge Lock / Contested / Pheromone | Observation instrumentation |
| 21 | PDL RL Dataset | Reinforcement learning dataset (sealed) |
| 22 | Alpha Termination Point | "Death is a process, not an event" |
| 23 | Alpha Energy Trajectory | E≤0 once → near-certain death |
| 24 | Central Axis Drift | Axis tracks energy position, not magnitude |
| 25-29 | Census / Interference / SBII / Geometry / Micro Origin | Structural topology mapping |
| 30 | Energy–Axis Drift | Axis position > axis magnitude |
| 31 | Relative Observer Frame | Stability in co-moving frame can be "weightless" |
| 32 | Orbit Closure (AOC) | Closed orbit = high WR |
| 34-35 | Frame Switch / Frame Cost | Stable coordinates can hide information |
| 36-38 | Shadow Geometry / Accumulation / Phase Transition | "Alpha reveals itself through shadow, not light" |
| 39-40 | Relative Gate / ARG Attach | ALLOW = living worldline; attachment triggers phase transition |

### Phase 2 — Link & Learning (EXP-41~64)
Connecting observation to execution through Bayesian learning.

| EXP | Title | Key Finding |
|-----|-------|-------------|
| 41-43 | Threshold / ZOMBIE / Compute Skip | Bayesian p_exec learning + sealed judge calibration |
| 44-45 | ECL Execution / Energy Exit | Energy conservation law: reward shaping forbidden |
| 46 | Observer Learning | Sealed — contributed to judge, not runtime |
| 47 | Minimal Distillation | 12→6 features, performance preserved (+improved) |
| 48-50 | Sharp Boundary / IMMORTAL-tight / Delay | "Execute or not" > "when to execute" |
| 51 | Cross-Market | NQ laws preserved; ES/BTC need calibration |
| 52-54 | Worldline Pruning / Hierarchical / Deferred | "Dead worldlines don't compute" — 47.8% savings |
| 55 | Global Stress Test | 4 markets, 90K+ bars, 2,910 trades — NQ ALL PASS |
| 56 | Market Calibration | FalseExec is structural, not tunable |
| 57 | Execution Probability | Learned p_exec dominates Sharp Boundary |
| 58 | World A/B Test | NQ_Tick FE -40.9%p; NQ_1min PnL +$10,426 |
| 59 | Learning Stability v2 | Constitution updated — final judgment window ≥200 |
| 60 | Posterior Shape | 57 bins: 70.2% maturity, 0 UNDECIDED |
| 61 | Bin Coalescence | ❌ REJECTED — merging borderline bins increases FE |
| 61v2a | Forbidden Band | 57-bin = MDU. "Cannot merge = structure is already optimal" |
| 62 | Confidence-Gated Execution | Non-linear FE curve; IMMORTAL clusters in infant bins |
| 63 | Infant Bin Growth Map | 19% IMMORTAL in infant bins — structural data scarcity |
| 64a | Observation Scheduling | Safe to p=0.25 for mature bins |
| 64b | JR Boundary Compression | ⚠️ PARTIAL — emergence is data-driven |
| 64c | PDL Directed Pheromone | ❌ NO EFFECT — E- events in mature bins only |

### Phase 3 — Variance (EXP-65~70)
Reducing fold-to-fold variance while preserving IMMORTAL capture.

| EXP | Title | Key Finding |
|-----|-------|-------------|
| 65 | Walk-Forward Constitution | 70/30 gap is partition artifact, not temporal shift |
| 66 | Shrinkage Posterior | FE_std↓ but IMM↓ — Pareto frontier mapped |
| 68/68b | Bagged Posterior | NO EFFECT — variance from test fold composition |
| 69 | CI Wait | ✅ DOMINATES shrinkage: FE_std 5.7%, IMM 84.3% |
| 70 | Boundary Hysteresis | NO EFFECT — flip rate 0.0%, no instability exists |

## Reproduction

```bash
cd scripts
bash run_repro.sh
python integrity_check.py
```

## Repository Structure

```
src/core/          v3_runtime execution grammar (production path ONLY)
src/observer/      Offline analysis, posterior learning
src/experiments/   Reproducibility scripts (never imported by runtime)
data/evidence/     Per-experiment JSON/PNG results
docs/              Phase summaries + per-EXP documentation
scripts/           Reproduction and integrity verification
```

**Key invariant**: `src/core/` imports NOTHING from `src/experiments/` or `src/observer/`. Verified by 6-Level Integrity Check.

## Sealed Laws

- **MDU Law** (EXP-61 v2a): 57-bin structure is minimal distinct unit
- **ECL** (EXP-44): Energy conserved; only execution converts to PnL
- **Constitution R1-R5**: Judge sees no energy, curiosity in thinker only, structure at ENTRY only, judge irreversible, all-on-or-collapse

## License

Research use only. Not financial advice.
