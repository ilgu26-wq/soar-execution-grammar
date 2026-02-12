# SOAR Experiment Manifest

> v3_runtime freeze confirmed — 2026-02-12
> 6-Level Integrity Check: ALL PASS
> Reproducibility Hash: `9ec07cd1a73ae484`
> Total: 1,344 decisions (E=469, D=746, W=129)

## Classification

| Category | Definition | Experiments |
|----------|-----------|-------------|
| **A — Runtime Law** | Execution laws, pruning, gate rules. In production path. | EXP-48, 44, 47, 52/53/54 |
| **B — Judge Learning** | Posterior calibration. Sealed — contributed to judge, not runtime. | EXP-41, 42, 43, 46 |
| **C — Observation** | Pure observation/verification. Never delete — physics proof archive. | EXP-01~40, 45, 49, 50, 51 |

## Full Index

| # | Name | Category | Verdict | One-Line |
|---|------|----------|---------|----------|
| 01 | RAW vs SOAR | C | ✅ PASS | SOAR grammar reduces DD/streak |
| 02 | Gate Ablation | C | ✅ PASS | Each gate component contributes |
| 03 | Kill Switch | C | ✅ PASS | Constitution violation → collapse |
| 04 | Core v2 vs v1 | C | ✅ PASS | v2 PF=1.15 / v1 PF=0.94 |
| 05 | v2 with v1 Overlay | C | ❌ REJECTED | v1 overlay kills PnL (73% retention) |
| 06 | Trade Count Calibration | C | ✅ PASS | v2 SOLO confirmed |
| 07 | Prop Deployment Sim | C | ✅ PASS | Prop-firm gate compatible |
| 08 | Boundary Sensitivity | C | ✅ PASS | Boundary parameters robust |
| 09 | Alpha Condition Refinement | C | ✅ PASS | Weight pipeline established |
| 10 | Proposal Shaping | C | ✅ PASS | Low-score candidates filtered |
| 12 | Regime Condition Resolution | C | ✅ PASS | Regime × Condition interaction mapped |
| 13 | Motion Watchdog | C | ✅ PASS | Motion observation instrumented |
| 14 | Motion Penalty | C | ✅ PASS | Motion-based weight penalties added |
| 15 | Failure Commitment | C | ✅ PASS | FCL irreversible classification |
| 16 | Alpha Orbit | C | ✅ PASS | AOCL orbit commitment |
| 17 | Observer Gauge | C | ✅ PASS | Gauge instrumentation |
| 18a | Gauge Lock v2 | C | ✅ PASS | Stabilized orbit evaluation |
| 19 | Contested Micro Orbit | C | ✅ PASS | Contested orbit resolution |
| 20 | Pheromone Drift | C | ✅ PASS | PDL drift measurement |
| 21 | PDL RL Dataset | C | ✅ SEALED | RL dataset generated |
| 22 | Alpha Termination | C | ✅ PASS | ATP: death is a process |
| 23 | Alpha Energy | C | ✅ PASS | E≤0 → near-certain death |
| 24 | Central Axis | C | ✅ PASS | Axis tracks position, not magnitude |
| 25 | Alpha Census | C | ✅ PASS | Population topology mapped |
| 26 | Interference | C | ✅ PASS | Interference patterns catalogued |
| 27 | SBII | C | ✅ PASS | Structure-based info inequality |
| 28 | Geometry Drift | C | ✅ PASS | Geometric drift measured |
| 29 | Micro Origin | C | ✅ PASS | Microstate origin traced |
| 30 | Energy–Axis Drift | C | ✅ PASS | Position > magnitude |
| 31 | Relative Frame | C | ✅ PASS | Co-moving stability = "weightless" |
| 32 | Orbit Closure | C | ✅ PASS | Closed orbit = high WR |
| 34 | Frame Switch | C | ✅ PASS | Stable coords hide info |
| 35 | Frame Cost | C | ✅ PASS | Frame switching has cost |
| 36 | Shadow Geometry | C | ✅ PASS | Shadow reveals alpha |
| 37 | Shadow Accumulation | C | ✅ PASS | Accumulation → phase transition |
| 38 | Phase Transition | C | ✅ PASS | Shadow-driven transitions |
| 39 | Relative Gate Shadow | C | ✅ PASS | ALLOW = living worldline |
| 40 | ARG Attach | C | ✅ PASS | Attachment triggers phase transition |
| 41 | Threshold Learning | B | ✅ SEALED | Bayesian p_exec threshold |
| 42 | ZOMBIE Learning | B | ✅ SEALED | ZOMBIE selection sealed |
| 43 | Computation Skip | B | ✅ SEALED | Dead compute eliminated |
| 44 | ECL Execution | A | ✅ PASS | Energy conservation law |
| 45 | Energy Exit | C | ✅ PASS | Exit mechanics observed |
| 46 | Observer Learning | B | ✅ SEALED | Observer calibration sealed |
| 47 | Minimal Distillation | A | ✅ PASS | 12→6 features preserved |
| 48 | Sharp Boundary | A | ✅ PASS | "Execute or not" dominates |
| 49 | IMMORTAL-tight | C | ✅ PASS | IMMORTAL boundary tightened |
| 50 | Execution Delay | C | ✅ PASS | Delay effect measured |
| 51 | Cross-Market | C | ⚠️ PARTIAL | NQ PASS, ES/BTC need calibration |
| 52 | Worldline Pruning | A | ✅ PASS | Dead worldlines pruned |
| 53 | Hierarchical Pruning | A | ✅ PASS | 3-tier pruning hierarchy |
| 54 | Deferred Compute | A | ✅ PASS | 47.8% orbit savings |
| 55 | Global Stress Test | C | ✅ PASS | 90K+ bars, NQ ALL PASS |
| 56 | Market Calibration | C | ⚠️ PARTIAL | FalseExec independent of pruning |
| 57 | Execution Probability | A | ✅ PASS | Learned p_exec dominates |
| 58 | World A/B Test | C | ✅ PASS | World B: FE -40.9%p |
| 59 | Learning Stability | C | ✅ PASS | Constitution updated, STABLE |
| 60 | Posterior Shape | C | ✅ PASS | 70.2% maturity |
| 61 | Bin Coalescence | C | ❌ REJECTED | Merging increases FE |
| 61v2a | Forbidden Band | C | ❌ SEALED | MDU = 57 bins final |
| 62 | Confidence-Gated | C | ⚠️ PARTIAL | Non-linear FE curve |
| 63 | Infant Bin Map | C | ✅ PASS | 19% IMMORTAL in infants |
| 64a | Observation Scheduling | C | ⚠️ PARTIAL | Safe to p=0.25 |
| 64b | JR Boundary Compression | C | ⚠️ PARTIAL | Emergence is data-driven |
| 64c | PDL Directed Pheromone | C | ❌ NO EFFECT | E- in mature bins only |
| 65 | Walk-Forward Constitution | C | ✅ PASS | 70/30 = partition artifact |
| 66 | Shrinkage Posterior | C | ✅ PASS | Pareto frontier mapped |
| 68 | Bagged Posterior | C | ❌ NO EFFECT | Variance ≠ posterior uncertainty |
| 68b | Bagging × CI Wait | C | ⚠️ SOFT PASS | CI alone better than combined |
| 69 | CI Wait | A | ✅ PASS | DOMINATES shrinkage on both axes |
| 70 | Boundary Hysteresis | C | ❌ NO EFFECT | Flip rate 0.0% |

## Sealed Laws

1. **MDU Law** (EXP-61 v2a): 57-bin structure is minimal distinct unit. No merge/split/redesign.
2. **ECL** (EXP-44): Energy conserved; only execution converts energy to PnL.
3. **Constitution R1-R5**: All laws must be ON or system collapses.
4. **v3_runtime Frozen Laws (Category A)**: EXP-48 (Sharp Boundary), EXP-44 (ECL), EXP-47 (Distillation), EXP-52/53/54 (Pruning).
5. **Bin topology frozen**: Only posterior values (α,β) may evolve via observation.
