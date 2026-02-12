# SOAR System Overview

## Core Concept

SOAR (Structural Observation & Alpha Recognition) is an execution grammar for NQ/MNQ futures that achieves performance through structural filtering — deciding **when NOT to execute** rather than predicting market direction.

The system is built on a single insight: **learning-free intelligence emerges from structure, not optimization.**

## Alpha Lifecycle

An alpha (trading opportunity) follows an irreversible lifecycle:

1. **Birth** — A structural coordinate enters the observable range. Energy is positive, shadow geometry begins forming.
2. **Growth** — Energy accumulates through favorable microstate transitions. The alpha's orbit expands.
3. **Maturity** — The orbit closes (AOC). Win rate peaks. This is the execution window.
4. **Decay** — Energy leaks. The central axis drifts. Shadow accumulation slows.
5. **Termination (ATP)** — "Death is a process, not an event." Once E ≤ 0, near-certain death follows. The alpha cannot be revived.

The system observes this lifecycle but never intervenes in it. Observation is passive; only execution is active.

## Energy Conservation Law (ECL)

Established in EXP-44:

- Energy is conserved within the decision system.
- Only **execution** converts energy to PnL.
- Judgment does not alter energy.
- Reward shaping is forbidden — it would violate conservation.

This means the system cannot "create" alpha through clever optimization. It can only recognize and permit execution of alpha that already exists structurally.

## Shadow Geometry

Discovered in EXP-36~38:

> "Alpha reveals itself through shadow, not light."

Shadow geometry is the indirect observable trace of alpha activity. Rather than measuring alpha directly (which would require future information), the system measures shadows — structural residuals that indicate alpha presence without revealing its magnitude.

Shadow accumulation precedes phase transitions. When shadow density crosses a threshold, the system recognizes a structural regime change.

## Gate: ALLOW / DENY Binary

The EV Gate is a fail-closed binary decision:

- **P = True** + boundary clear → **ALLOW**
- Everything else → **DENY**

The gate evaluates a 5-field EV message: Status (S), Instruction (I), Boundary (B), Permission (P), Criticality (C). The permission bit P is the root — if P is false, nothing else matters.

Gate decisions are not predictions. They are structural permissions.

## Judge Irreversibility

The Judge (JudgeIR) follows three absolute rules:

1. **Input = structural coordinate deltas only.** No energy magnitude, no trends, no performance, no outcomes.
2. **COMMIT is one-way.** Once a judgment is sealed, it cannot be revised. The reasoning hash is immutable.
3. **Results never re-enter Judge input.** No feedback loops. No self-reinforcement.

This irreversibility prevents the chaotic feedback loops that plague adaptive systems. The judge cannot "learn" from its own mistakes because it never sees its own results.

## Boundary-First Execution

Nothing fires without a boundary event. The Boundary module detects phase transitions through three structural indicators:

- **STI** (Structural Transition Index) — Microstate transition selectivity
- **EDG** (Execution Density Gradient) — Entropy of execution distribution
- **MHI** (Memory-History Index) — History-dependent execution bias

Only when a phase transition is detected, energy exceeds threshold, and cooldown has elapsed does the system open an execution window. Everything else is silence.

## CI Wait: 3-Way Decision

Introduced in EXP-69, CI Wait extends the binary EXECUTE/DENY to a 3-way decision:

- **EXECUTE** — Posterior confidence interval is above the upper threshold (85th percentile)
- **DENY** — Posterior confidence interval is below the lower threshold (15th percentile)
- **WAIT** — Confidence interval spans the boundary; insufficient evidence to decide

"Waiting is better than killing" — Waiting is better than killing.

CI Wait dominates all other variance reduction methods (shrinkage, bagging, hysteresis) because it addresses the root cause: uncertain bins should not be forced into binary decisions.

Results: FE_std 5.7%, IMM 84.3%, SG +74.8. CI Wait is the final variance solution.
