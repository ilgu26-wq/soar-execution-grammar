#!/usr/bin/env python3
"""
EXP-30: Alpha Energy–Axis Drift (ATP before after axis Movement)
===================================================
"alpha when it dies, the axis collapses — is this true?"

MOTIVATION:
 EXP-22(ATP)from alpha's/of irreversible Termination,
 EXP-23(Energy)from energy trajectory,
 EXP-24(Central Axis)from among axis Movement eacheach observation.
 Now three integration: ATP before afterto/as energy among axis how Movementdo?

DESIGN:
 Energy Axis A_E(W):
 one/a winalsoright W's/of tradesat about
 A_E(W) = (1/N) Σ E_i(W) · x_i
 E_i = peak_energy, x_i = RC feature space coordinates (one-hot normalized)
 → "energy where loaded exist"

 Axis Drift Δθ_E:
 consecutive winalsoright between energy axis eachalso change
 Δθ_E = ∠(A_E(W_t), A_E(W_{t+1}))

 ATP Alignment:
 each ATP event reference/criteria [-3, +3] winalsorightfrom
 Δθ_E, E_mean, dE/dt by aligning pattern analysis

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED
 Execution: ZERO changes — observation-only

 === Energy Axis Statistics ===
 Windows: 55 (size=20, step=5)
 Δθ_E mean: 21.38° (axis considerably many moves)
 Δθ_E std: 10.91°
 Δθ_E max: 55.10° (maximum 55°until rotation)
 E_mean range: [5.24, 96.38]
 Global corr(Δθ_E, E_mean): +0.039 (almost independent)

 === ATP-Aligned Patterns ===

 ZOMBIE (n=51):
 Pre-ATP: Δθ_E=21.87° E=+17.91 dE/dt=+0.086
 At-ATP: Δθ_E=19.55° E=+19.48 dE/dt=+0.406 ← energy sharp increase!
 Post-ATP: Δθ_E=20.98° E=+17.25 dE/dt=+0.209
 → ATP point in timefrom dE/dt +0.406to/as jump (energy recovery)
 → axis contraction after Stable (21.87° → 19.55° → 20.98°)

 TERMINATED (n=105):
 Pre-ATP: Δθ_E=21.83° E=+14.91 dE/dt=-0.051 ← death before energy reduction
 At-ATP: Δθ_E=21.38° E=+15.93 dE/dt=+0.012 ← almost stop
 Post-ATP: Δθ_E=21.82° E=+15.97 dE/dt=+0.011
 → axis movement Stable (21.83° → 21.38° → 21.82°)
 → dE/dt death before -0.051from 0to/as convergence

 STILLBORN (n=68):
 Pre-ATP: Δθ_E=22.89° E=+17.16 dE/dt=+0.051
 At-ATP: Δθ_E=18.65° E=+16.98 dE/dt=-0.063
 Post-ATP: Δθ_E=23.41° E=+18.21 dE/dt=-0.039
 → ATPfrom axis contraction (22.89° → 18.65°) after expansion (→ 23.41°)
 → die alpha rather/instead axis shakes

 NON-ATP (baseline):
 Δθ_E=20.73° E=+18.08 dE/dt=+0.148

 === Hypothesis Test ===

 H-30a (Physical Termination — TERMINATED):
 dE/dt at ATP: +0.012 (≥ 0) — NOT SUPPORTED
 → TERMINATED's/of energy collapse ATP 'before'at occurrence (pre: -0.051)
 → ATP point in timeat already energy barapproachesto/as dE/dt ≈ 0
 → "Death does not occur at ATP. ATP is the death Confirmedstand."

 H-30b (Zombie Revival): SUPPORTED ✓
 Post-ATP dE/dt: +0.209 (> 0) — energy recovery ✓
 Post Δθ_E < Pre Δθ_E (20.98° < 21.87°) — axis return ✓
 → ZOMBIE ATP afteratalso energy recovering
 axis original directionto/as returns
 → "ZOMBIEis not resurrection, but something that had not yet died"

 === Axis-Energy Correlation ===
 corr(Δθ_E, dE/dt) = -0.051
 → axis Movementand energy change almost independent
 → energy axis Movement energy total's/of function not... but
 'where energy whether loaded''s/of geometric phenomenon

PHYSICAL INTERPRETATION:

 "alpha when it dies the axis collapses' — partially true.

 more accurately:
 1. TERMINATED: energy first reductionand do (pre dE/dt=-0.051),
 ATP already ended after's/of 'death Confirmed'is.
 axis Stable (change almost none).

 2. ZOMBIE: ATP point in timefrom energy sharp increase (dE/dt=+0.406),
 Axis temporarily contracts then returns to original direction.
 → "die pretending didonly energy existed"

 3. STILLBORN: ATPfrom axis contraction (18.65°)But/However
 after expansion (23.41°). energy reductionand axis expansion simultaneousat.
 → "bornknow unable alpha geometric shakes"

 4. axis Movementand energy independent (corr = -0.051):
 → axis move thing energy total not... but
 energy's/of 'spaceever/instance re-deployment'at 's/ofone/a thing
 → "axis energy is not just positive quantity but positional tracking"

 Conclusion:
 "death death trades not... but andfixedis.
 ATP end not... but Confirmedis.
 ZOMBIEis not resurrection but an incomplete death."
"""

if __name__ == '__main__':
 print("EXP-30: Alpha Energy-Axis Drift — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp30_energy_axis/")
