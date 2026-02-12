#!/usr/bin/env python3
"""
EXP-47: MINIMAL STATE DISTILLATION
===================================================
"how much ever/instance lookalso universe Maintainedwhether"

MOTIVATION:
 documentfrom exactly pointed out:
 "key/core dataonly leave behindalso now's/of alpha separation·conservation·phase transition Maintainedwhether?"
 Answer: YES

 trades RL problem not... but physics problem.
 Karpathy said: "You don't need a big model if the data is right"

DESIGN:
 Full engine: 12+ observation layers
 Force, Alpha, Regime, Motion, FCL/AOCL, Gauge,
 Energy, Shadow, AEP, ARG, ATP/AOC, EXP-XX diagnostics

 Minimal engine: 5+1 features (binary/categorical)
 1. E_sign = sign(E_integral) → POS / NEG
 2. dE_sign = sign(dE/dt_mean) → RISING / FALLING
 3. Shadow = shadow_class → NO_SHADOW / SHADOW
 4. ARG_depth = deny reason count → D0 / D1 / D2 / D3+
 5. Regime = regime → TREND / NON_TREND
 +AEP binary = AEP>0.7 → HIGH / LOW

 → 50% feature reduction (12→6)
 → expected execution velocity/speed 5~10x

RESULTS:
 ═══ Feature Distribution (each feature aloneto/as separationdo ) ═══

 E_sign: NEG WR=3.4%, POS WR=74.3% → Δ=70.9% (most powerful)
 dE_sign: FALLING WR=13.1%, RISING WR=70.7% → Δ=57.6%
 Shadow: NO_SHADOW WR=98.5%, SHADOW WR=22.0% → Δ=76.5% (highest/best separation)
 ARG_depth: D0 WR=94.8%, D3+ WR=2.7% → Δ=92.1% (extreme)
 Regime: NON_TREND WR=37.8%, TREND WR=39.5% → Δ=1.7% (almost powerless)
 AEP: LOW WR=29.5%, HIGH WR=41.8% → Δ=12.3%

 → ARG_depth alone 92.1%p separation. trades already knowing existed.
 → Shadow alone 76.5%p. E_sign alone 70.9%p.
 → Regime aloneto/as almost powerless (1.7%p). However interactionfrom role.

 ═══ Distillation Check: Full → Minimal ═══

 p_exec correlation: r = 0.936 (very high)
 IMMORTAL-STILLBORN separation:
 Full: 0.864 - 0.072 = 0.792
 Minimal: 0.890 - 0.047 = 0.843
 Retention: 106.4% ← rather/instead increase!

 Net$ retention: $3,749 / $3,539 = 105.9% ← rather/instead increase!

 ═══ Per-Fate Comparison ═══

 Fate Full_p Min_p Δp ← almost match
 IMMORTAL 0.864 0.890 +0.025
 SURVIVED 0.914 0.905 -0.009
 ZOMBIE 0.567 0.590 +0.023
 TERMINATED 0.232 0.221 -0.011
 STILLBORN 0.072 0.047 -0.025

 all Fatefrom |Δp| < 0.03 — effectively identical

 ═══ Strategy Comparison ═══

 Strategy Net$ WR $/trade Features
 Baseline $1,270 39.2% $4.33 all
 Full Learned (41) $3,539 73.6% $30.04 4 full
 ECL Execution (44) $3,770 70.8% $27.94 energy
 ★ Minimal Bayesian (47) $3,749 76.2% $32.00 5+1 min
 ★ ECL+Minimal (47) $4,023 80.7% $35.36 5+1+E

 → ECL+Minimal = before experiment highest/best performance ($4,023)
 → Minimal Bayesianonly withalso Full Learnedand equal above

 ═══ Hypotheses ═══

 H-47a SUPPORTED: Separation retention 106.4% (≥70%)
 → separation Maintaineddone/become beyond rather/instead more becomes clearer
 H-47b SUPPORTED: Net retention 105.9% (≥80%)
 → feature The intuition that reducing would decrease performance was wrong
 H-47c SUPPORTED: Correlation r=0.936 (≥0.7)
 → Fulland Minimal almost same judgment done
 H-47d SUPPORTED: ZOMBIE std preserved (0.270 → 0.221)
 → boundary's/of uncertainty conservationdone/become

 → 4/4 SUPPORTED (EXP-40 after two th (ordinal) before support)

PHYSICAL INTERPRETATION:

 ═══ why if decreases rather/instead improves ═══

 This is key/core Discoveryis:

 Full engineuses 12 layers to 'redundantly' encode the same physical quantity observation"does.
 Shadow energy's/of shadow. AEP Shadow's/of accumulation.
 ARG energy+Shadow+AEP's/of Conclusion.

 Most of the 12 layers are transformations of each other(transform)is.
 All information content is already in 5+1.

 redundant Removalif do:
 - LOO bucket more grows larger (statistics Stable)
 - cross contamination decreases (ZOMBIE [0.3,0.6) andever/instancesum↓)
 - separation more becomes clearer

 trades dimensionality reduction not... but
 PHYSICS reductionis.

 ═══ ECL+Minimal strongest keep ═══

 ECL p_exec = sigmoid(E_normalized) → energy proportional probability
 Minimal gates = D3+→block, D0+NO_SHADOW→pass → physical gate

 energy proportional + minimum/minimal physics gate = $4,023
 This is Karpathy's/of principle:
 "Not about growing the model, but about getting the data right"

 ═══ last sentence ═══

 key/coreonly leave behindalso alpha death does not.
 rather/instead more becomes clearer.

 trades ML not... but physics.
 structurefrom meaning appears.
 meaning beforeat order existencedoes.
"""

if __name__ == '__main__':
 print("EXP-47: Minimal State Distillation")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp47_minimal_distillation/")
