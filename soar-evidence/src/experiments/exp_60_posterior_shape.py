#!/usr/bin/env python3
"""
EXP-60: POSTERIOR SHAPE LEARNING
================================================================
"Not when the mean rises, but when the distribution becomes sharper execution increases."

DESIGN:
  binper/star Beta posterior's/of shape observation:
  1. Sharpness = α + β (total evidence → distribution also)
  2. Variance = αβ / ((α+β)²(α+β+1))
  3. Convergence = 1 - variance * 4  (0=uniform, 1=point mass)
  4. Extremization = |p_exec - 0.5| * 2  (0=undecided, 1=extreme)

  success signal:
  - work/day Sharp Boundaryfrom p_exec > θ interval natural widens
  - Variance reduction (conviction accumulation)
  - Extremization increase (determinationever/instance )

  failure signal:
  - Variance increase (unstable)
  - p_exec ≈ 0.5 interval residual (change)

    determination changedo not does not.
  observationonly does.
"""

import sys, os, json, time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.v2_locked import validate_lock, LOCK_VERSION
from experiments.exp_55_global_stress_test import (
    load_1min_bars, generate_signals_multi,
    run_pipeline_deferred,
)
from experiments.exp_51_cross_market import (
    load_ticks, aggregate_5s,
    compute_shadow_geometry, compute_aep, compute_arg_deny,
    extract_minimal_features, apply_sharp_boundary,
    NumpyEncoder,
)
from observer.learning.p_exec_posterior import BetaPosterior

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
THETA = 0.5


def analyze_bin_shape(alpha, beta):
    total = alpha + beta
    p_exec = alpha / total
    variance = (alpha * beta) / (total ** 2 * (total + 1))
    convergence = max(0, 1 - variance * 4)
    extremization = abs(p_exec - 0.5) * 2
    sharpness = total - 2

    return {
        'p_exec': round(p_exec, 4),
        'alpha': alpha,
        'beta': beta,
        'sharpness': round(sharpness, 1),
        'variance': round(variance, 6),
        'convergence': round(convergence, 4),
        'extremization': round(extremization, 4),
    }


def classify_bin_state(shape):
    if shape['sharpness'] < 3:
        return 'INFANT'
    elif shape['convergence'] > 0.9 and shape['extremization'] > 0.6:
        return 'DECIDED'
    elif shape['convergence'] > 0.7:
        return 'CONVERGING'
    elif shape['extremization'] < 0.2:
        return 'UNDECIDED'
    else:
        return 'LEARNING'


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-60: POSTERIOR SHAPE LEARNING")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'Not when the mean rises, but when the distribution becomes sharper execution increases.'")
    print("=" * 70)

    posterior_path = os.path.join(EVIDENCE_DIR, 'exp57_execution_probability', 'posterior.json')
    if not os.path.exists(posterior_path):
        print("  ❌ No posterior found. Run EXP-57 first.")
        return

    posterior = BetaPosterior()
    posterior.load(posterior_path)

    all_bins = posterior.get_all_bins()
    print(f"\n  Total bins: {len(all_bins)}")

    print(f"\n  ═══ PHASE 1: BIN SHAPE ANALYSIS ═══")

    shapes = {}
    states = {'INFANT': 0, 'DECIDED': 0, 'CONVERGING': 0, 'UNDECIDED': 0, 'LEARNING': 0}

    for key, b in all_bins.items():
        shape = analyze_bin_shape(b['alpha'], b['beta'])
        state = classify_bin_state(shape)
        shape['state'] = state
        shape['n'] = b['n']
        shapes[key] = shape
        states[state] += 1

    print(f"\n  Bin state distribution:")
    for state, count in sorted(states.items(), key=lambda x: -x[1]):
        pct = count / max(len(shapes), 1) * 100
        bar = '█' * int(pct / 2)
        print(f"    {state:>12s}: {count:>3d} ({pct:>5.1f}%) {bar}")

    print(f"\n  ═══ PHASE 2: DECIDED BINS (execution confirmation interval) ═══")

    decided_exec = []
    decided_deny = []

    for key, shape in sorted(shapes.items(), key=lambda x: -x[1]['extremization']):
        if shape['state'] == 'DECIDED':
            if shape['p_exec'] >= THETA:
                decided_exec.append((key, shape))
            else:
                decided_deny.append((key, shape))

    print(f"\n  DECIDED → EXECUTE ({len(decided_exec)} bins):")
    for key, shape in decided_exec[:10]:
        print(f"    {key:>55s}  p={shape['p_exec']:.3f}  n={shape['n']:>4d}  "
              f"conv={shape['convergence']:.3f}  ext={shape['extremization']:.3f}")

    print(f"\n  DECIDED → DENY ({len(decided_deny)} bins):")
    for key, shape in decided_deny[:10]:
        print(f"    {key:>55s}  p={shape['p_exec']:.3f}  n={shape['n']:>4d}  "
              f"conv={shape['convergence']:.3f}  ext={shape['extremization']:.3f}")

    print(f"\n  ═══ PHASE 3: UNDECIDED BINS (change interval) ═══")

    undecided = [(k, s) for k, s in shapes.items() if s['state'] == 'UNDECIDED']
    for key, shape in undecided[:10]:
        print(f"    {key:>55s}  p={shape['p_exec']:.3f}  n={shape['n']:>4d}  "
              f"conv={shape['convergence']:.3f}  ext={shape['extremization']:.3f}")

    if not undecided:
        print(f"    (None — all bins have directionality)")

    print(f"\n  ═══ PHASE 4: CONVERGENCE STATISTICS ═══")

    all_variances = [s['variance'] for s in shapes.values() if s['n'] >= 3]
    all_convergences = [s['convergence'] for s in shapes.values() if s['n'] >= 3]
    all_extremizations = [s['extremization'] for s in shapes.values() if s['n'] >= 3]
    all_sharpness = [s['sharpness'] for s in shapes.values() if s['n'] >= 3]

    if all_variances:
        print(f"  Variance:       mean={np.mean(all_variances):.6f}  std={np.std(all_variances):.6f}")
        print(f"  Convergence:    mean={np.mean(all_convergences):.4f}  std={np.std(all_convergences):.4f}")
        print(f"  Extremization:  mean={np.mean(all_extremizations):.4f}  std={np.std(all_extremizations):.4f}")
        print(f"  Sharpness:      mean={np.mean(all_sharpness):.1f}  max={max(all_sharpness):.1f}")

    print(f"\n  ═══ PHASE 5: EXECUTION EXPANSION POTENTIAL ═══")

    sharp_exec_count = 0
    learned_exec_count = 0
    for key, shape in shapes.items():
        parts = key.split('|')
        if len(parts) == 6:
            e_sign, de_sign, shadow, depth, regime, aep = parts
            from experiments.exp_51_cross_market import apply_sharp_boundary
            mf = [{'e_sign': e_sign, 'de_sign': de_sign, 'shadow_binary': shadow,
                    'arg_depth': depth, 'regime_coarse': regime, 'aep_binary': aep}]
            sharp_p = apply_sharp_boundary(mf)[0]
            if sharp_p >= THETA:
                sharp_exec_count += 1
            if shape['p_exec'] >= THETA:
                learned_exec_count += 1

    print(f"  Sharp Boundary: {sharp_exec_count} exec bins")
    print(f"  Learned p_exec: {learned_exec_count} exec bins")
    expansion = learned_exec_count - sharp_exec_count
    print(f"  Expansion: {expansion:+d} bins")

    if expansion > 0:
        print(f"  ✅ learning execution possible interval expansiondid (natural discovery)")
    elif expansion == 0:
        print(f"  ⬜ execution Same interval — learning only improves precision")
    else:
        print(f"  🔻 execution interval shrinkage — conservative learning progress  (normal)")

    print(f"\n  ═══ PHASE 6: EVOLUTION READINESS ═══")

    ready_for_61 = False
    converging_or_decided = states['DECIDED'] + states['CONVERGING']
    total_active = len([s for s in shapes.values() if s['n'] >= 3])
    maturity_pct = converging_or_decided / max(total_active, 1) * 100

    print(f"  Mature bins (DECIDED + CONVERGING): {converging_or_decided}/{total_active} ({maturity_pct:.0f}%)")

    infant_pct = states['INFANT'] / max(len(shapes), 1) * 100
    undecided_pct = states['UNDECIDED'] / max(len(shapes), 1) * 100

    print(f"  INFANT bins: {states['INFANT']} ({infant_pct:.0f}%)")
    print(f"  UNDECIDED bins: {states['UNDECIDED']} ({undecided_pct:.0f}%)")

    if maturity_pct >= 60 and undecided_pct <= 10:
        ready_for_61 = True
        print(f"\n  ✅ EXP-61 (Bin Coalescence) progress possible")
        print(f"     posterior sufficient maturity. adjacent bin merge  allow.")
    elif maturity_pct >= 40:
        print(f"\n  ⚠️ part maturity. addition datato/as learning persist  EXP-61 progress recommended.")
    else:
        print(f"\n  ❌ maturity. learning continues needed. EXP-61 progress impossible.")

    exp60_dir = os.path.join(EVIDENCE_DIR, 'exp60_posterior_shape')
    os.makedirs(exp60_dir, exist_ok=True)

    exp60_data = {
        'experiment': 'EXP-60 Posterior Shape Learning',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'n_bins_total': len(all_bins),
        'n_bins_active': total_active,
        'state_distribution': states,
        'maturity_pct': round(maturity_pct, 1),
        'ready_for_exp61': ready_for_61,
        'statistics': {
            'variance_mean': round(np.mean(all_variances), 6) if all_variances else None,
            'convergence_mean': round(np.mean(all_convergences), 4) if all_convergences else None,
            'extremization_mean': round(np.mean(all_extremizations), 4) if all_extremizations else None,
            'sharpness_mean': round(np.mean(all_sharpness), 1) if all_sharpness else None,
        },
        'execution_expansion': expansion,
        'decided_exec_bins': [{'key': k, **s} for k, s in decided_exec],
        'decided_deny_bins': [{'key': k, **s} for k, s in decided_deny],
        'all_shapes': {k: v for k, v in sorted(shapes.items(), key=lambda x: -x[1]['n'])},
    }

    with open(os.path.join(exp60_dir, 'posterior_shape_results.json'), 'w') as f:
        json.dump(exp60_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-60 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'conviction accumulate where execution grows.'")


if __name__ == '__main__':
    main()
