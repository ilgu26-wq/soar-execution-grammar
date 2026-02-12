#!/usr/bin/env python3
"""
EXP-59: LEARNING STABILITY TEST (v2 — constitution adjustment ever/instanceuse)
================================================================
"learning law does not erode ?"

STABILITY CONSTITUTION (v2):
  1. most determination window ≥ 200fromonly does
  2. window < 100 violation "noise-only warning"to/as 
  3. rollback composite conditionto/asonly triggered:
     IMMORTAL capture < 80%
     AND False Execute increase
     AND Sharp Gap reduction
     → 3 simultaneous satisfaction timeatonly ROLLBACK

  single indicator judgment prohibited. worldline's/of caseintensity combinationto/as judgmentdoes.

  per/star indicator warning limit (warning, not rollback):
  - Sharp Gap: ≥ 60%p
  - Fate Sep: ≥ 70%p
  - AEP: ≥ 0.7
  - False Exec: ≤ 15%

  Rolling windows: 50, 100, 200 trades
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
    extract_minimal_features, apply_sharp_boundary, measure_invariants,
    NumpyEncoder,
)
from observer.learning.p_exec_posterior import BetaPosterior

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')

STABILITY_LIMITS = {
    'sharp_gap_min': 60.0,
    'fate_sep_min': 70.0,
    'aep_min': 0.7,
    'false_exec_max': 15.0,
}

ROLLBACK_COMPOSITE = {
    'immortal_capture_min': 80.0,
    'false_exec_increase': True,
    'sharp_gap_decrease': True,
}

FINAL_JUDGMENT_WINDOW = 200

WINDOW_SIZES = [50, 100, 200]


def compute_window_invariants(trades_window):
    if len(trades_window) < 10:
        return None

    shadow_results = []
    for t in trades_window:
        traj = t.get('energy_trajectory', [])
        sg = compute_shadow_geometry(traj, t.get('atp_bar'), t.get('alpha_fate', 'UNKNOWN'))
        shadow_results.append(sg if sg else {'shadow_class': 'NO_SHADOW'})

    aep_results = compute_aep(trades_window)
    arg_results = compute_arg_deny(trades_window, shadow_results, aep_results)
    minimal_features = extract_minimal_features(trades_window, arg_results, shadow_results, aep_results)
    sharp_p_exec = apply_sharp_boundary(minimal_features)
    invariants = measure_invariants(minimal_features, sharp_p_exec, aep_results)
    return invariants


def check_stability(invariants, limits):
    violations = []
    if invariants['sharp_gap'] < limits['sharp_gap_min']:
        violations.append(f"SharpGap {invariants['sharp_gap']:.1f} < {limits['sharp_gap_min']}")
    if invariants['fate_separation'] < limits['fate_sep_min']:
        violations.append(f"FateSep {invariants['fate_separation']:.1f} < {limits['fate_sep_min']}")
    if invariants['aep_median'] < limits['aep_min']:
        violations.append(f"AEP {invariants['aep_median']:.4f} < {limits['aep_min']}")
    if invariants['false_exec_rate'] > limits['false_exec_max']:
        violations.append(f"FalseExec {invariants['false_exec_rate']:.1f} > {limits['false_exec_max']}")
    return violations


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-59: LEARNING STABILITY TEST")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'learning law does not erode ?'")
    print("=" * 70)

    nq_combined_path = os.path.join(ASSETS, 'NQ1_1min_combined.csv')
    nq_tick_path = os.path.join(ASSETS, 'NinjaTrader_FullOrderFlow_1770877659150.csv')

    all_trades = []

    if os.path.exists(nq_tick_path):
        ticks_df = load_ticks(nq_tick_path)
        nq_5s = aggregate_5s(ticks_df)
        signals = generate_signals_multi(nq_5s, tick_size=0.25)
        trades, _, _ = run_pipeline_deferred(signals, nq_5s, 5.0, 0.25)
        all_trades.extend(trades)
        print(f"  NQ_Tick_5s: {len(trades)} trades")

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None:
            signals = generate_signals_multi(nq_1m, tick_size=0.25)
            trades, _, _ = run_pipeline_deferred(signals, nq_1m, 5.0, 0.25)
            all_trades.extend(trades)
            print(f"  NQ_1min: {len(trades)} trades")

    print(f"  Total trades: {len(all_trades)}")

    print(f"\n  Stability limits:")
    for k, v in STABILITY_LIMITS.items():
        print(f"    {k}: {v}")

    print(f"\n  ═══ ROLLING WINDOW STABILITY ═══")

    stability_log = []
    rollback_needed = False

    for window_size in WINDOW_SIZES:
        if len(all_trades) < window_size:
            print(f"\n  Window {window_size}: ⚠️ Insufficient trades ({len(all_trades)})")
            continue

        print(f"\n  ── Window: {window_size} trades ──")

        n_windows = max(1, (len(all_trades) - window_size) // (window_size // 2) + 1)
        window_results = []

        for w_idx in range(n_windows):
            start = w_idx * (window_size // 2)
            end = min(start + window_size, len(all_trades))
            if end - start < 20:
                continue

            window_trades = all_trades[start:end]
            inv = compute_window_invariants(window_trades)
            if inv is None:
                continue

            violations = check_stability(inv, STABILITY_LIMITS)
            status = '✅' if not violations else '❌'

            window_results.append({
                'window_idx': w_idx,
                'start': start,
                'end': end,
                'n_trades': len(window_trades),
                'sharp_gap': inv['sharp_gap'],
                'fate_sep': inv['fate_separation'],
                'aep': inv['aep_median'],
                'false_exec': inv['false_exec_rate'],
                'violations': violations,
                'stable': not bool(violations),
            })

            if violations:
                rollback_needed = True
                print(f"    [{w_idx}] trades [{start}:{end}] {status} VIOLATIONS: {violations}")
            else:
                print(f"    [{w_idx}] trades [{start}:{end}] {status} "
                      f"SG={inv['sharp_gap']:+.1f} FS={inv['fate_separation']:+.1f} "
                      f"AEP={inv['aep_median']:.4f} FE={inv['false_exec_rate']:.1f}%")

        stable_count = sum(1 for wr in window_results if wr['stable'])
        total_count = len(window_results)
        print(f"    Stable: {stable_count}/{total_count} windows")

        stability_log.append({
            'window_size': window_size,
            'n_windows': total_count,
            'stable_count': stable_count,
            'stable_pct': round(stable_count / max(total_count, 1) * 100, 1),
            'windows': window_results,
        })

    print(f"\n  ═══ OVERALL STABILITY ANALYSIS ═══")

    if stability_log:
        sharp_gaps = []
        fate_seps = []
        aeps = []
        false_execs = []

        for sl in stability_log:
            for wr in sl['windows']:
                sharp_gaps.append(wr['sharp_gap'])
                fate_seps.append(wr['fate_sep'])
                aeps.append(wr['aep'])
                false_execs.append(wr['false_exec'])

        if sharp_gaps:
            print(f"  Sharp Gap:  mean={np.mean(sharp_gaps):+.1f}  std={np.std(sharp_gaps):.1f}  "
                  f"min={min(sharp_gaps):+.1f}  max={max(sharp_gaps):+.1f}")
            print(f"  Fate Sep:   mean={np.mean(fate_seps):+.1f}  std={np.std(fate_seps):.1f}  "
                  f"min={min(fate_seps):+.1f}  max={max(fate_seps):+.1f}")
            print(f"  AEP:        mean={np.mean(aeps):.4f}  std={np.std(aeps):.4f}  "
                  f"min={min(aeps):.4f}  max={max(aeps):.4f}")
            print(f"  FalseExec:  mean={np.mean(false_execs):.1f}%  std={np.std(false_execs):.1f}%  "
                  f"min={min(false_execs):.1f}%  max={max(false_execs):.1f}%")

    print(f"\n  ═══ COMPOSITE ROLLBACK CHECK (v2 Constitution) ═══")

    large_window_log = [sl for sl in stability_log if sl['window_size'] >= FINAL_JUDGMENT_WINDOW]
    small_window_log = [sl for sl in stability_log if sl['window_size'] < 100]

    large_stable_pct = 0
    large_violation_count = 0
    large_total = 0
    for sl in large_window_log:
        large_total += sl['n_windows']
        large_stable_pct = sl['stable_pct']
        large_violation_count += sl['n_windows'] - sl['stable_count']

    small_violations = 0
    small_total = 0
    for sl in small_window_log:
        small_total += sl['n_windows']
        small_violations += sl['n_windows'] - sl['stable_count']

    print(f"  Window ≥ 200 determination: {large_stable_pct:.0f}% stable ({large_violation_count} violations / {large_total} windows)")
    if small_total > 0:
        print(f"  Window < 100 noise: {small_violations} warnings / {small_total} windows (determination )")

    posterior_path = os.path.join(EVIDENCE_DIR, 'exp57_execution_probability', 'posterior.json')
    composite_rollback = False

    if os.path.exists(posterior_path):
        from experiments.exp_57_execution_probability import train_posterior, evaluate_boundary
        posterior = BetaPosterior()
        posterior.load(posterior_path)

        shadow_results_all = []
        for t in all_trades:
            traj = t.get('energy_trajectory', [])
            sg = compute_shadow_geometry(traj, t.get('atp_bar'), t.get('alpha_fate', 'UNKNOWN'))
            shadow_results_all.append(sg if sg else {'shadow_class': 'NO_SHADOW'})
        aep_results_all = compute_aep(all_trades)
        arg_results_all = compute_arg_deny(all_trades, shadow_results_all, aep_results_all)
        mf_all = extract_minimal_features(all_trades, arg_results_all, shadow_results_all, aep_results_all)
        sharp_p = apply_sharp_boundary(mf_all)

        learned_p = []
        for mf in mf_all:
            p = posterior.get_p_exec(
                e_sign=mf['e_sign'], de_sign=mf['de_sign'],
                shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
                regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
            )
            learned_p.append(p)

        sharp_eval = evaluate_boundary(all_trades, mf_all, sharp_p, 0.5, 'Sharp')
        learned_eval = evaluate_boundary(all_trades, mf_all, learned_p, 0.5, 'Learned')

        if sharp_eval and learned_eval:
            imm_ok = learned_eval['immortal_capture_rate'] >= ROLLBACK_COMPOSITE['immortal_capture_min']
            fe_ok = learned_eval['false_exec_rate'] <= sharp_eval['false_exec_rate']
            sg_ok = learned_eval['sharp_gap'] >= sharp_eval['sharp_gap']

            print(f"\n  Composite health check:")
            print(f"    IMMORTAL capture: {learned_eval['immortal_capture_rate']:.1f}% "
                  f"(min {ROLLBACK_COMPOSITE['immortal_capture_min']}%) {'✅' if imm_ok else '⚠️'}")
            print(f"    FalseExec trend:  {learned_eval['false_exec_rate']:.1f}% vs sharp {sharp_eval['false_exec_rate']:.1f}% "
                  f"{'✅ decreased' if fe_ok else '❌ increased'}")
            print(f"    SharpGap trend:   {learned_eval['sharp_gap']:+.1f} vs sharp {sharp_eval['sharp_gap']:+.1f} "
                  f"{'✅ improved' if sg_ok else '❌ decreased'}")

            if not imm_ok and not fe_ok and not sg_ok:
                composite_rollback = True
                print(f"\n    ❌ ALL THREE CONDITIONS MET → COMPOSITE ROLLBACK TRIGGERED")
            else:
                print(f"\n    ✅ Composite health: PASS (rollback NOT triggered)")

    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  STABILITY VERDICT (v2 Constitution)                            ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")

    if composite_rollback:
        print(f"  ❌ ROLLBACK REQUIRED — composite condition satisfaction")
        print(f"     IMMORTAL < 80% AND FalseExec increase AND SharpGap reduction")
        print(f"     posterior must initialize does.")
    elif large_stable_pct < 80:
        print(f"  ⚠️ WARNING — large window stablenature/property {large_stable_pct:.0f}% (target 80%+)")
        print(f"     rollback non-/fireneeded. learning continues possibleone monitoring reinforcement needed.")
    else:
        print(f"  ✅ STABLE — window ≥ 200 reference/criteria {large_stable_pct:.0f}% stable")
        print(f"     learning progress possible. invention law does not erode did not.")

    rollback_needed = composite_rollback

    exp59_dir = os.path.join(EVIDENCE_DIR, 'exp59_learning_stability')
    os.makedirs(exp59_dir, exist_ok=True)

    exp59_data = {
        'experiment': 'EXP-59 Learning Stability Test',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'stability_limits': STABILITY_LIMITS,
        'n_total_trades': len(all_trades),
        'rollback_needed': rollback_needed,
        'stability_log': stability_log,
        'summary': {
            'sharp_gap': {'mean': round(np.mean(sharp_gaps), 1), 'std': round(np.std(sharp_gaps), 1)} if sharp_gaps else None,
            'fate_sep': {'mean': round(np.mean(fate_seps), 1), 'std': round(np.std(fate_seps), 1)} if fate_seps else None,
            'aep': {'mean': round(np.mean(aeps), 4), 'std': round(np.std(aeps), 4)} if aeps else None,
            'false_exec': {'mean': round(np.mean(false_execs), 1), 'std': round(np.std(false_execs), 1)} if false_execs else None,
        },
    }

    with open(os.path.join(exp59_dir, 'stability_results.json'), 'w') as f:
        json.dump(exp59_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-59 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
