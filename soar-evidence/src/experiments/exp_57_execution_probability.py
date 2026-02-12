#!/usr/bin/env python3
"""
EXP-57: BOUNDARY-ANCHORED EXECUTION PROBABILITY LEARNING
================================================================
"win rate targetto/as learningdo not does not. execution probability adjust merely/only."

Phase 3  constitution:
  - rule addition ❌, Manual parameter tuning ❌, intuitive adjustment ❌
  - learning p_execonly adjustment
  - result separation indicatorto/as determination (PnL/WR reference)

DESIGN:
  1. Full pipeline execution → trades generation
  2. Extract minimal features from each trade (E_sign, dE_sign, Shadow, ARG_depth, Regime, AEP)
  3. Sharp Boundary p_exec = anchor (reference/criteria coordinate frame)
  4. Beta posteriorto/as learned p_exec generation
  5. θ = 0.5 reference/criteriato/as EXECUTE/DENY 
  6. Sharp vs Learned separation indicator comparison

LOCKED (never learning prohibited):
  - Gate (irreversibility)
  - Energy computation
  - Sharp Boundary definition (anchorto/asonly use)
  - Fate  reference/criteria
  - AEP definition

LEARNED (thisonly change):
  p_exec = Beta_posterior(E_sign, dE_sign, Shadow, ARG_depth, Regime, AEP_zone)
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.v2_locked import validate_lock, LOCK_VERSION
from experiments.exp_55_global_stress_test import (
    load_1min_bars, generate_signals_multi,
    run_pipeline_deferred, compute_invariants,
)
from experiments.exp_51_cross_market import (
    load_ticks, aggregate_5s,
    compute_shadow_geometry, compute_aep, compute_arg_deny,
    extract_minimal_features, apply_sharp_boundary, measure_invariants,
    NumpyEncoder,
)
from observer.learning.p_exec_posterior import BetaPosterior

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
THETA = 0.5


def train_posterior(trades, posterior):
    shadow_results = []
    for t in trades:
        traj = t.get('energy_trajectory', [])
        sg = compute_shadow_geometry(traj, t.get('atp_bar'), t.get('alpha_fate', 'UNKNOWN'))
        shadow_results.append(sg if sg else {'shadow_class': 'NO_SHADOW'})

    aep_results = compute_aep(trades)
    arg_results = compute_arg_deny(trades, shadow_results, aep_results)
    minimal_features = extract_minimal_features(trades, arg_results, shadow_results, aep_results)

    for i, mf in enumerate(minimal_features):
        posterior.update(
            e_sign=mf['e_sign'],
            de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'],
            arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'],
            aep_zone=mf['aep_binary'],
            is_win=mf['is_win'],
        )

    return minimal_features, shadow_results, aep_results, arg_results


def evaluate_boundary(trades, minimal_features, p_exec_list, theta, label):
    exec_idx = [i for i, p in enumerate(p_exec_list) if p >= theta]
    deny_idx = [i for i, p in enumerate(p_exec_list) if p < theta]

    if not exec_idx:
        return None

    exec_trades = [trades[i] for i in exec_idx]
    deny_trades = [trades[i] for i in deny_idx]

    exec_wins = sum(1 for t in exec_trades if t['is_win'])
    exec_wr = exec_wins / len(exec_trades) * 100
    deny_wins = sum(1 for t in deny_trades if t['is_win']) if deny_trades else 0
    deny_wr = deny_wins / max(len(deny_trades), 1) * 100

    false_exec = sum(1 for t in exec_trades if not t['is_win'])
    false_exec_rate = false_exec / len(exec_trades) * 100

    exec_pnl = sum(t['pnl'] for t in exec_trades)
    deny_pnl = sum(t['pnl'] for t in deny_trades)

    imm_total = sum(1 for mf in minimal_features if mf['fate'] == 'IMMORTAL')
    imm_captured = sum(1 for i in exec_idx if minimal_features[i]['fate'] == 'IMMORTAL')
    imm_capture_rate = imm_captured / max(imm_total, 1) * 100

    return {
        'label': label,
        'n_exec': len(exec_idx),
        'n_deny': len(deny_idx),
        'exec_wr': round(exec_wr, 1),
        'deny_wr': round(deny_wr, 1),
        'sharp_gap': round(exec_wr - deny_wr, 1),
        'false_exec_rate': round(false_exec_rate, 1),
        'exec_pnl': round(exec_pnl, 2),
        'deny_pnl': round(deny_pnl, 2),
        'immortal_capture_rate': round(imm_capture_rate, 1),
        'immortal_total': imm_total,
        'immortal_captured': imm_captured,
    }


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-57: BOUNDARY-ANCHORED EXECUTION PROBABILITY LEARNING")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'win rate targetto/as learningdo not does not. execution probability adjust merely/only.'")
    print("=" * 70)

    nq_tick_path = os.path.join(ASSETS, 'NinjaTrader_FullOrderFlow_1770877659150.csv')
    nq_combined_path = os.path.join(ASSETS, 'NQ1_1min_combined.csv')

    datasets = []

    if os.path.exists(nq_tick_path):
        ticks_df = load_ticks(nq_tick_path)
        nq_5s = aggregate_5s(ticks_df)
        datasets.append(('NQ_Tick_5s', nq_5s, 0.25, 5.0))

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None and len(nq_1m) > 200:
            datasets.append(('NQ_1min', nq_1m, 0.25, 5.0))

    print(f"\n  Datasets: {[d[0] for d in datasets]}")

    posterior = BetaPosterior(alpha_prior=1.0, beta_prior=1.0)

    all_trades = []
    dataset_trades = {}

    print(f"\n  ═══ PHASE 1: PIPELINE EXECUTION ═══")
    for ds_name, bars_df, tick_size, tick_value in datasets:
        signals = generate_signals_multi(bars_df, tick_size=tick_size)
        trades, denied, stats = run_pipeline_deferred(signals, bars_df, tick_value, tick_size)
        print(f"  {ds_name}: {len(trades)} trades ({stats['alive']} alive, {stats['hard_dead']} pruned)")
        all_trades.extend(trades)
        dataset_trades[ds_name] = trades

    print(f"\n  Total trades for learning: {len(all_trades)}")

    print(f"\n  ═══ PHASE 2: POSTERIOR TRAINING ═══")

    minimal_features, shadow_results, aep_results, arg_results = train_posterior(all_trades, posterior)

    active_bins = posterior.get_active_bins(min_n=3)
    print(f"  Active bins (n≥3): {len(active_bins)}")
    print(f"  Total unique bins: {len(posterior.get_all_bins())}")

    print(f"\n  Top bins by p_exec:")
    sorted_bins = sorted(active_bins.items(), key=lambda x: -x[1]['p_exec'])
    for key, val in sorted_bins[:10]:
        print(f"    {key:>55s}  p={val['p_exec']:.3f}  n={val['n']:>4d}")

    print(f"\n  Bottom bins by p_exec:")
    for key, val in sorted_bins[-5:]:
        print(f"    {key:>55s}  p={val['p_exec']:.3f}  n={val['n']:>4d}")

    print(f"\n  ═══ PHASE 3: BOUNDARY COMPARISON ═══")

    sharp_p_exec = apply_sharp_boundary(minimal_features)

    learned_p_exec = []
    for mf in minimal_features:
        p = posterior.get_p_exec(
            e_sign=mf['e_sign'], de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
        )
        learned_p_exec.append(p)

    sharp_result = evaluate_boundary(all_trades, minimal_features, sharp_p_exec, THETA, 'Sharp_Boundary')
    learned_result = evaluate_boundary(all_trades, minimal_features, learned_p_exec, THETA, 'Learned_p_exec')

    for r in [sharp_result, learned_result]:
        if r:
            print(f"\n  ── {r['label']} ──")
            print(f"     EXEC: {r['n_exec']}  DENY: {r['n_deny']}")
            print(f"     Exec WR: {r['exec_wr']:.1f}%  Deny WR: {r['deny_wr']:.1f}%")
            print(f"     Sharp Gap: {r['sharp_gap']:+.1f}%p")
            print(f"     False Execute: {r['false_exec_rate']:.1f}%")
            print(f"     IMMORTAL capture: {r['immortal_captured']}/{r['immortal_total']} ({r['immortal_capture_rate']:.1f}%)")
            print(f"     PnL: EXEC=${r['exec_pnl']:,.0f}  DENY=${r['deny_pnl']:,.0f}")

    print(f"\n  ═══ PHASE 4: COMPARISON VERDICT ═══")

    if sharp_result and learned_result:
        d_gap = learned_result['sharp_gap'] - sharp_result['sharp_gap']
        d_fe = learned_result['false_exec_rate'] - sharp_result['false_exec_rate']
        d_imm = learned_result['immortal_capture_rate'] - sharp_result['immortal_capture_rate']
        d_pnl = learned_result['exec_pnl'] - sharp_result['exec_pnl']

        print(f"  Δ Sharp Gap:        {d_gap:+.1f}%p {'✅' if d_gap > 0 else '⚠️'}")
        print(f"  Δ False Execute:    {d_fe:+.1f}%p {'✅' if d_fe < 0 else '⚠️'}")
        print(f"  Δ IMMORTAL capture: {d_imm:+.1f}%p {'✅' if d_imm >= 0 else '⚠️'}")
        print(f"  Δ Exec PnL:        ${d_pnl:+,.0f} {'✅' if d_pnl > 0 else '⚠️'}")

        improvements = sum([d_gap > 0, d_fe < 0, d_imm >= 0, d_pnl > 0])
        if improvements >= 3:
            print(f"\n  ✅ LEARNED p_exec DOMINATES ({improvements}/4 improvements)")
        elif improvements >= 2:
            print(f"\n  ⚠️ MIXED RESULTS ({improvements}/4 improvements)")
        else:
            print(f"\n  ❌ SHARP BOUNDARY STILL SUPERIOR ({improvements}/4 improvements)")

    print(f"\n  ═══ PHASE 5: FULL INVARIANT CHECK ═══")

    invariants_sharp = compute_invariants(all_trades)
    if invariants_sharp:
        print(f"  Sharp Boundary invariants:")
        print(f"    SharpGap: {invariants_sharp['sharp_gap']:+.1f}%  FateSep: {invariants_sharp['fate_separation']:+.1f}%")
        print(f"    FalseExec: {invariants_sharp['false_exec_rate']:.1f}%  AEP: {invariants_sharp['aep_median']:.4f}")
        print(f"    Laws: {'✅ ALL PASS' if invariants_sharp.get('all_pass') else '❌'}")

    posterior_path = os.path.join(EVIDENCE_DIR, 'exp57_execution_probability', 'posterior.json')
    posterior.save(posterior_path)

    exp57_data = {
        'experiment': 'EXP-57 Boundary-Anchored Execution Probability Learning',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'theta': THETA,
        'n_trades': len(all_trades),
        'n_active_bins': len(active_bins),
        'sharp_result': sharp_result,
        'learned_result': learned_result,
        'comparison': {
            'd_sharp_gap': round(d_gap, 1) if sharp_result and learned_result else None,
            'd_false_exec': round(d_fe, 1) if sharp_result and learned_result else None,
            'd_immortal_capture': round(d_imm, 1) if sharp_result and learned_result else None,
            'd_exec_pnl': round(d_pnl, 2) if sharp_result and learned_result else None,
        },
        'invariants': invariants_sharp,
        'top_bins': dict(list(sorted_bins[:10])),
        'bottom_bins': dict(list(sorted_bins[-5:])),
    }

    exp57_dir = os.path.join(EVIDENCE_DIR, 'exp57_execution_probability')
    os.makedirs(exp57_dir, exist_ok=True)
    with open(os.path.join(exp57_dir, 'exp57_results.json'), 'w') as f:
        json.dump(exp57_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-57 Saved ---")
    print(f"  Posterior: {posterior_path}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'less wrongly executionthis — this is all there is to learning.'")


if __name__ == '__main__':
    main()
