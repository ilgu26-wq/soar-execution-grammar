#!/usr/bin/env python3
"""
EXP-58: WORLD A/B ONLINE TEST
================================================================
"work/day market, work/day time point. trade selectiononly is different."

DESIGN:
  World A: p_exec = 1.0 (all EXECUTE execution — baseline)
  World B: p_exec = learned posterior (θ aboveonly execution)

  work/day condition:
  - signal / alpha / timing work/day
  - pipeline work/day (deferred pruning included)
  - same tradesat different execution filteronly ever/instanceuse

  determination indicator:
  ❌ WR difference (target not)
  ✅ IMMORTAL capture rate
  ✅ False Execute reductionrate
  ✅ Net PnL improvement
  ✅ law conservation
"""

import sys, os, json, time
import numpy as np
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


def compute_features(trades):
    shadow_results = []
    for t in trades:
        traj = t.get('energy_trajectory', [])
        sg = compute_shadow_geometry(traj, t.get('atp_bar'), t.get('alpha_fate', 'UNKNOWN'))
        shadow_results.append(sg if sg else {'shadow_class': 'NO_SHADOW'})
    aep_results = compute_aep(trades)
    arg_results = compute_arg_deny(trades, shadow_results, aep_results)
    minimal_features = extract_minimal_features(trades, arg_results, shadow_results, aep_results)
    return minimal_features


def run_world(trades, minimal_features, p_exec_list, theta, world_name):
    exec_mask = [p >= theta for p in p_exec_list]
    exec_trades = [t for t, m in zip(trades, exec_mask) if m]
    deny_trades = [t for t, m in zip(trades, exec_mask) if not m]
    exec_features = [mf for mf, m in zip(minimal_features, exec_mask) if m]
    deny_features = [mf for mf, m in zip(minimal_features, exec_mask) if not m]

    if not exec_trades:
        return None

    exec_wins = sum(1 for t in exec_trades if t['is_win'])
    exec_wr = exec_wins / len(exec_trades) * 100
    deny_wins = sum(1 for t in deny_trades if t['is_win'])
    deny_wr = deny_wins / max(len(deny_trades), 1) * 100

    false_exec = sum(1 for t in exec_trades if not t['is_win'])
    false_exec_rate = false_exec / len(exec_trades) * 100

    exec_pnl = sum(t['pnl'] for t in exec_trades)

    imm_total = sum(1 for mf in minimal_features if mf['fate'] == 'IMMORTAL')
    imm_exec = sum(1 for mf in exec_features if mf['fate'] == 'IMMORTAL')
    imm_rate = imm_exec / max(imm_total, 1) * 100

    still_total = sum(1 for mf in minimal_features if mf['fate'] == 'STILLBORN')
    still_denied = sum(1 for mf in deny_features if mf['fate'] == 'STILLBORN')
    still_deny_rate = still_denied / max(still_total, 1) * 100

    equity = 100_000
    peak = equity
    max_dd = 0
    for t in exec_trades:
        equity += t['pnl']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        'world': world_name,
        'n_exec': len(exec_trades),
        'n_deny': len(deny_trades),
        'exec_wr': round(exec_wr, 1),
        'deny_wr': round(deny_wr, 1),
        'sharp_gap': round(exec_wr - deny_wr, 1),
        'false_exec_rate': round(false_exec_rate, 1),
        'exec_pnl': round(exec_pnl, 2),
        'immortal_capture_rate': round(imm_rate, 1),
        'stillborn_deny_rate': round(still_deny_rate, 1),
        'max_dd': round(max_dd * 100, 2),
        'final_equity': round(equity, 2),
    }


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-58: WORLD A/B ONLINE TEST")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'work/day market, work/day time point. trade selectiononly is different.'")
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
        if nq_1m is not None:
            datasets.append(('NQ_1min', nq_1m, 0.25, 5.0))

    posterior_path = os.path.join(EVIDENCE_DIR, 'exp57_execution_probability', 'posterior.json')
    if not os.path.exists(posterior_path):
        print("  ❌ No trained posterior found. Run EXP-57 first.")
        return

    posterior = BetaPosterior()
    posterior.load(posterior_path)
    print(f"  Posterior loaded: {len(posterior.get_all_bins())} bins")

    for ds_name, bars_df, tick_size, tick_value in datasets:
        print(f"\n  ═══ {ds_name} A/B TEST ═══")

        signals = generate_signals_multi(bars_df, tick_size=tick_size)
        trades, denied, stats = run_pipeline_deferred(signals, bars_df, tick_value, tick_size)
        print(f"  Trades: {len(trades)}")

        if len(trades) < 20:
            print(f"  ⚠️ INSUFFICIENT TRADES — skipped")
            continue

        minimal_features = compute_features(trades)

        p_exec_baseline = [1.0] * len(trades)

        p_exec_learned = []
        for mf in minimal_features:
            p = posterior.get_p_exec(
                e_sign=mf['e_sign'], de_sign=mf['de_sign'],
                shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
                regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
            )
            p_exec_learned.append(p)

        world_a = run_world(trades, minimal_features, p_exec_baseline, 0.0, 'World_A_Baseline')
        world_b = run_world(trades, minimal_features, p_exec_learned, THETA, 'World_B_Learned')

        for w in [world_a, world_b]:
            if w:
                print(f"\n  ── {w['world']} ──")
                print(f"     EXEC: {w['n_exec']}  DENY: {w['n_deny']}")
                print(f"     Exec WR: {w['exec_wr']:.1f}%  Deny WR: {w['deny_wr']:.1f}%")
                print(f"     Sharp Gap: {w['sharp_gap']:+.1f}%p")
                print(f"     False Exec: {w['false_exec_rate']:.1f}%")
                print(f"     IMMORTAL capture: {w['immortal_capture_rate']:.1f}%")
                print(f"     STILLBORN denied: {w['stillborn_deny_rate']:.1f}%")
                print(f"     PnL: ${w['exec_pnl']:,.0f}  DD: {w['max_dd']:.1f}%")

        if world_a and world_b:
            print(f"\n  ── A/B COMPARISON ──")
            d_fe = world_b['false_exec_rate'] - world_a['false_exec_rate']
            d_imm = world_b['immortal_capture_rate'] - world_a['immortal_capture_rate']
            d_pnl = world_b['exec_pnl'] - world_a['exec_pnl']
            d_dd = world_b['max_dd'] - world_a['max_dd']
            exec_reduction = (1 - world_b['n_exec'] / max(world_a['n_exec'], 1)) * 100

            print(f"     Δ False Execute:    {d_fe:+.1f}%p {'✅' if d_fe < 0 else '⚠️'}")
            print(f"     Δ IMMORTAL capture: {d_imm:+.1f}%p {'✅' if abs(d_imm) < 5 else '⚠️'}")
            print(f"     Δ PnL:              ${d_pnl:+,.0f}")
            print(f"     Δ Max DD:           {d_dd:+.1f}%p {'✅' if d_dd <= 0 else '⚠️'}")
            print(f"     Exec reduction:     {exec_reduction:.1f}%")

            if d_fe < 0 and abs(d_imm) < 10:
                print(f"\n  ✅ World B advantage: False Execute reduction, IMMORTAL conservation")
            elif d_fe < 0:
                print(f"\n  ⚠️ Trade-off: FE reduction but IMMORTAL loss")
            else:
                print(f"\n  ❌ World A advantage: learning maturity or θ adjustment needed")

    exp58_dir = os.path.join(EVIDENCE_DIR, 'exp58_ab_world_test')
    os.makedirs(exp58_dir, exist_ok=True)
    exp58_data = {
        'experiment': 'EXP-58 World A/B Online Test',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'theta': THETA,
        'datasets': [d[0] for d in datasets],
    }
    with open(os.path.join(exp58_dir, 'ab_test_results.json'), 'w') as f:
        json.dump(exp58_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-58 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
