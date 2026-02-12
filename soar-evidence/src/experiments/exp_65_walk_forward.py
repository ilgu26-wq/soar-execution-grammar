#!/usr/bin/env python3
"""
EXP-65: WALK-FORWARD CONSTITUTION — flat frame sealeddoes
================================================================
"70/30 one/a time partition onlyany illusion removaland do, true cheap generalization nature/property looks at."

PREREQUISITE:
  EXP-64c — FE gap 13.4%p temporal regime shift.
             observation scheduling/pheromoneto/as do impossible.

DESIGN:
  fixed 70/30 disposal → Expanding Window Walk-Forward

  time west/standing maintained:
    Fold 1: Train [0..N1]     → Test [N1..N2]
    Fold 2: Train [0..N2]     → Test [N2..N3]
    Fold 3: Train [0..N3]     → Test [N3..N4]
    ...
    In each fold: learn posterior from scratch (Train), Test does not update OFF

  output:
    foldper FE_test, SG_test, IMM_test, PnL_test + fold variance

VERDICT CRITERIA:
  GapThis 'always same direction' → true temporal shift (structural limit)
  Gap "folddirection changes each time' → simple sampling noise

  Laws: IMM ≥80%, SG ≥70%p (fold average reference/criteria)

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ Gate/execution/sizing change prohibited
  ✔️ flat frameonly change (walk-forward)
"""

import sys, os, json, time, copy
import numpy as np
from datetime import datetime
from collections import defaultdict

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
from experiments.exp_57_execution_probability import evaluate_boundary
from observer.learning.p_exec_posterior import BetaPosterior

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
THETA = 0.5
N_FOLDS = 5
MIN_TRAIN_FRAC = 0.30
TEST_WINDOW_FRAC = 0.14

np.random.seed(42)


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


def train_posterior_on_range(trades, minimal_features, start, end):
    posterior = BetaPosterior(alpha_prior=1.0, beta_prior=1.0)
    for i in range(start, end):
        mf = minimal_features[i]
        posterior.update(
            e_sign=mf['e_sign'], de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
            is_win=trades[i].get('is_win', False),
        )
    return posterior


def evaluate_range(trades, minimal_features, posterior, start, end, label):
    p_exec_list = []
    for i in range(start, end):
        mf = minimal_features[i]
        p = posterior.get_p_exec(
            e_sign=mf['e_sign'], de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
        )
        p_exec_list.append(p)
    return evaluate_boundary(
        trades[start:end], minimal_features[start:end],
        p_exec_list, THETA, label
    )


def run_walk_forward(trades, minimal_features, market_name, n_folds=N_FOLDS):
    n = len(trades)

    min_train = int(n * MIN_TRAIN_FRAC)
    test_size = max(int(n * TEST_WINDOW_FRAC), 20)

    remaining = n - min_train
    if remaining < test_size * 2:
        actual_folds = 2
        test_size = remaining // 2
    else:
        actual_folds = min(n_folds, remaining // test_size)

    step = remaining // actual_folds if actual_folds > 0 else remaining

    folds = []
    for f in range(actual_folds):
        test_start = min_train + f * step
        test_end = min(test_start + step, n)
        train_start = 0
        train_end = test_start

        if test_end <= test_start or train_end <= train_start:
            continue

        folds.append({
            'fold': f + 1,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_size': train_end - train_start,
            'test_size': test_end - test_start,
        })

    print(f"\n  ── {market_name}: {len(folds)} folds, {n} trades ──")
    for fd in folds:
        print(f"     Fold {fd['fold']}: Train [{fd['train_start']}..{fd['train_end']}] ({fd['train_size']}) → Test [{fd['test_start']}..{fd['test_end']}] ({fd['test_size']})")

    fold_results = []

    for fd in folds:
        posterior = train_posterior_on_range(
            trades, minimal_features, fd['train_start'], fd['train_end']
        )

        train_eval = evaluate_range(
            trades, minimal_features, posterior,
            fd['train_start'], fd['train_end'], f"fold{fd['fold']}_train"
        )
        test_eval = evaluate_range(
            trades, minimal_features, posterior,
            fd['test_start'], fd['test_end'], f"fold{fd['fold']}_test"
        )

        n_bins_active = len(posterior.get_active_bins(min_n=1))
        n_bins_mature = len(posterior.get_active_bins(min_n=20))

        fe_train = train_eval['false_exec_rate'] if train_eval else None
        fe_test = test_eval['false_exec_rate'] if test_eval else None
        fe_gap = abs(fe_test - fe_train) if (fe_train is not None and fe_test is not None) else None
        fe_gap_signed = (fe_test - fe_train) if (fe_train is not None and fe_test is not None) else None

        sg_train = train_eval['sharp_gap'] if train_eval else None
        sg_test = test_eval['sharp_gap'] if test_eval else None
        imm_train = train_eval['immortal_capture_rate'] if train_eval else None
        imm_test = test_eval['immortal_capture_rate'] if test_eval else None

        pnl_train = train_eval['exec_pnl'] if train_eval else None
        pnl_test = test_eval['exec_pnl'] if test_eval else None

        fr = {
            'fold': fd['fold'],
            'train_size': fd['train_size'],
            'test_size': fd['test_size'],
            'n_bins_active': n_bins_active,
            'n_bins_mature': n_bins_mature,
            'fe_train': fe_train,
            'fe_test': fe_test,
            'fe_gap': fe_gap,
            'fe_gap_signed': fe_gap_signed,
            'sg_train': sg_train,
            'sg_test': sg_test,
            'imm_train': imm_train,
            'imm_test': imm_test,
            'pnl_train': pnl_train,
            'pnl_test': pnl_test,
            'train_eval': train_eval,
            'test_eval': test_eval,
        }
        fold_results.append(fr)

        sign = '+' if fe_gap_signed and fe_gap_signed > 0 else ''
        print(f"\n     Fold {fd['fold']}:")
        if train_eval:
            print(f"       Train: Exec={train_eval['n_exec']} FE={fe_train:.1f}% SG={sg_train:+.1f} IMM={imm_train:.1f}% PnL=${pnl_train:,.0f}")
        if test_eval:
            print(f"       Test:  Exec={test_eval['n_exec']} FE={fe_test:.1f}% SG={sg_test:+.1f} IMM={imm_test:.1f}% PnL=${pnl_test:,.0f}")
        if fe_gap is not None:
            print(f"       Gap:   FE {sign}{fe_gap_signed:.1f}%p  |  Bins: {n_bins_active} active, {n_bins_mature} mature")

    fe_tests = [f['fe_test'] for f in fold_results if f['fe_test'] is not None]
    sg_tests = [f['sg_test'] for f in fold_results if f['sg_test'] is not None]
    imm_tests = [f['imm_test'] for f in fold_results if f['imm_test'] is not None]
    pnl_tests = [f['pnl_test'] for f in fold_results if f['pnl_test'] is not None]
    fe_gaps_signed = [f['fe_gap_signed'] for f in fold_results if f['fe_gap_signed'] is not None]

    fe_trains = [f['fe_train'] for f in fold_results if f['fe_train'] is not None]

    summary = {
        'market': market_name,
        'n_trades': n,
        'n_folds': len(folds),
        'fe_test_mean': round(np.mean(fe_tests), 1) if fe_tests else None,
        'fe_test_std': round(np.std(fe_tests), 1) if fe_tests else None,
        'fe_test_min': round(min(fe_tests), 1) if fe_tests else None,
        'fe_test_max': round(max(fe_tests), 1) if fe_tests else None,
        'fe_train_mean': round(np.mean(fe_trains), 1) if fe_trains else None,
        'sg_test_mean': round(np.mean(sg_tests), 1) if sg_tests else None,
        'sg_test_std': round(np.std(sg_tests), 1) if sg_tests else None,
        'imm_test_mean': round(np.mean(imm_tests), 1) if imm_tests else None,
        'imm_test_std': round(np.std(imm_tests), 1) if imm_tests else None,
        'pnl_test_mean': round(np.mean(pnl_tests), 1) if pnl_tests else None,
        'pnl_test_total': round(sum(pnl_tests), 1) if pnl_tests else None,
        'fe_gap_signed_mean': round(np.mean(fe_gaps_signed), 1) if fe_gaps_signed else None,
        'fe_gap_signed_std': round(np.std(fe_gaps_signed), 1) if fe_gaps_signed else None,
        'fe_gap_all_positive': all(g > 0 for g in fe_gaps_signed) if fe_gaps_signed else None,
        'fe_gap_direction_changes': sum(1 for i in range(1, len(fe_gaps_signed)) if fe_gaps_signed[i] * fe_gaps_signed[i-1] < 0) if len(fe_gaps_signed) > 1 else 0,
    }

    print(f"\n  ── {market_name} SUMMARY ──")
    print(f"     FE_test:  mean={summary['fe_test_mean']:.1f}% std={summary['fe_test_std']:.1f}% [{summary['fe_test_min']:.1f}%..{summary['fe_test_max']:.1f}%]")
    print(f"     FE_train: mean={summary['fe_train_mean']:.1f}%")
    print(f"     SG_test:  mean={summary['sg_test_mean']:+.1f} std={summary['sg_test_std']:.1f}")
    print(f"     IMM_test: mean={summary['imm_test_mean']:.1f}% std={summary['imm_test_std']:.1f}%")
    print(f"     PnL_test: total=${summary['pnl_test_total']:,.0f} mean=${summary['pnl_test_mean']:,.0f}")
    print(f"     FE Gap:   mean={summary['fe_gap_signed_mean']:+.1f}%p std={summary['fe_gap_signed_std']:.1f}%p")
    print(f"     Gap direction: {'ALL POSITIVE (test > train)' if summary['fe_gap_all_positive'] else 'MIXED (direction changes: ' + str(summary['fe_gap_direction_changes']) + ')'}")

    return fold_results, summary


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-65: WALK-FORWARD CONSTITUTION")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  '70/30 illusion removaland the stagnation of true generalization performance sealeddoes.'")
    print("=" * 70)

    nq_tick_path = os.path.join(ASSETS, 'NinjaTrader_FullOrderFlow_1770877659150.csv')
    nq_combined_path = os.path.join(ASSETS, 'NQ1_1min_combined.csv')

    all_results = {}

    if os.path.exists(nq_tick_path):
        ticks_df = load_ticks(nq_tick_path)
        nq_5s = aggregate_5s(ticks_df)
        signals = generate_signals_multi(nq_5s, tick_size=0.25)
        trades_tick, _, _ = run_pipeline_deferred(signals, nq_5s, 5.0, 0.25)
        print(f"\n  NQ_Tick_5s: {len(trades_tick)} trades loaded")

        mf_tick = compute_features(trades_tick)
        folds_tick, summary_tick = run_walk_forward(trades_tick, mf_tick, 'NQ_Tick_5s')
        all_results['NQ_Tick_5s'] = {
            'folds': folds_tick,
            'summary': summary_tick,
        }

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None and len(nq_1m) > 200:
            signals = generate_signals_multi(nq_1m, tick_size=0.25)
            trades_1m, _, _ = run_pipeline_deferred(signals, nq_1m, 5.0, 0.25)
            print(f"\n  NQ_1min: {len(trades_1m)} trades loaded")

            mf_1m = compute_features(trades_1m)
            folds_1m, summary_1m = run_walk_forward(trades_1m, mf_1m, 'NQ_1min')
            all_results['NQ_1min'] = {
                'folds': folds_1m,
                'summary': summary_1m,
            }

    combined_trades = []
    combined_mf = []
    for market_key in ['NQ_Tick_5s', 'NQ_1min']:
        if market_key == 'NQ_Tick_5s' and os.path.exists(nq_tick_path):
            combined_trades.extend(trades_tick)
            combined_mf.extend(mf_tick)
        elif market_key == 'NQ_1min' and os.path.exists(nq_combined_path):
            combined_trades.extend(trades_1m)
            combined_mf.extend(mf_1m)

    if combined_trades:
        print(f"\n  Combined: {len(combined_trades)} trades loaded")
        folds_comb, summary_comb = run_walk_forward(combined_trades, combined_mf, 'NQ_Combined')
        all_results['NQ_Combined'] = {
            'folds': folds_comb,
            'summary': summary_comb,
        }

    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  ═══ WALK-FORWARD VERDICT ═══")
    print(f"  ═══════════════════════════════════════════════════")

    for mkt, data in all_results.items():
        s = data['summary']
        print(f"\n  {mkt}:")
        print(f"    FE_test:  {s['fe_test_mean']:.1f}% (std={s['fe_test_std']:.1f}%)")
        print(f"    SG_test:  {s['sg_test_mean']:+.1f} (std={s['sg_test_std']:.1f})")
        print(f"    IMM_test: {s['imm_test_mean']:.1f}% (std={s['imm_test_std']:.1f}%)")

        imm_ok = s['imm_test_mean'] >= 80.0
        sg_ok = s['sg_test_mean'] >= 70.0
        print(f"    IMM ≥80%? {'PASS' if imm_ok else 'FAIL'} ({s['imm_test_mean']:.1f}%)")
        print(f"    SG ≥70?   {'PASS' if sg_ok else 'FAIL'} ({s['sg_test_mean']:+.1f})")

        if s['fe_gap_all_positive']:
            print(f"    Gap pattern: CONSISTENT POSITIVE ({s['fe_gap_signed_mean']:+.1f}%p)")
            print(f"      → TRUE TEMPORAL SHIFT: test FE is systematically higher")
            print(f"      → This is a structural limit, not sampling noise")
        else:
            n_changes = s['fe_gap_direction_changes']
            print(f"    Gap pattern: MIXED ({n_changes} direction change(s), mean={s['fe_gap_signed_mean']:+.1f}%p)")
            if s['fe_gap_signed_std'] > abs(s['fe_gap_signed_mean']):
                print(f"      → SAMPLING NOISE: gap std ({s['fe_gap_signed_std']:.1f}) > |gap mean| ({abs(s['fe_gap_signed_mean']):.1f})")
                print(f"      → The 70/30 gap was an artifact of partition choice")
            else:
                print(f"      → MIXED SIGNAL: some temporal component + noise")

    print(f"\n  ═══ INTERPRETATION FRAMEWORK ═══")
    print(f"  If ALL markets show CONSISTENT POSITIVE gap:")
    print(f"    → Temporal regime shift is real and structural")
    print(f"    → Proceed to EXP-66 (regime-aware posterior)")
    print(f"  If ANY market shows MIXED gap:")
    print(f"    → 70/30 illusion component existence")
    print(f"    → Walk-forward is the correct evaluation, not 70/30")
    print(f"  If fold std(FE) is large:")
    print(f"    → Generalization performance is unstable")
    print(f"    → Need more data or simpler model")

    exp65_dir = os.path.join(EVIDENCE_DIR, 'exp65_walk_forward')
    os.makedirs(exp65_dir, exist_ok=True)

    serializable_results = {}
    for mkt, data in all_results.items():
        clean_folds = []
        for f in data['folds']:
            cf = {k: v for k, v in f.items() if k not in ['train_eval', 'test_eval']}
            cf['train_eval'] = f['train_eval']
            cf['test_eval'] = f['test_eval']
            clean_folds.append(cf)
        serializable_results[mkt] = {
            'folds': clean_folds,
            'summary': data['summary'],
        }

    exp_data = {
        'experiment': 'EXP-65 Walk-Forward Constitution',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'theta': THETA,
        'n_folds_target': N_FOLDS,
        'min_train_frac': MIN_TRAIN_FRAC,
        'test_window_frac': TEST_WINDOW_FRAC,
        'results': serializable_results,
    }

    with open(os.path.join(exp65_dir, 'walk_forward_results.json'), 'w') as f:
        json.dump(exp_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-65 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Discard the illusion of fixed partitioning, keep only structures that survive over time.'")


if __name__ == '__main__':
    main()
