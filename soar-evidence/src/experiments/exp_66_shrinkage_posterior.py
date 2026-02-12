#!/usr/bin/env python3
"""
EXP-66: HIERARCHICAL SHRINKAGE POSTERIOR — variance reduction 1time batter/other
================================================================
"average is good. variance is the problem."

PREREQUISITE:
  EXP-65 — 70/30 gap = partition artifact.
           Walk-forward raisebar flat frame.
           Real problem: FE_test std=7.9% (variance, not bias)

DESIGN:
  BinShrink per-bin Beta posterior toward the global prior:
    - global prior:  learning data's/of aggregate statistics Beta(a0, b0)
    - bin posterior: Beta(ai, bi)
    - shrunk posterior: Beta(ai + λ*a0, bi + λ*b0)
    → λ when large, bins are pulled toward the global (infant bin stabilization)
    → λ=0 surface/if baseline(contraction none)

  λ sweep: {0, 0.5, 1, 2, 4, 8}

  Walk-forward 5-fold flat (EXP-65 frame use):
    In each fold: posterior learning + shrinkage application + test evaluation

METRICS:
  Primary: FE_test std reduction ( 7.9%, target ≤5%)
  Secondary: FE_test mean maintained or improvement
  Laws: IMM ≥80%, SG ≥70%p (walk-forward average reference/criteria)

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ Gate/execution/sizing change prohibited
  ✔️ posterior computation methodonly change (shrinkage)
"""

import sys, os, json, time, copy, math
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
LAMBDA_VALUES = [0, 0.5, 1, 2, 4, 8]

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


def get_bin_key(mf):
    return f"{mf['e_sign']}|{mf['de_sign']}|{mf['shadow_binary']}|{mf['arg_depth']}|{mf['regime_coarse']}|{mf['aep_binary']}"


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


def compute_global_prior(posterior):
    total_alpha = 0
    total_beta = 0
    n_bins = 0
    for key, b in posterior.bins.items():
        if b['n'] > 0:
            total_alpha += (b['alpha'] - 1.0)
            total_beta += (b['beta'] - 1.0)
            n_bins += 1
    if n_bins == 0:
        return 1.0, 1.0
    a0 = total_alpha / n_bins + 1.0
    b0 = total_beta / n_bins + 1.0
    return a0, b0


def get_shrunk_p_exec(posterior, mf, lam, a0, b0):
    key = get_bin_key(mf)
    b = posterior.bins[key]
    shrunk_alpha = b['alpha'] + lam * (a0 - 1.0)
    shrunk_beta = b['beta'] + lam * (b0 - 1.0)
    return shrunk_alpha / (shrunk_alpha + shrunk_beta)


def evaluate_range_shrunk(trades, minimal_features, posterior, start, end, lam, a0, b0, label):
    p_exec_list = []
    for i in range(start, end):
        mf = minimal_features[i]
        if lam == 0:
            p = posterior.get_p_exec(
                e_sign=mf['e_sign'], de_sign=mf['de_sign'],
                shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
                regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
            )
        else:
            p = get_shrunk_p_exec(posterior, mf, lam, a0, b0)
        p_exec_list.append(p)
    return evaluate_boundary(
        trades[start:end], minimal_features[start:end],
        p_exec_list, THETA, label
    )


def make_folds(n_trades):
    n = n_trades
    min_train = int(n * MIN_TRAIN_FRAC)
    test_size = max(int(n * TEST_WINDOW_FRAC), 20)

    remaining = n - min_train
    if remaining < test_size * 2:
        actual_folds = 2
        test_size = remaining // 2
    else:
        actual_folds = min(N_FOLDS, remaining // test_size)

    step = remaining // actual_folds if actual_folds > 0 else remaining

    folds = []
    for f in range(actual_folds):
        test_start = min_train + f * step
        test_end = min(test_start + step, n)
        train_end = test_start
        if test_end <= test_start or train_end <= 0:
            continue
        folds.append({
            'fold': f + 1,
            'train_start': 0, 'train_end': train_end,
            'test_start': test_start, 'test_end': test_end,
            'train_size': train_end, 'test_size': test_end - test_start,
        })
    return folds


def run_shrinkage_walk_forward(trades, minimal_features, market_name, lam):
    folds = make_folds(len(trades))

    fold_results = []
    for fd in folds:
        posterior = train_posterior_on_range(
            trades, minimal_features, fd['train_start'], fd['train_end']
        )
        a0, b0 = compute_global_prior(posterior)

        train_eval = evaluate_range_shrunk(
            trades, minimal_features, posterior,
            fd['train_start'], fd['train_end'], lam, a0, b0,
            f"fold{fd['fold']}_train"
        )
        test_eval = evaluate_range_shrunk(
            trades, minimal_features, posterior,
            fd['test_start'], fd['test_end'], lam, a0, b0,
            f"fold{fd['fold']}_test"
        )

        fe_train = train_eval['false_exec_rate'] if train_eval else None
        fe_test = test_eval['false_exec_rate'] if test_eval else None
        sg_test = test_eval['sharp_gap'] if test_eval else None
        imm_test = test_eval['immortal_capture_rate'] if test_eval else None
        pnl_test = test_eval['exec_pnl'] if test_eval else None
        n_exec_test = test_eval['n_exec'] if test_eval else 0

        fold_results.append({
            'fold': fd['fold'],
            'fe_train': fe_train,
            'fe_test': fe_test,
            'sg_test': sg_test,
            'imm_test': imm_test,
            'pnl_test': pnl_test,
            'n_exec_test': n_exec_test,
            'train_eval': train_eval,
            'test_eval': test_eval,
        })

    fe_tests = [f['fe_test'] for f in fold_results if f['fe_test'] is not None]
    sg_tests = [f['sg_test'] for f in fold_results if f['sg_test'] is not None]
    imm_tests = [f['imm_test'] for f in fold_results if f['imm_test'] is not None]
    pnl_tests = [f['pnl_test'] for f in fold_results if f['pnl_test'] is not None]
    n_execs = [f['n_exec_test'] for f in fold_results if f['n_exec_test'] is not None]

    summary = {
        'lambda': lam,
        'market': market_name,
        'n_folds': len(folds),
        'fe_test_mean': round(np.mean(fe_tests), 1) if fe_tests else None,
        'fe_test_std': round(np.std(fe_tests), 1) if fe_tests else None,
        'fe_test_values': [round(v, 1) for v in fe_tests],
        'sg_test_mean': round(np.mean(sg_tests), 1) if sg_tests else None,
        'imm_test_mean': round(np.mean(imm_tests), 1) if imm_tests else None,
        'imm_test_std': round(np.std(imm_tests), 1) if imm_tests else None,
        'pnl_test_total': round(sum(pnl_tests), 0) if pnl_tests else None,
        'n_exec_mean': round(np.mean(n_execs), 1) if n_execs else None,
    }

    return fold_results, summary


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-66: HIERARCHICAL SHRINKAGE POSTERIOR")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'average is good. variance is the problem. — contractionto/as stabilizationdoes.'")
    print("=" * 70)

    nq_tick_path = os.path.join(ASSETS, 'NinjaTrader_FullOrderFlow_1770877659150.csv')
    nq_combined_path = os.path.join(ASSETS, 'NQ1_1min_combined.csv')

    datasets = {}

    if os.path.exists(nq_tick_path):
        ticks_df = load_ticks(nq_tick_path)
        nq_5s = aggregate_5s(ticks_df)
        signals = generate_signals_multi(nq_5s, tick_size=0.25)
        trades_tick, _, _ = run_pipeline_deferred(signals, nq_5s, 5.0, 0.25)
        mf_tick = compute_features(trades_tick)
        datasets['NQ_Tick'] = (trades_tick, mf_tick)
        print(f"  NQ_Tick: {len(trades_tick)} trades")

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None and len(nq_1m) > 200:
            signals = generate_signals_multi(nq_1m, tick_size=0.25)
            trades_1m, _, _ = run_pipeline_deferred(signals, nq_1m, 5.0, 0.25)
            mf_1m = compute_features(trades_1m)
            datasets['NQ_1min'] = (trades_1m, mf_1m)
            print(f"  NQ_1min: {len(trades_1m)} trades")

    if 'NQ_Tick' in datasets and 'NQ_1min' in datasets:
        combined_trades = list(datasets['NQ_Tick'][0]) + list(datasets['NQ_1min'][0])
        combined_mf = list(datasets['NQ_Tick'][1]) + list(datasets['NQ_1min'][1])
        datasets['NQ_Combined'] = (combined_trades, combined_mf)
        print(f"  NQ_Combined: {len(combined_trades)} trades")

    all_results = {}

    for mkt_name, (trades, mf) in datasets.items():
        print(f"\n  ═══ {mkt_name}: λ SWEEP ═══")

        mkt_results = {}
        for lam in LAMBDA_VALUES:
            np.random.seed(42)
            folds, summary = run_shrinkage_walk_forward(trades, mf, mkt_name, lam)
            mkt_results[lam] = {
                'folds': folds,
                'summary': summary,
            }

        print(f"\n  {'λ':>4s}  {'FE_mean':>8s}  {'FE_std':>7s}  {'SG':>7s}  {'IMM':>7s}  {'PnL':>10s}  {'Exec/fold':>10s}  FE_folds")
        print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*30}")

        baseline_std = mkt_results[0]['summary']['fe_test_std']

        for lam in LAMBDA_VALUES:
            s = mkt_results[lam]['summary']
            fe_vals = s['fe_test_values']
            fe_str = ', '.join(f"{v:.0f}%" for v in fe_vals)
            std_marker = ''
            if s['fe_test_std'] is not None and baseline_std is not None:
                if s['fe_test_std'] < baseline_std:
                    std_marker = ' ↓'
                elif s['fe_test_std'] > baseline_std:
                    std_marker = ' ↑'
            print(f"  {lam:>4.1f}  {s['fe_test_mean']:>7.1f}%  {s['fe_test_std']:>5.1f}%{std_marker}  {s['sg_test_mean']:>+6.1f}  {s['imm_test_mean']:>6.1f}%  ${s['pnl_test_total']:>9,.0f}  {s['n_exec_mean']:>9.1f}  [{fe_str}]")

        all_results[mkt_name] = mkt_results

    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  ═══ SHRINKAGE VERDICT ═══")
    print(f"  ═══════════════════════════════════════════════════")

    comb = all_results.get('NQ_Combined', all_results.get('NQ_1min', {}))
    if comb:
        baseline_s = comb[0]['summary']
        print(f"\n  Baseline (λ=0): FE_mean={baseline_s['fe_test_mean']:.1f}% FE_std={baseline_s['fe_test_std']:.1f}%")
        print(f"                  SG={baseline_s['sg_test_mean']:+.1f} IMM={baseline_s['imm_test_mean']:.1f}%")

        best_lam = None
        best_score = None
        for lam in LAMBDA_VALUES:
            s = comb[lam]['summary']
            if s['imm_test_mean'] is None or s['imm_test_mean'] < 80.0:
                continue
            if s['sg_test_mean'] is None or s['sg_test_mean'] < 60.0:
                continue
            score = -(s['fe_test_std'] or 999) - 0.1 * (s['fe_test_mean'] or 999)
            if best_score is None or score > best_score:
                best_score = score
                best_lam = lam

        if best_lam is not None:
            best_s = comb[best_lam]['summary']
            std_reduction = baseline_s['fe_test_std'] - best_s['fe_test_std']
            print(f"\n  Best λ={best_lam}: FE_mean={best_s['fe_test_mean']:.1f}% FE_std={best_s['fe_test_std']:.1f}%")
            print(f"                  SG={best_s['sg_test_mean']:+.1f} IMM={best_s['imm_test_mean']:.1f}%")
            print(f"                  FE_std reduction: {std_reduction:+.1f}%p")

            imm_ok = best_s['imm_test_mean'] >= 80.0
            sg_ok = best_s['sg_test_mean'] >= 70.0
            std_reduced = best_s['fe_test_std'] < baseline_s['fe_test_std']
            std_target = best_s['fe_test_std'] <= 5.0

            print(f"\n  Checks:")
            print(f"    IMM ≥80%?         {'✅ PASS' if imm_ok else '❌ FAIL'} ({best_s['imm_test_mean']:.1f}%)")
            print(f"    SG ≥70?           {'✅ PASS' if sg_ok else '❌ FAIL'} ({best_s['sg_test_mean']:+.1f})")
            print(f"    FE_std reduced?   {'✅ PASS' if std_reduced else '❌ FAIL'} ({baseline_s['fe_test_std']:.1f}→{best_s['fe_test_std']:.1f})")
            print(f"    FE_std ≤5%?       {'✅ PASS' if std_target else '❌ FAIL'} ({best_s['fe_test_std']:.1f}%)")

            if imm_ok and sg_ok and std_reduced:
                print(f"\n  ✅ SHRINKAGE EFFECTIVE (λ={best_lam})")
                print(f"     Variance reduced while laws preserved.")
                if std_target:
                    print(f"     Target FE_std ≤5% ACHIEVED.")
                else:
                    print(f"     FE_std still above 5% target — further experiments needed.")
            else:
                print(f"\n  ❌ SHRINKAGE INSUFFICIENT")
                print(f"     Laws violated or variance not reduced.")
        else:
            print(f"\n  ❌ No λ value satisfies law constraints")

        print(f"\n  ── Monotonicity Analysis ──")
        stds = []
        for lam in LAMBDA_VALUES:
            s = comb[lam]['summary']
            stds.append(s['fe_test_std'])
        diffs = [stds[i+1] - stds[i] for i in range(len(stds)-1)]
        monotone_dec = all(d <= 0 for d in diffs)
        monotone_inc = all(d >= 0 for d in diffs)
        print(f"    FE_std by λ: {['%.1f'%s for s in stds]}")
        print(f"    Δ(FE_std):   {['%+.1f'%d for d in diffs]}")
        if monotone_dec:
            print(f"    → MONOTONIC DECREASE: more shrinkage = less variance (diminishing returns check needed)")
        elif monotone_inc:
            print(f"    → MONOTONIC INCREASE: shrinkage hurts variance (unexpected)")
        else:
            print(f"    → NON-MONOTONIC: optimal λ exists between extremes")

    exp66_dir = os.path.join(EVIDENCE_DIR, 'exp66_shrinkage_posterior')
    os.makedirs(exp66_dir, exist_ok=True)

    serializable = {}
    for mkt, mkt_data in all_results.items():
        serializable[mkt] = {}
        for lam, data in mkt_data.items():
            clean_folds = []
            for f in data['folds']:
                cf = {k: v for k, v in f.items() if k not in ['train_eval', 'test_eval']}
                cf['train_eval'] = f['train_eval']
                cf['test_eval'] = f['test_eval']
                clean_folds.append(cf)
            serializable[mkt][str(lam)] = {
                'folds': clean_folds,
                'summary': data['summary'],
            }

    exp_data = {
        'experiment': 'EXP-66 Hierarchical Shrinkage Posterior',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'theta': THETA,
        'lambda_values': LAMBDA_VALUES,
        'n_folds': N_FOLDS,
        'results': serializable,
    }

    with open(os.path.join(exp66_dir, 'shrinkage_results.json'), 'w') as f:
        json.dump(exp_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-66 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Where samples are few, trust the global; where samples are many, trust the local.'")


if __name__ == '__main__':
    main()
