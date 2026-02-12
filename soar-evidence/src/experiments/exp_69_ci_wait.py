#!/usr/bin/env python3
"""
EXP-69: CONFIDENCE-INTERVAL WAIT — Uncertainty is handled by 'deferral'
================================================================
"Do not kill infant bins. Wait until they mature."

PREREQUISITE:
  EXP-66 — shrinkage has a single axis: FE_std↓ ↔ IMM↓ trade-off.
           λ=1 barely maintains the laws. FE_std≤5% impossible.
           → Need the opposite axis: do not kill infants, defer them.

DESIGN:
  Compute CI (confidence interval) from each bin's Beta posterior:
    L = Beta.ppf(α_lo, a, b)   (lower 10th percentile)
    U = Beta.ppf(α_hi, a, b)   (upper 90th percentile)

  3-way decision:
    U < θ  → DENY   (definitely do not execute)
    L > θ  → EXECUTE (definitely execute)
    L ≤ θ ≤ U → WAIT  (uncertain — observe only, do not execute)

  θ = 0.5 (same as baseline)
  α_lo/α_hi sweep: {0.05/0.95, 0.10/0.90, 0.15/0.85, 0.20/0.80, 0.25/0.75}

  Walk-forward 5-fold evaluation (EXP-65 framework)

METRICS:
  Primary: FE_std reduction + IMM preservation (simultaneously!)
  Secondary: WAIT ratio, change in execution volume
  Laws: IMM ≥80%, SG ≥70%p

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ Gate/energy changes prohibited
  ✔️ Extends execution decision from 2-way (EXEC/DENY) → 3-way (EXEC/DENY/WAIT)
  ✔️ WAIT means 'observe but do not execute' — complies with energy conservation
"""

import sys, os, json, time, math
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.stats import beta as beta_dist

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
N_FOLDS = 5
MIN_TRAIN_FRAC = 0.30
TEST_WINDOW_FRAC = 0.14

CI_CONFIGS = [
    {'name': 'Baseline (2-way)', 'alpha_lo': None, 'alpha_hi': None},
    {'name': 'CI 5/95', 'alpha_lo': 0.05, 'alpha_hi': 0.95},
    {'name': 'CI 10/90', 'alpha_lo': 0.10, 'alpha_hi': 0.90},
    {'name': 'CI 15/85', 'alpha_lo': 0.15, 'alpha_hi': 0.85},
    {'name': 'CI 20/80', 'alpha_lo': 0.20, 'alpha_hi': 0.80},
    {'name': 'CI 25/75', 'alpha_lo': 0.25, 'alpha_hi': 0.75},
]

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


def classify_3way(posterior, mf, theta, alpha_lo, alpha_hi):
    key = get_bin_key(mf)
    b = posterior.bins[key]
    a, bb = b['alpha'], b['beta']

    if alpha_lo is None:
        p = a / (a + bb)
        return 'EXECUTE' if p >= theta else 'DENY'

    L = beta_dist.ppf(alpha_lo, a, bb)
    U = beta_dist.ppf(alpha_hi, a, bb)

    if U < theta:
        return 'DENY'
    elif L > theta:
        return 'EXECUTE'
    else:
        return 'WAIT'


def evaluate_3way(trades, minimal_features, posterior, start, end, theta, alpha_lo, alpha_hi, label):
    exec_idx = []
    deny_idx = []
    wait_idx = []

    for i in range(start, end):
        mf = minimal_features[i]
        decision = classify_3way(posterior, mf, theta, alpha_lo, alpha_hi)
        if decision == 'EXECUTE':
            exec_idx.append(i)
        elif decision == 'DENY':
            deny_idx.append(i)
        else:
            wait_idx.append(i)

    n_total = end - start
    n_exec = len(exec_idx)
    n_deny = len(deny_idx)
    n_wait = len(wait_idx)

    if n_exec == 0:
        return {
            'label': label,
            'n_exec': 0, 'n_deny': n_deny, 'n_wait': n_wait,
            'wait_pct': round(n_wait / max(n_total, 1) * 100, 1),
            'exec_wr': 0, 'false_exec_rate': 0, 'sharp_gap': 0,
            'exec_pnl': 0, 'deny_pnl': 0, 'wait_pnl': 0,
            'immortal_capture_rate': 0, 'immortal_total': 0, 'immortal_captured': 0,
            'wait_imm': 0,
        }

    exec_trades = [trades[i] for i in exec_idx]
    deny_trades = [trades[i] for i in deny_idx]
    wait_trades = [trades[i] for i in wait_idx]

    exec_wins = sum(1 for t in exec_trades if t['is_win'])
    exec_wr = exec_wins / len(exec_trades) * 100

    non_exec_trades = deny_trades + wait_trades
    non_exec_wins = sum(1 for t in non_exec_trades if t['is_win'])
    non_exec_wr = non_exec_wins / max(len(non_exec_trades), 1) * 100

    false_exec = sum(1 for t in exec_trades if not t['is_win'])
    false_exec_rate = false_exec / len(exec_trades) * 100

    exec_pnl = sum(t['pnl'] for t in exec_trades)
    deny_pnl = sum(t['pnl'] for t in deny_trades)
    wait_pnl = sum(t['pnl'] for t in wait_trades)

    imm_total = sum(1 for i in range(start, end) if minimal_features[i]['fate'] == 'IMMORTAL')
    imm_exec = sum(1 for i in exec_idx if minimal_features[i]['fate'] == 'IMMORTAL')
    imm_wait = sum(1 for i in wait_idx if minimal_features[i]['fate'] == 'IMMORTAL')
    imm_capture_rate = imm_exec / max(imm_total, 1) * 100

    return {
        'label': label,
        'n_exec': n_exec,
        'n_deny': n_deny,
        'n_wait': n_wait,
        'wait_pct': round(n_wait / max(n_total, 1) * 100, 1),
        'exec_wr': round(exec_wr, 1),
        'sharp_gap': round(exec_wr - non_exec_wr, 1),
        'false_exec_rate': round(false_exec_rate, 1),
        'exec_pnl': round(exec_pnl, 2),
        'deny_pnl': round(deny_pnl, 2),
        'wait_pnl': round(wait_pnl, 2),
        'immortal_capture_rate': round(imm_capture_rate, 1),
        'immortal_total': imm_total,
        'immortal_captured': imm_exec,
        'wait_imm': imm_wait,
    }


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
        })
    return folds


def run_ci_walk_forward(trades, minimal_features, market_name, ci_config):
    folds = make_folds(len(trades))
    alpha_lo = ci_config['alpha_lo']
    alpha_hi = ci_config['alpha_hi']

    fold_results = []
    for fd in folds:
        posterior = train_posterior_on_range(
            trades, minimal_features, fd['train_start'], fd['train_end']
        )

        train_eval = evaluate_3way(
            trades, minimal_features, posterior,
            fd['train_start'], fd['train_end'], THETA, alpha_lo, alpha_hi,
            f"fold{fd['fold']}_train"
        )
        test_eval = evaluate_3way(
            trades, minimal_features, posterior,
            fd['test_start'], fd['test_end'], THETA, alpha_lo, alpha_hi,
            f"fold{fd['fold']}_test"
        )

        fold_results.append({
            'fold': fd['fold'],
            'fe_train': train_eval['false_exec_rate'],
            'fe_test': test_eval['false_exec_rate'],
            'sg_test': test_eval['sharp_gap'],
            'imm_test': test_eval['immortal_capture_rate'],
            'pnl_test': test_eval['exec_pnl'],
            'n_exec_test': test_eval['n_exec'],
            'n_wait_test': test_eval['n_wait'],
            'wait_pct_test': test_eval['wait_pct'],
            'wait_imm_test': test_eval['wait_imm'],
            'wait_pnl_test': test_eval['wait_pnl'],
            'train_eval': train_eval,
            'test_eval': test_eval,
        })

    fe_tests = [f['fe_test'] for f in fold_results]
    sg_tests = [f['sg_test'] for f in fold_results]
    imm_tests = [f['imm_test'] for f in fold_results]
    pnl_tests = [f['pnl_test'] for f in fold_results]
    wait_pcts = [f['wait_pct_test'] for f in fold_results]
    n_execs = [f['n_exec_test'] for f in fold_results]
    wait_imms = [f['wait_imm_test'] for f in fold_results]

    summary = {
        'config': ci_config['name'],
        'market': market_name,
        'n_folds': len(folds),
        'fe_test_mean': round(np.mean(fe_tests), 1),
        'fe_test_std': round(np.std(fe_tests), 1),
        'fe_test_values': [round(v, 1) for v in fe_tests],
        'sg_test_mean': round(np.mean(sg_tests), 1),
        'imm_test_mean': round(np.mean(imm_tests), 1),
        'imm_test_std': round(np.std(imm_tests), 1),
        'pnl_test_total': round(sum(pnl_tests), 0),
        'n_exec_mean': round(np.mean(n_execs), 1),
        'wait_pct_mean': round(np.mean(wait_pcts), 1),
        'wait_imm_total': sum(wait_imms),
    }

    return fold_results, summary


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-69: CONFIDENCE-INTERVAL WAIT")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'Do not kill infant bins. Wait until they mature.'")
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
        print(f"\n  ═══ {mkt_name}: CI SWEEP ═══")

        mkt_results = {}
        for cfg in CI_CONFIGS:
            np.random.seed(42)
            folds, summary = run_ci_walk_forward(trades, mf, mkt_name, cfg)
            mkt_results[cfg['name']] = {
                'folds': folds,
                'summary': summary,
            }

        print(f"\n  {'Config':<18s}  {'FE_mean':>8s}  {'FE_std':>7s}  {'SG':>7s}  {'IMM':>7s}  {'Wait%':>6s}  {'WaitIMM':>8s}  {'PnL':>10s}  {'Exec/f':>7s}")
        print(f"  {'─'*18}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*7}")

        baseline_std = mkt_results['Baseline (2-way)']['summary']['fe_test_std']
        baseline_imm = mkt_results['Baseline (2-way)']['summary']['imm_test_mean']

        for cfg in CI_CONFIGS:
            s = mkt_results[cfg['name']]['summary']
            std_marker = ''
            if s['fe_test_std'] < baseline_std:
                std_marker = '↓'
            elif s['fe_test_std'] > baseline_std:
                std_marker = '↑'
            imm_marker = ''
            if s['imm_test_mean'] > baseline_imm + 1:
                imm_marker = '↑'
            elif s['imm_test_mean'] < baseline_imm - 1:
                imm_marker = '↓'
            print(f"  {cfg['name']:<18s}  {s['fe_test_mean']:>7.1f}%  {s['fe_test_std']:>5.1f}%{std_marker}  {s['sg_test_mean']:>+6.1f}  {s['imm_test_mean']:>5.1f}%{imm_marker}  {s['wait_pct_mean']:>5.1f}%  {s['wait_imm_total']:>8d}  ${s['pnl_test_total']:>9,.0f}  {s['n_exec_mean']:>6.1f}")

        all_results[mkt_name] = mkt_results

    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  ═══ CI WAIT VERDICT ═══")
    print(f"  ═══════════════════════════════════════════════════")

    comb = all_results.get('NQ_Combined', all_results.get('NQ_1min', {}))
    if comb:
        bl = comb['Baseline (2-way)']['summary']
        print(f"\n  Baseline: FE_mean={bl['fe_test_mean']:.1f}% FE_std={bl['fe_test_std']:.1f}% IMM={bl['imm_test_mean']:.1f}% SG={bl['sg_test_mean']:+.1f}")

        best_cfg = None
        best_score = None
        for cfg in CI_CONFIGS:
            if cfg['alpha_lo'] is None:
                continue
            s = comb[cfg['name']]['summary']
            if s['imm_test_mean'] < 80.0:
                continue
            if s['sg_test_mean'] < 60.0:
                continue
            score = -(s['fe_test_std']) + 0.1 * s['imm_test_mean'] - 0.05 * s['fe_test_mean']
            if best_score is None or score > best_score:
                best_score = score
                best_cfg = cfg

        if best_cfg:
            bs = comb[best_cfg['name']]['summary']
            std_delta = bl['fe_test_std'] - bs['fe_test_std']
            imm_delta = bs['imm_test_mean'] - bl['imm_test_mean']

            print(f"\n  Best: {best_cfg['name']}")
            print(f"    FE_mean: {bl['fe_test_mean']:.1f}% → {bs['fe_test_mean']:.1f}% (Δ{bs['fe_test_mean']-bl['fe_test_mean']:+.1f}%p)")
            print(f"    FE_std:  {bl['fe_test_std']:.1f}% → {bs['fe_test_std']:.1f}% (Δ{-std_delta:+.1f}%p)")
            print(f"    IMM:     {bl['imm_test_mean']:.1f}% → {bs['imm_test_mean']:.1f}% (Δ{imm_delta:+.1f}%p)")
            print(f"    SG:      {bl['sg_test_mean']:+.1f} → {bs['sg_test_mean']:+.1f}")
            print(f"    WAIT:    {bs['wait_pct_mean']:.1f}% of trades deferred")
            print(f"    Wait-IMM: {bs['wait_imm_total']} IMMORTAL trades in WAIT")

            imm_ok = bs['imm_test_mean'] >= 80.0
            sg_ok = bs['sg_test_mean'] >= 70.0
            std_improved = bs['fe_test_std'] < bl['fe_test_std']
            imm_preserved = bs['imm_test_mean'] >= bl['imm_test_mean'] - 5.0

            print(f"\n  Checks:")
            print(f"    IMM ≥80%?           {'✅' if imm_ok else '❌'} ({bs['imm_test_mean']:.1f}%)")
            print(f"    SG ≥70?             {'✅' if sg_ok else '❌'} ({bs['sg_test_mean']:+.1f})")
            print(f"    FE_std reduced?     {'✅' if std_improved else '❌'} ({bl['fe_test_std']:.1f}→{bs['fe_test_std']:.1f})")
            print(f"    IMM preserved?      {'✅' if imm_preserved else '❌'} (Δ{imm_delta:+.1f}%p)")

            if std_improved and imm_ok and sg_ok:
                if imm_preserved:
                    print(f"\n  ✅ CI WAIT SUCCESSFUL — FE_std↓ + IMM preserved")
                    print(f"     'The 'wait until mature' strategy broke through the shrinkage trade-off.")
                else:
                    print(f"\n  ⚠️  CI WAIT PARTIAL — FE_std↓ but IMM somewhat reduced")
            elif imm_ok and sg_ok:
                print(f"\n  ⚠️  CI WAIT: laws preserved but FE_std not reduced")
            else:
                print(f"\n  ❌ CI WAIT: laws violated")
        else:
            print(f"\n  ❌ No CI config satisfies law constraints")

        print(f"\n  ── WAIT Analysis ──")
        for cfg in CI_CONFIGS:
            if cfg['alpha_lo'] is None:
                continue
            s = comb[cfg['name']]['summary']
            print(f"    {cfg['name']}: WAIT={s['wait_pct_mean']:.1f}% | Wait-IMM={s['wait_imm_total']} | IMM_exec={s['imm_test_mean']:.1f}%")

        print(f"\n  ── Comparison: Shrinkage(λ=1) vs CI Wait ──")
        print(f"    Shrinkage λ=1: FE_std=7.0% IMM=80.3% (from EXP-66)")
        if best_cfg:
            bs = comb[best_cfg['name']]['summary']
            print(f"    CI Wait {best_cfg['name']}: FE_std={bs['fe_test_std']:.1f}% IMM={bs['imm_test_mean']:.1f}%")
            if bs['fe_test_std'] < 7.0 and bs['imm_test_mean'] > 80.3:
                print(f"    → CI Wait DOMINATES shrinkage on both axes!")
            elif bs['imm_test_mean'] > 80.3:
                print(f"    → CI Wait preserves IMM better")
            elif bs['fe_test_std'] < 7.0:
                print(f"    → CI Wait reduces variance more")
            else:
                print(f"    → Neither strictly dominates")

    exp69_dir = os.path.join(EVIDENCE_DIR, 'exp69_ci_wait')
    os.makedirs(exp69_dir, exist_ok=True)

    serializable = {}
    for mkt, mkt_data in all_results.items():
        serializable[mkt] = {}
        for cfg_name, data in mkt_data.items():
            clean_folds = []
            for f in data['folds']:
                cf = {k: v for k, v in f.items() if k not in ['train_eval', 'test_eval']}
                cf['train_eval'] = f['train_eval']
                cf['test_eval'] = f['test_eval']
                clean_folds.append(cf)
            serializable[mkt][cfg_name] = {
                'folds': clean_folds,
                'summary': data['summary'],
            }

    exp_data = {
        'experiment': 'EXP-69 Confidence-Interval Wait',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'theta': THETA,
        'ci_configs': CI_CONFIGS,
        'n_folds': N_FOLDS,
        'results': serializable,
    }

    with open(os.path.join(exp69_dir, 'ci_wait_results.json'), 'w') as f:
        json.dump(exp_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-69 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Execute what is certain, wait for what is uncertain. That is maturity.'")


if __name__ == '__main__':
    main()
