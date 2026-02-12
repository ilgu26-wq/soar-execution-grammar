#!/usr/bin/env python3
"""
EXP-68: BAGGED POSTERIOR + EXP-68b: BAGGING × CI WAIT
================================================================
"CI Wait bias reduced and, Bagging variance reduces."
"two  if synthesize Pareto boundary push number/can exists."

PREREQUISITE:
  EXP-66 — shrinkage: FE_std↓ ↔ IMM↓ trade-off. λ=1 sweet spot.
  EXP-69 — CI Wait: shrinkage positive/amount breakthrough. CI 15/85 best.
           remain wall: FE_std≤5% & IMM≥80% simultaneous satisfaction.

DESIGN (EXP-68 Bagging Only):
  From each fold's train set, bootstrap resample K → generate K posteriors
  binper/star p_exec aggregatorto/as synthesis:
    - mean:   p_hat = mean(p_k)
    - median: p_hat = median(p_k)  ← outlier resistant
    - q25:    p_hat = quantile(p_k, 0.25)  ← conservative

  Sweep A: K ∈ {5,10,20,40}, aggregator=median
  Sweep B: K=20, aggregator ∈ {mean, median, q25}

DESIGN (EXP-68b Bagging + CI Wait):
  method 1 ("CI on bagged posterior"):
    each posterior_kfrom L_k, U_k computation
    L = median(L_k), U = median(U_k)
    WAIT rule: L≤θ≤U → WAIT

  CI level: 15/85 (EXP-69from mostever/instance)

GO CONDITIONS (Phase 3 constitution):
  PASS:      IMM ≥80% AND SG ≥70% AND FE_std ≤5.0%
  SOFT PASS: IMM ≥82% AND SG ≥72% AND FE_std ≤5.7%

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ Gate/energy changes prohibited
  ✔️ Bootstrap ensemble of posteriors
  ✔️ energy conservation compliance
"""

import sys, os, json, time
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
    extract_minimal_features, NumpyEncoder,
)
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
    return extract_minimal_features(trades, arg_results, shadow_results, aep_results)


def get_bin_key(mf):
    return f"{mf['e_sign']}|{mf['de_sign']}|{mf['shadow_binary']}|{mf['arg_depth']}|{mf['regime_coarse']}|{mf['aep_binary']}"


def train_posterior_on_indices(trades, minimal_features, indices):
    posterior = BetaPosterior(alpha_prior=1.0, beta_prior=1.0)
    for i in indices:
        mf = minimal_features[i]
        posterior.update(
            e_sign=mf['e_sign'], de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
            is_win=trades[i].get('is_win', False),
        )
    return posterior


def train_posterior_on_range(trades, minimal_features, start, end):
    return train_posterior_on_indices(trades, minimal_features, range(start, end))


def bootstrap_posteriors(trades, minimal_features, start, end, K, rng):
    n = end - start
    indices_pool = list(range(start, end))
    posteriors = []
    for _ in range(K):
        boot_idx = rng.choice(indices_pool, size=n, replace=True)
        posteriors.append(train_posterior_on_indices(trades, minimal_features, boot_idx))
    return posteriors


def collect_all_bin_keys(posteriors):
    keys = set()
    for p in posteriors:
        keys.update(p.bins.keys())
    return sorted(keys)


def aggregate_p_exec(posteriors, bin_key, aggregator='median'):
    p_values = []
    for p in posteriors:
        b = p.bins[bin_key]
        p_values.append(b['alpha'] / (b['alpha'] + b['beta']))

    if aggregator == 'mean':
        return np.mean(p_values)
    elif aggregator == 'median':
        return np.median(p_values)
    elif aggregator == 'q25':
        return np.percentile(p_values, 25)
    else:
        return np.median(p_values)


def aggregate_ci(posteriors, bin_key, alpha_lo, alpha_hi, aggregator='median'):
    L_values = []
    U_values = []
    for p in posteriors:
        b = p.bins[bin_key]
        a, bb = b['alpha'], b['beta']
        L_values.append(beta_dist.ppf(alpha_lo, a, bb))
        U_values.append(beta_dist.ppf(alpha_hi, a, bb))

    if aggregator == 'median':
        return np.median(L_values), np.median(U_values)
    else:
        return np.mean(L_values), np.mean(U_values)


def classify_bagged(posteriors, mf, theta, aggregator):
    key = get_bin_key(mf)
    p_hat = aggregate_p_exec(posteriors, key, aggregator)
    return 'EXECUTE' if p_hat >= theta else 'DENY'


def classify_bagged_ci(posteriors, mf, theta, alpha_lo, alpha_hi, aggregator):
    key = get_bin_key(mf)
    L, U = aggregate_ci(posteriors, key, alpha_lo, alpha_hi, aggregator)

    if U < theta:
        return 'DENY'
    elif L > theta:
        return 'EXECUTE'
    else:
        return 'WAIT'


def evaluate_decisions(trades, minimal_features, decisions, start, end, label):
    exec_idx = []
    deny_idx = []
    wait_idx = []

    for i, d in zip(range(start, end), decisions):
        if d == 'EXECUTE':
            exec_idx.append(i)
        elif d == 'DENY':
            deny_idx.append(i)
        else:
            wait_idx.append(i)

    n_total = end - start
    n_exec = len(exec_idx)
    n_deny = len(deny_idx)
    n_wait = len(wait_idx)

    if n_exec == 0:
        return {
            'label': label, 'n_exec': 0, 'n_deny': n_deny, 'n_wait': n_wait,
            'wait_pct': round(n_wait / max(n_total, 1) * 100, 1),
            'exec_wr': 0, 'false_exec_rate': 0, 'sharp_gap': 0,
            'exec_pnl': 0, 'deny_pnl': 0, 'wait_pnl': 0,
            'immortal_capture_rate': 0, 'immortal_total': 0,
            'immortal_captured': 0, 'wait_imm': 0,
        }

    exec_trades = [trades[i] for i in exec_idx]
    deny_trades = [trades[i] for i in deny_idx]
    wait_trades = [trades[i] for i in wait_idx]

    exec_wins = sum(1 for t in exec_trades if t['is_win'])
    exec_wr = exec_wins / len(exec_trades) * 100

    non_exec = deny_trades + wait_trades
    non_exec_wr = sum(1 for t in non_exec if t['is_win']) / max(len(non_exec), 1) * 100

    false_exec = sum(1 for t in exec_trades if not t['is_win'])
    fe_rate = false_exec / len(exec_trades) * 100

    exec_pnl = sum(t['pnl'] for t in exec_trades)
    deny_pnl = sum(t['pnl'] for t in deny_trades)
    wait_pnl = sum(t['pnl'] for t in wait_trades)

    imm_total = sum(1 for i in range(start, end) if minimal_features[i]['fate'] == 'IMMORTAL')
    imm_exec = sum(1 for i in exec_idx if minimal_features[i]['fate'] == 'IMMORTAL')
    imm_wait = sum(1 for i in wait_idx if minimal_features[i]['fate'] == 'IMMORTAL')

    return {
        'label': label, 'n_exec': n_exec, 'n_deny': n_deny, 'n_wait': n_wait,
        'wait_pct': round(n_wait / max(n_total, 1) * 100, 1),
        'exec_wr': round(exec_wr, 1),
        'sharp_gap': round(exec_wr - non_exec_wr, 1),
        'false_exec_rate': round(fe_rate, 1),
        'exec_pnl': round(exec_pnl, 2), 'deny_pnl': round(deny_pnl, 2),
        'wait_pnl': round(wait_pnl, 2),
        'immortal_capture_rate': round(imm_exec / max(imm_total, 1) * 100, 1),
        'immortal_total': imm_total, 'immortal_captured': imm_exec,
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


def run_bagging_walk_forward(trades, mf, market, K, aggregator, ci_mode=False, alpha_lo=None, alpha_hi=None):
    folds = make_folds(len(trades))

    fold_results = []
    for fd in folds:
        rng = np.random.RandomState(42 + fd['fold'])

        posteriors = bootstrap_posteriors(trades, mf, fd['train_start'], fd['train_end'], K, rng)

        test_decisions = []
        for i in range(fd['test_start'], fd['test_end']):
            m = mf[i]
            if ci_mode:
                d = classify_bagged_ci(posteriors, m, THETA, alpha_lo, alpha_hi, aggregator)
            else:
                d = classify_bagged(posteriors, m, THETA, aggregator)
            test_decisions.append(d)

        train_decisions = []
        for i in range(fd['train_start'], fd['train_end']):
            m = mf[i]
            if ci_mode:
                d = classify_bagged_ci(posteriors, m, THETA, alpha_lo, alpha_hi, aggregator)
            else:
                d = classify_bagged(posteriors, m, THETA, aggregator)
            train_decisions.append(d)

        train_eval = evaluate_decisions(trades, mf, train_decisions, fd['train_start'], fd['train_end'], f"fold{fd['fold']}_train")
        test_eval = evaluate_decisions(trades, mf, test_decisions, fd['test_start'], fd['test_end'], f"fold{fd['fold']}_test")

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
        })

    fe_tests = [f['fe_test'] for f in fold_results]
    sg_tests = [f['sg_test'] for f in fold_results]
    imm_tests = [f['imm_test'] for f in fold_results]
    pnl_tests = [f['pnl_test'] for f in fold_results]
    wait_pcts = [f['wait_pct_test'] for f in fold_results]
    n_execs = [f['n_exec_test'] for f in fold_results]
    wait_imms = [f['wait_imm_test'] for f in fold_results]

    return {
        'folds': fold_results,
        'summary': {
            'market': market,
            'K': K,
            'aggregator': aggregator,
            'ci_mode': ci_mode,
            'alpha_lo': alpha_lo,
            'alpha_hi': alpha_hi,
            'n_folds': len(folds),
            'fe_test_mean': round(np.mean(fe_tests), 1),
            'fe_test_std': round(np.std(fe_tests), 1),
            'fe_test_values': [round(v, 1) for v in fe_tests],
            'sg_test_mean': round(np.mean(sg_tests), 1),
            'imm_test_mean': round(np.mean(imm_tests), 1),
            'pnl_test_total': round(sum(pnl_tests), 0),
            'n_exec_mean': round(np.mean(n_execs), 1),
            'wait_pct_mean': round(np.mean(wait_pcts), 1),
            'wait_imm_total': sum(wait_imms),
        }
    }


def print_summary_table(label, configs, baseline_std, baseline_imm):
    print(f"\n  {'Config':<28s}  {'FE_mean':>8s}  {'FE_std':>7s}  {'SG':>7s}  {'IMM':>7s}  {'Wait%':>6s}  {'WaitIMM':>8s}  {'PnL':>10s}  {'Exec/f':>7s}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*7}")

    for name, s in configs:
        std_m = '↓' if s['fe_test_std'] < baseline_std else ('↑' if s['fe_test_std'] > baseline_std else ' ')
        imm_m = '↑' if s['imm_test_mean'] > baseline_imm + 1 else ('↓' if s['imm_test_mean'] < baseline_imm - 1 else ' ')
        print(f"  {name:<28s}  {s['fe_test_mean']:>7.1f}%  {s['fe_test_std']:>5.1f}%{std_m}  {s['sg_test_mean']:>+6.1f}  {s['imm_test_mean']:>5.1f}%{imm_m}  {s['wait_pct_mean']:>5.1f}%  {s['wait_imm_total']:>8d}  ${s['pnl_test_total']:>9,.0f}  {s['n_exec_mean']:>6.1f}")


def check_go(s, label=''):
    imm = s['imm_test_mean']
    sg = s['sg_test_mean']
    fe_std = s['fe_test_std']

    hard_pass = imm >= 80.0 and sg >= 70.0 and fe_std <= 5.0
    soft_pass = imm >= 82.0 and sg >= 72.0 and fe_std <= 5.7

    return {
        'label': label,
        'hard_pass': hard_pass,
        'soft_pass': soft_pass,
        'imm': imm, 'sg': sg, 'fe_std': fe_std,
    }


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-68: BAGGED POSTERIOR + EXP-68b: BAGGING × CI WAIT")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'CI Wait bias decreases, Bagging variance reduces.'")
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

    for mkt_name in ['NQ_Combined']:
        if mkt_name not in datasets:
            continue
        trades, mf = datasets[mkt_name]

        print(f"\n  {'='*60}")
        print(f"  ═══ {mkt_name}: SWEEP A — K sweep (aggregator=median) ═══")
        print(f"  {'='*60}")

        baseline_result = run_bagging_walk_forward(trades, mf, mkt_name, K=1, aggregator='median', ci_mode=False)
        baseline_s = baseline_result['summary']
        baseline_std = baseline_s['fe_test_std']
        baseline_imm = baseline_s['imm_test_mean']

        sweep_a = [('Baseline (K=1)', baseline_s)]
        for K in [5, 10, 20, 40]:
            r = run_bagging_walk_forward(trades, mf, mkt_name, K=K, aggregator='median')
            sweep_a.append((f'Bag K={K} median', r['summary']))

        print_summary_table("Sweep A", sweep_a, baseline_std, baseline_imm)

        print(f"\n  {'='*60}")
        print(f"  ═══ {mkt_name}: SWEEP B — aggregator comparison (K=20) ═══")
        print(f"  {'='*60}")

        sweep_b = [('Baseline (K=1)', baseline_s)]
        for agg in ['mean', 'median', 'q25']:
            r = run_bagging_walk_forward(trades, mf, mkt_name, K=20, aggregator=agg)
            sweep_b.append((f'Bag K=20 {agg}', r['summary']))

        print_summary_table("Sweep B", sweep_b, baseline_std, baseline_imm)

        print(f"\n  {'='*60}")
        print(f"  ═══ {mkt_name}: EXP-68b — BAGGING × CI WAIT ═══")
        print(f"  {'='*60}")

        sweep_68b = [('Baseline (K=1)', baseline_s)]

        ci_only = run_bagging_walk_forward(trades, mf, mkt_name, K=1, aggregator='median',
                                           ci_mode=True, alpha_lo=0.15, alpha_hi=0.85)
        sweep_68b.append(('CI 15/85 only (K=1)', ci_only['summary']))

        for K in [5, 10, 20, 40]:
            r = run_bagging_walk_forward(trades, mf, mkt_name, K=K, aggregator='median',
                                         ci_mode=True, alpha_lo=0.15, alpha_hi=0.85)
            sweep_68b.append((f'Bag K={K} + CI 15/85', r['summary']))

        for K in [20]:
            r10 = run_bagging_walk_forward(trades, mf, mkt_name, K=K, aggregator='median',
                                            ci_mode=True, alpha_lo=0.10, alpha_hi=0.90)
            sweep_68b.append((f'Bag K={K} + CI 10/90', r10['summary']))
            r12 = run_bagging_walk_forward(trades, mf, mkt_name, K=K, aggregator='median',
                                            ci_mode=True, alpha_lo=0.12, alpha_hi=0.88)
            sweep_68b.append((f'Bag K={K} + CI 12/88', r12['summary']))

        print_summary_table("EXP-68b", sweep_68b, baseline_std, baseline_imm)

        all_results[mkt_name] = {
            'sweep_a': sweep_a,
            'sweep_b': sweep_b,
            'sweep_68b': sweep_68b,
        }

    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  ═══ FINAL VERDICT ═══")
    print(f"  ═══════════════════════════════════════════════════")

    comb_data = all_results.get('NQ_Combined', {})
    if comb_data:
        all_configs = []
        for sweep_name in ['sweep_a', 'sweep_b', 'sweep_68b']:
            for name, s in comb_data[sweep_name]:
                if name == 'Baseline (K=1)' and sweep_name != 'sweep_a':
                    continue
                all_configs.append((name, s))

        print(f"\n  ── GO Condition Check ──")
        print(f"  PASS:      IMM≥80% AND SG≥70% AND FE_std≤5.0%")
        print(f"  SOFT PASS: IMM≥82% AND SG≥72% AND FE_std≤5.7%")
        print()

        any_hard = False
        any_soft = False
        for name, s in all_configs:
            go = check_go(s, name)
            if go['hard_pass'] or go['soft_pass']:
                tag = '✅ PASS' if go['hard_pass'] else '⚠️  SOFT'
                print(f"    {tag}  {name:<28s}  FE_std={go['fe_std']:.1f}%  IMM={go['imm']:.1f}%  SG={go['sg']:+.1f}")
                if go['hard_pass']:
                    any_hard = True
                if go['soft_pass']:
                    any_soft = True

        if not any_hard and not any_soft:
            print(f"    ❌ No config achieves PASS or SOFT PASS")

        print(f"\n  ── Comparison: Previous Best vs Bagging ──")
        print(f"    Shrinkage λ=1:    FE_std=7.0%  IMM=80.3%  SG=+72.4  (EXP-66)")
        print(f"    CI Wait 15/85:    FE_std=6.8%  IMM=84.3%  SG=+73.4  (EXP-69)")

        best_score = None
        best_entry = None
        for name, s in all_configs:
            if s['imm_test_mean'] < 78.0 or s['sg_test_mean'] < 65.0:
                continue
            score = -s['fe_test_std'] + 0.15 * s['imm_test_mean'] + 0.05 * s['sg_test_mean']
            if best_score is None or score > best_score:
                best_score = score
                best_entry = (name, s)

        if best_entry:
            bn, bs = best_entry
            print(f"    Best EXP-68:      FE_std={bs['fe_test_std']:.1f}%  IMM={bs['imm_test_mean']:.1f}%  SG={bs['sg_test_mean']:+.1f}  [{bn}]")

            if bs['fe_test_std'] < 6.8 and bs['imm_test_mean'] > 84.3:
                print(f"    → Bagging DOMINATES all previous methods!")
            elif bs['fe_test_std'] < 6.8:
                print(f"    → Bagging achieves lower variance than CI Wait")
            elif bs['imm_test_mean'] > 84.3:
                print(f"    → Bagging achieves higher IMM than CI Wait")

        if any_hard:
            print(f"\n  ✅ HARD PASS ACHIEVED — Phase 3 constitution satisfaction!")
        elif any_soft:
            print(f"\n  ⚠️  SOFT PASS — realityever/instance reference/criteria satisfaction, FE_std≤5% ")
        else:
            print(f"\n  ❌ GO CONDITIONS NOT MET — addition  needed")

    exp_dir = os.path.join(EVIDENCE_DIR, 'exp68_bagged_posterior')
    os.makedirs(exp_dir, exist_ok=True)

    save_data = {
        'experiment': 'EXP-68 Bagged Posterior + EXP-68b Bagging x CI Wait',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'theta': THETA,
        'results': {}
    }
    for mkt, mkt_data in all_results.items():
        save_data['results'][mkt] = {}
        for sweep_name, configs in mkt_data.items():
            save_data['results'][mkt][sweep_name] = [
                {'name': n, 'summary': s} for n, s in configs
            ]

    with open(os.path.join(exp_dir, 'bagged_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-68 + 68b Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'multiple observationConsensus of judges is stronger than individual conviction.'")


if __name__ == '__main__':
    main()
