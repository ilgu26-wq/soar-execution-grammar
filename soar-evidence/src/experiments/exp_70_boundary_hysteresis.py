#!/usr/bin/env python3
"""
EXP-70: BOUNDARY HYSTERESIS — boundarylinefrom does not shake does not
================================================================
"θ A single cut causes boundary trades to flip per fold, increasing FE_std.
 physicslearning hysteresis: two criticalvalueto/as dead band onlyfeels."

PREREQUISITE:
  EXP-68 — Bagging alone effectand none. variance's/of origin posterior estimation not... but
           θ boundaryline near trade's/of selection variance.
  EXP-69 — CI Wait 15/85 SOFT PASS (FE_std 5.7%, IMM 84.3%).
  → boundaryline itself physicalto/as must organize HARD PASS possible.

DESIGN:
  existing: p_exec ≥ θ(=0.5) → EXECUTE, else DENY
  hysteresis:
    p_exec ≥ θ_hi → EXECUTE  (definitely execute)
    p_exec ≤ θ_lo → DENY     (certain rain/non-execution)
    θ_lo < p_exec < θ_hi → HOLD (directly determination maintained / or WAIT)

  θ_hi/θ_lo sweep:
    Band width Δ ∈ {0.02, 0.05, 0.08, 0.10, 0.15, 0.20}
    θ_hi = 0.5 + Δ/2,  θ_lo = 0.5 - Δ/2

  HOLD policy 2know:
    Mode A: HOLD → WAIT (execution not done, observationonly)
    Mode B: HOLD → directly fold determination maintained (memory)
    → Mode A CI Waitand compatibleto/as default

  Walk-forward 5-fold flat

METRICS:
  Primary: FE_std, IMM, SG
  Secondary: flip_rate (fold determination flipping rate), band_pct (band not trade ratio)
  GO: HARD PASS = IMM≥80% AND SG≥70% AND FE_std≤5.0%

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ Gate/energy changes prohibited
  ✔️ execution boundary single θ → (θ_lo, θ_hi)  criticalvalueto/as expansion
  ✔️ irreversible execution θ_hi must exceedonly — philosophyand 100% match
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
N_FOLDS = 5
MIN_TRAIN_FRAC = 0.30
TEST_WINDOW_FRAC = 0.14

BAND_WIDTHS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

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


def classify_hysteresis(posterior, mf, theta_lo, theta_hi):
    key = get_bin_key(mf)
    b = posterior.bins[key]
    p = b['alpha'] / (b['alpha'] + b['beta'])

    if p >= theta_hi:
        return 'EXECUTE'
    elif p <= theta_lo:
        return 'DENY'
    else:
        return 'WAIT'


def evaluate_decisions(trades, minimal_features, start, end, posterior, theta_lo, theta_hi, label):
    exec_idx = []
    deny_idx = []
    wait_idx = []

    for i in range(start, end):
        mf = minimal_features[i]
        d = classify_hysteresis(posterior, mf, theta_lo, theta_hi)
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
            'band_pct': round(n_wait / max(n_total, 1) * 100, 1),
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
        'band_pct': round(n_wait / max(n_total, 1) * 100, 1),
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


def compute_flip_rate(trades, mf, folds, theta_lo, theta_hi):
    if len(folds) < 2:
        return 0.0

    fold_decisions = {}
    for fd in folds:
        posterior = train_posterior_on_range(trades, mf, fd['train_start'], fd['train_end'])

        all_idx = set(range(fd['test_start'], fd['test_end']))
        decisions = {}
        for i in all_idx:
            d = classify_hysteresis(posterior, mf[i], theta_lo, theta_hi)
            decisions[i] = d
        fold_decisions[fd['fold']] = decisions

    fold_nums = sorted(fold_decisions.keys())
    total_comparisons = 0
    flips = 0
    for f_idx in range(len(fold_nums) - 1):
        f1, f2 = fold_nums[f_idx], fold_nums[f_idx + 1]
        common_idx = set(fold_decisions[f1].keys()) & set(fold_decisions[f2].keys())
        for i in common_idx:
            d1 = fold_decisions[f1][i]
            d2 = fold_decisions[f2][i]
            if d1 in ('EXECUTE', 'DENY') and d2 in ('EXECUTE', 'DENY'):
                total_comparisons += 1
                if d1 != d2:
                    flips += 1

    return round(flips / max(total_comparisons, 1) * 100, 1)


def run_hysteresis_walk_forward(trades, mf, market, delta):
    theta_lo = 0.5 - delta / 2
    theta_hi = 0.5 + delta / 2
    folds = make_folds(len(trades))

    fold_results = []
    for fd in folds:
        posterior = train_posterior_on_range(trades, mf, fd['train_start'], fd['train_end'])

        test_eval = evaluate_decisions(
            trades, mf, fd['test_start'], fd['test_end'],
            posterior, theta_lo, theta_hi, f"fold{fd['fold']}_test"
        )

        fold_results.append({
            'fold': fd['fold'],
            'fe_test': test_eval['false_exec_rate'],
            'sg_test': test_eval['sharp_gap'],
            'imm_test': test_eval['immortal_capture_rate'],
            'pnl_test': test_eval['exec_pnl'],
            'n_exec_test': test_eval['n_exec'],
            'n_wait_test': test_eval['n_wait'],
            'wait_pct_test': test_eval['wait_pct'],
            'band_pct_test': test_eval['band_pct'],
            'wait_imm_test': test_eval['wait_imm'],
        })

    fe_tests = [f['fe_test'] for f in fold_results]
    sg_tests = [f['sg_test'] for f in fold_results]
    imm_tests = [f['imm_test'] for f in fold_results]
    pnl_tests = [f['pnl_test'] for f in fold_results]
    wait_pcts = [f['wait_pct_test'] for f in fold_results]
    band_pcts = [f['band_pct_test'] for f in fold_results]
    n_execs = [f['n_exec_test'] for f in fold_results]
    wait_imms = [f['wait_imm_test'] for f in fold_results]

    flip_rate = compute_flip_rate(trades, mf, folds, theta_lo, theta_hi)

    return {
        'folds': fold_results,
        'summary': {
            'market': market,
            'delta': delta,
            'theta_lo': round(theta_lo, 3),
            'theta_hi': round(theta_hi, 3),
            'n_folds': len(folds),
            'fe_test_mean': round(np.mean(fe_tests), 1),
            'fe_test_std': round(np.std(fe_tests), 1),
            'fe_test_values': [round(v, 1) for v in fe_tests],
            'sg_test_mean': round(np.mean(sg_tests), 1),
            'imm_test_mean': round(np.mean(imm_tests), 1),
            'pnl_test_total': round(sum(pnl_tests), 0),
            'n_exec_mean': round(np.mean(n_execs), 1),
            'wait_pct_mean': round(np.mean(wait_pcts), 1),
            'band_pct_mean': round(np.mean(band_pcts), 1),
            'wait_imm_total': sum(wait_imms),
            'flip_rate': flip_rate,
        }
    }


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-70: BOUNDARY HYSTERESIS")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'boundarylinefrom does not shake does not. if exceeds execution, cannot if exceeds waits.'")
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

        print(f"\n  ═══ {mkt_name}: HYSTERESIS Δ SWEEP ═══")

        configs = []
        for delta in BAND_WIDTHS:
            r = run_hysteresis_walk_forward(trades, mf, mkt_name, delta)
            configs.append((delta, r))

        baseline_s = configs[0][1]['summary']
        baseline_std = baseline_s['fe_test_std']
        baseline_imm = baseline_s['imm_test_mean']

        print(f"\n  {'Δ':>5s}  {'θ_lo':>5s}  {'θ_hi':>5s}  {'FE_mean':>8s}  {'FE_std':>7s}  {'SG':>7s}  {'IMM':>7s}  {'Band%':>6s}  {'WtIMM':>6s}  {'Flip%':>6s}  {'PnL':>10s}  {'Exec/f':>7s}")
        print(f"  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*7}")

        for delta, r in configs:
            s = r['summary']
            std_m = '↓' if s['fe_test_std'] < baseline_std else ('↑' if s['fe_test_std'] > baseline_std else ' ')
            imm_m = '↓' if s['imm_test_mean'] < baseline_imm - 1 else (' ' if abs(s['imm_test_mean'] - baseline_imm) <= 1 else '↑')
            print(f"  {delta:>5.2f}  {s['theta_lo']:>5.3f}  {s['theta_hi']:>5.3f}  {s['fe_test_mean']:>7.1f}%  {s['fe_test_std']:>5.1f}%{std_m}  {s['sg_test_mean']:>+6.1f}  {s['imm_test_mean']:>5.1f}%{imm_m}  {s['band_pct_mean']:>5.1f}%  {s['wait_imm_total']:>6d}  {s['flip_rate']:>5.1f}%  ${s['pnl_test_total']:>9,.0f}  {s['n_exec_mean']:>6.1f}")

        all_results[mkt_name] = configs

    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  ═══ HYSTERESIS VERDICT ═══")
    print(f"  ═══════════════════════════════════════════════════")

    comb_configs = all_results.get('NQ_Combined', [])
    if comb_configs:
        bl_s = comb_configs[0][1]['summary']
        print(f"\n  Baseline (Δ=0): FE_std={bl_s['fe_test_std']:.1f}% IMM={bl_s['imm_test_mean']:.1f}% SG={bl_s['sg_test_mean']:+.1f} Flip={bl_s['flip_rate']:.1f}%")

        best_delta = None
        best_score = None
        for delta, r in comb_configs:
            s = r['summary']
            if s['imm_test_mean'] < 80.0 or s['sg_test_mean'] < 65.0:
                continue
            score = -s['fe_test_std'] + 0.15 * s['imm_test_mean'] + 0.05 * s['sg_test_mean']
            if best_score is None or score > best_score:
                best_score = score
                best_delta = delta

        if best_delta is not None:
            bs = None
            for d, r in comb_configs:
                if d == best_delta:
                    bs = r['summary']
                    break

            print(f"\n  Best: Δ={best_delta:.2f} (θ_lo={bs['theta_lo']:.3f}, θ_hi={bs['theta_hi']:.3f})")
            print(f"    FE_std:  {bl_s['fe_test_std']:.1f}% → {bs['fe_test_std']:.1f}% (Δ{bs['fe_test_std']-bl_s['fe_test_std']:+.1f}%p)")
            print(f"    IMM:     {bl_s['imm_test_mean']:.1f}% → {bs['imm_test_mean']:.1f}% (Δ{bs['imm_test_mean']-bl_s['imm_test_mean']:+.1f}%p)")
            print(f"    SG:      {bl_s['sg_test_mean']:+.1f} → {bs['sg_test_mean']:+.1f}")
            print(f"    Flip:    {bl_s['flip_rate']:.1f}% → {bs['flip_rate']:.1f}%")
            print(f"    Band:    {bs['band_pct_mean']:.1f}% trades in dead band")
            print(f"    WaitIMM: {bs['wait_imm_total']} IMMORTAL in band")

            hard_pass = bs['imm_test_mean'] >= 80 and bs['sg_test_mean'] >= 70 and bs['fe_test_std'] <= 5.0
            soft_pass = bs['imm_test_mean'] >= 82 and bs['sg_test_mean'] >= 72 and bs['fe_test_std'] <= 5.7

            print(f"\n  GO Checks:")
            print(f"    IMM ≥80%?       {'✅' if bs['imm_test_mean'] >= 80 else '❌'} ({bs['imm_test_mean']:.1f}%)")
            print(f"    SG ≥70?         {'✅' if bs['sg_test_mean'] >= 70 else '❌'} ({bs['sg_test_mean']:+.1f})")
            print(f"    FE_std ≤5.0%?   {'✅' if bs['fe_test_std'] <= 5.0 else '❌'} ({bs['fe_test_std']:.1f}%)")
            print(f"    HARD PASS?      {'✅' if hard_pass else '❌'}")
            print(f"    SOFT PASS?      {'✅' if soft_pass else '❌'}")

            if hard_pass:
                print(f"\n  ✅ HARD PASS — Phase 3 constitution  satisfaction!")
            elif soft_pass:
                print(f"\n  ⚠️  SOFT PASS")
            else:
                print(f"\n  ❌ GO CONDITIONS NOT MET")

        print(f"\n  ── Comparison: All Methods ──")
        print(f"    Shrinkage λ=1:    FE_std=7.0%  IMM=80.3%  SG=+72.4   (EXP-66)")
        print(f"    CI Wait 15/85:    FE_std=5.7%  IMM=84.3%  SG=+74.8   (EXP-68b)")
        if best_delta is not None:
            print(f"    Hysteresis Δ={best_delta:.2f}: FE_std={bs['fe_test_std']:.1f}%  IMM={bs['imm_test_mean']:.1f}%  SG={bs['sg_test_mean']:+.1f}   (EXP-70)")

        print(f"\n  ── FE_std vs Flip Rate Curve ──")
        for delta, r in comb_configs:
            s = r['summary']
            bar_std = '█' * int(s['fe_test_std'] * 3)
            bar_flip = '▒' * int(s['flip_rate'] * 2)
            print(f"    Δ={delta:.2f}: FE_std={s['fe_test_std']:>5.1f}% {bar_std}  Flip={s['flip_rate']:>5.1f}% {bar_flip}")

    exp_dir = os.path.join(EVIDENCE_DIR, 'exp70_boundary_hysteresis')
    os.makedirs(exp_dir, exist_ok=True)

    save_data = {
        'experiment': 'EXP-70 Boundary Hysteresis',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'band_widths': BAND_WIDTHS,
        'results': {}
    }
    for mkt, configs in all_results.items():
        save_data['results'][mkt] = [
            {'delta': d, 'summary': r['summary'], 'folds': r['folds']}
            for d, r in configs
        ]

    with open(os.path.join(exp_dir, 'hysteresis_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-70 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'If it crosses, execute. If not, wait. Oscillation at the boundary is a structural defect.'")


if __name__ == '__main__':
    main()
