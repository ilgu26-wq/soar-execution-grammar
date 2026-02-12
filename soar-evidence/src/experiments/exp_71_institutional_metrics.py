#!/usr/bin/env python3
"""
EXP-71: INSTITUTIONAL METRICS OVERLAY
================================================================
"SOAR does not exist to raise Sharpe.
 It measures Sharpe only to prove it does not collapse."

PURPOSE:
  Verify that SOAR's structural execution grammar produces
  non-pathological risk profiles under standard institutional
  evaluation frameworks — WITHOUT optimizing for them.

DESIGN:
  Using the sealed v3_runtime and CI Wait 15/85 (unchanged):
  1. Reconstruct equity curves from per-trade PnL
  2. Compute institutional metrics as NON-OPTIMIZED overlays
  3. Compare 4 systems on identical trade universe

COMPARISON SYSTEMS:
  Baseline    — Always-Execute (no gate, all trades taken)
  Sharp       — EXEC/DENY only (2-way, Sharp Boundary)
  SOAR v3     — EXEC/DENY/WAIT (3-way, CI 15/85)
  Random Gate — Same execution frequency as SOAR v3, random selection

METRICS:
  Sharpe Ratio (per-trade returns)
  Sortino Ratio (downside deviation only)
  Max Drawdown (peak-to-trough)
  Calmar Ratio (annualized return / MDD)
  Tail Loss 95%, 99% (VaR)
  Hit Rate (win %)

CONSTITUTION:
  ❌ No parameter changes (v3_runtime sealed)
  ❌ No bin structure changes (MDU Law)
  ❌ No optimization toward these metrics
  ✔️ Pure measurement overlay on frozen system
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
CI_ALPHA_LO = 0.15
CI_ALPHA_HI = 0.85
N_RANDOM_SEEDS = 20
TRADES_PER_YEAR_APPROX = 252


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


def train_full_posterior(trades, minimal_features):
    posterior = BetaPosterior(alpha_prior=1.0, beta_prior=1.0)
    for i, mf in enumerate(minimal_features):
        posterior.update(
            e_sign=mf['e_sign'], de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
            is_win=trades[i].get('is_win', False),
        )
    return posterior


def classify_3way(posterior, mf, theta):
    key = get_bin_key(mf)
    b = posterior.bins[key]
    a, bb = b['alpha'], b['beta']
    L = beta_dist.ppf(CI_ALPHA_LO, a, bb)
    U = beta_dist.ppf(CI_ALPHA_HI, a, bb)
    if U < theta:
        return 'DENY'
    elif L > theta:
        return 'EXECUTE'
    else:
        return 'WAIT'


def classify_2way(posterior, mf, theta):
    key = get_bin_key(mf)
    b = posterior.bins[key]
    a, bb = b['alpha'], b['beta']
    p = a / (a + bb)
    return 'EXECUTE' if p >= theta else 'DENY'


def build_equity_curve(pnl_series):
    equity = [0.0]
    for pnl in pnl_series:
        equity.append(equity[-1] + pnl)
    return equity


def compute_max_drawdown(equity):
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe(returns):
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r == 0:
        return 0.0
    return float(mean_r / std_r * math.sqrt(TRADES_PER_YEAR_APPROX))


def compute_sortino(returns):
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    downside = returns[returns < 0]
    if len(downside) < 1:
        return float('inf') if mean_r > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(mean_r / downside_std * math.sqrt(TRADES_PER_YEAR_APPROX))


def compute_calmar(total_return, max_dd, n_trades):
    if max_dd == 0 or n_trades == 0:
        return 0.0
    annualized = total_return * (TRADES_PER_YEAR_APPROX / n_trades)
    return float(annualized / max_dd)


def compute_tail_loss(returns, percentile):
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, percentile))


def compute_institutional_metrics(pnl_list, label):
    returns = np.array(pnl_list, dtype=float)
    n = len(returns)
    if n == 0:
        return {
            'label': label, 'n_trades': 0,
            'total_pnl': 0, 'sharpe': 0, 'sortino': 0,
            'max_drawdown': 0, 'calmar': 0,
            'tail_5pct': 0, 'tail_1pct': 0,
            'hit_rate': 0, 'avg_win': 0, 'avg_loss': 0,
            'profit_factor': 0,
        }

    equity = build_equity_curve(returns)
    total_pnl = float(sum(returns))
    mdd = compute_max_drawdown(equity)

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    hit_rate = len(wins) / n * 100 if n > 0 else 0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    return {
        'label': label,
        'n_trades': n,
        'total_pnl': round(total_pnl, 2),
        'sharpe': round(compute_sharpe(returns), 3),
        'sortino': round(compute_sortino(returns), 3),
        'max_drawdown': round(mdd, 2),
        'calmar': round(compute_calmar(total_pnl, mdd, n), 3),
        'tail_5pct': round(compute_tail_loss(returns, 5), 2),
        'tail_1pct': round(compute_tail_loss(returns, 1), 2),
        'hit_rate': round(hit_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 'inf',
    }


def run_institutional_overlay(trades, minimal_features, market_name):
    posterior = train_full_posterior(trades, minimal_features)

    all_pnl = [t['pnl'] for t in trades]

    sharp_exec_pnl = []
    sharp_deny_pnl = []
    soar_exec_pnl = []
    soar_deny_pnl = []
    soar_wait_pnl = []

    soar_exec_count = 0
    for i, (t, mf) in enumerate(zip(trades, minimal_features)):
        decision_2way = classify_2way(posterior, mf, THETA)
        if decision_2way == 'EXECUTE':
            sharp_exec_pnl.append(t['pnl'])
        else:
            sharp_deny_pnl.append(t['pnl'])

        decision_3way = classify_3way(posterior, mf, THETA)
        if decision_3way == 'EXECUTE':
            soar_exec_pnl.append(t['pnl'])
            soar_exec_count += 1
        elif decision_3way == 'DENY':
            soar_deny_pnl.append(t['pnl'])
        else:
            soar_wait_pnl.append(t['pnl'])

    random_results = []
    exec_ratio = soar_exec_count / max(len(trades), 1)
    for seed in range(N_RANDOM_SEEDS):
        rng = np.random.RandomState(seed + 100)
        mask = rng.random(len(trades)) < exec_ratio
        rand_pnl = [t['pnl'] for i, t in enumerate(trades) if mask[i]]
        if len(rand_pnl) > 0:
            random_results.append(compute_institutional_metrics(rand_pnl, f'Random_{seed}'))

    baseline_metrics = compute_institutional_metrics(all_pnl, 'Baseline (Always-Execute)')
    sharp_metrics = compute_institutional_metrics(sharp_exec_pnl, 'Sharp Boundary (EXEC/DENY)')
    soar_metrics = compute_institutional_metrics(soar_exec_pnl, 'SOAR v3 (CI Wait 15/85)')

    rand_avg = {}
    if random_results:
        for key in ['sharpe', 'sortino', 'max_drawdown', 'calmar', 'tail_5pct', 'tail_1pct',
                     'hit_rate', 'total_pnl', 'n_trades', 'avg_win', 'avg_loss']:
            vals = [r[key] for r in random_results if isinstance(r[key], (int, float))]
            rand_avg[key] = round(np.mean(vals), 3) if vals else 0
        pf_vals = [r['profit_factor'] for r in random_results
                   if isinstance(r['profit_factor'], (int, float))]
        rand_avg['profit_factor'] = round(np.mean(pf_vals), 3) if pf_vals else 0
        rand_avg['label'] = f'Random Gate (avg of {N_RANDOM_SEEDS} seeds)'

    denied_metrics = compute_institutional_metrics(sharp_deny_pnl, 'Sharp Denied Trades')
    waited_metrics = compute_institutional_metrics(soar_wait_pnl, 'SOAR Waited Trades')

    return {
        'market': market_name,
        'baseline': baseline_metrics,
        'sharp': sharp_metrics,
        'soar_v3': soar_metrics,
        'random_gate': rand_avg,
        'random_individual': random_results,
        'denied_trades': denied_metrics,
        'waited_trades': waited_metrics,
        'soar_exec_ratio': round(exec_ratio * 100, 1),
    }


def print_comparison_table(result):
    systems = [result['baseline'], result['sharp'], result['soar_v3'], result['random_gate']]

    print(f"\n  {'System':<32s}  {'Sharpe':>7s}  {'Sortino':>8s}  {'MDD':>10s}  {'Calmar':>7s}  {'Tail5%':>8s}  {'Tail1%':>8s}  {'HitR%':>6s}  {'PF':>6s}  {'PnL':>10s}  {'N':>5s}")
    print(f"  {'─'*32}  {'─'*7}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*5}")

    for s in systems:
        pf_str = f"{s['profit_factor']:>6.2f}" if isinstance(s.get('profit_factor'), (int, float)) else f"{'inf':>6s}"
        print(f"  {s['label']:<32s}  {s.get('sharpe',0):>+7.3f}  {s.get('sortino',0):>+8.3f}  ${s.get('max_drawdown',0):>9,.2f}  {s.get('calmar',0):>+7.3f}  ${s.get('tail_5pct',0):>7,.2f}  ${s.get('tail_1pct',0):>7,.2f}  {s.get('hit_rate',0):>5.1f}%  {pf_str}  ${s.get('total_pnl',0):>9,.2f}  {int(s.get('n_trades',0)):>5d}")


def print_tail_analysis(result):
    print(f"\n  ── Tail Risk Analysis ──")
    print(f"  Gate changes tail structure, not expected return.")

    bl = result['baseline']
    sv = result['soar_v3']

    if bl.get('tail_5pct', 0) != 0:
        tail5_change = (sv.get('tail_5pct', 0) - bl.get('tail_5pct', 0))
        tail1_change = (sv.get('tail_1pct', 0) - bl.get('tail_1pct', 0))
        print(f"    Tail 5% shift: ${tail5_change:+,.2f} (Baseline ${bl['tail_5pct']:,.2f} → SOAR ${sv['tail_5pct']:,.2f})")
        print(f"    Tail 1% shift: ${tail1_change:+,.2f} (Baseline ${bl['tail_1pct']:,.2f} → SOAR ${sv['tail_1pct']:,.2f})")

    print(f"\n  ── WAIT Impact on Drawdown ──")
    print(f"    Baseline MDD:  ${bl.get('max_drawdown', 0):,.2f}")
    print(f"    SOAR v3 MDD:   ${sv.get('max_drawdown', 0):,.2f}")
    mdd_reduction = bl.get('max_drawdown', 0) - sv.get('max_drawdown', 0)
    if bl.get('max_drawdown', 0) > 0:
        mdd_pct = mdd_reduction / bl['max_drawdown'] * 100
        print(f"    MDD reduction: ${mdd_reduction:,.2f} ({mdd_pct:+.1f}%)")


def print_random_comparison(result):
    sv = result['soar_v3']
    rg = result['random_gate']
    print(f"\n  ── Structure vs Luck ──")
    print(f"    SOAR v3 Sharpe:    {sv.get('sharpe', 0):+.3f}")
    print(f"    Random Gate Sharpe: {rg.get('sharpe', 0):+.3f} (mean of {N_RANDOM_SEEDS} seeds)")
    sharpe_delta = sv.get('sharpe', 0) - rg.get('sharpe', 0)
    print(f"    Structural alpha:   {sharpe_delta:+.3f}")

    rand_sharpes = [r['sharpe'] for r in result['random_individual']]
    if rand_sharpes:
        above = sum(1 for rs in rand_sharpes if rs >= sv.get('sharpe', 0))
        print(f"    SOAR beats {N_RANDOM_SEEDS - above}/{N_RANDOM_SEEDS} random gates ({(N_RANDOM_SEEDS - above)/N_RANDOM_SEEDS*100:.0f}% dominance)")


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-71: INSTITUTIONAL METRICS OVERLAY")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'SOAR does not exist to raise Sharpe.")
    print(f"   It measures Sharpe only to prove it does not collapse.'")
    print("=" * 70)

    nq_tick_path = os.path.join(ASSETS, 'NinjaTrader_FullOrderFlow_1770877659150.csv')
    nq_combined_path = os.path.join(ASSETS, 'NQ1_1min_combined.csv')

    datasets = {}

    if os.path.exists(nq_tick_path):
        try:
            ticks_df = load_ticks(nq_tick_path)
            nq_5s = aggregate_5s(ticks_df)
            signals = generate_signals_multi(nq_5s, tick_size=0.25)
            trades_tick, _, _ = run_pipeline_deferred(signals, nq_5s, 5.0, 0.25)
            mf_tick = compute_features(trades_tick)
            datasets['NQ_Tick'] = (trades_tick, mf_tick)
            print(f"  NQ_Tick: {len(trades_tick)} trades loaded")
        except Exception as e:
            print(f"  NQ_Tick: skipped ({e})")

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None and len(nq_1m) > 200:
            signals = generate_signals_multi(nq_1m, tick_size=0.25)
            trades_1m, _, _ = run_pipeline_deferred(signals, nq_1m, 5.0, 0.25)
            mf_1m = compute_features(trades_1m)
            datasets['NQ_1min'] = (trades_1m, mf_1m)
            print(f"  NQ_1min: {len(trades_1m)} trades loaded")

    if 'NQ_Tick' in datasets and 'NQ_1min' in datasets:
        combined_trades = list(datasets['NQ_Tick'][0]) + list(datasets['NQ_1min'][0])
        combined_mf = list(datasets['NQ_Tick'][1]) + list(datasets['NQ_1min'][1])
        datasets['NQ_Combined'] = (combined_trades, combined_mf)
        print(f"  NQ_Combined: {len(combined_trades)} trades loaded")

    all_results = {}

    for mkt_name, (trades, mf) in datasets.items():
        print(f"\n  {'═' * 60}")
        print(f"  ═══ {mkt_name}: INSTITUTIONAL METRICS ═══")
        print(f"  {'═' * 60}")

        result = run_institutional_overlay(trades, mf, mkt_name)
        all_results[mkt_name] = result

        print_comparison_table(result)
        print_tail_analysis(result)
        print_random_comparison(result)

    print(f"\n  {'═' * 60}")
    print(f"  ═══ EXP-71 VERDICT ═══")
    print(f"  {'═' * 60}")

    primary = all_results.get('NQ_Combined', all_results.get('NQ_1min', {}))
    if primary:
        bl = primary['baseline']
        sv = primary['soar_v3']
        rg = primary['random_gate']

        sharpe_ok = sv.get('sharpe', 0) > 0
        sortino_ok = sv.get('sortino', 0) > 0
        mdd_ok = sv.get('max_drawdown', 0) <= bl.get('max_drawdown', 0)
        beats_random = sv.get('sharpe', 0) > rg.get('sharpe', 0)

        print(f"\n  Pathology Checks:")
        print(f"    Sharpe > 0?              {'PASS' if sharpe_ok else 'FAIL'} ({sv.get('sharpe',0):+.3f})")
        print(f"    Sortino > 0?             {'PASS' if sortino_ok else 'FAIL'} ({sv.get('sortino',0):+.3f})")
        print(f"    MDD ≤ Baseline?          {'PASS' if mdd_ok else 'FAIL'} (${sv.get('max_drawdown',0):,.2f} vs ${bl.get('max_drawdown',0):,.2f})")
        print(f"    Beats Random Gate?       {'PASS' if beats_random else 'FAIL'} ({sv.get('sharpe',0):+.3f} vs {rg.get('sharpe',0):+.3f})")

        all_pass = sharpe_ok and sortino_ok and mdd_ok and beats_random

        if all_pass:
            print(f"\n  VERDICT: PASS — No pathological risk profile detected.")
            print(f"  Structural execution does not introduce institutional red flags.")
        else:
            fails = []
            if not sharpe_ok: fails.append('Sharpe≤0')
            if not sortino_ok: fails.append('Sortino≤0')
            if not mdd_ok: fails.append('MDD>Baseline')
            if not beats_random: fails.append('Loses to Random')
            print(f"\n  VERDICT: FAIL — {', '.join(fails)}")

        print(f"\n  NOTE: These metrics are NON-OPTIMIZED overlays.")
        print(f"  SOAR's objectives remain: Sharp Gap, IMM Capture, False Execute.")
        print(f"  Institutional metrics confirm absence of pathology, not fitness.")

    exp71_dir = os.path.join(EVIDENCE_DIR, 'exp71_institutional_metrics')
    os.makedirs(exp71_dir, exist_ok=True)

    exp_data = {
        'experiment': 'EXP-71 Institutional Metrics Overlay',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'theta': THETA,
        'ci_alpha': [CI_ALPHA_LO, CI_ALPHA_HI],
        'n_random_seeds': N_RANDOM_SEEDS,
        'trades_per_year_approx': TRADES_PER_YEAR_APPROX,
        'design': {
            'purpose': 'Verify non-pathological risk profile under institutional metrics',
            'constraint': 'NO optimization — pure measurement overlay on sealed v3_runtime',
            'systems': ['Baseline (Always-Execute)', 'Sharp Boundary (EXEC/DENY)',
                       'SOAR v3 (CI Wait 15/85)', 'Random Gate (same freq)'],
        },
        'results': all_results,
    }

    with open(os.path.join(exp71_dir, 'institutional_metrics_results.json'), 'w') as f:
        json.dump(exp_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-71 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Risk-adjusted return is a language, not an objective.'")


if __name__ == '__main__':
    main()
