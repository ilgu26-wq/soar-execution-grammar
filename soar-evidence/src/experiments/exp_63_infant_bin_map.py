#!/usr/bin/env python3
"""
EXP-63: INFANT BIN GROWTH MAP (observation insufficient knowalso)
================================================================
"IMMORTALWhere is it hiding, and why is it not yet visible?"

PREREQUISITE:
  MDU Law — 57-bin topology sealed
  EXP-62 — IMMORTAL trades cluster in infant bins (low sharpness)

PURPOSE:
  1. IMMORTAL concentrated infant bin exactly identification
  2. each bin why infantwhether decomposition (Insufficient data's/of structural person/of)
  3. "rare alpha bin top-K" list create
  4.  binplural which/what regime/vol/timeat does not rise analysis

OUTPUT:
  - binper/star (α+β), sharpness, IMM count, win/loss, fate distribution
  - Infant bin's/of market condition to/asfile
  - observation priority above map (where observation more must do)

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ p_exec value manipulation prohibited
  ✔️ bin's/of internal structureonly analysis (read-only)
"""

import sys, os, json, time
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
from observer.learning.p_exec_posterior import BetaPosterior

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')

THETA = 0.5

INFANT_THRESHOLD = 10
DECIDED_THRESHOLD = 20


def classify_maturity(sharpness):
    if sharpness < 3:
        return 'NEWBORN'
    elif sharpness < INFANT_THRESHOLD:
        return 'INFANT'
    elif sharpness < DECIDED_THRESHOLD:
        return 'ADOLESCENT'
    else:
        return 'MATURE'


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-63: INFANT BIN GROWTH MAP")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'IMMORTALWhere is it hiding, and why is it not yet visible?'")
    print("=" * 70)

    posterior_path = os.path.join(EVIDENCE_DIR, 'exp57_execution_probability', 'posterior.json')
    if not os.path.exists(posterior_path):
        print("  ❌ No posterior found. Run EXP-57 first.")
        return

    posterior = BetaPosterior()
    posterior.load(posterior_path)
    all_bins = posterior.get_all_bins()
    print(f"\n  Bins: {len(all_bins)} (MDU-sealed)")

    nq_tick_path = os.path.join(ASSETS, 'NinjaTrader_FullOrderFlow_1770877659150.csv')
    nq_combined_path = os.path.join(ASSETS, 'NQ1_1min_combined.csv')

    all_trades = []
    trade_sources = []

    if os.path.exists(nq_tick_path):
        ticks_df = load_ticks(nq_tick_path)
        nq_5s = aggregate_5s(ticks_df)
        signals = generate_signals_multi(nq_5s, tick_size=0.25)
        trades, _, _ = run_pipeline_deferred(signals, nq_5s, 5.0, 0.25)
        all_trades.extend(trades)
        trade_sources.extend(['NQ_Tick'] * len(trades))
        print(f"  NQ_Tick_5s: {len(trades)} trades")

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None:
            signals = generate_signals_multi(nq_1m, tick_size=0.25)
            trades, _, _ = run_pipeline_deferred(signals, nq_1m, 5.0, 0.25)
            all_trades.extend(trades)
            trade_sources.extend(['NQ_1min'] * len(trades))
            print(f"  NQ_1min: {len(trades)} trades")

    print(f"  Total trades: {len(all_trades)}")

    shadow_results = []
    for t in all_trades:
        traj = t.get('energy_trajectory', [])
        sg = compute_shadow_geometry(traj, t.get('atp_bar'), t.get('alpha_fate', 'UNKNOWN'))
        shadow_results.append(sg if sg else {'shadow_class': 'NO_SHADOW'})
    aep_results = compute_aep(all_trades)
    arg_results = compute_arg_deny(all_trades, shadow_results, aep_results)
    minimal_features = extract_minimal_features(all_trades, arg_results, shadow_results, aep_results)

    bin_profiles = {}

    for key, b_raw in posterior.bins.items():
        p = b_raw['alpha'] / (b_raw['alpha'] + b_raw['beta'])
        sharpness = b_raw['alpha'] + b_raw['beta'] - 2.0
        maturity = classify_maturity(sharpness)
        direction = 'EXEC' if p >= THETA else 'DENY'

        bin_profiles[key] = {
            'alpha': b_raw['alpha'],
            'beta': b_raw['beta'],
            'n': b_raw['n'],
            'p_exec': round(p, 4),
            'sharpness': round(sharpness, 1),
            'maturity': maturity,
            'direction': direction,
            'trades': [],
            'fates': defaultdict(int),
            'wins': 0,
            'losses': 0,
            'immortal_count': 0,
            'pnl': 0.0,
            'sources': defaultdict(int),
            'regimes': defaultdict(int),
            'energy_signs': defaultdict(int),
        }

    for i, mf in enumerate(minimal_features):
        key = f"{mf['e_sign']}|{mf['de_sign']}|{mf['shadow_binary']}|{mf['arg_depth']}|{mf['regime_coarse']}|{mf['aep_binary']}"
        t = all_trades[i]
        fate = mf.get('fate', 'UNKNOWN')

        if key not in bin_profiles:
            continue

        bp = bin_profiles[key]
        bp['trades'].append(i)
        bp['fates'][fate] += 1
        bp['sources'][trade_sources[i]] += 1
        bp['regimes'][mf['regime_coarse']] += 1
        bp['energy_signs'][mf['e_sign']] += 1

        if t.get('is_win', False):
            bp['wins'] += 1
        else:
            bp['losses'] += 1
        bp['pnl'] += t.get('pnl', 0)

        if fate == 'IMMORTAL':
            bp['immortal_count'] += 1

    print(f"\n  ═══ MATURITY DISTRIBUTION ═══")
    maturity_groups = defaultdict(list)
    for key, bp in bin_profiles.items():
        maturity_groups[bp['maturity']].append(key)

    for mat in ['NEWBORN', 'INFANT', 'ADOLESCENT', 'MATURE']:
        keys = maturity_groups.get(mat, [])
        n_bins = len(keys)
        total_trades = sum(len(bin_profiles[k]['trades']) for k in keys)
        total_imm = sum(bin_profiles[k]['immortal_count'] for k in keys)
        exec_bins = sum(1 for k in keys if bin_profiles[k]['direction'] == 'EXEC')
        deny_bins = n_bins - exec_bins
        print(f"\n  {mat:>12s}: {n_bins:3d} bins  ({exec_bins} EXEC / {deny_bins} DENY)")
        print(f"               {total_trades:4d} trades, {total_imm:3d} IMMORTAL")

    print(f"\n  ═══ IMMORTAL CONCENTRATION MAP ═══")
    imm_bins = [(key, bp) for key, bp in bin_profiles.items() if bp['immortal_count'] > 0]
    imm_bins.sort(key=lambda x: x[1]['immortal_count'], reverse=True)

    total_immortal = sum(bp['immortal_count'] for _, bp in imm_bins)
    print(f"  Total IMMORTAL trades: {total_immortal}")
    print(f"  Bins with IMMORTAL: {len(imm_bins)} / {len(bin_profiles)}")

    print(f"\n  {'Rank':>4s}  {'Bin Key':>50s}  {'IMM':>4s}  {'n':>4s}  {'sharp':>6s}  {'mat':>12s}  {'p':>6s}  {'WR':>5s}")
    print(f"  {'─'*4}  {'─'*50}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*12}  {'─'*6}  {'─'*5}")

    cumulative_imm = 0
    for rank, (key, bp) in enumerate(imm_bins):
        wr = bp['wins'] / max(bp['wins'] + bp['losses'], 1) * 100
        cumulative_imm += bp['immortal_count']
        pct = cumulative_imm / total_immortal * 100
        marker = ' ◀ INFANT' if bp['maturity'] in ('NEWBORN', 'INFANT') else ''
        print(f"  {rank+1:>4d}  {key:>50s}  {bp['immortal_count']:>4d}  {bp['n']:>4d}  "
              f"{bp['sharpness']:>5.0f}  {bp['maturity']:>12s}  {bp['p_exec']:>.3f}  "
              f"{wr:>4.0f}%{marker}")
        if pct >= 95 and rank >= 5:
            remaining = len(imm_bins) - rank - 1
            if remaining > 0:
                print(f"       ... {remaining} more bins with ≤1 IMMORTAL each ...")
            break

    print(f"\n  ═══ INFANT + EXEC BINS: RARE ALPHA TARGETS ═══")
    print(f"  (bins where p≥θ AND sharpness<{INFANT_THRESHOLD} — potential growth targets)")

    rare_alpha_bins = []
    for key, bp in bin_profiles.items():
        if bp['direction'] == 'EXEC' and bp['maturity'] in ('NEWBORN', 'INFANT'):
            rare_alpha_bins.append((key, bp))
    rare_alpha_bins.sort(key=lambda x: x[1]['immortal_count'], reverse=True)

    total_rare_imm = sum(bp['immortal_count'] for _, bp in rare_alpha_bins)
    total_rare_trades = sum(len(bp['trades']) for _, bp in rare_alpha_bins)
    print(f"\n  Rare alpha bins: {len(rare_alpha_bins)}")
    print(f"  Trades in rare alpha bins: {total_rare_trades}")
    print(f"  IMMORTAL in rare alpha bins: {total_rare_imm} / {total_immortal} ({total_rare_imm/max(total_immortal,1)*100:.0f}%)")

    for key, bp in rare_alpha_bins:
        wr = bp['wins'] / max(bp['wins'] + bp['losses'], 1) * 100
        fates_str = ', '.join(f"{f}:{c}" for f, c in sorted(bp['fates'].items(), key=lambda x: -x[1]))
        sources_str = ', '.join(f"{s}:{c}" for s, c in sorted(bp['sources'].items(), key=lambda x: -x[1]))
        print(f"\n  ▸ {key}")
        print(f"    p={bp['p_exec']:.3f}  sharp={bp['sharpness']:.0f}  n={bp['n']}  WR={wr:.0f}%  IMM={bp['immortal_count']}")
        print(f"    α={bp['alpha']:.0f}  β={bp['beta']:.0f}  PnL=${bp['pnl']:,.0f}")
        print(f"    Fates: {fates_str}")
        print(f"    Sources: {sources_str}")

    print(f"\n  ═══ OBSERVATION PRIORITY MAP ═══")
    print(f"  (bins ranked by observation need: high IMMORTAL density + low sharpness)")

    obs_priority = []
    for key, bp in bin_profiles.items():
        if bp['direction'] != 'EXEC':
            continue
        imm_density = bp['immortal_count'] / max(len(bp['trades']), 1)
        growth_need = max(0, DECIDED_THRESHOLD - bp['sharpness'])
        priority_score = imm_density * growth_need
        obs_priority.append({
            'key': key,
            'priority_score': round(priority_score, 2),
            'imm_density': round(imm_density, 3),
            'growth_need': round(growth_need, 1),
            'sharpness': bp['sharpness'],
            'immortal_count': bp['immortal_count'],
            'n': bp['n'],
            'maturity': bp['maturity'],
            'p_exec': bp['p_exec'],
        })

    obs_priority.sort(key=lambda x: x['priority_score'], reverse=True)

    print(f"\n  {'Rank':>4s}  {'Priority':>8s}  {'ImmDens':>7s}  {'Need':>5s}  {'Sharp':>6s}  {'IMM':>4s}  {'n':>4s}  {'Mat':>12s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*4}  {'─'*4}  {'─'*12}")

    for rank, op in enumerate(obs_priority[:15]):
        print(f"  {rank+1:>4d}  {op['priority_score']:>8.2f}  {op['imm_density']:>7.3f}  "
              f"{op['growth_need']:>5.1f}  {op['sharpness']:>5.0f}  "
              f"{op['immortal_count']:>4d}  {op['n']:>4d}  {op['maturity']:>12s}")
        print(f"        {op['key']}")

    print(f"\n  ═══ GROWTH FORECAST ═══")
    print(f"  (how many additional observations needed to mature each infant EXEC bin)")

    for key, bp in rare_alpha_bins:
        needed = max(0, DECIDED_THRESHOLD - bp['sharpness'])
        if needed > 0:
            current_rate = len(bp['trades']) / max(len(all_trades), 1)
            if current_rate > 0:
                bars_needed = int(needed / current_rate)
            else:
                bars_needed = float('inf')
            print(f"  {key}")
            print(f"    sharpness: {bp['sharpness']:.0f} → {DECIDED_THRESHOLD}  (need +{needed:.0f} obs)")
            print(f"    current rate: {current_rate:.4f} ({len(bp['trades'])}/{len(all_trades)})")
            if bars_needed != float('inf'):
                print(f"    estimated bars to mature: ~{bars_needed:,d}")
            else:
                print(f"    estimated bars to mature: ∞ (no observations yet)")

    exp63_dir = os.path.join(EVIDENCE_DIR, 'exp63_infant_bin_map')
    os.makedirs(exp63_dir, exist_ok=True)

    serializable_profiles = {}
    for key, bp in bin_profiles.items():
        serializable_profiles[key] = {
            'alpha': bp['alpha'],
            'beta': bp['beta'],
            'n': bp['n'],
            'p_exec': bp['p_exec'],
            'sharpness': bp['sharpness'],
            'maturity': bp['maturity'],
            'direction': bp['direction'],
            'n_trades': len(bp['trades']),
            'wins': bp['wins'],
            'losses': bp['losses'],
            'immortal_count': bp['immortal_count'],
            'pnl': round(bp['pnl'], 2),
            'fates': dict(bp['fates']),
            'sources': dict(bp['sources']),
        }

    exp63_data = {
        'experiment': 'EXP-63 Infant Bin Growth Map',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'n_bins': len(all_bins),
        'n_trades': len(all_trades),
        'total_immortal': total_immortal,
        'maturity_summary': {
            mat: {
                'n_bins': len(keys),
                'n_trades': sum(len(bin_profiles[k]['trades']) for k in keys),
                'n_immortal': sum(bin_profiles[k]['immortal_count'] for k in keys),
            }
            for mat, keys in maturity_groups.items()
        },
        'rare_alpha_summary': {
            'n_bins': len(rare_alpha_bins),
            'n_trades': total_rare_trades,
            'n_immortal': total_rare_imm,
            'pct_immortal': round(total_rare_imm / max(total_immortal, 1) * 100, 1),
        },
        'observation_priority': obs_priority[:20],
        'bin_profiles': serializable_profiles,
    }

    with open(os.path.join(exp63_dir, 'infant_bin_map.json'), 'w') as f:
        json.dump(exp63_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-63 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Rare things are invisible due to scarce data. More observation reveals them.'")


if __name__ == '__main__':
    main()
