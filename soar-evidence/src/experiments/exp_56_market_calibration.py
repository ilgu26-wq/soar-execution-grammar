#!/usr/bin/env python3
"""
EXP-56: MARKET-SPECIFIC CALIBRATION
================================================================
"Laws are fixed. Only thresholds are fitted per market."

MOTIVATION:
  EXP-55 result:
  - NQ: ALL PASS (FPR 8.4~10.8%, FalseExec 7.4~8.7%)
  - ES: 4/5 PASS (FPR 27.0%, FalseExec 21.0%)
  - BTC: 4/5 PASS (FPR 29.7%, FalseExec 15.4%)

  failure person/of: fast_death_check criticalvalue NQ reference/criteriato/as corrected
  → PRUNE_MIN_CONDITIONS, force_dir_con, mae_mfe_ratio per-market adjustment needed

DESIGN:
  Phase 1 — Threshold Grid Search
    PRUNE_MIN_CONDITIONS: [2, 3]
    force_dir_con_threshold: [0.2, 0.3, 0.4]
    mae_mfe_ratio: [0.3, 0.5, 0.7]

  Phase 2 — Best Config Selection
    Optimization target: FPR ≤ 15% + M4 PASS (FalseExec ≤ 10%) + Laws PASS

  Phase 3 — Validation
    mostever/instance configto/as full pipeline execution, law conservation confirmed

CONSTRAINT:
  Law structure (Sharp Boundary, ECL, Fate Sep) is absolutely immutable.
  pruning intensityonly adjustment.
"""

import sys, os, json, time, itertools
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.v2_locked import (
    DD_THRESHOLD, CONSEC_LOSS_PAUSE, CONSEC_LOSS_COOLDOWN_BARS,
    VOL_GATE_HIGH, HIGH_VOL_DD_MULTIPLIER, WARMUP_BARS,
    STOP_TICKS, MIN_SIGNAL_GAP, ER_FLOOR, Z_NORM_THRESHOLD, ER_MULTIPLIER,
    LOOKBACK_BARS, DenyReason, validate_lock, LOCK_VERSION,
)
from core.regime_layer import classify_regime, RegimeMemory, RegimeLogger
from core.force_engine import ForceEngine
from core.alpha_layer import AlphaGenerator, AlphaMemory
from core.motion_watchdog import analyze_trade_motion
from core.pheromone_drift import PheromoneDriftLayer
from core.alpha_termination import detect_atp, classify_alpha_fate
from core.alpha_energy import compute_energy_trajectory, summarize_energy
from core.failure_commitment import (
    evaluate_failure_trajectory, FCLMemory,
    evaluate_alpha_trajectory, AOCLMemory,
    stabilized_orbit_evaluation,
)
from experiments.exp_51_cross_market import (
    compute_shadow_geometry, compute_aep, compute_arg_deny,
    extract_minimal_features, apply_sharp_boundary, measure_invariants,
    NumpyEncoder,
)
from experiments.exp_54_deferred_compute import (
    tier2_soft_death_check, fast_orbit_energy,
    deferred_recovery_check, classify_hard_dead_zone,
    _build_pruned_trade, _build_alive_trade,
    DEFERRED_E_UPPER, DEFERRED_E_LOWER, INSTANT_DEAD_THRESHOLD,
)
from experiments.exp_55_global_stress_test import (
    load_1min_bars, generate_signals_multi, compute_regime_features,
)

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
EPS = 1e-10


def fast_death_check_calibrated(df, bar_idx, direction, force_state, tick_size,
                                 min_conditions=2, force_threshold=0.3, mae_ratio=0.5):
    n = len(df)
    if bar_idx + 2 >= n:
        return False, 0, {}

    close = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    if direction == 1:
        mfe_1 = (highs[bar_idx + 1] - close[bar_idx]) / tick_size
        mae_1 = (close[bar_idx] - lows[bar_idx + 1]) / tick_size
    else:
        mfe_1 = (close[bar_idx] - lows[bar_idx + 1]) / tick_size
        mae_1 = (highs[bar_idx + 1] - close[bar_idx]) / tick_size

    e_excursion_1 = mfe_1 - mae_1
    move_1 = (close[bar_idx + 1] - close[bar_idx]) * direction / tick_size
    move_2 = (close[bar_idx + 2] - close[bar_idx]) * direction / tick_size
    de_sign = move_2 - move_1
    force_dir_con = getattr(force_state, 'dir_consistency', 0) or 0

    conditions = 0
    details = {}

    if e_excursion_1 <= 0:
        conditions += 1
        details['E_sign_neg'] = True

    if de_sign <= 0 and move_1 <= 0:
        conditions += 1
        details['dE_falling'] = True

    if force_dir_con < force_threshold:
        conditions += 1
        details['force_misaligned'] = True

    if mae_1 > 0 and mfe_1 < mae_1 * mae_ratio:
        conditions += 1
        details['mae_dominant'] = True

    details['n_conditions'] = conditions
    details['e_excursion_1'] = round(e_excursion_1, 2)
    details['move_1'] = round(move_1, 2)
    details['move_2'] = round(move_2, 2)
    details['de_sign'] = round(de_sign, 2)
    details['force_dir_con'] = round(force_dir_con, 3)

    is_dead = conditions >= min_conditions
    return is_dead, conditions, details


def run_calibration_pipeline(signals, df, tick_value, tick_size,
                              min_conditions=2, force_threshold=0.3, mae_ratio=0.5):
    n = len(df)
    dE_vals = df['dE'].values.astype(float)
    vol_short = np.zeros(n)
    vol_long = np.zeros(n)
    for i in range(n):
        lo = max(0, i - 20)
        lo2 = max(0, i - 100)
        vol_short[i] = np.std(dE_vals[lo:i+1]) if i >= 1 else 0
        vol_long[i] = np.std(dE_vals[lo2:i+1]) if i >= 1 else 0

    sig_map = {}
    for sig in signals:
        sig_map.setdefault(sig['bar_idx'], []).append(sig)
    sig_indices = sorted(sig_map.keys())
    sig_count_cache = {}
    for idx in sig_indices:
        lo = max(0, idx - 100)
        sig_count_cache[idx] = sum(1 for s in sig_indices if lo <= s <= idx)

    force_engine = ForceEngine()
    force_engine.compute_all(df)
    alpha_mem = AlphaMemory()
    alpha_gen = AlphaGenerator(memory=alpha_mem)
    pdl = PheromoneDriftLayer()
    alpha_gen.set_pheromone_layer(pdl)

    equity = 100_000.0
    peak = equity
    max_dd = 0.0
    consec_losses = 0
    paused_until = -1
    trades = []
    regime_mem = RegimeMemory()
    regime_log = RegimeLogger()
    fcl_mem = FCLMemory()
    aocl_mem = AOCLMemory()

    stats = {
        'total': 0, 'denied': 0,
        'hard_dead': 0, 'soft_dead': 0, 'alive': 0,
        'false_prunes_hard': 0, 'false_prunes_soft': 0,
        'orbit_calls_full': 0, 'orbit_calls_fast': 0,
        'deferred_total': 0, 'deferred_promoted': 0, 'deferred_dead': 0,
    }

    for i in range(n):
        if i < WARMUP_BARS or i not in sig_map:
            continue
        force_state = force_engine.get_state(i)
        vr_r, sd_r, abr_r, da_r = compute_regime_features(df, i, sig_count_cache.get(i, 0))
        regime_label = classify_regime(vr_r, sd_r, abr_r, da_r)
        alpha_gen.set_regime(regime_label)
        alpha_candidates = alpha_gen.generate(df, i, force_state)

        for sig in sig_map[i]:
            stats['total'] += 1
            pnl_per = sig['pnl_ticks'] * tick_value
            pnl_total = pnl_per * 1
            dd_pct = (peak - equity) / peak if peak > 0 else 0
            vr = vol_short[i] / (vol_long[i] + EPS)
            matching_alphas = [c for c in alpha_candidates if c.direction == sig['direction']]
            deny_reasons = []
            if dd_pct > DD_THRESHOLD:
                deny_reasons.append(DenyReason.DD_BREACH)
            if consec_losses >= CONSEC_LOSS_PAUSE and i < paused_until:
                deny_reasons.append(DenyReason.CONSEC_LOSS_PAUSE)
            vol_regime = 'HIGH' if vr > VOL_GATE_HIGH else 'MID'
            if vol_regime == 'HIGH' and dd_pct > DD_THRESHOLD * HIGH_VOL_DD_MULTIPLIER:
                deny_reasons.append(DenyReason.HIGH_VOL_CAUTION)

            if deny_reasons:
                stats['denied'] += 1
                continue

            size_hint = regime_mem.get_size_hint(regime_label)
            effective_pnl = pnl_total * size_hint
            is_win = sig['pnl_ticks'] > 0

            is_dead, n_conds, death_details = fast_death_check_calibrated(
                df, i, sig['direction'], force_state, tick_size,
                min_conditions=min_conditions, force_threshold=force_threshold, mae_ratio=mae_ratio
            )

            tier = 'ALIVE'
            deferred_details = None

            if is_dead:
                e1 = death_details.get('e_excursion_1', -999)
                zone = classify_hard_dead_zone(e1)

                if zone == 'DEFERRED':
                    stats['deferred_total'] += 1
                    promote, def_details = deferred_recovery_check(df, i, sig['direction'], tick_size)
                    deferred_details = def_details
                    if promote:
                        tier = 'ALIVE'
                        stats['deferred_promoted'] += 1
                    else:
                        tier = 'HARD_DEAD'
                        stats['deferred_dead'] += 1
                else:
                    tier = 'HARD_DEAD'
            elif n_conds == 1:
                is_soft, soft_details = tier2_soft_death_check(df, i, sig['direction'], death_details, tick_size)
                death_details.update(soft_details)
                if is_soft:
                    tier = 'SOFT_DEAD'
                else:
                    tier = 'ALIVE'

            if tier in ('HARD_DEAD', 'SOFT_DEAD'):
                if tier == 'HARD_DEAD':
                    stats['hard_dead'] += 1
                    if is_win:
                        stats['false_prunes_hard'] += 1
                else:
                    stats['soft_dead'] += 1
                    if is_win:
                        stats['false_prunes_soft'] += 1

                stats['orbit_calls_fast'] += 1
                fast_be, fast_etraj, fast_esum = fast_orbit_energy(df, i, sig['direction'], tick_size)
                fast_atp = detect_atp(fast_be, 'NEUTRAL', 'TIE', aocl_oct=None)
                fast_fate = classify_alpha_fate(fast_atp, 'NEUTRAL')
                trade = _build_pruned_trade(sig, effective_pnl, is_win, regime_label, size_hint,
                                           fast_be, fast_etraj, fast_esum, fast_atp, fast_fate,
                                           tier, death_details, deferred_details)
                trades.append(trade)
            else:
                stats['alive'] += 1
                stats['orbit_calls_full'] += 1
                for ac in matching_alphas:
                    alpha_mem.record_allowed(ac.alpha_type, effective_pnl, ac.condition, regime=regime_label)
                motion = analyze_trade_motion(df, i, sig['direction'], tick_size=tick_size, force_state=force_state)
                stab_result = stabilized_orbit_evaluation(df, i, sig['direction'], force_state, tick_size=tick_size)
                trade = _build_alive_trade(sig, effective_pnl, is_win, regime_label, size_hint,
                                          matching_alphas, alpha_mem, motion, fcl_mem, aocl_mem,
                                          force_state, df, i, stab_result)
                trades.append(trade)

            equity += effective_pnl
            if equity > peak:
                peak = equity
            dd_now = (peak - equity) / peak if peak > 0 else 0
            if dd_now > max_dd:
                max_dd = dd_now
            if effective_pnl > 0:
                consec_losses = 0
            else:
                consec_losses += 1
                if consec_losses >= CONSEC_LOSS_PAUSE:
                    paused_until = i + CONSEC_LOSS_COOLDOWN_BARS
            regime_mem.record(regime_label, effective_pnl, is_win)

    n_traded = stats['hard_dead'] + stats['soft_dead'] + stats['alive']
    total_pruned = stats['hard_dead'] + stats['soft_dead']
    false_prunes = stats['false_prunes_hard'] + stats['false_prunes_soft']
    fpr = false_prunes / max(total_pruned, 1) * 100 if total_pruned > 0 else 0
    orbit_saved = stats['orbit_calls_fast'] / max(stats['orbit_calls_full'] + stats['orbit_calls_fast'], 1) * 100

    return trades, {
        'n_traded': n_traded,
        'n_pruned': total_pruned,
        'n_alive': stats['alive'],
        'false_prunes': false_prunes,
        'fpr': round(fpr, 1),
        'orbit_saved_pct': round(orbit_saved, 1),
        'max_dd': round(max_dd * 100, 2),
        'final_equity': round(equity, 2),
        'wr': round(sum(1 for t in trades if t['is_win']) / max(len(trades), 1) * 100, 1),
        'net_pnl': round(sum(t['pnl'] for t in trades), 2),
    }


def compute_invariants_quick(trades_list):
    if len(trades_list) < 10:
        return None
    shadow_results_list = []
    for t in trades_list:
        traj = t.get('energy_trajectory', [])
        sg = compute_shadow_geometry(traj, t.get('atp_bar'), t.get('alpha_fate', 'UNKNOWN'))
        shadow_results_list.append(sg if sg else {'shadow_class': 'NO_SHADOW'})
    aep_results_list = compute_aep(trades_list)
    arg_results = compute_arg_deny(trades_list, shadow_results_list, aep_results_list)
    minimal_features = extract_minimal_features(trades_list, arg_results, shadow_results_list, aep_results_list)
    sharp_p_exec = apply_sharp_boundary(minimal_features)
    invariants = measure_invariants(minimal_features, sharp_p_exec, aep_results_list)
    return invariants


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-56: MARKET-SPECIFIC CALIBRATION")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'Laws are fixed. Only thresholds are fitted per market.'")
    print("=" * 70)

    markets = {}

    es_files = sorted([f for f in os.listdir(ASSETS) if 'CME_MINI_ES1!' in f and f.endswith('.csv') and ',_1_' in f])
    if es_files:
        es_dfs = []
        for ef in es_files[:3]:
            edf = load_1min_bars(os.path.join(ASSETS, ef), tick_size=0.25)
            if edf is not None:
                es_dfs.append(edf)
        if es_dfs:
            es_combined = pd.concat(es_dfs, ignore_index=True).sort_values('time').reset_index(drop=True)
            es_combined['dE'] = es_combined['close'].diff().fillna(0)
            es_combined['d2E'] = es_combined['dE'].diff().fillna(0)
            rm = es_combined['close'].rolling(50, min_periods=1).mean()
            rs = es_combined['close'].rolling(50, min_periods=1).std().fillna(1)
            es_combined['z_norm'] = (es_combined['close'] - rm) / (rs + EPS)
            r20 = es_combined['close'].rolling(20, min_periods=1)
            es_combined['dc'] = ((es_combined['close'] - r20.min()) / (r20.max() - r20.min() + EPS)).fillna(0.5)
            sv = es_combined['close'].rolling(20, min_periods=1).std()
            lv = es_combined['close'].rolling(100, min_periods=1).std()
            es_combined['vol_ratio'] = (sv / (lv + EPS)).fillna(1.0)
            es_combined['ch_range'] = ((es_combined['high'] - es_combined['low']) / 0.25).fillna(0)
            markets['ES'] = {'df': es_combined, 'tick_size': 0.25, 'tick_value': 12.50}

    btc_files = sorted([f for f in os.listdir(ASSETS) if 'CME_BTC1!' in f and f.endswith('.csv') and ',_1_' in f])
    if btc_files:
        btc_dfs = []
        for bf in btc_files[:3]:
            bdf = load_1min_bars(os.path.join(ASSETS, bf), tick_size=5.0)
            if bdf is not None:
                btc_dfs.append(bdf)
        if btc_dfs:
            btc_combined = pd.concat(btc_dfs, ignore_index=True).sort_values('time').reset_index(drop=True)
            btc_combined['dE'] = btc_combined['close'].diff().fillna(0)
            btc_combined['d2E'] = btc_combined['dE'].diff().fillna(0)
            rm = btc_combined['close'].rolling(50, min_periods=1).mean()
            rs = btc_combined['close'].rolling(50, min_periods=1).std().fillna(1)
            btc_combined['z_norm'] = (btc_combined['close'] - rm) / (rs + EPS)
            r20 = btc_combined['close'].rolling(20, min_periods=1)
            btc_combined['dc'] = ((btc_combined['close'] - r20.min()) / (r20.max() - r20.min() + EPS)).fillna(0.5)
            sv = btc_combined['close'].rolling(20, min_periods=1).std()
            lv = btc_combined['close'].rolling(100, min_periods=1).std()
            btc_combined['vol_ratio'] = (sv / (lv + EPS)).fillna(1.0)
            btc_combined['ch_range'] = ((btc_combined['high'] - btc_combined['low']) / 5.0).fillna(0)
            markets['BTC'] = {'df': btc_combined, 'tick_size': 5.0, 'tick_value': 5.0}

    print(f"\n  Markets loaded: {list(markets.keys())}")

    param_grid = {
        'min_conditions': [2, 3],
        'force_threshold': [0.2, 0.3, 0.4],
        'mae_ratio': [0.3, 0.5, 0.7],
    }

    combos = list(itertools.product(
        param_grid['min_conditions'],
        param_grid['force_threshold'],
        param_grid['mae_ratio'],
    ))

    print(f"  Grid: {len(combos)} combinations per market")
    print(f"  Optimizing for: FPR ≤ 15% + M4 PASS (FalseExec ≤ 10%) + orbit_saved ≥ 30%")

    all_results = {}

    for market_name, market_data in markets.items():
        print(f"\n  ═══ {market_name} CALIBRATION ═══")
        df = market_data['df']
        tick_size = market_data['tick_size']
        tick_value = market_data['tick_value']

        signals = generate_signals_multi(df, tick_size=tick_size)
        print(f"  Signals: {len(signals)}")

        if len(signals) < 20:
            print(f"  ⚠️ INSUFFICIENT SIGNALS — skipped")
            continue

        grid_results = []

        for idx, (mc, ft, mr) in enumerate(combos):
            trades, run_stats = run_calibration_pipeline(
                signals, df, tick_value, tick_size,
                min_conditions=mc, force_threshold=ft, mae_ratio=mr
            )

            invariants = None
            false_exec = None
            all_pass = False

            if len(trades) >= 10:
                invariants = compute_invariants_quick(trades)
                if invariants:
                    false_exec = invariants.get('false_exec_rate', 999)
                    all_pass = invariants.get('all_pass', False)

            result = {
                'min_conditions': mc,
                'force_threshold': ft,
                'mae_ratio': mr,
                'n_trades': len(trades),
                'fpr': run_stats['fpr'],
                'orbit_saved': run_stats['orbit_saved_pct'],
                'false_exec': false_exec,
                'all_pass': all_pass,
                'wr': run_stats['wr'],
                'net_pnl': run_stats['net_pnl'],
                'max_dd': run_stats['max_dd'],
                'sharp_gap': invariants['sharp_gap'] if invariants else None,
                'fate_sep': invariants['fate_separation'] if invariants else None,
                'aep_median': invariants['aep_median'] if invariants else None,
            }
            grid_results.append(result)

            status = '✅' if all_pass else '  '
            fpr_mark = '✓' if run_stats['fpr'] <= 15 else '✗'
            fe_mark = '✓' if (false_exec is not None and false_exec <= 10) else '✗'
            os_mark = '✓' if run_stats['orbit_saved_pct'] >= 30 else '✗'

            print(f"  [{idx+1:2d}/{len(combos)}] mc={mc} ft={ft:.1f} mr={mr:.1f}  "
                  f"FPR={run_stats['fpr']:>5.1f}%{fpr_mark}  FE={false_exec if false_exec else '---':>5}%{fe_mark}  "
                  f"OS={run_stats['orbit_saved_pct']:>4.1f}%{os_mark}  "
                  f"WR={run_stats['wr']:>4.1f}%  PnL=${run_stats['net_pnl']:>8,.0f}  {status}")

        candidates = [r for r in grid_results
                      if r['all_pass'] and r['fpr'] <= 15 and r['orbit_saved'] >= 30]

        if not candidates:
            candidates = [r for r in grid_results
                          if r['all_pass'] and r['fpr'] <= 20]

        if not candidates:
            candidates = sorted(grid_results, key=lambda x: (x['fpr'], -(x['orbit_saved'] or 0)))[:3]

        best = min(candidates, key=lambda x: x['fpr'])

        print(f"\n  ── {market_name} BEST CONFIG ──")
        print(f"     min_conditions: {best['min_conditions']}")
        print(f"     force_threshold: {best['force_threshold']}")
        print(f"     mae_ratio: {best['mae_ratio']}")
        print(f"     FPR: {best['fpr']:.1f}%  FalseExec: {best['false_exec']}%")
        print(f"     Orbit saved: {best['orbit_saved']:.1f}%")
        print(f"     Laws: {'✅ ALL PASS' if best['all_pass'] else '❌ BROKEN'}")
        print(f"     WR: {best['wr']:.1f}%  PnL: ${best['net_pnl']:,.0f}")

        all_results[market_name] = {
            'grid_results': grid_results,
            'best_config': best,
            'n_signals': len(signals),
            'n_bars': len(df),
        }

    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  CALIBRATION RESULTS                                            ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")

    calibration_data = {
        'experiment': 'EXP-56 Market-Specific Calibration',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'nq_baseline': {
            'min_conditions': 2,
            'force_threshold': 0.3,
            'mae_ratio': 0.5,
            'status': 'v3_runtime_frozen',
        },
    }

    for market_name, mr in all_results.items():
        best = mr['best_config']
        calibration_data[market_name] = {
            'n_bars': mr['n_bars'],
            'n_signals': mr['n_signals'],
            'best_config': {
                'min_conditions': best['min_conditions'],
                'force_threshold': best['force_threshold'],
                'mae_ratio': best['mae_ratio'],
            },
            'metrics': {
                'fpr': best['fpr'],
                'false_exec': best['false_exec'],
                'orbit_saved': best['orbit_saved'],
                'all_pass': best['all_pass'],
                'wr': best['wr'],
                'net_pnl': best['net_pnl'],
                'sharp_gap': best['sharp_gap'],
                'fate_sep': best['fate_sep'],
                'aep_median': best['aep_median'],
            },
            'improvement': {
                'fpr_before': 27.0 if market_name == 'ES' else 29.7,
                'fpr_after': best['fpr'],
                'all_pass_before': False,
                'all_pass_after': best['all_pass'],
            },
        }

        status = '✅ CALIBRATED' if best['all_pass'] else '⚠️ PARTIAL'
        print(f"\n  {market_name}: {status}")
        print(f"    Config: mc={best['min_conditions']} ft={best['force_threshold']} mr={best['mae_ratio']}")
        print(f"    FPR: {27.0 if market_name == 'ES' else 29.7:.1f}% → {best['fpr']:.1f}%")
        print(f"    Laws: {'ALL PASS' if best['all_pass'] else 'BROKEN'}")

    exp56_dir = os.path.join(EVIDENCE_DIR, 'exp56_market_calibration')
    os.makedirs(exp56_dir, exist_ok=True)
    exp56_path = os.path.join(exp56_dir, 'calibration_results.json')
    with open(exp56_path, 'w') as f:
        json.dump(calibration_data, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-56 Calibration Saved ---")
    print(f"  {exp56_path}")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
