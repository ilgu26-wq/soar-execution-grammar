#!/usr/bin/env python3
"""
EXP-52: WORLDLINE PRUNING — worldline blocking
================================================================
"already die worldline computationdifference do not does not."

MOTIVATION:
  EXP-43 proof: 66.2% trade computation not doalso done/become
  EXP-47 proof: Only 6 core states; the rest are redundant transformations
  Current problem: STILLBORN/TERMINATED also compute all orbit·shadow·AEP

  Reason for slowdown = not an optimization problem but worldline blocking problem

DESIGN:
  1. fast_death_check(): bar 0-1from O(1) death determination
     - E_excursion(bar1) ≤ 0 (MFE < MAE)
     - dE ≤ 0 (bar2 excursion ≤ bar1)
     - force direction mismatch
     - initial MAE dominanceever/instance
     condition 2 above → DEAD

  2. DEAD determination time:
     - stabilized_orbit_evaluation SKIP
     - compute_energy_trajectory SKIP
     - compute_shadow_geometry SKIP
     - compute_aep SKIP (previous trades' shadow still tracked)
     - detect_atp SKIP
     - fate = STILLBORN directly allocation

  3. ALIVE determination time: full pipeline youto/as

VALIDATION:
  - Pruned pipeline result vs Full pipeline result comparison
  - ALIVE trade's/of invariant metrics work/day whether
  - velocity/speed difference measurement
  - false pruning rate (alive even though killed determinationone/a ratio) measurement

CRITICAL CONSTRAINT:
  law conservationmust become does.
  pruningChanging Sharp Boundary / Fate Separation → failure.
"""

import sys, os, json, time
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
    progressive_orbit_evaluation,
    stabilized_orbit_evaluation,
)
from experiments.exp_51_cross_market import (
    load_ticks, aggregate_5s, generate_signals, compute_regime_features,
    compute_shadow_geometry, compute_aep, compute_arg_deny,
    extract_minimal_features, apply_sharp_boundary, measure_invariants,
    NumpyEncoder,
)

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
EPS = 1e-10
PRUNE_MIN_CONDITIONS = 2


def fast_death_check(df, bar_idx, direction, force_state, tick_size=0.25):
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

    if force_dir_con < 0.3:
        conditions += 1
        details['force_misaligned'] = True

    if mae_1 > 0 and mfe_1 < mae_1 * 0.5:
        conditions += 1
        details['mae_dominant'] = True

    is_dead = conditions >= PRUNE_MIN_CONDITIONS
    details['n_conditions'] = conditions
    details['e_excursion_1'] = round(e_excursion_1, 2)
    details['move_1'] = round(move_1, 2)
    details['move_2'] = round(move_2, 2)
    details['de_sign'] = round(de_sign, 2)
    details['force_dir_con'] = round(force_dir_con, 3)

    return is_dead, conditions, details


FAST_ORBIT_WINDOW = 10

def fast_orbit_energy(df, bar_idx, direction, tick_size=0.25):
    n = len(df)
    close = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    entry_price = close[bar_idx]
    max_k = min(FAST_ORBIT_WINDOW, n - bar_idx - 1)

    bar_evolution = []
    for k in range(1, max_k + 1):
        end_idx = min(bar_idx + k + 1, n)
        future = close[bar_idx + 1:end_idx]
        if len(future) == 0:
            break
        moves = (future - entry_price) * direction / tick_size
        mfe = float(max(0, max(moves)))
        mae = float(abs(min(0, min(moves))))

        bar_evolution.append({
            'k': k,
            'fcl_raw': 0, 'aocl_raw': 0,
            'fcl_stab': 0, 'aocl_stab': 0,
            'running_fcl': 0, 'running_aocl': 0,
            'leader': 'TIE',
            'dir_stable': False,
            'mfe': round(mfe, 2),
            'mae': round(mae, 2),
        })

    energy_traj = compute_energy_trajectory(bar_evolution)
    energy_summary = summarize_energy(energy_traj, atp_bar=0)

    return bar_evolution, energy_traj, energy_summary


def run_pipeline_full(signals, df, tick_value=5.0):
    return _run_pipeline(signals, df, tick_value, pruning=False)


def run_pipeline_pruned(signals, df, tick_value=5.0):
    return _run_pipeline(signals, df, tick_value, pruning=True)


def _run_pipeline(signals, df, tick_value, pruning):
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
    consec_losses = 0
    paused_until = -1
    trades = []
    denied = []
    regime_mem = RegimeMemory()
    regime_log = RegimeLogger()
    fcl_mem = FCLMemory()
    aocl_mem = AOCLMemory()

    stats = {'total': 0, 'pruned': 0, 'alive': 0, 'denied': 0,
             'layers_full': 0, 'layers_pruned': 0, 'false_prunes': 0}

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
                for ac in matching_alphas:
                    alpha_mem.record_denied(ac.alpha_type, ac.condition, regime=regime_label)
                    alpha_mem.record_anti_soar(ac.alpha_type, ac.condition, pnl_total, regime=regime_label)
                regime_log.append(sig['time'], regime_label, pnl_total, dd_pct, denied_reason=deny_reasons[0])
                denied.append({'time': sig['time'], 'price': sig['price'], 'pnl': round(pnl_total, 2),
                               'reasons': deny_reasons, 'regime': regime_label})
                continue

            size_hint = regime_mem.get_size_hint(regime_label)
            effective_pnl = pnl_total * size_hint
            is_win = sig['pnl_ticks'] > 0

            if pruning:
                is_dead, n_conds, death_details = fast_death_check(df, i, sig['direction'], force_state)
            else:
                is_dead = False
                n_conds = 0
                death_details = {}

            if pruning and is_dead:
                stats['pruned'] += 1
                stats['layers_pruned'] += 6

                if is_win:
                    stats['false_prunes'] += 1

                fast_be, fast_etraj, fast_esum = fast_orbit_energy(df, i, sig['direction'])

                fast_atp = detect_atp(fast_be, 'NEUTRAL', 'TIE', aocl_oct=None)
                fast_fate = classify_alpha_fate(fast_atp, 'NEUTRAL')

                trades.append({
                    'time': sig['time'], 'price': sig['price'], 'direction': sig['direction'],
                    'pnl_ticks': sig['pnl_ticks'], 'pnl': round(effective_pnl, 2),
                    'is_win': is_win, 'regime': regime_label,
                    'size_hint': round(size_hint, 2),
                    'is_committed': False, 'fcl_conditions': [],
                    'is_alpha_orbit': False, 'aocl_conditions': [],
                    'bar_evolution': fast_be,
                    'crossover_bar': None,
                    'first_leader': 'TIE',
                    'final_leader': 'TIE',
                    'contested_lean': None,
                    'dominant_orbit': 'NEUTRAL',
                    'stab_aocl_oct': None,
                    'stab_oss_fcl': None,
                    'stab_oss_aocl': None,
                    'atp_bar': fast_atp['atp_bar'],
                    'alpha_fate': fast_fate,
                    'had_aocl_lead': False,
                    'energy_trajectory': fast_etraj,
                    'energy_summary': fast_esum,
                    '_pruned': True,
                    '_death_details': death_details,
                })
            else:
                stats['alive'] += 1
                stats['layers_full'] += 6

                for ac in matching_alphas:
                    alpha_mem.record_allowed(ac.alpha_type, effective_pnl, ac.condition, regime=regime_label)
                motion = analyze_trade_motion(df, i, sig['direction'], tick_size=0.25, force_state=force_state)
                for ac in matching_alphas:
                    alpha_mem.record_motion(ac.alpha_type, ac.condition, regime_label, motion['motion_tag'])

                is_committed = False
                fcl_conditions = []
                is_alpha_orbit = False
                aocl_conditions = []
                for ac in matching_alphas:
                    rc_key = f"{ac.alpha_type}.{ac.condition}@{regime_label}"
                    fcl_mem.record_trade(rc_key)
                    aocl_mem.record_trade(rc_key)
                    committed, conds, fcl_details = evaluate_failure_trajectory(motion, force_state, df, i, sig['direction'])
                    if committed:
                        is_committed = True
                        fcl_conditions = conds
                        fcl_mem.record(rc_key, i, conds, fcl_details, effective_pnl)
                    a_committed, a_conds, aocl_details = evaluate_alpha_trajectory(motion, force_state, df, i, sig['direction'])
                    if a_committed:
                        is_alpha_orbit = True
                        aocl_conditions = a_conds
                        aocl_mem.record(rc_key, i, a_conds, aocl_details, effective_pnl)

                stab_result = stabilized_orbit_evaluation(df, i, sig['direction'], force_state, tick_size=0.25)

                trades.append({
                    'time': sig['time'], 'price': sig['price'], 'direction': sig['direction'],
                    'pnl_ticks': sig['pnl_ticks'], 'pnl': round(effective_pnl, 2),
                    'is_win': is_win, 'regime': regime_label,
                    'size_hint': round(size_hint, 2),
                    'is_committed': is_committed, 'fcl_conditions': fcl_conditions,
                    'is_alpha_orbit': is_alpha_orbit, 'aocl_conditions': aocl_conditions,
                    'bar_evolution': stab_result['bar_evolution'],
                    'crossover_bar': stab_result['crossover_bar'],
                    'first_leader': stab_result['first_leader'],
                    'final_leader': stab_result['final_leader'],
                    'contested_lean': stab_result['contested_lean'],
                    'dominant_orbit': stab_result['dominant_orbit'],
                    'stab_aocl_oct': stab_result['stab_aocl_oct'],
                    'stab_oss_fcl': stab_result['stab_oss_fcl'],
                    'stab_oss_aocl': stab_result['stab_oss_aocl'],
                    '_pruned': False,
                })

                atp_result = detect_atp(stab_result['bar_evolution'], stab_result['dominant_orbit'],
                                        stab_result['first_leader'], aocl_oct=stab_result['stab_aocl_oct'])
                alpha_fate = classify_alpha_fate(atp_result, stab_result['dominant_orbit'])
                trades[-1]['atp_bar'] = atp_result['atp_bar']
                trades[-1]['alpha_fate'] = alpha_fate
                trades[-1]['had_aocl_lead'] = atp_result['had_aocl_lead']

                energy_traj = compute_energy_trajectory(stab_result['bar_evolution'],
                                                        force_dir_con=force_state.dir_consistency)
                energy_summary = summarize_energy(energy_traj, atp_bar=atp_result['atp_bar'])
                trades[-1]['energy_trajectory'] = energy_traj
                trades[-1]['energy_summary'] = energy_summary

            equity += effective_pnl
            if equity > peak:
                peak = equity
            if effective_pnl > 0:
                consec_losses = 0
            else:
                consec_losses += 1
                if consec_losses >= CONSEC_LOSS_PAUSE:
                    paused_until = i + CONSEC_LOSS_COOLDOWN_BARS
            regime_mem.record(regime_label, effective_pnl, is_win)

    return trades, denied, stats


def compute_invariants(trades_list):
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

    tick_path = os.path.join(os.path.dirname(__file__), '..',
                             'attached_assets', 'NinjaTrader_FullOrderFlow_1770877659150.csv')
    if not os.path.exists(tick_path):
        print("ERROR: Tick data not found")
        sys.exit(1)

    print("=" * 70)
    print(f"  EXP-52: WORLDLINE PRUNING")
    print(f"  SOAR CORE {LOCK_VERSION} — worldline blocking")
    print(f"  'already die worldline computationdifference do not does not.'")
    print("=" * 70)

    print(f"\n  Loading data...")
    ticks_df = load_ticks(tick_path)
    bars_df = aggregate_5s(ticks_df)
    signals = generate_signals(bars_df)
    print(f"  Bars: {len(bars_df):,}  Signals: {len(signals)}")

    print(f"\n  ═══ PHASE 1: FULL PIPELINE (Baseline) ═══")
    t1 = time.time()
    trades_full, denied_full, stats_full = run_pipeline_full(signals, bars_df)
    t_full = time.time() - t1
    print(f"  Full pipeline: {len(trades_full)} trades, {len(denied_full)} denied in {t_full:.2f}s")

    print(f"\n  ═══ PHASE 2: PRUNED PIPELINE ═══")
    t2 = time.time()
    trades_pruned, denied_pruned, stats_pruned = run_pipeline_pruned(signals, bars_df)
    t_pruned = time.time() - t2
    print(f"  Pruned pipeline: {len(trades_pruned)} trades, {len(denied_pruned)} denied in {t_pruned:.2f}s")

    n_pruned = stats_pruned['pruned']
    n_alive = stats_pruned['alive']
    n_total = stats_pruned['total']
    n_denied = stats_pruned['denied']
    n_traded = n_pruned + n_alive
    prune_rate = n_pruned / max(n_traded, 1) * 100
    false_prune_rate = stats_pruned['false_prunes'] / max(n_pruned, 1) * 100

    layers_full = stats_pruned['layers_full']
    layers_pruned = stats_pruned['layers_pruned']
    layers_saved_pct = layers_pruned / max(layers_full + layers_pruned, 1) * 100
    speedup = t_full / max(t_pruned, 0.001)

    print(f"\n  ═══ PRUNING STATISTICS ═══")
    print(f"  Total signals:    {n_total}")
    print(f"  Gate-denied:      {n_denied}")
    print(f"  Traded:           {n_traded}")
    print(f"    Pruned (DEAD):  {n_pruned} ({prune_rate:.1f}%)")
    print(f"    Alive (FULL):   {n_alive} ({100-prune_rate:.1f}%)")
    print(f"  False prunes:     {stats_pruned['false_prunes']} ({false_prune_rate:.1f}%)")
    print(f"  Layers computed:  {layers_full} (full) + {layers_pruned} (skipped)")
    print(f"  Layer savings:    {layers_saved_pct:.1f}%")

    print(f"\n  ═══ SPEED BENCHMARK ═══")
    print(f"  Full pipeline:   {t_full:.2f}s")
    print(f"  Pruned pipeline: {t_pruned:.2f}s")
    print(f"  Speedup:         {speedup:.2f}x")
    print(f"  Time saved:      {t_full - t_pruned:.2f}s ({(1 - t_pruned/max(t_full,0.001))*100:.0f}%)")

    print(f"\n  ═══ PRUNED TRADE ANALYSIS ═══")
    pruned_trades = [t for t in trades_pruned if t.get('_pruned')]
    alive_trades = [t for t in trades_pruned if not t.get('_pruned')]
    pruned_wr = sum(1 for t in pruned_trades if t['is_win']) / max(len(pruned_trades), 1) * 100
    alive_wr = sum(1 for t in alive_trades if t['is_win']) / max(len(alive_trades), 1) * 100
    pruned_pnl = sum(t['pnl'] for t in pruned_trades)
    alive_pnl = sum(t['pnl'] for t in alive_trades)

    print(f"  Pruned trades:  n={len(pruned_trades)}  WR={pruned_wr:.1f}%  PnL=${pruned_pnl:.0f}")
    print(f"  Alive trades:   n={len(alive_trades)}  WR={alive_wr:.1f}%  PnL=${alive_pnl:.0f}")
    print(f"  Separation:     {alive_wr - pruned_wr:.1f}%p")

    if pruned_trades:
        print(f"\n  Death condition distribution:")
        cond_counts = {}
        for t in pruned_trades:
            dd = t.get('_death_details', {})
            for k in ['E_sign_neg', 'dE_falling', 'force_misaligned', 'mae_dominant']:
                if dd.get(k):
                    cond_counts[k] = cond_counts.get(k, 0) + 1
        for k, v in sorted(cond_counts.items(), key=lambda x: -x[1]):
            print(f"    {k:>20s}: {v:>4d} ({v/len(pruned_trades)*100:.1f}%)")

        depth_dist = {}
        for t in pruned_trades:
            d = t.get('_death_details', {}).get('n_conditions', 0)
            depth_dist[d] = depth_dist.get(d, 0) + 1
        print(f"\n  Death depth distribution:")
        for d in sorted(depth_dist.keys()):
            n_d = depth_dist[d]
            sub = [t for t in pruned_trades if t.get('_death_details', {}).get('n_conditions', 0) == d]
            wr_d = sum(1 for t in sub if t['is_win']) / max(len(sub), 1) * 100
            print(f"    depth={d}: {n_d:>4d} trades  WR={wr_d:.1f}%")

    print(f"\n  ═══ PHASE 3: LAW PRESERVATION CHECK ═══")
    print(f"  Computing invariants for FULL pipeline...")
    inv_full = compute_invariants(trades_full)
    print(f"  Computing invariants for PRUNED pipeline...")
    inv_pruned = compute_invariants(trades_pruned)

    if inv_full and inv_pruned:
        print(f"\n  {'Metric':>25s}  {'Full':>10s}  {'Pruned':>10s}  {'Delta':>10s}  {'Status':>8s}")
        print(f"  {'-'*70}")

        checks = []

        d_gap = inv_pruned['sharp_gap'] - inv_full['sharp_gap']
        gap_ok = abs(d_gap) <= 15
        checks.append(gap_ok)
        print(f"  {'Sharp Gap':>25s}  {inv_full['sharp_gap']:>+9.1f}%  {inv_pruned['sharp_gap']:>+9.1f}%  {d_gap:>+9.1f}%  {'✅' if gap_ok else '❌':>8s}")

        d_sep = inv_pruned['fate_separation'] - inv_full['fate_separation']
        sep_ok = abs(d_sep) <= 15
        checks.append(sep_ok)
        print(f"  {'Fate Separation':>25s}  {inv_full['fate_separation']:>+9.1f}%  {inv_pruned['fate_separation']:>+9.1f}%  {d_sep:>+9.1f}%  {'✅' if sep_ok else '❌':>8s}")

        d_fe = inv_pruned['false_exec_rate'] - inv_full['false_exec_rate']
        fe_ok = abs(d_fe) <= 5
        checks.append(fe_ok)
        print(f"  {'False Execute Rate':>25s}  {inv_full['false_exec_rate']:>9.1f}%  {inv_pruned['false_exec_rate']:>9.1f}%  {d_fe:>+9.1f}%  {'✅' if fe_ok else '❌':>8s}")

        d_aep = inv_pruned['aep_median'] - inv_full['aep_median']
        aep_ok = abs(d_aep) <= 0.05
        checks.append(aep_ok)
        print(f"  {'AEP Median':>25s}  {inv_full['aep_median']:>9.4f}   {inv_pruned['aep_median']:>9.4f}   {d_aep:>+9.4f}   {'✅' if aep_ok else '❌':>8s}")

        all_preserved = all(checks)
        print(f"\n  Law preservation: {'✅ ALL PRESERVED' if all_preserved else '❌ LAWS BROKEN'}")
    else:
        all_preserved = False
        print(f"  ⚠️ Could not compute invariants (insufficient data)")

    print(f"\n  ═══ PHASE 4: FULL vs PRUNED — fate consistency (alive trades only) ═══")
    alive_full_fates = {}
    for t in trades_full:
        key = (str(t['time']), t['price'], t['direction'])
        alive_full_fates[key] = t.get('alpha_fate', 'UNKNOWN')

    fate_matches = 0
    fate_mismatches = 0
    for t in alive_trades:
        key = (str(t['time']), t['price'], t['direction'])
        full_fate = alive_full_fates.get(key)
        if full_fate == t.get('alpha_fate'):
            fate_matches += 1
        else:
            fate_mismatches += 1

    total_fate = fate_matches + fate_mismatches
    fate_consistency = fate_matches / max(total_fate, 1) * 100
    print(f"  Alive trades fate match: {fate_matches}/{total_fate} ({fate_consistency:.1f}%)")

    full_fate_match_for_pruned = 0
    full_fate_mismatch_for_pruned = 0
    full_fates_for_pruned = {}
    for t in pruned_trades:
        key = (str(t['time']), t['price'], t['direction'])
        full_fate = alive_full_fates.get(key, 'UNKNOWN')
        full_fates_for_pruned[key] = full_fate
        if full_fate in ('STILLBORN', 'TERMINATED'):
            full_fate_match_for_pruned += 1
        else:
            full_fate_mismatch_for_pruned += 1

    total_pf = full_fate_match_for_pruned + full_fate_mismatch_for_pruned
    pruned_accuracy = full_fate_match_for_pruned / max(total_pf, 1) * 100
    print(f"  Pruned trades: {full_fate_match_for_pruned}/{total_pf} were STILLBORN/TERMINATED in full pipeline ({pruned_accuracy:.1f}%)")

    if full_fate_mismatch_for_pruned > 0:
        print(f"  ⚠️ Mismatches (pruned but not STILLBORN/TERMINATED in full):")
        for t in pruned_trades:
            key = (str(t['time']), t['price'], t['direction'])
            full_fate = full_fates_for_pruned.get(key, 'UNKNOWN')
            if full_fate not in ('STILLBORN', 'TERMINATED'):
                print(f"    {str(t['time'])[:19]} dir={t['direction']} pnl={t['pnl_ticks']} full_fate={full_fate} win={t['is_win']}")

    print(f"\n  ═══ HYPOTHESIS TESTS ═══")

    h52a = prune_rate >= 40
    print(f"\n  H-52a: ≥40% trade prunable?")
    print(f"    Pruned: {prune_rate:.1f}%")
    print(f"    {'SUPPORTED' if h52a else 'NOT SUPPORTED'}")

    h52b = false_prune_rate <= 15
    print(f"\n  H-52b: False prune rate ≤15%?")
    print(f"    False prunes: {stats_pruned['false_prunes']}/{n_pruned} ({false_prune_rate:.1f}%)")
    print(f"    {'SUPPORTED' if h52b else 'NOT SUPPORTED'}")

    h52c = all_preserved
    print(f"\n  H-52c: Law invariants preserved after pruning?")
    print(f"    {'SUPPORTED' if h52c else 'NOT SUPPORTED'}")

    h52d = speedup >= 1.2
    print(f"\n  H-52d: ≥1.2x speedup?")
    print(f"    Speedup: {speedup:.2f}x")
    print(f"    {'SUPPORTED' if h52d else 'NOT SUPPORTED'}")

    supported = sum([h52a, h52b, h52c, h52d])
    print(f"\n  ═══ RESULT: {supported}/4 SUPPORTED ═══")

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  ENGINEERING VERDICT                                    ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    if all_preserved and speedup >= 1.2:
        print(f"  ✅ WORLDLINE PRUNING VALIDATED")
        print(f"     velocity/speed ↑ {speedup:.1f}x, law conservation,  ever/instanceuse possible")
    elif all_preserved:
        print(f"  ⚠️ LAWS PRESERVED but speed gain marginal ({speedup:.2f}x)")
        print(f"     Pruning condition reinforcement needed")
    else:
        print(f"  ❌ LAWS BROKEN — Pruning condition design needed")

    exp52_dir = os.path.join(EVIDENCE_DIR, 'exp52_worldline_pruning')
    os.makedirs(exp52_dir, exist_ok=True)

    exp52_data = {
        'experiment': 'EXP-52 Worldline Pruning',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'pruning_conditions': {
            'min_conditions': PRUNE_MIN_CONDITIONS,
            'checks': ['E_sign_neg', 'dE_falling', 'force_misaligned', 'mae_dominant'],
        },
        'stats': {
            'total_signals': n_total,
            'gate_denied': n_denied,
            'traded': n_traded,
            'pruned': n_pruned,
            'alive': n_alive,
            'prune_rate': round(prune_rate, 1),
            'false_prunes': stats_pruned['false_prunes'],
            'false_prune_rate': round(false_prune_rate, 1),
        },
        'speed': {
            'full_time_s': round(t_full, 2),
            'pruned_time_s': round(t_pruned, 2),
            'speedup': round(speedup, 2),
            'layer_savings_pct': round(layers_saved_pct, 1),
        },
        'pruned_analysis': {
            'pruned_wr': round(pruned_wr, 1),
            'alive_wr': round(alive_wr, 1),
            'separation': round(alive_wr - pruned_wr, 1),
            'pruned_pnl': round(pruned_pnl, 2),
            'alive_pnl': round(alive_pnl, 2),
        },
        'law_preservation': {
            'full_invariants': inv_full,
            'pruned_invariants': inv_pruned,
            'all_preserved': all_preserved,
        },
        'fate_validation': {
            'alive_fate_consistency': round(fate_consistency, 1),
            'pruned_accuracy': round(pruned_accuracy, 1),
        },
        'hypotheses': {
            'H52a_prune_rate': {'threshold': '≥40%', 'value': round(prune_rate, 1), 'supported': h52a},
            'H52b_false_prune': {'threshold': '≤15%', 'value': round(false_prune_rate, 1), 'supported': h52b},
            'H52c_law_preserved': {'supported': h52c},
            'H52d_speedup': {'threshold': '≥1.2x', 'value': round(speedup, 2), 'supported': h52d},
        },
        'supported_count': f'{supported}/4',
    }

    exp52_path = os.path.join(exp52_dir, 'worldline_pruning.json')
    with open(exp52_path, 'w') as f:
        json.dump(exp52_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-52 Worldline Pruning Saved ---")
    print(f"  {exp52_path}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'alive exist worldlineonly leave engineering''s/of first foot/occurrence.")


if __name__ == '__main__':
    main()
