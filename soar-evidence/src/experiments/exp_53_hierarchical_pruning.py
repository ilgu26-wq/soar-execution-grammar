#!/usr/bin/env python3
"""
EXP-53: HIERARCHICAL PRUNING — hierarchyever/instance worldline blocking
================================================================
"bar 0Stop what is dead, look one more step at what is uncertain, and compute only what is alive to the end."

MOTIVATION:
  EXP-52 proof: 45.1% prunable, law conservation, velocity/speed  (1.00x)
  Cause: binary (DEAD/ALIVE) determination → ambiguous trades are unconditionally full orbit
  do: 3stage hierarchyever/instance determinationto/as  hierarchy addition

DESIGN:
  Tier 1 — HARD DEAD (bar 0-1):
    fast_death_check() ≥2 conditions → fast_orbit_energy (MFE/MAE only)
    orbit analysis  skip. most cheap.

  Tier 2 — SOFT DEAD (bar 0-2):
    fast_death_check() == 1 condition (uncertain)
    → fast_orbit_energy 2 barsonly computation
    → E_excursion(bar2) ≤ 0 AND bar2_mfe == 0 → SOFT DEAD
    → else → promote to ALIVE

  Tier 3 — ALIVE:
    0 conditions OR Tier 2 promoted
    → full stabilized_orbit_evaluation

HYPOTHESES:
  H-53a: Tier 2captures 10%p or more additionally? (soft_dead_rate ≥ 10%p)
  H-53b:  pruning rate ≥ 50%?
  H-53c: False prune rate ≤ 15%?
  H-53d: law invariant conservation?
  H-53e: per-trade orbit cost measurement (full vs fast comparison)

VALIDATION:
  EXP-52 flat pruning result vs EXP-53 hierarchical result comparison
  ZOMBIE mismatch analysis (EXP-54 preparation)

CRITICAL CONSTRAINT:
  law conservationmust become does.
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
FAST_ORBIT_WINDOW = 10


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

    details['n_conditions'] = conditions
    details['e_excursion_1'] = round(e_excursion_1, 2)
    details['move_1'] = round(move_1, 2)
    details['move_2'] = round(move_2, 2)
    details['de_sign'] = round(de_sign, 2)
    details['force_dir_con'] = round(force_dir_con, 3)

    return conditions >= PRUNE_MIN_CONDITIONS, conditions, details


def tier2_soft_death_check(df, bar_idx, direction, death_details, tick_size=0.25):
    n = len(df)
    if bar_idx + 3 >= n:
        return False, {}

    close = df['close'].values
    entry_price = close[bar_idx]

    moves = (close[bar_idx + 1:bar_idx + 3] - entry_price) * direction / tick_size

    mfe_2 = float(max(0, max(moves)))
    mae_2 = float(abs(min(0, min(moves))))
    e_exc_2 = mfe_2 - mae_2

    move_3 = (close[bar_idx + 3] - entry_price) * direction / tick_size if bar_idx + 3 < n else 0
    move_2 = float(moves[-1]) if len(moves) > 0 else 0
    de_2to3 = move_3 - move_2

    soft_details = {
        'e_excursion_2': round(e_exc_2, 2),
        'mfe_2': round(mfe_2, 2),
        'mae_2': round(mae_2, 2),
        'de_2to3': round(de_2to3, 2),
    }

    is_soft_dead = (e_exc_2 <= 0) and (mfe_2 < 1.0) and (de_2to3 <= 0)

    return is_soft_dead, soft_details


def fast_orbit_energy(df, bar_idx, direction, tick_size=0.25):
    n = len(df)
    close = df['close'].values
    entry_price = close[bar_idx]
    max_k = min(FAST_ORBIT_WINDOW, n - bar_idx - 1)

    bar_evolution = []
    for k in range(1, max_k + 1):
        end_idx = min(bar_idx + k + 1, n)
        future = close[bar_idx + 1:end_idx]
        if len(future) == 0:
            break
        moves_arr = (future - entry_price) * direction / tick_size
        mfe = float(max(0, max(moves_arr)))
        mae = float(abs(min(0, min(moves_arr))))

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


def _run_pipeline(signals, df, tick_value, mode='full'):
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

    stats = {
        'total': 0, 'denied': 0,
        'hard_dead': 0, 'soft_dead': 0, 'alive': 0,
        'soft_dead_promoted': 0,
        'false_prunes_hard': 0, 'false_prunes_soft': 0,
        'orbit_calls_full': 0, 'orbit_calls_fast': 0,
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

            tier = 'ALIVE'
            death_details = {}

            if mode == 'hierarchical':
                is_dead, n_conds, death_details = fast_death_check(df, i, sig['direction'], force_state)

                if is_dead:
                    tier = 'HARD_DEAD'
                elif n_conds == 1:
                    is_soft, soft_details = tier2_soft_death_check(df, i, sig['direction'], death_details)
                    death_details.update(soft_details)
                    if is_soft:
                        tier = 'SOFT_DEAD'
                    else:
                        tier = 'ALIVE'
                        stats['soft_dead_promoted'] += 1

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
                    '_tier': tier,
                    '_death_details': death_details,
                })
            else:
                stats['alive'] += 1
                stats['orbit_calls_full'] += 1

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
                    '_tier': 'ALIVE',
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
    print(f"  EXP-53: HIERARCHICAL PRUNING")
    print(f"  SOAR CORE {LOCK_VERSION} — hierarchyever/instance worldline blocking")
    print(f"  'bar 0from die thing stop and, uncertainone/a thing one/a foot/occurrence more looks at.'")
    print("=" * 70)

    print(f"\n  Loading data...")
    ticks_df = load_ticks(tick_path)
    bars_df = aggregate_5s(ticks_df)
    signals = generate_signals(bars_df)
    print(f"  Bars: {len(bars_df):,}  Signals: {len(signals)}")

    print(f"\n  ═══ PHASE 1: FULL PIPELINE (Baseline) ═══")
    t1 = time.time()
    trades_full, denied_full, stats_full = _run_pipeline(signals, bars_df, 5.0, mode='full')
    t_full = time.time() - t1
    print(f"  Full pipeline: {len(trades_full)} trades, {len(denied_full)} denied in {t_full:.2f}s")

    print(f"\n  ═══ PHASE 2: HIERARCHICAL PRUNED PIPELINE ═══")
    t2 = time.time()
    trades_hier, denied_hier, stats_hier = _run_pipeline(signals, bars_df, 5.0, mode='hierarchical')
    t_hier = time.time() - t2
    print(f"  Hierarchical pipeline: {len(trades_hier)} trades, {len(denied_hier)} denied in {t_hier:.2f}s")

    n_total = stats_hier['total']
    n_denied = stats_hier['denied']
    n_hard = stats_hier['hard_dead']
    n_soft = stats_hier['soft_dead']
    n_alive = stats_hier['alive']
    n_promoted = stats_hier['soft_dead_promoted']
    n_traded = n_hard + n_soft + n_alive
    total_pruned = n_hard + n_soft
    prune_rate = total_pruned / max(n_traded, 1) * 100
    hard_rate = n_hard / max(n_traded, 1) * 100
    soft_rate = n_soft / max(n_traded, 1) * 100
    false_prunes_total = stats_hier['false_prunes_hard'] + stats_hier['false_prunes_soft']
    false_prune_rate = false_prunes_total / max(total_pruned, 1) * 100

    speedup = t_full / max(t_hier, 0.001)

    print(f"\n  ═══ PRUNING STATISTICS ═══")
    print(f"  Total signals:       {n_total}")
    print(f"  Gate-denied:         {n_denied}")
    print(f"  Traded:              {n_traded}")
    print(f"    Tier 1 HARD DEAD:  {n_hard} ({hard_rate:.1f}%)")
    print(f"    Tier 2 SOFT DEAD:  {n_soft} ({soft_rate:.1f}%)")
    print(f"    Tier 2 → ALIVE:    {n_promoted} (promoted)")
    print(f"    Tier 3 ALIVE:      {n_alive} ({n_alive/max(n_traded,1)*100:.1f}%)")
    print(f"  Total pruned:        {total_pruned} ({prune_rate:.1f}%)")
    print(f"  False prunes:        {false_prunes_total} ({false_prune_rate:.1f}%)")
    print(f"    Hard false:        {stats_hier['false_prunes_hard']}")
    print(f"    Soft false:        {stats_hier['false_prunes_soft']}")

    print(f"\n  ═══ SPEED BENCHMARK ═══")
    print(f"  Full pipeline:         {t_full:.2f}s")
    print(f"  Hierarchical pipeline: {t_hier:.2f}s")
    print(f"  Speedup:               {speedup:.2f}x")
    print(f"  Time saved:            {t_full - t_hier:.2f}s ({(1 - t_hier/max(t_full,0.001))*100:.0f}%)")
    print(f"  Full orbit calls:      {stats_hier['orbit_calls_full']}")
    print(f"  Fast orbit calls:      {stats_hier['orbit_calls_fast']}")

    orbit_saved_pct = stats_hier['orbit_calls_fast'] / max(stats_hier['orbit_calls_full'] + stats_hier['orbit_calls_fast'], 1) * 100
    print(f"  Orbit call savings:    {orbit_saved_pct:.1f}%")

    print(f"\n  ═══ TIER ANALYSIS ═══")
    pruned_trades = [t for t in trades_hier if t.get('_pruned')]
    alive_trades = [t for t in trades_hier if not t.get('_pruned')]

    hard_trades = [t for t in pruned_trades if t.get('_tier') == 'HARD_DEAD']
    soft_trades = [t for t in pruned_trades if t.get('_tier') == 'SOFT_DEAD']

    for label, subset in [('HARD DEAD', hard_trades), ('SOFT DEAD', soft_trades), ('ALIVE', alive_trades)]:
        if not subset:
            print(f"  {label:>12s}: n=0")
            continue
        wr = sum(1 for t in subset if t['is_win']) / len(subset) * 100
        pnl = sum(t['pnl'] for t in subset)
        print(f"  {label:>12s}: n={len(subset):>4d}  WR={wr:>5.1f}%  PnL=${pnl:>8.0f}")

    if hard_trades and soft_trades:
        hard_wr = sum(1 for t in hard_trades if t['is_win']) / len(hard_trades) * 100
        soft_wr = sum(1 for t in soft_trades if t['is_win']) / len(soft_trades) * 100
        alive_wr = sum(1 for t in alive_trades if t['is_win']) / max(len(alive_trades), 1) * 100
        print(f"\n  Tier WR separation:")
        print(f"    HARD→ALIVE: {alive_wr - hard_wr:+.1f}%p")
        print(f"    SOFT→ALIVE: {alive_wr - soft_wr:+.1f}%p")
        print(f"    HARD→SOFT:  {soft_wr - hard_wr:+.1f}%p")

    if pruned_trades:
        print(f"\n  Death condition distribution (HARD):")
        cond_counts = {}
        for t in hard_trades:
            dd = t.get('_death_details', {})
            for k in ['E_sign_neg', 'dE_falling', 'force_misaligned', 'mae_dominant']:
                if dd.get(k):
                    cond_counts[k] = cond_counts.get(k, 0) + 1
        for k, v in sorted(cond_counts.items(), key=lambda x: -x[1]):
            print(f"    {k:>20s}: {v:>4d} ({v/max(len(hard_trades),1)*100:.1f}%)")

    print(f"\n  ═══ PHASE 3: LAW PRESERVATION CHECK ═══")
    print(f"  Computing invariants for FULL pipeline...")
    inv_full = compute_invariants(trades_full)
    print(f"  Computing invariants for HIERARCHICAL pipeline...")
    inv_hier = compute_invariants(trades_hier)

    all_preserved = False
    if inv_full and inv_hier:
        print(f"\n  {'Metric':>25s}  {'Full':>10s}  {'Hier':>10s}  {'Delta':>10s}  {'Status':>8s}")
        print(f"  {'-'*70}")

        checks = []

        d_gap = inv_hier['sharp_gap'] - inv_full['sharp_gap']
        gap_ok = abs(d_gap) <= 15
        checks.append(gap_ok)
        print(f"  {'Sharp Gap':>25s}  {inv_full['sharp_gap']:>+9.1f}%  {inv_hier['sharp_gap']:>+9.1f}%  {d_gap:>+9.1f}%  {'✅' if gap_ok else '❌':>8s}")

        d_sep = inv_hier['fate_separation'] - inv_full['fate_separation']
        sep_ok = abs(d_sep) <= 15
        checks.append(sep_ok)
        print(f"  {'Fate Separation':>25s}  {inv_full['fate_separation']:>+9.1f}%  {inv_hier['fate_separation']:>+9.1f}%  {d_sep:>+9.1f}%  {'✅' if sep_ok else '❌':>8s}")

        d_fe = inv_hier['false_exec_rate'] - inv_full['false_exec_rate']
        fe_ok = abs(d_fe) <= 5
        checks.append(fe_ok)
        print(f"  {'False Execute Rate':>25s}  {inv_full['false_exec_rate']:>9.1f}%  {inv_hier['false_exec_rate']:>9.1f}%  {d_fe:>+9.1f}%  {'✅' if fe_ok else '❌':>8s}")

        d_aep = inv_hier['aep_median'] - inv_full['aep_median']
        aep_ok = abs(d_aep) <= 0.05
        checks.append(aep_ok)
        print(f"  {'AEP Median':>25s}  {inv_full['aep_median']:>9.4f}   {inv_hier['aep_median']:>9.4f}   {d_aep:>+9.4f}   {'✅' if aep_ok else '❌':>8s}")

        all_preserved = all(checks)
        print(f"\n  Law preservation: {'✅ ALL PRESERVED' if all_preserved else '❌ LAWS BROKEN'}")
    else:
        print(f"  ⚠️ Could not compute invariants")

    print(f"\n  ═══ PHASE 4: ZOMBIE MISMATCH ANALYSIS (EXP-54 preparation) ═══")
    alive_full_fates = {}
    for t in trades_full:
        key = (str(t['time']), t['price'], t['direction'])
        alive_full_fates[key] = t.get('alpha_fate', 'UNKNOWN')

    zombie_mismatches = []
    for t in pruned_trades:
        key = (str(t['time']), t['price'], t['direction'])
        full_fate = alive_full_fates.get(key, 'UNKNOWN')
        if full_fate not in ('STILLBORN', 'TERMINATED'):
            zombie_mismatches.append({
                'time': str(t['time'])[:19],
                'direction': t['direction'],
                'pnl_ticks': t['pnl_ticks'],
                'full_fate': full_fate,
                'is_win': t['is_win'],
                'tier': t.get('_tier'),
                'n_conditions': t.get('_death_details', {}).get('n_conditions', 0),
                'e_excursion_1': t.get('_death_details', {}).get('e_excursion_1', None),
            })

    n_zombie_mm = len(zombie_mismatches)
    zombie_winners = sum(1 for z in zombie_mismatches if z['is_win'])
    zombie_losers = n_zombie_mm - zombie_winners

    print(f"  ZOMBIE/IMMORTAL mismatches: {n_zombie_mm}")
    print(f"    Winners wrongly pruned:   {zombie_winners}")
    print(f"    Losers wrongly pruned:    {zombie_losers}")

    if zombie_mismatches:
        hard_zm = [z for z in zombie_mismatches if z['tier'] == 'HARD_DEAD']
        soft_zm = [z for z in zombie_mismatches if z['tier'] == 'SOFT_DEAD']
        print(f"    From HARD_DEAD tier:      {len(hard_zm)}")
        print(f"    From SOFT_DEAD tier:      {len(soft_zm)}")

        print(f"\n  ZOMBIE mismatch details:")
        for z in zombie_mismatches[:15]:
            print(f"    {z['time']} dir={z['direction']} pnl={z['pnl_ticks']} "
                  f"fate={z['full_fate']} win={z['is_win']} tier={z['tier']} "
                  f"E1={z['e_excursion_1']}")

        zombie_e1_vals = [z['e_excursion_1'] for z in zombie_mismatches if z['e_excursion_1'] is not None]
        if zombie_e1_vals:
            print(f"\n  ZOMBIE E_excursion(bar1) distribution:")
            print(f"    mean={np.mean(zombie_e1_vals):.2f}  median={np.median(zombie_e1_vals):.2f}")
            print(f"    min={min(zombie_e1_vals):.2f}  max={max(zombie_e1_vals):.2f}")

    print(f"\n  ═══ HYPOTHESIS TESTS ═══")

    soft_added_pct = n_soft / max(n_traded, 1) * 100
    h53a = soft_added_pct >= 5
    print(f"\n  H-53a: Tier 2 addition 5%p above grab?")
    print(f"    Soft dead: {n_soft} ({soft_added_pct:.1f}%p)")
    print(f"    {'SUPPORTED' if h53a else 'NOT SUPPORTED'}")

    h53b = prune_rate >= 50
    print(f"\n  H-53b:  pruning rate ≥ 50%?")
    print(f"    Total pruned: {prune_rate:.1f}%")
    print(f"    {'SUPPORTED' if h53b else 'NOT SUPPORTED'}")

    h53c = false_prune_rate <= 15
    print(f"\n  H-53c: False prune rate ≤ 15%?")
    print(f"    False prunes: {false_prunes_total}/{total_pruned} ({false_prune_rate:.1f}%)")
    print(f"    {'SUPPORTED' if h53c else 'NOT SUPPORTED'}")

    h53d = all_preserved
    print(f"\n  H-53d: law invariant conservation?")
    print(f"    {'SUPPORTED' if h53d else 'NOT SUPPORTED'}")

    h53e_orbit_pct = orbit_saved_pct
    h53e = h53e_orbit_pct >= 40
    print(f"\n  H-53e: Orbit call savings ≥ 40%?")
    print(f"    Savings: {h53e_orbit_pct:.1f}%")
    print(f"    {'SUPPORTED' if h53e else 'NOT SUPPORTED'}")

    supported = sum([h53a, h53b, h53c, h53d, h53e])
    print(f"\n  ═══ RESULT: {supported}/5 SUPPORTED ═══")

    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  ENGINEERING VERDICT                                            ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")
    if all_preserved and prune_rate >= 50:
        print(f"  ✅ HIERARCHICAL PRUNING VALIDATED")
        print(f"     Tier 1 (HARD): {hard_rate:.1f}%, Tier 2 (SOFT): {soft_rate:.1f}%")
        print(f"     law conservation, orbit {orbit_saved_pct:.0f}% reduction")
    elif all_preserved:
        print(f"  ⚠️ LAWS PRESERVED, prune rate {prune_rate:.1f}% (target: ≥50%)")
        print(f"     Tier 2 condition reinforcement or Tier expansion needed")
    else:
        print(f"  ❌ LAWS BROKEN — hierarchyever/instance determination design needed")

    if zombie_mismatches:
        print(f"\n  EXP-54 PREVIEW: {n_zombie_mm} ZOMBIE mismatches detected")
        print(f"     → Deferred compute target: quantum boundary oscillation trade")

    exp53_dir = os.path.join(EVIDENCE_DIR, 'exp53_hierarchical_pruning')
    os.makedirs(exp53_dir, exist_ok=True)

    exp53_data = {
        'experiment': 'EXP-53 Hierarchical Pruning',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'design': {
            'tier_1': 'HARD_DEAD: fast_death_check ≥2 conditions → fast_orbit_energy',
            'tier_2': 'SOFT_DEAD: 1 condition + bar2 E≤0 & mfe<1 & dE≤0 → fast_orbit_energy',
            'tier_3': 'ALIVE: 0 conditions or promoted → full stabilized_orbit_evaluation',
        },
        'stats': {
            'total_signals': n_total,
            'gate_denied': n_denied,
            'traded': n_traded,
            'hard_dead': n_hard,
            'soft_dead': n_soft,
            'soft_dead_promoted': n_promoted,
            'alive': n_alive,
            'total_pruned': total_pruned,
            'prune_rate': round(prune_rate, 1),
            'hard_rate': round(hard_rate, 1),
            'soft_rate': round(soft_rate, 1),
            'false_prunes_total': false_prunes_total,
            'false_prunes_hard': stats_hier['false_prunes_hard'],
            'false_prunes_soft': stats_hier['false_prunes_soft'],
            'false_prune_rate': round(false_prune_rate, 1),
        },
        'speed': {
            'full_time_s': round(t_full, 2),
            'hier_time_s': round(t_hier, 2),
            'speedup': round(speedup, 2),
            'orbit_calls_full': stats_hier['orbit_calls_full'],
            'orbit_calls_fast': stats_hier['orbit_calls_fast'],
            'orbit_saved_pct': round(orbit_saved_pct, 1),
        },
        'tier_analysis': {
            'hard_wr': round(sum(1 for t in hard_trades if t['is_win']) / max(len(hard_trades), 1) * 100, 1) if hard_trades else None,
            'soft_wr': round(sum(1 for t in soft_trades if t['is_win']) / max(len(soft_trades), 1) * 100, 1) if soft_trades else None,
            'alive_wr': round(sum(1 for t in alive_trades if t['is_win']) / max(len(alive_trades), 1) * 100, 1),
            'hard_pnl': round(sum(t['pnl'] for t in hard_trades), 2) if hard_trades else None,
            'soft_pnl': round(sum(t['pnl'] for t in soft_trades), 2) if soft_trades else None,
            'alive_pnl': round(sum(t['pnl'] for t in alive_trades), 2),
        },
        'law_preservation': {
            'full_invariants': inv_full,
            'hier_invariants': inv_hier,
            'all_preserved': all_preserved,
        },
        'zombie_analysis': {
            'total_mismatches': n_zombie_mm,
            'winners_wrongly_pruned': zombie_winners,
            'losers_wrongly_pruned': zombie_losers,
            'mismatches': zombie_mismatches[:20],
        },
        'hypotheses': {
            'H53a_soft_dead_rate': {'threshold': '≥5%p', 'value': round(soft_added_pct, 1), 'supported': h53a},
            'H53b_total_prune_rate': {'threshold': '≥50%', 'value': round(prune_rate, 1), 'supported': h53b},
            'H53c_false_prune': {'threshold': '≤15%', 'value': round(false_prune_rate, 1), 'supported': h53c},
            'H53d_law_preserved': {'supported': h53d},
            'H53e_orbit_savings': {'threshold': '≥40%', 'value': round(h53e_orbit_pct, 1), 'supported': h53e},
        },
        'supported_count': f'{supported}/5',
    }

    exp53_path = os.path.join(exp53_dir, 'hierarchical_pruning.json')
    with open(exp53_path, 'w') as f:
        json.dump(exp53_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-53 Hierarchical Pruning Saved ---")
    print(f"  {exp53_path}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Hierarchical blocking: stop the dead, look one more step at the uncertain, only the alive go to the end.'")


if __name__ == '__main__':
    main()
