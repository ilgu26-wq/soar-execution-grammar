#!/usr/bin/env python3
"""
EXP-54: DEFERRED COMPUTE FOR MARGINAL HARD_DEAD
================================================================
"die as if does not lookonly still/yet observation does not end  worldline"

MOTIVATION:
  EXP-53 Result: 21 ZOMBIE mismatches, all occurring in HARD_DEAD tier
  E_excursion(bar1) mean=-3.67, median=-3.00, range=[-10, -1]
  → bar 0-1 number/can knowonly, orbit analysis  recoverydo "quantum boundary oscillation" worldline

  This is  law .
  This is an experiment that aligns already existing laws along the time axis.

DESIGN:
  HARD_DEAD E_excursion(bar1) reference/criteriato/as granularity:

  Zone A — INSTANT DEAD:  E_excursion(bar1) ≤ -10
    → That istime HARD_DEAD. recovery Impossible.

  Zone B — CONFIRMED DEAD: E_excursion(bar1) ∈ (-10, -3]
    → HARD_DEAD maintained. recovery probability extremely low.

  Zone C — DEFERRED:       E_excursion(bar1) ∈ (-3, -1]
    → short-term deferred orbit window (3 bars)
    → When E turns positive at bar 2-4, promoted to ALIVE → full orbit
    → positive/amounttransition none surface/if DEAD confirmation

  Zone D — ALIVE:          E_excursion(bar1) > -1
    → existing logic. (already ALIVE determination)

  Tier 2 SOFT_DEAD: EXP-53 youto/as maintained

HYPOTHESES:
  H-54a: Deferred zone ZOMBIE 21  50% above recoverymake?
  H-54b: False prune rate < EXP-53 (10.0%)?
  H-54c: law conservation?
  H-54d: addition orbit cost  savings's/of 10% ?
  H-54e: recoverybecome trade's/of WR >  HARD_DEAD WR?

CRITICAL CONSTRAINT:
  law conservationmust become does.
    none. observation time allocationonly change.
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
DEFERRED_WINDOW = 3

DEFERRED_E_UPPER = -1.0
DEFERRED_E_LOWER = -3.0
INSTANT_DEAD_THRESHOLD = -10.0


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


def deferred_recovery_check(df, bar_idx, direction, tick_size=0.25):
    n = len(df)
    close = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    entry_price = close[bar_idx]

    window_end = min(bar_idx + 1 + DEFERRED_WINDOW, n)
    if window_end <= bar_idx + 1:
        return False, {}

    recovery_detected = False
    max_recovery_e = -999.0
    recovery_bar = None

    for k in range(2, window_end - bar_idx):
        idx = bar_idx + k
        if idx >= n:
            break

        if direction == 1:
            mfe_k = (max(highs[bar_idx + 1:idx + 1]) - entry_price) / tick_size
            mae_k = (entry_price - min(lows[bar_idx + 1:idx + 1])) / tick_size
        else:
            mfe_k = (entry_price - min(lows[bar_idx + 1:idx + 1])) / tick_size
            mae_k = (max(highs[bar_idx + 1:idx + 1]) - entry_price) / tick_size

        e_k = mfe_k - mae_k

        if e_k > max_recovery_e:
            max_recovery_e = e_k

        if e_k > 0:
            recovery_detected = True
            recovery_bar = k
            break

    move_last = (close[min(bar_idx + DEFERRED_WINDOW, n - 1)] - entry_price) * direction / tick_size
    de_trend = move_last - ((close[bar_idx + 1] - entry_price) * direction / tick_size)

    details = {
        'recovery_detected': recovery_detected,
        'recovery_bar': recovery_bar,
        'max_recovery_e': round(max_recovery_e, 2),
        'de_trend': round(de_trend, 2),
        'window_bars': window_end - bar_idx - 1,
    }

    promote = recovery_detected or (max_recovery_e > DEFERRED_E_UPPER and de_trend > 0)

    return promote, details


def classify_hard_dead_zone(e_excursion_1):
    if e_excursion_1 <= INSTANT_DEAD_THRESHOLD:
        return 'INSTANT_DEAD'
    elif e_excursion_1 <= DEFERRED_E_LOWER:
        return 'CONFIRMED_DEAD'
    elif e_excursion_1 <= DEFERRED_E_UPPER:
        return 'DEFERRED'
    else:
        return 'NOT_HARD_DEAD'


def _build_pruned_trade(sig, effective_pnl, is_win, regime_label, size_hint,
                        fast_be, fast_etraj, fast_esum, fast_atp, fast_fate,
                        tier, death_details, deferred_details=None):
    trade = {
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
    }
    if deferred_details:
        trade['_deferred_details'] = deferred_details
    return trade


def _build_alive_trade(sig, effective_pnl, is_win, regime_label, size_hint,
                       matching_alphas, alpha_mem, motion, fcl_mem, aocl_mem,
                       force_state, df, i, stab_result):
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

    trade = {
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
    }

    atp_result = detect_atp(stab_result['bar_evolution'], stab_result['dominant_orbit'],
                            stab_result['first_leader'], aocl_oct=stab_result['stab_aocl_oct'])
    alpha_fate = classify_alpha_fate(atp_result, stab_result['dominant_orbit'])
    trade['atp_bar'] = atp_result['atp_bar']
    trade['alpha_fate'] = alpha_fate
    trade['had_aocl_lead'] = atp_result['had_aocl_lead']

    energy_traj = compute_energy_trajectory(stab_result['bar_evolution'],
                                            force_dir_con=force_state.dir_consistency)
    energy_summary = summarize_energy(energy_traj, atp_bar=atp_result['atp_bar'])
    trade['energy_trajectory'] = energy_traj
    trade['energy_summary'] = energy_summary

    return trade


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
        'deferred_total': 0, 'deferred_promoted': 0, 'deferred_dead': 0,
        'deferred_orbit_calls': 0,
        'instant_dead': 0, 'confirmed_dead': 0,
        'false_prunes_deferred_dead': 0,
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
            deferred_details = None

            if mode == 'deferred':
                is_dead, n_conds, death_details = fast_death_check(df, i, sig['direction'], force_state)

                if is_dead:
                    e1 = death_details.get('e_excursion_1', -999)
                    zone = classify_hard_dead_zone(e1)

                    if zone == 'DEFERRED':
                        stats['deferred_total'] += 1
                        stats['deferred_orbit_calls'] += 1
                        promote, def_details = deferred_recovery_check(df, i, sig['direction'])
                        deferred_details = def_details
                        deferred_details['zone'] = zone
                        deferred_details['e_excursion_1'] = e1

                        if promote:
                            tier = 'ALIVE'
                            stats['deferred_promoted'] += 1
                            death_details['deferred_promoted'] = True
                        else:
                            tier = 'HARD_DEAD'
                            stats['deferred_dead'] += 1
                            if is_win:
                                stats['false_prunes_deferred_dead'] += 1
                    else:
                        tier = 'HARD_DEAD'
                        death_details['zone'] = zone
                        if zone == 'INSTANT_DEAD':
                            stats['instant_dead'] += 1
                        else:
                            stats['confirmed_dead'] += 1

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

                trade = _build_pruned_trade(sig, effective_pnl, is_win, regime_label, size_hint,
                                           fast_be, fast_etraj, fast_esum, fast_atp, fast_fate,
                                           tier, death_details, deferred_details)
                trades.append(trade)
            else:
                stats['alive'] += 1
                stats['orbit_calls_full'] += 1

                for ac in matching_alphas:
                    alpha_mem.record_allowed(ac.alpha_type, effective_pnl, ac.condition, regime=regime_label)
                motion = analyze_trade_motion(df, i, sig['direction'], tick_size=0.25, force_state=force_state)
                for ac in matching_alphas:
                    alpha_mem.record_motion(ac.alpha_type, ac.condition, regime_label, motion['motion_tag'])

                stab_result = stabilized_orbit_evaluation(df, i, sig['direction'], force_state, tick_size=0.25)

                trade = _build_alive_trade(sig, effective_pnl, is_win, regime_label, size_hint,
                                          matching_alphas, alpha_mem, motion, fcl_mem, aocl_mem,
                                          force_state, df, i, stab_result)
                trades.append(trade)

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
    print(f"  EXP-54: DEFERRED COMPUTE FOR MARGINAL HARD_DEAD")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'die as if does not lookonly still/yet observation does not end  worldline'")
    print("=" * 70)

    print(f"\n  Design:")
    print(f"    Zone A — INSTANT DEAD:   E_excursion(bar1) ≤ {INSTANT_DEAD_THRESHOLD}")
    print(f"    Zone B — CONFIRMED DEAD: E_excursion(bar1) ∈ ({INSTANT_DEAD_THRESHOLD}, {DEFERRED_E_LOWER}]")
    print(f"    Zone C — DEFERRED:       E_excursion(bar1) ∈ ({DEFERRED_E_LOWER}, {DEFERRED_E_UPPER}]")
    print(f"    Zone D — ALIVE:          E_excursion(bar1) > {DEFERRED_E_UPPER}")
    print(f"    Deferred window: {DEFERRED_WINDOW} bars")

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

    print(f"\n  ═══ PHASE 2: DEFERRED PIPELINE ═══")
    t2 = time.time()
    trades_def, denied_def, stats_def = _run_pipeline(signals, bars_df, 5.0, mode='deferred')
    t_def = time.time() - t2
    print(f"  Deferred pipeline: {len(trades_def)} trades, {len(denied_def)} denied in {t_def:.2f}s")

    n_total = stats_def['total']
    n_denied = stats_def['denied']
    n_hard = stats_def['hard_dead']
    n_soft = stats_def['soft_dead']
    n_alive = stats_def['alive']
    n_traded = n_hard + n_soft + n_alive
    total_pruned = n_hard + n_soft
    prune_rate = total_pruned / max(n_traded, 1) * 100
    false_prunes_total = stats_def['false_prunes_hard'] + stats_def['false_prunes_soft']
    false_prune_rate = false_prunes_total / max(total_pruned, 1) * 100

    speedup = t_full / max(t_def, 0.001)
    orbit_saved_pct = stats_def['orbit_calls_fast'] / max(stats_def['orbit_calls_full'] + stats_def['orbit_calls_fast'], 1) * 100

    print(f"\n  ═══ PRUNING STATISTICS ═══")
    print(f"  Total signals:         {n_total}")
    print(f"  Gate-denied:           {n_denied}")
    print(f"  Traded:                {n_traded}")
    print(f"    HARD DEAD:           {n_hard}")
    print(f"      Instant dead:      {stats_def['instant_dead']}")
    print(f"      Confirmed dead:    {stats_def['confirmed_dead']}")
    print(f"      Deferred → dead:   {stats_def['deferred_dead']}")
    print(f"    SOFT DEAD:           {n_soft}")
    print(f"    ALIVE:               {n_alive}")
    print(f"      Deferred → alive:  {stats_def['deferred_promoted']}")
    print(f"      Soft → alive:      {stats_def['soft_dead_promoted']}")
    print(f"  Total pruned:          {total_pruned} ({prune_rate:.1f}%)")
    print(f"  False prunes:          {false_prunes_total} ({false_prune_rate:.1f}%)")

    print(f"\n  ═══ DEFERRED ZONE ANALYSIS ═══")
    n_deferred = stats_def['deferred_total']
    n_def_promoted = stats_def['deferred_promoted']
    n_def_dead = stats_def['deferred_dead']
    recovery_rate = n_def_promoted / max(n_deferred, 1) * 100

    print(f"  Deferred candidates:   {n_deferred}")
    print(f"    Promoted → ALIVE:    {n_def_promoted} ({recovery_rate:.1f}%)")
    print(f"    Confirmed → DEAD:    {n_def_dead} ({100 - recovery_rate:.1f}%)")
    print(f"  Deferred orbit calls:  {stats_def['deferred_orbit_calls']}")

    promoted_trades = [t for t in trades_def if t.get('_deferred_details', {}).get('deferred_promoted')]
    if promoted_trades:
        prom_wr = sum(1 for t in promoted_trades if t['is_win']) / len(promoted_trades) * 100
        prom_pnl = sum(t['pnl'] for t in promoted_trades)
        print(f"\n  Promoted trade quality:")
        print(f"    WR: {prom_wr:.1f}%")
        print(f"    PnL: ${prom_pnl:.0f}")

    deferred_still_dead = [t for t in trades_def if t.get('_tier') == 'HARD_DEAD'
                           and t.get('_deferred_details', {}).get('zone') == 'DEFERRED'
                           and not t.get('_death_details', {}).get('deferred_promoted')]

    print(f"\n  ═══ SPEED BENCHMARK ═══")
    print(f"  Full pipeline:         {t_full:.2f}s")
    print(f"  Deferred pipeline:     {t_def:.2f}s")
    print(f"  Speedup:               {speedup:.2f}x")
    print(f"  Full orbit calls:      {stats_def['orbit_calls_full']}")
    print(f"  Fast orbit calls:      {stats_def['orbit_calls_fast']}")
    print(f"  Orbit call savings:    {orbit_saved_pct:.1f}%")
    deferred_overhead = stats_def['deferred_orbit_calls'] / max(stats_def['orbit_calls_fast'] + stats_def['orbit_calls_full'], 1) * 100
    print(f"  Deferred overhead:     {deferred_overhead:.1f}% of total calls")

    print(f"\n  ═══ TIER ANALYSIS ═══")
    pruned_trades = [t for t in trades_def if t.get('_pruned')]
    alive_trades = [t for t in trades_def if not t.get('_pruned')]
    hard_trades = [t for t in pruned_trades if t.get('_tier') == 'HARD_DEAD']
    soft_trades = [t for t in pruned_trades if t.get('_tier') == 'SOFT_DEAD']

    for label, subset in [('HARD DEAD', hard_trades), ('SOFT DEAD', soft_trades), ('ALIVE', alive_trades)]:
        if not subset:
            print(f"  {label:>12s}: n=0")
            continue
        wr = sum(1 for t in subset if t['is_win']) / len(subset) * 100
        pnl = sum(t['pnl'] for t in subset)
        print(f"  {label:>12s}: n={len(subset):>4d}  WR={wr:>5.1f}%  PnL=${pnl:>8.0f}")

    print(f"\n  ═══ PHASE 3: ZOMBIE MISMATCH COMPARISON ═══")
    alive_full_fates = {}
    for t in trades_full:
        key = (str(t['time']), t['price'], t['direction'])
        alive_full_fates[key] = t.get('alpha_fate', 'UNKNOWN')

    zombie_mismatches_exp53 = []
    zombie_mismatches_exp54 = []
    for t in trades_def:
        if not t.get('_pruned'):
            continue
        key = (str(t['time']), t['price'], t['direction'])
        full_fate = alive_full_fates.get(key, 'UNKNOWN')
        if full_fate not in ('STILLBORN', 'TERMINATED'):
            zombie_mismatches_exp54.append({
                'time': str(t['time'])[:19],
                'direction': t['direction'],
                'pnl_ticks': t['pnl_ticks'],
                'full_fate': full_fate,
                'is_win': t['is_win'],
                'tier': t.get('_tier'),
                'e_excursion_1': t.get('_death_details', {}).get('e_excursion_1', None),
                'zone': t.get('_death_details', {}).get('zone',
                        t.get('_deferred_details', {}).get('zone', 'UNKNOWN')),
            })

    for t in trades_full:
        key = (str(t['time']), t['price'], t['direction'])
        pass

    print(f"  EXP-53 baseline ZOMBIE mismatches: 21 (from previous run)")
    print(f"  EXP-54 ZOMBIE mismatches:          {len(zombie_mismatches_exp54)}")
    zombies_recovered = 21 - len(zombie_mismatches_exp54)
    print(f"  Zombies recovered by deferred:     {zombies_recovered}")

    if zombie_mismatches_exp54:
        zombie_e1_vals = [z['e_excursion_1'] for z in zombie_mismatches_exp54 if z['e_excursion_1'] is not None]
        if zombie_e1_vals:
            print(f"\n  Remaining ZOMBIE E_excursion(bar1):")
            print(f"    mean={np.mean(zombie_e1_vals):.2f}  median={np.median(zombie_e1_vals):.2f}")
            print(f"    min={min(zombie_e1_vals):.2f}  max={max(zombie_e1_vals):.2f}")

        for z in zombie_mismatches_exp54[:10]:
            print(f"    {z['time']} dir={z['direction']} pnl={z['pnl_ticks']} "
                  f"fate={z['full_fate']} E1={z['e_excursion_1']} zone={z['zone']}")

    print(f"\n  ═══ PHASE 4: LAW PRESERVATION CHECK ═══")
    print(f"  Computing invariants for FULL pipeline...")
    inv_full = compute_invariants(trades_full)
    print(f"  Computing invariants for DEFERRED pipeline...")
    inv_def = compute_invariants(trades_def)

    all_preserved = False
    if inv_full and inv_def:
        print(f"\n  {'Metric':>25s}  {'Full':>10s}  {'Deferred':>10s}  {'Delta':>10s}  {'Status':>8s}")
        print(f"  {'-'*75}")

        checks = []

        d_gap = inv_def['sharp_gap'] - inv_full['sharp_gap']
        gap_ok = abs(d_gap) <= 15
        checks.append(gap_ok)
        print(f"  {'Sharp Gap':>25s}  {inv_full['sharp_gap']:>+9.1f}%  {inv_def['sharp_gap']:>+9.1f}%  {d_gap:>+9.1f}%  {'✅' if gap_ok else '❌':>8s}")

        d_sep = inv_def['fate_separation'] - inv_full['fate_separation']
        sep_ok = abs(d_sep) <= 15
        checks.append(sep_ok)
        print(f"  {'Fate Separation':>25s}  {inv_full['fate_separation']:>+9.1f}%  {inv_def['fate_separation']:>+9.1f}%  {d_sep:>+9.1f}%  {'✅' if sep_ok else '❌':>8s}")

        d_fe = inv_def['false_exec_rate'] - inv_full['false_exec_rate']
        fe_ok = abs(d_fe) <= 5
        checks.append(fe_ok)
        print(f"  {'False Execute Rate':>25s}  {inv_full['false_exec_rate']:>9.1f}%  {inv_def['false_exec_rate']:>9.1f}%  {d_fe:>+9.1f}%  {'✅' if fe_ok else '❌':>8s}")

        d_aep = inv_def['aep_median'] - inv_full['aep_median']
        aep_ok = abs(d_aep) <= 0.05
        checks.append(aep_ok)
        print(f"  {'AEP Median':>25s}  {inv_full['aep_median']:>9.4f}   {inv_def['aep_median']:>9.4f}   {d_aep:>+9.4f}   {'✅' if aep_ok else '❌':>8s}")

        all_preserved = all(checks)
        print(f"\n  Law preservation: {'✅ ALL PRESERVED' if all_preserved else '❌ LAWS BROKEN'}")
    else:
        print(f"  ⚠️ Could not compute invariants")

    print(f"\n  ═══ HYPOTHESIS TESTS ═══")

    h54a = zombies_recovered >= 10
    print(f"\n  H-54a: Deferred zone ZOMBIE 21  50% above recoverymake?")
    print(f"    Recovered: {zombies_recovered}/21 ({zombies_recovered/21*100:.0f}%)")
    print(f"    {'SUPPORTED' if h54a else 'NOT SUPPORTED'}")

    exp53_fpr = 10.0
    h54b = false_prune_rate < exp53_fpr
    print(f"\n  H-54b: False prune rate < EXP-53 ({exp53_fpr}%)?")
    print(f"    EXP-54: {false_prune_rate:.1f}%  vs  EXP-53: {exp53_fpr}%")
    print(f"    {'SUPPORTED' if h54b else 'NOT SUPPORTED'}")

    h54c = all_preserved
    print(f"\n  H-54c: law conservation?")
    print(f"    {'SUPPORTED' if h54c else 'NOT SUPPORTED'}")

    total_calls = stats_def['orbit_calls_full'] + stats_def['orbit_calls_fast']
    deferred_overhead_pct = stats_def['deferred_orbit_calls'] / max(total_calls, 1) * 100
    h54d = deferred_overhead_pct <= 10
    print(f"\n  H-54d: addition orbit cost  savings's/of 10% ?")
    print(f"    Deferred overhead: {deferred_overhead_pct:.1f}%")
    print(f"    {'SUPPORTED' if h54d else 'NOT SUPPORTED'}")

    if promoted_trades and hard_trades:
        hard_wr = sum(1 for t in hard_trades if t['is_win']) / max(len(hard_trades), 1) * 100
        prom_wr_val = sum(1 for t in promoted_trades if t['is_win']) / len(promoted_trades) * 100
        h54e = prom_wr_val > hard_wr
        print(f"\n  H-54e: recoverybecome trade's/of WR >  HARD_DEAD WR?")
        print(f"    Promoted WR: {prom_wr_val:.1f}%  vs  HARD_DEAD WR: {hard_wr:.1f}%")
        print(f"    {'SUPPORTED' if h54e else 'NOT SUPPORTED'}")
    else:
        h54e = False
        print(f"\n  H-54e: No promoted trades to compare")

    supported = sum([h54a, h54b, h54c, h54d, h54e])
    print(f"\n  ═══ RESULT: {supported}/5 SUPPORTED ═══")

    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  ENGINEERING VERDICT                                            ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")

    if all_preserved and zombies_recovered > 0:
        print(f"  ✅ DEFERRED COMPUTE VALIDATED")
        print(f"     ZOMBIE recovery: {zombies_recovered}/21")
        print(f"     False prune: {false_prune_rate:.1f}% (EXP-53: {exp53_fpr}%)")
        print(f"     law conservation, deferred overhead {deferred_overhead_pct:.1f}%")
        print(f"\n  Final pruning architecture:")
        print(f"    E ≤ {INSTANT_DEAD_THRESHOLD:>5}  → INSTANT DEAD  ({stats_def['instant_dead']})")
        print(f"    E ∈ ({INSTANT_DEAD_THRESHOLD}, {DEFERRED_E_LOWER}] → CONFIRMED DEAD ({stats_def['confirmed_dead']})")
        print(f"    E ∈ ({DEFERRED_E_LOWER}, {DEFERRED_E_UPPER}] → DEFERRED      ({n_deferred} → {n_def_promoted} alive, {n_def_dead} dead)")
        print(f"    E > {DEFERRED_E_UPPER:>5}  → ALIVE          ({n_alive})")
    elif all_preserved:
        print(f"  ⚠️ LAWS PRESERVED but no ZOMBIE recovery")
        print(f"     Deferred zone boundaries may need adjustment")
    else:
        print(f"  ❌ LAWS BROKEN — deferred compute design needed")

    exp54_dir = os.path.join(EVIDENCE_DIR, 'exp54_deferred_compute')
    os.makedirs(exp54_dir, exist_ok=True)

    exp54_data = {
        'experiment': 'EXP-54 Deferred Compute for Marginal HARD_DEAD',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'design': {
            'instant_dead_threshold': INSTANT_DEAD_THRESHOLD,
            'deferred_lower': DEFERRED_E_LOWER,
            'deferred_upper': DEFERRED_E_UPPER,
            'deferred_window_bars': DEFERRED_WINDOW,
            'zone_A': f'E ≤ {INSTANT_DEAD_THRESHOLD} → INSTANT DEAD',
            'zone_B': f'E ∈ ({INSTANT_DEAD_THRESHOLD}, {DEFERRED_E_LOWER}] → CONFIRMED DEAD',
            'zone_C': f'E ∈ ({DEFERRED_E_LOWER}, {DEFERRED_E_UPPER}] → DEFERRED',
            'zone_D': f'E > {DEFERRED_E_UPPER} → ALIVE',
        },
        'stats': {
            'total_signals': n_total,
            'gate_denied': n_denied,
            'traded': n_traded,
            'hard_dead': n_hard,
            'instant_dead': stats_def['instant_dead'],
            'confirmed_dead': stats_def['confirmed_dead'],
            'deferred_total': n_deferred,
            'deferred_promoted': n_def_promoted,
            'deferred_dead': n_def_dead,
            'soft_dead': n_soft,
            'alive': n_alive,
            'total_pruned': total_pruned,
            'prune_rate': round(prune_rate, 1),
            'false_prunes_total': false_prunes_total,
            'false_prune_rate': round(false_prune_rate, 1),
        },
        'zombie_recovery': {
            'exp53_mismatches': 21,
            'exp54_mismatches': len(zombie_mismatches_exp54),
            'zombies_recovered': zombies_recovered,
            'recovery_rate': round(zombies_recovered / 21 * 100, 1),
            'remaining_mismatches': zombie_mismatches_exp54[:20],
        },
        'speed': {
            'full_time_s': round(t_full, 2),
            'deferred_time_s': round(t_def, 2),
            'speedup': round(speedup, 2),
            'orbit_calls_full': stats_def['orbit_calls_full'],
            'orbit_calls_fast': stats_def['orbit_calls_fast'],
            'deferred_orbit_calls': stats_def['deferred_orbit_calls'],
            'orbit_saved_pct': round(orbit_saved_pct, 1),
            'deferred_overhead_pct': round(deferred_overhead_pct, 1),
        },
        'law_preservation': {
            'full_invariants': inv_full,
            'deferred_invariants': inv_def,
            'all_preserved': all_preserved,
        },
        'hypotheses': {
            'H54a_zombie_recovery': {'threshold': '≥50% of 21', 'value': zombies_recovered, 'supported': h54a},
            'H54b_false_prune_improvement': {'threshold': f'< {exp53_fpr}%', 'value': round(false_prune_rate, 1), 'supported': h54b},
            'H54c_law_preserved': {'supported': h54c},
            'H54d_deferred_overhead': {'threshold': '≤10%', 'value': round(deferred_overhead_pct, 1), 'supported': h54d},
            'H54e_promoted_quality': {'supported': h54e},
        },
        'supported_count': f'{supported}/5',
    }

    exp54_path = os.path.join(exp54_dir, 'deferred_compute.json')
    with open(exp54_path, 'w') as f:
        json.dump(exp54_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-54 Deferred Compute Saved ---")
    print(f"  {exp54_path}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'SOARlearns when to stop thinking — without parameters, observation timeonly with.'")


if __name__ == '__main__':
    main()
