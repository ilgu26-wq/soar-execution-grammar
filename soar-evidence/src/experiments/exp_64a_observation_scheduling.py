#!/usr/bin/env python3
"""
EXP-64a: OBSERVATION SCHEDULING (observation deployment learning)
================================================================
"Precise computation only for rare bins. Less computation elsewhere. Speed+data solved simultaneously."

PREREQUISITE:
  MDU Law — 57-bin topology sealed
  EXP-63 — 23 rare alpha bins (INFANT+EXEC), 19% IMMORTAL

DESIGN:
  Two worlds, same data, processed chronologically:
  
  World A (Baseline): Update all bins uniformly
  World B (Scheduled): 
    - INFANT/NEWBORN bins: always update (full observation)
    - MATURE bins: update with probability p_sample (simulated pruning)
    - ADOLESCENT bins: always update

  Temporal processing:
    1. Start with prior posterior (α=1, β=1 for all)
    2. Process trades in chronological order
    3. Track posterior evolution at checkpoints
    4. Measure infant bin growth rate, counterexample discovery

  p_sample sweep: {1.0, 0.5, 0.25, 0.1}
    p_sample=1.0 = World A (no pruning)
    p_sample=0.1 = aggressive mature pruning (90% compute saved on mature)

METRICS:
  - Infant bin sharpness growth rate (Δsharp/100bars)
  - Counterexample discovery (first loss in 100% WR infant bins)
  - Mature bin information loss (posterior drift from full update)
  - Simulated compute savings (% of updates skipped)
  - Final evaluation: FE, SG, IMM at each p_sample level

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ p_exec value directly manipulation prohibited
  ✔️ posterior update emptyalsoonly control (observation deployment)
"""

import sys, os, json, time, copy
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

P_SAMPLE_VALUES = [1.0, 0.5, 0.25, 0.1]

INFANT_THRESHOLD = 10
DECIDED_THRESHOLD = 20

CHECKPOINT_INTERVAL = 100

np.random.seed(42)


def classify_maturity(sharpness):
    if sharpness < 3:
        return 'NEWBORN'
    elif sharpness < INFANT_THRESHOLD:
        return 'INFANT'
    elif sharpness < DECIDED_THRESHOLD:
        return 'ADOLESCENT'
    else:
        return 'MATURE'


def get_bin_key(mf):
    return f"{mf['e_sign']}|{mf['de_sign']}|{mf['shadow_binary']}|{mf['arg_depth']}|{mf['regime_coarse']}|{mf['aep_binary']}"


def run_scheduled_learning(trades, minimal_features, p_sample_mature):
    posterior = BetaPosterior(alpha_prior=1.0, beta_prior=1.0)

    checkpoints = []
    updates_done = 0
    updates_skipped = 0

    infant_first_loss = {}
    infant_history = defaultdict(list)

    for i, (t, mf) in enumerate(zip(trades, minimal_features)):
        key = get_bin_key(mf)

        b = posterior.bins[key]
        sharpness = b['alpha'] + b['beta'] - 2.0
        maturity = classify_maturity(sharpness)

        do_update = True
        if maturity == 'MATURE' and p_sample_mature < 1.0:
            if np.random.random() > p_sample_mature:
                do_update = False

        if do_update:
            is_win = t.get('is_win', False)
            posterior.update(
                e_sign=mf['e_sign'], de_sign=mf['de_sign'],
                shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
                regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
                is_win=is_win,
            )
            updates_done += 1

            if maturity in ('NEWBORN', 'INFANT'):
                if not is_win and key not in infant_first_loss:
                    infant_first_loss[key] = {
                        'trade_idx': i,
                        'sharpness_at_loss': round(sharpness, 1),
                        'p_exec_at_loss': round(b['alpha'] / (b['alpha'] + b['beta']), 4),
                    }
        else:
            updates_skipped += 1

        if maturity in ('NEWBORN', 'INFANT'):
            b_after = posterior.bins[key]
            infant_history[key].append({
                'idx': i,
                'alpha': b_after['alpha'],
                'beta': b_after['beta'],
                'sharpness': round(b_after['alpha'] + b_after['beta'] - 2.0, 1),
            })

        if (i + 1) % CHECKPOINT_INTERVAL == 0 or i == len(trades) - 1:
            all_bins = posterior.bins
            infant_bins = []
            mature_bins = []
            for k, bv in all_bins.items():
                s = bv['alpha'] + bv['beta'] - 2.0
                m = classify_maturity(s)
                p = bv['alpha'] / (bv['alpha'] + bv['beta'])
                info = {'key': k, 'sharpness': round(s, 1), 'p': round(p, 4), 'n': bv['n'], 'maturity': m}
                if m in ('NEWBORN', 'INFANT'):
                    infant_bins.append(info)
                elif m == 'MATURE':
                    mature_bins.append(info)

            checkpoints.append({
                'trade_idx': i,
                'n_infant': len(infant_bins),
                'n_mature': len(mature_bins),
                'avg_infant_sharpness': round(np.mean([b['sharpness'] for b in infant_bins]), 1) if infant_bins else 0,
                'avg_mature_sharpness': round(np.mean([b['sharpness'] for b in mature_bins]), 1) if mature_bins else 0,
                'updates_done': updates_done,
                'updates_skipped': updates_skipped,
            })

    return posterior, checkpoints, infant_first_loss, infant_history, updates_done, updates_skipped


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-64a: OBSERVATION SCHEDULING")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'Precise computation only for rare bins. Less computation elsewhere.'")
    print("=" * 70)

    nq_tick_path = os.path.join(ASSETS, 'NinjaTrader_FullOrderFlow_1770877659150.csv')
    nq_combined_path = os.path.join(ASSETS, 'NQ1_1min_combined.csv')

    all_trades = []
    if os.path.exists(nq_tick_path):
        ticks_df = load_ticks(nq_tick_path)
        nq_5s = aggregate_5s(ticks_df)
        signals = generate_signals_multi(nq_5s, tick_size=0.25)
        trades, _, _ = run_pipeline_deferred(signals, nq_5s, 5.0, 0.25)
        all_trades.extend(trades)
        print(f"  NQ_Tick_5s: {len(trades)} trades")

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None:
            signals = generate_signals_multi(nq_1m, tick_size=0.25)
            trades, _, _ = run_pipeline_deferred(signals, nq_1m, 5.0, 0.25)
            all_trades.extend(trades)
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

    print(f"\n  ═══ CHRONOLOGICAL LEARNING WITH OBSERVATION SCHEDULING ═══")

    world_results = {}

    for p_sample in P_SAMPLE_VALUES:
        label = f"p={p_sample:.2f}"
        np.random.seed(42)

        posterior, checkpoints, infant_losses, infant_history, n_done, n_skip = \
            run_scheduled_learning(all_trades, minimal_features, p_sample)

        p_exec_list = []
        for mf in minimal_features:
            p = posterior.get_p_exec(
                e_sign=mf['e_sign'], de_sign=mf['de_sign'],
                shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
                regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
            )
            p_exec_list.append(p)

        ev = evaluate_boundary(all_trades, minimal_features, p_exec_list, THETA, label)

        compute_savings = n_skip / max(n_done + n_skip, 1) * 100

        all_bins_data = posterior.bins
        infant_bins_final = []
        mature_bins_final = []
        for k, bv in all_bins_data.items():
            s = bv['alpha'] + bv['beta'] - 2.0
            m = classify_maturity(s)
            if m in ('NEWBORN', 'INFANT'):
                infant_bins_final.append({'key': k, 'sharpness': s, 'n': bv['n']})
            elif m == 'MATURE':
                mature_bins_final.append({'key': k, 'sharpness': s, 'n': bv['n']})

        print(f"\n  ── World: p_sample = {p_sample:.2f} ──")
        print(f"     Updates: {n_done:,d} done / {n_skip:,d} skipped ({compute_savings:.1f}% saved)")
        if ev:
            print(f"     Exec: {ev['n_exec']}  FE: {ev['false_exec_rate']:.1f}%  SG: {ev['sharp_gap']:+.1f}%p")
            print(f"     IMM: {ev['immortal_capture_rate']:.1f}%  PnL: ${ev['exec_pnl']:,.0f}")
        print(f"     Infant bins remaining: {len(infant_bins_final)}")
        print(f"     Mature bins: {len(mature_bins_final)}")

        if infant_losses:
            print(f"     ⚡ Counterexamples discovered in infant bins: {len(infant_losses)}")
            for k, info in sorted(infant_losses.items(), key=lambda x: x[1]['trade_idx']):
                print(f"        {k}")
                print(f"          at trade #{info['trade_idx']}, sharp={info['sharpness_at_loss']:.0f}, p={info['p_exec_at_loss']:.3f}")
        else:
            print(f"     ⬜ No counterexamples in infant bins (all 100% WR maintained)")

        growth_trajectories = {}
        for k, history in infant_history.items():
            if len(history) >= 2:
                start_s = history[0]['sharpness']
                end_s = history[-1]['sharpness']
                growth = end_s - start_s
                growth_trajectories[k] = {
                    'start_sharp': round(start_s, 1),
                    'end_sharp': round(end_s, 1),
                    'growth': round(growth, 1),
                    'observations': len(history),
                }

        if growth_trajectories:
            top_growers = sorted(growth_trajectories.items(), key=lambda x: x[1]['growth'], reverse=True)[:5]
            print(f"     Top growing infant bins:")
            for k, gt in top_growers:
                print(f"       {k}: sharp {gt['start_sharp']}→{gt['end_sharp']} (+{gt['growth']}, {gt['observations']} obs)")

        world_results[p_sample] = {
            'p_sample': p_sample,
            'updates_done': n_done,
            'updates_skipped': n_skip,
            'compute_savings_pct': round(compute_savings, 1),
            'eval': ev,
            'infant_bins_remaining': len(infant_bins_final),
            'mature_bins': len(mature_bins_final),
            'counterexamples': len(infant_losses),
            'infant_losses': {k: v for k, v in infant_losses.items()},
            'growth_trajectories': growth_trajectories,
            'checkpoints': checkpoints,
        }

    print(f"\n  ═══ SWEEP SUMMARY ═══")
    print(f"  {'p_sample':>8s}  {'Saved':>6s}  {'Exec':>5s}  {'FE':>6s}  {'SG':>7s}  {'IMM':>6s}  {'PnL':>9s}  {'InfBins':>7s}  {'CtrEx':>5s}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*5}")

    baseline_ev = world_results[1.0]['eval']

    for p_sample in P_SAMPLE_VALUES:
        wr = world_results[p_sample]
        ev = wr['eval']
        if ev:
            print(f"  {p_sample:>8.2f}  {wr['compute_savings_pct']:>5.1f}%  {ev['n_exec']:>5d}  "
                  f"{ev['false_exec_rate']:>5.1f}%  {ev['sharp_gap']:>+6.1f}  "
                  f"{ev['immortal_capture_rate']:>5.1f}%  ${ev['exec_pnl']:>8,.0f}  "
                  f"{wr['infant_bins_remaining']:>7d}  {wr['counterexamples']:>5d}")

    print(f"\n  ═══ VERDICT ═══")

    drift_acceptable = True
    for p_sample in P_SAMPLE_VALUES[1:]:
        ev = world_results[p_sample]['eval']
        if ev and baseline_ev:
            d_fe = ev['false_exec_rate'] - baseline_ev['false_exec_rate']
            d_sg = ev['sharp_gap'] - baseline_ev['sharp_gap']
            d_imm = ev['immortal_capture_rate'] - baseline_ev['immortal_capture_rate']
            if abs(d_fe) > 2.0 or abs(d_sg) > 3.0 or d_imm < -5.0:
                drift_acceptable = False

    if drift_acceptable:
        best_savings = max(P_SAMPLE_VALUES[1:], key=lambda p: world_results[p]['compute_savings_pct'])
        bs = world_results[best_savings]
        print(f"\n  ✅ OBSERVATION SCHEDULING VIABLE")
        print(f"     Best savings: p_sample={best_savings:.2f} ({bs['compute_savings_pct']:.1f}% compute saved)")
        print(f"     No significant drift in FE/SG/IMM")
        print(f"     Infant bins maintain full observation")
        print(f"\n  SCHEDULING LAW:")
        print(f"     'INFANT/NEWBORN: always observe (full compute)'")
        print(f"     'MATURE: sample at p={best_savings} (save {bs['compute_savings_pct']:.0f}% compute)'")
        print(f"     'observation density maturityalsoat inversely proportionaldoes'")
    else:
        safe_p = None
        for p_sample in [0.5, 0.25, 0.1]:
            ev = world_results[p_sample]['eval']
            if ev and baseline_ev:
                d_fe = ev['false_exec_rate'] - baseline_ev['false_exec_rate']
                d_sg = ev['sharp_gap'] - baseline_ev['sharp_gap']
                d_imm = ev['immortal_capture_rate'] - baseline_ev['immortal_capture_rate']
                if abs(d_fe) <= 2.0 and abs(d_sg) <= 3.0 and d_imm >= -5.0:
                    safe_p = p_sample
        if safe_p:
            print(f"\n  ⚠️  PARTIAL SCHEDULING — safe up to p_sample={safe_p:.2f}")
        else:
            print(f"\n  ❌ SCHEDULING NOT VIABLE — mature pruning causes drift")

    any_losses = any(world_results[p]['counterexamples'] > 0 for p in P_SAMPLE_VALUES)
    if any_losses:
        print(f"\n  ⚡ COUNTEREXAMPLE DISCOVERY:")
        print(f"     Some infant bins received their first loss!")
        print(f"     This is GOOD — it means the 100% WR was survivorship bias, now corrected.")
        loss_bins = set()
        for p in P_SAMPLE_VALUES:
            loss_bins.update(world_results[p]['infant_losses'].keys())
        print(f"     Bins with counterexamples: {len(loss_bins)}")
    else:
        print(f"\n  ⬜ NO COUNTEREXAMPLES — all infant bins maintain 100% WR")
        print(f"     This could mean:")
        print(f"     (a) True structural alpha (these bins genuinely only win)")
        print(f"     (b) Insufficient data to find losses yet")
        print(f"     → EXP-65 (cross-world prior) needed to distinguish")

    exp64_dir = os.path.join(EVIDENCE_DIR, 'exp64a_observation_scheduling')
    os.makedirs(exp64_dir, exist_ok=True)

    clean_results = {}
    for p, wr in world_results.items():
        cr = {k: v for k, v in wr.items() if k not in ('checkpoints',)}
        clean_results[str(p)] = cr

    exp64_data = {
        'experiment': 'EXP-64a Observation Scheduling',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'n_trades': len(all_trades),
        'p_sample_values': P_SAMPLE_VALUES,
        'world_results': clean_results,
        'drift_acceptable': drift_acceptable,
        'counterexamples_found': any_losses,
    }

    with open(os.path.join(exp64_dir, 'observation_scheduling_results.json'), 'w') as f:
        json.dump(exp64_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-64a Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Precise where it matters, light elsewhere. This is the economics of observation.'")


if __name__ == '__main__':
    main()
