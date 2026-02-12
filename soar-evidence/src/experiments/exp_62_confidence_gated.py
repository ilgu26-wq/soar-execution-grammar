#!/usr/bin/env python3
"""
EXP-62: CONFIDENCE-GATED EXECUTION
================================================================
"representation casedoes not act without, only acts when conviction accumulates."

PREREQUISITE:
  EXP-61 v2a — MDU Law sealed: 57-bin topology most distinction unit
  bin merge/partition/design prohibited

DESIGN:
  Gate:  EXECUTE  iff  p̂ ≥ θ  AND  sharpness ≥ S
  
  sharpness = α + β - 2  (effective evidence count, excluding prior)
  
  S sweep: {0, 5, 10, 20, 40, 80, 160}
    S=0  :  baseline (sharpness limit none)
    S=5  : minimal evidence
    S=10 : moderate evidence
    S=20+: strong evidence only

  Key question:
    "how much conviction accumulated when execution allowwill thingis it??"

  expectation:
    S↑ → Exec count ↓, FalseExec ↓, IMMORTAL capture ↓
    mostever/instance S FalseExec reductionand IMMORTAL conservation's/of intersection

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ p_exec value manipulation prohibited
  ❌ θ change prohibited
  ✔️ posterior's/of shape(sharpness)only use by doing execution gate addition

SUCCESS:
  - FalseExec ↓ (nonlinear reduction)
  - IMMORTAL capture ≥ 80%
  - SharpGap maintained or improvement
  - At optimal S, achieve 'less wrong + sufficiently many'
"""

import sys, os, json, time
import numpy as np
from datetime import datetime

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

S_VALUES = [0, 5, 10, 20, 40, 80, 160]


def compute_bin_sharpness(bin_data):
    return bin_data['alpha'] + bin_data['beta'] - 2.0


def evaluate_with_confidence_gate(trades, minimal_features, posterior, theta, s_threshold, label):
    all_bins = posterior.get_all_bins()

    exec_idx = []
    deny_idx = []
    gated_idx = []

    for i, mf in enumerate(minimal_features):
        key = f"{mf['e_sign']}|{mf['de_sign']}|{mf['shadow_binary']}|{mf['arg_depth']}|{mf['regime_coarse']}|{mf['aep_binary']}"

        b = posterior.bins.get(key, None)
        if b is None:
            b = {'alpha': posterior.alpha_prior, 'beta': posterior.beta_prior, 'n': 0}

        p_exec = b['alpha'] / (b['alpha'] + b['beta'])
        sharpness = compute_bin_sharpness(b)

        if p_exec >= theta and sharpness >= s_threshold:
            exec_idx.append(i)
        elif p_exec >= theta and sharpness < s_threshold:
            gated_idx.append(i)
        else:
            deny_idx.append(i)

    if not exec_idx:
        return None

    exec_trades = [trades[i] for i in exec_idx]
    deny_trades = [trades[i] for i in deny_idx]
    gated_trades = [trades[i] for i in gated_idx]

    exec_wins = sum(1 for t in exec_trades if t['is_win'])
    exec_wr = exec_wins / len(exec_trades) * 100
    deny_wins = sum(1 for t in deny_trades if t['is_win']) if deny_trades else 0
    deny_wr = deny_wins / max(len(deny_trades), 1) * 100

    false_exec = sum(1 for t in exec_trades if not t['is_win'])
    false_exec_rate = false_exec / len(exec_trades) * 100

    exec_pnl = sum(t['pnl'] for t in exec_trades)
    deny_pnl = sum(t['pnl'] for t in deny_trades)
    gated_pnl = sum(t['pnl'] for t in gated_trades)

    imm_total = sum(1 for mf in minimal_features if mf['fate'] == 'IMMORTAL')
    imm_captured = sum(1 for i in exec_idx if minimal_features[i]['fate'] == 'IMMORTAL')
    imm_capture_rate = imm_captured / max(imm_total, 1) * 100

    imm_gated = sum(1 for i in gated_idx if minimal_features[i]['fate'] == 'IMMORTAL')

    gated_wins = sum(1 for t in gated_trades if t['is_win']) if gated_trades else 0
    gated_wr = gated_wins / max(len(gated_trades), 1) * 100

    return {
        'label': label,
        'n_exec': len(exec_idx),
        'n_deny': len(deny_idx),
        'n_gated': len(gated_idx),
        'exec_wr': round(exec_wr, 1),
        'deny_wr': round(deny_wr, 1),
        'gated_wr': round(gated_wr, 1),
        'sharp_gap': round(exec_wr - deny_wr, 1),
        'false_exec_rate': round(false_exec_rate, 1),
        'exec_pnl': round(exec_pnl, 2),
        'deny_pnl': round(deny_pnl, 2),
        'gated_pnl': round(gated_pnl, 2),
        'immortal_capture_rate': round(imm_capture_rate, 1),
        'immortal_total': imm_total,
        'immortal_captured': imm_captured,
        'immortal_gated': imm_gated,
    }


def analyze_gated_bins(posterior, theta, s_threshold):
    all_bins_raw = posterior.bins
    exec_bins = []
    gated_bins = []
    deny_bins = []

    for key, b in all_bins_raw.items():
        p = b['alpha'] / (b['alpha'] + b['beta'])
        s = compute_bin_sharpness(b)
        n = b['n']
        info = {'key': key, 'p': round(p, 4), 'sharpness': round(s, 1), 'n': n}

        if p >= theta and s >= s_threshold:
            exec_bins.append(info)
        elif p >= theta and s < s_threshold:
            gated_bins.append(info)
        else:
            deny_bins.append(info)

    return exec_bins, gated_bins, deny_bins


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-62: CONFIDENCE-GATED EXECUTION")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'representation casedoes not act without, only acts when conviction accumulates.'")
    print("=" * 70)

    posterior_path = os.path.join(EVIDENCE_DIR, 'exp57_execution_probability', 'posterior.json')
    if not os.path.exists(posterior_path):
        print("  ❌ No posterior found. Run EXP-57 first.")
        return

    posterior = BetaPosterior()
    posterior.load(posterior_path)
    all_bins = posterior.get_all_bins()
    print(f"\n  Bins: {len(all_bins)} (MDU-sealed)")

    print(f"\n  ── Bin Sharpness Distribution ──")
    sharpness_list = []
    for key, b in posterior.bins.items():
        s = compute_bin_sharpness(b)
        sharpness_list.append(s)
    sharpness_arr = np.array(sharpness_list)
    print(f"     Count: {len(sharpness_arr)}")
    print(f"     Min: {sharpness_arr.min():.0f}  Max: {sharpness_arr.max():.0f}")
    print(f"     Mean: {sharpness_arr.mean():.1f}  Median: {np.median(sharpness_arr):.1f}")
    for s_val in S_VALUES:
        n_above = np.sum(sharpness_arr >= s_val)
        print(f"     sharpness ≥ {s_val:3d}: {n_above:3d} bins ({n_above/len(sharpness_arr)*100:.0f}%)")

    nq_tick_path = os.path.join(ASSETS, 'NinjaTrader_FullOrderFlow_1770877659150.csv')
    nq_combined_path = os.path.join(ASSETS, 'NQ1_1min_combined.csv')

    all_trades = []
    all_minimal = []

    datasets_loaded = []
    if os.path.exists(nq_tick_path):
        ticks_df = load_ticks(nq_tick_path)
        nq_5s = aggregate_5s(ticks_df)
        signals = generate_signals_multi(nq_5s, tick_size=0.25)
        trades, _, _ = run_pipeline_deferred(signals, nq_5s, 5.0, 0.25)
        datasets_loaded.append(('NQ_Tick_5s', len(trades)))
        all_trades.extend(trades)

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None:
            signals = generate_signals_multi(nq_1m, tick_size=0.25)
            trades, _, _ = run_pipeline_deferred(signals, nq_1m, 5.0, 0.25)
            datasets_loaded.append(('NQ_1min', len(trades)))
            all_trades.extend(trades)

    for ds_name, n in datasets_loaded:
        print(f"  {ds_name}: {n} trades")
    print(f"  Total trades: {len(all_trades)}")

    shadow_results = []
    for t in all_trades:
        traj = t.get('energy_trajectory', [])
        sg = compute_shadow_geometry(traj, t.get('atp_bar'), t.get('alpha_fate', 'UNKNOWN'))
        shadow_results.append(sg if sg else {'shadow_class': 'NO_SHADOW'})
    aep_results = compute_aep(all_trades)
    arg_results = compute_arg_deny(all_trades, shadow_results, aep_results)
    minimal_features = extract_minimal_features(all_trades, arg_results, shadow_results, aep_results)

    print(f"\n  ═══ CONFIDENCE GATE SWEEP: S ∈ {S_VALUES} ═══")

    sweep_results = []
    baseline = None

    for s_val in S_VALUES:
        ev = evaluate_with_confidence_gate(
            all_trades, minimal_features, posterior, THETA, s_val, f'S={s_val}'
        )

        if ev is None:
            print(f"\n  ── S = {s_val} ──  ⬜ (no executions)")
            sweep_results.append({
                's': s_val, 'n_exec': 0, 'n_gated': 0, 'n_deny': len(all_trades),
                'exec_wr': 0, 'sharp_gap': 0, 'false_exec_rate': 0,
                'immortal_capture': 0, 'exec_pnl': 0, 'gated_pnl': 0,
            })
            continue

        if s_val == 0:
            baseline = ev

        exec_bins, gated_bins, deny_bins = analyze_gated_bins(posterior, THETA, s_val)

        d_fe = ev['false_exec_rate'] - baseline['false_exec_rate'] if baseline else 0
        d_sg = ev['sharp_gap'] - baseline['sharp_gap'] if baseline else 0

        print(f"\n  ── S = {s_val} ──")
        print(f"     Bins:  EXEC={len(exec_bins)}  GATED={len(gated_bins)}  DENY={len(deny_bins)}")
        print(f"     Trades: EXEC={ev['n_exec']}  GATED={ev['n_gated']}  DENY={ev['n_deny']}")
        print(f"     Exec WR: {ev['exec_wr']:.1f}%  Deny WR: {ev['deny_wr']:.1f}%  Sharp Gap: {ev['sharp_gap']:+.1f}%p")
        print(f"     FalseExec: {ev['false_exec_rate']:.1f}%  (Δ {d_fe:+.1f}%p)")
        print(f"     IMMORTAL: {ev['immortal_capture_rate']:.1f}%  ({ev['immortal_captured']}/{ev['immortal_total']})")
        if ev['immortal_gated'] > 0:
            print(f"     ⚠️  IMMORTAL gated (blocked): {ev['immortal_gated']}")
        print(f"     PnL: EXEC ${ev['exec_pnl']:,.0f}  GATED ${ev['gated_pnl']:,.0f}  DENY ${ev['deny_pnl']:,.0f}")

        if gated_bins and s_val > 0:
            print(f"     Gated bins:")
            for gb in sorted(gated_bins, key=lambda x: x['sharpness'], reverse=True)[:5]:
                print(f"       {gb['key']}  p={gb['p']:.3f}  sharp={gb['sharpness']:.0f}  n={gb['n']}")

        result = {
            's': s_val,
            'n_exec': ev['n_exec'],
            'n_gated': ev['n_gated'],
            'n_deny': ev['n_deny'],
            'n_exec_bins': len(exec_bins),
            'n_gated_bins': len(gated_bins),
            'exec_wr': ev['exec_wr'],
            'deny_wr': ev['deny_wr'],
            'gated_wr': ev['gated_wr'],
            'sharp_gap': ev['sharp_gap'],
            'false_exec_rate': ev['false_exec_rate'],
            'd_false_exec': round(d_fe, 1),
            'd_sharp_gap': round(d_sg, 1),
            'immortal_capture': ev['immortal_capture_rate'],
            'immortal_captured': ev['immortal_captured'],
            'immortal_total': ev['immortal_total'],
            'immortal_gated': ev['immortal_gated'],
            'exec_pnl': ev['exec_pnl'],
            'deny_pnl': ev['deny_pnl'],
            'gated_pnl': ev['gated_pnl'],
            'gated_bins': gated_bins,
        }
        sweep_results.append(result)

    print(f"\n  ═══ SWEEP SUMMARY ═══")
    print(f"  {'S':>5s}  {'Exec':>5s}  {'Gated':>5s}  {'ExBin':>5s}  {'ExWR':>6s}  {'SG':>7s}  {'FE':>6s}  {'ΔFE':>6s}  {'IMM':>6s}  {'PnL':>9s}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*9}")

    for r in sweep_results:
        pnl_str = f"${r.get('exec_pnl', 0):,.0f}" if r.get('exec_pnl') else '$0'
        print(f"  {r['s']:>5d}  {r['n_exec']:>5d}  {r['n_gated']:>5d}  "
              f"{r.get('n_exec_bins', 0):>5d}  {r.get('exec_wr', 0):>5.1f}%  "
              f"{r.get('sharp_gap', 0):>+6.1f}  {r.get('false_exec_rate', 0):>5.1f}%  "
              f"{r.get('d_false_exec', 0):>+5.1f}  {r.get('immortal_capture', 0):>5.1f}%  {pnl_str:>9s}")

    print(f"\n  ═══ VERDICT ═══")

    if baseline:
        optimal_s = None
        optimal_score = -float('inf')

        for r in sweep_results:
            if r['n_exec'] == 0:
                continue
            imm_ok = r.get('immortal_capture', 0) >= 80.0
            fe_ok = r.get('false_exec_rate', 0) <= baseline['false_exec_rate']

            if imm_ok and fe_ok:
                score = (r.get('immortal_capture', 0) * 0.4
                         + (100 - r.get('false_exec_rate', 0)) * 0.3
                         + r.get('sharp_gap', 0) * 0.2
                         + r['n_exec'] / max(baseline['n_exec'], 1) * 100 * 0.1)
                if score > optimal_score:
                    optimal_score = score
                    optimal_s = r['s']

        if optimal_s is not None:
            opt = next(r for r in sweep_results if r['s'] == optimal_s)
            print(f"\n  ✅ OPTIMAL CONFIDENCE GATE FOUND")
            print(f"     S* = {optimal_s}  (sharpness threshold)")
            print(f"     Exec: {opt['n_exec']}  ({opt['n_exec']}/{baseline['n_exec']} of baseline)")
            print(f"     FalseExec: {opt['false_exec_rate']:.1f}%  (baseline {baseline['false_exec_rate']:.1f}%)")
            print(f"     IMMORTAL: {opt['immortal_capture']:.1f}%")
            print(f"     SharpGap: {opt['sharp_gap']:+.1f}%p")
            print(f"     PnL: ${opt['exec_pnl']:,.0f}")
            print(f"\n  CONFIDENCE GATE LAW:")
            print(f"     'p̂ ≥ θ AND sharpness ≥ {optimal_s} → EXECUTE'")
            print(f"     'conviction {optimal_s} if below, does not act even if the direction is correct.'")
        else:
            print(f"\n  ⚠️  NO IMPROVEMENT FOUND — baseline is already optimal")
            print(f"     all confidence gate baseline versus/compared to improvement none")
            print(f"     already learningbecome p_exec sufficient precisiondo")

        fe_curve = [(r['s'], r.get('false_exec_rate', 0)) for r in sweep_results if r['n_exec'] > 0]
        imm_curve = [(r['s'], r.get('immortal_capture', 0)) for r in sweep_results if r['n_exec'] > 0]

        if len(fe_curve) >= 3:
            fe_diffs = [fe_curve[i+1][1] - fe_curve[i][1] for i in range(len(fe_curve)-1)]
            print(f"\n  FE curve (ΔFE between S steps): {[f'{d:+.1f}' for d in fe_diffs]}")

            nonlinear = any(abs(fe_diffs[i+1] - fe_diffs[i]) > 1.0
                          for i in range(len(fe_diffs)-1)) if len(fe_diffs) >= 2 else False
            if nonlinear:
                print(f"  → Non-linear FE reduction detected ✅")

    exp62_dir = os.path.join(EVIDENCE_DIR, 'exp62_confidence_gated')
    os.makedirs(exp62_dir, exist_ok=True)

    clean_results = []
    for r in sweep_results:
        cr = {k: v for k, v in r.items() if k != 'gated_bins'}
        clean_results.append(cr)

    exp62_data = {
        'experiment': 'EXP-62 Confidence-Gated Execution',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'n_bins': len(all_bins),
        'theta': THETA,
        's_values': S_VALUES,
        'baseline': {
            'n_exec': baseline['n_exec'] if baseline else 0,
            'false_exec_rate': baseline['false_exec_rate'] if baseline else 0,
            'sharp_gap': baseline['sharp_gap'] if baseline else 0,
            'immortal_capture': baseline['immortal_capture_rate'] if baseline else 0,
            'exec_pnl': baseline['exec_pnl'] if baseline else 0,
        },
        'sweep_results': clean_results,
        'optimal_s': optimal_s if baseline else None,
    }

    with open(os.path.join(exp62_dir, 'confidence_gated_results.json'), 'w') as f:
        json.dump(exp62_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-62 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Acting without conviction leads to mistakes. Acting only with conviction leads to fewer mistakes.'")


if __name__ == '__main__':
    main()
