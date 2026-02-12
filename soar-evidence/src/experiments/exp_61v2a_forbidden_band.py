#!/usr/bin/env python3
"""
EXP-61 v2a: FORBIDDEN BAND COALESCENCE — MARGIN SWEEP
================================================================
"θ near phase transition boundary. same is a signalso different above(phase)is."

EXP-61 FINDING:
  POS|RISING|SHADOW|D2+D0 (p≈0.53) merge → FalseExec +7.5%p surge
  Root cause: borderline bin marginal trades EXECUTEto/as attraction

NEW LAW CANDIDATE — Coalescence Forbidden Band (CFB):
  |p̂ - θ| < m person/of bin merge prohibited
  merged backatalso |p̂_merge - θ| ≥ m must done

DESIGN:
  margin sweep: m ∈ {0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20}
  each mat about:
    1. Forbidden band ever/instanceuse  merge  mountain/living fixed
    2. Greedy merge execution
    3. Post-merge evaluation

  pass condition:
    SharpGap drift ≥ -3%p
    FalseExec drift ≤ +2%p
    IMMORTAL capture ≥ 85%
    merge count ≥ 1
"""

import sys, os, json, time, copy
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
from experiments.exp_57_execution_probability import evaluate_boundary
from experiments.exp_61_bin_coalescence import (
    symmetric_kl, are_adjacent, compute_bin_fe_rate,
    assign_trades_to_bins, parse_bin_key,
    KL_EPSILON, FE_DELTA, FEATURE_NAMES,
)
from observer.learning.p_exec_posterior import BetaPosterior

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')

THETA = 0.5

MARGINS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

PASS_CRITERIA = {
    'sharp_gap_drift_min': -3.0,
    'false_exec_drift_max': 2.0,
    'immortal_capture_min': 85.0,
    'min_merges': 1,
}


def run_merge_with_margin(all_bins, bin_trades, margin, posterior_original):
    active_bins = {k: v for k, v in all_bins.items() if v['n'] >= 3}
    active_keys = list(active_bins.keys())

    adjacent_pairs = []
    for i, key_a in enumerate(active_keys):
        for key_b in active_keys[i+1:]:
            is_adj, dim = are_adjacent(key_a, key_b)
            if is_adj:
                adjacent_pairs.append((key_a, key_b, dim))

    merge_candidates = []
    rejected_by_cfb = 0
    rejected_by_post_cfb = 0

    for key_a, key_b, dim in adjacent_pairs:
        bin_a = all_bins[key_a]
        bin_b = all_bins[key_b]

        skl = symmetric_kl(bin_a['alpha'], bin_a['beta'], bin_b['alpha'], bin_b['beta'])

        p_a = bin_a['alpha'] / (bin_a['alpha'] + bin_a['beta'])
        p_b = bin_b['alpha'] / (bin_b['alpha'] + bin_b['beta'])
        same_direction = (p_a >= THETA) == (p_b >= THETA)

        fe_a = compute_bin_fe_rate(bin_trades.get(key_a, []))
        fe_b = compute_bin_fe_rate(bin_trades.get(key_b, []))
        fe_similar = abs(fe_a - fe_b) < FE_DELTA

        base_eligible = (skl < KL_EPSILON) and same_direction and fe_similar

        if not base_eligible:
            continue

        in_forbidden_a = abs(p_a - THETA) < margin
        in_forbidden_b = abs(p_b - THETA) < margin
        if in_forbidden_a or in_forbidden_b:
            rejected_by_cfb += 1
            continue

        merged_alpha = bin_a['alpha'] + bin_b['alpha'] - posterior_original.alpha_prior
        merged_beta = bin_a['beta'] + bin_b['beta'] - posterior_original.beta_prior
        p_merged = merged_alpha / (merged_alpha + merged_beta)

        if abs(p_merged - THETA) < margin:
            rejected_by_post_cfb += 1
            continue

        merge_candidates.append({
            'key_a': key_a,
            'key_b': key_b,
            'dim': dim,
            'skl': skl,
            'p_a': p_a,
            'p_b': p_b,
            'p_merged': p_merged,
            'merged_alpha': merged_alpha,
            'merged_beta': merged_beta,
            'n_a': bin_a['n'],
            'n_b': bin_b['n'],
        })

    merged_bins_set = set()
    merge_plan = []
    for mc in sorted(merge_candidates, key=lambda x: x['skl']):
        if mc['key_a'] in merged_bins_set or mc['key_b'] in merged_bins_set:
            continue
        merge_plan.append(mc)
        merged_bins_set.add(mc['key_a'])
        merged_bins_set.add(mc['key_b'])

    posterior_merged = BetaPosterior(
        alpha_prior=posterior_original.alpha_prior,
        beta_prior=posterior_original.beta_prior,
    )
    for key, b in all_bins.items():
        posterior_merged.bins[key] = copy.deepcopy(b)

    merge_details = []
    for mc in merge_plan:
        key_a = mc['key_a']
        key_b = mc['key_b']

        parts_a = parse_bin_key(key_a)
        parts_b = parse_bin_key(key_b)
        dim = mc['dim']
        merged_parts = list(parts_a)
        merged_parts[dim] = f"{parts_a[dim]}+{parts_b[dim]}"
        merged_key = '|'.join(merged_parts)

        posterior_merged.bins[merged_key] = {
            'alpha': mc['merged_alpha'],
            'beta': mc['merged_beta'],
            'n': mc['n_a'] + mc['n_b'],
        }
        del posterior_merged.bins[key_a]
        del posterior_merged.bins[key_b]

        merge_details.append({
            'merged_key': merged_key,
            'from': [key_a, key_b],
            'p_merged': round(mc['p_merged'], 4),
            'dim_name': FEATURE_NAMES[dim],
        })

    return {
        'n_merges': len(merge_plan),
        'n_rejected_cfb': rejected_by_cfb,
        'n_rejected_post_cfb': rejected_by_post_cfb,
        'n_post_bins': len(posterior_merged.get_all_bins()),
        'merge_details': merge_details,
        'posterior_merged': posterior_merged,
    }


def get_p_exec_for_trade(mf, posterior_obj):
    key = f"{mf['e_sign']}|{mf['de_sign']}|{mf['shadow_binary']}|{mf['arg_depth']}|{mf['regime_coarse']}|{mf['aep_binary']}"
    if key in posterior_obj.bins:
        b = posterior_obj.bins[key]
        return b['alpha'] / (b['alpha'] + b['beta'])
    for bk, bv in posterior_obj.bins.items():
        parts = bk.split('|')
        if len(parts) != 6:
            continue
        match = True
        mf_vals = [mf['e_sign'], mf['de_sign'], mf['shadow_binary'],
                   mf['arg_depth'], mf['regime_coarse'], mf['aep_binary']]
        for i, (pk, mk) in enumerate(zip(parts, mf_vals)):
            if '+' in pk:
                if mk not in pk.split('+'):
                    match = False
                    break
            elif pk != mk:
                match = False
                break
        if match:
            return bv['alpha'] / (bv['alpha'] + bv['beta'])
    return posterior_obj.alpha_prior / (posterior_obj.alpha_prior + posterior_obj.beta_prior)


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-61 v2a: FORBIDDEN BAND COALESCENCE — MARGIN SWEEP")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'θ near phase transition boundary. same is a signalso different aboveis.'")
    print("=" * 70)

    posterior_path = os.path.join(EVIDENCE_DIR, 'exp57_execution_probability', 'posterior.json')
    if not os.path.exists(posterior_path):
        print("  ❌ No posterior found. Run EXP-57 first.")
        return

    posterior_original = BetaPosterior()
    posterior_original.load(posterior_path)
    all_bins = posterior_original.get_all_bins()
    print(f"\n  Original bins: {len(all_bins)}")

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
    bin_trades = assign_trades_to_bins(all_trades, minimal_features)

    pre_p_exec = []
    for mf in minimal_features:
        p = posterior_original.get_p_exec(
            e_sign=mf['e_sign'], de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
        )
        pre_p_exec.append(p)

    pre_eval = evaluate_boundary(all_trades, minimal_features, pre_p_exec, THETA, 'Pre-Merge')

    print(f"\n  ── Baseline (Pre-Merge) ──")
    if pre_eval:
        print(f"     EXEC: {pre_eval['n_exec']}  Sharp Gap: {pre_eval['sharp_gap']:+.1f}%p")
        print(f"     FalseExec: {pre_eval['false_exec_rate']:.1f}%  IMMORTAL: {pre_eval['immortal_capture_rate']:.1f}%")
        print(f"     PnL: ${pre_eval['exec_pnl']:,.0f}")

    print(f"\n  ═══ MARGIN SWEEP: m ∈ {MARGINS} ═══")

    sweep_results = []

    for margin in MARGINS:
        merge_result = run_merge_with_margin(all_bins, bin_trades, margin, posterior_original)

        post_p_exec = []
        for mf in minimal_features:
            p = get_p_exec_for_trade(mf, merge_result['posterior_merged'])
            post_p_exec.append(p)

        post_eval = evaluate_boundary(all_trades, minimal_features, post_p_exec, THETA, f'm={margin}')

        d_sg = post_eval['sharp_gap'] - pre_eval['sharp_gap'] if post_eval and pre_eval else None
        d_fe = post_eval['false_exec_rate'] - pre_eval['false_exec_rate'] if post_eval and pre_eval else None
        imm = post_eval['immortal_capture_rate'] if post_eval else 0
        d_pnl = post_eval['exec_pnl'] - pre_eval['exec_pnl'] if post_eval and pre_eval else None
        d_exec = post_eval['n_exec'] - pre_eval['n_exec'] if post_eval and pre_eval else 0

        sg_pass = d_sg is not None and d_sg >= PASS_CRITERIA['sharp_gap_drift_min']
        fe_pass = d_fe is not None and d_fe <= PASS_CRITERIA['false_exec_drift_max']
        imm_pass = imm >= PASS_CRITERIA['immortal_capture_min']
        merge_pass = merge_result['n_merges'] >= PASS_CRITERIA['min_merges']
        all_pass = sg_pass and fe_pass and imm_pass and merge_pass

        status = '✅' if all_pass else ('⚠️' if merge_result['n_merges'] == 0 else '❌')

        result = {
            'margin': margin,
            'n_merges': merge_result['n_merges'],
            'n_rejected_cfb': merge_result['n_rejected_cfb'],
            'n_rejected_post_cfb': merge_result['n_rejected_post_cfb'],
            'n_post_bins': merge_result['n_post_bins'],
            'merge_details': merge_result['merge_details'],
            'd_sharp_gap': round(d_sg, 1) if d_sg is not None else None,
            'd_false_exec': round(d_fe, 1) if d_fe is not None else None,
            'immortal_capture': round(imm, 1),
            'd_pnl': round(d_pnl, 2) if d_pnl is not None else None,
            'd_exec': d_exec,
            'pass': all_pass,
            'sg_pass': sg_pass,
            'fe_pass': fe_pass,
            'imm_pass': imm_pass,
            'merge_pass': merge_pass,
        }
        sweep_results.append(result)

        print(f"\n  ── m = {margin:.2f} ──  {status}")
        print(f"     Merges: {merge_result['n_merges']}  "
              f"(CFB rejected: {merge_result['n_rejected_cfb']}+{merge_result['n_rejected_post_cfb']})")
        if merge_result['n_merges'] > 0:
            print(f"     Bins: {len(all_bins)} → {merge_result['n_post_bins']}")
            print(f"     Δ SG: {d_sg:+.1f}%p  Δ FE: {d_fe:+.1f}%p  IMM: {imm:.1f}%  Δ PnL: ${d_pnl:+,.0f}")
            print(f"     Δ Exec: {d_exec:+d}")
            for md in merge_result['merge_details']:
                print(f"       [{md['dim_name']}] {md['merged_key']}  p={md['p_merged']:.3f}")
        else:
            print(f"     (No merges at this margin)")

    print(f"\n  ═══ SWEEP SUMMARY ═══")
    print(f"  {'m':>6s}  {'merges':>6s}  {'ΔSG':>7s}  {'ΔFE':>7s}  {'IMM':>6s}  {'ΔPnL':>8s}  {'ΔExec':>6s}  {'PASS':>4s}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*4}")

    for r in sweep_results:
        sg_str = f"{r['d_sharp_gap']:+.1f}" if r['d_sharp_gap'] is not None else 'N/A'
        fe_str = f"{r['d_false_exec']:+.1f}" if r['d_false_exec'] is not None else 'N/A'
        pnl_str = f"${r['d_pnl']:+,.0f}" if r['d_pnl'] is not None else 'N/A'
        pass_str = '✅' if r['pass'] else '❌'
        print(f"  {r['margin']:>6.2f}  {r['n_merges']:>6d}  {sg_str:>7s}  {fe_str:>7s}  "
              f"{r['immortal_capture']:>5.1f}%  {pnl_str:>8s}  {r['d_exec']:>+6d}  {pass_str:>4s}")

    print(f"\n  ═══ VERDICT ═══")

    passing_margins = [r for r in sweep_results if r['pass']]

    if passing_margins:
        best = max(passing_margins, key=lambda r: r['n_merges'])
        print(f"\n  ✅ FORBIDDEN BAND CONFIRMED")
        print(f"     Optimal margin: m = {best['margin']:.2f}")
        print(f"     Merges: {best['n_merges']}  Bins: {len(all_bins)} → {best['n_post_bins']}")
        print(f"     Δ SG: {best['d_sharp_gap']:+.1f}%p  Δ FE: {best['d_false_exec']:+.1f}%p")
        print(f"     IMM: {best['immortal_capture']:.1f}%  Δ PnL: ${best['d_pnl']:+,.0f}")
        print(f"\n  CFB LAW CANDIDATE:")
        print(f"     '|p̂ - θ| < {best['margin']} person/of bin merge prohibited'")
        print(f"     'merged backatalso |p̂_merge - θ| ≥ {best['margin']} must done'")

        safe_margins = [r['margin'] for r in passing_margins]
        unsafe_margins = [r['margin'] for r in sweep_results if not r['pass'] and r['n_merges'] > 0]
        print(f"\n  Safe margins: {safe_margins}")
        print(f"  Unsafe margins: {unsafe_margins}")
    else:
        merging_but_failing = [r for r in sweep_results if r['n_merges'] > 0 and not r['pass']]
        no_merge = [r for r in sweep_results if r['n_merges'] == 0]

        if merging_but_failing:
            print(f"\n  ❌ NO SAFE MARGIN FOUND — all merge law erosion")
            print(f"     Current bin structure is already the minimum distinct unit")
            print(f"     'Cannot merge = structure is already optimal' → constitutionever/instance sealed")
        else:
            print(f"\n  ⬜ ALL MARGINS BLOCK ALL MERGES — forbidden band  covered")
            print(f"     merge target itself boundarylayerat  exist")

    exp61v2_dir = os.path.join(EVIDENCE_DIR, 'exp61v2a_forbidden_band')
    os.makedirs(exp61v2_dir, exist_ok=True)

    exp61v2_data = {
        'experiment': 'EXP-61 v2a Forbidden Band Coalescence — Margin Sweep',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'n_original_bins': len(all_bins),
        'theta': THETA,
        'margins_tested': MARGINS,
        'pass_criteria': PASS_CRITERIA,
        'pre_eval': pre_eval,
        'sweep_results': [{k: v for k, v in r.items() if k != 'posterior_merged'} for r in sweep_results],
        'passing_margins': [r['margin'] for r in passing_margins] if passing_margins else [],
        'optimal_margin': passing_margins[0]['margin'] if passing_margins else None,
    }

    with open(os.path.join(exp61v2_dir, 'forbidden_band_results.json'), 'w') as f:
        json.dump(exp61v2_data, f, indent=2, cls=NumpyEncoder)

    if passing_margins:
        best = max(passing_margins, key=lambda r: r['n_merges'])
        merged_result = run_merge_with_margin(all_bins, bin_trades, best['margin'], posterior_original)
        merged_path = os.path.join(exp61v2_dir, f"posterior_merged_m{best['margin']:.2f}.json")
        merged_result['posterior_merged'].save(merged_path)
        print(f"\n  Best merged posterior saved: {merged_path}")

    elapsed = time.time() - t0
    print(f"\n  --- EXP-61 v2a Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'boundarylayer phase transition intervalis. merge's/of physical limit existencedoes.'")


if __name__ == '__main__':
    main()
