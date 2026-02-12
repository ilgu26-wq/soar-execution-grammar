#!/usr/bin/env python3
"""
EXP-61: BIN COALESCENCE (natural merge)
================================================================
"distinguish needed none worldline to/as merges."

CONSTITUTION:
  ❌ criticalvalue does not lower does not
  ❌ p_exec does not manipulate does not
  ❌ execution trying to increase do not does not
  ✔️ statisticsever/instanceto/as distinction Impossibleone/a adjacent bin oneto/as mergedoes

DESIGN:
  1. adjacency definition: 6-feature  exactly 1only different bin pair
  2. merge condition (ALL must hold):
     a. KL(A‖B) < ε  (posterior overlap)
     b. Direction match (two  ≥θ or two  <θ)
     c. FalseExec rate difference < δ
  3. merge method: α = α_A + α_B, β = β_A + β_B
  4. original conservation (rollback possible)

SUCCESS:
  - DECIDED bin ratio ↑
  - Posterior variance ↓
  - work/day θfrom p_exec > θ region ↑ (natural increase)

FAILURE:
  - SharpGap reduction > 5%p
  - FalseExec increase
  - IMMORTAL capture < 85%
"""

import sys, os, json, time, copy, math
import numpy as np
from datetime import datetime
from itertools import combinations
from scipy import special

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
from experiments.exp_60_posterior_shape import analyze_bin_shape, classify_bin_state
from observer.learning.p_exec_posterior import BetaPosterior

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')

THETA = 0.5

KL_EPSILON = 0.15
FE_DELTA = 0.10

FEATURE_DIMS = {
    0: ['POS', 'NEG'],
    1: ['RISING', 'FALLING'],
    2: ['NO_SHADOW', 'SHADOW'],
    3: ['D0', 'D1', 'D2', 'D3+'],
    4: ['TREND', 'NON_TREND'],
    5: ['HIGH', 'LOW'],
}

FEATURE_NAMES = ['E_sign', 'dE_sign', 'Shadow', 'ARG_depth', 'Regime', 'AEP_zone']


def kl_beta(a1, b1, a2, b2):
    try:
        kl = (special.betaln(a2, b2) - special.betaln(a1, b1)
              + (a1 - a2) * special.digamma(a1)
              + (b1 - b2) * special.digamma(b1)
              + (a2 - a1 + b2 - b1) * special.digamma(a1 + b1))
        return max(0, float(kl))
    except Exception:
        return float('inf')


def symmetric_kl(a1, b1, a2, b2):
    return (kl_beta(a1, b1, a2, b2) + kl_beta(a2, b2, a1, b1)) / 2


def parse_bin_key(key):
    return key.split('|')


def are_adjacent(key_a, key_b):
    parts_a = parse_bin_key(key_a)
    parts_b = parse_bin_key(key_b)
    if len(parts_a) != 6 or len(parts_b) != 6:
        return False, -1
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(parts_a, parts_b)) if a != b]
    if len(diffs) == 1:
        return True, diffs[0][0]
    return False, -1


def compute_bin_fe_rate(trades_in_bin):
    if not trades_in_bin:
        return 0.0
    losses = sum(1 for t in trades_in_bin if not t.get('is_win', False))
    return losses / len(trades_in_bin)


def assign_trades_to_bins(trades, minimal_features):
    bin_trades = {}
    for i, mf in enumerate(minimal_features):
        key = f"{mf['e_sign']}|{mf['de_sign']}|{mf['shadow_binary']}|{mf['arg_depth']}|{mf['regime_coarse']}|{mf['aep_binary']}"
        if key not in bin_trades:
            bin_trades[key] = []
        bin_trades[key].append(trades[i])
    return bin_trades


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-61: BIN COALESCENCE (natural merge)")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'distinguish needed none worldline to/as merges.'")
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
    datasets = []

    if os.path.exists(nq_tick_path):
        ticks_df = load_ticks(nq_tick_path)
        nq_5s = aggregate_5s(ticks_df)
        datasets.append(('NQ_Tick_5s', nq_5s, 0.25, 5.0))

    if os.path.exists(nq_combined_path):
        nq_1m = load_1min_bars(nq_combined_path, tick_size=0.25)
        if nq_1m is not None:
            datasets.append(('NQ_1min', nq_1m, 0.25, 5.0))

    for ds_name, bars_df, tick_size, tick_value in datasets:
        signals = generate_signals_multi(bars_df, tick_size=tick_size)
        trades, _, stats = run_pipeline_deferred(signals, bars_df, tick_value, tick_size)
        all_trades.extend(trades)
        print(f"  {ds_name}: {len(trades)} trades")

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

    print(f"\n  ═══ PHASE 1: ADJACENCY SCAN ═══")

    active_bins = {k: v for k, v in all_bins.items() if v['n'] >= 3}
    active_keys = list(active_bins.keys())
    print(f"  Active bins (n≥3): {len(active_keys)}")

    adjacent_pairs = []
    for i, key_a in enumerate(active_keys):
        for key_b in active_keys[i+1:]:
            is_adj, dim = are_adjacent(key_a, key_b)
            if is_adj:
                adjacent_pairs.append((key_a, key_b, dim))

    print(f"  Adjacent pairs found: {len(adjacent_pairs)}")

    print(f"\n  ═══ PHASE 2: MERGE ELIGIBILITY ═══")

    merge_candidates = []

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

        eligible = (skl < KL_EPSILON) and same_direction and fe_similar

        candidate = {
            'key_a': key_a,
            'key_b': key_b,
            'dim': dim,
            'dim_name': FEATURE_NAMES[dim],
            'skl': round(skl, 4),
            'p_a': round(p_a, 4),
            'p_b': round(p_b, 4),
            'fe_a': round(fe_a, 3),
            'fe_b': round(fe_b, 3),
            'same_direction': same_direction,
            'fe_similar': fe_similar,
            'eligible': eligible,
            'n_a': bin_a['n'],
            'n_b': bin_b['n'],
        }

        if eligible:
            merge_candidates.append(candidate)

    print(f"  Merge-eligible pairs: {len(merge_candidates)}")

    for mc in sorted(merge_candidates, key=lambda x: x['skl'])[:15]:
        print(f"    [{mc['dim_name']:>10s}] SKL={mc['skl']:.4f}  "
              f"p={mc['p_a']:.3f}/{mc['p_b']:.3f}  "
              f"FE={mc['fe_a']:.2f}/{mc['fe_b']:.2f}  "
              f"n={mc['n_a']}+{mc['n_b']}")
        print(f"      A: {mc['key_a']}")
        print(f"      B: {mc['key_b']}")

    if not merge_candidates:
        print(f"\n  ⬜ No eligible merges. Bins are already maximally distinct.")
        print(f"     This is failureis not — the structure is already minimized.")

        exp61_dir = os.path.join(EVIDENCE_DIR, 'exp61_bin_coalescence')
        os.makedirs(exp61_dir, exist_ok=True)
        exp61_data = {
            'experiment': 'EXP-61 Bin Coalescence',
            'timestamp': datetime.now().isoformat(),
            'lock_version': LOCK_VERSION,
            'n_original_bins': len(all_bins),
            'n_active_bins': len(active_keys),
            'n_adjacent_pairs': len(adjacent_pairs),
            'n_merge_candidates': 0,
            'merges_performed': 0,
            'verdict': 'NO_MERGE_NEEDED',
        }
        with open(os.path.join(exp61_dir, 'coalescence_results.json'), 'w') as f:
            json.dump(exp61_data, f, indent=2, cls=NumpyEncoder)
        return

    print(f"\n  ═══ PHASE 3: GREEDY MERGE EXECUTION ═══")

    merged_bins = set()
    merge_plan = []

    for mc in sorted(merge_candidates, key=lambda x: x['skl']):
        if mc['key_a'] in merged_bins or mc['key_b'] in merged_bins:
            continue
        merge_plan.append(mc)
        merged_bins.add(mc['key_a'])
        merged_bins.add(mc['key_b'])

    print(f"  Merges to execute (greedy, no overlap): {len(merge_plan)}")

    posterior_merged = BetaPosterior(
        alpha_prior=posterior_original.alpha_prior,
        beta_prior=posterior_original.beta_prior,
    )

    for key, b in all_bins.items():
        posterior_merged.bins[key] = copy.deepcopy(b)

    merge_records = []

    for mc in merge_plan:
        key_a = mc['key_a']
        key_b = mc['key_b']

        bin_a = posterior_merged.bins[key_a]
        bin_b = posterior_merged.bins[key_b]

        merged_alpha = bin_a['alpha'] + bin_b['alpha'] - posterior_merged.alpha_prior
        merged_beta = bin_a['beta'] + bin_b['beta'] - posterior_merged.beta_prior
        merged_n = bin_a['n'] + bin_b['n']

        parts_a = parse_bin_key(key_a)
        parts_b = parse_bin_key(key_b)
        dim = mc['dim']
        merged_parts = list(parts_a)
        merged_parts[dim] = f"{parts_a[dim]}+{parts_b[dim]}"
        merged_key = '|'.join(merged_parts)

        posterior_merged.bins[merged_key] = {
            'alpha': merged_alpha,
            'beta': merged_beta,
            'n': merged_n,
        }

        del posterior_merged.bins[key_a]
        del posterior_merged.bins[key_b]

        merged_p = merged_alpha / (merged_alpha + merged_beta)

        merge_records.append({
            'key_a': key_a,
            'key_b': key_b,
            'merged_key': merged_key,
            'dim_name': mc['dim_name'],
            'skl': mc['skl'],
            'merged_p_exec': round(merged_p, 4),
            'merged_n': merged_n,
            'merged_alpha': round(merged_alpha, 1),
            'merged_beta': round(merged_beta, 1),
        })

        print(f"    MERGED: {merged_key}")
        print(f"      ← {key_a} + {key_b}")
        print(f"      p_exec={merged_p:.3f}  n={merged_n}  SKL={mc['skl']:.4f}")

    new_bins = posterior_merged.get_all_bins()
    print(f"\n  Bins after merge: {len(new_bins)} (was {len(all_bins)}, Δ={len(new_bins)-len(all_bins)})")

    print(f"\n  ═══ PHASE 4: POST-MERGE SHAPE ANALYSIS ═══")

    pre_shapes = {}
    post_shapes = {}

    for key, b in all_bins.items():
        if b['n'] >= 3:
            pre_shapes[key] = analyze_bin_shape(b['alpha'], b['beta'])

    for key, b in new_bins.items():
        if b['n'] >= 3:
            post_shapes[key] = analyze_bin_shape(b['alpha'], b['beta'])

    pre_states = {}
    post_states = {}
    for s in pre_shapes.values():
        st = classify_bin_state(s)
        pre_states[st] = pre_states.get(st, 0) + 1
    for s in post_shapes.values():
        st = classify_bin_state(s)
        post_states[st] = post_states.get(st, 0) + 1

    print(f"\n  State distribution (before → after):")
    all_state_names = sorted(set(list(pre_states.keys()) + list(post_states.keys())))
    for st in all_state_names:
        pre = pre_states.get(st, 0)
        post = post_states.get(st, 0)
        delta = post - pre
        print(f"    {st:>12s}: {pre:>3d} → {post:>3d}  ({delta:+d})")

    pre_var = np.mean([s['variance'] for s in pre_shapes.values()]) if pre_shapes else 0
    post_var = np.mean([s['variance'] for s in post_shapes.values()]) if post_shapes else 0
    pre_conv = np.mean([s['convergence'] for s in pre_shapes.values()]) if pre_shapes else 0
    post_conv = np.mean([s['convergence'] for s in post_shapes.values()]) if post_shapes else 0

    print(f"\n  Variance:    {pre_var:.6f} → {post_var:.6f} ({post_var-pre_var:+.6f})")
    print(f"  Convergence: {pre_conv:.4f} → {post_conv:.4f} ({post_conv-pre_conv:+.4f})")

    print(f"\n  ═══ PHASE 5: EXECUTION COMPARISON ═══")

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

    pre_p_exec = []
    post_p_exec = []
    for mf in minimal_features:
        pre_p = posterior_original.get_p_exec(
            e_sign=mf['e_sign'], de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
        )
        post_p = get_p_exec_for_trade(mf, posterior_merged)
        pre_p_exec.append(pre_p)
        post_p_exec.append(post_p)

    sharp_p_exec = apply_sharp_boundary(minimal_features)

    pre_eval = evaluate_boundary(all_trades, minimal_features, pre_p_exec, THETA, 'Pre-Merge')
    post_eval = evaluate_boundary(all_trades, minimal_features, post_p_exec, THETA, 'Post-Merge')
    sharp_eval = evaluate_boundary(all_trades, minimal_features, sharp_p_exec, THETA, 'Sharp_Anchor')

    for r in [sharp_eval, pre_eval, post_eval]:
        if r:
            print(f"\n  ── {r['label']} ──")
            print(f"     EXEC: {r['n_exec']}  DENY: {r['n_deny']}")
            print(f"     Exec WR: {r['exec_wr']:.1f}%  Sharp Gap: {r['sharp_gap']:+.1f}%p")
            print(f"     False Exec: {r['false_exec_rate']:.1f}%")
            print(f"     IMMORTAL: {r['immortal_captured']}/{r['immortal_total']} ({r['immortal_capture_rate']:.1f}%)")
            print(f"     PnL: ${r['exec_pnl']:,.0f}")

    print(f"\n  ═══ PHASE 6: VERDICT ═══")

    verdict = 'UNKNOWN'
    if pre_eval and post_eval:
        d_sg = post_eval['sharp_gap'] - pre_eval['sharp_gap']
        d_fe = post_eval['false_exec_rate'] - pre_eval['false_exec_rate']
        d_imm = post_eval['immortal_capture_rate'] - pre_eval['immortal_capture_rate']
        d_exec = post_eval['n_exec'] - pre_eval['n_exec']
        d_pnl = post_eval['exec_pnl'] - pre_eval['exec_pnl']

        print(f"  Δ Sharp Gap:        {d_sg:+.1f}%p {'✅' if d_sg >= -5 else '❌'}")
        print(f"  Δ False Exec:       {d_fe:+.1f}%p {'✅' if d_fe <= 0 else '❌'}")
        print(f"  Δ IMMORTAL capture: {d_imm:+.1f}%p {'✅' if post_eval['immortal_capture_rate'] >= 85 else '❌'}")
        print(f"  Δ Exec count:       {d_exec:+d} {'📈' if d_exec > 0 else '⬜'}")
        print(f"  Δ PnL:              ${d_pnl:+,.0f}")

        sg_ok = d_sg >= -5
        fe_ok = d_fe <= 0
        imm_ok = post_eval['immortal_capture_rate'] >= 85

        if sg_ok and fe_ok and imm_ok:
            verdict = 'PASS'
            print(f"\n  ✅ COALESCENCE PASS — merge law does not erode did not")
            if d_exec > 0:
                print(f"     📈 execution interval natural expansion: +{d_exec} trades")
                print(f"     'Nobody said to increase it, yet it increased — this is invention.'")
            else:
                print(f"     ⬜ execution interval work/day — merge precisiononly improvement")
        elif not imm_ok:
            verdict = 'FAIL_IMMORTAL'
            print(f"\n  ❌ IMMORTAL capture < 85% — merge rejection")
        elif not fe_ok:
            verdict = 'FAIL_FE'
            print(f"\n  ❌ FalseExec increase — merge rejection")
        else:
            verdict = 'FAIL_SG'
            print(f"\n  ❌ SharpGap reduction > 5%p — merge rejection")

    if verdict == 'PASS':
        merged_posterior_path = os.path.join(EVIDENCE_DIR, 'exp61_bin_coalescence', 'posterior_merged.json')
        posterior_merged.save(merged_posterior_path)
        print(f"\n  Merged posterior saved: {merged_posterior_path}")

    exp61_dir = os.path.join(EVIDENCE_DIR, 'exp61_bin_coalescence')
    os.makedirs(exp61_dir, exist_ok=True)

    exp61_data = {
        'experiment': 'EXP-61 Bin Coalescence',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'n_original_bins': len(all_bins),
        'n_active_bins': len(active_keys),
        'n_adjacent_pairs': len(adjacent_pairs),
        'n_merge_candidates': len(merge_candidates),
        'merges_performed': len(merge_plan),
        'n_post_bins': len(new_bins),
        'merge_records': merge_records,
        'pre_eval': pre_eval,
        'post_eval': post_eval,
        'sharp_eval': sharp_eval,
        'shape_changes': {
            'pre_variance': round(pre_var, 6),
            'post_variance': round(post_var, 6),
            'pre_convergence': round(pre_conv, 4),
            'post_convergence': round(post_conv, 4),
            'pre_states': pre_states,
            'post_states': post_states,
        },
        'verdict': verdict,
        'original_posterior_backup': posterior_path,
    }

    with open(os.path.join(exp61_dir, 'coalescence_results.json'), 'w') as f:
        json.dump(exp61_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-61 Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'same conclusion true worldline mergedalso physics conservationbecomes.'")


if __name__ == '__main__':
    main()
