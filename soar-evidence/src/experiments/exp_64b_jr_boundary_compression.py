#!/usr/bin/env python3
"""
EXP-64b: JR BOUNDARY COMPRESSION — " onlyenters not... but, boundary thinly do"
================================================================
"INFANT↔MATURE between boundarysurface/if pressure do consecutiveever/instance maturity possiblemake does."

PREREQUISITE:
  MDU Law — 57-bin topology sealed
  EXP-63 — ADOLESCENT=0 IMMORTAL ( stage wealth/department)
  EXP-64a — Scheduling Law (infant=full, mature=sample)

DESIGN:
  JR boundary score b ∈ [0,1]:
    sharp = |p - 0.5| * √n   (confidence measure)
    b = sigmoid(k * (S0 - sharp))
    b high → uncertainty surface (boundarysurface/if)
    b low → confirmed interior (confirmation internal)

  Three worlds on 70/30 train/test split:
    Baseline: 64a only (infant=full, mature p_sample=0.25)
    JR-BC: 64a + boundary promotion (mature with b≥threshold → full)
    Over-BC: stress test (very low threshold → too much full compute)

HYPOTHESES:
  H-64b-a (Compression): BT decreases ≥20% relative during training
  H-64b-b (Adolescent Emergence): ADOLESCENT bins ≥3 or trades ≥30
  H-64b-c (Law Preservation): All invariants PASS + IMM ≥80%
  H-64b-d (Efficiency): full_compute increase ≤+10%p vs 64a

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ p_exec value directly manipulation prohibited
  ❌ Gate/sizing/energy change prohibited
  ✔️ observation/learning density distributiononly change (JR score based on)
"""

import sys, os, json, time, math
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
TRAIN_RATIO = 0.70
P_SAMPLE_MATURE = 0.25

INFANT_THRESHOLD = 10
DECIDED_THRESHOLD = 20

np.random.seed(42)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if abs(x) < 500 else (1.0 if x > 0 else 0.0)


def compute_boundary_score(alpha, beta, k=1.0, s0=3.0):
    p = alpha / (alpha + beta)
    n = alpha + beta - 2.0
    sharp = abs(p - 0.5) * math.sqrt(max(n, 0))
    b = sigmoid(k * (s0 - sharp))
    return b, sharp


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


def run_jr_learning(trades, minimal_features, b_full_threshold, k=1.0, s0=3.0):
    posterior = BetaPosterior(alpha_prior=1.0, beta_prior=1.0)

    n_train = int(len(trades) * TRAIN_RATIO)
    train_trades = trades[:n_train]
    train_mf = minimal_features[:n_train]

    full_computes = 0
    fast_computes = 0
    bt_history = []
    maturity_history = []

    for i, (t, mf) in enumerate(zip(train_trades, train_mf)):
        key = get_bin_key(mf)
        b_data = posterior.bins[key]
        sharpness = b_data['alpha'] + b_data['beta'] - 2.0
        maturity = classify_maturity(sharpness)

        b_score, sharp_val = compute_boundary_score(b_data['alpha'], b_data['beta'], k, s0)

        do_full = True
        if maturity == 'MATURE':
            if b_score >= b_full_threshold:
                do_full = True
            else:
                if np.random.random() > P_SAMPLE_MATURE:
                    do_full = False

        if do_full:
            is_win = t.get('is_win', False)
            posterior.update(
                e_sign=mf['e_sign'], de_sign=mf['de_sign'],
                shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
                regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
                is_win=is_win,
            )
            full_computes += 1
        else:
            fast_computes += 1

        if (i + 1) % 100 == 0 or i == len(train_trades) - 1:
            all_b_scores = []
            mat_counts = {'NEWBORN': 0, 'INFANT': 0, 'ADOLESCENT': 0, 'MATURE': 0}
            for bk, bv in posterior.bins.items():
                s = bv['alpha'] + bv['beta'] - 2.0
                m = classify_maturity(s)
                mat_counts[m] = mat_counts.get(m, 0) + 1
                bs, _ = compute_boundary_score(bv['alpha'], bv['beta'], k, s0)
                all_b_scores.append(bs)

            bt = np.mean(all_b_scores) if all_b_scores else 0
            p_above_05 = sum(1 for bs in all_b_scores if bs > 0.5) / max(len(all_b_scores), 1)
            bt_history.append({
                'trade_idx': i,
                'mean_b': round(bt, 4),
                'p_above_05': round(p_above_05, 4),
                'n_adolescent': mat_counts.get('ADOLESCENT', 0),
            })
            maturity_history.append(dict(mat_counts))

    p_exec_list = []
    for mf in minimal_features:
        p = posterior.get_p_exec(
            e_sign=mf['e_sign'], de_sign=mf['de_sign'],
            shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
            regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
        )
        p_exec_list.append(p)

    train_eval = evaluate_boundary(
        trades[:n_train], minimal_features[:n_train],
        p_exec_list[:n_train], THETA, 'train'
    )
    test_eval = evaluate_boundary(
        trades[n_train:], minimal_features[n_train:],
        p_exec_list[n_train:], THETA, 'test'
    )
    full_eval = evaluate_boundary(
        trades, minimal_features,
        p_exec_list, THETA, 'full'
    )

    final_mat = {'NEWBORN': 0, 'INFANT': 0, 'ADOLESCENT': 0, 'MATURE': 0}
    adolescent_bins = []
    for bk, bv in posterior.bins.items():
        s = bv['alpha'] + bv['beta'] - 2.0
        m = classify_maturity(s)
        final_mat[m] = final_mat.get(m, 0) + 1
        if m == 'ADOLESCENT':
            adolescent_bins.append({
                'key': bk,
                'sharpness': round(s, 1),
                'n': bv['n'],
                'p': round(bv['alpha'] / (bv['alpha'] + bv['beta']), 4),
            })

    total_ops = full_computes + fast_computes
    compute_savings = fast_computes / max(total_ops, 1) * 100
    full_pct = full_computes / max(total_ops, 1) * 100

    return {
        'posterior': posterior,
        'full_computes': full_computes,
        'fast_computes': fast_computes,
        'compute_savings_pct': round(compute_savings, 1),
        'full_compute_pct': round(full_pct, 1),
        'bt_history': bt_history,
        'maturity_history': maturity_history,
        'final_maturity': final_mat,
        'adolescent_bins': adolescent_bins,
        'train_eval': train_eval,
        'test_eval': test_eval,
        'full_eval': full_eval,
    }


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-64b: JR BOUNDARY COMPRESSION")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  ' onlyenters not... but, boundary thinly do.'")
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

    n_total = len(all_trades)
    n_train = int(n_total * TRAIN_RATIO)
    n_test = n_total - n_train
    print(f"  Total: {n_total} trades (Train: {n_train} / Test: {n_test})")

    shadow_results = []
    for t in all_trades:
        traj = t.get('energy_trajectory', [])
        sg = compute_shadow_geometry(traj, t.get('atp_bar'), t.get('alpha_fate', 'UNKNOWN'))
        shadow_results.append(sg if sg else {'shadow_class': 'NO_SHADOW'})
    aep_results = compute_aep(all_trades)
    arg_results = compute_arg_deny(all_trades, shadow_results, aep_results)
    minimal_features = extract_minimal_features(all_trades, arg_results, shadow_results, aep_results)

    configs = [
        {'name': 'Baseline (64a only)', 'b_threshold': 2.0, 'k': 1.0, 's0': 3.0},
        {'name': 'JR-BC (b≥0.65)', 'b_threshold': 0.65, 'k': 1.0, 's0': 3.0},
        {'name': 'JR-BC (b≥0.50)', 'b_threshold': 0.50, 'k': 1.0, 's0': 3.0},
        {'name': 'JR-BC (b≥0.35)', 'b_threshold': 0.35, 'k': 1.5, 's0': 3.0},
        {'name': 'Over-BC stress (b≥0.20)', 'b_threshold': 0.20, 'k': 1.0, 's0': 3.0},
    ]

    print(f"\n  ═══ JR BOUNDARY COMPRESSION SWEEP ═══")
    print(f"  JR score: b = sigmoid(k * (S0 - sharp))")
    print(f"  sharp = |p - 0.5| * √n")
    print(f"  Mature bins with b ≥ threshold → full compute (promoted)")
    print(f"  Train/Test split: {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%")

    world_results = {}

    for cfg in configs:
        np.random.seed(42)
        label = cfg['name']
        print(f"\n  ── {label} ──")
        print(f"     b_threshold={cfg['b_threshold']}, k={cfg['k']}, S0={cfg['s0']}")

        result = run_jr_learning(
            all_trades, minimal_features,
            b_full_threshold=cfg['b_threshold'],
            k=cfg['k'], s0=cfg['s0'],
        )

        bth = result['bt_history']
        bt_start = bth[0]['mean_b'] if bth else 0
        bt_end = bth[-1]['mean_b'] if bth else 0
        bt_reduction = (bt_start - bt_end) / max(bt_start, 0.001) * 100

        p05_start = bth[0]['p_above_05'] if bth else 0
        p05_end = bth[-1]['p_above_05'] if bth else 0
        p05_reduction = (p05_start - p05_end) / max(p05_start, 0.001) * 100

        print(f"     Compute: {result['full_computes']} full / {result['fast_computes']} fast ({result['compute_savings_pct']:.1f}% saved)")
        print(f"     BT: mean_b {bt_start:.4f} → {bt_end:.4f} (Δ{bt_reduction:+.1f}%)")
        print(f"     P(b>0.5): {p05_start:.3f} → {p05_end:.3f} (Δ{p05_reduction:+.1f}%)")

        fm = result['final_maturity']
        print(f"     Maturity: NB={fm['NEWBORN']} INF={fm['INFANT']} ADO={fm['ADOLESCENT']} MAT={fm['MATURE']}")

        if result['adolescent_bins']:
            print(f"     ⚡ ADOLESCENT bins emerged: {len(result['adolescent_bins'])}")
            for ab in sorted(result['adolescent_bins'], key=lambda x: x['sharpness'], reverse=True)[:5]:
                print(f"       {ab['key']}: sharp={ab['sharpness']}, n={ab['n']}, p={ab['p']}")

        te = result['train_eval']
        tst = result['test_eval']
        fe = result['full_eval']
        if te:
            print(f"     Train: Exec={te['n_exec']} FE={te['false_exec_rate']:.1f}% SG={te['sharp_gap']:+.1f} IMM={te['immortal_capture_rate']:.1f}%")
        if tst:
            print(f"     Test:  Exec={tst['n_exec']} FE={tst['false_exec_rate']:.1f}% SG={tst['sharp_gap']:+.1f} IMM={tst['immortal_capture_rate']:.1f}%")
        if fe:
            print(f"     Full:  Exec={fe['n_exec']} FE={fe['false_exec_rate']:.1f}% SG={fe['sharp_gap']:+.1f} IMM={fe['immortal_capture_rate']:.1f}%")

        world_results[label] = {
            'config': cfg,
            'full_computes': result['full_computes'],
            'fast_computes': result['fast_computes'],
            'compute_savings_pct': result['compute_savings_pct'],
            'full_compute_pct': result['full_compute_pct'],
            'bt_start': round(bt_start, 4),
            'bt_end': round(bt_end, 4),
            'bt_reduction_pct': round(bt_reduction, 1),
            'p05_start': round(p05_start, 4),
            'p05_end': round(p05_end, 4),
            'p05_reduction_pct': round(p05_reduction, 1),
            'final_maturity': fm,
            'n_adolescent': fm['ADOLESCENT'],
            'adolescent_bins': result['adolescent_bins'],
            'train_eval': te,
            'test_eval': tst,
            'full_eval': fe,
            'bt_history': result['bt_history'],
        }

    print(f"\n  ═══ SWEEP COMPARISON ═══")
    print(f"  {'Config':<28s}  {'Saved':>6s}  {'BT Δ':>7s}  {'ADO':>4s}  {'FE(full)':>8s}  {'SG(full)':>8s}  {'IMM(full)':>9s}")
    print(f"  {'─'*28}  {'─'*6}  {'─'*7}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*9}")

    for label, wr in world_results.items():
        fe = wr['full_eval']
        fe_str = f"{fe['false_exec_rate']:.1f}%" if fe else 'N/A'
        sg_str = f"{fe['sharp_gap']:+.1f}" if fe else 'N/A'
        imm_str = f"{fe['immortal_capture_rate']:.1f}%" if fe else 'N/A'
        print(f"  {label:<28s}  {wr['compute_savings_pct']:>5.1f}%  {wr['bt_reduction_pct']:>+6.1f}%  {wr['n_adolescent']:>4d}  {fe_str:>8s}  {sg_str:>8s}  {imm_str:>9s}")

    baseline = world_results.get('Baseline (64a only)')

    print(f"\n  ═══ HYPOTHESIS TESTING ═══")

    h_results = {}

    best_bc = None
    for label, wr in world_results.items():
        if 'JR-BC' in label and wr['bt_reduction_pct'] > (best_bc['bt_reduction_pct'] if best_bc else -999):
            best_bc = wr
            best_bc['label'] = label

    if best_bc:
        h_a = best_bc['bt_reduction_pct'] >= 20.0
        print(f"\n  H-64b-a (Compression): BT reduction = {best_bc['bt_reduction_pct']:+.1f}%")
        print(f"    Threshold: ≥20% relative decrease")
        print(f"    → {'✅ PASS' if h_a else '❌ FAIL'}")
        h_results['H-64b-a'] = {'pass': h_a, 'value': best_bc['bt_reduction_pct']}

        h_b = best_bc['n_adolescent'] >= 3
        print(f"\n  H-64b-b (Adolescent Emergence): {best_bc['n_adolescent']} ADOLESCENT bins")
        print(f"    Threshold: ≥3 bins")
        print(f"    → {'✅ PASS' if h_b else '❌ FAIL'}")
        h_results['H-64b-b'] = {'pass': h_b, 'value': best_bc['n_adolescent']}

        fe = best_bc.get('full_eval')
        if fe:
            imm_ok = fe['immortal_capture_rate'] >= 80.0
            fe_ok = fe['false_exec_rate'] <= 25.0
            sg_ok = fe['sharp_gap'] >= 60.0
            h_c = imm_ok and fe_ok and sg_ok
            print(f"\n  H-64b-c (Law Preservation):")
            print(f"    IMM capture: {fe['immortal_capture_rate']:.1f}% (≥80%) → {'✅' if imm_ok else '❌'}")
            print(f"    FE: {fe['false_exec_rate']:.1f}% (≤25%) → {'✅' if fe_ok else '❌'}")
            print(f"    SG: {fe['sharp_gap']:+.1f}%p (≥60%) → {'✅' if sg_ok else '❌'}")
            print(f"    → {'✅ PASS' if h_c else '❌ FAIL'}")
            h_results['H-64b-c'] = {'pass': h_c, 'imm': fe['immortal_capture_rate'], 'fe': fe['false_exec_rate'], 'sg': fe['sharp_gap']}

        if baseline:
            d_full = best_bc['full_compute_pct'] - baseline['full_compute_pct']
            h_d = d_full <= 10.0
            print(f"\n  H-64b-d (Efficiency): full_compute Δ = {d_full:+.1f}%p vs baseline")
            print(f"    Threshold: ≤+10%p increase")
            print(f"    → {'✅ PASS' if h_d else '❌ FAIL'}")
            h_results['H-64b-d'] = {'pass': h_d, 'delta': round(d_full, 1)}

    print(f"\n  ═══ VERDICT ═══")

    all_pass = all(h.get('pass', False) for h in h_results.values())
    n_pass = sum(1 for h in h_results.values() if h.get('pass', False))
    n_total_h = len(h_results)

    if all_pass:
        print(f"\n  ✅ JR BOUNDARY COMPRESSION SUCCESSFUL ({n_pass}/{n_total_h} hypotheses PASS)")
        print(f"     boundarysurface/if pressure becomes, ADOLESCENT generationbecomes, law conservationdone/become.")
        print(f"     → EXP-64c (PDL Directed Pheromone) preparation complete")
    elif n_pass >= 2:
        print(f"\n  ⚠️  PARTIAL SUCCESS ({n_pass}/{n_total_h} hypotheses PASS)")
        failed = [k for k, v in h_results.items() if not v.get('pass')]
        print(f"     Failed: {', '.join(failed)}")
        if not h_results.get('H-64b-b', {}).get('pass'):
            print(f"     ADOLESCENT generation = phase transition essenceever/instance → 64cfrom learning case definition needed")
        if not h_results.get('H-64b-a', {}).get('pass'):
            print(f"     BT reduction = boundarysurface already minimal → data problem, not structural problem")
    else:
        print(f"\n  ❌ JR BOUNDARY COMPRESSION FAILED ({n_pass}/{n_total_h} hypotheses PASS)")
        print(f"     boundary Compression is not valid in current data")
        print(f"     → phase transition(phase transition) essenceever/instance is a characteristic possiblenature/property high")

    over_bc = world_results.get('Over-BC stress (b≥0.20)')
    if over_bc and over_bc.get('full_eval'):
        ov_fe = over_bc['full_eval']
        print(f"\n  Over-BC stress test:")
        print(f"    FE={ov_fe['false_exec_rate']:.1f}% SG={ov_fe['sharp_gap']:+.1f} IMM={ov_fe['immortal_capture_rate']:.1f}%")
        print(f"    Compute savings: {over_bc['compute_savings_pct']:.1f}%")
        if ov_fe['immortal_capture_rate'] < 80.0 or ov_fe['false_exec_rate'] > 25.0:
            print(f"    ⚠️  Overcompression detected — too much full compute causes posterior instability")
        else:
            print(f"    ✅ Even aggressive promotion preserves laws")

    exp64b_dir = os.path.join(EVIDENCE_DIR, 'exp64b_jr_boundary_compression')
    os.makedirs(exp64b_dir, exist_ok=True)

    clean_results = {}
    for label, wr in world_results.items():
        cr = {k: v for k, v in wr.items() if k != 'bt_history'}
        clean_results[label] = cr

    exp_data = {
        'experiment': 'EXP-64b JR Boundary Compression',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'n_trades': len(all_trades),
        'n_train': n_train,
        'n_test': n_test,
        'train_ratio': TRAIN_RATIO,
        'theta': THETA,
        'p_sample_mature': P_SAMPLE_MATURE,
        'configs': configs,
        'world_results': clean_results,
        'hypotheses': h_results,
        'all_pass': all_pass,
    }

    with open(os.path.join(exp64b_dir, 'jr_boundary_compression_results.json'), 'w') as f:
        json.dump(exp_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-64b Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'Making the boundary thin does not increase win rate — it creates growth paths.'")


if __name__ == '__main__':
    main()
