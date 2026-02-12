#!/usr/bin/env python3
"""
EXP-64c: PDL DIRECTED PHEROMONE — Learning-Event Field
================================================================
"A guiding field that follows 'wrong conviction', not money."

PREREQUISITE:
  EXP-64a — Scheduling Law (infant=full, mature=sample)
  EXP-64b — ADOLESCENT is transient, JR promotion doesn't change outcomes
           Train/Test FE gap = 11.4%p (overfit signal)

DESIGN:
  Learning event classification per trade:
    E- : high confidence + wrong (sharp > s_high AND loss) — MOST IMPORTANT
    E+ : low confidence + right (sharp < s_low AND win)
    E0 : neutral (everything else)

  Directed PDL deposit:
    E- → strong pheromone on bin (φ += ε_neg)
    E+ → weak pheromone on bin (φ += ε_pos)
    E0 → decay only (φ = 1 + (φ-1)*λ)
    Cap: φ ∈ [1.0, 1.2]

  Pheromone usage (observation scheduling only):
    p_full = base_p_full + γ*(φ-1)
    Higher φ → more full compute on that bin
    Gate/execution/sizing: UNTOUCHED

  Comparison (70/30 train/test):
    Baseline: 64a scheduling
    PDL-E-: E- only deposit
    PDL-E-+E+: E- strong + E+ weak deposit
    PDL-Aggressive: higher ε, lower λ (stress test)

METRICS:
  Primary: Train/Test FE gap reduction (current: 11.4%p, target: ≤6%p)
  Secondary: E- density decrease, compute efficiency ≥30%
  Laws: IMM ≥80%, SG ≥70%p, FE ≤ baseline

CONSTITUTION:
  ❌ bin structure changes prohibited (MDU Law)
  ❌ Gate/execution/sizing change prohibited
  ✔️ Only observation density distribution changed (pheromone-guided scheduling)
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


def classify_learning_event(alpha, beta, is_win, s_low=2.0, s_high=5.0):
    p = alpha / (alpha + beta)
    n = alpha + beta - 2.0
    sharp = abs(p - 0.5) * math.sqrt(max(n, 0))

    is_exec = p >= THETA

    if is_exec and not is_win:
        return 'E-'
    if not is_exec and is_win and n >= 3:
        return 'E-'
    if sharp < s_low and is_win:
        return 'E+'
    return 'E0'


class DirectedPDL:
    def __init__(self, eps_neg=0.05, eps_pos=0.01, decay=0.98, cap=1.20):
        self.eps_neg = eps_neg
        self.eps_pos = eps_pos
        self.decay = decay
        self.cap = cap
        self.pheromones = defaultdict(lambda: 1.0)
        self.e_neg_count = 0
        self.e_pos_count = 0
        self.e0_count = 0

    def process_event(self, bin_key, event_type):
        for k in list(self.pheromones.keys()):
            old = self.pheromones[k]
            self.pheromones[k] = 1.0 + (old - 1.0) * self.decay

        if event_type == 'E-':
            old = self.pheromones[bin_key]
            self.pheromones[bin_key] = min(old + self.eps_neg, self.cap)
            self.e_neg_count += 1
        elif event_type == 'E+':
            old = self.pheromones[bin_key]
            self.pheromones[bin_key] = min(old + self.eps_pos, self.cap)
            self.e_pos_count += 1
        else:
            self.e0_count += 1

    def get_phi(self, bin_key):
        return self.pheromones[bin_key]

    def get_active_bins(self):
        return {k: v for k, v in self.pheromones.items() if v > 1.001}


def run_directed_pdl_learning(trades, minimal_features, config):
    posterior = BetaPosterior(alpha_prior=1.0, beta_prior=1.0)

    n_train = int(len(trades) * TRAIN_RATIO)
    train_trades = trades[:n_train]
    train_mf = minimal_features[:n_train]

    use_e_neg = config.get('use_e_neg', True)
    use_e_pos = config.get('use_e_pos', False)
    eps_neg = config.get('eps_neg', 0.05)
    eps_pos = config.get('eps_pos', 0.01)
    decay = config.get('decay', 0.98)
    gamma = config.get('gamma', 0.3)

    pdl = DirectedPDL(eps_neg=eps_neg, eps_pos=eps_pos if use_e_pos else 0.0, decay=decay)

    full_computes = 0
    fast_computes = 0
    event_counts = {'E-': 0, 'E+': 0, 'E0': 0}
    e_neg_by_bin = defaultdict(int)
    phi_history = []

    for i, (t, mf) in enumerate(zip(train_trades, train_mf)):
        key = get_bin_key(mf)
        b_data = posterior.bins[key]
        sharpness = b_data['alpha'] + b_data['beta'] - 2.0
        maturity = classify_maturity(sharpness)

        phi = pdl.get_phi(key)

        do_full = True
        if maturity == 'MATURE':
            p_full = P_SAMPLE_MATURE + gamma * (phi - 1.0)
            p_full = min(p_full, 1.0)
            if np.random.random() > p_full:
                do_full = False

        is_win = t.get('is_win', False)

        if do_full:
            posterior.update(
                e_sign=mf['e_sign'], de_sign=mf['de_sign'],
                shadow=mf['shadow_binary'], arg_depth=mf['arg_depth'],
                regime=mf['regime_coarse'], aep_zone=mf['aep_binary'],
                is_win=is_win,
            )
            full_computes += 1
        else:
            fast_computes += 1

        event = classify_learning_event(b_data['alpha'], b_data['beta'], is_win)
        event_counts[event] += 1
        if event == 'E-':
            e_neg_by_bin[key] += 1

        if use_e_neg and event == 'E-':
            pdl.process_event(key, 'E-')
        elif use_e_pos and event == 'E+':
            pdl.process_event(key, 'E+')
        else:
            pdl.process_event(key, 'E0')

        if (i + 1) % 200 == 0 or i == len(train_trades) - 1:
            active = pdl.get_active_bins()
            phi_history.append({
                'trade_idx': i,
                'active_bins': len(active),
                'max_phi': round(max(active.values()), 4) if active else 1.0,
                'mean_phi_active': round(np.mean(list(active.values())), 4) if active else 1.0,
                'e_neg_total': pdl.e_neg_count,
            })

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

    fe_train = train_eval['false_exec_rate'] if train_eval else 0
    fe_test = test_eval['false_exec_rate'] if test_eval else 0
    fe_gap = abs(fe_test - fe_train)

    total_ops = full_computes + fast_computes
    compute_savings = fast_computes / max(total_ops, 1) * 100

    return {
        'train_eval': train_eval,
        'test_eval': test_eval,
        'full_eval': full_eval,
        'fe_gap': round(fe_gap, 1),
        'event_counts': event_counts,
        'e_neg_by_bin': dict(e_neg_by_bin),
        'full_computes': full_computes,
        'fast_computes': fast_computes,
        'compute_savings_pct': round(compute_savings, 1),
        'phi_history': phi_history,
        'active_bins_final': len(pdl.get_active_bins()),
    }


def main():
    t0 = time.time()
    validate_lock()

    ASSETS = os.path.join(os.path.dirname(__file__), '..', 'attached_assets')

    print("=" * 70)
    print(f"  EXP-64c: PDL DIRECTED PHEROMONE — Learning-Event Field")
    print(f"  SOAR CORE {LOCK_VERSION}")
    print(f"  'A guiding field that follows wrong conviction, not money.'")
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
        {
            'name': 'Baseline (64a)',
            'use_e_neg': False, 'use_e_pos': False,
            'eps_neg': 0, 'eps_pos': 0, 'decay': 1.0, 'gamma': 0,
        },
        {
            'name': 'PDL-E- (ε=0.02)',
            'use_e_neg': True, 'use_e_pos': False,
            'eps_neg': 0.02, 'eps_pos': 0, 'decay': 0.98, 'gamma': 0.3,
        },
        {
            'name': 'PDL-E- (ε=0.05)',
            'use_e_neg': True, 'use_e_pos': False,
            'eps_neg': 0.05, 'eps_pos': 0, 'decay': 0.98, 'gamma': 0.3,
        },
        {
            'name': 'PDL-E-+E+ (ε=0.05/0.01)',
            'use_e_neg': True, 'use_e_pos': True,
            'eps_neg': 0.05, 'eps_pos': 0.01, 'decay': 0.98, 'gamma': 0.3,
        },
        {
            'name': 'PDL-E- (ε=0.05, γ=0.5)',
            'use_e_neg': True, 'use_e_pos': False,
            'eps_neg': 0.05, 'eps_pos': 0, 'decay': 0.98, 'gamma': 0.5,
        },
        {
            'name': 'PDL-Aggressive (ε=0.10, λ=0.95)',
            'use_e_neg': True, 'use_e_pos': True,
            'eps_neg': 0.10, 'eps_pos': 0.02, 'decay': 0.95, 'gamma': 0.5,
        },
    ]

    print(f"\n  ═══ PDL DIRECTED PHEROMONE SWEEP ═══")
    print(f"  Event classification:")
    print(f"    E-: high confidence + wrong (learning pressure)")
    print(f"    E+: low confidence + right (rare alpha signal)")
    print(f"    E0: neutral")
    print(f"  Pheromone → observation density (not execution)")
    print(f"  Train/Test: {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%")

    world_results = {}

    for cfg in configs:
        np.random.seed(42)
        label = cfg['name']
        print(f"\n  ── {label} ──")

        result = run_directed_pdl_learning(all_trades, minimal_features, cfg)

        te = result['train_eval']
        tst = result['test_eval']
        fe = result['full_eval']
        ec = result['event_counts']

        print(f"     Events: E-={ec['E-']}  E+={ec['E+']}  E0={ec['E0']}")
        print(f"     Compute: {result['full_computes']} full / {result['fast_computes']} fast ({result['compute_savings_pct']:.1f}% saved)")
        print(f"     Active pheromone bins: {result['active_bins_final']}")

        if te:
            print(f"     Train: Exec={te['n_exec']} FE={te['false_exec_rate']:.1f}% SG={te['sharp_gap']:+.1f} IMM={te['immortal_capture_rate']:.1f}%")
        if tst:
            print(f"     Test:  Exec={tst['n_exec']} FE={tst['false_exec_rate']:.1f}% SG={tst['sharp_gap']:+.1f} IMM={tst['immortal_capture_rate']:.1f}%")
        if fe:
            print(f"     Full:  Exec={fe['n_exec']} FE={fe['false_exec_rate']:.1f}% SG={fe['sharp_gap']:+.1f} IMM={fe['immortal_capture_rate']:.1f}%")
        print(f"     FE Gap (Train→Test): {result['fe_gap']:.1f}%p")

        if result['e_neg_by_bin']:
            top_e_neg = sorted(result['e_neg_by_bin'].items(), key=lambda x: -x[1])[:3]
            print(f"     Top E- bins: {', '.join(f'{k}({v})' for k,v in top_e_neg)}")

        world_results[label] = {
            'config': {k: v for k, v in cfg.items() if k != 'name'},
            'train_eval': te,
            'test_eval': tst,
            'full_eval': fe,
            'fe_gap': result['fe_gap'],
            'event_counts': ec,
            'full_computes': result['full_computes'],
            'fast_computes': result['fast_computes'],
            'compute_savings_pct': result['compute_savings_pct'],
            'active_bins_final': result['active_bins_final'],
            'e_neg_by_bin': result['e_neg_by_bin'],
        }

    print(f"\n  ═══ SWEEP COMPARISON ═══")
    print(f"  {'Config':<30s}  {'Saved':>6s}  {'FE-Tr':>6s}  {'FE-Ts':>6s}  {'Gap':>5s}  {'SG':>7s}  {'IMM':>6s}  {'E-':>4s}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*4}")

    baseline = world_results.get('Baseline (64a)')
    baseline_gap = baseline['fe_gap'] if baseline else 0

    for label, wr in world_results.items():
        te = wr['train_eval']
        tst = wr['test_eval']
        fe = wr['full_eval']
        fe_tr = te['false_exec_rate'] if te else 0
        fe_ts = tst['false_exec_rate'] if tst else 0
        sg = fe['sharp_gap'] if fe else 0
        imm = fe['immortal_capture_rate'] if fe else 0
        e_neg = wr['event_counts']['E-']
        print(f"  {label:<30s}  {wr['compute_savings_pct']:>5.1f}%  {fe_tr:>5.1f}%  {fe_ts:>5.1f}%  {wr['fe_gap']:>4.1f}  {sg:>+6.1f}  {imm:>5.1f}%  {e_neg:>4d}")

    print(f"\n  ═══ VERDICT ═══")

    best_pdl = None
    best_gap_reduction = 0
    for label, wr in world_results.items():
        if 'PDL' in label:
            gap_reduction = baseline_gap - wr['fe_gap']
            if gap_reduction > best_gap_reduction:
                best_gap_reduction = gap_reduction
                best_pdl = wr
                best_pdl['label'] = label

    if best_pdl:
        fe = best_pdl['full_eval']
        imm_ok = fe and fe['immortal_capture_rate'] >= 80.0
        sg_ok = fe and fe['sharp_gap'] >= 70.0

        gap_halved = best_pdl['fe_gap'] <= baseline_gap / 2
        gap_reduced = best_gap_reduction > 0

        compute_ok = best_pdl['compute_savings_pct'] >= 30.0

        print(f"\n  Best config: {best_pdl['label']}")
        print(f"  FE Gap: {baseline_gap:.1f}%p → {best_pdl['fe_gap']:.1f}%p (Δ{-best_gap_reduction:+.1f}%p)")
        print(f"    Gap halved (≤{baseline_gap/2:.1f}%p)? {'✅' if gap_halved else '❌'}")
        print(f"    Gap reduced? {'✅' if gap_reduced else '❌'}")
        print(f"    IMM ≥80%? {'✅' if imm_ok else '❌'} ({fe['immortal_capture_rate']:.1f}%)" if fe else "")
        print(f"    SG ≥70%p? {'✅' if sg_ok else '❌'} ({fe['sharp_gap']:+.1f}%p)" if fe else "")
        print(f"    Compute ≥30% saved? {'✅' if compute_ok else '❌'} ({best_pdl['compute_savings_pct']:.1f}%)")

        if gap_halved and imm_ok and sg_ok:
            print(f"\n  ✅ PDL DIRECTED PHEROMONE SUCCESSFUL")
            print(f"     'wrong conviction' following guiding field overfit breakraised.")
        elif gap_reduced and imm_ok and sg_ok:
            print(f"\n  ⚠️  PARTIAL — gap reduced but not halved")
            print(f"     PDLDirection is correct but effect is insufficient")
            print(f"     → More data or more refined E-definition needed")
        elif imm_ok and sg_ok:
            print(f"\n  ⚠️  PARTIAL — laws preserved but gap not reduced")
            print(f"     PDL observation density bardreamedonly overfit resolvedat insufficient")
        else:
            print(f"\n  ❌ PDL FAILED — laws violated or gap increased")
    else:
        print(f"\n  ❌ No PDL config improved over baseline")

    for label, wr in world_results.items():
        if 'Aggressive' in label:
            fe = wr['full_eval']
            if fe:
                print(f"\n  Stress test ({label}):")
                print(f"    FE={fe['false_exec_rate']:.1f}% SG={fe['sharp_gap']:+.1f} IMM={fe['immortal_capture_rate']:.1f}%")
                if fe['immortal_capture_rate'] < 80.0:
                    print(f"    ⚠️  Aggressive PDL breaks IMMORTAL preservation")
                else:
                    print(f"    ✅ Even aggressive PDL preserves laws")

    exp64c_dir = os.path.join(EVIDENCE_DIR, 'exp64c_pdl_directed_pheromone')
    os.makedirs(exp64c_dir, exist_ok=True)

    exp_data = {
        'experiment': 'EXP-64c PDL Directed Pheromone',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'n_trades': n_total,
        'n_train': n_train,
        'n_test': n_test,
        'train_ratio': TRAIN_RATIO,
        'theta': THETA,
        'baseline_fe_gap': baseline_gap,
        'world_results': world_results,
    }

    with open(os.path.join(exp64c_dir, 'pdl_directed_results.json'), 'w') as f:
        json.dump(exp_data, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    print(f"\n  --- EXP-64c Saved ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  'learning casea guiding field that follows — chasing information, not money.'")


if __name__ == '__main__':
    main()
