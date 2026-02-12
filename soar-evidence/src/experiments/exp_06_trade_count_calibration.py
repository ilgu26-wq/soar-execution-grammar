#!/usr/bin/env python3
"""
EXP-06: Trade Count Matched Calibration
=========================================
Goal: Fair comparison after matching v1 and v2 trade counts

Method: Relax v1's policy threshold to raise trades to v2 level(~900)to/as raise
     Sweep multiple threshold levels to find optimal matching point

Key question:
  "Does v1 have a real edge, or is it just a brake?"

Comparison:
  - Direct comparison of PF/DD/PnL at similar trade counts
  - v1 with PF > 1.0 + DD < v2 → v1 has edge
  - v1 with PF < 1.0 or DD ≥ v2 → v1 is just a brake

Execution:
  python experiments/exp_06_trade_count_calibration.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'data', 'NQ_5s.csv'),
    os.path.join(os.path.dirname(__file__), '..', 'chat-observation-engine', 'quant', 'data', 'NQ_5s.csv'),
]
EPS = 1e-10

N_BOUNDARY_BINS = 3
N_FLIP_BINS = 2
N_DIRECTION_BINS = 3
N_ROUTES = N_BOUNDARY_BINS * N_FLIP_BINS * N_DIRECTION_BINS
P_MAX = 10.0


def find_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    print("ERROR: NQ_5s.csv not found")
    sys.exit(1)


def load_data(path):
    df = pd.read_csv(path)
    if 'dE' not in df.columns:
        df['dE'] = df['close'].diff().fillna(0)
    if 'd2E' not in df.columns:
        df['d2E'] = df['dE'].diff().fillna(0)
    if 'z_norm' not in df.columns:
        rm = df['close'].rolling(50, min_periods=1).mean()
        rs = df['close'].rolling(50, min_periods=1).std().fillna(1)
        df['z_norm'] = (df['close'] - rm) / (rs + 1e-10)
    if 'dc' not in df.columns:
        r = df['close'].rolling(20, min_periods=1)
        df['dc'] = ((df['close'] - r.min()) / (r.max() - r.min() + 1e-10)).fillna(0.5)
    if 'vol_ratio' not in df.columns:
        sv = df['close'].rolling(20, min_periods=1).std()
        lv = df['close'].rolling(100, min_periods=1).std()
        df['vol_ratio'] = (sv / (lv + 1e-10)).fillna(1.0)
    if 'ch_range' not in df.columns:
        r20 = df['close'].rolling(20, min_periods=1)
        df['ch_range'] = (r20.max() - r20.min()).fillna(0)
    return df


def generate_signals(df, stop_ticks=5.0, min_gap=10):
    signals = []
    n = len(df)
    last_idx = -999
    dE = df['dE'].values
    z_norm = df['z_norm'].values
    close = df['close'].values
    er_vals = np.abs(dE)
    er_20 = pd.Series(er_vals).rolling(20, min_periods=1).mean().values
    for i in range(100, n - 20):
        if i - last_idx < min_gap:
            continue
        er = er_20[i]
        if er < 0.5:
            continue
        if abs(z_norm[i]) > 1.0 and abs(dE[i]) > er * 0.8:
            direction = 1 if dE[i] > 0 else -1
            pnl_ticks = 0.0
            for j in range(1, min(20, n - i)):
                move = (close[i + j] - close[i]) * direction / 0.25
                if move <= -stop_ticks:
                    pnl_ticks = -stop_ticks
                    break
                if move >= stop_ticks * 2:
                    pnl_ticks = stop_ticks * 2
                    break
                pnl_ticks = move
            signals.append({'bar_idx': i, 'direction': direction,
                            'pnl_ticks': round(pnl_ticks, 2)})
            last_idx = i
    return signals


def make_records(df):
    records = []
    for _, row in df.iterrows():
        rec = {}
        for c in ['dE', 'd2E', 'z_norm', 'dc', 'vol_ratio', 'ch_range']:
            rec[c] = float(row.get(c, 0))
        rec['pnl'] = float(row.get('dE', 0))
        rec['storm_flag'] = 0
        records.append(rec)
    return records


def loss_clustering(pnls):
    streaks = []
    cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    return max(streaks) if streaks else 0


def _discretize_route(rec, d2e_p70):
    margins = [
        1 - abs(rec['storm_flag']),
        max((rec['dc'] - 0.3) / 0.7, 0),
        max((50 - rec['ch_range']) / 50, 0),
        max((3.0 - rec['vol_ratio']) / 3.0, 0),
        max((d2e_p70 - abs(rec['d2E'])) / (d2e_p70 + EPS), 0),
    ]
    b = float(np.mean(margins))
    f = float(min(abs(rec['dE']) / (d2e_p70 + EPS), 1.0))
    d = float(rec['dE'] * 0.5 + (rec['dc'] - 0.5) * 2.0 + rec['z_norm'] * 0.3)
    b_bin = min(int(b * N_BOUNDARY_BINS), N_BOUNDARY_BINS - 1)
    f_bin = min(int(f * N_FLIP_BINS), N_FLIP_BINS - 1)
    d_bin = 0 if d < -0.3 else (2 if d > 0.3 else 1)
    return b_bin * (N_FLIP_BINS * N_DIRECTION_BINS) + f_bin * N_DIRECTION_BINS + d_bin


def run_v1_calibrated(signals, records, policy_cap, prob_floor=0.0,
                      tick_value=5.0, warmup=300, decay=0.05):
    """
    v1 with adjustable policy_cap and prob_floor.
    Higher policy_cap + lower prob_floor = more trades allowed.
    """
    n = len(records)
    rng = np.random.RandomState(42)
    rolls = rng.random(n)
    pnl_scale = np.std([r['pnl'] for r in records]) + EPS
    d2e_p70 = float(np.percentile([abs(r['dE']) for r in records], 70))

    pheromone = np.zeros(N_ROUTES)
    policy_level = 0.0
    prev_coord_delta = 0.0

    sig_map = {}
    for sig in signals:
        sig_map.setdefault(sig['bar_idx'], []).append(sig)

    equity = 100_000.0
    peak = equity
    pnls = []
    denied = 0

    for i, rec in enumerate(records):
        route_id = _discretize_route(rec, d2e_p70)
        reward_norm = rec['pnl'] / pnl_scale

        sp_level = pheromone[route_id]
        shadow_prob = 0.5 + 0.4 * np.tanh(sp_level * 2.0)
        shadow_decision = 'execute' if rolls[i] < shadow_prob else 'skip'

        pheromone *= (1 - decay)
        if shadow_decision == 'execute' and reward_norm > 0:
            pheromone[route_id] += reward_norm
        pheromone = np.clip(pheromone, 0, P_MAX)

        if i < warmup:
            continue

        pher_std = float(np.std(pheromone))
        kinetic = abs(reward_norm) * pher_std
        coord_delta = kinetic - prev_coord_delta
        prev_coord_delta = kinetic
        forbidden_zone = abs(reward_norm) > 2.0

        input_signal = coord_delta
        if forbidden_zone:
            input_signal *= 2.0
        boundary_crossed = abs(coord_delta) > 0.5
        should_act = boundary_crossed and abs(input_signal) > 0.05

        action = None
        if should_act:
            action = 'restrict' if input_signal > 0 else 'release'
            if action == 'restrict':
                policy_level += 0.1
            elif action == 'release':
                policy_level = max(0, policy_level - 0.05)

        if i in sig_map:
            for sig in sig_map[i]:
                pnl_dollar = sig['pnl_ticks'] * tick_value
                gp_level = float(pheromone[route_id])
                exec_prob = 0.5 + 0.4 * np.tanh(gp_level * 2.0)

                effective_prob = max(exec_prob, prob_floor)
                policy_ok = (policy_level < policy_cap) or (action == 'release')
                prob_ok = rolls[i] < effective_prob

                if policy_ok and prob_ok:
                    equity += pnl_dollar
                    pnls.append(pnl_dollar)
                    if equity > peak:
                        peak = equity
                else:
                    denied += 1

    max_dd = 0.0
    eq = 100_000.0
    pk = eq
    for p in pnls:
        eq += p
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk if pk > 0 else 0
        if dd > max_dd:
            max_dd = dd

    wins = sum(1 for p in pnls if p > 0)
    gp = sum(p for p in pnls if p > 0)
    gl = sum(abs(p) for p in pnls if p <= 0)
    pf = gp / gl if gl > 0 else float('inf')
    ms = loss_clustering(pnls)

    return {
        'trades': len(pnls),
        'denied': denied,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'max_loss_streak': ms,
        'policy_cap': policy_cap,
        'prob_floor': prob_floor,
    }


def run_v2(signals, records, dd_limit=0.03, consec_pause=3,
           warmup=300, tick_value=5.0):
    equity = 100_000.0
    peak = equity
    pnls = []
    denied = 0
    consec_losses = 0
    paused_until = -1

    dE_vals = np.array([r.get('dE', 0) for r in records], dtype=float)
    vol_short = np.zeros(len(records))
    vol_long = np.zeros(len(records))
    for i in range(len(records)):
        lo = max(0, i - 20)
        lo2 = max(0, i - 100)
        vol_short[i] = np.std(dE_vals[lo:i+1]) if i >= 1 else 0
        vol_long[i] = np.std(dE_vals[lo2:i+1]) if i >= 1 else 0

    sig_map = {}
    for sig in signals:
        sig_map.setdefault(sig['bar_idx'], []).append(sig)

    for i in range(len(records)):
        if i < warmup or i not in sig_map:
            continue
        for sig in sig_map[i]:
            pnl = sig['pnl_ticks'] * tick_value
            dd_pct = (peak - equity) / peak if peak > 0 else 0
            vr = vol_short[i] / (vol_long[i] + EPS)
            regime = 'HIGH' if vr > 1.3 else ('LOW' if vr < 0.7 else 'MID')

            block = False
            if dd_pct > dd_limit:
                block = True
            elif consec_losses >= consec_pause and i < paused_until:
                block = True
            elif regime == 'HIGH' and dd_pct > dd_limit * 0.5:
                block = True

            if block:
                denied += 1
            else:
                equity += pnl
                pnls.append(pnl)
                if pnl > 0:
                    consec_losses = 0
                else:
                    consec_losses += 1
                    if consec_losses >= consec_pause:
                        paused_until = i + 50
                if equity > peak:
                    peak = equity

    max_dd = 0.0
    eq = 100_000.0
    pk = eq
    for p in pnls:
        eq += p
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk if pk > 0 else 0
        if dd > max_dd:
            max_dd = dd

    wins = sum(1 for p in pnls if p > 0)
    gp = sum(p for p in pnls if p > 0)
    gl = sum(abs(p) for p in pnls if p <= 0)
    pf = gp / gl if gl > 0 else float('inf')
    ms = loss_clustering(pnls)

    return {
        'trades': len(pnls),
        'denied': denied,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'max_loss_streak': ms,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  EXP-06: Trade Count Matched Calibration")
    print("  Question: Does v1 have real edge, or is it just a brake?")
    print("=" * 70)

    data_path = find_data()
    df = load_data(data_path)
    signals = generate_signals(df)
    records = make_records(df)
    print(f"\n  Data: {os.path.basename(data_path)} ({len(df)} bars)")
    print(f"  Signals: {len(signals)} candidates")

    v2_result = run_v2(signals, records)
    v2_trades = v2_result['trades']
    print(f"\n  V2 baseline: {v2_trades} trades, PF={v2_result['pf']}, "
          f"DD={v2_result['max_dd_pct']}%")

    print(f"\n  Sweeping v1 calibrations to match ~{v2_trades} trades...")
    print(f"  {'policy_cap':>12s} {'prob_floor':>12s} {'trades':>8s} {'PF':>8s} "
          f"{'DD%':>8s} {'PnL$':>10s} {'WR%':>8s} {'streak':>8s}")
    print(f"  {'-'*76}")

    sweep_results = []
    policy_caps = [2.0, 5.0, 10.0, 50.0, 999.0]
    prob_floors = [0.0, 0.3, 0.5, 0.7, 0.9]

    for pc in policy_caps:
        for pf in prob_floors:
            r = run_v1_calibrated(signals, records, policy_cap=pc, prob_floor=pf)
            sweep_results.append(r)
            print(f"  {pc:>12.1f} {pf:>12.1f} {r['trades']:>8d} {r['pf']:>8.2f} "
                  f"{r['max_dd_pct']:>8.2f} {r['net_pnl']:>10.0f} "
                  f"{r['win_rate']:>8.1f} {r['max_loss_streak']:>8d}")

    best_match = None
    best_diff = float('inf')
    for r in sweep_results:
        diff = abs(r['trades'] - v2_trades)
        if diff < best_diff:
            best_diff = diff
            best_match = r

    candidates = [r for r in sweep_results
                  if abs(r['trades'] - v2_trades) <= max(v2_trades * 0.3, 100)]
    if not candidates:
        candidates = sorted(sweep_results, key=lambda x: abs(x['trades'] - v2_trades))[:3]

    print(f"\n  {'='*76}")
    print(f"  BEST MATCH (closest to {v2_trades} trades)")
    print(f"  {'='*76}")
    print(f"  v1 config: policy_cap={best_match['policy_cap']}, "
          f"prob_floor={best_match['prob_floor']}")
    print(f"  v1 trades: {best_match['trades']}")

    header = f"\n  {'Metric':<20s} {'V2':>12s} {'V1_matched':>12s} {'Delta':>12s}"
    print(header)
    print(f"  {'-'*56}")

    compare_rows = [
        ('Trades', 'trades', 'd'),
        ('PF', 'pf', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Max DD %', 'max_dd_pct', '.2f'),
        ('Net PnL $', 'net_pnl', '.0f'),
        ('Avg PnL $', 'avg_pnl', '.2f'),
        ('Max Loss Streak', 'max_loss_streak', 'd'),
    ]

    for label, key, fmt in compare_rows:
        v2v = v2_result[key]
        v1v = best_match[key]
        d = v1v - v2v
        sign = '+' if d > 0 else ''
        print(f"  {label:<20s} {v2v:>12{fmt}} {v1v:>12{fmt}} {sign}{d:>11{fmt}}")

    print(f"\n  {'='*56}")
    print(f"  CALIBRATION VERDICT")
    print(f"  {'='*56}")

    v1_has_edge = best_match['pf'] > 1.0
    v1_has_dd_advantage = best_match['max_dd_pct'] < v2_result['max_dd_pct']
    v1_profitable = best_match['net_pnl'] > 0

    if v1_has_edge and v1_has_dd_advantage:
        conclusion = "V1_HAS_EDGE"
        print(f"  [EDGE] v1 has genuine edge: PF>{1.0} AND lower DD")
        print(f"  → v1 is a structural filter, not just a brake")
    elif v1_has_edge and not v1_has_dd_advantage:
        conclusion = "V1_EDGE_NO_DD"
        print(f"  [PARTIAL] v1 profitable but DD not better")
        print(f"  → v1 has edge but v2 is better for DD protection")
    elif not v1_has_edge and v1_has_dd_advantage:
        conclusion = "V1_BRAKE_ONLY"
        print(f"  [BRAKE] v1 reduces DD but kills PF — it's just a brake")
        print(f"  → v1 is just a brake. No edge.")
    else:
        conclusion = "V1_NO_VALUE"
        print(f"  [NONE] v1 neither profitable nor protective")
        print(f"  → v1 has no value in this signal pool")

    print(f"\n  CONCLUSION: {conclusion}")

    if conclusion in ('V1_BRAKE_ONLY', 'V1_NO_VALUE'):
        print(f"  → v2 standalone production deployment confirmed")
        print(f"  → v1 for research sealed")
    elif conclusion == 'V1_HAS_EDGE':
        print(f"  → Integrate v1's scoring as overlay into v2 (see EXP-05 results)")
    else:
        print(f"  → v2 main + v1 optional DD overlay")

    elapsed = time.time() - t0
    print(f"\n  EXP-06 STATUS: {conclusion}")
    print(f"  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'experiment': 'EXP-06-Trade-Count-Calibration',
        'timestamp': datetime.now().isoformat(),
        'data': os.path.basename(data_path),
        'bars': len(df),
        'signals': len(signals),
        'v2_baseline': v2_result,
        'best_match': best_match,
        'all_sweeps': sweep_results,
        'conclusion': conclusion,
    }
    path = os.path.join(EVIDENCE_DIR, 'exp_06_evidence.json')
    with open(path, 'w') as f:
        json.dump(evidence, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer))
                  else float(o) if isinstance(o, np.floating) else None)
    print(f"  Evidence: {path}")


if __name__ == '__main__':
    main()
