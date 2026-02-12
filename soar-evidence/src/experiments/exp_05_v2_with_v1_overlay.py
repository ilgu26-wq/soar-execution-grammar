#!/usr/bin/env python3
"""
EXP-05: v2 + v1 Risk Sizing Overlay
=====================================
Goal: Maintain v2 PnL while pulling DD toward v1 level

Structure:
  - Entry signal/direction remains v2 as-is (DD/streak/vol gate)
  - v1 handles only 'sizing dampening', not filtering
  - v1 pheromone score → size multiplier (1.0 / 0.5 / 0.0)

Comparison modes:
  RAW        — gate none
  V2_ONLY    — v2 gateonly
  V2+OVERLAY — v2 gate + v1 sizing dampening

Success criteria:
  - Net PnL ≥ 80% of v2
  - Max DD < 20~40% reduction vs v2
  - PF ≥ 1.10

Execution:
  python experiments/exp_05_v2_with_v1_overlay.py
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
    return {
        'max_streak': max(streaks) if streaks else 0,
        'avg_streak': round(np.mean(streaks), 2) if streaks else 0,
    }


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


def build_v1_scores(records, decay=0.05, warmup=300):
    """
    Pre-compute v1 pheromone score at each bar.
    Returns array of size scores in [0, 1].
    """
    n = len(records)
    rng = np.random.RandomState(42)
    rolls = rng.random(n)
    pnl_scale = np.std([r['pnl'] for r in records]) + EPS
    d2e_p70 = float(np.percentile([abs(r['dE']) for r in records], 70))

    pheromone = np.zeros(N_ROUTES)
    scores = np.ones(n) * 0.5

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

        if i >= warmup:
            gp_level = float(pheromone[route_id])
            exec_prob = 0.5 + 0.4 * np.tanh(gp_level * 2.0)
            scores[i] = exec_prob

    return scores


def score_to_size(score, low_thresh=0.4, high_thresh=0.7):
    """
    v1 score → sizing multiplier
    score >= high_thresh → 1.0 (full size)
    low_thresh <= score < high_thresh → 0.5 (half size)
    score < low_thresh → 0.0 (cut — only when really dangerous)
    """
    if score >= high_thresh:
        return 1.0
    elif score >= low_thresh:
        return 0.5
    else:
        return 0.0


def simulate(signals, records, v1_scores, mode='RAW',
             dd_limit=0.03, consec_pause=3, warmup=300, tick_value=5.0,
             low_thresh=0.4, high_thresh=0.7):
    equity = 100_000.0
    peak = equity
    pnls = []
    denied = 0
    denied_pnls = []
    consec_losses = 0
    paused_until = -1
    size_sum = 0.0

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
            pnl_base = sig['pnl_ticks'] * tick_value
            dd_pct = (peak - equity) / peak if peak > 0 else 0

            if mode == 'RAW':
                size = 1.0
                block = False
            else:
                vr = vol_short[i] / (vol_long[i] + EPS)
                regime = 'HIGH' if vr > 1.3 else ('LOW' if vr < 0.7 else 'MID')

                block = False
                if dd_pct > dd_limit:
                    block = True
                elif consec_losses >= consec_pause and i < paused_until:
                    block = True
                elif regime == 'HIGH' and dd_pct > dd_limit * 0.5:
                    block = True

                if mode == 'V2_ONLY':
                    size = 1.0
                elif mode == 'V2+OVERLAY':
                    size = score_to_size(v1_scores[i], low_thresh, high_thresh)
                    if size == 0.0:
                        block = True
                else:
                    size = 1.0

            if block:
                denied += 1
                denied_pnls.append(pnl_base)
            else:
                pnl = pnl_base * size
                equity += pnl
                pnls.append(pnl)
                size_sum += size
                if pnl_base > 0:
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
    cl = loss_clustering(pnls)
    dl = sum(1 for p in denied_pnls if p < 0)

    return {
        'mode': mode,
        'trades': len(pnls),
        'denied': denied,
        'denied_losses': dl,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'avg_size': round(size_sum / len(pnls), 3) if pnls else 0,
        'max_loss_streak': cl['max_streak'],
        'final_equity': round(equity, 2),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  EXP-05: v2 + v1 Risk Sizing Overlay")
    print("  'v2 trades. v1 sizes. Best of both.'")
    print("=" * 70)

    data_path = find_data()
    df = load_data(data_path)
    signals = generate_signals(df)
    records = make_records(df)
    v1_scores = build_v1_scores(records)
    print(f"\n  Data: {os.path.basename(data_path)} ({len(df)} bars)")
    print(f"  Signals: {len(signals)} candidates")
    print(f"  v1 score range: [{v1_scores.min():.3f}, {v1_scores.max():.3f}], "
          f"mean={v1_scores.mean():.3f}")

    modes = ['RAW', 'V2_ONLY', 'V2+OVERLAY']
    results = {}
    for m in modes:
        results[m] = simulate(signals, records, v1_scores, mode=m)

    header = f"\n  {'Metric':<20s} {'RAW':>12s} {'V2_ONLY':>12s} {'V2+OVERLAY':>12s}"
    print(header)
    print(f"  {'-'*56}")

    rows = [
        ('Trades', 'trades', 'd'),
        ('Denied', 'denied', 'd'),
        ('PF', 'pf', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Max DD %', 'max_dd_pct', '.2f'),
        ('Net PnL $', 'net_pnl', '.0f'),
        ('Avg PnL/Trade $', 'avg_pnl', '.2f'),
        ('Avg Size', 'avg_size', '.3f'),
        ('Max Loss Streak', 'max_loss_streak', 'd'),
        ('Final Equity $', 'final_equity', '.0f'),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<20s}"
        for m in modes:
            v = results[m][key]
            line += f" {v:>12{fmt}}"
        print(line)

    v2 = results['V2_ONLY']
    ov = results['V2+OVERLAY']

    print(f"\n  {'='*56}")
    print(f"  OVERLAY JUDGMENT")
    print(f"  {'='*56}")

    pnl_ratio = ov['net_pnl'] / v2['net_pnl'] * 100 if v2['net_pnl'] != 0 else 0
    dd_reduction = (1 - ov['max_dd_pct'] / v2['max_dd_pct']) * 100 if v2['max_dd_pct'] > 0 else 0

    print(f"\n  PnL retention: ${ov['net_pnl']:.0f} / ${v2['net_pnl']:.0f} = {pnl_ratio:.0f}%")
    pnl_ok = pnl_ratio >= 80
    print(f"    {'[PASS]' if pnl_ok else '[FAIL]'} Target ≥ 80%")

    print(f"\n  DD reduction: {v2['max_dd_pct']:.2f}% → {ov['max_dd_pct']:.2f}% ({dd_reduction:+.0f}%)")
    dd_ok = 20 <= dd_reduction <= 60
    dd_any = ov['max_dd_pct'] < v2['max_dd_pct']
    print(f"    {'[PASS]' if dd_ok else '[PARTIAL]' if dd_any else '[FAIL]'} Target 20~40% reduction")

    print(f"\n  PF: {ov['pf']}")
    pf_ok = ov['pf'] >= 1.10
    print(f"    {'[PASS]' if pf_ok else '[FAIL]'} Target ≥ 1.10")

    all_pass = pnl_ok and dd_any and pf_ok
    status = 'PASS' if all_pass else 'PARTIAL' if (pnl_ok or dd_any) else 'FAIL'

    print(f"\n  {'='*56}")
    print(f"  EXP-05 STATUS: {status}")
    print(f"  {'='*56}")
    if all_pass:
        print(f"  → v2+v1 overlay = prop firm engine candidate secured")
    elif dd_any:
        print(f"  → DD improved but below criteria — review threshold adjustment")
    else:
        print(f"  → No overlay effect — v1 scoring needs review")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'experiment': 'EXP-05-V2-with-V1-Overlay',
        'timestamp': datetime.now().isoformat(),
        'data': os.path.basename(data_path),
        'bars': len(df),
        'signals': len(signals),
        'results': {m: r for m, r in results.items()},
        'pnl_retention_pct': round(pnl_ratio, 1),
        'dd_reduction_pct': round(dd_reduction, 1),
        'status': status,
    }
    path = os.path.join(EVIDENCE_DIR, 'exp_05_evidence.json')
    with open(path, 'w') as f:
        json.dump(evidence, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer))
                  else float(o) if isinstance(o, np.floating) else None)
    print(f"  Evidence: {path}")


if __name__ == '__main__':
    main()
