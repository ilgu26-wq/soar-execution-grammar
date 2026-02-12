#!/usr/bin/env python3
"""
EXP-04: CORE v1 vs CORE v2
============================
"Same market. Same signals. Which survives?"

v1 (CORE): Pheromone routing + probabilistic execution
       Observer → Boundary → Judge → Reactor
       Execution probability = pheromone level + phase transition + policy level

v2 (SOAR): Structural DD/streak/vol gate
       DD breach → DENY
       Consecutive loss pause → DENY
       High vol + DD → DENY
       Otherwise → ALLOW

Comparison principles:
  ❌ Different signals prohibited
  ❌ Different data prohibited
  ❌ Parameter tuning prohibited
  ✅ Only execution allow/block logic differs

Execution:
  python experiments/exp_04_core_vs_v2.py
"""
import sys, os, json, time, hashlib
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
            signals.append({
                'bar_idx': i,
                'direction': direction,
                'pnl_ticks': round(pnl_ticks, 2),
            })
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


N_BOUNDARY_BINS = 3
N_FLIP_BINS = 2
N_DIRECTION_BINS = 3
N_ROUTES = N_BOUNDARY_BINS * N_FLIP_BINS * N_DIRECTION_BINS
P_MAX = 10.0
N_DC_BINS = 3
N_DE_BINS = 2
N_MICRO_STATES = 3 * N_DC_BINS * N_DE_BINS


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


def _discretize_micro(rec):
    z = rec['z_norm']
    z_bin = 0 if z < -0.1 else (2 if z > 0.1 else 1)
    dc = rec['dc']
    dc_bin = 0 if dc < 0.4 else (2 if dc > 0.6 else 1)
    de_bin = 0 if rec['dE'] < 0 else 1
    return z_bin * (N_DC_BINS * N_DE_BINS) + dc_bin * N_DE_BINS + de_bin


def _compute_sti(micro, decs, theta=0.5):
    trans_exec = defaultdict(int)
    trans_total = defaultdict(int)
    for i in range(len(micro) - 1):
        tkey = (micro[i], micro[i + 1])
        trans_total[tkey] += 1
        if decs[i] == 'execute':
            trans_exec[tkey] += 1
    if not trans_total:
        return 0.0
    selected = measured = 0
    for tkey in trans_total:
        if trans_total[tkey] >= 3:
            rate = trans_exec[tkey] / trans_total[tkey]
            measured += 1
            if rate >= theta:
                selected += 1
    return selected / measured if measured else 0.0


def _compute_edg(micro, decs):
    density = np.zeros(N_MICRO_STATES)
    for i, ms in enumerate(micro):
        if decs[i] == 'execute':
            density[ms] += 1
    total = density.sum()
    if total == 0:
        return 0.0
    p = density / total
    p_pos = p[p > 0]
    h = float(-np.sum(p_pos * np.log(p_pos)))
    return 1.0 - (h / np.log(N_MICRO_STATES))


def _normalize(val, lo, hi):
    return max(0.0, min(1.0, (val - lo) / (hi - lo + EPS)))


def _f_mid(z, mu=0.5, sigma=0.25):
    return float(np.exp(-0.5 * ((z - mu) / sigma) ** 2))


CALIB_STI_RANGE = (0.05, 0.35)
CALIB_EDG_RANGE = (0.1, 0.5)
CALIB_MHI_RANGE = (0.05, 0.25)
CALIB_STI_MU = 0.18
CALIB_EDG_MU = 0.28
CALIB_MHI_MU = 0.12


def run_v1(signals, records, tick_value=5.0, warmup=300, decay=0.05):
    """
    CORE v1: Pheromone probability + Policy level gate
    At each signal bar, v1 decides execute/deny based on:
      1. Pheromone level at that route → execution probability
      2. Policy level (Judge restrict/release accumulation)
      3. Policy must be < 2.0 (or Judge just released)
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
    denied_pnls = []

    for i, rec in enumerate(records):
        route_id = _discretize_route(rec, d2e_p70)
        reward_raw = rec['pnl']
        reward_norm = reward_raw / pnl_scale

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

                policy_ok = (policy_level < 2.0) or (action == 'release')
                prob_ok = rolls[i] < exec_prob

                if policy_ok and prob_ok:
                    equity += pnl_dollar
                    pnls.append(pnl_dollar)
                    if equity > peak:
                        peak = equity
                else:
                    denied += 1
                    denied_pnls.append(pnl_dollar)

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
        'mode': 'CORE_V1',
        'trades': len(pnls),
        'denied': denied,
        'denied_losses': dl,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'max_loss_streak': cl['max_streak'],
        'worst_5': sorted([round(p, 2) for p in pnls])[:5],
        'final_equity': round(equity, 2),
    }


def run_v2(signals, records, tick_value=5.0, dd_limit=0.03,
           consec_pause=3, warmup=300):
    """CORE v2: Structural DD/streak/vol gate"""
    equity = 100_000.0
    peak = equity
    pnls = []
    denied = 0
    denied_pnls = []
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
                denied_pnls.append(pnl)
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
    cl = loss_clustering(pnls)
    dl = sum(1 for p in denied_pnls if p < 0)

    return {
        'mode': 'CORE_V2',
        'trades': len(pnls),
        'denied': denied,
        'denied_losses': dl,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'max_loss_streak': cl['max_streak'],
        'worst_5': sorted([round(p, 2) for p in pnls])[:5],
        'final_equity': round(equity, 2),
    }


def run_raw(signals, tick_value=5.0):
    """No gate at all (baseline)"""
    equity = 100_000.0
    peak = equity
    pnls = []
    for sig in signals:
        pnl = sig['pnl_ticks'] * tick_value
        equity += pnl
        pnls.append(pnl)
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

    return {
        'mode': 'RAW',
        'trades': len(pnls),
        'denied': 0,
        'denied_losses': 0,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'max_loss_streak': cl['max_streak'],
        'worst_5': sorted([round(p, 2) for p in pnls])[:5],
        'final_equity': round(equity, 2),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  EXP-04: CORE v1 vs CORE v2")
    print("  'Same market. Same signals. Which survives?'")
    print("=" * 70)

    data_path = find_data()
    df = load_data(data_path)
    signals = generate_signals(df)
    records = make_records(df)
    print(f"\n  Data: {os.path.basename(data_path)} ({len(df)} bars)")
    print(f"  Signals: {len(signals)} candidates (identical for all modes)")
    wins = sum(1 for s in signals if s['pnl_ticks'] > 0)
    print(f"  Raw pool: {wins}W / {len(signals)-wins}L ({wins/len(signals)*100:.1f}% WR)")

    print(f"\n  Running 3 modes...")
    raw = run_raw(signals)
    v1 = run_v1(signals, records)
    v2 = run_v2(signals, records)

    modes = ['RAW', 'CORE_V1', 'CORE_V2']
    results = {'RAW': raw, 'CORE_V1': v1, 'CORE_V2': v2}

    header = f"\n  {'Metric':<20s} {'RAW':>12s} {'CORE v1':>12s} {'CORE v2':>12s}"
    print(header)
    print(f"  {'-'*56}")

    rows = [
        ('Trades', 'trades', 'd'),
        ('Denied', 'denied', 'd'),
        ('Denied (losses)', 'denied_losses', 'd'),
        ('PF', 'pf', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Max DD %', 'max_dd_pct', '.2f'),
        ('Net PnL $', 'net_pnl', '.0f'),
        ('Avg PnL/Trade $', 'avg_pnl', '.2f'),
        ('Max Loss Streak', 'max_loss_streak', 'd'),
        ('Final Equity $', 'final_equity', '.0f'),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<20s}"
        for m in modes:
            v = results[m][key]
            line += f" {v:>12{fmt}}"
        print(line)

    print(f"\n  Worst 5 Trades (PnL $):")
    for m in modes:
        tag = {'RAW': 'RAW', 'CORE_V1': 'v1', 'CORE_V2': 'v2'}[m]
        print(f"    {tag:>4s}: {results[m]['worst_5']}")

    print(f"\n  {'='*56}")
    print(f"  HEAD-TO-HEAD JUDGMENT")
    print(f"  {'='*56}")

    v1_dd = v1['max_dd_pct']
    v2_dd = v2['max_dd_pct']
    raw_dd = raw['max_dd_pct']

    print(f"\n  DD comparison (lower = better):")
    print(f"    RAW:  {raw_dd:.2f}%")
    print(f"    v1:   {v1_dd:.2f}%")
    print(f"    v2:   {v2_dd:.2f}%")
    if v2_dd < v1_dd:
        pct = (1 - v2_dd / v1_dd) * 100 if v1_dd > 0 else 0
        print(f"    → v2 wins DD by {pct:.0f}%")
    elif v1_dd < v2_dd:
        pct = (1 - v1_dd / v2_dd) * 100 if v2_dd > 0 else 0
        print(f"    → v1 wins DD by {pct:.0f}%")
    else:
        print(f"    → DD tied")

    print(f"\n  PF comparison (higher = better):")
    print(f"    RAW:  {raw['pf']}")
    print(f"    v1:   {v1['pf']}")
    print(f"    v2:   {v2['pf']}")
    if v2['pf'] >= v1['pf']:
        print(f"    → v2 wins PF ({v2['pf']} >= {v1['pf']})")
    else:
        print(f"    → v1 wins PF ({v1['pf']} > {v2['pf']})")

    print(f"\n  WR comparison:")
    print(f"    RAW:  {raw['win_rate']}%")
    print(f"    v1:   {v1['win_rate']}%")
    print(f"    v2:   {v2['win_rate']}%")
    wr_diff = abs(v1['win_rate'] - v2['win_rate'])
    print(f"    → Δ = {wr_diff:.1f}% {'(within ±5%)' if wr_diff < 5 else '(significant shift)'}")

    v2_dd_wins = v2_dd <= v1_dd
    v2_pf_holds = v2['pf'] >= v1['pf'] * 0.8
    wr_stable = wr_diff < 5.0

    print(f"\n  {'='*56}")
    print(f"  FINAL VERDICT")
    print(f"  {'='*56}")

    if v2_dd_wins and v2_pf_holds:
        winner = 'CORE_V2'
        reason = 'DD ↓ + PF maintained'
    elif not v2_dd_wins and v1['pf'] > v2['pf']:
        winner = 'CORE_V1'
        reason = 'Better DD + higher PF'
    else:
        winner = 'DRAW'
        reason = 'No clear structural advantage'

    print(f"  WINNER: {winner}")
    print(f"  REASON: {reason}")
    print(f"  DD:  v1={v1_dd:.2f}% vs v2={v2_dd:.2f}%")
    print(f"  PF:  v1={v1['pf']} vs v2={v2['pf']}")
    print(f"  WR:  v1={v1['win_rate']}% vs v2={v2['win_rate']}% (Δ={wr_diff:.1f}%)")

    if winner == 'CORE_V2':
        print(f"\n  → v2 production adoption basis secured")
        print(f"  → v1 sealed for research only")
    elif winner == 'CORE_V1':
        print(f"\n  → v1 still superior — v2 improvement needed")
    else:
        print(f"\n  → Additional verification needed")

    status = 'V2_WINS' if winner == 'CORE_V2' else ('V1_WINS' if winner == 'CORE_V1' else 'DRAW')
    elapsed = time.time() - t0
    print(f"\n  EXP-04 STATUS: {status}")
    print(f"  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'experiment': 'EXP-04-CORE-vs-V2',
        'timestamp': datetime.now().isoformat(),
        'data': os.path.basename(data_path),
        'bars': len(df),
        'signals': len(signals),
        'raw': {k: v for k, v in raw.items()},
        'core_v1': {k: v for k, v in v1.items()},
        'core_v2': {k: v for k, v in v2.items()},
        'winner': winner,
        'reason': reason,
        'status': status,
    }
    path = os.path.join(EVIDENCE_DIR, 'exp_04_evidence.json')
    with open(path, 'w') as f:
        json.dump(evidence, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer))
                  else float(o) if isinstance(o, np.floating) else None)
    print(f"  Evidence: {path}")


if __name__ == '__main__':
    main()
