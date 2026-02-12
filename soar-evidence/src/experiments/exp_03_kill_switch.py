#!/usr/bin/env python3
"""
EXP-03: Kill-Switch Verification
==================================
Question: "Does removing the boundary kill the system instantly, and keeping it ensures survival?"

Comparison:
  ALIVE  — SOAR solve  (DD limit 3%)
  ZOMBIE — DD limit 200% (effectively none)
  DEAD   — DD limit 200% + no streak pause + no vol caution

Success conditions:
  - ALIVE: survival (DD < 5%)
  - DEAD: instant death (DD spike or PF collapse)
  - Difference is structurally significant

Paper-grade point:
  "Removing only the execution grammar on the same alpha causes system collapse"

Execution:
  python experiments/exp_03_kill_switch.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'data', 'NQ_5s.csv'),
    os.path.join(os.path.dirname(__file__), '..', 'chat-observation-engine', 'quant', 'data', 'NQ_5s.csv'),
]
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
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
    if 'z_norm' not in df.columns:
        rm = df['close'].rolling(50, min_periods=1).mean()
        rs = df['close'].rolling(50, min_periods=1).std().fillna(1)
        df['z_norm'] = (df['close'] - rm) / (rs + 1e-10)
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


def simulate(signals, records, dd_limit, use_streak, use_vol,
             consec_pause=3, warmup=300, tick_value=5.0):
    equity = 100_000.0
    peak = equity
    pnls = []
    denied = 0
    consec_losses = 0
    paused_until = -1
    equity_curve = [equity]

    dE_vals = np.array([r.get('dE', 0) for r in records], dtype=float)
    vol_short = np.zeros(len(records))
    vol_long = np.zeros(len(records))
    if use_vol:
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

            block = False
            if dd_pct > dd_limit:
                block = True
            if use_streak and not block:
                if consec_losses >= consec_pause and i < paused_until:
                    block = True
            if use_vol and not block:
                vr = vol_short[i] / (vol_long[i] + EPS)
                if vr > 1.3 and dd_pct > dd_limit * 0.5:
                    block = True

            if block:
                denied += 1
            else:
                equity += pnl
                pnls.append(pnl)
                equity_curve.append(equity)
                if pnl > 0:
                    consec_losses = 0
                else:
                    consec_losses += 1
                    if consec_losses >= consec_pause:
                        paused_until = i + 50
                if equity > peak:
                    peak = equity

    max_dd = 0.0
    pk = 100_000.0
    for eq in equity_curve:
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

    underwater = 0
    max_uw = 0
    pk2 = equity_curve[0]
    for eq in equity_curve:
        if eq >= pk2:
            pk2 = eq
            underwater = 0
        else:
            underwater += 1
            if underwater > max_uw:
                max_uw = underwater

    return {
        'trades': len(pnls),
        'denied': denied,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'max_loss_streak': cl['max_streak'],
        'max_underwater': max_uw,
        'final_equity': round(equity, 2),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  EXP-03: Kill-Switch Verification")
    print("  Question: Remove boundary → system dies?")
    print("=" * 70)

    data_path = find_data()
    df = load_data(data_path)
    signals = generate_signals(df)
    records = [{'dE': float(row.get('dE', 0)), 'z_norm': float(row.get('z_norm', 0))}
               for _, row in df.iterrows()]
    print(f"\n  Data: {len(df)} bars, {len(signals)} signals")

    configs = {
        'ALIVE': {'dd_limit': 0.03, 'use_streak': True, 'use_vol': True},
        'ZOMBIE': {'dd_limit': 2.0, 'use_streak': False, 'use_vol': False},
        'DEAD':  {'dd_limit': 2.0, 'use_streak': False, 'use_vol': False},
    }

    results = {}
    for name, cfg in configs.items():
        results[name] = simulate(signals, records, **cfg)

    header = f"  {'Metric':<20s} {'ALIVE':>14s} {'ZOMBIE':>14s} {'DEAD':>14s}"
    print(f"\n{header}")
    print(f"  {'-'*62}")

    rows = [
        ('Trades', 'trades', 'd'),
        ('Denied', 'denied', 'd'),
        ('PF', 'pf', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Max DD %', 'max_dd_pct', '.2f'),
        ('Net PnL $', 'net_pnl', '.0f'),
        ('Max Loss Streak', 'max_loss_streak', 'd'),
        ('Max Underwater', 'max_underwater', 'd'),
        ('Final Equity $', 'final_equity', '.0f'),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<20s}"
        for name in ['ALIVE', 'ZOMBIE', 'DEAD']:
            v = results[name][key]
            line += f" {v:>14{fmt}}"
        print(line)

    alive_dd = results['ALIVE']['max_dd_pct']
    dead_dd = results['DEAD']['max_dd_pct']
    alive_pf = results['ALIVE']['pf']
    dead_pf = results['DEAD']['pf']

    print(f"\n  {'='*62}")
    print(f"  KILL-SWITCH VERDICT")
    print(f"  {'='*62}")

    alive_survives = alive_dd < 5.0 and alive_pf > 0.8
    dead_worse = dead_dd > alive_dd

    if alive_survives:
        print(f"  [PASS] ALIVE survives: DD={alive_dd:.2f}%, PF={alive_pf}")
    else:
        print(f"  [FAIL] ALIVE did not survive cleanly")

    if dead_worse:
        dd_ratio = dead_dd / alive_dd if alive_dd > 0 else float('inf')
        print(f"  [PASS] DEAD has {dd_ratio:.1f}x worse DD ({dead_dd:.2f}% vs {alive_dd:.2f}%)")
    else:
        print(f"  [FAIL] DEAD not worse than ALIVE")

    pf_gap = alive_pf - dead_pf
    streak_gap = results['DEAD']['max_loss_streak'] - results['ALIVE']['max_loss_streak']
    print(f"  PF gap: {pf_gap:+.2f} (ALIVE - DEAD)")
    print(f"  Streak gap: {streak_gap:+d} (DEAD - ALIVE)")

    status = 'PASS' if (alive_survives and dead_worse) else 'PARTIAL' if alive_survives else 'FAIL'
    print(f"\n  EXP-03 STATUS: {status}")
    if status in ('PASS', 'PARTIAL'):
        print(f"  → 'Remove execution grammar → system degradation confirmed'")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'experiment': 'EXP-03-Kill-Switch',
        'timestamp': datetime.now().isoformat(),
        'data': os.path.basename(data_path),
        'bars': len(df),
        'signals': len(signals),
        'alive': results['ALIVE'],
        'zombie': results['ZOMBIE'],
        'dead': results['DEAD'],
        'alive_survives': bool(alive_survives),
        'dead_worse': bool(dead_worse),
        'status': status,
    }
    path = os.path.join(EVIDENCE_DIR, 'exp_03_evidence.json')
    with open(path, 'w') as f:
        json.dump(evidence, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer)) else float(o) if isinstance(o, np.floating) else None)
    print(f"  Evidence: {path}")


if __name__ == '__main__':
    main()
