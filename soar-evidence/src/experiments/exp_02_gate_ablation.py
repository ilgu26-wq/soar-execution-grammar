#!/usr/bin/env python3
"""
EXP-02: Same Alpha, Gate Component Ablation
=============================================
Question: "If we enable gate components one by one on the same alpha,
       what protective effect does each have?"

Comparison modes:
  A) RAW          — gate none (baseline)
  B) DD-only      — DD breachonly blocking
  C) DD+Streak    — DD breach + consecutiveloss pause
  D) FULL (SOAR)  — DD + Streak + High-vol caution

Measurement:
  - Win rate preservation
  - DD change
  - Consecutive loss streak length (max / avg)

Execution:
  python experiments/exp_02_gate_ablation.py
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
        'count': len(streaks),
    }


def run_mode(signals, records, mode='RAW', dd_limit=0.03,
             consec_pause=3, warmup=300, tick_value=5.0):
    """
    mode:
      RAW        = no gate
      DD_ONLY    = DD breach only
      DD_STREAK  = DD + consecutive loss pause
      FULL       = DD + streak + high-vol caution
    """
    equity = 100_000.0
    peak = equity
    pnls = []
    denied = 0
    consec_losses = 0
    paused_until = -1

    dE_vals = np.array([r.get('dE', 0) for r in records], dtype=float)
    vol_short = np.zeros(len(records))
    vol_long = np.zeros(len(records))
    if mode == 'FULL':
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

            if mode in ('DD_ONLY', 'DD_STREAK', 'FULL'):
                if dd_pct > dd_limit:
                    block = True

            if mode in ('DD_STREAK', 'FULL') and not block:
                if consec_losses >= consec_pause and i < paused_until:
                    block = True

            if mode == 'FULL' and not block:
                vr = vol_short[i] / (vol_long[i] + EPS)
                if vr > 1.3 and dd_pct > dd_limit * 0.5:
                    block = True

            if mode == 'RAW':
                block = False

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
    cl = loss_clustering(pnls)

    return {
        'mode': mode,
        'trades': len(pnls),
        'denied': denied,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'max_loss_streak': cl['max_streak'],
        'avg_loss_streak': cl['avg_streak'],
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  EXP-02: Gate Component Ablation")
    print("  Question: Which gate component contributes what protection?")
    print("=" * 70)

    data_path = find_data()
    df = load_data(data_path)
    signals = generate_signals(df)
    records = [{'dE': float(row.get('dE', 0)), 'z_norm': float(row.get('z_norm', 0))}
               for _, row in df.iterrows()]
    print(f"\n  Data: {len(df)} bars, {len(signals)} signals")

    modes = ['RAW', 'DD_ONLY', 'DD_STREAK', 'FULL']
    labels = {
        'RAW': 'A) No Gate',
        'DD_ONLY': 'B) DD Only',
        'DD_STREAK': 'C) DD+Streak',
        'FULL': 'D) FULL (SOAR)',
    }

    results = {}
    for m in modes:
        results[m] = run_mode(signals, records, mode=m)

    header = f"  {'Metric':<20s}"
    for m in modes:
        header += f" {labels[m]:>14s}"
    print(f"\n{header}")
    print(f"  {'-'*76}")

    rows = [
        ('Trades', 'trades', 'd'),
        ('Denied', 'denied', 'd'),
        ('PF', 'pf', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Max DD %', 'max_dd_pct', '.2f'),
        ('Net PnL $', 'net_pnl', '.0f'),
        ('Max Loss Streak', 'max_loss_streak', 'd'),
        ('Avg Loss Streak', 'avg_loss_streak', '.2f'),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<20s}"
        for m in modes:
            v = results[m][key]
            line += f" {v:>14{fmt}}"
        print(line)

    print(f"\n  {'='*76}")
    print(f"  ABLATION ANALYSIS")
    print(f"  {'='*76}")

    raw_dd = results['RAW']['max_dd_pct']
    for m in ['DD_ONLY', 'DD_STREAK', 'FULL']:
        dd = results[m]['max_dd_pct']
        delta = raw_dd - dd
        wr_diff = results[m]['win_rate'] - results['RAW']['win_rate']
        print(f"  {labels[m]:>14s}: DD Δ={delta:+.2f}%, WR Δ={wr_diff:+.1f}%, "
              f"streak {results['RAW']['max_loss_streak']}→{results[m]['max_loss_streak']}")

    full_dd_ok = results['FULL']['max_dd_pct'] < results['RAW']['max_dd_pct']
    wr_held = abs(results['FULL']['win_rate'] - results['RAW']['win_rate']) < 5.0
    status = 'PASS' if (full_dd_ok and wr_held) else 'FAIL'
    print(f"\n  EXP-02 STATUS: {status}")
    if full_dd_ok:
        print(f"  [PASS] DD reduced with FULL gate")
    else:
        print(f"  [FAIL] DD not reduced")
    if wr_held:
        print(f"  [PASS] Win rate preserved (Δ < 5%)")
    else:
        print(f"  [WARN] Win rate shifted significantly")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'experiment': 'EXP-02-Gate-Ablation',
        'timestamp': datetime.now().isoformat(),
        'data': os.path.basename(data_path),
        'bars': len(df),
        'signals': len(signals),
        'results': {m: {k: v for k, v in r.items()} for m, r in results.items()},
        'status': status,
    }
    path = os.path.join(EVIDENCE_DIR, 'exp_02_evidence.json')
    with open(path, 'w') as f:
        json.dump(evidence, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer)) else float(o) if isinstance(o, np.floating) else None)
    print(f"  Evidence: {path}")


if __name__ == '__main__':
    main()
