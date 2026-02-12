#!/usr/bin/env python3
"""
EXP-01: RAW vs SOAR (Baseline)
================================
Question: "If we keep alpha unchanged and only change the execution grammar (SOAR),
       does the left-tail structurally decrease?"

Measurement:
  - Trades
  - Max DD
  - Worst 5 trades PnL
  - Blocked trades (count / loss ratio)

Execution:
  python experiments/exp_01_raw_vs_soar.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from engine import ExecutionEngine
from gate import EVGate, GateDecision

DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'data', 'NQ_5s.csv'),
    os.path.join(os.path.dirname(__file__), '..', 'chat-observation-engine', 'quant', 'data', 'NQ_5s.csv'),
]
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')


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
    d2E = df['d2E'].values
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
                'er': round(er, 4),
                'z': round(z_norm[i], 4),
            })
            last_idx = i
    return signals


def make_records(df):
    records = []
    for _, row in df.iterrows():
        rec = {}
        for c in ['dE', 'd2E', 'z_norm', 'dc', 'vol_ratio']:
            rec[c] = float(row.get(c, 0))
        records.append(rec)
    return records


def worst_n(pnls, n=5):
    s = sorted(pnls)
    return s[:n]


def loss_clustering(pnls):
    max_streak = 0
    cur = 0
    streaks = []
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
        'max_loss_streak': max(streaks) if streaks else 0,
        'avg_loss_streak': round(np.mean(streaks), 2) if streaks else 0,
        'loss_streak_count': len(streaks),
    }


def run_raw_detailed(signals, tick_value=5.0):
    equity = 100_000.0
    peak = equity
    pnls = []
    max_dd = 0.0

    for sig in signals:
        pnl = sig['pnl_ticks'] * tick_value
        equity += pnl
        pnls.append(pnl)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
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
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'worst_5': [round(p, 2) for p in worst_n(pnls, 5)],
        'max_loss_streak': cl['max_loss_streak'],
        'avg_loss_streak': cl['avg_loss_streak'],
        'pnl_list': pnls,
    }


def run_soar_detailed(signals, records, tick_value=5.0, dd_limit=0.03,
                      consec_pause=3, warmup=300):
    equity = 100_000.0
    peak = equity
    pnls = []
    denied_pnls = []
    max_dd = 0.0
    consec_losses = 0
    paused_until = -1
    EPS = 1e-10

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

    deny_reasons = {'DD_BREACH': 0, 'CONSEC_LOSS_PAUSE': 0, 'HIGH_VOL_CAUTION': 0}

    for i in range(len(records)):
        if i < warmup or i not in sig_map:
            continue
        for sig in sig_map[i]:
            pnl = sig['pnl_ticks'] * tick_value
            dd_pct = (peak - equity) / peak if peak > 0 else 0
            vr = vol_short[i] / (vol_long[i] + EPS)
            regime = 'HIGH' if vr > 1.3 else ('LOW' if vr < 0.7 else 'MID')

            deny = None
            if dd_pct > dd_limit:
                deny = 'DD_BREACH'
            elif consec_losses >= consec_pause and i < paused_until:
                deny = 'CONSEC_LOSS_PAUSE'
            elif regime == 'HIGH' and dd_pct > dd_limit * 0.5:
                deny = 'HIGH_VOL_CAUTION'

            if deny:
                denied_pnls.append(pnl)
                deny_reasons[deny] += 1
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
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

    wins = sum(1 for p in pnls if p > 0)
    gp = sum(p for p in pnls if p > 0)
    gl = sum(abs(p) for p in pnls if p <= 0)
    pf = gp / gl if gl > 0 else float('inf')
    cl = loss_clustering(pnls)
    denied_losses = sum(1 for p in denied_pnls if p < 0)

    return {
        'mode': 'SOAR',
        'trades': len(pnls),
        'denied': len(denied_pnls),
        'denied_losses_blocked': denied_losses,
        'denied_loss_ratio': round(denied_losses / len(denied_pnls) * 100, 1) if denied_pnls else 0,
        'denied_avg_pnl': round(np.mean(denied_pnls), 2) if denied_pnls else 0,
        'deny_reasons': deny_reasons,
        'pf': round(pf, 2),
        'win_rate': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'max_dd_pct': round(max_dd * 100, 2),
        'net_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'worst_5': [round(p, 2) for p in worst_n(pnls, 5)],
        'max_loss_streak': cl['max_loss_streak'],
        'avg_loss_streak': cl['avg_loss_streak'],
        'pnl_list': pnls,
    }


def main():
    t0 = time.time()
    print("=" * 66)
    print("  EXP-01: RAW vs SOAR (Baseline)")
    print("  Question: Does execution grammar reduce left-tail risk?")
    print("=" * 66)

    data_path = find_data()
    df = load_data(data_path)
    print(f"\n  Data: {os.path.basename(data_path)} ({len(df)} bars)")

    signals = generate_signals(df)
    records = make_records(df)
    print(f"  Signals: {len(signals)} candidates")
    wins = sum(1 for s in signals if s['pnl_ticks'] > 0)
    print(f"  Raw pool WR: {wins}/{len(signals)} = {wins/len(signals)*100:.1f}%")

    raw = run_raw_detailed(signals)
    soar = run_soar_detailed(signals, records)

    print(f"\n  {'Metric':<25s} {'RAW':>12s} {'SOAR':>12s} {'Delta':>12s}")
    print(f"  {'-'*61}")
    rows = [
        ('Trades', 'trades', 'd'),
        ('Profit Factor', 'pf', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Max DD %', 'max_dd_pct', '.2f'),
        ('Net PnL $', 'net_pnl', '.2f'),
        ('Avg PnL/Trade $', 'avg_pnl', '.2f'),
        ('Max Loss Streak', 'max_loss_streak', 'd'),
        ('Avg Loss Streak', 'avg_loss_streak', '.2f'),
    ]
    for label, key, fmt in rows:
        rv, sv = raw[key], soar[key]
        d = sv - rv
        sign = '+' if d > 0 else ''
        print(f"  {label:<25s} {rv:>12{fmt}} {sv:>12{fmt}} {sign}{d:>11{fmt}}")

    print(f"\n  Worst 5 Trades (PnL $):")
    print(f"    RAW:  {raw['worst_5']}")
    print(f"    SOAR: {soar['worst_5']}")

    print(f"\n  Blocked Trades:")
    print(f"    Total denied: {soar['denied']}")
    print(f"    Denied were losses: {soar['denied_losses_blocked']} ({soar['denied_loss_ratio']}%)")
    print(f"    Avg denied PnL: ${soar['denied_avg_pnl']}")
    print(f"    Deny reasons: {soar['deny_reasons']}")

    dd_reduced = soar['max_dd_pct'] < raw['max_dd_pct']
    streak_reduced = soar['max_loss_streak'] <= raw['max_loss_streak']

    print(f"\n  {'='*61}")
    print(f"  VERDICT")
    print(f"  {'='*61}")
    if dd_reduced:
        pct = (1 - soar['max_dd_pct'] / raw['max_dd_pct']) * 100
        print(f"  [PASS] DD reduced {raw['max_dd_pct']:.2f}% -> {soar['max_dd_pct']:.2f}% (-{pct:.0f}%)")
    else:
        print(f"  [FAIL] DD not reduced: {raw['max_dd_pct']:.2f}% -> {soar['max_dd_pct']:.2f}%")
    if streak_reduced:
        print(f"  [PASS] Loss streak: {raw['max_loss_streak']} -> {soar['max_loss_streak']}")
    else:
        print(f"  [WARN] Loss streak increased: {raw['max_loss_streak']} -> {soar['max_loss_streak']}")

    status = 'PASS' if dd_reduced else 'FAIL'
    print(f"\n  EXP-01 STATUS: {status}")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'experiment': 'EXP-01-RAW-vs-SOAR',
        'timestamp': datetime.now().isoformat(),
        'data': os.path.basename(data_path),
        'bars': len(df),
        'signals': len(signals),
        'raw': {k: v for k, v in raw.items() if k != 'pnl_list'},
        'soar': {k: v for k, v in soar.items() if k != 'pnl_list'},
        'dd_reduced': bool(dd_reduced),
        'streak_reduced': bool(streak_reduced),
        'status': status,
    }
    path = os.path.join(EVIDENCE_DIR, 'exp_01_evidence.json')
    with open(path, 'w') as f:
        json.dump(evidence, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer)) else float(o) if isinstance(o, np.floating) else None)
    print(f"  Evidence: {path}")


if __name__ == '__main__':
    main()
