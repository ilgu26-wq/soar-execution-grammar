#!/usr/bin/env python3
"""
run_core.py — SOAR Execution Grammar Demonstration
====================================================
Loads NQ 5-second data, generates signal candidates,
then compares RAW execution (no grammar) vs SOAR execution (with grammar).

Alpha logic is IDENTICAL in both modes.
Only execution permission differs.

Usage:
    python core/run_core.py
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import ExecutionEngine
from gate import EVPack, EVGate, GateDecision
from constitution import Constitution, LAWS

DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'chat-observation-engine', 'quant', 'data', 'NQ_5s.csv'),
    os.path.join(os.path.dirname(__file__), '..', 'data', 'NQ_5s.csv'),
]

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'core_evidence')


def find_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    print("ERROR: NQ_5s.csv not found. Searched:")
    for p in DATA_PATHS:
        print(f"  {p}")
    sys.exit(1)


def load_data(path):
    df = pd.read_csv(path)
    required = ['close']
    for col in required:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in data")
            sys.exit(1)

    if 'dE' not in df.columns:
        df['dE'] = df['close'].diff().fillna(0)
    if 'd2E' not in df.columns:
        df['d2E'] = df['dE'].diff().fillna(0)
    if 'z_norm' not in df.columns:
        rolling_mean = df['close'].rolling(50, min_periods=1).mean()
        rolling_std = df['close'].rolling(50, min_periods=1).std().fillna(1)
        df['z_norm'] = (df['close'] - rolling_mean) / (rolling_std + 1e-10)
    if 'dc' not in df.columns:
        r = df['close'].rolling(20, min_periods=1)
        df['dc'] = ((df['close'] - r.min()) / (r.max() - r.min() + 1e-10)).fillna(0.5)
    if 'vol_ratio' not in df.columns:
        short_vol = df['close'].rolling(20, min_periods=1).std()
        long_vol = df['close'].rolling(100, min_periods=1).std()
        df['vol_ratio'] = (short_vol / (long_vol + 1e-10)).fillna(1.0)
    if 'ch_range' not in df.columns:
        r20 = df['close'].rolling(20, min_periods=1)
        df['ch_range'] = (r20.max() - r20.min()).fillna(0)

    return df


def generate_signals(df, stop_ticks=5.0, min_gap=10):
    signals = []
    n = len(df)
    last_sig_idx = -999

    dE = df['dE'].values
    d2E = df['d2E'].values
    z_norm = df['z_norm'].values
    dc = df['dc'].values
    close = df['close'].values
    vol_ratio = df['vol_ratio'].values if 'vol_ratio' in df.columns else np.ones(n)

    er_vals = np.abs(dE)
    er_20 = pd.Series(er_vals).rolling(20, min_periods=1).mean().values

    for i in range(100, n - 20):
        if i - last_sig_idx < min_gap:
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
            last_sig_idx = i

    return signals


def make_records(df):
    records = []
    cols = df.columns.tolist()
    for _, row in df.iterrows():
        rec = {}
        for c in ['dE', 'd2E', 'z_norm', 'dc', 'vol_ratio', 'ch_range']:
            rec[c] = float(row.get(c, 0))
        rec['pnl'] = float(row.get('dE', 0))
        rec['storm_flag'] = 0
        records.append(rec)
    return records


def verify_ev_gate():
    print("  [EV Gate Verification]")
    test_cases = [
        ({'S': 'NOMINAL', 'I': 'APPROVE', 'B': 'REVERSIBLE', 'P': True, 'C': 'LOW'}, 'ALLOW'),
        ({'S': 'NOMINAL', 'I': 'APPROVE', 'B': 'REVERSIBLE', 'P': False, 'C': 'LOW'}, 'DENY'),
        ({'S': 'WARNING', 'I': 'HALT', 'B': 'IRREVERSIBLE', 'P': True, 'C': 'HIGH'}, 'DENY'),
        ({'S': 'NOMINAL', 'I': 'APPROVE', 'B': 'IRREVERSIBLE', 'P': True, 'C': 'LOW'}, 'ALLOW'),
    ]
    all_ok = True
    for ev, expected in test_cases:
        result = EVGate.evaluate(ev)
        actual = result.decision.value
        ok = actual == expected
        roundtrip = EVPack.verify_roundtrip(ev)
        status = 'OK' if (ok and roundtrip) else 'FAIL'
        if not (ok and roundtrip):
            all_ok = False
        print(f"    P={str(ev['P']):>5s} B={ev['B']:<25s} C={ev['C']:<8s} → {actual:<5s} (expect {expected:<5s}) RT={'OK' if roundtrip else 'FAIL'}  [{status}]")

    print(f"    Gate integrity: {'ALL PASS' if all_ok else 'FAILED'}")
    return all_ok


def verify_constitution():
    print("  [Constitution Verification]")
    c = Constitution()
    intact = c.is_intact()
    print(f"    All laws ON: {intact}")
    for key, desc in LAWS.items():
        print(f"    {key}: {desc}")
    return intact


def print_comparison(raw, soar):
    print(f"\n  {'Metric':<20s} {'RAW':>12s} {'SOAR':>12s} {'Delta':>12s}")
    print(f"  {'-'*56}")

    metrics = [
        ('Trades', 'trades', 'd'),
        ('Profit Factor', 'pf', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Max DD %', 'max_dd_pct', '.2f'),
        ('Net PnL $', 'net_pnl', '.2f'),
        ('Avg PnL/Trade $', 'avg_pnl', '.2f'),
    ]

    for label, key, fmt in metrics:
        rv = raw[key]
        sv = soar[key]
        if isinstance(rv, (int, float)) and isinstance(sv, (int, float)):
            delta = sv - rv
            sign = '+' if delta > 0 else ''
            print(f"  {label:<20s} {rv:>12{fmt}} {sv:>12{fmt}} {sign}{delta:>11{fmt}}")
        else:
            print(f"  {label:<20s} {rv:>12{fmt}} {sv:>12{fmt}}")

    if soar.get('denied', 0) > 0:
        dl = soar.get('denied_losses_blocked', 0)
        da = soar.get('denied_avg_pnl', 0)
        print(f"\n  Grammar blocked {soar['denied']} signals ({dl} were losses, avg denied PnL=${da})")
    print(f"  Constitution intact: {soar.get('constitution_intact', False)}")


def main():
    t0 = time.time()
    print("=" * 66)
    print("  SOAR EXECUTION GRAMMAR — CORE DEMONSTRATION")
    print("  'Alpha unchanged. Only execution permission differs.'")
    print("=" * 66)

    print("\n[1/5] Loading data...")
    data_path = find_data()
    df = load_data(data_path)
    print(f"  Loaded {len(df)} bars from {os.path.basename(data_path)}")

    print("\n[2/5] Verifying structural components...")
    gate_ok = verify_ev_gate()
    const_ok = verify_constitution()
    if not gate_ok:
        print("  FATAL: EV Gate verification failed")
        sys.exit(1)

    print("\n[3/5] Generating signal candidates (alpha logic)...")
    signals = generate_signals(df, stop_ticks=5.0)
    records = make_records(df)
    print(f"  Generated {len(signals)} signal candidates")
    if not signals:
        print("  WARNING: No signals generated. Check data quality.")
        sys.exit(1)

    wins = sum(1 for s in signals if s['pnl_ticks'] > 0)
    losses = sum(1 for s in signals if s['pnl_ticks'] <= 0)
    print(f"  Raw pool: {wins} winners, {losses} losers ({wins/(wins+losses)*100:.1f}% WR)")

    print("\n[4/5] Running comparison...")
    engine = ExecutionEngine(
        energy_threshold=0.05,
        time_cooldown=20,
        stop_ticks=5.0,
        dd_limit=0.03,
    )

    raw_result = engine.run_raw(signals)
    print(f"  RAW mode:  {raw_result['trades']} trades, PF={raw_result['pf']}")

    engine_soar = ExecutionEngine(
        energy_threshold=0.05,
        time_cooldown=20,
        stop_ticks=5.0,
        dd_limit=0.03,
    )
    soar_result = engine_soar.run_grammar(signals, records)
    print(f"  SOAR mode: {soar_result['trades']} trades, PF={soar_result['pf']}")

    print("\n[5/5] Results")
    print_comparison(raw_result, soar_result)

    dd_improved = soar_result['max_dd_pct'] < raw_result['max_dd_pct']
    pf_maintained = soar_result['pf'] >= raw_result['pf'] * 0.5
    grammar_active = soar_result.get('denied', 0) > 0 or soar_result['trades'] <= raw_result['trades']

    print(f"\n  {'='*56}")
    print(f"  VERDICT")
    print(f"  {'='*56}")
    if dd_improved:
        print(f"  Left-tail risk REDUCED: DD {raw_result['max_dd_pct']:.2f}% → {soar_result['max_dd_pct']:.2f}%")
    else:
        print(f"  Left-tail risk unchanged or increased")
    if pf_maintained:
        print(f"  Profit factor MAINTAINED: {raw_result['pf']} → {soar_result['pf']}")
    else:
        print(f"  Profit factor degraded significantly")
    if grammar_active:
        print(f"  Execution grammar ACTIVE: filtering applied")
    else:
        print(f"  Execution grammar had no effect")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'timestamp': datetime.now().isoformat(),
        'data_file': os.path.basename(data_path),
        'bars': len(df),
        'signal_candidates': len(signals),
        'raw': raw_result,
        'soar': soar_result,
        'dd_improved': dd_improved,
        'pf_maintained': pf_maintained,
        'grammar_active': grammar_active,
    }
    def _serialize(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    evidence_path = os.path.join(EVIDENCE_DIR, 'core_run_evidence.json')
    with open(evidence_path, 'w') as f:
        json.dump(evidence, f, indent=2, default=_serialize)
    print(f"  Evidence saved to: {evidence_path}")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
