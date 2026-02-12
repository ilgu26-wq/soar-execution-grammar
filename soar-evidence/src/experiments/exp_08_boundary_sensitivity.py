#!/usr/bin/env python3
"""
EXP-08: Boundary Sensitivity Sweep (SOAR vs Anti-SOAR)
========================================================
v2 LOCK maintained. No conclusion changes. Boundary sensitivity only.

Structure:
  - Compute slack_score for all trades denied by v2
  - slack_score = average slack to each gate condition
  - Analyze actual PnL distribution by slack segment
  - Measure ratio of 'conservatively blocked profits'

Key question:
  Q1. Were blocked trades with slack >= 0.7 profitable?
  Q2. At which slack segment does the midpoint appear?

Anti-SOAR role:
  - No execution authority
  - Only records 'why was it rejected, and how close was it'
  - No v2 parameter modification suggestions

Execution:
  python experiments/exp_08_boundary_sensitivity.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.v2_locked import (
    SOAR_CORE_V2_LOCKED, LOCK_VERSION,
    DD_THRESHOLD, CONSEC_LOSS_PAUSE, CONSEC_LOSS_COOLDOWN_BARS,
    VOL_GATE_HIGH, HIGH_VOL_DD_MULTIPLIER, WARMUP_BARS,
    STOP_TICKS, MIN_SIGNAL_GAP, ER_FLOOR, Z_NORM_THRESHOLD, ER_MULTIPLIER,
    LOOKBACK_BARS, DenyReason, validate_lock,
)

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'data', 'NQ_5s.csv'),
    os.path.join(os.path.dirname(__file__), '..', 'chat-observation-engine', 'quant', 'data', 'NQ_5s.csv'),
]
EPS = 1e-10

SLACK_BUCKETS = [
    (0.0, 0.3, 'HARD_DENY'),
    (0.3, 0.5, 'MODERATE_DENY'),
    (0.5, 0.7, 'SOFT_DENY'),
    (0.7, 0.85, 'NEAR_ALLOW'),
    (0.85, 1.01, 'ALMOST_ALLOW'),
]


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


def generate_signals(df):
    signals = []
    n = len(df)
    last_idx = -999
    dE = df['dE'].values
    z_norm = df['z_norm'].values
    close = df['close'].values
    er_vals = np.abs(dE)
    er_20 = pd.Series(er_vals).rolling(20, min_periods=1).mean().values
    for i in range(100, n - LOOKBACK_BARS):
        if i - last_idx < MIN_SIGNAL_GAP:
            continue
        er = er_20[i]
        if er < ER_FLOOR:
            continue
        if abs(z_norm[i]) > Z_NORM_THRESHOLD and abs(dE[i]) > er * ER_MULTIPLIER:
            direction = 1 if dE[i] > 0 else -1
            pnl_ticks = 0.0
            for j in range(1, min(LOOKBACK_BARS, n - i)):
                move = (close[i + j] - close[i]) * direction / 0.25
                if move <= -STOP_TICKS:
                    pnl_ticks = -STOP_TICKS
                    break
                if move >= STOP_TICKS * 2:
                    pnl_ticks = STOP_TICKS * 2
                    break
                pnl_ticks = move
            signals.append({
                'bar_idx': i,
                'direction': direction,
                'pnl_ticks': round(pnl_ticks, 2),
            })
            last_idx = i
    return signals


def compute_slack(dd_pct, consec_losses, paused_until, bar_idx, vol_ratio):
    """
    Anti-SOAR: compute slack_score for each gate condition.
    slack = how far from triggering each deny rule.
    0.0 = rule triggered (hard deny)
    1.0 = maximum distance from any rule
    """
    slacks = []

    dd_slack = max(0.0, 1.0 - dd_pct / DD_THRESHOLD)
    slacks.append(dd_slack)

    if consec_losses >= CONSEC_LOSS_PAUSE and bar_idx < paused_until:
        streak_slack = 0.0
    else:
        streak_slack = max(0.0, 1.0 - consec_losses / CONSEC_LOSS_PAUSE)
    slacks.append(streak_slack)

    if vol_ratio > VOL_GATE_HIGH:
        vol_dd_threshold = DD_THRESHOLD * HIGH_VOL_DD_MULTIPLIER
        vol_slack = max(0.0, 1.0 - dd_pct / vol_dd_threshold) if dd_pct < vol_dd_threshold else 0.0
    else:
        vol_slack = max(0.0, min(1.0, (VOL_GATE_HIGH - vol_ratio) / VOL_GATE_HIGH))
    slacks.append(vol_slack)

    return float(np.mean(slacks)), {
        'dd_slack': round(dd_slack, 3),
        'streak_slack': round(streak_slack, 3),
        'vol_slack': round(vol_slack, 3),
    }


def run_sensitivity(signals, df, tick_value=5.0):
    """
    Run v2 engine with full slack logging for every signal.
    """
    n = len(df)
    dE_vals = df['dE'].values.astype(float)
    vol_short = np.zeros(n)
    vol_long = np.zeros(n)
    for i in range(n):
        lo = max(0, i - 20)
        lo2 = max(0, i - 100)
        vol_short[i] = np.std(dE_vals[lo:i+1]) if i >= 1 else 0
        vol_long[i] = np.std(dE_vals[lo2:i+1]) if i >= 1 else 0

    sig_map = {}
    for sig in signals:
        sig_map.setdefault(sig['bar_idx'], []).append(sig)

    equity = 100_000.0
    peak = equity
    consec_losses = 0
    paused_until = -1

    allowed_log = []
    denied_log = []

    for i in range(n):
        if i < WARMUP_BARS or i not in sig_map:
            continue

        for sig in sig_map[i]:
            pnl_dollar = sig['pnl_ticks'] * tick_value
            dd_pct = (peak - equity) / peak if peak > 0 else 0
            vr = vol_short[i] / (vol_long[i] + EPS)

            slack_score, slack_detail = compute_slack(
                dd_pct, consec_losses, paused_until, i, vr
            )

            deny_reasons = []
            if dd_pct > DD_THRESHOLD:
                deny_reasons.append(DenyReason.DD_BREACH)
            if consec_losses >= CONSEC_LOSS_PAUSE and i < paused_until:
                deny_reasons.append(DenyReason.CONSEC_LOSS_PAUSE)
            regime = 'HIGH' if vr > VOL_GATE_HIGH else 'MID'
            if regime == 'HIGH' and dd_pct > DD_THRESHOLD * HIGH_VOL_DD_MULTIPLIER:
                deny_reasons.append(DenyReason.HIGH_VOL_CAUTION)

            entry = {
                'bar': i,
                'pnl': round(pnl_dollar, 2),
                'pnl_ticks': sig['pnl_ticks'],
                'is_win': pnl_dollar > 0,
                'slack_score': round(slack_score, 3),
                'slack_detail': slack_detail,
                'dd_pct': round(dd_pct * 100, 3),
                'consec_losses': consec_losses,
                'vol_ratio': round(vr, 3),
            }

            if deny_reasons:
                entry['deny_reasons'] = deny_reasons
                denied_log.append(entry)
            else:
                allowed_log.append(entry)
                equity += pnl_dollar
                if pnl_dollar > 0:
                    consec_losses = 0
                else:
                    consec_losses += 1
                    if consec_losses >= CONSEC_LOSS_PAUSE:
                        paused_until = i + CONSEC_LOSS_COOLDOWN_BARS
                if equity > peak:
                    peak = equity

    return allowed_log, denied_log


def analyze_buckets(denied_log):
    """Analyze denied trades by slack bucket."""
    bucket_stats = {}
    for lo, hi, label in SLACK_BUCKETS:
        trades = [d for d in denied_log if lo <= d['slack_score'] < hi]
        if not trades:
            bucket_stats[label] = {
                'range': f'{lo:.1f}~{hi:.1f}',
                'count': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'pct_of_denied': 0,
            }
            continue

        wins = sum(1 for t in trades if t['is_win'])
        pnls = [t['pnl'] for t in trades]
        bucket_stats[label] = {
            'range': f'{lo:.1f}~{hi:.1f}',
            'count': len(trades),
            'win_rate': round(wins / len(trades) * 100, 1),
            'avg_pnl': round(np.mean(pnls), 2),
            'total_pnl': round(sum(pnls), 2),
            'pct_of_denied': round(len(trades) / len(denied_log) * 100, 1),
        }
    return bucket_stats


def main():
    t0 = time.time()
    validate_lock()

    print("=" * 70)
    print(f"  EXP-08: Boundary Sensitivity Sweep")
    print(f"  SOAR v2 = LOCKED. Anti-SOAR = Observer only.")
    print(f"  'Find the middle ground without moving the line.'")
    print("=" * 70)

    data_path = find_data()
    df = load_data(data_path)
    signals = generate_signals(df)
    print(f"\n  Data: {os.path.basename(data_path)} ({len(df)} bars)")
    print(f"  Signals: {len(signals)} candidates")

    allowed_log, denied_log = run_sensitivity(signals, df)
    print(f"\n  Allowed: {len(allowed_log)} trades")
    print(f"  Denied:  {len(denied_log)} trades")

    denied_slacks = [d['slack_score'] for d in denied_log]
    if denied_slacks:
        print(f"  Denied slack range: [{min(denied_slacks):.3f}, {max(denied_slacks):.3f}], "
              f"mean={np.mean(denied_slacks):.3f}")

    bucket_stats = analyze_buckets(denied_log)

    print(f"\n  {'='*70}")
    print(f"  SLACK BUCKET ANALYSIS (Denied Trades Only)")
    print(f"  {'='*70}")
    print(f"\n  {'Bucket':<18s} {'Range':>8s} {'Count':>7s} {'WR%':>7s} "
          f"{'AvgPnL':>9s} {'TotalPnL':>10s} {'% Denied':>9s}")
    print(f"  {'-'*68}")

    for lo, hi, label in SLACK_BUCKETS:
        b = bucket_stats[label]
        print(f"  {label:<18s} {b['range']:>8s} {b['count']:>7d} {b['win_rate']:>7.1f} "
              f"{b['avg_pnl']:>9.2f} {b['total_pnl']:>10.2f} {b['pct_of_denied']:>9.1f}")

    print(f"\n  {'='*70}")
    print(f"  ANTI-SOAR OBSERVATIONS")
    print(f"  {'='*70}")

    near_allow = [d for d in denied_log if d['slack_score'] >= 0.7]
    hard_deny = [d for d in denied_log if d['slack_score'] < 0.3]

    if near_allow:
        na_wins = sum(1 for t in near_allow if t['is_win'])
        na_pnl = sum(t['pnl'] for t in near_allow)
        print(f"\n  NEAR-ALLOW (slack >= 0.7): {len(near_allow)} trades")
        print(f"    Win rate: {na_wins/len(near_allow)*100:.1f}%")
        print(f"    Total PnL if allowed: ${na_pnl:,.2f}")
        print(f"    Avg PnL: ${np.mean([t['pnl'] for t in near_allow]):,.2f}")
        if na_pnl > 0:
            print(f"    >>> OBSERVATION: v2 is being conservative here — money left on table")
        else:
            print(f"    >>> OBSERVATION: v2 correctly denied — these would have lost money")
    else:
        print(f"\n  NEAR-ALLOW (slack >= 0.7): 0 trades — v2 never barely denied")

    if hard_deny:
        hd_wins = sum(1 for t in hard_deny if t['is_win'])
        hd_pnl = sum(t['pnl'] for t in hard_deny)
        print(f"\n  HARD-DENY (slack < 0.3): {len(hard_deny)} trades")
        print(f"    Win rate: {hd_wins/len(hard_deny)*100:.1f}%")
        print(f"    Total PnL if allowed: ${hd_pnl:,.2f}")
        print(f"    >>> {'CONFIRMED: v2 correctly blocked danger zone' if hd_pnl <= 0 else 'NOTE: some profitable trades in danger zone'}")

    allowed_pnls = [a['pnl'] for a in allowed_log]
    allowed_total = sum(allowed_pnls)
    denied_pnls = [d['pnl'] for d in denied_log]
    denied_total = sum(denied_pnls)

    print(f"\n  {'='*70}")
    print(f"  VALUE SUMMARY")
    print(f"  {'='*70}")
    print(f"  Allowed trades PnL:  ${allowed_total:>10,.2f}")
    print(f"  Denied trades PnL:   ${denied_total:>10,.2f} (would-be)")
    if denied_total > 0:
        print(f"  >>> v2 left ${denied_total:,.0f} on the table — review NEAR-ALLOW bucket")
    else:
        print(f"  >>> v2 denied ${abs(denied_total):,.0f} in losses — gate is adding value")

    mid_candidates = []
    for lo, hi, label in SLACK_BUCKETS:
        b = bucket_stats[label]
        if b['count'] > 0 and b['win_rate'] > 50 and b['avg_pnl'] > 0:
            mid_candidates.append(label)

    print(f"\n  {'='*70}")
    print(f"  MIDDLE GROUND CANDIDATES")
    print(f"  {'='*70}")
    if mid_candidates:
        print(f"  Buckets with WR > 50% and positive avg PnL:")
        for mc in mid_candidates:
            b = bucket_stats[mc]
            print(f"    {mc}: WR={b['win_rate']}%, AvgPnL=${b['avg_pnl']}, "
                  f"Count={b['count']}")
        print(f"\n  >>> These denied trades could be selectively re-allowed")
        print(f"  >>> in a future EXP-09 (non-destructive, shadow mode only)")
    else:
        print(f"  No bucket has WR > 50% with positive PnL.")
        print(f"  >>> v2 gate is OPTIMAL — no middle ground exists.")
        print(f"  >>> The current boundary IS the answer.")

    overall_status = 'OPTIMAL' if not mid_candidates else 'MIDDLE_FOUND'

    print(f"\n  {'='*70}")
    print(f"  EXP-08 STATUS: {overall_status}")
    print(f"  {'='*70}")
    if overall_status == 'OPTIMAL':
        print(f"  v2 gate is already at the optimal boundary.")
        print(f"  No further adjustment needed.")
    else:
        print(f"  Middle ground detected in {len(mid_candidates)} bucket(s).")
        print(f"  Next: EXP-09 shadow test on those buckets (v2 unchanged).")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'experiment': 'EXP-08-Boundary-Sensitivity-Sweep',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'data': os.path.basename(data_path),
        'bars': len(df),
        'signals': len(signals),
        'allowed_count': len(allowed_log),
        'denied_count': len(denied_log),
        'allowed_pnl': round(allowed_total, 2),
        'denied_pnl': round(denied_total, 2),
        'bucket_stats': bucket_stats,
        'near_allow_count': len(near_allow) if near_allow else 0,
        'near_allow_wr': round(na_wins / len(near_allow) * 100, 1) if near_allow else 0,
        'near_allow_pnl': round(na_pnl, 2) if near_allow else 0,
        'middle_candidates': mid_candidates,
        'status': overall_status,
    }
    path = os.path.join(EVIDENCE_DIR, 'exp_08_evidence.json')
    with open(path, 'w') as f:
        json.dump(evidence, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer))
                  else float(o) if isinstance(o, np.floating) else None)
    print(f"  Evidence: {path}")


if __name__ == '__main__':
    main()
