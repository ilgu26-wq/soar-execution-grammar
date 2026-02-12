#!/usr/bin/env python3
"""
EXP-07: Prop Firm Deployment Simulation
=========================================
Deployment simulation, not research.

Structure:
  - v2 standalone (LOCKED parameters)
  - Prop firm rules applied (daily DD / trailing DD / position limit)
  - Daily reset
  - Fixed position size

Verification points:
  1. Daily DD breach = 0
  2. Forced liquidation = 0
  3. Trade denial reason distribution normal
  4. Prop firm evaluation PASS/FAIL determination

Execution:
  python experiments/exp_07_prop_deployment_sim.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.v2_locked import (
    SOAR_CORE_V2_LOCKED, LOCK_VERSION,
    DD_THRESHOLD, CONSEC_LOSS_PAUSE, CONSEC_LOSS_COOLDOWN_BARS,
    VOL_GATE_HIGH, VOL_GATE_LOW, HIGH_VOL_DD_MULTIPLIER, WARMUP_BARS,
    STOP_TICKS, MIN_SIGNAL_GAP, ER_FLOOR, Z_NORM_THRESHOLD, ER_MULTIPLIER,
    LOOKBACK_BARS, PropProfile, PROP_PROFILES, DenyReason, validate_lock,
)

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'data', 'NQ_5s.csv'),
    os.path.join(os.path.dirname(__file__), '..', 'chat-observation-engine', 'quant', 'data', 'NQ_5s.csv'),
]
EPS = 1e-10

BARS_PER_DAY = 4680


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
                'day': i // BARS_PER_DAY,
            })
            last_idx = i
    return signals


def simulate_prop(signals, df, profile):
    """
    Full prop deployment simulation with daily resets.
    """
    n = len(df)
    tv = profile.tick_value
    contracts = profile.max_position

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

    account_equity = float(profile.account_size)
    account_peak = account_equity
    trailing_high = account_equity

    daily_start = account_equity
    daily_pnl = 0.0
    current_day = 0

    consec_losses = 0
    paused_until = -1

    trades = []
    denied_log = []
    daily_summaries = []
    forced_liquidations = 0
    daily_dd_breaches = 0

    for i in range(n):
        day = i // BARS_PER_DAY
        if day != current_day:
            daily_summaries.append({
                'day': current_day,
                'pnl': round(daily_pnl, 2),
                'trades': sum(1 for t in trades if t['day'] == current_day),
                'denied': sum(1 for d in denied_log if d['day'] == current_day),
                'equity': round(account_equity, 2),
                'dd_from_start': round((daily_start - account_equity) / daily_start * 100, 2)
                                 if daily_start > 0 else 0,
            })
            daily_start = account_equity
            daily_pnl = 0.0
            current_day = day
            consec_losses = 0
            paused_until = -1

        if i < WARMUP_BARS or i not in sig_map:
            continue

        for sig in sig_map[i]:
            pnl_per_contract = sig['pnl_ticks'] * tv
            pnl_total = pnl_per_contract * contracts

            deny_reasons = []

            dd_from_peak = (account_peak - account_equity) / account_peak if account_peak > 0 else 0
            if dd_from_peak > DD_THRESHOLD:
                deny_reasons.append(DenyReason.DD_BREACH)

            if consec_losses >= CONSEC_LOSS_PAUSE and i < paused_until:
                deny_reasons.append(DenyReason.CONSEC_LOSS_PAUSE)

            vr = vol_short[i] / (vol_long[i] + EPS)
            if vr > VOL_GATE_HIGH and dd_from_peak > DD_THRESHOLD * HIGH_VOL_DD_MULTIPLIER:
                deny_reasons.append(DenyReason.HIGH_VOL_CAUTION)

            daily_dd = (daily_start - account_equity) / daily_start if daily_start > 0 else 0
            if daily_dd * 100 >= profile.daily_dd_pct:
                deny_reasons.append(DenyReason.DAILY_DD_PROP)

            trailing_dd = (trailing_high - account_equity) / trailing_high if trailing_high > 0 else 0
            if trailing_dd * 100 >= profile.trailing_dd_pct:
                deny_reasons.append(DenyReason.TRAILING_DD_PROP)

            if deny_reasons:
                denied_log.append({
                    'bar': i,
                    'day': day,
                    'reasons': deny_reasons,
                    'pnl_would_be': round(pnl_total, 2),
                })
                if DenyReason.DAILY_DD_PROP in deny_reasons:
                    daily_dd_breaches += 1
                if DenyReason.TRAILING_DD_PROP in deny_reasons:
                    forced_liquidations += 1
            else:
                account_equity += pnl_total
                daily_pnl += pnl_total

                if account_equity > account_peak:
                    account_peak = account_equity
                if account_equity > trailing_high:
                    trailing_high = account_equity

                is_win = pnl_total > 0
                if is_win:
                    consec_losses = 0
                else:
                    consec_losses += 1
                    if consec_losses >= CONSEC_LOSS_PAUSE:
                        paused_until = i + CONSEC_LOSS_COOLDOWN_BARS

                trades.append({
                    'bar': i,
                    'day': day,
                    'pnl': round(pnl_total, 2),
                    'equity': round(account_equity, 2),
                    'contracts': contracts,
                })

    daily_summaries.append({
        'day': current_day,
        'pnl': round(daily_pnl, 2),
        'trades': sum(1 for t in trades if t['day'] == current_day),
        'denied': sum(1 for d in denied_log if d['day'] == current_day),
        'equity': round(account_equity, 2),
        'dd_from_start': round((daily_start - account_equity) / daily_start * 100, 2)
                         if daily_start > 0 else 0,
    })

    trade_pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in trade_pnls if p > 0)
    gp = sum(p for p in trade_pnls if p > 0)
    gl = sum(abs(p) for p in trade_pnls if p <= 0)
    pf = gp / gl if gl > 0 else float('inf')

    max_dd_pct = 0.0
    eq = float(profile.account_size)
    pk = eq
    for p in trade_pnls:
        eq += p
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk if pk > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

    streaks = []
    cur = 0
    for p in trade_pnls:
        if p < 0:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    max_streak = max(streaks) if streaks else 0

    deny_reason_counts = Counter()
    for d in denied_log:
        for r in d['reasons']:
            deny_reason_counts[r] += 1

    profitable_days = sum(1 for d in daily_summaries if d['pnl'] > 0)
    losing_days = sum(1 for d in daily_summaries if d['pnl'] < 0)
    flat_days = sum(1 for d in daily_summaries if d['pnl'] == 0)

    return {
        'profile': profile.name,
        'account_size': profile.account_size,
        'contracts': contracts,
        'tick_value': tv,
        'total_trades': len(trades),
        'total_denied': len(denied_log),
        'pf': round(pf, 2),
        'win_rate': round(wins / len(trades) * 100, 1) if trades else 0,
        'max_dd_pct': round(max_dd_pct * 100, 2),
        'net_pnl': round(sum(trade_pnls), 2),
        'avg_pnl': round(np.mean(trade_pnls), 2) if trade_pnls else 0,
        'max_loss_streak': max_streak,
        'final_equity': round(account_equity, 2),
        'daily_dd_breaches': daily_dd_breaches,
        'forced_liquidations': forced_liquidations,
        'deny_reasons': dict(deny_reason_counts),
        'profitable_days': profitable_days,
        'losing_days': losing_days,
        'flat_days': flat_days,
        'total_days': len(daily_summaries),
        'daily_summaries': daily_summaries,
    }


def main():
    t0 = time.time()
    validate_lock()

    print("=" * 70)
    print(f"  EXP-07: Prop Firm Deployment Simulation")
    print(f"  SOAR CORE {LOCK_VERSION} — LOCKED PARAMETERS")
    print(f"  'This is not research. This is deployment.'")
    print("=" * 70)

    data_path = find_data()
    df = load_data(data_path)
    signals = generate_signals(df)
    print(f"\n  Data: {os.path.basename(data_path)} ({len(df)} bars)")
    print(f"  Signals: {len(signals)} candidates")
    print(f"  Simulated days: ~{len(df) // BARS_PER_DAY + 1}")

    profiles_to_test = ['APEX_50K', 'TOPSTEP_50K', 'MNQ_MICRO']
    all_results = {}

    for pkey in profiles_to_test:
        profile = PROP_PROFILES[pkey]
        print(f"\n  {'─'*60}")
        print(f"  Profile: {profile.name}")
        print(f"  Account: ${profile.account_size:,} | "
              f"Daily DD: {profile.daily_dd_pct}% (${profile.daily_dd_dollars:,.0f}) | "
              f"Trailing DD: {profile.trailing_dd_pct}% (${profile.trailing_dd_dollars:,.0f})")
        print(f"  Position: {profile.max_position} contracts × ${profile.tick_value}/tick")
        print(f"  {'─'*60}")

        result = simulate_prop(signals, df, profile)
        all_results[pkey] = result

        print(f"\n  Trades: {result['total_trades']} | Denied: {result['total_denied']}")
        print(f"  PF: {result['pf']} | WR: {result['win_rate']}%")
        print(f"  Max DD: {result['max_dd_pct']}%")
        print(f"  Net PnL: ${result['net_pnl']:,.2f}")
        print(f"  Final Equity: ${result['final_equity']:,.2f}")
        print(f"  Max Loss Streak: {result['max_loss_streak']}")

        print(f"\n  --- Prop Safety Checks ---")
        print(f"  Daily DD breaches:    {result['daily_dd_breaches']}", end="")
        print(f"  {'[PASS]' if result['daily_dd_breaches'] == 0 else '[FAIL]'}")
        print(f"  Forced liquidations:  {result['forced_liquidations']}", end="")
        print(f"  {'[PASS]' if result['forced_liquidations'] == 0 else '[FAIL]'}")

        print(f"\n  --- Denial Distribution ---")
        for reason in DenyReason.ALL:
            count = result['deny_reasons'].get(reason, 0)
            if count > 0:
                print(f"    {reason:<25s}: {count}")

        print(f"\n  --- Day Stats ---")
        print(f"  Profitable days: {result['profitable_days']}")
        print(f"  Losing days:     {result['losing_days']}")
        print(f"  Flat days:       {result['flat_days']}")

        is_pass = (
            result['daily_dd_breaches'] == 0 and
            result['forced_liquidations'] == 0 and
            result['pf'] >= 1.0 and
            result['net_pnl'] > 0
        )
        result['prop_pass'] = is_pass
        verdict = "PROP_PASS" if is_pass else "PROP_FAIL"
        print(f"\n  >>> {profile.name}: {verdict} <<<")

    print(f"\n\n  {'='*60}")
    print(f"  FINAL DEPLOYMENT VERDICT")
    print(f"  {'='*60}")

    all_pass = all(r['prop_pass'] for r in all_results.values())
    for pkey, r in all_results.items():
        status = "PASS" if r['prop_pass'] else "FAIL"
        print(f"  {r['profile']:<25s}: {status} | PF={r['pf']} | "
              f"DD={r['max_dd_pct']}% | PnL=${r['net_pnl']:,.0f}")

    if all_pass:
        print(f"\n  ALL PROFILES PASSED.")
        print(f"  SOAR CORE v2 is DEPLOYMENT-READY.")
        print(f"  → Ready for production deployment")
    else:
        failed = [r['profile'] for r in all_results.values() if not r['prop_pass']]
        print(f"\n  FAILED: {', '.join(failed)}")
        print(f"  → Parameter readjustment needed")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence = {
        'experiment': 'EXP-07-Prop-Deployment-Sim',
        'timestamp': datetime.now().isoformat(),
        'lock_version': LOCK_VERSION,
        'data': os.path.basename(data_path),
        'bars': len(df),
        'signals': len(signals),
        'locked_params': {
            'DD_THRESHOLD': DD_THRESHOLD,
            'CONSEC_LOSS_PAUSE': CONSEC_LOSS_PAUSE,
            'VOL_GATE_HIGH': VOL_GATE_HIGH,
            'STOP_TICKS': STOP_TICKS,
            'WARMUP_BARS': WARMUP_BARS,
        },
        'results': {},
        'all_pass': all_pass,
    }
    for pkey, r in all_results.items():
        safe_r = {k: v for k, v in r.items() if k != 'daily_summaries'}
        safe_r['daily_summary_count'] = len(r['daily_summaries'])
        safe_r['daily_pnl_list'] = [d['pnl'] for d in r['daily_summaries']]
        evidence['results'][pkey] = safe_r

    path = os.path.join(EVIDENCE_DIR, 'exp_07_evidence.json')
    with open(path, 'w') as f:
        json.dump(evidence, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer))
                  else float(o) if isinstance(o, np.floating) else None)
    print(f"  Evidence: {path}")


if __name__ == '__main__':
    main()
