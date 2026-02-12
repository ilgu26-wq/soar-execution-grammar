"""
MOTION WATCHDOG v0.1 — Post-Entry Movement Quality Observer
=============================================================
Observes "did the price actually move after entry?" as a structural
quality metric. This is NOT about profit — it's about whether the
market responded to the signal direction at all.

Motion Tags:
  HEALTHY      — MFE >= threshold within k bars, no fast adverse
  NO_FOLLOW    — MFE too small (price didn't move in signal direction)
  FAST_ADVERSE — MAE grew too fast (immediate reversal)
  LOW_FORCE    — Force engine dir_consistency was too low at entry
  STALL        — Price stayed flat (neither MFE nor MAE significant)

Rules:
  - Observation only (EXP-13). No weight changes.
  - Gate is LOCKED. Watchdog does NOT deny trades.
  - Tags are recorded per (alpha, condition, regime) for map building.
  - EXP-14 adds motion-based weight penalties via AlphaMemory.update_motion_weights().
"""

import numpy as np

MOTION_VERSION = "v0.1.0"

MOTION_HEALTHY = "HEALTHY"
MOTION_NO_FOLLOW = "NO_FOLLOW"
MOTION_FAST_ADVERSE = "FAST_ADVERSE"
MOTION_LOW_FORCE = "LOW_FORCE"
MOTION_STALL = "STALL"

ALL_MOTION_TAGS = [
    MOTION_HEALTHY, MOTION_NO_FOLLOW, MOTION_FAST_ADVERSE,
    MOTION_LOW_FORCE, MOTION_STALL,
]

MFE_WINDOW = 5
MAE_WINDOW = 3
MFE_MIN_TICKS = 1.0
MAE_FAST_TICKS = 4.0
STALL_THRESHOLD_TICKS = 0.5
DIR_CONSISTENCY_FLOOR = 0.45


def compute_mfe_mae(df, bar_idx, direction, tick_size=0.25, window=10):
    """
    Compute Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    in ticks for a given entry bar and direction.
    """
    n = len(df)
    close = df['close'].values
    entry_price = close[bar_idx]

    end_idx = min(bar_idx + window + 1, n)
    if end_idx <= bar_idx + 1:
        return 0.0, 0.0, [], []

    future_prices = close[bar_idx + 1:end_idx]
    moves_ticks = (future_prices - entry_price) * direction / tick_size

    mfe_cumulative = []
    mae_cumulative = []
    running_mfe = 0.0
    running_mae = 0.0
    for m in moves_ticks:
        if m > running_mfe:
            running_mfe = m
        if m < running_mae:
            running_mae = m
        mfe_cumulative.append(round(running_mfe, 2))
        mae_cumulative.append(round(running_mae, 2))

    mfe = running_mfe
    mae = abs(running_mae)

    return round(mfe, 2), round(mae, 2), mfe_cumulative, mae_cumulative


def classify_motion(mfe, mae, mfe_at_k, mae_at_k, force_state=None):
    """
    Classify the post-entry movement quality.

    Priority order (first match wins):
      1. LOW_FORCE — structural weakness at entry (dir_consistency too low)
      2. FAST_ADVERSE — MAE grew too fast within first k bars
      3. NO_FOLLOW — MFE too small within first k bars
      4. STALL — neither MFE nor MAE significant
      5. HEALTHY — everything else
    """
    if force_state is not None:
        if force_state.dir_consistency < DIR_CONSISTENCY_FLOOR:
            return MOTION_LOW_FORCE

    if mae_at_k >= MAE_FAST_TICKS:
        return MOTION_FAST_ADVERSE

    if mfe_at_k < MFE_MIN_TICKS:
        if mae_at_k < STALL_THRESHOLD_TICKS:
            return MOTION_STALL
        return MOTION_NO_FOLLOW

    return MOTION_HEALTHY


def analyze_trade_motion(df, bar_idx, direction, force_state=None, tick_size=0.25):
    """
    Full motion analysis for a single trade.
    Returns motion_tag and detailed metrics.
    """
    mfe, mae, mfe_cum, mae_cum = compute_mfe_mae(
        df, bar_idx, direction, tick_size, window=max(MFE_WINDOW, MAE_WINDOW) + 2)

    mfe_at_k = mfe_cum[MFE_WINDOW - 1] if len(mfe_cum) >= MFE_WINDOW else mfe
    mae_at_k = abs(mae_cum[MAE_WINDOW - 1]) if len(mae_cum) >= MAE_WINDOW else mae

    motion_tag = classify_motion(mfe, mae, mfe_at_k, mae_at_k, force_state)

    return {
        'motion_tag': motion_tag,
        'mfe': mfe,
        'mae': mae,
        'mfe_at_5': round(mfe_at_k, 2),
        'mae_at_3': round(mae_at_k, 2),
        'dir_consistency': round(force_state.dir_consistency, 3) if force_state else None,
    }
