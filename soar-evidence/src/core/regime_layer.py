"""
REGIME LEARNING LAYER — sits ON TOP of v2 LOCKED
==================================================
v2 gate parameters are NEVER touched.
This layer only:
  1. Classifies market regime (TREND / CHOP / STORM / DEAD)
  2. Records regime-level performance (passive memory)
  3. Provides size_hint (0.8 ~ 1.0 ONLY, no amplification)

Rules:
  - v2 gate must PASS before size_hint applies
  - size_hint is a REDUCTION only (max -20%)
  - No stop/entry/denial changes
  - Regime memory needs n >= 50 before any hint activates

Architecture:
  observation_layer (classify) → passive_memory (record) → size_hint (reduce only)
"""

import json, os
from datetime import datetime
from collections import defaultdict

REGIME_LAYER_VERSION = "v0.1.0"

REGIME_TREND = "TREND"
REGIME_CHOP = "CHOP"
REGIME_STORM = "STORM"
REGIME_DEAD = "DEAD"
ALL_REGIMES = [REGIME_TREND, REGIME_CHOP, REGIME_STORM, REGIME_DEAD]

MIN_SAMPLES_FOR_HINT = 50
SIZE_HINT_FLOOR = 0.8
SIZE_HINT_CEIL = 1.0


def classify_regime(vol_ratio, signal_density, avg_bar_range, dE_accel):
    """
    Classify current market regime from bar-level features.

    Inputs (all computed from existing bar data, no new indicators):
      vol_ratio:       short_vol / long_vol (already in v2)
      signal_density:  signals per 100 bars in recent window
      avg_bar_range:   avg (high - low) over recent window
      dE_accel:        abs(d2E) mean — second derivative of price

    Output: regime string

    Logic:
      STORM:  vol_ratio > 1.5 AND dE_accel high → chaotic, dangerous
      TREND:  vol_ratio 0.8~1.5 AND signal_density moderate → directional
      CHOP:   vol_ratio 0.8~1.5 AND signal_density high, range tight → mean-reverting noise
      DEAD:   vol_ratio < 0.8 OR signal_density very low → no movement
    """
    if vol_ratio > 1.5 and dE_accel > 1.0:
        return REGIME_STORM
    if vol_ratio < 0.7 or signal_density < 0.5:
        return REGIME_DEAD
    if signal_density > 3.0 and avg_bar_range < 2.0:
        return REGIME_CHOP
    return REGIME_TREND


class RegimeMemory:
    """
    Passive performance memory per regime.
    Records only — no execution decisions.
    """
    def __init__(self):
        self.data = {r: {'wins': 0, 'losses': 0, 'total_pnl': 0.0,
                         'trades': 0, 'gross_profit': 0.0, 'gross_loss': 0.0}
                     for r in ALL_REGIMES}

    def record(self, regime, pnl, is_win):
        d = self.data[regime]
        d['trades'] += 1
        d['total_pnl'] += pnl
        if is_win:
            d['wins'] += 1
            d['gross_profit'] += pnl
        else:
            d['losses'] += 1
            d['gross_loss'] += abs(pnl)

    def get_stats(self, regime):
        d = self.data[regime]
        n = d['trades']
        if n == 0:
            return {'n': 0, 'WR': 0, 'EV': 0, 'PF': 0}
        wr = d['wins'] / n * 100
        ev = d['total_pnl'] / n
        pf = d['gross_profit'] / d['gross_loss'] if d['gross_loss'] > 0 else float('inf')
        return {'n': n, 'WR': round(wr, 1), 'EV': round(ev, 2), 'PF': round(pf, 2)}

    def get_size_hint(self, regime):
        """
        Returns size multiplier: 0.8 ~ 1.0
        Only active if regime has >= MIN_SAMPLES_FOR_HINT trades.
        """
        stats = self.get_stats(regime)
        if stats['n'] < MIN_SAMPLES_FOR_HINT:
            return SIZE_HINT_CEIL

        if stats['EV'] <= 0:
            return SIZE_HINT_FLOOR

        if stats['PF'] >= 1.5:
            return 1.0
        elif stats['PF'] >= 1.2:
            return 0.9
        elif stats['PF'] >= 1.0:
            return 0.85
        else:
            return SIZE_HINT_FLOOR

    def summary_table(self):
        rows = []
        for r in ALL_REGIMES:
            s = self.get_stats(r)
            hint = self.get_size_hint(r)
            active = "YES" if s['n'] >= MIN_SAMPLES_FOR_HINT else "NO"
            rows.append({
                'regime': r,
                'n': s['n'],
                'WR': s['WR'],
                'EV': s['EV'],
                'PF': s['PF'],
                'size_hint': hint,
                'hint_active': active,
            })
        return rows

    def to_dict(self):
        result = {}
        for r in ALL_REGIMES:
            s = self.get_stats(r)
            s['size_hint'] = self.get_size_hint(r)
            s['hint_active'] = s['n'] >= MIN_SAMPLES_FOR_HINT
            result[r] = s
        return result


class RegimeLogger:
    """
    Trade-level regime log.
    Every trade gets a regime tag for post-analysis.
    """
    def __init__(self):
        self.log = []

    def append(self, timestamp, regime, pnl, dd_at_entry, denied_reason=None):
        self.log.append({
            'time': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'regime': regime,
            'pnl': round(pnl, 2),
            'dd_at_entry': round(dd_at_entry, 4),
            'denied_reason': denied_reason,
        })

    def to_list(self):
        return self.log
