"""
FORCE ENGINE v2 — High-Resolution Market Physics
===================================================
Observation-only layer. Records force state per bar.
NEVER makes execution decisions.

Resolution levels:
  Level 0 (v1): vol_ratio, range, DD — scalars only
  Level 1 (v2): + gradient, curvature, directional consistency — vectors

All outputs are TRACES for post-analysis.
v2 gate reads NONE of this.
"""

import numpy as np

FORCE_ENGINE_VERSION = "v2.0.0"

EPS = 1e-10


class ForceState:
    """Immutable snapshot of force at a single bar."""
    __slots__ = [
        'bar_idx', 'force_magnitude', 'force_gradient', 'force_curvature',
        'dir_consistency', 'pressure_buy', 'pressure_sell', 'net_pressure',
        'momentum_3', 'momentum_10', 'acceleration',
    ]

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k, 0.0))

    def to_dict(self):
        return {k: round(getattr(self, k), 6) for k in self.__slots__}


class ForceEngine:
    """
    Computes high-resolution force traces from bar data.

    Input: bars DataFrame (must have dE, d2E, volume, delta, buy_vol, sell_vol, high, low, close)
    Output: list of ForceState per bar
    """

    def __init__(self):
        self.traces = []

    def compute_all(self, df):
        """Pre-compute force traces for all bars."""
        n = len(df)
        dE = df['dE'].values.astype(float)
        d2E = df['d2E'].values.astype(float)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        delta = df['delta'].values.astype(float)

        buy_vol = df['buy_vol'].values.astype(float) if 'buy_vol' in df.columns else np.maximum(delta, 0)
        sell_vol = df['sell_vol'].values.astype(float) if 'sell_vol' in df.columns else np.maximum(-delta, 0)

        force_raw = np.abs(dE) * np.sqrt(np.maximum(volume, 1))

        force_grad = np.zeros(n)
        force_curv = np.zeros(n)
        for i in range(1, n):
            force_grad[i] = force_raw[i] - force_raw[i - 1]
        for i in range(2, n):
            force_curv[i] = force_raw[i] - 2 * force_raw[i - 1] + force_raw[i - 2]

        dir_con = np.zeros(n)
        window = 10
        for i in range(window, n):
            price_signs = np.sign(dE[i - window:i])
            delta_signs = np.sign(delta[i - window:i])
            matches = np.sum(price_signs == delta_signs)
            dir_con[i] = matches / window

        mom3 = np.zeros(n)
        mom10 = np.zeros(n)
        for i in range(3, n):
            mom3[i] = close[i] - close[i - 3]
        for i in range(10, n):
            mom10[i] = close[i] - close[i - 10]

        accel = np.zeros(n)
        for i in range(1, n):
            accel[i] = mom3[i] - mom3[i - 1] if i >= 4 else 0

        self.traces = []
        for i in range(n):
            total_vol = buy_vol[i] + sell_vol[i]
            fs = ForceState(
                bar_idx=i,
                force_magnitude=force_raw[i],
                force_gradient=force_grad[i],
                force_curvature=force_curv[i],
                dir_consistency=dir_con[i],
                pressure_buy=buy_vol[i] / (total_vol + EPS),
                pressure_sell=sell_vol[i] / (total_vol + EPS),
                net_pressure=(buy_vol[i] - sell_vol[i]) / (total_vol + EPS),
                momentum_3=mom3[i],
                momentum_10=mom10[i],
                acceleration=accel[i],
            )
            self.traces.append(fs)

        return self.traces

    def get_state(self, bar_idx):
        if 0 <= bar_idx < len(self.traces):
            return self.traces[bar_idx]
        return ForceState()

    def summary_stats(self):
        """Aggregate stats for reporting."""
        if not self.traces:
            return {}
        mags = [t.force_magnitude for t in self.traces]
        grads = [t.force_gradient for t in self.traces]
        curvs = [t.force_curvature for t in self.traces]
        dcons = [t.dir_consistency for t in self.traces if t.bar_idx >= 10]
        return {
            'bars': len(self.traces),
            'force_mag_mean': round(np.mean(mags), 4),
            'force_mag_std': round(np.std(mags), 4),
            'force_grad_mean': round(np.mean(grads), 4),
            'force_curv_mean': round(np.mean(curvs), 4),
            'dir_consistency_mean': round(np.mean(dcons), 4) if dcons else 0,
            'dir_consistency_min': round(np.min(dcons), 4) if dcons else 0,
        }
