"""
ExecutionEngine — Core Grammar Runner
=======================================
Connects Boundary → Judge → Gate.
No learning. No tuning. No optimization.

Takes a list of signal candidates (with pre-computed PnL)
and applies the execution grammar to filter them.

Two modes:
  RAW   — execute all signals (no grammar)
  SOAR  — execute only what the grammar permits
"""
import numpy as np
try:
    from .boundary import PhaseDetector, Boundary
    from .judge import JudgeIR
    from .gate import EVGate, EVPack, GateDecision
    from .constitution import Constitution
except ImportError:
    from boundary import PhaseDetector, Boundary
    from judge import JudgeIR
    from gate import EVGate, EVPack, GateDecision
    from constitution import Constitution

EPS = 1e-10


class ExecutionEngine:
    def __init__(self, energy_threshold=0.05, time_cooldown=20,
                 stop_ticks=5.0, dd_limit=0.03):
        self.boundary = Boundary(energy_threshold=energy_threshold,
                                 time_cooldown=time_cooldown)
        self.detector = PhaseDetector(window=500)
        self.judge = JudgeIR()
        self.constitution = Constitution()
        self.stop_ticks = stop_ticks
        self.dd_limit = dd_limit

    def run_raw(self, signals, tick_value=5.0):
        equity = 100_000.0
        peak = equity
        trades = 0
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        max_dd_pct = 0.0
        pnls = []

        for sig in signals:
            pnl_ticks = sig['pnl_ticks']
            pnl_dollar = pnl_ticks * tick_value
            equity += pnl_dollar
            trades += 1
            pnls.append(pnl_dollar)

            if pnl_dollar > 0:
                wins += 1
                gross_profit += pnl_dollar
            else:
                gross_loss += abs(pnl_dollar)

            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        wr = wins / trades if trades > 0 else 0

        return {
            'mode': 'RAW',
            'trades': trades,
            'pf': round(pf, 2),
            'win_rate': round(wr * 100, 1),
            'max_dd_pct': round(max_dd_pct * 100, 2),
            'net_pnl': round(sum(pnls), 2),
            'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
        }

    def run_grammar(self, signals, records, tick_value=5.0, warmup=300,
                    consec_loss_pause=3):
        equity = 100_000.0
        peak = equity
        trades = 0
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        max_dd_pct = 0.0
        pnls = []
        denied = 0
        denied_pnls = []
        consec_losses = 0
        paused_until = -1

        dE_vals = np.array([r.get('dE', 0) for r in records], dtype=float)
        z_vals = np.array([r.get('z_norm', 0) for r in records], dtype=float)
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
            if i < warmup:
                continue

            if i not in sig_map:
                continue

            for sig in sig_map[i]:
                pnl_dollar = sig['pnl_ticks'] * tick_value
                dd_pct = (peak - equity) / peak if peak > 0 else 0

                vr = vol_short[i] / (vol_long[i] + EPS)
                regime = 'HIGH' if vr > 1.3 else ('LOW' if vr < 0.7 else 'MID')

                deny_cause = None
                if dd_pct > self.dd_limit:
                    deny_cause = 'DD_BREACH'
                elif consec_losses >= consec_loss_pause and i < paused_until:
                    deny_cause = 'CONSEC_LOSS_PAUSE'
                elif regime == 'HIGH' and dd_pct > self.dd_limit * 0.5:
                    deny_cause = 'HIGH_VOL_CAUTION'

                ev = self._build_ev(regime, dd_pct, deny_cause)
                gate_result = EVGate.evaluate(ev)

                if gate_result.decision == GateDecision.ALLOW:
                    equity += pnl_dollar
                    trades += 1
                    pnls.append(pnl_dollar)

                    if pnl_dollar > 0:
                        wins += 1
                        gross_profit += pnl_dollar
                        consec_losses = 0
                    else:
                        gross_loss += abs(pnl_dollar)
                        consec_losses += 1
                        if consec_losses >= consec_loss_pause:
                            paused_until = i + 50

                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak if peak > 0 else 0
                    if dd > max_dd_pct:
                        max_dd_pct = dd
                else:
                    denied += 1
                    denied_pnls.append(pnl_dollar)

        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        wr = wins / trades if trades > 0 else 0
        judge_stats = self.judge.get_stats()

        denied_losses = sum(1 for p in denied_pnls if p < 0)
        denied_avg = round(np.mean(denied_pnls), 2) if denied_pnls else 0

        return {
            'mode': 'SOAR',
            'trades': trades,
            'denied': denied,
            'denied_losses_blocked': denied_losses,
            'denied_avg_pnl': denied_avg,
            'pf': round(pf, 2),
            'win_rate': round(wr * 100, 1),
            'max_dd_pct': round(max_dd_pct * 100, 2),
            'net_pnl': round(sum(pnls), 2),
            'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'constitution_intact': self.constitution.is_intact(),
        }

    def _build_ev(self, regime, dd_pct, deny_cause):
        if deny_cause == 'DD_BREACH':
            s = 'WARNING'
            p = False
            c = 'CRITICAL'
            b = 'IRREVERSIBLE'
        elif deny_cause == 'CONSEC_LOSS_PAUSE':
            s = 'CONSTRAINED'
            p = False
            c = 'HIGH'
            b = 'IRREVERSIBLE'
        elif deny_cause == 'HIGH_VOL_CAUTION':
            s = 'FLAGGED'
            p = True
            c = 'HIGH'
            b = 'IRREVERSIBLE'
        else:
            s = 'NOMINAL'
            p = True
            c = 'LOW'
            b = 'REVERSIBLE'

        return {
            'S': s,
            'I': 'APPROVE' if p else 'HOLD',
            'B': b,
            'P': p,
            'C': c,
        }
