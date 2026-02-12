"""
Boundary — Phase detection and ENTRY gate
==========================================
Without boundary, nothing fires.

Phase detector: STI / EDG / MHI → SUB / CRITICAL / SUPER
ENTRY: phase transition + energy threshold + time cooldown

Only ENTRY opens the execution window.
Everything else is silence.
"""
import numpy as np
from collections import defaultdict

EPS = 1e-10
N_DC_BINS = 3
N_DE_BINS = 2
N_MICRO_STATES = 3 * N_DC_BINS * N_DE_BINS

CALIB_STI_RANGE = (0.05, 0.35)
CALIB_EDG_RANGE = (0.1, 0.5)
CALIB_MHI_RANGE = (0.05, 0.25)
CALIB_STI_MU = 0.18
CALIB_EDG_MU = 0.28
CALIB_MHI_MU = 0.12


def _normalize(val, lo, hi):
    return max(0.0, min(1.0, (val - lo) / (hi - lo + EPS)))


def _f_mid(z, mu=0.5, sigma=0.25):
    return float(np.exp(-0.5 * ((z - mu) / sigma) ** 2))


def _discretize_micro(rec):
    z = rec.get('z_norm', 0)
    z_bin = 0 if z < -0.1 else (2 if z > 0.1 else 1)
    dc = rec.get('dc', 0.5)
    dc_bin = 0 if dc < 0.4 else (2 if dc > 0.6 else 1)
    de_bin = 0 if rec.get('dE', 0) < 0 else 1
    return z_bin * (N_DC_BINS * N_DE_BINS) + dc_bin * N_DE_BINS + de_bin


def _compute_sti(micro, decs, theta=0.5):
    n = len(micro)
    trans_exec = defaultdict(int)
    trans_total = defaultdict(int)
    for i in range(n - 1):
        tkey = (micro[i], micro[i + 1])
        trans_total[tkey] += 1
        if decs[i] == 'execute':
            trans_exec[tkey] += 1
    if not trans_total:
        return 0.0
    selected = measured = 0
    for tkey in trans_total:
        if trans_total[tkey] >= 3:
            rate = trans_exec[tkey] / trans_total[tkey]
            measured += 1
            if rate >= theta:
                selected += 1
    return selected / measured if measured else 0.0


def _compute_edg(micro, decs):
    density = np.zeros(N_MICRO_STATES)
    for i, ms in enumerate(micro):
        if decs[i] == 'execute':
            density[ms] += 1
    total = density.sum()
    if total == 0:
        return 0.0
    p = density / total
    p_pos = p[p > 0]
    h = float(-np.sum(p_pos * np.log(p_pos)))
    return 1.0 - (h / np.log(N_MICRO_STATES))


def _compute_mhi(micro, decs, hw=10):
    n = len(micro)
    heavy = defaultdict(list)
    light = defaultdict(list)
    for i in range(hw, n):
        ms = micro[i]
        recent = decs[i - hw:i]
        frac = sum(1 for d in recent if d == 'execute') / hw
        did = 1.0 if decs[i] == 'execute' else 0.0
        if frac > 0.5:
            heavy[ms].append(did)
        else:
            light[ms].append(did)
    gaps = []
    for ms in range(N_MICRO_STATES):
        h_list = heavy.get(ms, [])
        l_list = light.get(ms, [])
        if len(h_list) >= 3 and len(l_list) >= 3:
            gaps.append(abs(np.mean(h_list) - np.mean(l_list)))
    return float(np.mean(gaps)) if gaps else 0.0


class PhaseDetector:
    def __init__(self, window=500):
        self.window = window
        self.buffer_records = []
        self.buffer_decisions = []
        self.buffer_micro = []

    def update(self, rec, decision):
        ms = _discretize_micro(rec)
        self.buffer_records.append(rec)
        self.buffer_decisions.append(decision)
        self.buffer_micro.append(ms)
        if len(self.buffer_records) > self.window * 2:
            trim = len(self.buffer_records) - self.window
            self.buffer_records = self.buffer_records[trim:]
            self.buffer_decisions = self.buffer_decisions[trim:]
            self.buffer_micro = self.buffer_micro[trim:]

    def detect(self):
        if len(self.buffer_micro) < 50:
            return 'SUB', 0.0
        micro = self.buffer_micro[-self.window:]
        decs = self.buffer_decisions[-self.window:]
        sti = _normalize(_compute_sti(micro, decs), *CALIB_STI_RANGE)
        edg = _normalize(_compute_edg(micro, decs), *CALIB_EDG_RANGE)
        mhi = _normalize(_compute_mhi(micro, decs), *CALIB_MHI_RANGE)

        s_sub = (1 - sti) + (1 - edg) + (1 - mhi)
        s_crit = (_f_mid(sti, mu=_normalize(CALIB_STI_MU, *CALIB_STI_RANGE)) +
                  _f_mid(edg, mu=_normalize(CALIB_EDG_MU, *CALIB_EDG_RANGE)) +
                  _f_mid(mhi, mu=_normalize(CALIB_MHI_MU, *CALIB_MHI_RANGE)))
        s_super = sti + edg + (1 - mhi) * 0.5

        scores = {'SUB': s_sub, 'CRITICAL': s_crit, 'SUPER': s_super}
        phase = max(scores, key=scores.get)
        sv = sorted(scores.values(), reverse=True)
        confidence = float((sv[0] - sv[1]) / (sv[0] + EPS))
        return phase, confidence


class Boundary:
    def __init__(self, energy_threshold=0.05, time_cooldown=20):
        self.energy_threshold = energy_threshold
        self.time_cooldown = time_cooldown
        self.last_entry_step = -999
        self.entry_count = 0

    def check(self, kinetic_energy, has_transition, step):
        energy_ok = abs(kinetic_energy) > self.energy_threshold
        time_ok = (step - self.last_entry_step) > self.time_cooldown
        is_entry = has_transition and energy_ok and time_ok
        if is_entry:
            self.last_entry_step = step
            self.entry_count += 1
        return is_entry
