"""
ORBIT COMMITMENT LAYER v0.2 — Irreversible Trajectory Classification
=====================================================================
Dual orbit classification: Failure Orbit (FCL) + Alpha Orbit (AOCL).

"Failure orbit entry is irreversible. Alpha orbit entry is also irreversible."
"Alpha is not a 'winning trade' but a 'winning orbit.'"

FCL (Failure Commitment Layer — EXP-15):
  "Just because it survived by luck does not mean it was not in a failure orbit"
  Failure Trajectory Conditions (2+ must fire simultaneously):
    1. FAST_ADVERSE motion tag
    2. MFE/MAE ratio < 0.7
    3. Force sign flip within first 3 bars
    4. Adverse speed: MAE at bar 3 > 6 ticks
    5. Low force at entry: dir_consistency < 0.40

AOCL (Alpha Orbit Commitment Layer — EXP-16):
  "Just because it lost by bad luck does not mean it was not in an alpha orbit"
  Alpha Trajectory Conditions (2+ must fire simultaneously):
    1. HEALTHY motion tag
    2. MFE/MAE ratio >= 1.5 (favorable excursion dominates)
    3. Force sustained: no sign flip within 3 bars
    4. Low initial adverse: MAE at bar 3 < 2 ticks
    5. High force at entry: dir_consistency >= 0.65

Once committed (either orbit):
  - Mark is permanent, regardless of final PnL
  - Feeds into weight update pipeline as evidence
  - AOCL does NOT amplify. Reduces penalty dampening only.

Rules:
  - Gate LOCKED. Neither FCL nor AOCL deny trades.
  - Neither changes weights immediately.
  - Both are input to delayed weight update (next cycle).
"""

import numpy as np

FCL_VERSION = "v0.1.0"
AOCL_VERSION = "v0.1.0"
ORBIT_VERSION = "v0.4.0"

GAUGE_LOCK_VERSION = "v0.1.0"
TEMPORAL_LOCK_WINDOW = 3
TEMPORAL_LOCK_RATIO = 0.67
DIR_HYSTERESIS_BARS = 2
SHADOW_THRESHOLD = 0.5

FCL_MIN_CONDITIONS = 2
FCL_MFE_MAE_THRESHOLD = 0.7
FCL_ADVERSE_SPEED_TICKS = 6.0
FCL_DIR_CONSISTENCY_FLOOR = 0.40
FCL_FORCE_FLIP_WINDOW = 3

COND_FAST_ADVERSE = "FAST_ADVERSE"
COND_LOW_MFE_MAE = "LOW_MFE_MAE"
COND_FORCE_FLIP = "FORCE_FLIP"
COND_HIGH_ADVERSE_SPEED = "HIGH_ADVERSE_SPEED"
COND_LOW_FORCE_ENTRY = "LOW_FORCE_ENTRY"

ALL_FCL_CONDITIONS = [
    COND_FAST_ADVERSE, COND_LOW_MFE_MAE, COND_FORCE_FLIP,
    COND_HIGH_ADVERSE_SPEED, COND_LOW_FORCE_ENTRY,
]

AOCL_MIN_CONDITIONS = 2
AOCL_MFE_MAE_THRESHOLD = 1.5
AOCL_LOW_ADVERSE_TICKS = 2.0
AOCL_DIR_CONSISTENCY_FLOOR = 0.65

ACOND_HEALTHY = "HEALTHY_MOTION"
ACOND_HIGH_MFE_MAE = "HIGH_MFE_MAE"
ACOND_FORCE_SUSTAINED = "FORCE_SUSTAINED"
ACOND_LOW_INITIAL_ADVERSE = "LOW_INITIAL_ADVERSE"
ACOND_HIGH_FORCE_ENTRY = "HIGH_FORCE_ENTRY"

ALL_AOCL_CONDITIONS = [
    ACOND_HEALTHY, ACOND_HIGH_MFE_MAE, ACOND_FORCE_SUSTAINED,
    ACOND_LOW_INITIAL_ADVERSE, ACOND_HIGH_FORCE_ENTRY,
]


def check_force_flip(df, bar_idx, direction, window=FCL_FORCE_FLIP_WINDOW):
    """Check if force direction flips within window bars after entry."""
    n = len(df)
    if 'close' not in df.columns:
        return False

    close = df['close'].values
    entry_price = close[bar_idx]
    end_idx = min(bar_idx + window + 1, n)
    if end_idx <= bar_idx + 1:
        return False

    future = close[bar_idx + 1:end_idx]
    moves = (future - entry_price) * direction

    if len(moves) < 2:
        return False

    sign_changes = 0
    for i in range(1, len(moves)):
        if moves[i] * moves[i-1] < 0:
            sign_changes += 1

    return sign_changes >= 2


def evaluate_failure_trajectory(motion_result, force_state, df, bar_idx, direction):
    """
    Evaluate whether a trade has entered a failure trajectory.
    Returns (is_committed, conditions_fired, details).

    A trade is committed as failure if 2+ conditions fire simultaneously.
    """
    conditions_fired = []
    details = {}

    if motion_result['motion_tag'] == 'FAST_ADVERSE':
        conditions_fired.append(COND_FAST_ADVERSE)
        details['fast_adverse'] = True

    mfe = motion_result['mfe']
    mae = motion_result['mae']
    mfe_mae_ratio = mfe / max(mae, 0.01)
    details['mfe_mae_ratio'] = round(mfe_mae_ratio, 3)
    if mae > 0 and mfe_mae_ratio < FCL_MFE_MAE_THRESHOLD:
        conditions_fired.append(COND_LOW_MFE_MAE)

    mae_at_3 = motion_result.get('mae_at_3', 0)
    details['mae_at_3'] = mae_at_3
    if mae_at_3 > FCL_ADVERSE_SPEED_TICKS:
        conditions_fired.append(COND_HIGH_ADVERSE_SPEED)

    if force_state is not None:
        details['dir_consistency'] = round(force_state.dir_consistency, 3)
        if force_state.dir_consistency < FCL_DIR_CONSISTENCY_FLOOR:
            conditions_fired.append(COND_LOW_FORCE_ENTRY)

    if check_force_flip(df, bar_idx, direction):
        conditions_fired.append(COND_FORCE_FLIP)
        details['force_flip'] = True

    is_committed = len(conditions_fired) >= FCL_MIN_CONDITIONS

    return is_committed, conditions_fired, details


class FailureCommitmentRecord:
    """A single irreversible failure commitment mark."""
    __slots__ = ['bar_idx', 'conditions', 'details', 'pnl', 'survived']

    def __init__(self, bar_idx, conditions, details, pnl=0.0):
        self.bar_idx = bar_idx
        self.conditions = conditions
        self.details = details
        self.pnl = pnl
        self.survived = pnl > 0

    def to_dict(self):
        return {
            'bar_idx': self.bar_idx,
            'conditions': self.conditions,
            'condition_count': len(self.conditions),
            'pnl': round(self.pnl, 2),
            'survived': self.survived,
            'details': self.details,
        }


class FCLMemory:
    """
    Tracks failure commitments per (alpha, condition, regime) slice.
    Provides commitment_rate as additional input to weight updates.
    """
    def __init__(self):
        self.records = []
        self.rc_commits = {}
        self.rc_totals = {}
        self.condition_frequency = {c: 0 for c in ALL_FCL_CONDITIONS}

    def record(self, rc_key, bar_idx, conditions, details, pnl=0.0):
        rec = FailureCommitmentRecord(bar_idx, conditions, details, pnl)
        self.records.append(rec)

        if rc_key not in self.rc_commits:
            self.rc_commits[rc_key] = 0
            self.rc_totals[rc_key] = 0

        self.rc_commits[rc_key] += 1

        for c in conditions:
            self.condition_frequency[c] += 1

    def record_trade(self, rc_key):
        if rc_key not in self.rc_totals:
            self.rc_totals[rc_key] = 0
            self.rc_commits[rc_key] = 0
        self.rc_totals[rc_key] += 1

    def get_commitment_rate(self, rc_key):
        total = self.rc_totals.get(rc_key, 0)
        if total == 0:
            return 0.0
        commits = self.rc_commits.get(rc_key, 0)
        return commits / total

    def summary(self):
        total_trades = sum(self.rc_totals.values())
        total_commits = len(self.records)
        survived = sum(1 for r in self.records if r.survived)
        survived_pnl = sum(r.pnl for r in self.records if r.survived)
        failed_pnl = sum(r.pnl for r in self.records if not r.survived)

        return {
            'total_trades': total_trades,
            'total_commits': total_commits,
            'commitment_rate': round(total_commits / max(total_trades, 1), 3),
            'survived_count': survived,
            'survived_pnl': round(survived_pnl, 2),
            'true_failure_count': total_commits - survived,
            'true_failure_pnl': round(failed_pnl, 2),
            'condition_frequency': dict(self.condition_frequency),
        }

    def rc_table(self):
        rows = []
        for rc_key in sorted(self.rc_totals.keys()):
            total = self.rc_totals[rc_key]
            commits = self.rc_commits.get(rc_key, 0)
            if total == 0:
                continue
            rows.append({
                'rc_key': rc_key,
                'total': total,
                'commits': commits,
                'commitment_rate': round(commits / total, 3),
            })
        return rows

    def to_dict(self):
        return {
            'summary': self.summary(),
            'rc_table': self.rc_table(),
            'records_sample': [r.to_dict() for r in self.records[:20]],
        }


def evaluate_alpha_trajectory(motion_result, force_state, df, bar_idx, direction):
    """
    Evaluate whether a trade has entered an alpha (success) trajectory.
    Returns (is_committed, conditions_fired, details).

    A trade is committed as alpha orbit if 2+ conditions fire simultaneously.
    Mirror of evaluate_failure_trajectory.
    """
    conditions_fired = []
    details = {}

    if motion_result['motion_tag'] == 'HEALTHY':
        conditions_fired.append(ACOND_HEALTHY)
        details['healthy'] = True

    mfe = motion_result['mfe']
    mae = motion_result['mae']
    mfe_mae_ratio = mfe / max(mae, 0.01)
    details['mfe_mae_ratio'] = round(mfe_mae_ratio, 3)
    if mfe_mae_ratio >= AOCL_MFE_MAE_THRESHOLD:
        conditions_fired.append(ACOND_HIGH_MFE_MAE)

    mae_at_3 = motion_result.get('mae_at_3', 0)
    details['mae_at_3'] = mae_at_3
    if mae_at_3 < AOCL_LOW_ADVERSE_TICKS:
        conditions_fired.append(ACOND_LOW_INITIAL_ADVERSE)

    if force_state is not None:
        details['dir_consistency'] = round(force_state.dir_consistency, 3)
        if force_state.dir_consistency >= AOCL_DIR_CONSISTENCY_FLOOR:
            conditions_fired.append(ACOND_HIGH_FORCE_ENTRY)

    if not check_force_flip(df, bar_idx, direction):
        conditions_fired.append(ACOND_FORCE_SUSTAINED)
        details['force_sustained'] = True

    is_committed = len(conditions_fired) >= AOCL_MIN_CONDITIONS

    return is_committed, conditions_fired, details


class AlphaOrbitRecord:
    """A single irreversible alpha orbit commitment mark."""
    __slots__ = ['bar_idx', 'conditions', 'details', 'pnl', 'lost']

    def __init__(self, bar_idx, conditions, details, pnl=0.0):
        self.bar_idx = bar_idx
        self.conditions = conditions
        self.details = details
        self.pnl = pnl
        self.lost = pnl <= 0

    def to_dict(self):
        return {
            'bar_idx': self.bar_idx,
            'conditions': self.conditions,
            'condition_count': len(self.conditions),
            'pnl': round(self.pnl, 2),
            'lost': self.lost,
            'details': self.details,
        }


class AOCLMemory:
    """
    Tracks alpha orbit commitments per (alpha, condition, regime) slice.
    Provides alpha_orbit_rate as input to weight updates.
    AOCL does NOT amplify weights. It only provides evidence to reduce
    penalty dampening for slices with high alpha orbit rates.
    """
    def __init__(self):
        self.records = []
        self.rc_commits = {}
        self.rc_totals = {}
        self.condition_frequency = {c: 0 for c in ALL_AOCL_CONDITIONS}

    def record(self, rc_key, bar_idx, conditions, details, pnl=0.0):
        rec = AlphaOrbitRecord(bar_idx, conditions, details, pnl)
        self.records.append(rec)

        if rc_key not in self.rc_commits:
            self.rc_commits[rc_key] = 0
            self.rc_totals[rc_key] = 0

        self.rc_commits[rc_key] += 1

        for c in conditions:
            self.condition_frequency[c] += 1

    def record_trade(self, rc_key):
        if rc_key not in self.rc_totals:
            self.rc_totals[rc_key] = 0
            self.rc_commits[rc_key] = 0
        self.rc_totals[rc_key] += 1

    def get_alpha_orbit_rate(self, rc_key):
        total = self.rc_totals.get(rc_key, 0)
        if total == 0:
            return 0.0
        commits = self.rc_commits.get(rc_key, 0)
        return commits / total

    def summary(self):
        total_trades = sum(self.rc_totals.values())
        total_commits = len(self.records)
        lost = sum(1 for r in self.records if r.lost)
        lost_pnl = sum(r.pnl for r in self.records if r.lost)
        won_pnl = sum(r.pnl for r in self.records if not r.lost)

        return {
            'total_trades': total_trades,
            'total_commits': total_commits,
            'alpha_orbit_rate': round(total_commits / max(total_trades, 1), 3),
            'won_count': total_commits - lost,
            'won_pnl': round(won_pnl, 2),
            'lost_despite_orbit_count': lost,
            'lost_despite_orbit_pnl': round(lost_pnl, 2),
            'condition_frequency': dict(self.condition_frequency),
        }

    def rc_table(self):
        rows = []
        for rc_key in sorted(self.rc_totals.keys()):
            total = self.rc_totals[rc_key]
            commits = self.rc_commits.get(rc_key, 0)
            if total == 0:
                continue
            rows.append({
                'rc_key': rc_key,
                'total': total,
                'commits': commits,
                'alpha_orbit_rate': round(commits / total, 3),
            })
        return rows

    def to_dict(self):
        return {
            'summary': self.summary(),
            'rc_table': self.rc_table(),
            'records_sample': [r.to_dict() for r in self.records[:20]],
        }


GAUGE_EVAL_WINDOW = 10


def _partial_motion_at_bar(df, entry_idx, direction, k, tick_size=0.25):
    """Compute partial MFE/MAE/motion_tag up to bar k after entry."""
    n = len(df)
    close = df['close'].values
    entry_price = close[entry_idx]

    end_idx = min(entry_idx + k + 1, n)
    if end_idx <= entry_idx + 1:
        return None

    future = close[entry_idx + 1:end_idx]
    moves = (future - entry_price) * direction / tick_size

    running_mfe = 0.0
    running_mae = 0.0
    for m in moves:
        if m > running_mfe:
            running_mfe = m
        if m < running_mae:
            running_mae = m

    mfe = running_mfe
    mae = abs(running_mae)

    from core.motion_watchdog import (
        MFE_MIN_TICKS, MAE_FAST_TICKS, STALL_THRESHOLD_TICKS,
    )

    mae_at_3 = 0.0
    if k >= 3 and len(moves) >= 3:
        mae_at_3 = abs(min(moves[:3]))
    mfe_at_5 = 0.0
    if k >= 5 and len(moves) >= 5:
        mfe_at_5 = max(moves[:5])

    check_mae = mae_at_3 if k >= 3 else mae
    check_mfe = mfe_at_5 if k >= 5 else mfe

    if check_mae >= MAE_FAST_TICKS:
        tag = 'FAST_ADVERSE'
    elif check_mfe < MFE_MIN_TICKS:
        if check_mae < STALL_THRESHOLD_TICKS:
            tag = 'STALL'
        else:
            tag = 'NO_FOLLOW'
    else:
        tag = 'HEALTHY'

    return {
        'motion_tag': tag,
        'mfe': round(mfe, 2),
        'mae': round(mae, 2),
        'mae_at_3': round(mae_at_3, 2),
        'mfe_at_5': round(mfe_at_5, 2),
    }


def _check_force_flip_at_k(df, bar_idx, direction, k):
    """Check force flip using only bars up to k after entry."""
    n = len(df)
    if 'close' not in df.columns:
        return False
    close = df['close'].values
    entry_price = close[bar_idx]
    end_idx = min(bar_idx + k + 1, n)
    if end_idx <= bar_idx + 1:
        return False
    future = close[bar_idx + 1:end_idx]
    moves = (future - entry_price) * direction
    if len(moves) < 2:
        return False
    sign_changes = 0
    for i in range(1, len(moves)):
        if moves[i] * moves[i-1] < 0:
            sign_changes += 1
    return sign_changes >= 2


def _count_fcl_conditions_at_k(partial_motion, force_state, df, bar_idx, direction, k):
    """Count how many FCL conditions fire at bar offset k."""
    count = 0
    if partial_motion['motion_tag'] == 'FAST_ADVERSE':
        count += 1
    mfe = partial_motion['mfe']
    mae = partial_motion['mae']
    ratio = mfe / max(mae, 0.01)
    if mae > 0 and ratio < FCL_MFE_MAE_THRESHOLD:
        count += 1
    if k >= 3 and partial_motion['mae_at_3'] > FCL_ADVERSE_SPEED_TICKS:
        count += 1
    if force_state is not None and force_state.dir_consistency < FCL_DIR_CONSISTENCY_FLOOR:
        count += 1
    if k >= 3 and _check_force_flip_at_k(df, bar_idx, direction, min(k, FCL_FORCE_FLIP_WINDOW)):
        count += 1
    return count


def _count_aocl_conditions_at_k(partial_motion, force_state, df, bar_idx, direction, k):
    """Count how many AOCL conditions fire at bar offset k."""
    count = 0
    if partial_motion['motion_tag'] == 'HEALTHY':
        count += 1
    mfe = partial_motion['mfe']
    mae = partial_motion['mae']
    ratio = mfe / max(mae, 0.01)
    if ratio >= AOCL_MFE_MAE_THRESHOLD:
        count += 1
    if k >= 3 and partial_motion['mae_at_3'] < AOCL_LOW_ADVERSE_TICKS:
        count += 1
    if force_state is not None and force_state.dir_consistency >= AOCL_DIR_CONSISTENCY_FLOOR:
        count += 1
    if k >= 3 and not _check_force_flip_at_k(df, bar_idx, direction, min(k, FCL_FORCE_FLIP_WINDOW)):
        count += 1
    return count


def progressive_orbit_evaluation(df, bar_idx, direction, force_state, tick_size=0.25):
    """
    EXP-17: Progressive bar-by-bar orbit evaluation.

    For each bar k after entry (k=1..10), evaluates FCL and AOCL
    conditions to find:
      - OCT: First bar where orbit commits (2+ conditions fire)
      - OSS: After commit, how often the opposite orbit fires (stability)

    Returns dict with fcl_oct, aocl_oct, oss_fcl, oss_aocl, bar_timeline.
    """
    n = len(df)
    max_k = min(GAUGE_EVAL_WINDOW, n - bar_idx - 1)
    if max_k < 1:
        return {
            'fcl_oct': None, 'aocl_oct': None,
            'oss_fcl': None, 'oss_aocl': None,
            'bar_timeline': [],
        }

    fcl_oct = None
    aocl_oct = None
    bar_timeline = []

    fcl_post_commit_opposite_fires = 0
    fcl_post_commit_bars = 0
    aocl_post_commit_opposite_fires = 0
    aocl_post_commit_bars = 0

    for k in range(1, max_k + 1):
        partial = _partial_motion_at_bar(df, bar_idx, direction, k, tick_size)
        if partial is None:
            break

        fcl_c = _count_fcl_conditions_at_k(partial, force_state, df, bar_idx, direction, k)
        aocl_c = _count_aocl_conditions_at_k(partial, force_state, df, bar_idx, direction, k)

        bar_timeline.append({
            'k': k,
            'fcl_conds': fcl_c,
            'aocl_conds': aocl_c,
            'motion_tag': partial['motion_tag'],
            'mfe': partial['mfe'],
            'mae': partial['mae'],
        })

        if fcl_oct is None and fcl_c >= FCL_MIN_CONDITIONS:
            fcl_oct = k
        if aocl_oct is None and aocl_c >= AOCL_MIN_CONDITIONS:
            aocl_oct = k

        if fcl_oct is not None:
            fcl_post_commit_bars += 1
            if aocl_c >= AOCL_MIN_CONDITIONS:
                fcl_post_commit_opposite_fires += 1

        if aocl_oct is not None:
            aocl_post_commit_bars += 1
            if fcl_c >= FCL_MIN_CONDITIONS:
                aocl_post_commit_opposite_fires += 1

    oss_fcl = None
    if fcl_oct is not None and fcl_post_commit_bars > 0:
        oss_fcl = round(1.0 - fcl_post_commit_opposite_fires / fcl_post_commit_bars, 3)

    oss_aocl = None
    if aocl_oct is not None and aocl_post_commit_bars > 0:
        oss_aocl = round(1.0 - aocl_post_commit_opposite_fires / aocl_post_commit_bars, 3)

    return {
        'fcl_oct': fcl_oct,
        'aocl_oct': aocl_oct,
        'oss_fcl': oss_fcl,
        'oss_aocl': oss_aocl,
        'bar_timeline': bar_timeline,
    }


def _temporal_lock_filter(history, k, min_conditions=2):
    """
    Temporal Lock: condition must fire consistently over recent bars.
    Look at last TEMPORAL_LOCK_WINDOW bars and check if conditions fired
    in >= TEMPORAL_LOCK_RATIO of them.
    """
    window_start = max(0, len(history) - TEMPORAL_LOCK_WINDOW)
    recent = history[window_start:]
    if not recent:
        return 0
    fire_count = sum(1 for h in recent if h >= min_conditions)
    if fire_count / len(recent) >= TEMPORAL_LOCK_RATIO:
        return recent[-1]
    return 0


def _check_dir_hysteresis(df, bar_idx, direction, k):
    """
    Directional Hysteresis: force direction must be sustained for
    DIR_HYSTERESIS_BARS consecutive bars to count as stable.
    Returns True if direction is stable.
    """
    n = len(df)
    close = df['close'].values
    entry_price = close[bar_idx]

    start_bar = max(1, k - DIR_HYSTERESIS_BARS + 1)
    end_bar = k

    consecutive = 0
    for j in range(start_bar, end_bar + 1):
        idx = bar_idx + j
        if idx >= n:
            break
        move = (close[idx] - entry_price) * direction
        if move > 0:
            consecutive += 1
        else:
            consecutive = 0

    return consecutive >= DIR_HYSTERESIS_BARS


def stabilized_orbit_evaluation(df, bar_idx, direction, force_state, tick_size=0.25):
    """
    EXP-18a: Gauge Lock v2 — Stabilized orbit evaluation.

    Three stabilization mechanisms:
      1. Temporal Lock: conditions must fire consistently over recent bars
      2. Directional Hysteresis: force direction must sustain for k bars
      3. Orbit Dominance Rule: only the dominant orbit is "locked",
         the other becomes a shadow event

    Returns dict with:
      - raw_* : EXP-17 original values (for comparison)
      - stabilized_* : after gauge lock v2
      - dominant_orbit: ALPHA / FAILURE / NEUTRAL / CONTESTED
      - shadow_events: count of suppressed orbit fires
      - dir_stable_bars: how many bars had stable direction
    """
    n = len(df)
    max_k = min(GAUGE_EVAL_WINDOW, n - bar_idx - 1)
    if max_k < 1:
        return {
            'raw_fcl_oct': None, 'raw_aocl_oct': None,
            'raw_oss_fcl': None, 'raw_oss_aocl': None,
            'stab_fcl_oct': None, 'stab_aocl_oct': None,
            'stab_oss_fcl': None, 'stab_oss_aocl': None,
            'dominant_orbit': 'NEUTRAL',
            'shadow_events': 0,
            'fcl_fire_bars': 0, 'aocl_fire_bars': 0,
            'dir_stable_bars': 0,
        }

    raw_fcl_oct = None
    raw_aocl_oct = None
    stab_fcl_oct = None
    stab_aocl_oct = None

    fcl_cond_history = []
    aocl_cond_history = []

    raw_fcl_post_opp = 0
    raw_fcl_post_bars = 0
    raw_aocl_post_opp = 0
    raw_aocl_post_bars = 0

    stab_fcl_post_opp = 0
    stab_fcl_post_bars = 0
    stab_aocl_post_opp = 0
    stab_aocl_post_bars = 0

    total_fcl_fires = 0
    total_aocl_fires = 0
    dir_stable_bars = 0
    bar_evolution = []

    for k in range(1, max_k + 1):
        partial = _partial_motion_at_bar(df, bar_idx, direction, k, tick_size)
        if partial is None:
            break

        fcl_c = _count_fcl_conditions_at_k(partial, force_state, df, bar_idx, direction, k)
        aocl_c = _count_aocl_conditions_at_k(partial, force_state, df, bar_idx, direction, k)

        fcl_cond_history.append(fcl_c)
        aocl_cond_history.append(aocl_c)

        dir_stable = _check_dir_hysteresis(df, bar_idx, direction, k)
        if dir_stable:
            dir_stable_bars += 1

        if raw_fcl_oct is None and fcl_c >= FCL_MIN_CONDITIONS:
            raw_fcl_oct = k
        if raw_aocl_oct is None and aocl_c >= AOCL_MIN_CONDITIONS:
            raw_aocl_oct = k

        if raw_fcl_oct is not None:
            raw_fcl_post_bars += 1
            if aocl_c >= AOCL_MIN_CONDITIONS:
                raw_fcl_post_opp += 1
        if raw_aocl_oct is not None:
            raw_aocl_post_bars += 1
            if fcl_c >= FCL_MIN_CONDITIONS:
                raw_aocl_post_opp += 1

        stab_fcl_c = _temporal_lock_filter(fcl_cond_history, k, FCL_MIN_CONDITIONS)
        stab_aocl_c = _temporal_lock_filter(aocl_cond_history, k, AOCL_MIN_CONDITIONS)

        if dir_stable:
            pass
        else:
            if stab_aocl_c >= AOCL_MIN_CONDITIONS:
                stab_aocl_c = max(0, stab_aocl_c - 1)

        if stab_fcl_c >= FCL_MIN_CONDITIONS:
            total_fcl_fires += 1
        if stab_aocl_c >= AOCL_MIN_CONDITIONS:
            total_aocl_fires += 1

        running_fcl = total_fcl_fires
        running_aocl = total_aocl_fires
        if running_fcl > running_aocl:
            bar_leader = 'FCL'
        elif running_aocl > running_fcl:
            bar_leader = 'AOCL'
        else:
            bar_leader = 'TIE'
        bar_evolution.append({
            'k': k,
            'fcl_raw': fcl_c,
            'aocl_raw': aocl_c,
            'fcl_stab': stab_fcl_c,
            'aocl_stab': stab_aocl_c,
            'running_fcl': running_fcl,
            'running_aocl': running_aocl,
            'leader': bar_leader,
            'dir_stable': dir_stable,
            'mfe': partial['mfe'],
            'mae': partial['mae'],
        })

        if stab_fcl_oct is None and stab_fcl_c >= FCL_MIN_CONDITIONS:
            stab_fcl_oct = k
        if stab_aocl_oct is None and stab_aocl_c >= AOCL_MIN_CONDITIONS:
            stab_aocl_oct = k

        if stab_fcl_oct is not None:
            stab_fcl_post_bars += 1
            if stab_aocl_c >= AOCL_MIN_CONDITIONS:
                stab_fcl_post_opp += 1
        if stab_aocl_oct is not None:
            stab_aocl_post_bars += 1
            if stab_fcl_c >= FCL_MIN_CONDITIONS:
                stab_aocl_post_opp += 1

    raw_oss_fcl = None
    if raw_fcl_oct is not None and raw_fcl_post_bars > 0:
        raw_oss_fcl = round(1.0 - raw_fcl_post_opp / raw_fcl_post_bars, 3)
    raw_oss_aocl = None
    if raw_aocl_oct is not None and raw_aocl_post_bars > 0:
        raw_oss_aocl = round(1.0 - raw_aocl_post_opp / raw_aocl_post_bars, 3)

    stab_oss_fcl = None
    if stab_fcl_oct is not None and stab_fcl_post_bars > 0:
        stab_oss_fcl = round(1.0 - stab_fcl_post_opp / stab_fcl_post_bars, 3)
    stab_oss_aocl = None
    if stab_aocl_oct is not None and stab_aocl_post_bars > 0:
        stab_oss_aocl = round(1.0 - stab_aocl_post_opp / stab_aocl_post_bars, 3)

    shadow_events = 0
    if total_fcl_fires > 0 and total_aocl_fires > 0:
        if total_fcl_fires > total_aocl_fires:
            ratio = total_aocl_fires / total_fcl_fires
            if ratio < SHADOW_THRESHOLD:
                dominant_orbit = 'FAILURE'
                shadow_events = total_aocl_fires
            else:
                dominant_orbit = 'CONTESTED'
        elif total_aocl_fires > total_fcl_fires:
            ratio = total_fcl_fires / total_aocl_fires
            if ratio < SHADOW_THRESHOLD:
                dominant_orbit = 'ALPHA'
                shadow_events = total_fcl_fires
            else:
                dominant_orbit = 'CONTESTED'
        else:
            dominant_orbit = 'CONTESTED'
    elif total_fcl_fires > 0:
        dominant_orbit = 'FAILURE'
    elif total_aocl_fires > 0:
        dominant_orbit = 'ALPHA'
    else:
        dominant_orbit = 'NEUTRAL'

    crossover_bar = None
    if len(bar_evolution) >= 2:
        for bi in range(1, len(bar_evolution)):
            prev_leader = bar_evolution[bi - 1]['leader']
            curr_leader = bar_evolution[bi]['leader']
            if prev_leader != curr_leader and prev_leader != 'TIE' and curr_leader != 'TIE':
                crossover_bar = bar_evolution[bi]['k']
                break

    first_leader = bar_evolution[0]['leader'] if bar_evolution else 'NONE'
    final_leader = bar_evolution[-1]['leader'] if bar_evolution else 'NONE'

    if dominant_orbit == 'CONTESTED':
        if total_aocl_fires > total_fcl_fires:
            contested_lean = 'ALPHA_LEANING'
        elif total_fcl_fires > total_aocl_fires:
            contested_lean = 'FAILURE_LEANING'
        else:
            contested_lean = 'BALANCED'
    else:
        contested_lean = None

    return {
        'raw_fcl_oct': raw_fcl_oct,
        'raw_aocl_oct': raw_aocl_oct,
        'raw_oss_fcl': raw_oss_fcl,
        'raw_oss_aocl': raw_oss_aocl,
        'stab_fcl_oct': stab_fcl_oct,
        'stab_aocl_oct': stab_aocl_oct,
        'stab_oss_fcl': stab_oss_fcl,
        'stab_oss_aocl': stab_oss_aocl,
        'dominant_orbit': dominant_orbit,
        'shadow_events': shadow_events,
        'fcl_fire_bars': total_fcl_fires,
        'aocl_fire_bars': total_aocl_fires,
        'dir_stable_bars': dir_stable_bars,
        'bar_evolution': bar_evolution,
        'crossover_bar': crossover_bar,
        'first_leader': first_leader,
        'final_leader': final_leader,
        'contested_lean': contested_lean,
    }
