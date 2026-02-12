"""
Alpha Termination Point (ATP) Detection — EXP-22
=================================================
"Alpha does not end with a price outcome,
 but at the point in state space where it can no longer be reversed."

ATP = first bar t* where:
  ∀ε>0, S_{t*+ε} ∉ Reachable(A)

Four physical irreversibility channels:

  IR-1: ORBIT LOCK — Leader flips AOCL→FCL and stays for LOCK_PERSIST bars
        (orbit committed to failure, no recovery window)

  IR-2: MFE/MAE COLLAPSE — ratio < 1.0 for COLLAPSE_PERSIST consecutive bars
        (price trajectory structurally adverse, alpha unreachable)

  IR-3: ADVERSE PERSISTENCE — FAST_ADVERSE motion for ADVERSE_PERSIST consecutive bars
        (motion locked into failure mode, no healthy recovery)

  IR-4: DIRECTION INSTABILITY — no dir_stable for UNSTABLE_PERSIST consecutive bars
        after initial alpha commitment (AOCL fired but direction never stabilizes)

ATP fires on the FIRST bar where ANY channel triggers.
This is observation-only: Gate/Alpha/Size untouched.
"""

ATP_VERSION = "0.1.0"

LOCK_PERSIST = 2
COLLAPSE_PERSIST = 2
ADVERSE_PERSIST = 2
UNSTABLE_PERSIST = 3

IR_ORBIT_LOCK = "IR-1:ORBIT_LOCK"
IR_MFE_MAE_COLLAPSE = "IR-2:MFE_MAE_COLLAPSE"
IR_ADVERSE_PERSIST = "IR-3:ADVERSE_PERSIST"
IR_DIR_UNSTABLE = "IR-4:DIR_UNSTABLE"


def detect_atp(bar_evolution, dominant_orbit, first_leader, aocl_oct=None):
    """
    Detect Alpha Termination Point from bar_evolution timeline.

    Args:
        bar_evolution: list of dicts from stabilized_orbit_evaluation,
            each with keys: k, fcl_stab, aocl_stab, running_fcl,
            running_aocl, leader, dir_stable, mfe, mae
        dominant_orbit: final orbit classification (ALPHA/FAILURE/CONTESTED/NEUTRAL)
        first_leader: who led first (AOCL/FCL/TIE)
        aocl_oct: bar where AOCL first committed (None if never)

    Returns:
        dict with:
            atp_bar: int or None (bar index where alpha terminates)
            atp_channel: str or None (which IR channel triggered)
            alpha_lifespan: int or None (bars from entry to ATP)
            channels_active: dict of channel → first_trigger_bar
            post_atp_bars: int (bars remaining after ATP)
            was_alpha: bool (did this trade ever show alpha characteristics)
    """
    if not bar_evolution:
        return _empty_result()

    was_alpha = (first_leader in ('AOCL', 'TIE') or
                 aocl_oct is not None or
                 dominant_orbit in ('ALPHA', 'CONTESTED'))

    had_aocl_lead = False

    fcl_lead_streak = 0
    collapse_streak = 0
    adverse_streak = 0
    unstable_streak = 0

    channels_active = {}
    atp_bar = None
    atp_channel = None

    for bar in bar_evolution:
        k = bar['k']
        leader = bar['leader']
        mfe = bar.get('mfe', 0)
        mae = bar.get('mae', 0)
        dir_stable = bar.get('dir_stable', False)
        motion_tag = bar.get('motion_tag', '')

        aocl_stab = bar.get('aocl_stab', 0)
        if aocl_stab is None:
            aocl_stab = 0
        if aocl_stab >= 2 or leader in ('AOCL', 'TIE'):
            had_aocl_lead = True

        if had_aocl_lead and leader == 'FCL':
            fcl_lead_streak += 1
        else:
            fcl_lead_streak = 0

        if fcl_lead_streak >= LOCK_PERSIST and IR_ORBIT_LOCK not in channels_active:
            channels_active[IR_ORBIT_LOCK] = k

        mfe_mae_ratio = mfe / max(mae, 0.01)
        if mae > 0 and mfe_mae_ratio < 1.0:
            collapse_streak += 1
        else:
            collapse_streak = 0

        if collapse_streak >= COLLAPSE_PERSIST and IR_MFE_MAE_COLLAPSE not in channels_active:
            channels_active[IR_MFE_MAE_COLLAPSE] = k

        if motion_tag == 'FAST_ADVERSE':
            adverse_streak += 1
        else:
            adverse_streak = 0

        if adverse_streak >= ADVERSE_PERSIST and IR_ADVERSE_PERSIST not in channels_active:
            channels_active[IR_ADVERSE_PERSIST] = k

        if had_aocl_lead:
            if not dir_stable:
                unstable_streak += 1
            else:
                unstable_streak = 0

            if unstable_streak >= UNSTABLE_PERSIST and IR_DIR_UNSTABLE not in channels_active:
                channels_active[IR_DIR_UNSTABLE] = k

    if channels_active:
        atp_bar = min(channels_active.values())
        for ch, b in channels_active.items():
            if b == atp_bar:
                atp_channel = ch
                break

    total_bars = len(bar_evolution)
    alpha_lifespan = atp_bar - 1 if atp_bar is not None else None
    post_atp_bars = total_bars - atp_bar if atp_bar is not None else None

    return {
        'atp_bar': atp_bar,
        'atp_channel': atp_channel,
        'alpha_lifespan': alpha_lifespan,
        'channels_active': channels_active,
        'post_atp_bars': post_atp_bars,
        'was_alpha': was_alpha,
        'had_aocl_lead': had_aocl_lead,
        'total_bars': total_bars,
    }


def _empty_result():
    return {
        'atp_bar': None,
        'atp_channel': None,
        'alpha_lifespan': None,
        'channels_active': {},
        'post_atp_bars': None,
        'was_alpha': False,
        'had_aocl_lead': False,
        'total_bars': 0,
    }


def classify_alpha_fate(atp_result, dominant_orbit):
    """
    Classify the fate of an alpha based on ATP and orbit.

    Returns one of:
        SURVIVED    — alpha orbit, no ATP fired (alpha lived to the end)
        TERMINATED  — ATP fired, alpha died during trade
        STILLBORN   — never showed alpha characteristics
        IMMORTAL    — alpha orbit AND no ATP (strongest alphas)
        ZOMBIE      — ATP fired but ended in alpha orbit (structural anomaly)
    """
    atp_bar = atp_result['atp_bar']
    was_alpha = atp_result['was_alpha']

    if not was_alpha:
        return 'STILLBORN'

    if atp_bar is None and dominant_orbit in ('ALPHA',):
        return 'IMMORTAL'

    if atp_bar is None and dominant_orbit in ('CONTESTED',):
        return 'SURVIVED'

    if atp_bar is None:
        return 'SURVIVED'

    if atp_bar is not None and dominant_orbit in ('ALPHA',):
        return 'ZOMBIE'

    return 'TERMINATED'
