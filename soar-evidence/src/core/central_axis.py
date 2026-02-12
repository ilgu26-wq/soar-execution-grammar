"""
Central Axis Drift — EXP-24
============================
"Where does alpha get captured"

Measures how the Judge's decision-space reference axis shifts
around classification events (AOCL commit, FCL commit, ATP, Zombie revival).

Central Axis (CA) at bar k:
  CA(k) = <E_total(k), E_orbit(k), E_stability(k), leader(k)>

Axis Drift (ΔCA) around event at k₀:
  ΔE_axis     = E_total(k₀+δ)     - E_total(k₀-δ)
  ΔE_orbit    = E_orbit(k₀+δ)     - E_orbit(k₀-δ)
  ΔE_stability= E_stability(k₀+δ) - E_stability(k₀-δ)

Events detected:
  AOCL_COMMIT   — stab_aocl_oct bar (alpha orbit crystallizes)
  FCL_COMMIT    — stab_fcl_oct bar (failure orbit crystallizes)
  ATP           — alpha termination point
  ZOMBIE_REVIVAL— E_total goes negative then returns positive
  CROSSOVER     — leader switches (AOCL↔FCL)

No learning. No optimization. Pure measurement.
"""

CA_VERSION = "0.1.0"

LEADER_NUMERIC = {'AOCL': +1.0, 'TIE': 0.0, 'FCL': -1.0}

EVENT_AOCL_COMMIT = 'AOCL_COMMIT'
EVENT_FCL_COMMIT = 'FCL_COMMIT'
EVENT_ATP = 'ATP'
EVENT_ZOMBIE_REVIVAL = 'ZOMBIE_REVIVAL'
EVENT_CROSSOVER = 'CROSSOVER'


def _ca_at_bar(energy_traj, k):
    for step in energy_traj:
        if step['k'] == k:
            return {
                'k': k,
                'e_total': step['e_total'],
                'e_orbit': step['e_orbit'],
                'e_stability': step['e_stability'],
                'leader': step['leader'],
                'leader_num': LEADER_NUMERIC.get(step['leader'], 0.0),
                'e_excursion': step['e_excursion'],
            }
    return None


def _drift_between(ca_pre, ca_post):
    if ca_pre is None or ca_post is None:
        return None
    return {
        'delta_e_axis': round(ca_post['e_total'] - ca_pre['e_total'], 3),
        'delta_e_orbit': round(ca_post['e_orbit'] - ca_pre['e_orbit'], 4),
        'delta_e_stability': round(ca_post['e_stability'] - ca_pre['e_stability'], 1),
        'delta_e_excursion': round(ca_post['e_excursion'] - ca_pre['e_excursion'], 2),
        'delta_leader': round(ca_post['leader_num'] - ca_pre['leader_num'], 1),
        'leader_pre': ca_pre['leader'],
        'leader_post': ca_post['leader'],
    }


def detect_events(energy_traj, stab_aocl_oct=None, stab_fcl_oct=None,
                  atp_bar=None, alpha_fate=None):
    events = []

    if stab_aocl_oct is not None:
        events.append({
            'type': EVENT_AOCL_COMMIT,
            'bar': stab_aocl_oct,
        })

    if stab_fcl_oct is not None:
        events.append({
            'type': EVENT_FCL_COMMIT,
            'bar': stab_fcl_oct,
        })

    if atp_bar is not None:
        events.append({
            'type': EVENT_ATP,
            'bar': atp_bar,
        })

    in_collapse = False
    for step in energy_traj:
        if step['e_total'] < 0:
            in_collapse = True
        elif in_collapse and step['e_total'] > 0:
            events.append({
                'type': EVENT_ZOMBIE_REVIVAL,
                'bar': step['k'],
            })
            in_collapse = False

    prev_leader = None
    for step in energy_traj:
        leader = step['leader']
        if prev_leader is not None and leader != prev_leader and leader != 'TIE' and prev_leader != 'TIE':
            events.append({
                'type': EVENT_CROSSOVER,
                'bar': step['k'],
            })
        prev_leader = leader

    events.sort(key=lambda e: e['bar'])
    return events


def compute_axis_drift(energy_traj, events, delta=1):
    max_k = max(s['k'] for s in energy_traj) if energy_traj else 0
    min_k = min(s['k'] for s in energy_traj) if energy_traj else 0

    results = []
    for event in events:
        k0 = event['bar']
        pre_k = max(k0 - delta, min_k)
        post_k = min(k0 + delta, max_k)

        ca_at = _ca_at_bar(energy_traj, k0)
        ca_pre = _ca_at_bar(energy_traj, pre_k)
        ca_post = _ca_at_bar(energy_traj, post_k)

        drift = _drift_between(ca_pre, ca_post)

        results.append({
            'event_type': event['type'],
            'event_bar': k0,
            'ca_at': ca_at,
            'ca_pre': ca_pre,
            'ca_post': ca_post,
            'drift': drift,
        })

    return results


def classify_axis_movement(drift):
    if drift is None:
        return 'NO_DATA'
    de = drift['delta_e_axis']
    do = drift['delta_e_orbit']
    ds = drift['delta_e_stability']

    if de > 0.5 and do > 0:
        if ds > 0:
            return 'ALPHA_LOCK'
        return 'ALPHA_DRIFT'
    elif de < -0.5 and do < 0:
        if ds < 0:
            return 'FAILURE_LOCK'
        return 'FAILURE_DRIFT'
    elif abs(de) <= 0.5 and abs(do) <= 0.05:
        return 'NEUTRAL_COMPRESSION'
    elif (drift['leader_pre'] == 'AOCL' and drift['leader_post'] == 'FCL') or \
         (drift['leader_pre'] == 'FCL' and drift['leader_post'] == 'AOCL'):
        return 'AXIS_FLIP'
    elif de > 0.5 and do <= 0:
        return 'ENERGY_NO_ORBIT'
    elif de <= 0.5 and do > 0:
        return 'ORBIT_NO_ENERGY'
    return 'MIXED'


def summarize_axis(axis_drift_results, alpha_fate=None):
    if not axis_drift_results:
        return {
            'n_events': 0,
            'events': [],
            'dominant_movement': 'NO_DATA',
            'alpha_fate': alpha_fate,
        }

    event_summaries = []
    movements = []
    for r in axis_drift_results:
        movement = classify_axis_movement(r['drift'])
        movements.append(movement)
        summary = {
            'event_type': r['event_type'],
            'event_bar': r['event_bar'],
            'movement': movement,
        }
        if r['drift']:
            summary['delta_e_axis'] = r['drift']['delta_e_axis']
            summary['delta_e_orbit'] = r['drift']['delta_e_orbit']
            summary['delta_e_stability'] = r['drift']['delta_e_stability']
        if r['ca_at']:
            summary['e_total_at'] = r['ca_at']['e_total']
            summary['leader_at'] = r['ca_at']['leader']
        event_summaries.append(summary)

    from collections import Counter
    movement_counts = Counter(movements)
    dominant = movement_counts.most_common(1)[0][0] if movement_counts else 'NO_DATA'

    return {
        'n_events': len(axis_drift_results),
        'events': event_summaries,
        'dominant_movement': dominant,
        'alpha_fate': alpha_fate,
        'movement_counts': dict(movement_counts),
    }
