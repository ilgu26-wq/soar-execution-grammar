"""
Alpha Energy Trajectory — EXP-23
=================================
"Alpha receives, maintains, and loses energy.
 This flow determines the entire lifecycle of the alpha."

Computes per-bar energy trajectory from bar_evolution data.

Energy Components:
  E_excursion(k) = MFE_k - MAE_k
    Raw net excursion in ticks. This is the "stored energy" of the alpha.
    Positive = alpha has favorable position. Negative = adverse.

  E_orbit(k) = (running_aocl_k - running_fcl_k) / max(running_aocl_k + running_fcl_k, 1)
    Orbit coherence [-1, +1]. Measures which orbit dominates.
    +1 = pure alpha orbit. -1 = pure failure orbit.

  E_stability(k) = 1 if dir_stable else 0
    Directional coherence. Binary: direction is sustained or not.

  dE(k) = E_excursion(k) - E_excursion(k-1)
    Energy flow rate. Positive = gaining energy. Negative = losing.

  E_total(k) = E_excursion(k) + ORBIT_WEIGHT * E_orbit(k) + STABILITY_WEIGHT * E_stability(k)
    Composite energy. Weights chosen for dimensional balance:
    E_excursion is in ticks (typically 0-20), E_orbit is [-1,+1],
    E_stability is [0,1]. Weights scale orbit/stability to ~tick units.

No learning. No optimization. Pure measurement.
"""

ENERGY_VERSION = "0.1.0"

ORBIT_WEIGHT = 4.0
STABILITY_WEIGHT = 2.0


def compute_energy_trajectory(bar_evolution, force_dir_con=None):
    """
    Compute per-bar energy trajectory from bar_evolution.

    Args:
        bar_evolution: list of dicts from stabilized_orbit_evaluation
        force_dir_con: directional consistency at entry (optional, for initial E_force)

    Returns:
        list of dicts, one per bar:
            k, e_excursion, e_orbit, e_stability, de_dt, e_total,
            mfe, mae, leader, dir_stable
    """
    if not bar_evolution:
        return []

    trajectory = []
    prev_excursion = 0.0

    for bar in bar_evolution:
        k = bar['k']
        mfe = bar.get('mfe', 0) or 0
        mae = bar.get('mae', 0) or 0
        running_aocl = bar.get('running_aocl', 0) or 0
        running_fcl = bar.get('running_fcl', 0) or 0
        dir_stable = bar.get('dir_stable', False)
        leader = bar.get('leader', 'TIE')

        e_excursion = mfe - mae

        orbit_total = running_aocl + running_fcl
        e_orbit = (running_aocl - running_fcl) / max(orbit_total, 1)

        e_stability = 1.0 if dir_stable else 0.0

        de_dt = e_excursion - prev_excursion

        e_total = e_excursion + ORBIT_WEIGHT * e_orbit + STABILITY_WEIGHT * e_stability

        trajectory.append({
            'k': k,
            'e_excursion': round(e_excursion, 2),
            'e_orbit': round(e_orbit, 3),
            'e_stability': e_stability,
            'de_dt': round(de_dt, 2),
            'e_total': round(e_total, 2),
            'mfe': mfe,
            'mae': mae,
            'leader': leader,
            'dir_stable': dir_stable,
        })

        prev_excursion = e_excursion

    return trajectory


def summarize_energy(trajectory, atp_bar=None):
    """
    Summarize energy trajectory statistics.

    Returns dict with:
        peak_energy, min_energy, final_energy,
        peak_bar, collapse_bar (first bar where e_total goes negative),
        energy_at_atp, energy_integral (total area under curve),
        de_mean, de_std
    """
    if not trajectory:
        return {
            'peak_energy': None, 'min_energy': None, 'final_energy': None,
            'peak_bar': None, 'collapse_bar': None, 'energy_at_atp': None,
            'energy_integral': 0, 'de_mean': 0, 'de_std': 0,
        }

    energies = [t['e_total'] for t in trajectory]
    de_values = [t['de_dt'] for t in trajectory]

    peak_energy = max(energies)
    min_energy = min(energies)
    final_energy = energies[-1]
    peak_bar = trajectory[energies.index(peak_energy)]['k']

    collapse_bar = None
    for t in trajectory:
        if t['e_total'] < 0:
            collapse_bar = t['k']
            break

    energy_at_atp = None
    if atp_bar is not None:
        for t in trajectory:
            if t['k'] == atp_bar:
                energy_at_atp = t['e_total']
                break

    import numpy as np
    energy_integral = sum(energies)
    de_mean = float(np.mean(de_values)) if de_values else 0
    de_std = float(np.std(de_values)) if de_values else 0

    return {
        'peak_energy': round(peak_energy, 2),
        'min_energy': round(min_energy, 2),
        'final_energy': round(final_energy, 2),
        'peak_bar': peak_bar,
        'collapse_bar': collapse_bar,
        'energy_at_atp': round(energy_at_atp, 2) if energy_at_atp is not None else None,
        'energy_integral': round(energy_integral, 2),
        'de_mean': round(de_mean, 3),
        'de_std': round(de_std, 3),
    }
