"""
PHEROMONE DRIFT LAYER v0.1.0 — EXP-20
=======================================================================
"Gate protects the world. Alpha lives in its orbit. Judge leaves a scent."

Pheromone = structural scent deposited on CONTESTED→ALPHA paths.
NOT reward, NOT reinforcement, NOT amplification.

Design:
  - Key: rc_key (alpha_type.condition@regime) — same as existing RC system
  - Deposit condition: CONTESTED + ALPHA_LEANING + first_leader in {AOCL, TIE}
  - Accumulation: +ε per qualifying trade (default ε=0.01)
  - Pheromone range: [1.0, PHEROMONE_CAP] (cap=1.20)
  - Application: effective_weight = min(base_weight * pheromone, 1.0)
  - Never exceeds 1.0 effective weight — pheromone only recovers penalized paths
  - Never penalizes — failure paths are left alone

What it does:
  Paths where CONTESTED births turned ALPHA accumulate scent.
  Future proposals on those paths resist decay slightly.
  "Scent lingers where alpha was born at the boundary"

What it does NOT do:
  - Amplify beyond 1.0
  - Penalize failure paths
  - Touch Gate in any way
  - Act immediately (delayed by nature of backtest flow)

Invariants:
  - Gate: UNTOUCHED
  - SOAR v2: LOCKED
  - PnL/WR/DD: must remain IDENTICAL when ε is small enough
"""

from collections import defaultdict

PDL_VERSION = "v0.1.0"

PHEROMONE_EPSILON = 0.01
PHEROMONE_BASE = 1.0
PHEROMONE_CAP = 1.20


class PheromoneDriftLayer:

    def __init__(self, epsilon=PHEROMONE_EPSILON, cap=PHEROMONE_CAP):
        self.epsilon = epsilon
        self.cap = cap
        self.pheromones = defaultdict(lambda: PHEROMONE_BASE)
        self.deposit_log = []
        self.total_deposits = 0
        self.total_skips = 0
        self.path_detail = defaultdict(lambda: {
            'deposits': 0,
            'first_leaders': defaultdict(int),
        })

    def deposit(self, rc_key, first_leader, dominant_orbit, contested_lean):
        if dominant_orbit != 'CONTESTED':
            self.total_skips += 1
            return False
        if contested_lean != 'ALPHA_LEANING':
            self.total_skips += 1
            return False
        if first_leader not in ('AOCL', 'TIE'):
            self.total_skips += 1
            return False

        old_val = self.pheromones[rc_key]
        new_val = min(old_val + self.epsilon, self.cap)
        self.pheromones[rc_key] = round(new_val, 4)

        self.total_deposits += 1
        self.path_detail[rc_key]['deposits'] += 1
        self.path_detail[rc_key]['first_leaders'][first_leader] += 1

        self.deposit_log.append({
            'rc_key': rc_key,
            'first_leader': first_leader,
            'old': round(old_val, 4),
            'new': round(new_val, 4),
        })

        return True

    def get_strength(self, rc_key):
        return self.pheromones[rc_key]

    def apply_to_weight(self, base_weight, rc_key):
        strength = self.get_strength(rc_key)
        effective = base_weight * strength
        return min(effective, 1.0)

    def get_active_paths(self):
        return {k: v for k, v in self.pheromones.items() if v > PHEROMONE_BASE}

    def summary(self):
        active = self.get_active_paths()
        return {
            'version': PDL_VERSION,
            'epsilon': self.epsilon,
            'cap': self.cap,
            'total_deposits': self.total_deposits,
            'total_skips': self.total_skips,
            'active_paths': len(active),
            'total_paths_observed': len(self.pheromones),
            'max_pheromone': max(self.pheromones.values()) if self.pheromones else PHEROMONE_BASE,
            'min_pheromone': min(self.pheromones.values()) if self.pheromones else PHEROMONE_BASE,
            'paths': {
                k: {
                    'strength': v,
                    'deposits': self.path_detail[k]['deposits'],
                    'first_leaders': dict(self.path_detail[k]['first_leaders']),
                }
                for k, v in sorted(active.items(), key=lambda x: -x[1])
            },
        }
