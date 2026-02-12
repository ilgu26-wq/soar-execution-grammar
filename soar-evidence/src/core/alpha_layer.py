"""
ALPHA DISCOVERY LAYER v1.4 — Motion-Aware Alpha Penalty
=======================================================================
Generates multiple alpha candidates per bar.
SOAR gate decides which pass — Alpha NEVER decides execution.

v1.4 Change (EXP-14):
  - Motion-based weight penalty on RC slices
  - motion_score = healthy_rate - fast_adverse_rate per (alpha, cond, regime)
  - penalty = clip(k * |motion_score|, 0.0, 0.02) when motion_score < 0
  - k = 0.5 (very conservative), max penalty per cycle = 2%
  - Floor 0.80 (never fully remove), ceiling 1.00 (never amplify)
  - "EXP-10 learned WHAT; EXP-12 WHEN; EXP-13 mapped WHERE; EXP-14 penalizes"
  - Gate reads NONE of this. Gate is LOCKED.

v1.3 (EXP-12):
  - Regime-conditional (RC) stats: key = (alpha, condition, regime)
  - RC-level proposal weights: lookup priority RC > condition > default

v1.2 (EXP-10):
  - Proposal Probability Shaping per (alpha, condition) pair
  - Score = 0.5*legitimacy + 0.3*norm_EV + 0.2*(1 - anti_soar_fail_rate)
  - Weight range: [0.80, 1.00] — NO AMPLIFICATION (never > 1.0)

v1.1 (EXP-09):
  - Condition tags per alpha from Force state
  - Memory tracks per (alpha_type, condition) pair

Alpha Learning (NOT reinforcement learning):
  - "Legitimacy" = allowed / proposed (NOT profit-based)
  - Proposal shaping = "sensory correction", not reward optimization
  - Suppression threshold: legitimacy < 0.10 AND n >= 100

Condition Tags (Force-derived):
  - LOW_CURV / HIGH_CURV    — curvature magnitude split
  - NARROW_RNG / WIDE_RNG   — channel range split
  - ALIGNED / MISALIGNED     — directional consistency split
  - STRONG_F / WEAK_F        — force magnitude split

Rules:
  - Alpha generates CANDIDATES only
  - SOAR gate is the ONLY decider
  - Proposal weight shapes proposal frequency, NEVER gate decisions
  - Weight can only DECREASE (reduce noise), never amplify
  - Suppression threshold: legitimacy < 0.10 AND n >= 100
"""

import numpy as np
from collections import defaultdict

ALPHA_LAYER_VERSION = "v1.4.0"

ALPHA_Z_MOMENTUM = "Z_MOMENTUM"
ALPHA_RANGE_BREAKOUT = "RANGE_BREAKOUT"
ALPHA_VOL_CONTRACTION = "VOL_CONTRACTION"
ALPHA_MEAN_REVERT = "MEAN_REVERT"
ALPHA_MICRO_MOMENTUM = "MICRO_MOMENTUM"
ALPHA_FLOW_IMBALANCE = "FLOW_IMBALANCE"

ALL_ALPHAS = [
    ALPHA_Z_MOMENTUM, ALPHA_RANGE_BREAKOUT, ALPHA_VOL_CONTRACTION,
    ALPHA_MEAN_REVERT, ALPHA_MICRO_MOMENTUM, ALPHA_FLOW_IMBALANCE,
]

SUPPRESSION_MIN_N = 100
SUPPRESSION_THRESHOLD = 0.10

COND_LOW_CURV = "LOW_CURV"
COND_HIGH_CURV = "HIGH_CURV"
COND_NARROW_RNG = "NARROW_RNG"
COND_WIDE_RNG = "WIDE_RNG"
COND_ALIGNED = "ALIGNED"
COND_MISALIGNED = "MISALIGNED"
COND_STRONG_F = "STRONG_F"
COND_WEAK_F = "WEAK_F"
COND_NONE = "NONE"

ALL_CONDITIONS = [
    COND_LOW_CURV, COND_HIGH_CURV,
    COND_NARROW_RNG, COND_WIDE_RNG,
    COND_ALIGNED, COND_MISALIGNED,
    COND_STRONG_F, COND_WEAK_F,
]

CURVATURE_THRESHOLD = 5.0
DIR_CONSISTENCY_THRESHOLD = 0.6
FORCE_MAG_THRESHOLD = 15.0
RANGE_WIDTH_THRESHOLD = 5.0

PROPOSAL_WEIGHT_MIN = 0.80
PROPOSAL_WEIGHT_MAX = 1.00
PROPOSAL_WEIGHT_DEFAULT = 1.00
PROPOSAL_STEP_MAX = 0.05
PROPOSAL_LEARNING_MIN_N = 100
PROPOSAL_OBSERVE_MIN_N = 50

RC_LEARNING_MIN_N = 30
RC_OBSERVE_MIN_N = 15

MOTION_PENALTY_K = 0.5
MOTION_PENALTY_MAX = 0.02
MOTION_PENALTY_MIN_N = 30

SCORE_W_LEGITIMACY = 0.5
SCORE_W_EV = 0.3
SCORE_W_ANTI_SOAR = 0.2

EPS = 1e-10


def classify_condition(alpha_type, force_state, features=None):
    """
    Derive condition tag from Force state for a given alpha type.
    Each alpha type gets ONE condition from its most relevant dimension.
    """
    if force_state is None:
        return COND_NONE

    if alpha_type == ALPHA_Z_MOMENTUM:
        return COND_HIGH_CURV if abs(force_state.force_curvature) > CURVATURE_THRESHOLD else COND_LOW_CURV

    elif alpha_type == ALPHA_RANGE_BREAKOUT:
        rng = (features or {}).get('range', 0)
        return COND_WIDE_RNG if rng > RANGE_WIDTH_THRESHOLD else COND_NARROW_RNG

    elif alpha_type == ALPHA_MICRO_MOMENTUM:
        return COND_ALIGNED if force_state.dir_consistency > DIR_CONSISTENCY_THRESHOLD else COND_MISALIGNED

    elif alpha_type == ALPHA_FLOW_IMBALANCE:
        return COND_STRONG_F if force_state.force_magnitude > FORCE_MAG_THRESHOLD else COND_WEAK_F

    elif alpha_type == ALPHA_MEAN_REVERT:
        return COND_HIGH_CURV if abs(force_state.force_curvature) > CURVATURE_THRESHOLD else COND_LOW_CURV

    elif alpha_type == ALPHA_VOL_CONTRACTION:
        return COND_STRONG_F if force_state.force_magnitude > FORCE_MAG_THRESHOLD else COND_WEAK_F

    return COND_NONE


class AlphaCandidate:
    """A single alpha proposal for a specific bar, now with condition tag."""
    __slots__ = ['bar_idx', 'alpha_type', 'direction', 'strength', 'features', 'condition']

    def __init__(self, bar_idx, alpha_type, direction, strength, features=None, condition=None):
        self.bar_idx = bar_idx
        self.alpha_type = alpha_type
        self.direction = direction
        self.strength = strength
        self.features = features or {}
        self.condition = condition or COND_NONE

    def to_dict(self):
        return {
            'bar_idx': self.bar_idx,
            'alpha_type': self.alpha_type,
            'condition': self.condition,
            'direction': self.direction,
            'strength': round(self.strength, 4),
        }

    @property
    def full_tag(self):
        return f"{self.alpha_type}.{self.condition}"


def _empty_stat():
    return {'proposed': 0, 'allowed': 0, 'denied': 0,
            'total_pnl': 0.0, 'wins': 0, 'losses': 0}


def _empty_anti_soar():
    return {'denied_total': 0, 'denied_would_win': 0, 'denied_pnl_sum': 0.0}


class AlphaMemory:
    """
    Tracks per-alpha-type, per-condition, AND per-regime-condition statistics.
    Learning is structural: legitimacy = allowed / proposed.

    EXP-12: Regime × Condition resolution — same score, same weight range,
    but weights are now per (alpha, condition, regime) triple.
    Lookup priority: RC-weight > condition-weight > default.
    """
    def __init__(self):
        self.stats = {a: _empty_stat() for a in ALL_ALPHAS}
        self.cond_stats = defaultdict(_empty_stat)
        self.rc_stats = defaultdict(_empty_stat)
        self.anti_soar = defaultdict(_empty_anti_soar)
        self.rc_anti_soar = defaultdict(_empty_anti_soar)
        self.proposal_weights = defaultdict(lambda: PROPOSAL_WEIGHT_DEFAULT)
        self.rc_weights = defaultdict(lambda: PROPOSAL_WEIGHT_DEFAULT)
        self.motion_stats = defaultdict(lambda: defaultdict(int))
        self._update_count = 0

    def _rc_key(self, alpha_type, condition, regime):
        return f"{alpha_type}.{condition}@{regime}"

    def record_proposal(self, alpha_type, condition=COND_NONE, regime=None):
        self.stats[alpha_type]['proposed'] += 1
        self.cond_stats[f"{alpha_type}.{condition}"]['proposed'] += 1
        if regime:
            self.rc_stats[self._rc_key(alpha_type, condition, regime)]['proposed'] += 1

    def record_allowed(self, alpha_type, pnl=0.0, condition=COND_NONE, regime=None):
        self.stats[alpha_type]['allowed'] += 1
        self.stats[alpha_type]['total_pnl'] += pnl
        if pnl > 0:
            self.stats[alpha_type]['wins'] += 1
        else:
            self.stats[alpha_type]['losses'] += 1
        cs = self.cond_stats[f"{alpha_type}.{condition}"]
        cs['allowed'] += 1
        cs['total_pnl'] += pnl
        if pnl > 0:
            cs['wins'] += 1
        else:
            cs['losses'] += 1
        if regime:
            rc = self.rc_stats[self._rc_key(alpha_type, condition, regime)]
            rc['allowed'] += 1
            rc['total_pnl'] += pnl
            if pnl > 0:
                rc['wins'] += 1
            else:
                rc['losses'] += 1

    def record_denied(self, alpha_type, condition=COND_NONE, regime=None):
        self.stats[alpha_type]['denied'] += 1
        self.cond_stats[f"{alpha_type}.{condition}"]['denied'] += 1
        if regime:
            self.rc_stats[self._rc_key(alpha_type, condition, regime)]['denied'] += 1

    def get_legitimacy(self, alpha_type):
        s = self.stats[alpha_type]
        if s['proposed'] == 0:
            return 1.0
        return s['allowed'] / s['proposed']

    def get_cond_legitimacy(self, alpha_type, condition):
        key = f"{alpha_type}.{condition}"
        s = self.cond_stats[key]
        if s['proposed'] == 0:
            return 1.0
        return s['allowed'] / s['proposed']

    def is_suppressed(self, alpha_type):
        s = self.stats[alpha_type]
        if s['proposed'] < SUPPRESSION_MIN_N:
            return False
        return self.get_legitimacy(alpha_type) < SUPPRESSION_THRESHOLD

    def summary_table(self):
        rows = []
        for a in ALL_ALPHAS:
            s = self.stats[a]
            leg = self.get_legitimacy(a)
            sup = self.is_suppressed(a)
            wr = s['wins'] / s['allowed'] * 100 if s['allowed'] > 0 else 0
            ev = s['total_pnl'] / s['allowed'] if s['allowed'] > 0 else 0
            rows.append({
                'alpha': a,
                'proposed': s['proposed'],
                'allowed': s['allowed'],
                'denied': s['denied'],
                'legitimacy': round(leg, 3),
                'WR': round(wr, 1),
                'EV': round(ev, 2),
                'suppressed': sup,
            })
        return rows

    def record_anti_soar(self, alpha_type, condition, pnl_if_executed, regime=None):
        """Anti-SOAR feedback: what would have happened if gate allowed this."""
        key = f"{alpha_type}.{condition}"
        a = self.anti_soar[key]
        a['denied_total'] += 1
        a['denied_pnl_sum'] += pnl_if_executed
        if pnl_if_executed > 0:
            a['denied_would_win'] += 1
        if regime:
            rc_key = self._rc_key(alpha_type, condition, regime)
            ra = self.rc_anti_soar[rc_key]
            ra['denied_total'] += 1
            ra['denied_pnl_sum'] += pnl_if_executed
            if pnl_if_executed > 0:
                ra['denied_would_win'] += 1

    def get_anti_soar_fail_rate(self, tag):
        """Fraction of denied trades that would have won (= gate was 'wrong')."""
        a = self.anti_soar[tag]
        if a['denied_total'] == 0:
            return 0.0
        return a['denied_would_win'] / a['denied_total']

    def get_rc_anti_soar_fail_rate(self, rc_key):
        a = self.rc_anti_soar[rc_key]
        if a['denied_total'] == 0:
            return 0.0
        return a['denied_would_win'] / a['denied_total']

    def get_proposal_weight(self, tag, regime=None):
        """
        Proposal weight lookup priority:
          1. RC-level weight (alpha.condition@regime) if regime given and n >= RC_OBSERVE_MIN_N
          2. Condition-level weight (alpha.condition)
          3. Default (1.0)
        """
        if regime:
            parts = tag.split('.')
            if len(parts) == 2:
                rc_key = f"{tag}@{regime}"
            else:
                rc_key = tag if '@' in tag else f"{tag}@{regime}"
            rc_s = self.rc_stats[rc_key]
            if rc_s['proposed'] >= RC_OBSERVE_MIN_N:
                return self.rc_weights[rc_key]
        return self.proposal_weights[tag]

    def compute_structural_score(self, tag):
        """
        Score = 0.5*legitimacy + 0.3*norm_EV + 0.2*(1 - anti_soar_fail_rate)

        All components are structural, not reward-based.
        """
        s = self.cond_stats[tag]
        if s['proposed'] < PROPOSAL_OBSERVE_MIN_N:
            return None

        legitimacy = s['allowed'] / s['proposed'] if s['proposed'] > 0 else 1.0

        ev = s['total_pnl'] / s['allowed'] if s['allowed'] > 0 else 0
        ev_norm = np.clip(ev / 20.0, -1.0, 1.0) * 0.5 + 0.5

        anti_fail = self.get_anti_soar_fail_rate(tag)

        score = (SCORE_W_LEGITIMACY * legitimacy +
                 SCORE_W_EV * ev_norm +
                 SCORE_W_ANTI_SOAR * (1.0 - anti_fail))

        return round(score, 4)

    def update_proposal_weights(self):
        """
        Delayed weight update. Only adjusts when n >= PROPOSAL_LEARNING_MIN_N.
        Max step: ±5%. Weight range: [0.80, 1.00]. No amplification.
        """
        self._update_count += 1
        updates = {}
        target_score = 0.65

        for tag, s in self.cond_stats.items():
            if s['proposed'] < PROPOSAL_LEARNING_MIN_N:
                continue

            score = self.compute_structural_score(tag)
            if score is None:
                continue

            current_w = self.proposal_weights[tag]
            delta = (score - target_score) * 0.1
            delta = np.clip(delta, -PROPOSAL_STEP_MAX, PROPOSAL_STEP_MAX)

            new_w = np.clip(current_w + delta,
                            PROPOSAL_WEIGHT_MIN, PROPOSAL_WEIGHT_MAX)

            if abs(new_w - current_w) > 0.001:
                self.proposal_weights[tag] = round(new_w, 4)
                updates[tag] = {
                    'old': round(current_w, 4),
                    'new': round(new_w, 4),
                    'score': score,
                    'delta': round(delta, 4),
                }

        return updates

    def compute_rc_structural_score(self, rc_key):
        """
        Same scoring formula as condition-level, but applied to RC slice.
        Uses RC-specific anti-soar data.
        """
        s = self.rc_stats[rc_key]
        if s['proposed'] < RC_OBSERVE_MIN_N:
            return None

        legitimacy = s['allowed'] / s['proposed'] if s['proposed'] > 0 else 1.0

        ev = s['total_pnl'] / s['allowed'] if s['allowed'] > 0 else 0
        ev_norm = np.clip(ev / 20.0, -1.0, 1.0) * 0.5 + 0.5

        anti_fail = self.get_rc_anti_soar_fail_rate(rc_key)

        score = (SCORE_W_LEGITIMACY * legitimacy +
                 SCORE_W_EV * ev_norm +
                 SCORE_W_ANTI_SOAR * (1.0 - anti_fail))

        return round(score, 4)

    def update_rc_weights(self):
        """
        Delayed RC-level weight update.
        Same formula as condition-level but applied per (alpha, condition, regime).
        n >= RC_LEARNING_MIN_N required.
        """
        rc_updates = {}
        target_score = 0.65

        for rc_key, s in self.rc_stats.items():
            if s['proposed'] < RC_LEARNING_MIN_N:
                continue

            score = self.compute_rc_structural_score(rc_key)
            if score is None:
                continue

            current_w = self.rc_weights[rc_key]
            delta = (score - target_score) * 0.1
            delta = np.clip(delta, -PROPOSAL_STEP_MAX, 0.0)

            new_w = np.clip(current_w + delta,
                            PROPOSAL_WEIGHT_MIN, PROPOSAL_WEIGHT_MAX)

            if abs(new_w - current_w) > 0.001:
                self.rc_weights[rc_key] = round(new_w, 4)
                rc_updates[rc_key] = {
                    'old': round(current_w, 4),
                    'new': round(new_w, 4),
                    'score': score,
                    'delta': round(delta, 4),
                }

        return rc_updates

    def compute_motion_score(self, rc_key):
        """
        motion_score = healthy_rate - fast_adverse_rate
        Range: [-1, +1]. Negative = more fast adverse than healthy.
        """
        tags = self.motion_stats.get(rc_key)
        if not tags:
            return None
        total = sum(tags.values())
        if total < MOTION_PENALTY_MIN_N:
            return None
        healthy_rate = tags.get('HEALTHY', 0) / total
        fast_adverse_rate = tags.get('FAST_ADVERSE', 0) / total
        return round(healthy_rate - fast_adverse_rate, 4)

    def update_motion_weights(self):
        """
        Motion-aware penalty (EXP-14). Applied AFTER structural score weights.
        Only penalizes RC slices where motion_score < 0 and n >= 30.
        penalty = clip(k * |motion_score|, 0.0, 0.02)
        new_weight = max(0.80, old_weight - penalty)
        """
        motion_updates = {}

        for rc_key in list(self.motion_stats.keys()):
            if '@' not in rc_key:
                continue

            motion_score = self.compute_motion_score(rc_key)
            if motion_score is None:
                continue

            if motion_score >= 0:
                continue

            penalty = np.clip(MOTION_PENALTY_K * abs(motion_score),
                              0.0, MOTION_PENALTY_MAX)

            current_w = self.rc_weights[rc_key]
            new_w = max(PROPOSAL_WEIGHT_MIN, current_w - penalty)
            new_w = round(new_w, 4)

            if abs(new_w - current_w) > 0.0005:
                self.rc_weights[rc_key] = new_w

                tags = self.motion_stats[rc_key]
                total = sum(tags.values())
                healthy_rate = tags.get('HEALTHY', 0) / total
                fast_adv_rate = tags.get('FAST_ADVERSE', 0) / total

                motion_updates[rc_key] = {
                    'old': round(current_w, 4),
                    'new': new_w,
                    'motion_score': motion_score,
                    'penalty': round(penalty, 4),
                    'healthy_pct': round(healthy_rate * 100, 1),
                    'fast_adv_pct': round(fast_adv_rate * 100, 1),
                    'n': total,
                }

        return motion_updates

    def condition_table(self):
        """Per-condition breakdown with proposal weights."""
        rows = []
        for a in ALL_ALPHAS:
            for c in ALL_CONDITIONS + [COND_NONE]:
                key = f"{a}.{c}"
                s = self.cond_stats[key]
                if s['proposed'] == 0:
                    continue
                leg = s['allowed'] / s['proposed']
                wr = s['wins'] / s['allowed'] * 100 if s['allowed'] > 0 else 0
                ev = s['total_pnl'] / s['allowed'] if s['allowed'] > 0 else 0
                score = self.compute_structural_score(key)
                rows.append({
                    'alpha': a,
                    'condition': c,
                    'tag': key,
                    'proposed': s['proposed'],
                    'allowed': s['allowed'],
                    'denied': s['denied'],
                    'legitimacy': round(leg, 3),
                    'WR': round(wr, 1),
                    'EV': round(ev, 2),
                    'score': score,
                    'weight': round(self.proposal_weights[key], 4),
                })
        return rows

    def rc_table(self):
        """Regime × Condition breakdown — EXP-12 primary output."""
        rows = []
        for rc_key, s in sorted(self.rc_stats.items()):
            if s['proposed'] == 0:
                continue
            leg = s['allowed'] / s['proposed']
            wr = s['wins'] / s['allowed'] * 100 if s['allowed'] > 0 else 0
            ev = s['total_pnl'] / s['allowed'] if s['allowed'] > 0 else 0
            score = self.compute_rc_structural_score(rc_key)
            rows.append({
                'rc_key': rc_key,
                'proposed': s['proposed'],
                'allowed': s['allowed'],
                'denied': s['denied'],
                'legitimacy': round(leg, 3),
                'WR': round(wr, 1),
                'EV': round(ev, 2),
                'score': score,
                'weight': round(self.rc_weights[rc_key], 4),
            })
        return rows

    def rc_legitimacy_gaps(self):
        """Find conditions where legitimacy differs by regime by >= 0.15."""
        from itertools import groupby
        cond_groups = defaultdict(list)
        for rc_key, s in self.rc_stats.items():
            if s['proposed'] < RC_OBSERVE_MIN_N:
                continue
            at_pos = rc_key.index('@')
            cond_tag = rc_key[:at_pos]
            regime = rc_key[at_pos+1:]
            leg = s['allowed'] / s['proposed']
            cond_groups[cond_tag].append({
                'regime': regime,
                'legitimacy': round(leg, 3),
                'proposed': s['proposed'],
                'rc_key': rc_key,
            })

        gaps = []
        for cond_tag, entries in cond_groups.items():
            if len(entries) < 2:
                continue
            legs = [e['legitimacy'] for e in entries]
            gap = max(legs) - min(legs)
            if gap >= 0.0:
                gaps.append({
                    'condition': cond_tag,
                    'gap': round(gap, 3),
                    'regimes': entries,
                })
        gaps.sort(key=lambda x: -x['gap'])
        return gaps

    def proposal_weight_table(self):
        """Summary of all non-default proposal weights."""
        rows = []
        for tag, w in sorted(self.proposal_weights.items()):
            if abs(w - PROPOSAL_WEIGHT_DEFAULT) < 0.001:
                continue
            s = self.cond_stats[tag]
            score = self.compute_structural_score(tag)
            rows.append({
                'tag': tag,
                'weight': round(w, 4),
                'score': score,
                'proposed': s['proposed'],
            })
        return rows

    def to_dict(self):
        result = {}
        for a in ALL_ALPHAS:
            s = self.stats[a]
            result[a] = {
                'proposed': s['proposed'],
                'allowed': s['allowed'],
                'denied': s['denied'],
                'legitimacy': round(self.get_legitimacy(a), 3),
                'suppressed': self.is_suppressed(a),
            }
        cond_result = {}
        for key, s in self.cond_stats.items():
            if s['proposed'] > 0:
                leg = s['allowed'] / s['proposed']
                cond_result[key] = {
                    'proposed': s['proposed'],
                    'allowed': s['allowed'],
                    'denied': s['denied'],
                    'legitimacy': round(leg, 3),
                    'score': self.compute_structural_score(key),
                    'weight': round(self.proposal_weights[key], 4),
                }
        result['_conditions'] = cond_result
        rc_result = {}
        for rc_key, s in self.rc_stats.items():
            if s['proposed'] > 0:
                leg = s['allowed'] / s['proposed']
                rc_result[rc_key] = {
                    'proposed': s['proposed'],
                    'allowed': s['allowed'],
                    'denied': s['denied'],
                    'legitimacy': round(leg, 3),
                    'score': self.compute_rc_structural_score(rc_key),
                    'weight': round(self.rc_weights[rc_key], 4),
                }
        result['_rc'] = rc_result
        weight_changes = {k: round(v, 4) for k, v in self.proposal_weights.items()
                          if abs(v - PROPOSAL_WEIGHT_DEFAULT) > 0.001}
        result['_proposal_weights'] = weight_changes
        rc_weight_changes = {k: round(v, 4) for k, v in self.rc_weights.items()
                              if abs(v - PROPOSAL_WEIGHT_DEFAULT) > 0.001}
        result['_rc_weights'] = rc_weight_changes
        result['_motion'] = dict(self.motion_stats)
        result['_update_count'] = self._update_count
        return result

    def record_motion(self, alpha_type, condition, regime, motion_tag):
        """Record motion quality observation for a gate-allowed trade."""
        cond_key = f"{alpha_type}.{condition}"
        rc_key = self._rc_key(alpha_type, condition, regime) if regime else cond_key
        self.motion_stats[cond_key][motion_tag] += 1
        if regime:
            self.motion_stats[rc_key][motion_tag] += 1

    def motion_table(self):
        """Motion failure map: where do movement failures cluster?"""
        rows = []
        for key in sorted(self.motion_stats.keys()):
            tags = self.motion_stats[key]
            total = sum(tags.values())
            if total == 0:
                continue
            healthy = tags.get('HEALTHY', 0)
            no_follow = tags.get('NO_FOLLOW', 0)
            fast_adv = tags.get('FAST_ADVERSE', 0)
            low_force = tags.get('LOW_FORCE', 0)
            stall = tags.get('STALL', 0)
            failure_count = no_follow + fast_adv + low_force + stall
            failure_rate = failure_count / total
            rows.append({
                'key': key,
                'total': total,
                'healthy': healthy,
                'no_follow': no_follow,
                'fast_adverse': fast_adv,
                'low_force': low_force,
                'stall': stall,
                'failure_rate': round(failure_rate, 3),
                'healthy_rate': round(healthy / total, 3),
            })
        return rows

    def motion_failure_gaps(self):
        """Find RC keys where failure rate diverges by regime."""
        cond_groups = defaultdict(list)
        for key, tags in self.motion_stats.items():
            if '@' not in key:
                continue
            total = sum(tags.values())
            if total < 10:
                continue
            at_pos = key.index('@')
            cond_tag = key[:at_pos]
            regime = key[at_pos+1:]
            healthy = tags.get('HEALTHY', 0)
            failure_rate = 1.0 - (healthy / total)
            cond_groups[cond_tag].append({
                'regime': regime,
                'failure_rate': round(failure_rate, 3),
                'total': total,
                'rc_key': key,
            })

        gaps = []
        for cond_tag, entries in cond_groups.items():
            if len(entries) < 2:
                continue
            rates = [e['failure_rate'] for e in entries]
            gap = max(rates) - min(rates)
            gaps.append({
                'condition': cond_tag,
                'gap': round(gap, 3),
                'regimes': entries,
            })
        gaps.sort(key=lambda x: -x['gap'])
        return gaps


class AlphaGenerator:
    """
    Generates alpha candidates from bar data.
    Each bar can produce 0 or more candidates from different alpha types.

    EXP-10: Proposal shaping — candidates with low structural score
    are probabilistically skipped based on proposal_weight.
    """

    def __init__(self, memory=None, rng_seed=42):
        self.memory = memory or AlphaMemory()
        self._rng = np.random.RandomState(rng_seed)
        self.skipped_count = 0
        self.total_pre_filter = 0
        self._current_regime = None
        self._pheromone_layer = None

    def set_regime(self, regime):
        """Set current regime for RC-aware weight lookup."""
        self._current_regime = regime

    def set_pheromone_layer(self, pdl):
        self._pheromone_layer = pdl

    def _should_propose(self, alpha_type, condition):
        """Check if this candidate should be proposed based on weight.
        Uses RC-level weight if regime is set and has enough data.
        EXP-20: applies pheromone drift if PDL is set."""
        tag = f"{alpha_type}.{condition}"
        w = self.memory.get_proposal_weight(tag, regime=self._current_regime)
        if self._pheromone_layer and self._current_regime and w < PROPOSAL_WEIGHT_MAX - 0.001:
            rc_key = f"{tag}@{self._current_regime}"
            w = self._pheromone_layer.apply_to_weight(w, rc_key)
        if w >= PROPOSAL_WEIGHT_MAX - 0.001:
            return True
        self.total_pre_filter += 1
        if self._rng.random() < w:
            return True
        self.skipped_count += 1
        return False

    def generate(self, df, bar_idx, force_state=None):
        """Generate all alpha candidates for a given bar, with condition tags and proposal shaping."""
        candidates = []
        n = len(df)
        if bar_idx < 20 or bar_idx >= n:
            return candidates

        close = df['close'].values
        dE = df['dE'].values
        z_norm = df['z_norm'].values
        vol_ratio = df['vol_ratio'].values
        ch_range = df['ch_range'].values

        high_20 = df['high'].rolling(20, min_periods=1).max().values
        low_20 = df['low'].rolling(20, min_periods=1).min().values

        c = close[bar_idx]
        z = z_norm[bar_idx]
        de = dE[bar_idx]
        vr = vol_ratio[bar_idx]

        if not self.memory.is_suppressed(ALPHA_Z_MOMENTUM):
            if abs(z) > 1.0 and abs(de) > 0.5:
                direction = 1 if de > 0 else -1
                strength = abs(z) * abs(de)
                feats = {'z': round(z, 3), 'dE': round(de, 3)}
                cond = classify_condition(ALPHA_Z_MOMENTUM, force_state, feats)
                if self._should_propose(ALPHA_Z_MOMENTUM, cond):
                    candidates.append(AlphaCandidate(
                        bar_idx, ALPHA_Z_MOMENTUM, direction, strength, feats, cond
                    ))

        if not self.memory.is_suppressed(ALPHA_RANGE_BREAKOUT):
            h20 = high_20[bar_idx]
            l20 = low_20[bar_idx]
            rng = h20 - l20
            if rng > 0:
                if c >= h20 - 0.25:
                    feats = {'breakout': 'HIGH', 'range': round(rng, 2)}
                    cond = classify_condition(ALPHA_RANGE_BREAKOUT, force_state, feats)
                    if self._should_propose(ALPHA_RANGE_BREAKOUT, cond):
                        candidates.append(AlphaCandidate(
                            bar_idx, ALPHA_RANGE_BREAKOUT, 1,
                            (c - l20) / rng, feats, cond
                        ))
                elif c <= l20 + 0.25:
                    feats = {'breakout': 'LOW', 'range': round(rng, 2)}
                    cond = classify_condition(ALPHA_RANGE_BREAKOUT, force_state, feats)
                    if self._should_propose(ALPHA_RANGE_BREAKOUT, cond):
                        candidates.append(AlphaCandidate(
                            bar_idx, ALPHA_RANGE_BREAKOUT, -1,
                            (h20 - c) / rng, feats, cond
                        ))

        if not self.memory.is_suppressed(ALPHA_VOL_CONTRACTION):
            if bar_idx >= 30:
                recent_vr = vol_ratio[bar_idx - 5:bar_idx]
                if len(recent_vr) > 0 and np.mean(recent_vr) < 0.7 and vr > 1.0:
                    direction = 1 if de > 0 else -1
                    feats = {'squeeze_ratio': round(vr / (np.mean(recent_vr) + EPS), 2)}
                    cond = classify_condition(ALPHA_VOL_CONTRACTION, force_state, feats)
                    if self._should_propose(ALPHA_VOL_CONTRACTION, cond):
                        candidates.append(AlphaCandidate(
                            bar_idx, ALPHA_VOL_CONTRACTION, direction,
                            vr / (np.mean(recent_vr) + EPS), feats, cond
                        ))

        if not self.memory.is_suppressed(ALPHA_MEAN_REVERT):
            if abs(z) > 2.0:
                direction = -1 if z > 0 else 1
                feats = {'z_extreme': round(z, 3)}
                cond = classify_condition(ALPHA_MEAN_REVERT, force_state, feats)
                if self._should_propose(ALPHA_MEAN_REVERT, cond):
                    candidates.append(AlphaCandidate(
                        bar_idx, ALPHA_MEAN_REVERT, direction,
                        abs(z) - 1.5, feats, cond
                    ))

        if not self.memory.is_suppressed(ALPHA_MICRO_MOMENTUM):
            if bar_idx >= 3:
                mom3 = close[bar_idx] - close[bar_idx - 3]
                if abs(mom3) > 2.0:
                    direction = 1 if mom3 > 0 else -1
                    feats = {'mom3': round(mom3, 3)}
                    cond = classify_condition(ALPHA_MICRO_MOMENTUM, force_state, feats)
                    if self._should_propose(ALPHA_MICRO_MOMENTUM, cond):
                        candidates.append(AlphaCandidate(
                            bar_idx, ALPHA_MICRO_MOMENTUM, direction,
                            abs(mom3), feats, cond
                        ))

        if not self.memory.is_suppressed(ALPHA_FLOW_IMBALANCE):
            if 'buy_vol' in df.columns and 'sell_vol' in df.columns:
                bv = df['buy_vol'].values[bar_idx]
                sv = df['sell_vol'].values[bar_idx]
                total = bv + sv
                if total > 5:
                    imbalance = (bv - sv) / total
                    if abs(imbalance) > 0.6:
                        direction = 1 if imbalance > 0 else -1
                        feats = {'imbalance': round(imbalance, 3), 'total_vol': int(total)}
                        cond = classify_condition(ALPHA_FLOW_IMBALANCE, force_state, feats)
                        if self._should_propose(ALPHA_FLOW_IMBALANCE, cond):
                            candidates.append(AlphaCandidate(
                                bar_idx, ALPHA_FLOW_IMBALANCE, direction,
                                abs(imbalance), feats, cond
                            ))

        for cand in candidates:
            self.memory.record_proposal(cand.alpha_type, cand.condition, regime=self._current_regime)

        return candidates
