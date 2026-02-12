"""
JudgeIR — Irreversible Judge
=============================
P1: Input = structural coordinate DELTAS only.
    NO energy magnitude. NO trends. NO performance. NO outcomes.
P2: COMMIT is one-way. reasoning_hash is SEALED.
P3: Results never re-enter Judge input.

R1: Judge_IR.input ∩ Energy = ∅
R4: Judge is irreversible.
"""
import hashlib


class JudgeIR:
    def __init__(self):
        self.committed = []
        self.sealed_hashes = []
        self.total_judgments = 0
        self.total_commits = 0
        self.total_abstains = 0
        self.swing_events = 0
        self.prev_action = None

    def _make_hash(self, action, boundary_info):
        raw = f"{action}|{boundary_info}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def decide(self, coord_delta, is_entry, forbidden_zone=False):
        self.total_judgments += 1

        input_signal = coord_delta
        if forbidden_zone:
            input_signal *= 2.0

        boundary_crossed = is_entry or abs(coord_delta) > 0.5
        should_act = boundary_crossed and abs(input_signal) > 0.05

        if should_act:
            action = 'restrict' if input_signal > 0 else 'release'
            boundary_info = f"delta={coord_delta:.4f}|fz={forbidden_zone}"
            reasoning_hash = self._make_hash(action, boundary_info)

            self.committed.append({
                'step': self.total_judgments,
                'action': action,
                'input_signal': round(input_signal, 6),
            })
            self.sealed_hashes.append(reasoning_hash)
            self.total_commits += 1

            if self.prev_action is not None and action != self.prev_action:
                self.swing_events += 1
            self.prev_action = action

            return action, reasoning_hash
        else:
            self.total_abstains += 1
            return None, None

    def get_stats(self):
        return {
            'total_judgments': self.total_judgments,
            'total_commits': self.total_commits,
            'total_abstains': self.total_abstains,
            'swing_events': self.swing_events,
            'commit_rate': round(self.total_commits / max(self.total_judgments, 1), 4),
            'swing_rate': round(self.swing_events / max(self.total_commits, 1), 4),
        }
