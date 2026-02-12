"""
Constitution — The 5 Structural Laws
======================================
R1: Judge_IR.input ∩ Energy = ∅
R2: Curiosity ∈ Thinker only
R3: Structure only at ENTRY
R4: Judge is irreversible
R5: Disable one → collapse (D4 proven)

Each law can be toggled for destruction testing.
ALL ON = stable system. Any OFF = collapse.
"""

LAWS = {
    'R1': 'Judge sees NO energy (Judge_IR.input ∩ Energy = ∅)',
    'R2': 'Curiosity lives in Thinker only (never leaks to action/judge)',
    'R3': 'Structure changes only at ENTRY (configs at boundary)',
    'R4': 'Judge is irreversible (no feedback, no result reflection)',
    'R5': 'All laws ON or collapse (constitution lock)',
}


class Constitution:
    def __init__(self):
        self.switches = {
            'R1_judge_no_energy': True,
            'R2_curiosity_thinker_only': True,
            'R3_structure_at_entry': True,
            'R4_judge_irreversible': True,
            'R5_all_on': True,
        }

    def disable(self, rule_key):
        if rule_key in self.switches:
            self.switches[rule_key] = False
            self.switches['R5_all_on'] = False

    def enable_all(self):
        for k in self.switches:
            self.switches[k] = True

    def is_intact(self):
        return all(self.switches.values())

    def get_status(self):
        return {
            'intact': self.is_intact(),
            'switches': dict(self.switches),
            'laws': dict(LAWS),
        }
