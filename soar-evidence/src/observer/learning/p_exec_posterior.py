"""
p_exec Posterior — Beta Distribution per Feature Bin
================================================================
"Only execution probability is learned. Laws are never changed."

DESIGN:
  6 features → bin key → Beta(α, β) posterior
  Features: E_sign, dE_sign, Shadow, ARG_depth, Regime, AEP_zone
  
  Update rule:
    win  → α += 1 (evidence for EXECUTE)
    loss → β += 1 (evidence against EXECUTE)
  
  p_exec = α / (α + β)  (posterior mean)
  
  Prior: Beta(1, 1) = Uniform (no bias)

CONSTRAINT:
  - This module does not modify the Gate
  - Does not modify the Sharp Boundary definition
  - Does not modify Energy calculations
  - Only provides p_exec values; execution decisions are made by the caller
"""

import json
import os
from collections import defaultdict


class BetaPosterior:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.bins = defaultdict(lambda: {'alpha': alpha_prior, 'beta': beta_prior, 'n': 0})

    def _make_key(self, e_sign, de_sign, shadow, arg_depth, regime, aep_zone):
        return f"{e_sign}|{de_sign}|{shadow}|{arg_depth}|{regime}|{aep_zone}"

    def update(self, e_sign, de_sign, shadow, arg_depth, regime, aep_zone, is_win):
        key = self._make_key(e_sign, de_sign, shadow, arg_depth, regime, aep_zone)
        if is_win:
            self.bins[key]['alpha'] += 1
        else:
            self.bins[key]['beta'] += 1
        self.bins[key]['n'] += 1

    def get_p_exec(self, e_sign, de_sign, shadow, arg_depth, regime, aep_zone):
        key = self._make_key(e_sign, de_sign, shadow, arg_depth, regime, aep_zone)
        b = self.bins[key]
        return b['alpha'] / (b['alpha'] + b['beta'])

    def get_confidence(self, e_sign, de_sign, shadow, arg_depth, regime, aep_zone):
        key = self._make_key(e_sign, de_sign, shadow, arg_depth, regime, aep_zone)
        b = self.bins[key]
        n = b['n']
        variance = (b['alpha'] * b['beta']) / ((b['alpha'] + b['beta'])**2 * (b['alpha'] + b['beta'] + 1))
        return {'p_exec': b['alpha'] / (b['alpha'] + b['beta']),
                'n': n, 'variance': round(variance, 6),
                'alpha': b['alpha'], 'beta': b['beta']}

    def get_all_bins(self):
        result = {}
        for key, b in self.bins.items():
            p = b['alpha'] / (b['alpha'] + b['beta'])
            result[key] = {'p_exec': round(p, 4), 'n': b['n'],
                          'alpha': b['alpha'], 'beta': b['beta']}
        return result

    def get_active_bins(self, min_n=5):
        return {k: v for k, v in self.get_all_bins().items() if v['n'] >= min_n}

    def save(self, path):
        data = {
            'alpha_prior': self.alpha_prior,
            'beta_prior': self.beta_prior,
            'bins': dict(self.bins),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.alpha_prior = data['alpha_prior']
        self.beta_prior = data['beta_prior']
        self.bins = defaultdict(lambda: {'alpha': self.alpha_prior, 'beta': self.beta_prior, 'n': 0})
        for k, v in data['bins'].items():
            self.bins[k] = v

    def reset(self):
        self.bins = defaultdict(lambda: {'alpha': self.alpha_prior, 'beta': self.beta_prior, 'n': 0})
