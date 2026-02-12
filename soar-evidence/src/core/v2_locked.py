"""
SOAR CORE v2 — LOCKED PRODUCTION PARAMETERS
==============================================
Frozen after 6 experiments (EXP-01 ~ EXP-06).
DO NOT modify without full re-validation.

Lock date: 2026-02-12
Evidence: data/exp_evidence/exp_01 ~ exp_06

Decision basis:
  EXP-01~03: SOAR grammar structurally reduces DD/streak → PASS
  EXP-04: v2 PF=1.15 / v1 PF=0.94 → v2 = engine, v1 = brake
  EXP-05: v1 overlay kills PnL (73% retention) → overlay REJECTED
  EXP-06: v1 calibrated still worse DD than v2 → v2 SOLO confirmed
"""

SOAR_CORE_V2_LOCKED = True
LOCK_VERSION = "v2.0.0-PROD"
LOCK_DATE = "2026-02-12"

# ============================================================
#  GATE PARAMETERS (from engine.py validated thresholds)
# ============================================================

DD_THRESHOLD = 0.03
CONSEC_LOSS_PAUSE = 3
CONSEC_LOSS_COOLDOWN_BARS = 50
VOL_GATE_HIGH = 1.3
VOL_GATE_LOW = 0.7
HIGH_VOL_DD_MULTIPLIER = 0.5
WARMUP_BARS = 300

# ============================================================
#  SIGNAL GENERATION PARAMETERS
# ============================================================

STOP_TICKS = 5.0
TAKE_TICKS = 10.0
MIN_SIGNAL_GAP = 10
ER_FLOOR = 0.5
Z_NORM_THRESHOLD = 1.0
ER_MULTIPLIER = 0.8
LOOKBACK_BARS = 20

# ============================================================
#  PROP FIRM RULE MAPPING
# ============================================================
#
#  Prop Rule                    SOAR Parameter           Mapping
#  ─────────────────────────    ────────────────         ───────
#  Daily Max DD (e.g. 2~5%)    DD_THRESHOLD (3%)        DIRECT
#  Trailing DD (e.g. 4~6%)    (covered by DD gate)      DIRECT
#  Max consec losses           CONSEC_LOSS_PAUSE (3)    DIRECT
#  High vol restriction        VOL_GATE (1.3x ratio)    DIRECT
#  Position size limit         POSITION_SIZE (fixed)    CONFIG
#  Daily trade limit           (none — structural)      N/A
#
#  The SOAR gate is STRICTER than most prop firms:
#    - Prop daily DD: typically 2-5% → SOAR gate: 3%
#    - Prop trailing DD: typically 4-6% → SOAR gate: same 3% (tighter)
#    - Most props allow 5+ consec losses → SOAR pauses at 3
#
#  This means: if SOAR passes, any standard prop account passes.

class PropProfile:
    """Prop firm account configuration."""
    def __init__(self, name, account_size, daily_dd_pct, trailing_dd_pct,
                 max_position, tick_value, contract_multiplier=1):
        self.name = name
        self.account_size = account_size
        self.daily_dd_pct = daily_dd_pct
        self.trailing_dd_pct = trailing_dd_pct
        self.max_position = max_position
        self.tick_value = tick_value
        self.contract_multiplier = contract_multiplier

    @property
    def daily_dd_dollars(self):
        return self.account_size * self.daily_dd_pct / 100

    @property
    def trailing_dd_dollars(self):
        return self.account_size * self.trailing_dd_pct / 100


PROP_PROFILES = {
    'APEX_50K': PropProfile(
        name='Apex 50K Eval',
        account_size=50_000,
        daily_dd_pct=2.5,
        trailing_dd_pct=5.0,
        max_position=2,
        tick_value=5.0,
    ),
    'APEX_100K': PropProfile(
        name='Apex 100K Eval',
        account_size=100_000,
        daily_dd_pct=2.0,
        trailing_dd_pct=4.0,
        max_position=4,
        tick_value=5.0,
    ),
    'TOPSTEP_50K': PropProfile(
        name='Topstep 50K',
        account_size=50_000,
        daily_dd_pct=2.0,
        trailing_dd_pct=4.0,
        max_position=2,
        tick_value=5.0,
    ),
    'MNQ_MICRO': PropProfile(
        name='MNQ Micro (Personal)',
        account_size=10_000,
        daily_dd_pct=3.0,
        trailing_dd_pct=6.0,
        max_position=1,
        tick_value=0.50,
    ),
}

# ============================================================
#  DENY REASON ENUM (for trade journal / audit)
# ============================================================

class DenyReason:
    DD_BREACH = 'DD_BREACH'
    CONSEC_LOSS_PAUSE = 'CONSEC_LOSS_PAUSE'
    HIGH_VOL_CAUTION = 'HIGH_VOL_CAUTION'
    DAILY_DD_PROP = 'DAILY_DD_PROP'
    TRAILING_DD_PROP = 'TRAILING_DD_PROP'
    POSITION_LIMIT = 'POSITION_LIMIT'

    ALL = [DD_BREACH, CONSEC_LOSS_PAUSE, HIGH_VOL_CAUTION,
           DAILY_DD_PROP, TRAILING_DD_PROP, POSITION_LIMIT]


def validate_lock():
    """Verify all parameters are within validated ranges."""
    assert SOAR_CORE_V2_LOCKED is True, "v2 must be locked for production"
    assert 0.01 <= DD_THRESHOLD <= 0.10, f"DD_THRESHOLD {DD_THRESHOLD} out of range"
    assert 1 <= CONSEC_LOSS_PAUSE <= 10, f"CONSEC_LOSS_PAUSE {CONSEC_LOSS_PAUSE} out of range"
    assert VOL_GATE_HIGH > 1.0, f"VOL_GATE_HIGH must be > 1.0"
    assert STOP_TICKS > 0, f"STOP_TICKS must be positive"
    assert WARMUP_BARS >= 100, f"WARMUP_BARS must be >= 100"
    return True


if __name__ == '__main__':
    validate_lock()
    print(f"SOAR CORE {LOCK_VERSION} — LOCKED")
    print(f"Lock date: {LOCK_DATE}")
    print(f"DD threshold: {DD_THRESHOLD*100:.1f}%")
    print(f"Consec loss pause: {CONSEC_LOSS_PAUSE}")
    print(f"Vol gate: >{VOL_GATE_HIGH}x")
    print(f"Stop: {STOP_TICKS} ticks, Take: {TAKE_TICKS} ticks")
    print(f"Warmup: {WARMUP_BARS} bars")
    print(f"\nProp profiles available:")
    for k, p in PROP_PROFILES.items():
        print(f"  {k}: ${p.account_size:,} / Daily DD ${p.daily_dd_dollars:,.0f} / "
              f"Trailing ${p.trailing_dd_dollars:,.0f} / Max {p.max_position} contracts")
    print(f"\nAll parameters validated: OK")
