"""
EVGate — P-root Binary Gate (ALLOW / DENY)
============================================
Fail-closed execution gate.
P=True + boundary clear → ALLOW.
Everything else → DENY.

EV message format: {S, I, B, P, C}
  S = Status vocabulary (13 states)
  I = Instruction vocabulary (13 actions)
  B = Boundary class (4 levels)
  P = Permission bit (True/False) — the root
  C = Criticality (4 levels)

Gate rule:
  P=False → DENY (P_FALSE)
  B=IRREVERSIBLE + C≥HIGH → DENY (BOUNDARY_OVERRIDE)
  Otherwise → ALLOW
"""
import struct
from enum import Enum
from dataclasses import dataclass
from typing import Optional

S_VOCAB = ['ANOMALOUS', 'CONSTRAINED', 'BREACH', 'CONFLICT', 'WARNING',
           'IN_PROGRESS', 'NOMINAL', 'DEGRADED', 'FLAGGED', 'READY',
           'CRITICAL_STATE', 'OVERLOADED', 'UNKNOWN_S']
I_VOCAB = ['HALT', 'DEFER', 'ISOLATE', 'HOLD', 'THROTTLE', 'PAUSE',
           'SAFE_STOP', 'APPROVE', 'CONTINUE', 'SCALE', 'MONITOR',
           'INVESTIGATE', 'UNKNOWN_I']
B_VOCAB = ['IRREVERSIBLE', 'REVERSIBLE', 'IRREVERSIBLE_IF_EXCEED',
           'IRREVERSIBLE_IF_CONTINUE']
C_VOCAB = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']


class DenyCause(Enum):
    CRC_FAIL = 'CRC_FAIL'
    P_FALSE = 'P_FALSE'
    INVALID_SLOT = 'INVALID_SLOT'
    BOUNDARY_OVERRIDE = 'BOUNDARY_OVERRIDE'


class GateDecision(Enum):
    ALLOW = 'ALLOW'
    DENY = 'DENY'


@dataclass
class GateResult:
    decision: GateDecision
    cause: Optional[DenyCause] = None
    ev: Optional[dict] = None


def _index(vocab, value):
    val = str(value).upper().strip()
    for i, v in enumerate(vocab):
        if v == val:
            return i
    return -1


def _crc3(bits_13):
    crc = 0
    for i in range(13):
        bit = (bits_13 >> (12 - i)) & 1
        crc = ((crc << 1) | bit) & 0x07
        if crc & 0x04:
            crc ^= 0x05
    return crc & 0x07


class EVPack:
    @staticmethod
    def pack(ev):
        s = _index(S_VOCAB, ev.get('S', 'UNKNOWN_S'))
        i = _index(I_VOCAB, ev.get('I', 'UNKNOWN_I'))
        b = _index(B_VOCAB, ev.get('B', 'REVERSIBLE'))
        p = 1 if ev.get('P', False) in (True, 'true', 'True', 1, 'TRUE') else 0
        c = _index(C_VOCAB, ev.get('C', 'LOW'))
        if s < 0: s = len(S_VOCAB) - 1
        if i < 0: i = len(I_VOCAB) - 1
        if b < 0: b = 1
        if c < 0: c = 0
        bits_13 = (s << 9) | (i << 5) | (b << 3) | (p << 2) | c
        crc = _crc3(bits_13)
        return struct.pack('>H', (bits_13 << 3) | crc)

    @staticmethod
    def unpack(data):
        if len(data) != 2:
            return None, DenyCause.CRC_FAIL
        bits_16, = struct.unpack('>H', data)
        crc_recv = bits_16 & 0x07
        bits_13 = (bits_16 >> 3) & 0x1FFF
        crc_calc = _crc3(bits_13)
        if crc_recv != crc_calc:
            return None, DenyCause.CRC_FAIL
        c = bits_13 & 0x03
        p = (bits_13 >> 2) & 0x01
        b = (bits_13 >> 3) & 0x03
        i = (bits_13 >> 5) & 0x0F
        s = (bits_13 >> 9) & 0x0F
        ev = {
            'S': S_VOCAB[s] if s < len(S_VOCAB) else 'UNKNOWN_S',
            'I': I_VOCAB[i] if i < len(I_VOCAB) else 'UNKNOWN_I',
            'B': B_VOCAB[b] if b < len(B_VOCAB) else 'REVERSIBLE',
            'P': bool(p),
            'C': C_VOCAB[c] if c < len(C_VOCAB) else 'LOW',
        }
        return ev, None

    @staticmethod
    def verify_roundtrip(ev):
        packed = EVPack.pack(ev)
        unpacked, err = EVPack.unpack(packed)
        if err:
            return False
        repacked = EVPack.pack(unpacked)
        return packed == repacked


class EVGate:
    @staticmethod
    def evaluate(ev):
        if not ev.get('P', False):
            return GateResult(
                decision=GateDecision.DENY,
                cause=DenyCause.P_FALSE,
                ev=ev,
            )
        b = str(ev.get('B', '')).upper()
        c = str(ev.get('C', '')).upper()
        c_idx = _index(C_VOCAB, c)
        if b == 'IRREVERSIBLE' and c_idx >= 2:
            return GateResult(
                decision=GateDecision.DENY,
                cause=DenyCause.BOUNDARY_OVERRIDE,
                ev=ev,
            )
        return GateResult(
            decision=GateDecision.ALLOW,
            ev=ev,
        )
