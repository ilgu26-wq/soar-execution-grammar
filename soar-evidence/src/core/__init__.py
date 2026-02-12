"""
SOAR Execution Grammar — Core
==============================
Boundary-first execution permission system.
Controls WHEN execution is permitted, not WHAT to execute.

Modules:
  boundary      — Phase detection + entry gate
  judge         — Irreversible judgment (sealed, one-way)
  gate          — EV binary gate (ALLOW / DENY)
  constitution  — The 5 structural laws
  engine        — Wires boundary → judge → gate
"""
from .boundary import Boundary, PhaseDetector
from .judge import JudgeIR
from .gate import EVGate, GateDecision
from .constitution import Constitution, LAWS
from .engine import ExecutionEngine
