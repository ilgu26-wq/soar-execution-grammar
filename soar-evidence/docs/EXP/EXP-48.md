# EXP-48: Sharp Boundary

## Question
Does the binary execute/not-execute boundary dominate timing-based decisions?

## Protocol
- Fixed: Core v2, 6-feature posterior
- Varied: Decision type: binary boundary vs continuous timing

## Signals
Performance under binary vs timing decisions, Sharp Gap metric

## Result
Execute or not dominates when to execute. Sharp Boundary captures nearly all performance. SG +76.8%p.

## Verdict
✅ PASS

## Implication
Sharp Boundary is the dominant decision. Timing refinement adds negligible value.
