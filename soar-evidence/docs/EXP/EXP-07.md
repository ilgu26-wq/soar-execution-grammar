# EXP-07: Prop Deployment Sim

## Question
Is the SOAR gate compatible with prop firm trading rules?

## Protocol
- Fixed: SOAR v2 grammar
- Varied: Prop firm rule profiles (Apex, Topstep)

## Signals
Daily DD compliance, trailing DD compliance, consecutive loss compliance

## Result
SOAR gate is stricter than standard prop firm rules. If SOAR passes, any prop account passes.

## Verdict
✅ PASS

## Implication
Prop-firm deployment is feasible. Gate parameters map directly to prop rules.
