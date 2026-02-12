# EXP-56: Market Calibration

## Question
Can FalseExec rate be reduced through parameter calibration?

## Protocol
- Fixed: v3_runtime, Sharp Boundary
- Varied: Calibration parameters per market

## Signals
FE rate before/after calibration, PF impact

## Result
FalseExec is structural, not tunable. Parameter adjustment cannot reduce FE without sacrificing capture.

## Verdict
⚠️ PARTIAL

## Implication
FE is a structural floor. Accept it or change the structure (which is frozen).
