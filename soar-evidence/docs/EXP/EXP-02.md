# EXP-02: Gate Ablation

## Question
Does each gate component contribute independently to performance?

## Protocol
- Fixed: SOAR grammar, market data, signal parameters
- Varied: Individual gate components removed one at a time

## Signals
PF, Win Rate, Max DD% per ablation variant

## Result
Removing any single gate component degrades performance. No component is redundant.

## Verdict
✅ PASS

## Implication
All gate components are necessary. The gate is minimal — nothing can be removed.
