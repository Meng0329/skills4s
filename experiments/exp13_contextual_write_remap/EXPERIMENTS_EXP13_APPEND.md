# EXP13 — Context-Conditioned Schema Write-Site Remapping

## Status
PLANNED

## EXP12 output audit

Actual EXP12 outputs were inspected before designing EXP13.

Key correction to the previous hypothesis: the frozen offsets correspond to
the same tokens across all families and wording variants, yet their causal V
effects still change sign. Therefore EXP12 cannot be explained as a simple
absolute-position artifact.

EXP13 tests whether the writer is jointly schema- and context-conditioned,
while keeping the replicated Q0/Q3/Q5 reader register frozen.

## Design

- exact EXP12 64 tasks imported from EXP12 run.py;
- 8 discovery / 8 held-out confirmation tasks per family;
- writer discovery separately for family × wording;
- nine fixed semantic anchors;
- bidirectional discovery score requiring both label directions;
- old absolute offsets and discovery runner-up as frozen controls;
- Q0/Q3/Q5 vs Q2/Q4/Q6 reader comparison;
- all7 reconstruction and zero-leakage sanity.

## Claim boundary

A successful result supports a context/schema-dependent writer feeding a
stable downstream reader register.

If writer remapping fails again while the reader register survives, further
position scans are terminated.
