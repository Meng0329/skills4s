# EXP05 — Preregistered Experiment Plan

## Status
PLANNED

## Previous Result

EXP04 supports coherent distributed residual-state mediation:

```text
common_H19_H28 = +0.0579
95% CI = [0.0523, 0.0636]
common_H15_H28 = +0.0796
common_H21_H28 ≈ 0
```

Self and same-state controls were approximately zero, and opposite-state
cross-wording transfer was positive.

## Main Question

Does the H15–H20 distributed residual state function as a selector that
reconfigures later attention/MLP computations?

## Primary Behavioral Test

Same-task, same-wording, opposite-state donor.

Patch:

```text
common suffix × H15–H20
```

Primary metric:

```text
signed transfer =
donor_sign *
[
(logP(src)-logP(test))_patched
-
(logP(src)-logP(test))_baseline
]
```

Primary success criterion:

```text
task-bootstrap 95% CI > 0
```

## Positive / Negative Controls

```text
full_H15_H28   # positive control
late_H21_H28   # expected near zero
self_H15_H20   # expected near zero
```

Additional semantic controls:

```text
same-state cross-wording H15–H20   # ~0 directional transfer
opposite-state cross-wording H15–H20  # positive if selector generalizes
```

## Pathway Readout

After patching only H15–H20, measure H21–H28 outputs without patching them:

- self-attention branch output;
- MLP branch output;
- decoder block output.

Primary pathway statistic:

```text
projection recovery =
((P-R) dot (D-R)) / ||D-R||^2
```

where:

- `P` = early-patched recipient;
- `R` = unpatched recipient;
- `D` = donor.

## Interpretation

Selector support requires:

1. positive `early_H15_H20` behavior effect;
2. `late_H21_H28` near zero;
3. downstream branch/block recovery toward donor;
4. same-state control near zero;
5. cross-wording opposite-state effect has same sign.

The experiment does not yet claim a specific head-level circuit.
