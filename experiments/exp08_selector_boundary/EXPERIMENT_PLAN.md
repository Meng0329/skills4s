# EXP08 — Preregistered Experiment Plan

## Status
PLANNED

## Previous evidence

- EXP04: coherent residual restoration is causally positive.
- EXP05: H15–H20 restoration recovers +0.0784, 98.4% of full H15–H28.
- EXP06: downstream MLP outputs are not causal portable mediators.
- EXP07: downstream post-attention head outputs are not causal portable
  mediators.

## Main question

Is the selector state genuinely distributed across H15–H20 depth, or is there
a compact causal handoff at H20?

## Primary contrasts

1. `single_H20`
2. `prefix_H15_H19`
3. `full_H15_H20`

Primary metrics:

```text
E20     = signed transfer(single_H20)
Epre19  = signed transfer(prefix_H15_H19)
Efull   = signed transfer(full_H15_H20)
```

Report:

```text
E20 / Efull
Epre19 / Efull
Efull - E20
Efull - Epre19
```

with task-level paired bootstrap confidence intervals for the differences.

## Handoff-support pattern

Evidence favors an H20 handoff if:

- E20 has CI > 0;
- E20 captures most of Efull;
- Efull - E20 is small;
- Epre19 is much smaller than Efull.

No hard ratio threshold is used as a significance test; the ratio is reported
descriptively.

## Distributed-depth pattern

Evidence favors a depth-distributed selector if:

- no single layer approaches Efull;
- cumulative prefix effects increase across several depths;
- H15-H19 retains substantial positive causal effect.

## Controls

H20-only and full H15-H20:

- self;
- same-state cross-wording;
- opposite-state cross-wording.

All patch activations are captured through decoder-layer forward hooks at the
exact pre-final-norm patch site in float32.
