# EXP08 — Selector Boundary / Handoff Localization

## Status
PLANNED

## Motivation

EXP05 established the causal sufficiency of common-suffix H15–H20 restoration,
but that experiment did not establish that all six layers jointly mediate the
effect.

EXP06 and EXP07 subsequently falsified portable downstream MLP and attention
outputs as major mediators.

Therefore the next mechanistic question is whether the selector is:

```text
a depth-distributed H15–H20 state
```

or instead:

```text
an H20 handoff state constructed by earlier layers
```

## Primary contrast

```text
single H20
vs
H15-H19
vs
H15-H20
```

## Decision

If H20 alone reproduces the full effect and H15-H19 without H20 is weak/null,
revise the project claim from "H15-H20 distributed selector" to:

> earlier layers construct a procedural selector that reaches a causal handoff
> state at H20.

If no single layer is sufficient and cumulative prefixes build the effect,
retain a depth-distributed selector account.

The result determines whether EXP09 should study the H20 consumer interface
(RMSNorm/QKV routing) or a multi-layer selector circuit.
