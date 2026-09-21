# EXP08 — Selector Boundary / Handoff Localization

## Motivation

EXP05–EXP07 establish that restoring the H15–H20 common-suffix residual state
causally redirects behavior, while portable downstream MLP and post-attention
outputs do not explain the effect.

However, the phrase **"H15–H20 selector window"** is still too strong.

We have not yet shown whether:

1. selector information is genuinely distributed across H15–H20; or
2. H15–H19 construct a state that is handed off at H20, with H20 alone being
   the sufficient causal interface.

EXP08 localizes that boundary.

## Main tests

Same task + same wording + opposite-state donor, full aligned common suffix.

### Single-layer sufficiency

```text
H15
H16
H17
H18
H19
H20
```

Patch only one hidden-state output.

### Cumulative prefixes

```text
H15-H16
H15-H17
H15-H18
H15-H19
H15-H20
```

This asks when the causal effect first appears while progressively restoring
the early trajectory.

### Terminal suffixes

```text
H16-H20
H17-H20
H18-H20
H19-H20
H20
```

This asks how much earlier depth can be removed while retaining the effect.

## Critical contrast

The most important comparison is:

```text
H20 only
vs
H15-H19
vs
H15-H20
```

### Handoff pattern

```text
H20 only   ≈ H15-H20
H15-H19    ≈ 0
```

Interpretation:

> H20 is a compact causal handoff state. Earlier layers construct it, but they
> do not need to be patched once H20 itself is restored.

### Distributed-depth pattern

```text
H20 only   << H15-H20
H15-H19    > 0
prefix effect rises progressively
```

Interpretation:

> selector mediation is distributed across depth, not only across token
> positions.

## Controls

For H20-only and full H15-H20:

- self patch;
- same-state cross-wording donor;
- opposite-state cross-wording donor.

## Run

```bash
python experiments/exp08_selector_boundary/run.py
```

## Outputs

```text
outputs/exp08_selector_boundary/
├── intervention_results.csv
├── config_effects.csv
├── paired_contrasts.csv
├── boundary_profile.csv
├── selector_boundary.png
├── run_manifest.json
└── summary.json
```

## Next experiment

If H20 is a handoff state, EXP09 should decompose the **consumer side** of that
handoff at decoder block 20:

```text
H20 residual
 -> input RMSNorm
 -> Q/K/V
 -> attention routing
 -> post-attention residual
 -> MLP
```

If H15–H20 is genuinely depth-distributed, EXP09 should instead use path
patching across the minimal multi-layer selector circuit.
