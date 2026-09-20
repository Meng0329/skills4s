# EXP03 — Preregistered Experiment Plan

## Status
PLANNED

## Date
2026-09-20

## Previous Result
EXP02 is a null result for single-layer linear mean-direction steering.

## Hypothesis

H3: The procedural state is context-conditioned/nonlinear rather than a global
linear steering vector. Exact same-task opposite-Skill activation interchange
will causally shift the next-action preference toward the donor Skill state.

## Confirmatory Site

```text
hidden-state index = 19
decoder block index = 18
token = final prompt token
```

## Primary Metric

```text
M = mean_logP(src action) - mean_logP(test action)
```

Signed transfer:

```text
T = s_donor * (M_patched - M_baseline)
```

where:

```text
s_donor = +1  for IMPLEMENTATION-first donor
s_donor = -1  for TEST-first donor
```

## Primary Test

Mean task-level signed transfer at hidden-state index 19.

Success requires:

- positive mean transfer;
- task-bootstrap 95% CI excludes zero;
- same sign in canonical and paraphrased wording;
- self-patch numerical effect approximately zero.

## Exploratory Test

Scan every decoder block output and locate where exact interchange transfer is
largest.

No claim that the exploratory peak is independently confirmed.

## Decision

- Positive exact patch + null linear steering:
  pursue nonlinear/context-specific procedural state.
- Null exact patch:
  move from single-site state representations to distributed circuit/pathway
  analysis.
