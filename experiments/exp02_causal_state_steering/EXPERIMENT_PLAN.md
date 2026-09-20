# EXP02 — Preregistered Experiment Plan

## Status
PLANNED

## Date
2026-09-20

## Motivation
EXP01b supports a Skill-sensitive representation of the prescribed next
procedural state at hidden-state index 19 and shows a weaker but positive
behavioral association. Decodability alone does not establish causal use.

## Hypothesis
A direction separating `INSPECT_IMPLEMENTATION` from `INSPECT_TEST` at the
preregistered representation site causally changes the model's next-action
preference.

## Preregistered Representation Site
EXP01/EXP01b hidden-state index:

```text
19
```

Corresponding Qwen decoder module:

```text
model.model.layers[18]
```

because Hugging Face hidden-state index 0 is the embedding output.

## Direction
Estimated within each training fold only:

```text
v = mean(h_impl) - mean(h_test)
```

## Primary Outcome
Candidate-action mean-log-probability margin:

```text
M = logP(src action) - logP(test action)
```

Primary symmetric steering effect at alpha = 1:

```text
E = [M(+1) - M(-1)] / 2
```

## Primary Controls
- equal-norm random direction orthogonal to `v`;
- equal-norm same-state direction;
- held-out task evaluation;
- canonical and paraphrase Skills.

## Exploratory Analysis
Dose response:

```text
alpha ∈ {-2, -1, -0.5, +0.5, +1, +2}
```

## Success Criteria
- `E_real > 0`;
- 95% task-bootstrap CI excludes 0;
- `E_real > E_same_state`;
- `E_real` exceeds the random-control distribution;
- consistent direction in canonical and paraphrased conditions;
- monotonic dose-response around zero.

## Interpretation Boundary
A positive result supports causal influence of the identified residual-stream
direction on next-action preference. It does not yet establish that the
direction is the complete or unique implementation of procedural state.
