# EXP04 — Preregistered Experiment Plan

## Status
PLANNED

## Previous evidence

- EXP01: stage information is decodable.
- EXP01b: Skill-prescribed next state generalizes across wording.
- EXP02: global linear steering is null.
- EXP03: one-token exact interchange is not sufficient.

## Competing explanations

A. **Single-site chimera**:
the donor final-token activation is inconsistent with unchanged recipient
states at neighboring tokens/layers.

B. **Distributed mediation**:
Skill-conditioned control is jointly carried across token positions and/or
layers.

## Primary configuration

```text
donor       = same task, same wording, opposite state
token span  = exact longest common suffix
layers      = hidden-state indices H19-H28
metric      = logP(src) - logP(test)
```

## Success criteria

1. primary task-bootstrap 95% CI excludes zero in the positive direction;
2. same-state cross-wording control is materially smaller;
3. self patch is approximately zero;
4. canonical and paraphrase recipients show the same sign;
5. opposite-state cross-wording donor yields concordant positive transfer.

A positive result supports a distributed residual-state mediator, not yet a
specific attention/MLP circuit.
