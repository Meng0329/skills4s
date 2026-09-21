# EXP06 — MLP Neuron-Group Causal Mediation

## Status
PLANNED

## Motivation

EXP05 localized the strongest downstream reconfiguration to the MLP branch:

```text
projection recovery:
MLP      = 0.875
Residual = 0.872
Attention= 0.700
```

The H15–H20 selector window captures 98.4% of the full H15–H28 behavioral
effect.

EXP06 moves from branch-level localization to MLP intermediate-neuron groups.

## Question

Which H21–H28 MLP neuron combinations are causally responsible for transferring
the H15–H20 selector state into the next-action policy?

## Causal standard

A candidate neuron group must pass both:

```text
sufficiency:
donor neuron-group transplant -> behavior moves toward donor

necessity:
selector restoration + clamp group to recipient -> selector effect decreases
```

Feature selection is performed on train tasks only; interventions are evaluated
on held-out tasks.

## Primary group size

```text
K = 256 layer-neuron pairs
```

with K=64 / 1024 exploratory and matched-random controls.

## Decision

If a compact group is both sufficient and necessary, proceed to EXP07:
compositional-neuron/SNMF interpretation, semantic labeling, and cross-Skill
reuse.

If only full-MLP interventions work, the mediator is broad/distributed within
MLP and sparse-neuron claims should be rejected.
