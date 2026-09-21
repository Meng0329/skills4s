# EXP07 — Preregistered Experiment Plan

## Status
PLANNED

## Previous result
EXP06:
- early selector effect ≈ +0.0784
- sparse MLP sufficiency/necessity < 0
- all-MLP sufficiency/necessity < 0

Thus downstream MLP recovery is not a causal mediator.

## Hypothesis
A subset of H21–H28 attention-head outputs mediates the H15–H20 selector
effect.

## Primary K
K=16 layer-head components.

Exploratory: K=4,32,64.

## Sufficiency
Patch selected donor head outputs into baseline recipient.

## Necessity
Restore H15–H20 selector, then clamp selected downstream head outputs to
recipient baseline.

## Controls
- all H21–H28 attention heads;
- five layer-count-matched random K=16 groups;
- held-out tasks;
- opposite-state cross-wording K=16;
- exact common-suffix token alignment.

## Success criteria
Sparse attention mediation is supported only if:
1. K=16 sufficiency CI > 0;
2. K=16 necessity-loss CI > 0;
3. selected-minus-random CI > 0 for both;
4. full-attention upper bound is positive;
5. cross-wording signs agree.
