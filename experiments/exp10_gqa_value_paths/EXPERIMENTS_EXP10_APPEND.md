# EXP10 — GQA Value-Head × Token-Position Causal Localization

## Status
PLANNED

## Motivation

EXP09 identifies H20 V/KV as the first downstream representation satisfying strong sufficiency and necessity simultaneously.

H20 V alone carries approximately 90% of the H20 residual effect, whereas K is near zero and Q is negative.

EXP10 localizes this V-routing interface across the four GQA KV heads and across token positions in the exact common suffix.

## Scope boundary

The position scan covers only the identical downstream common suffix. It does not yet localize the differing Skill text itself.

## Tests

1. four individual KV value heads: sufficiency + necessity;
2. leave-one-head-out groups;
3. six normalized common-suffix position bins;
4. 4 × 6 head-position interaction matrix;
5. full-V self/cross-wording controls.

## Statistical policy

Head-position cells are exploratory and receive sign-flip p-values plus Benjamini-Hochberg q-values.

## Decision

Sparse head and position concentration -> refine to exact token offsets and the seven query heads attached to the causal KV group.

Broad distribution -> characterize a distributed GQA value-routing field.
