# EXP11 — Preregistered Experiment Plan

## Status
PLANNED

## Previous result
EXP10:
- full V sufficiency +0.0434
- full V necessity +0.0438
- KV0 direct sufficiency +0.0471
- KV0 direct necessity +0.0472
- KV0×B5 sufficiency +0.0448
- KV0×B5 necessity +0.0445
- both BH q=0.0006

## Part A: exact offsets
Scan all exact negative offsets inside B5 using KV0-only V intervention.

Endpoints:
1. sufficiency;
2. H20-residual necessity loss;
3. leave-one-token-out loss within the full KV0×B5 V mediator.

Inference:
- task-level bootstrap 95% CI;
- sign-flip p;
- BH-FDR separately per endpoint.

## Part B: query readers
Prespecified Q0..Q6, because KV0 is repeated to seven query heads.

Path sufficiency:
baseline attention output + the actual Qh delta caused by KV0×B5.

Path necessity:
KV0×B5 V patch, but clamp Qh attention output to baseline.

Sanity:
- all seven heads reconstruct full V patch;
- removing all seven removes the full effect;
- Q7..Q27 response delta ~0;
- canonical/paraphrase subgroup signs concordant.

## Claim boundary
EXP11 localizes the read path from the verified B5 value state. It does not yet
identify which earlier differing Skill tokens write the state into B5.
