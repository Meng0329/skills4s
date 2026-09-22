# EXP11 — Exact KV0×Token → Query-Head Value Path

## Status
PLANNED

EXP10 collapses H20 V mediation to KV0×B5. EXP11 resolves:
1. exact token offsets inside B5;
2. the seven Q heads served by KV0.

Reader-head interventions transfer or block only the actual per-head
attention-output delta caused by the verified KV0×B5 V patch, rather than
transplanting arbitrary donor post-attention activations.

Sparse token + sparse reader head would define a candidate exact procedural
routing path, which must then be independently replicated.
