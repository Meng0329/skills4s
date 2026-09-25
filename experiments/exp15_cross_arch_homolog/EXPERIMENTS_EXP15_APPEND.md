# EXP15 — Cross-Architecture Functional Homolog Replication

## Status
PLANNED

EXP14 established same-architecture, cross-checkpoint functional homology with
partial fidelity in two Qwen2-family 7B checkpoints.

EXP15 is the first genuine cross-architecture test. It freezes the functional
causal organization but not any Qwen-specific layer/head index. Discovery and
confirmation remain strictly split.

Inherited fidelity metrics:
- same-state contamination;
- reader leakage;
- selected-vs-runner-up sufficiency and necessity;
- directional label0/label1 effects.

A negative result is scientifically valid and must be retained.
