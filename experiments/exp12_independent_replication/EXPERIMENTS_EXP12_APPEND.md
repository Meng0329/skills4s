# EXP12 — Frozen-Circuit Independent Replication

## Status
PLANNED

Frozen circuit:
H20 -> block20 V -> KV0 -> offsets {-13,-5,-3,-1} -> readers {Q0,Q3,Q5}

EXP12 performs no component discovery.

Independent data: 64 new tasks across test_edit, search_edit, config_command,
and docs_code. Action vocabulary and task templates are new.

Confirmatory tests:
- frozen V sufficiency / necessity
- frozen reader-path sufficiency / necessity
- frozen negative offset/read-head controls
- cross-wording transfer
- all7 reconstruction and leakage sanity

Specificity falsification:
the same tasks are rerun with direct action instructions instead of procedural
Skills. Equal transfer there would require broadening the claim to an
instruction-conditioned action-selection boundary circuit.

A successful EXP12 establishes same-model task-general replication, not yet
cross-model or real-agent generality.
