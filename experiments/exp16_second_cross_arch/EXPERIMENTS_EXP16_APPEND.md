# EXP16 — Second Cross-Architecture Homolog + Anchor-Conditioned Interface Discovery

## Status
PLANNED

EXP15 produced the first non-Qwen cross-architecture replication on Mistral, but its Stage-A interface search used USER_END/FINAL_INSTRUCTION_END while Stage-B later found GENERATION_BOUNDARY as the strong writer. EXP16 repairs this search mismatch by performing a final `(layer,KV)` refinement using the actually selected family×wording writer anchors.

EXP16 also removes the fixed top-3/bottom-3 reader quota so exact-zero tie heads cannot be mislabeled as register members.

Before EXP16 interpretation, EXP15 runner-up necessity is recomputed because the original control reused the selected anchor's residual reference when evaluating the runner-up anchor. This affects only the runner-up diagnostic, not EXP15's primary selected V / reader / cross-wording endpoints.

Preferred Model D is Gemma-2-9B-Instruct (second non-Qwen architecture, GQA geometry). Model selection is frozen before the behavioral gate.
