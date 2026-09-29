# EXP17 — Dialogue-Handoff Writer Relocation Falsification

## Status
PLANNED

EXP14–16 established a recurring late-prompt writer → causal V/KV subpath →
sparse positive-reader organization across Qwen2, Mistral, and Granite.

EXP17 freezes all previously discovered components and manipulates only prompt
topology to distinguish semantic instruction, user end, template terminator,
generation handoff, and old absolute-position explanations.

48 new tasks are generated; 32 are primary confirmatory and 16 are untouched
reserve robustness tasks. No component discovery or reselection is allowed.

The primary verdict must be written before reserve tasks are inspected.
