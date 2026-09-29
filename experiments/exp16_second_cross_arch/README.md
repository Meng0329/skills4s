# EXP16 — Second Cross-Architecture Homolog + Anchor-Conditioned Interface Discovery

EXP16 has two goals:

1. replicate the functional organization in a second non-Qwen architecture;
2. repair EXP15's discovery mismatch: EXP15 chose `(layer, KV)` using USER_END / FINAL_INSTRUCTION_END, while the final strong writer was GENERATION_BOUNDARY.

EXP16 makes the actually selected semantic writer participate in the final interface localization before confirmation.

## Preferred Model D

Primary recommendation: `google/gemma-2-9b-it` if access is available. Gemma-2 is a different architecture family, 42 layers, 16 query heads / 8 KV heads (GQA), and therefore preserves the shared-KV→multiple-reader geometry needed for a direct test of the Qwen/Mistral functional organization.

Fallbacks before the behavioral gate:

1. an official/local Llama-3.x-8B-Instruct snapshot;
2. another non-Qwen, non-Mistral GQA instruct decoder;
3. OLMo-2-1124-7B-Instruct only as an explicitly architecture-adapted MHA branch (not identical confirmatory geometry).

Model choice is frozen before the behavioral gate. Never switch models because mechanistic results are negative.

## Discovery factorization

### A1 — coarse residual writer candidates

Coarse relative layer × all semantic anchors. No KV search yet.

Score:

`min(mean residual donor effect label0, mean residual donor effect label1)`

Keep top 2 `(layer, anchor)` candidates.

### A2 — provisional V/KV interface

For each A1 candidate:

1. at the candidate layer, scan all KV groups;
2. retain top 4 KV groups by `min(suff0,suff1,nec0,nec1)`;
3. refine only those KV groups over ±2 layers.

Freeze provisional `(L, KV)`.

### B — contextual schema writer

At provisional `(L,KV)`, each family×wording stratum searches all semantic anchors with bidirectional V sufficiency+necessity. Freeze winner + runner-up.

### C — final anchor-conditioned interface

Using the **frozen stratum-specific anchors**:

1. scan all KV groups at provisional L;
2. keep top 4 KV groups;
3. refine those over ±2 layers;
4. freeze final `(L,KV)`.

Anchors are NOT re-selected afterward.

This is the key methodological repair relative to EXP15.

### D — reader register

At final `(L,KV)` and frozen anchors, measure per-query-head path sufficiency and necessity.

Unlike EXP15, no fixed top-3/bottom-3 quota is forced. Heads must pass a preregistered causal-effect threshold; zero-score ties are never called part of the register.

## EXP15 control repair

Run before interpreting EXP16:

`python experiments/exp16_second_cross_arch/diagnostics/recompute_exp15_runnerup.py`

EXP15's original runner-up necessity accidentally reused the selected anchor's residual reference. The diagnostic recomputes runner-up residual reference at the runner-up anchor itself and writes a non-destructive corrected diagnostic output.

## Run

Inventory:

`python experiments/exp16_second_cross_arch/run.py --phase inventory`

Full:

`python experiments/exp16_second_cross_arch/run.py --model_dir models/<ModelD> --phase all`
