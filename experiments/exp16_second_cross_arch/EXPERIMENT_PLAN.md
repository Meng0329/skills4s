# EXP16 — Preregistered Experiment Plan

## Status
PLANNED

## Research question

Does the schema-conditioned writer → V/KV interface → positive/inhibitory reader organization recur in a second genuinely non-Qwen architecture when the writer anchor is allowed to participate in final interface localization?

## Model D

Primary target: `google/gemma-2-9b-it` if accessible before the behavioral gate.

Model selection occurs before mechanistic results. A fallback is allowed only for genuine access/engineering reasons and must be documented before the gate.

## Data and inference

Reuse the controlled 64-task benchmark for direct model comparison. Discovery and confirmation remain disjoint. Task is the inference unit. Mean candidate-token log-prob margin and task-bootstrap 95% CI are unchanged.

## A1 residual candidate generation

Coarse layers at 10,20,30,40,50,60,70,80,90,96% relative depth × all semantic anchors.

For `(L, anchor)`:

`residual_score = min(mean residual effect label0, mean residual effect label1)`.

Keep top 2 `(L, anchor)` candidates. This is discovery candidate generation, not confirmatory evidence.

## A2 provisional V/KV interface

For each A1 candidate:

- scan all KV groups at its candidate L;
- score each with `min(suff0,suff1,nec0,nec1)`;
- keep top 4 KV groups;
- refine only those KV groups over L-2..L+2.

Choose global best provisional `(L,KV)`.

## B contextual writer

At provisional `(L,KV)`, each family×wording stratum searches:
SKILL_END, SYSTEM_END, ISSUE_END, DETAIL_END, ACTION0_END, ACTION1_END, FINAL_INSTRUCTION_END, USER_END, GENERATION_BOUNDARY.

Score = `min(suff0,suff1,nec0,nec1)`.

Freeze winner + runner-up per stratum.

## C final anchor-conditioned interface

With B's frozen stratum-specific anchors:

- scan all KV groups at provisional L;
- keep top 4;
- refine those over provisional L ±2;
- score = `min(suff0,suff1,nec0,nec1)` using each entry's frozen stratum anchor.

Freeze final `(L,KV)`. Do not reselect anchors after this.

## D reader register

Per query head:

- `positive_score = min(mean_suff, mean_nec)`
- `inhibitory_score = max(mean_suff, mean_nec)`

Positive threshold:
`score >= max(0.002, 0.05 * best_positive_score)`

Inhibitory threshold:
`score <= min(-0.002, 0.05 * best_inhibitory_score)`

At most 4 heads per set. Zero-score heads are never inserted to fill a quota.

If no inhibitory head qualifies, record an empty inhibitory set; that weakens the homolog verdict rather than forcing a fake register.

## Confirmation

No component is reselected.

Primary endpoints:
- selected V sufficiency > 0;
- selected V necessity > 0;
- cross-wording selected V > 0;
- label0 and label1 both positive;
- positive reader sufficiency/necessity > 0;
- if inhibitory set exists: positive-minus-inhibitory contrast > 0;
- all-associated-reader reconstruction/removal;
- matched negative anchor: ISSUE_END unless selected anchor is ISSUE_END, then DETAIL_END;
- same-state control;
- runner-up sufficiency/necessity using the runner-up's OWN residual reference;
- leakage.

## Verdict

STRONG RECURRING HOMOLOG: all core levels confirm and fidelity is clean.

PARTIAL / ALGORITHMIC HOMOLOG: substantial recurrence with fidelity degradation or missing inhibitory register.

ARCHITECTURE-DIVERGENT: behavioral gate passes but preregistered V/KV-mediated organization does not recur.

BEHAVIORAL GATE FAILURE: the model does not robustly express the controlled Skill behavior.

No post-hoc rescue changes the confirmatory verdict.
