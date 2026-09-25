# EXP15 — Preregistered Experiment Plan

## Status
PLANNED

## Question

Does the functional causal organization discovered in Qwen2-family checkpoints
replicate in a genuinely different model architecture?

The target is functional homology, not index identity.

## Frozen mechanism

context/schema-conditioned writer
→ causal attention value/KV interface
→ positive/inhibitory reader register
→ behavior

## Behavioral gate

The cross-architecture model must show positive Skill-conditioned next-action
preference overall and in every task family before circuit-null results are interpreted.

## Discovery / confirmation

Reuse the controlled 64-task benchmark and deterministic 32/32 split so model
differences are directly comparable. All component search is discovery-only.

### Phase A — writer/interface localization

Coarse relative-depth grid:

10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 96%

At each layer × every KV/value group, test USER_END and FINAL_INSTRUCTION_END.

Selection score:

min(suff_label0, suff_label1, nec_label0, nec_label1)

Refine only around the winning coarse layer: L*-2 ... L*+2.

### Phase B — schema anchor localization

At frozen (L,k), search:
SKILL_END, SYSTEM_END, ISSUE_END, DETAIL_END, ACTION0_END, ACTION1_END,
FINAL_INSTRUCTION_END, USER_END, GENERATION_BOUNDARY.

Each family×wording stratum selects by the same bidirectional score.

### Phase C — reader register

For every query head measure path sufficiency and necessity.

Positive score = min(mean_suff, mean_nec)
Negative score = max(mean_suff, mean_nec)

Freeze top 3 positive and bottom 3 inhibitory heads.

## Confirmation

No component is reselected.

Report:
- selected V sufficiency / necessity;
- cross-wording;
- label0 / label1;
- positive and negative reader registers;
- positive-minus-negative contrast;
- all-reader reconstruction/removal;
- same-state control;
- matched negative anchor;
- reader leakage;
- runner-up sufficiency/necessity.

## Verdict

STRONG HOMOLOG: writer + V/KV interface + reader-register organization confirm.
PARTIAL / ALGORITHMIC HOMOLOG: only part replicates or fidelity controls degrade.
ARCHITECTURE-SPECIFIC: behavioral gate passes but causal organization does not replicate.

No post-hoc rescue.
