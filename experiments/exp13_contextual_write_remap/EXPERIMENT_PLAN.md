# EXP13 — Preregistered Context-Conditioned Write-Site Remapping

## Status
PLANNED

## Empirical motivation from EXP12 outputs

EXP12 confirmatory replication failed for the frozen writer but succeeded for
the reader register.

Verified EXP12 output facts:

- frozen V sufficiency = -0.0131;
- frozen V necessity = -0.0150;
- negative-offset V sufficiency = +0.0109;
- target reader sufficiency = +0.0763;
- target reader necessity = +0.0620;
- target-minus-negative reader contrasts = +0.1609 / +0.1672;
- all7 reconstruction/removal exact; Q7..Q27 leakage = 0;
- frozen offset tokens are identical across family/wording;
- frozen V effects nevertheless vary strongly by family, wording, and label.

This rejects a simple absolute-position explanation.

## Frozen mechanism components

- source hidden state: H20
- consumer block: block20
- V head: KV0
- target readers: Q0,Q3,Q5
- negative readers: Q2,Q4,Q6
- old absolute control offsets: -13,-5,-3,-1

No head/layer/read-register re-selection is allowed.

## Dataset split

Exact EXP12 64-task set.

For each family:

- deterministic seeded shuffle;
- 8 discovery tasks;
- 8 confirmation tasks.

Split seed: 5313.

## Candidate schema anchors

All use exactly 6 tokens ending at the semantic boundary:

- SKILL_END
- SYSTEM_END
- ISSUE_END
- DETAIL_END
- ACTION0_END
- ACTION1_END
- FINAL_INSTRUCTION_END
- USER_END
- GENERATION_BOUNDARY

## Discovery strata

Eight fixed strata:

- four families × canonical/paraphrase.

Labels are not separate discovery strata.

## Selection score

For anchor a:

`score(a) = min(S0(a), S1(a), N0(a), N1(a))`

where S/N are discovery-task means for sufficiency/necessity and 0/1 is the
recipient label.

Choose maximum score. Tie-break uses the fixed anchor order above.

Also freeze the runner-up anchor.

## Held-out confirmation

For each selected family×wording writer:

1. selected V sufficiency / necessity;
2. old absolute V sufficiency / necessity;
3. runner-up V sufficiency / necessity;
4. label0 and label1 effects separately;
5. cross-wording transfer using recipient and donor wording-specific anchors;
6. same-state cross-wording control;
7. target-reader sufficiency / necessity;
8. negative-reader sufficiency / necessity;
9. all7/non-KV0 hard sanity checks.

## Primary pooled endpoints

- selected V sufficiency CI > 0
- selected V necessity CI > 0
- selected - old absolute is reported but is not a hard pass criterion because the old set has four sparse tokens while schema windows have six tokens
- selected - frozen runner-up > 0 for sufficiency and necessity
- target reader sufficiency CI > 0
- target reader necessity CI > 0
- target reader - negative reader > 0 for sufficiency and necessity
- cross-wording selected V sufficiency/necessity CI > 0

## Directionality requirement

The mechanism cannot be called a portable bidirectional writer if one action
direction remains systematically negative.

Report confirmation means separately for:

- recipient label 0
- recipient label 1

and for every family×wording×label stratum.

A `bidirectional_writer_pass` flag additionally requires both pooled label
means to be positive for sufficiency and necessity.

## Hard sanity

- all Q0..Q6 sufficiency = selected V effect
- all Q0..Q6 necessity = selected V effect
- Q7..Q27 = 0
- reader recapture baseline = baseline
- reader verified V effect = direct selected-schema V effect

## Decision

Writer confirmation and stable reader -> contextual/schema writer mechanism.

Writer fails but reader survives -> stop token-location search and move to
functional latent-state identification.
