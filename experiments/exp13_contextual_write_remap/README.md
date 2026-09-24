# EXP13 — Context-Conditioned Schema Write-Site Remapping

## Motivation

EXP12 independently replicated the reader register but rejected the frozen
absolute write positions from EXP11.

Direct inspection of the EXP12 outputs adds an important constraint:

- offsets `{-13,-5,-3,-1}` correspond to the same tokens across every family
  and wording;
- nevertheless, their V causal effect changes sign across family, wording,
  and intervention direction;
- canonical wording is systematically more positive than paraphrase;
- Q0/Q3/Q5 stay positive while Q2/Q4/Q6 stay inhibitory.

Therefore EXP13 no longer assumes that write location alone explains the
failure. It tests a stronger hypothesis:

> the write site and the local value code may be context/wording conditioned,
> while the downstream reader register remains stable.

## Remote / repository dependency

EXP13 imports the repository implementation:

`experiments/exp12_independent_replication/run.py`

It reuses the exact EXP12 64 tasks, messages, action candidates, and tokenizer
rendering. No task templates are duplicated.

## Split

Within each of the four EXP12 families, 16 tasks are split deterministically:

- 8 discovery
- 8 confirmation

Seed: `5313`.

The same task split is used for canonical and paraphrase wording.

## Schema anchors

Nine semantic anchors are tested, each as a fixed six-token window ending at
that anchor:

1. `SKILL_END`
2. `SYSTEM_END`
3. `ISSUE_END`
4. `DETAIL_END`
5. `ACTION0_END`
6. `ACTION1_END`
7. `FINAL_INSTRUCTION_END`
8. `USER_END`
9. `GENERATION_BOUNDARY`

The first two anchors allow the writer to live in the Skill/system region; the
remaining anchors cover the shared downstream context.

## Discovery unit: family × wording

EXP12 showed large canonical/paraphrase differences. Therefore writer discovery
is performed separately for the eight fixed strata:

`4 families × 2 wording families`.

No discovery is allowed by label/action direction.

### Bidirectional selection rule

For every anchor within a family×wording stratum, compute held-task means for:

- sufficiency when recipient label = 0;
- sufficiency when recipient label = 1;
- necessity when recipient label = 0;
- necessity when recipient label = 1.

The preregistered discovery score is:

`min(suff_label0, suff_label1, nec_label0, nec_label1)`

The selected writer is the anchor with the highest bidirectional score.

This prevents selecting a site that works only in one donor→recipient
direction, a major failure mode visible in EXP12.

The runner-up anchor is frozen at the same time.

## Held-out confirmation

On 32 confirmation tasks, for every family×wording stratum:

- selected schema-local H20 residual effect and V sufficiency / necessity;
- original EXP11 absolute offsets as historical control;
- frozen discovery runner-up as control;
- label0 and label1 effects reported separately;
- cross-wording transfer maps recipient and donor through their own frozen
  schema anchors;
- same-state cross-wording control;
- Q0/Q3/Q5 frozen reader register versus Q2/Q4/Q6;
- all-Q0..Q6 reconstruction/removal and Q7..Q27 zero-leakage sanity checks.

## Primary interpretation

### Pattern A — contextual writer + stable reader

Selected writer sites differ across family and/or wording, held-out V transfer
is positive in both directions, and Q0/Q3/Q5 remains superior to Q2/Q4/Q6.

### Pattern B — schema-stable writer + stable reader

The same semantic anchor is selected across strata and held-out confirmation
passes.

### Pattern C — no portable writer, stable reader

Schema remapping still fails or remains one-directional, while the frozen
reader register remains positive and superior to negative readers.

If Pattern C occurs, stop positional search. The next experiment should identify
the functional latent state consumed by the stable reader register rather than
continue scanning token locations.

## Run

```bash
python experiments/exp13_contextual_write_remap/run.py
```

Optional:

```bash
python experiments/exp13_contextual_write_remap/run.py --phase discovery
python experiments/exp13_contextual_write_remap/run.py --phase confirmation
```

## Outputs

`outputs/exp13_contextual_write_remap/`

Key files:

- `split.json`
- `schema_audit.csv`
- `discovery_results.csv`
- `discovery_summary.csv`
- `selected_schema.json`
- `confirmation_results.csv`
- `confirmation_summary.csv`
- `directional_confirmation.csv`
- `family_wording_confirmation.csv`
- `paired_contrasts.csv`
- `summary.json`
- `run_manifest.json`
