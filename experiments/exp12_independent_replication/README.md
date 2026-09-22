# EXP12 — Frozen-Circuit Independent Replication

## Motivation
EXP11 compressed the candidate mechanism to:

H20 -> block20 V -> KV0 -> offsets {-13,-5,-3,-1} -> readers {Q0,Q3,Q5} -> action preference

EXP12 freezes those components and changes the data. No head, layer, offset, or
path component is re-selected.

## Independent replication dataset
64 new tasks across four new procedural families:
1. test_edit
2. search_edit
3. config_command
4. docs_code

Each family contains 16 tasks, two opposite procedural Skills, and canonical /
paraphrased Skill wording. Candidate action vocabulary is new:
run_tests, open_file, search_code, inspect_config, run_command, search_docs.

The user prompt ends with the same frozen boundary sentence:
`Choose the single next action now.`

## Frozen targets
- H20 -> decoder block 20
- V head KV0
- offsets: -13,-5,-3,-1
- reader heads: Q0,Q3,Q5

Prespecified negative controls:
- offsets: -12,-10,-8,-6
- readers: Q2,Q4,Q6

## Primary replication tests
- H20 residual reference
- frozen V sufficiency / necessity
- frozen reader-path sufficiency / necessity
- negative offset and reader controls
- cross-wording V replication

## Specificity falsification
A second cohort uses the same tasks and candidates but replaces the procedural
Skill with a direct instruction naming the required next action.

If the frozen circuit transfers equally strongly under direct-choice
instructions, the mechanism should be described as a broader
instruction-conditioned action-selection boundary circuit rather than as
Skill-specific procedural machinery.

## Run
`python experiments/exp12_independent_replication/run.py`

Optional:
- `--phase replication`
- `--phase specificity`

## Outputs
`outputs/exp12_independent_replication/`
