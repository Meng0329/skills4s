# EXP17 — Preregistered Experiment Plan

## Status
PLANNED

## Question
Does the frozen causal writer track:
1. semantic final instruction;
2. end of user content;
3. rendered template terminator;
4. actual generation/dialogue handoff;
5. the old absolute writer position?

## Frozen components
| model | layer | KV/value group | positive readers |
|---|---:|---:|---|
| Qwen2.5-Coder | 20 | 0 | 0,3,5 |
| Mistral-7B-v0.3 | 30 | 4 | 16,18 |
| Granite-3.0-8B | 39 | 2 | 10,11 |

Mistral Q0 is excluded because its EXP15 reader score was exactly zero.

## Data
48 fresh tasks; 12/family.
- confirmatory: indices 0..7 (N=32)
- reserve: indices 8..11 (N=16)

Reserve tasks remain untouched until the confirmatory verdict is written.

## Topology conditions
native

suffix_short:
`Reference note: protocol marker R17.`

suffix_long:
`Reference note: protocol marker R17. Session metadata: channel standard; priority normal; checksum stable.`

instruction_early:
move `Choose the single next action now.` before the action list and end with
the same neutral metadata.

All additions are label- and wording-neutral.

## Sites
Fixed 6-token windows:
- FINAL_INSTRUCTION_END
- USER_CONTENT_END
- TEMPLATE_TERMINATOR
- GENERATION_BOUNDARY
- NATIVE_ABSOLUTE

NATIVE_ABSOLUTE uses the absolute token indices occupied by the prior frozen
writer in that entry's unmodified native prompt.

## Primary endpoints
At every model × condition × site:
- V/KV sufficiency
- local V/KV necessity

Task is the inference unit; paired task bootstrap 95% CI.

## Reader confirmation
For native and suffix_long, at:
- GENERATION_BOUNDARY
- FINAL_INSTRUCTION_END
- NATIVE_ABSOLUTE

measure frozen positive-reader path sufficiency and necessity.

## Controls
At actual generation boundary for native and suffix_long:
- opposite-state cross-wording
- same-state cross-wording
- direct-choice opposite-label

Direct-choice is descriptive specificity evidence; no direction is assumed.

## Pattern rules

HANDOFF_TRACKING:
1. suffix_long handoff sufficiency and necessity CI > 0;
2. suffix_long handoff - NATIVE_ABSOLUTE paired CI > 0;
3. NATIVE_ABSOLUTE effect decreases from native to suffix_long.

SEMANTIC_INSTRUCTION:
1. suffix_long FINAL_INSTRUCTION_END suff/nec CI > 0;
2. it is not significantly weaker than handoff.

ABSOLUTE_POSITION:
1. suffix_long NATIVE_ABSOLUTE suff/nec CI > 0;
2. it is not significantly weaker than handoff.

USER_END_TRACKING:
USER_CONTENT_END remains positive and is not significantly weaker than handoff.

If TEMPLATE_TERMINATOR and GENERATION_BOUNDARY windows have Jaccard >= 0.8,
report HANDOFF_OR_TEMPLATE_BOUNDARY rather than claiming they are separable.

MIXED:
multiple non-equivalent sites satisfy criteria.

## Claim boundary
EXP17 distinguishes relocation hypotheses for the frozen causal V/KV subpaths.
It does not prove the tested subpath is the only procedural-control mechanism.
