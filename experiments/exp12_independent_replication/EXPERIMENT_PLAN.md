# EXP12 — Preregistered Independent Replication

## Status
PLANNED

## Frozen discovery target
H20 -> block20 V -> KV0 -> offsets {-13,-5,-3,-1} -> readers {Q0,Q3,Q5}

No component may be changed after running EXP12.

## Dataset
64 new tasks:
- 16 test_edit
- 16 search_edit
- 16 config_command
- 16 docs_code

Each task has two opposite Skills and two wording variants.

## Primary endpoints
- H20 residual reference
- frozen V sufficiency
- frozen V necessity loss
- frozen reader-set path sufficiency
- frozen reader-set path necessity loss

## Negative controls
- frozen negative offsets {-12,-10,-8,-6}
- frozen negative readers {Q2,Q4,Q6}

Primary paired contrasts compare target versus negative controls for both
sufficiency and necessity.

## Cross-wording
Repeat the frozen V path with opposite-state donor under the other wording.

## Behavioral validity
Report baseline next-action accuracy before interpreting circuit replication.

## Architecture sanity
Require 28 Q heads, 4 KV heads, KV0 readers Q0..Q6.
Require offset -5 to decode as <|im_end|> and offset -3 as <|im_start|>.

## Specificity falsification
Same tasks, same action candidates, same final boundary sentence. Replace the
procedural Skill with a direct instruction specifying ACTION_0 or ACTION_1.

No direction is assumed for procedural-vs-direct comparison.

## Replication success
All four primary causal endpoints must have task-bootstrap 95% CI > 0.
Target offsets/readers must exceed the frozen negative controls in paired
analysis. Cross-wording V must preserve sign. Decomposition sanity must pass.
