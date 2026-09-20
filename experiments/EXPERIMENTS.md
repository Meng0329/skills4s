# skills4s Experiment Log

## EXP01 — Procedural Stage Probe

### Date
2026-09-20

### Commit
ed1de55

### Model
Qwen2.5-Coder-7B-Instruct

### Research Question
Can the current procedural stage be linearly decoded from the hidden
representations of a Skill-conditioned coding agent?

### Hypothesis
H1: Skill-conditioned hidden states contain linearly decodable information
about the current procedural stage beyond trajectory position and tool identity.

### Conditions
- No Skill
- Correct Skill
- Shuffled Skill

### Dataset
- 8 synthetic coding tasks
- 6 procedural stages
- 48 stage examples
- teacher-forced trajectories
- task-grouped cross-validation

### Main Results
- Best layer: 19
- Correct - Shuffled F1: 1.000
- Correct Skill F1: 0.978
- Shuffled Skill F1: 0.978
- No Skill F1: 0.956
- Position baseline F1: 0.422

Same-tool controls:
- INSPECT_TEST vs INSPECT_IMPLEMENTATION: 1.000
- TARGET_VERIFY vs REGRESSION_VERIFY: 1.000

### Initial Observation
Procedural-stage information is extremely strongly linearly decodable,
with the strongest signal around layer 19.

### Important Confound
No-Skill representations already achieve F1 = 0.956.

Therefore EXP01 does NOT establish that the representation is induced by
the Skill. The trajectory history itself contains strong semantic evidence
about the current stage.

### Conclusion
H1 is partially supported only at the representation level.

Supported:
- procedural stage is linearly represented;
- representation generalizes across held-out tasks;
- representation is not reducible to tool identity;
- position baseline alone does not explain the signal.

Not established:
- Skill causally creates the representation;
- Skill ordering causally controls the current procedural state;
- layer 19 representations control agent behavior.

### Decision
Do NOT perform activation steering yet.

Next experiment must isolate Skill-prescribed state while holding task,
trajectory history, tool history, and current position constant.

### Next Experiment
EXP01b — Counterbalanced Skill-Controlled Next-State Decoding

Target contrast:
INSPECT_TEST ↔ INSPECT_IMPLEMENTATION

Both use the same tool (`read_file`).

Only the Skill-defined next procedural state changes.

### Status
COMPLETED