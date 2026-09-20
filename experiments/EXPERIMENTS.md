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

# EXP01b — Counterbalanced Skill-Controlled Next-State Decoding

## Date
2026-09-20

## Commit
62aee2c

## Status
COMPLETED

## Motivation

EXP01 established that procedural stage information is strongly linearly
decodable from Qwen2.5-Coder-7B hidden states, but No-Skill representations
were also highly decodable (Macro-F1 = 0.956). Therefore EXP01 could not
isolate a Skill-induced procedural representation from semantic information
already present in the execution history.

EXP01b was designed to remove this confound.

## Research Question

When the task, execution history, current position, and next-tool identity are
held constant, does changing only the ordering specified by an Agent Skill
alter:

1. the internal representation of the prescribed next procedural state; and
2. the model's actual next-action preference?

## Hypothesis

H1: Skill ordering induces an internal representation of the prescribed next
procedural state that generalizes across tasks and across surface wording.

H2: This representation is behaviorally relevant: changing the Skill-defined
next state should shift the model's probability toward the corresponding next
action.

## Experimental Design

Primary contrast:

- `INSPECT_TEST`
- `INSPECT_IMPLEMENTATION`

Both states require the same tool:

```text
read_file(...)
```

Only the file argument differs:

```text
tests/...
```

versus:

```text
src/...
```

For every paired example, the following variables were held constant:

- coding task;
- reproduced failure;
- execution history;
- current trajectory position;
- next-tool type;
- relevant files.

Only the Skill-prescribed ordering was changed.

Two independent wording families were used:

- canonical wording;
- paraphrased wording.

The primary representation test used cross-task, cross-wording evaluation.
Hidden-state index 19 was preregistered as the confirmatory layer based on the
independent EXP01 result.

A randomized task-specific mapping control was included to estimate whether the
probe could exploit arbitrary task-specific associations.

Behavioral linkage was evaluated from the conditional log-probability of two
candidate actions:

```text
read_file('tests/...')
read_file('src/...')
```

## Main Results

### Representation

Confirmatory Layer:

```text
Layer 19
```

Cross-wording Macro-F1:

```text
0.709
```

Chance:

```text
0.500
```

The representation therefore generalizes across held-out tasks and across
different Skill wording families.

### Random Mapping Control

Mean control Macro-F1:

```text
0.458
```

This result is close to chance and substantially below the true-label probe,
arguing against a simple task-specific label memorization explanation.

A useful descriptive separation is:

```text
true-label F1 / control F1 ≈ 1.55
```

### Behavioral Effect

Overall next-action accuracy:

```text
0.651
```

Canonical-Skill action accuracy:

```text
0.719
```

Thus, Skill ordering not only leaves decodable information in the hidden state;
it also shifts the model's next-action preference in the prescribed direction.

### Representation–Behavior Relationship

Probe / behavior relationship:

```text
Pearson r = 0.345
```

The relationship is positive but moderate.

This indicates that the linearly decodable representation is associated with
behavioral preference, but the present experiment does not establish that the
decoded direction is itself the causal mechanism used by the model.

### No-Skill Behavioral Bias

Without a Skill, the model selected the implementation/source-file action in:

```text
97.9%
```

of cases.

This is a strong intrinsic baseline bias toward inspecting implementation code
before tests.

This bias likely reduces the apparent behavioral effect of TEST-first Skills and
must be explicitly controlled in causal intervention experiments.

## Interpretation

EXP01b resolves the main confound identified in EXP01.

Because task, history, position, and tool identity are matched while the Skill
ordering changes, the above-chance cross-wording decoding result supports the
existence of a Skill-sensitive internal representation of the prescribed next
procedural state.

The cross-wording result further argues that the representation is not solely a
classifier response to explicit labels such as `INSPECT_TEST` or
`INSPECT_IMPLEMENTATION`.

The behavioral log-probability results provide evidence that Skill ordering
affects the model's actual action policy, not only probe decodability.

However, the representation-level result is stronger than the behavioral
result:

```text
Representation Macro-F1 = 0.709
Behavior Accuracy        = 0.651
Probe–Behavior r         = 0.345
```

Therefore the current evidence supports:

```text
Skill ordering
    ↓
Skill-sensitive internal representation
    ↓
associated shift in next-action preference
```

but does NOT yet establish:

```text
decoded representation
    ↓
causally controls next action
```

## Supported Claims

The current experiments support the following claims:

1. The model contains information about Skill-prescribed procedural state.
2. This information generalizes across held-out tasks.
3. This information generalizes across substantially different Skill wording.
4. The effect cannot be explained by next-tool identity because both target
   states use `read_file`.
5. Random task-specific labels do not show comparable cross-task
   generalization.
6. Skill ordering produces a measurable change in next-action preference.

## Claims Not Yet Established

The current experiments do NOT establish that:

1. Layer 19 is the unique or necessary implementation site.
2. The linearly decoded procedural direction is actually used by the model.
3. The procedural representation is necessary for the Skill's behavioral
   effect.
4. Injecting a different procedural state is sufficient to redirect behavior.
5. The same mechanism persists over long multi-step autonomous agent
   trajectories.

## Important Confound / Follow-Up Control

The No-Skill model has a very strong implementation-first prior:

```text
P(prefer implementation | no skill) = 0.979
```

EXP02 must therefore analyze intervention effects relative to this baseline
rather than relying only on raw action accuracy.

Recommended primary outcome for EXP02:

```text
Δ log-odds(action_impl vs action_test)
```

before and after intervention.

This will distinguish a true state-dependent causal shift from the model's
pre-existing source-inspection bias.

## Scientific Conclusion

EXP01b provides positive representation-level and behavioral evidence for a
Skill-sensitive next-procedural-state signal.

The strongest defensible conclusion at this stage is:

> Holding task, history, trajectory position, and tool identity constant,
> changing the procedural ordering specified by an Agent Skill changes an
> internal representation that generalizes across Skill wording and is
> associated with a corresponding shift in next-action preference.

This is stronger than the result of EXP01, but it remains an associational
mechanistic result rather than a causal mechanistic result.

## Decision

H1: SUPPORTED

H2: PARTIALLY SUPPORTED

Proceed to causal intervention.

## Next Experiment

EXP02 — Causal Procedural-State Steering

Primary intervention:

```text
INSPECT_TEST ↔ INSPECT_IMPLEMENTATION
```

Preregistered intervention region:

```text
Layer 19
```

Candidate causal direction:

```text
v = μ(INSPECT_IMPLEMENTATION) - μ(INSPECT_TEST)
```

Test whether adding/removing this direction changes:

```text
log P(read_file(src/...))
-
log P(read_file(tests/...))
```

while preserving the Skill text and execution history.

Required controls:

- matched random direction with equal norm;
- same-state direction;
- multiple steering strengths;
- positive and negative intervention;
- held-out tasks;
- canonical and paraphrased Skills;
- explicit correction for the 97.9% No-Skill implementation-first bias.

A successful EXP02 would move the project from:

```text
Decodable + behavior-associated
```

to:

```text
Causally steerable
```

# EXP02 — Result Note

## Status
COMPLETED — NULL RESULT

## Commit
20dba87

## Main Result

The preregistered linear causal-steering hypothesis was not supported.

| Criterion | Result |
|---|---:|
| Real primary effect | +0.00086 |
| Real > same-state | No; difference ≈ -0.00030 |
| Real > random controls | No; empirical p = 0.167 |
| Behavioral reversal | No; 83.9% vs 84.4% |
| Dose-response span | ≈ 0.004 |

## Interpretation

EXP01b established that Skill-prescribed next state is linearly decodable from
the residual stream and associated with behavior.

EXP02 shows that the corresponding mean-difference direction at the
preregistered site is not sufficient to causally redirect next-action
preference under the tested intervention.

Therefore:

```text
decodable != globally linearly steerable
```

The result does not establish that the decoded information is causally unused.
Alternative possibilities include:

- context-dependent nonlinear representations;
- multiple interacting mediators;
- distributed representation across tokens/layers;
- a representation that is readable but not on the causal pathway.

## Decision

Do not increase alpha or search for a better linear separator as the immediate
next step.

Proceed to EXP03 exact paired interchange patching to remove the global-linear
direction assumption.
