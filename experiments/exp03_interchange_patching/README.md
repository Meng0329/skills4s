# EXP03 — Paired Interchange Patching

## Motivation

EXP01b found a Skill-sensitive procedural-state representation:

- cross-wording Macro-F1 = 0.709 at hidden-state index 19;
- behavior accuracy = 0.651.

EXP02 then found a null causal result for **linear mean-direction steering**:

- real symmetric effect ≈ +0.00086;
- no advantage over same-state/random controls;
- no meaningful behavioral flip;
- negligible dose response.

Therefore EXP02 rejects the simple hypothesis:

> a global linear direction at hidden-state index 19 is sufficient to control
> next procedural action.

It does **not** reject the stronger possibility that procedural state is
context-dependent or nonlinearly embedded.

## EXP03 question

Can the model's actual paired hidden state from the opposite Skill condition
causally transfer next-action preference when inserted into the same task?

This removes the linear-vector assumption.

For every task and wording family:

```text
TEST-first prompt  <----paired---->  IMPLEMENTATION-first prompt
```

Task, history, relevant files, and tool type are identical.

Only Skill ordering differs.

## Intervention

For a recipient prompt, take the exact last-prompt-token residual activation
from its same-task opposite-Skill donor and replace the recipient activation:

```text
h_recipient^(l) <- h_donor^(l)
```

Then continue the forward pass and re-score:

```text
read_file('tests/...')
read_file('src/...')
```

This is an interchange intervention / activation patch.

## Primary endpoint

For every patch:

```text
margin = logP(src action) - logP(test action)
```

The signed transfer effect is:

```text
+ [patched - baseline]   when donor = IMPLEMENTATION-first
- [patched - baseline]   when donor = TEST-first
```

Positive values mean the recipient was shifted toward the donor's prescribed
procedural state.

## Confirmatory site

EXP01/EXP01b independently selected:

```text
hidden-state index 19
```

which maps to:

```text
decoder block 18 output
```

The effect at index 19 is confirmatory.

The full layer scan is exploratory.

## Sanity control

At hidden-state index 19, EXP03 also performs a self-patch:

```text
h_recipient <- h_recipient
```

which should produce ~zero numerical change. A nonzero self-patch indicates an
implementation bug.

## Run

From repository root:

```bash
python experiments/exp03_interchange_patching/run.py
```

Recommended:

```bash
python experiments/exp03_interchange_patching/run.py --batch-size 4
```

If memory is limited:

```bash
python experiments/exp03_interchange_patching/run.py --batch-size 1
```

## Required previous artifacts

```text
outputs/exp01b_counterbalanced_next_state/
├── activations.npz
├── metadata.csv
├── behavior.csv
└── run_manifest.json
```

and:

```text
experiments/exp01b_counterbalanced_next_state/run.py
```

## Outputs

```text
outputs/exp03_interchange_patching/
├── patch_results.csv
├── layer_effects.csv
├── task_effects_confirmatory.csv
├── layer_effects.png
├── run_manifest.json
└── summary.json
```

## Interpretation

### If exact interchange works but EXP02 linear steering failed

This supports:

```text
linearly decodable
+
context-dependent / nonlinear causal state
```

rather than a globally steerable linear vector.

### If exact interchange also fails

Then the last-prompt-token residual state is likely not a sufficient mediator.
The next experiment should test distributed mechanisms:

- multiple token positions;
- multiple adjacent layers;
- attention/MLP pathway patching;
- sequential patching across the Skill-processing trajectory.
