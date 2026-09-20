# EXP01b — Counterbalanced Skill-Controlled Next-State Decoding

## Goal

Test whether changing only the procedure specified by a Skill changes the model's internal representation of the **prescribed next procedural state**.

EXP01 found strong stage decodability, but No-Skill was also highly decodable. EXP01b therefore holds task/history/position/tool identity constant and changes only Skill order.

## Confirmatory layer

Hidden-state index **19**, selected independently by EXP01. Full layer scans are exploratory.

## Contrast

Immediately after the failure has been reproduced:

- TEST-first Skill: next state = `INSPECT_TEST`
- IMPLEMENTATION-first Skill: next state = `INSPECT_IMPLEMENTATION`

Both next actions use `read_file`; only the file argument differs.

Two wording families are used:

- `canonical`: explicit stage labels
- `paraphrase`: independent wording without those labels

The key probe trains on one wording family and tests on the other while holding out whole tasks.

## Run

From repository root:

```bash
python experiments/exp01b_counterbalanced_next_state/run.py --num-tasks 48
```

The script uses only local Hugging Face-format weights under `./models/`.

## Outputs

`outputs/exp01b_counterbalanced_next_state/` contains:

- `summary.json`
- `run_manifest.json`
- `metadata.csv`
- `behavior.csv`
- `no_skill_behavior.csv`
- `layer_probe.csv`
- `control_probe.csv`
- `confirmatory_confusion_matrix.csv`
- `layer_probe.png`
- `activations.npz`
- `stimuli.json`

## Primary success pattern

1. Layer-19 cross-wording Macro-F1 is well above 0.5.
2. Random task-specific mapping control stays near chance.
3. Skill order reverses next-action log-probability in the prescribed direction.
4. Held-out probe score correlates with held-out action preference.
