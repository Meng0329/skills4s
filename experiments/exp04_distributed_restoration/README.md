# EXP04 — Distributed Procedural-State Restoration

EXP03 showed that single-token exact interchange is not sufficient. This does
not yet prove a distributed circuit: a one-token donor patch may simply create
an inconsistent donor/recipient hybrid state.

EXP04 tests whether a *coherent set* of donor activations across multiple
shared suffix tokens and/or multiple layers can transfer the donor Skill's
next-action preference.

## Main tests

Token distribution at H19:
- `last1_H19`
- `common_H19`

Layer distribution on the exact longest common token suffix:
- `common_H19_H24`
- `common_H19_H28`  **primary**
- `common_H21_H28`
- `common_H15_H28`

`H19` is Hugging Face hidden-state index 19, i.e. decoder block 18 output.

## Primary donor

Same task + same wording + opposite prescribed state.

## Controls

At the primary `common_H19_H28` configuration:
- self patch;
- same state + opposite wording;
- opposite state + opposite wording.

The last control tests whether a transferred causal state generalizes across
Skill surface wording.

## Primary metric

`margin = mean_logP(src action) - mean_logP(test action)`

Signed transfer is positive when the patched recipient moves toward the donor
Skill's prescribed state.

## Run

From repository root:

```bash
python experiments/exp04_distributed_restoration/run.py
```

The script is fully offline and auto-detects the local model under `./models`.

## Outputs

`outputs/exp04_distributed_restoration/`

Key files:
- `summary.json`
- `config_effects.csv`
- `task_effects_primary.csv`
- `common_suffix_lengths.csv`
- `distributed_restoration.png`
- `run_manifest.json`
