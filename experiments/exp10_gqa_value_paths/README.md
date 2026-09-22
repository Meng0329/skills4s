# EXP10 — GQA Value-Head × Token-Position Causal Localization

## Motivation

EXP09 identified H20 V/KV as the first strong portable downstream causal interface.

- H20 residual reference ≈ +0.0488
- H20 V sufficiency ≈ 89% of residual effect
- H20 V necessity ≈ 90%
- H20 KV sufficiency ≈ 94%
- H20 KV necessity ≈ 91%
- H20 K ≈ 0
- H20 Q < 0

The next question is: which of the four GQA value heads, at which aligned prompt positions, carry this effect?

## Architecture

Qwen2.5-Coder-7B-Instruct uses 28 query heads, 4 key/value heads, and head_dim=128.
EXP10 intervenes at decoder block 20 `v_proj` output, reshaped as:

[batch, seq, 4 KV heads, 128]

## Scope boundary

Position localization is restricted to the exact common suffix used in EXP04–EXP09.
This region has identical token IDs in donor and recipient and lies downstream of the Skill wording/order manipulation.

Therefore EXP10 localizes where the Skill-induced state is written into the shared downstream context; it does not yet localize the differing Skill tokens themselves.

## Experiment A — KV-head localization

For each KV head 0..3:
- single-head sufficiency
- single-head necessity

Also run all-except-one groups to detect redundancy/synergy.

Full-V intervention is retained as the positive reference.

## Experiment B — position localization

Partition each exact common suffix into six contiguous normalized bins B0..B5.

B0 is earliest in the aligned suffix; B5 is closest to generation.

Patch all four V heads only inside one bin and measure sufficiency/necessity.

## Experiment C — head × position matrix

Evaluate all 4 × 6 = 24 cells for both sufficiency and necessity.

The matrix is exploratory. Report task bootstrap 95% CI, sign-flip p-values, and Benjamini-Hochberg q-values.

## Controls

Full-suffix all-head V:
- self
- same-state cross-wording
- opposite-state cross-wording

Also reproduce the H20 residual reference.

## Token audit

`token_position_map.csv` records:
- task
- wording
- suffix token index
- offset from generation boundary
- normalized bin
- token id
- decoded token text

## Run

```bash
python experiments/exp10_gqa_value_paths/run.py
```

## Decision

- sparse head: move to its 7 associated query heads;
- sparse bin: refine to exact token offsets/semantic spans;
- sparse head×bin cells: candidate compact routing circuit;
- broad effects: characterize distributed GQA value-routing field.
