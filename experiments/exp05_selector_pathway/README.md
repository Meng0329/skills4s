# EXP05 — Selector → Pathway Reconfiguration

## Motivation

EXP04 produced the first positive causal restoration result:

- `last1_H19`: ~0
- `common_H19`: positive
- `common_H19_H24`: stronger positive
- `common_H19_H28`: +0.0579, 95% CI [0.0523, 0.0636]
- `common_H21_H28`: ~0
- `common_H15_H28`: +0.0796

Controls were near zero, and opposite-state cross-wording restoration reproduced
the effect.

This establishes that a coherent distributed residual state is sufficient to
transfer Skill-conditioned action preference.

The next question is *how* that state controls later computation.

## Hypothesis

H5: the H15–H20 residual state acts as a **selector state**.

If the donor H15–H20 common-suffix residual state is restored into the
recipient, then without directly patching H21–H28:

1. behavior should shift toward the donor Skill state;
2. later attention/MLP branch outputs should spontaneously move toward the
   donor computation.

This distinguishes a selector mechanism from a simple "carry the donor state
all the way to the output" mechanism.

## Why attention vs MLP can be separated

Qwen2 decoder layers use:

```text
input
  ↓
RMSNorm
  ↓
Self-Attention
  ↓
Residual Add
  ↓
RMSNorm
  ↓
MLP
  ↓
Residual Add
```

So attention and MLP branch outputs can be captured independently.

## Main behavioral configurations

Same task, same wording, opposite prescribed state donor:

```text
early_H15_H20     # restore only H15–H20
late_H21_H28      # negative control suggested by EXP04
full_H15_H28      # positive control / EXP04 replication
```

Primary selector criterion:

```text
early_H15_H20 > 0
```

If `early_H15_H20` is a substantial fraction of `full_H15_H28`, then an early
state is sufficient to redirect later unpatched computation.

## Downstream pathway readout

During the `early_H15_H20` patched prompt forward, EXP05 captures unpatched
downstream branch outputs in decoder layers 20–27 (H21–H28):

- attention output after `o_proj`;
- MLP output;
- full decoder-block output.

For every layer and branch, define donor-recovery projection:

```text
recovery =
((patched - recipient) · (donor - recipient))
/
||donor - recipient||^2
```

Interpretation:

```text
0  = no movement toward donor
1  = donor-equivalent movement along donor-recipient axis
<0 = moves away from donor
>1 = overshoots donor
```

The branch outputs are never patched in this analysis; they are measured after
only H15–H20 has been restored.

## Controls

- self-patch of H15–H20;
- same-state cross-wording H15–H20 patch;
- opposite-state cross-wording H15–H20 patch;
- `late_H21_H28` residual restoration;
- `full_H15_H28` positive-control restoration.

## Run

From repository root:

```bash
python experiments/exp05_selector_pathway/run.py
```

The script is fully offline and reuses the exact EXP01b stimuli.

## Outputs

```text
outputs/exp05_selector_pathway/
├── behavior_results.csv
├── config_effects.csv
├── pathway_recovery.csv
├── pathway_recovery_by_layer.csv
├── selector_behavior.png
├── pathway_recovery.png
├── run_manifest.json
└── summary.json
```

## Decision logic

### Pattern A — Selector-like mechanism

```text
early_H15_H20 behavior effect > 0
late_H21_H28 ~ 0
downstream attention/MLP outputs move toward donor
```

This supports:

```text
early distributed state
→ reconfigures later computation
→ changes action preference
```

### Pattern B — Carrier-like mechanism

```text
early_H15_H20 ~ 0
full_H15_H28 > 0
```

and downstream branch recovery is weak.

This suggests the donor state must be continuously carried/restored, rather
than acting as a selector.

### Pattern C — Attention-dominant selector

Attention recovery is positive and substantially larger than MLP recovery.

### Pattern D — MLP-dominant selector

MLP recovery is positive and substantially larger than attention recovery.

### Pattern E — Joint pathway

Both attention and MLP recover, or their effects are complementary.

A positive EXP05 does not yet identify individual heads or MLP features.
That becomes EXP06.
