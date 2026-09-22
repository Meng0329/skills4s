# EXP09 — Residual → Q/K/V Routing Mediation

## Motivation

EXP08 localized the causal selector core to H18–H20. H20 is the strongest
single layer, but H20 alone captures only ~62% of the full H15–H20 effect.
EXP06/07 rejected portable downstream MLP and post-attention outputs.

EXP09 tests the immediate consumer interface: Q/K/V projections in the decoder
block that consumes each residual handoff.

## Source → consumer mapping

- H18 → decoder block 18
- H19 → decoder block 19
- H20 → decoder block 20

Qwen2 computes q_proj/k_proj/v_proj first, then applies RoPE to Q/K. EXP09
patches projection outputs **before RoPE**.

## Components

Preregistered:
- Q
- K
- V
- KV
- QKV

H20 is confirmatory; H18/H19 are within-core replication/localization analyses.

## Causal tests

Sufficiency:
- baseline recipient
- transplant donor projection states only

Necessity:
- restore donor residual at Hk
- clamp selected consumer projections back to recipient baseline
- measure loss of the residual-patch effect

## Controls

For QKV at H18/H19/H20:
- self
- same-state cross-wording
- opposite-state cross-wording

Exploratory joint routing:
- chain_KV_sufficiency
- chain_QKV_sufficiency
across consumer blocks 18–20.

## Run

```bash
python experiments/exp09_qkv_routing/run.py
```

## Outputs

`outputs/exp09_qkv_routing/`

Key files:
- summary.json
- routing_profile.csv
- paired_contrasts.csv
- summary_by_config.csv
- intervention_results.csv
- qkv_routing.png
- run_manifest.json
