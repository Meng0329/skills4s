# EXP15 — Cross-Architecture Functional Homolog Replication

EXP15 tests whether the causal organization established in Qwen2-family checkpoints
generalizes to a different Transformer architecture.

This package reuses EXP14's tested patching/scoring machinery and replaces only
architecture assumptions and the discovery policy.

## Scientific target

Do not freeze Qwen indices such as H20 / KV0 / Q0,Q3,Q5.

Freeze only the functional organization:

context/schema-conditioned writer
→ single-block projection interface
→ positive + inhibitory reader register
→ next-action preference

## Architecture support

The runner targets Hugging Face decoder-only models whose decoder blocks expose
`self_attn.v_proj` and `self_attn.o_proj`, and whose decoder stack can be found at
common paths such as `model.model.layers`.

This covers intended Llama/Mistral/Gemma-style candidates. Unsupported architectures
fail before intervention instead of silently patching the wrong site.

## Changes from EXP14

- no 28-layer assumption;
- no 4-KV-head assumption;
- no Qwen special-token assumption;
- semantic anchors derived from rendered message content;
- coarse relative-depth layer search + local refinement;
- Phase-A writer selection requires sufficiency + necessity + both labels;
- reader discovery requires sufficiency + necessity;
- Q/KV geometry read from config;
- leakage works for arbitrary query-head counts.

## Usage

Inventory:

```bash
python experiments/exp15_cross_arch_homolog/run.py --phase inventory
```

Full run:

```bash
python experiments/exp15_cross_arch_homolog/run.py   --model_dir /path/to/local/cross_arch_model   --phase all
```

No model download is allowed.
