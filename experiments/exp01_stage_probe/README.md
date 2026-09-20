# Experiment 01 — Procedural Stage Probe

## Question

Does a natural-language Agent Skill induce an internal representation of the
**current procedural stage**, beyond what can be explained by trajectory
position or next-tool identity?

This is a representation-discovery experiment. It intentionally does **not**
perform activation steering yet.

## Local model only

Put one or more Hugging Face-format model directories under the repository
root:

```text
skills4s/
└── models/
    └── Qwen2.5-Coder-7B-Instruct/
        ├── config.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        ├── model-00001-of-....safetensors
        └── ...
```

No Hugging Face download is used. The script sets offline mode and calls
`from_pretrained(..., local_files_only=True)`.

If exactly one local model is found, simply run:

```bash
python experiments/exp01_stage_probe/run.py
```

If there are multiple local models:

```bash
python experiments/exp01_stage_probe/run.py \
  --model models/Qwen2.5-Coder-7B-Instruct
```

For the first real run:

```bash
python experiments/exp01_stage_probe/run.py --num-tasks 12
```

## Experimental conditions

For every task/stage pair the model sees the same task and teacher-forced
trajectory history under three conditions:

1. `no_skill`
2. `correct_skill`
3. `shuffled_skill`

`shuffled_skill` contains the same procedural blocks as the correct Skill but
in a fixed wrong order. The main contrast is therefore:

```text
Δ_order = h(correct_skill) - h(shuffled_skill)
```

This is stronger than only using `Skill - NoSkill`, because it reduces the
context-length/content confound.

## Controls

The six procedural stages are:

```text
REPRODUCE
INSPECT_TEST
INSPECT_IMPLEMENTATION
REPAIR
TARGET_VERIFY
REGRESSION_VERIFY
```

Same-tool controls:

```text
INSPECT_TEST           vs INSPECT_IMPLEMENTATION  -> read_file vs read_file
TARGET_VERIFY          vs REGRESSION_VERIFY       -> run_tests vs run_tests
```

Irrelevant tool calls are inserted into histories, so procedural stage is not
identical to raw tool-call index.

## Outputs

```text
outputs/exp01_stage_probe/
├── activations.npz
├── metadata.csv
├── layer_probe.csv
├── position_baseline.csv
├── same_tool_results.csv
├── confusion_matrix_best_layer.csv
├── layer_probe.png
└── summary.json
```

The first scientific decision should use:

- `best_order_delta_macro_f1`
- `best_layer`
- `position_baseline_macro_f1`
- both same-tool pair F1 values
- the layer-wise curve

Chance for six-way stage classification is approximately `0.167`.
